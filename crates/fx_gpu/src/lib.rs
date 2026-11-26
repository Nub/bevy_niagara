//! GPU compute simulation of cooked effects.
//!
//! Each emitter's module stacks compile to one WGSL module ([`codegen`]).
//! Particles live in two pooled structure-of-arrays storage buffers sized for
//! `max_particles` (ping-pong): a step reads the living particles from one,
//! updates them and appends the survivors and the newly spawned ones to the
//! other, so compaction is part of the update. Counts never come back to the
//! CPU: small single-thread kernels turn them into the indirect dispatch
//! arguments of the next kernel, and [`view::ViewDraw`] turns them into
//! indirect draw arguments.
//!
//! Instances of one program are batched ([`GpuBatch`], as Niagara batches
//! GPU emitters): every buffer holds a 256-byte-aligned part per slot, each
//! instance's per-step values (`Frame`: user parameters, age, spawn request,
//! where its parts are) go in one array, and a step of every staged instance
//! is one dispatch per kernel (`workgroup_id.y` picks the instance). A
//! [`GpuEffect`] made on its own is a batch of one.
//!
//! Nothing is allocated per step once built: the per-step uniforms go to a
//! preallocated ring of [`MAX_STEPS_PER_SUBMIT`] slots selected with dynamic
//! offsets, written from a fixed-size scratch array.
//!
//! The emitter stacks (spawn rate, bursts) only read user, engine and emitter
//! values, so they run on the CPU ([`emitter_cpu`]) with the same code path as
//! `fx_core::sim`; the GPU clamps the request to the free room.

pub mod codegen;
mod data;
mod emitter_cpu;
pub mod events;
pub mod gather;
pub mod lights;
pub mod ribbon;
pub mod view;
pub mod volume;

use std::borrow::Cow;
use std::sync::Arc;

use fx_core::cook::CookedSystem;
use fx_core::events::{EVENT_WORDS, Event, EventKind};
use fx_core::sim::{OwnerPose, Particle};

use crate::codegen::ShaderOptions;
use crate::events::EventLayout;
use fx_core::types::Value;
use wgpu::util::DeviceExt;

pub use wgpu;

/// 32-bit words per particle in the SoA layout (see `sim_common.wgsl`); every
/// emitter stores these.
pub const WORDS_PER_PARTICLE: u32 = 28;
/// 32-bit words per particle of an emitter that uses the extended attributes
/// (`fx_core::cook::CookedEmitter::uses_extended_attributes`): the base 28,
/// then Scale, MeshOrientation, RibbonID, RibbonWidth, RibbonTwist,
/// RibbonLinkOrder, LightRadius, DynamicMaterialParameter (words 28..43),
/// SubImageIndex (44), Initial.Scale (45..47), Initial.RibbonWidth (48) and
/// CameraOffset (49), SpriteFacing (50..52), SpriteAlignment (53..55),
/// DynamicMaterialParameter1 (56..59), RotationalVelocity (60..62),
/// Physics.RotationalForce (63..65) and Physics.RotationalDrag (66).
pub const WORDS_PER_PARTICLE_EXTENDED: u32 = 67;
/// 32-bit words per particle of an emitter that also uses
/// DynamicMaterialParameter2 and 3
/// (`fx_core::cook::CookedEmitter::uses_material_parameters_2_3`): the
/// extended 67, then DynamicMaterialParameter2 (67..70) and 3 (71..74).
pub const WORDS_PER_PARTICLE_EXTENDED_2: u32 = 75;
/// Word of `Particles.DynamicMaterialParameter2` (4 words; parameter 3 follows).
pub const WORD_DYNAMIC_MATERIAL_PARAMETER_2: u32 = 67;

/// Words per particle emitter `e` stores.
pub fn particle_words(e: &fx_core::cook::CookedEmitter) -> u32 {
    if e.uses_material_parameters_2_3() {
        WORDS_PER_PARTICLE_EXTENDED_2
    } else if e.uses_extended_attributes() {
        WORDS_PER_PARTICLE_EXTENDED
    } else {
        WORDS_PER_PARTICLE
    }
}

/// The renderers' `extended` level for an emitter storing `words` words per
/// particle: 0 base, 1 extended, 2 also DynamicMaterialParameter2 and 3.
pub fn extended_level(words: u32) -> u32 {
    if words >= WORDS_PER_PARTICLE_EXTENDED_2 {
        2
    } else if words > WORDS_PER_PARTICLE {
        1
    } else {
        0
    }
}
/// Word of `Particles.SpriteFacing` (3 words) in the extended layout.
pub const WORD_SPRITE_FACING: u32 = 50;
/// Word of `Particles.SpriteAlignment` (3 words) in the extended layout.
pub const WORD_SPRITE_ALIGNMENT: u32 = 53;
/// Word of `Particles.DynamicMaterialParameter1` (4 words) in the extended layout.
pub const WORD_DYNAMIC_MATERIAL_PARAMETER_1: u32 = 56;
/// Word of `Particles.CameraOffset` in the extended layout.
pub const WORD_CAMERA_OFFSET: u32 = 49;
/// Word of `Particles.SubImageIndex` in the extended layout.
pub const WORD_SUB_IMAGE_INDEX: u32 = 44;
/// `User.` parameters a system may have on the GPU.
pub const MAX_USER_PARAMS: usize = 16;
/// Steps that can be recorded between two queue submissions.
pub const MAX_STEPS_PER_SUBMIT: u32 = 8;
/// Workgroup size of the per-particle kernels.
pub const WORKGROUP: u32 = 256;
/// Event handlers an emitter may have on the GPU.
pub const MAX_EVENT_HANDLERS: usize = 16;

/// First word of the frame uniform's loop values (after the user parameters).
const FRAME_LOOP_WORD: usize = 12 + MAX_USER_PARAMS * 4;
/// First word of the frame's batch addressing (`src_base` .. `n_active`).
const FRAME_BASE_WORD: usize = FRAME_LOOP_WORD + 4;
/// First word of the owner poses (previous, then current: translation,
/// rotation, scale, a vec4 each), for world-space emitters.
const FRAME_POSE_WORD: usize = FRAME_BASE_WORD + 8;
const FRAME_BYTES: usize = (FRAME_POSE_WORD + 24) * 4;
/// Words of the emitter state buffer.
pub const STATE_WORDS: u64 = 32;
/// Word holding the living-particle count in the state buffer.
pub const STATE_COUNT: u64 = 0;
const STATE_NEXT_ID: usize = 2;

/// Builds a device for headless use (tests, benchmarks, tools), with the
/// adapter's own limits.
pub fn headless_device() -> Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no GPU adapter: {e}"))?;
    let info = adapter.get_info();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("fx_gpu headless"),
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("no GPU device: {e}"))?;
    Ok((device, queue, info))
}

/// Minimal executor for wgpu's native futures (they resolve on poll).
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, Waker};
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

struct Pipelines {
    begin: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    /// The update tracing rays in the ray-tracing scene (RayTraced collisions,
    /// when [`GpuOptions::ray_tracing_layout`] is given); used while a scene is bound.
    update_rt: Option<wgpu::ComputePipeline>,
    spawn_prep: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    /// Sorts the emitter's events (emitters that write events).
    event_rank: Option<wgpu::ComputePipeline>,
    /// Simulation stage kernels, in order.
    stages: Vec<(wgpu::ComputePipeline, codegen::StageKernel)>,
    /// The update and spawn read the data interfaces (group 1) and the scene (group 2).
    uses_scene: bool,
}

/// A part of a batch buffer: one instance's particles, state, grids or events.
/// Bind it with [`Region::binding`]; offsets are 256-byte aligned.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl Region {
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: wgpu::BufferSize::new(self.size),
        })
    }
}

/// One emitter of one instance: where its particles live in the batch
/// buffers, and its CPU-side state.
pub struct GpuEmitter {
    pub name: String,
    pub capacity: u32,
    /// Words per particle: [`WORDS_PER_PARTICLE`], [`WORDS_PER_PARTICLE_EXTENDED`] or [`WORDS_PER_PARTICLE_EXTENDED_2`].
    pub words: u32,
    /// The instance's two particle copies (ping-pong); `particles[current]`
    /// holds the living ones, word `w` of particle `i` at `w * capacity + i`.
    pub particles: [Region; 2],
    /// Grid collections ([`fx_core::custom`]): each grid twice, ping-pong.
    pub grids: Region,
    /// Counts, ids and spawn bookkeeping ([`STATE_WORDS`] u32s).
    pub state: Region,
    /// Words between particle copies in the batch buffer ([`GpuEmitter::particles`]).
    pub region_words: u32,
    pipelines: Arc<Pipelines>,
    pub current: usize,
    age: f32,
    spawn_remainder: f32,
    /// Fraction of a particle carried over by the spawn-count scale.
    scale_remainder: f32,
    /// Generated WGSL, for inspection.
    pub shader_source: Arc<str>,
}

/// One instance of a cooked system simulated on the GPU: a slot of a
/// [`GpuBatch`]. Made with [`GpuEffect::new`] it owns a batch of its own
/// and steps alone ([`GpuEffect::step`]); taken from a shared batch
/// ([`GpuBatch::alloc`]) it steps with the batch's other instances
/// ([`GpuBatch::stage`], [`GpuBatch::record`]).
pub struct GpuEffect {
    pub system: Arc<CookedSystem>,
    /// The compiled pipelines and layouts, shared by every instance made from it.
    pub program: Arc<GpuProgram>,
    /// Emitters spawn while active (Niagara Activate/Deactivate); living
    /// particles simulate either way.
    pub active: bool,
    /// Multiplies every spawn request (scalability); fractions carry over.
    pub spawn_scale: f32,
    /// Set by [`GpuEffect::reset`]: the next step clears the GPU counts first.
    clear_pending: bool,
    pub seed: u32,
    /// Random value mixed into the seed of emitters with `deterministic: false`.
    pub nonce: u32,
    /// The warm-up ran (or there is none); see [`GpuEffect::warm_up`].
    pub warmed: bool,
    /// Steps taken so far.
    pub frame: u32,
    /// Simulated seconds (`Engine.Time`).
    pub time: f32,
    pub user: Vec<Value>,
    /// `Engine.Owner.Velocity` (m/s), read by Inherit Velocity; set by the host.
    pub owner_velocity: [f32; 3],
    /// The owner's `world_from_local` at the end of the next step (set by
    /// the host): world-space emitters move new particles to world space
    /// with it, as [`fx_core::sim::Simulation::owner_pose`].
    pub owner_pose: OwnerPose,
    /// The owner's `world_from_local` at the start of the next step.
    pub previous_owner_pose: OwnerPose,
    pub emitters: Vec<GpuEmitter>,
    /// Where each emitter's events live (offsets within [`GpuEffect::event_counts`] and [`GpuEffect::event_data`]).
    pub event_layout: EventLayout,
    /// One atomic counter per event region.
    pub event_counts: Region,
    /// The events and their sorted order.
    pub event_data: Region,
    /// This instance's [`Scene`](SceneBindings) entry.
    scene: Region,
    /// The batch slot.
    pub slot: u32,
    batch_id: u64,
    /// The batch of an instance made on its own.
    own: Option<Box<GpuBatch>>,
}

/// Instances of one program simulated together: every buffer holds a part
/// per slot, and one step of all staged instances is one dispatch per
/// kernel (Niagara's GPU emitter batching).
pub struct GpuBatch {
    pub program: Arc<GpuProgram>,
    /// Instances it holds.
    pub slots: u32,
    id: u64,
    emitters: Vec<BatchEmitter>,
    event_counts: wgpu::Buffer,
    event_data: wgpu::Buffer,
    ev_count_stride: u32,
    ev_data_stride: u32,
    scene: SceneBindings,
    data: data::DataBindings,
    /// The ray-tracing scene bind group (group 3), when RayTraced collisions trace rays.
    ray_tracing_scene: Option<wgpu::BindGroup>,
    free: Vec<u32>,
    /// Step slot of the frame ring (reset by [`GpuBatch::submitted`]).
    ring: u32,
    /// Instances staged for the step being built.
    staged: u32,
    frame_stride: u64,
}

/// One emitter's buffers in a batch.
struct BatchEmitter {
    particles: wgpu::Buffer,
    state: wgpu::Buffer,
    grids: wgpu::Buffer,
    /// Per step of the ring: one `Frame` per staged instance.
    frames: wgpu::Buffer,
    /// Indirect dispatch arguments (zeroed by the kernels after use).
    args: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    args_bind_group: wgpu::BindGroup,
    region_words: u32,
    grid_words: u32,
    pipelines: Arc<Pipelines>,
    scratch: Vec<u8>,
}

/// What the host offers the simulation beyond the effect itself.
#[derive(Default)]
pub struct GpuOptions<'a> {
    /// The layout of the ray-tracing scene bind group (Solari's scene
    /// bindings; binding 6, the previous frame's TLAS, is read). Given, emitters with RayTraced
    /// collisions trace rays whenever [`GpuEffect::set_ray_tracing_scene`]
    /// has provided a bind group; otherwise they use the depth buffer.
    pub ray_tracing_layout: Option<&'a wgpu::BindGroupLayout>,
}

/// The camera whose depth buffer SceneDepth collisions read (the frame the
/// depth was rendered in), in world space.
#[derive(Clone, Copy, Debug)]
pub struct SceneView {
    pub clip_from_world: bevy_math::Mat4,
    pub camera_position: bevy_math::Vec3,
    /// Depth texture size in pixels.
    pub size: [u32; 2],
}

/// Bytes of the `Scene` uniform in `scene.wgsl`.
const SCENE_BYTES: usize = 4 * 64 + 16 + 16 + 16;

/// Group 2: the scene uniform and the depth texture (an `R32Float` copy of
/// the view's depth, reverse-Z).
struct SceneBindings {
    layout: wgpu::BindGroupLayout,
    /// One `Scene` per slot.
    scenes: wgpu::Buffer,
    dummy_depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl SceneBindings {
    fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx_gpu scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(SCENE_BYTES as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, slots: u32) -> Self {
        let layout = layout.clone();
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx scenes"),
            size: SCENE_STRIDE * slots as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("fx no depth"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCENE_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let bind_group = Self::bind(device, &layout, &uniform, &dummy_depth);
        SceneBindings {
            layout,
            scenes: uniform,
            dummy_depth,
            bind_group,
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx scene"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
            ],
        })
    }
}

/// Format of the depth texture SceneDepth collisions read.
pub const SCENE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One emitter's compiled kernels and sizes, shared by every instance.
struct EmitterProgram {
    pipelines: Arc<Pipelines>,
    shader_source: Arc<str>,
    capacity: u32,
    words: u32,
    grid_bytes: u64,
}

/// A cooked system compiled for the GPU: its pipelines and bind group
/// layouts. Instances made from one program ([`GpuEffect::from_program`])
/// share them, so a new instance only allocates its buffers.
pub struct GpuProgram {
    pub system: Arc<CookedSystem>,
    layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    data_layout: wgpu::BindGroupLayout,
    scene_layout: wgpu::BindGroupLayout,
    event_layout: EventLayout,
    emitters: Vec<EmitterProgram>,
    /// Bytes between the slots of the stage iteration uniform.
    iteration_stride: u64,
    iterations: wgpu::Buffer,
    /// Built with a ray-tracing scene layout.
    pub ray_tracing: bool,
    /// The device's largest storage binding.
    storage_limit: u64,
}

impl GpuProgram {
    /// Compiles every emitter of `system`.
    pub fn new(
        device: &wgpu::Device,
        system: Arc<CookedSystem>,
        options: GpuOptions,
    ) -> Result<Self, String> {
        if system.user_params.len() > MAX_USER_PARAMS {
            return Err(format!(
                "{} user parameters; the GPU path supports {MAX_USER_PARAMS}",
                system.user_params.len()
            ));
        }
        if let Some(e) = system
            .emitters
            .iter()
            .find(|e| e.event_handlers.len() > MAX_EVENT_HANDLERS)
        {
            return Err(format!(
                "emitter \"{}\" has {} event handlers; the GPU path supports {MAX_EVENT_HANDLERS}",
                e.name,
                e.event_handlers.len()
            ));
        }
        let limit = device.limits().max_storage_buffer_binding_size;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx_gpu sim"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(FRAME_BYTES as u64),
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx_gpu sim args"),
            entries: &[storage_entry(0, false)],
        });
        let scene_layout = SceneBindings::layout(device);
        let data_layout = data::DataBindings::layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx_gpu sim"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let args_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx_gpu sim args"),
            bind_group_layouts: &[Some(&layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let scene_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx_gpu sim scene"),
            bind_group_layouts: &[Some(&layout), Some(&data_layout), Some(&scene_layout)],
            immediate_size: 0,
        });
        let rt_pipeline_layout = options.ray_tracing_layout.map(|rt| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fx_gpu sim ray tracing"),
                bind_group_layouts: &[
                    Some(&layout),
                    Some(&data_layout),
                    Some(&scene_layout),
                    Some(rt),
                ],
                immediate_size: 0,
            })
        });

        // Iteration j of a simulation stage reads slot j (`it.j` in sim_common.wgsl).
        let iteration_stride = device.limits().min_uniform_buffer_offset_alignment.max(16) as u64;
        let mut iteration_bytes = vec![0u8; (iteration_stride * fx_core::custom::MAX_ITERATIONS as u64) as usize];
        for j in 0..fx_core::custom::MAX_ITERATIONS as u64 {
            let at = (j * iteration_stride) as usize;
            iteration_bytes[at..at + 4].copy_from_slice(&(j as u32).to_le_bytes());
        }
        let iterations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx stage iterations"),
            contents: &iteration_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let event_layout = EventLayout::new(&system);
        let event_bytes = event_layout.data_words.max(1) * 4;
        if event_bytes > limit {
            return Err(format!(
                "the effect's events need {event_bytes} bytes, over this GPU's storage binding limit of {limit}"
            ));
        }

        let mut emitters = Vec::new();
        for (ei, e) in system.emitters.iter().enumerate() {
            let capacity = e.max_particles.max(1);
            let words = particle_words(e);
            let bytes = capacity as u64 * words as u64 * 4;
            if bytes > limit {
                return Err(format!(
                    "emitter \"{}\": {} particles need {bytes} bytes, over this GPU's storage binding limit of {limit}",
                    e.name, e.max_particles
                ));
            }
            // Grid collections: two copies of each.
            let grid_words: u64 = e.grids.iter().map(|g| 2 * g.half_words() as u64).sum();
            let grid_bytes = (grid_words * 4).max(16);
            if grid_bytes > limit {
                return Err(format!(
                    "emitter \"{}\": its grids need {grid_bytes} bytes, over this GPU's storage binding limit of {limit}",
                    e.name
                ));
            }
            let shader = codegen::emitter_shader(&system, ei, &event_layout, ShaderOptions::default());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&e.name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader.source)),
            });
            let make = |module: &wgpu::ShaderModule, entry: &str, layout: &wgpu::PipelineLayout| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(layout),
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            // Kernels running module code read the data interfaces and the scene when used.
            let modules_read = shader.uses_scene || shader.uses_data;
            let pipe = |entry: &str| {
                let writes_args = entry == "begin" || entry == "spawn_prep" || entry == "finish";
                let runs_modules = entry == "update" || entry == "spawn" || entry.starts_with("stage_");
                make(
                    &module,
                    entry,
                    if writes_args {
                        &args_pipeline_layout
                    } else if runs_modules && modules_read {
                        &scene_pipeline_layout
                    } else {
                        &pipeline_layout
                    },
                )
            };
            // The ray-tracing variant of the update, when it differs.
            let update_rt = rt_pipeline_layout.as_ref().and_then(|rt_layout| {
                let rt = codegen::emitter_shader(
                    &system,
                    ei,
                    &event_layout,
                    ShaderOptions { ray_tracing: true },
                );
                if !rt.uses_tlas {
                    return None;
                }
                let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&e.name),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&rt.source)),
                });
                Some(make(&m, "update", rt_layout))
            });
            let pipelines = Pipelines {
                begin: pipe("begin"),
                update: pipe("update"),
                update_rt,
                spawn_prep: pipe("spawn_prep"),
                spawn: pipe("spawn"),
                finish: pipe("finish"),
                event_rank: shader.writes_events.then(|| pipe("event_rank")),
                stages: shader
                    .stages
                    .iter()
                    .map(|k| (pipe(&k.entry), k.clone()))
                    .collect(),
                uses_scene: modules_read,
            };
            emitters.push(EmitterProgram {
                pipelines: Arc::new(pipelines),
                shader_source: shader.source.into(),
                capacity,
                words,
                grid_bytes,
            });
        }
        Ok(GpuProgram {
            system,
            layout,
            args_layout,
            data_layout,
            scene_layout,
            event_layout,
            emitters,
            iteration_stride,
            iterations,
            ray_tracing: options.ray_tracing_layout.is_some(),
            storage_limit: limit,
        })
    }

    /// GPU bytes one instance (one batch slot) of this program takes (before data interfaces).
    pub fn instance_bytes(&self) -> u64 {
        self.emitters
            .iter()
            .map(|e| {
                align_words(e.capacity as u64 * e.words as u64) * 4 * 2
                    + STATE_STRIDE * 4
                    + FRAME_BYTES as u64 * MAX_STEPS_PER_SUBMIT as u64
                    + align_words(e.grid_bytes / 4) * 4
            })
            .sum::<u64>()
            + align_words(self.event_layout.regions.len().max(1) as u64) * 4
            + align_words(self.event_layout.data_words.max(1)) * 4
            + SCENE_STRIDE
    }

    /// Particles one instance of emitter `emitter` holds.
    pub fn emitter_capacity(&self, emitter: usize) -> u32 {
        self.emitters[emitter].capacity
    }

    /// The most slots a [`GpuBatch`] of this program can hold: every
    /// batch buffer fits one storage binding, and at most [`MAX_BATCH_SLOTS`].
    pub fn max_slots(&self) -> u32 {
        let per_slot = self
            .emitters
            .iter()
            .map(|e| {
                (align_words(e.capacity as u64 * e.words as u64) * 8)
                    .max(align_words(e.grid_bytes / 4) * 4)
            })
            .chain([
                align_words(self.event_layout.data_words.max(1)) * 4,
                FRAME_BYTES as u64,
                SCENE_STRIDE,
            ])
            .max()
            .unwrap_or(1)
            .max(1);
        (self.storage_limit / per_slot).clamp(1, MAX_BATCH_SLOTS as u64) as u32
    }
}

/// Words of the per-slot state (256 bytes, so a slot's state can be bound on its own).
pub const STATE_STRIDE: u64 = 64;
/// Bytes of a `Scene` entry (see `scene.wgsl`).
const SCENE_STRIDE: u64 = SCENE_BYTES as u64;
/// Slots a shared batch may hold at most.
pub const MAX_BATCH_SLOTS: u32 = 1024;

fn align_words(w: u64) -> u64 {
    w.div_ceil(64) * 64
}

static NEXT_BATCH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl GpuBatch {
    /// Allocates the buffers of `slots` instances (clamped to
    /// [`GpuProgram::max_slots`]); no shader compilation.
    pub fn new(device: &wgpu::Device, program: &Arc<GpuProgram>, slots: u32) -> Self {
        let slots = slots.clamp(1, program.max_slots());
        let system = &program.system;
        let scene = SceneBindings::new(device, &program.scene_layout, slots);
        let data = data::DataBindings::new(device, &program.data_layout, system);
        let event_layout = &program.event_layout;
        let ev_count_stride = align_words(event_layout.regions.len().max(1) as u64);
        let ev_data_stride = align_words(event_layout.data_words.max(1));
        let event_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx event counts"),
            size: ev_count_stride * 4 * slots as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx event data"),
            size: ev_data_stride * 4 * slots as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_stride = (slots as u64 * FRAME_BYTES as u64).div_ceil(256) * 256;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mut emitters = Vec::new();
        for ep in &program.emitters {
            let region_words = align_words(ep.capacity as u64 * ep.words as u64);
            let grid_words = align_words(ep.grid_bytes / 4);
            let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let particles = buf("fx particles", region_words * 4 * 2 * slots as u64, storage);
            let grids = buf("fx grids", grid_words * 4 * slots as u64, storage);
            let state = buf(
                "fx emitter state",
                STATE_STRIDE * 4 * slots as u64,
                storage | wgpu::BufferUsages::INDIRECT,
            );
            let args = buf(
                "fx sim dispatch args",
                64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            );
            let frames = buf(
                "fx frames",
                frame_stride * MAX_STEPS_PER_SUBMIT as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            // Zeroed slots: no particles, no events.
            let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx sim args"),
                layout: &program.args_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: args.as_entire_binding(),
                }],
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx sim"),
                layout: &program.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &frames,
                            offset: 0,
                            size: wgpu::BufferSize::new(slots as u64 * FRAME_BYTES as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: event_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: event_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: grids.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &program.iterations,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            });
            emitters.push(BatchEmitter {
                particles,
                state,
                grids,
                frames,
                args,
                bind_group,
                args_bind_group,
                region_words: region_words as u32,
                grid_words: grid_words as u32,
                pipelines: ep.pipelines.clone(),
                scratch: vec![0u8; slots as usize * FRAME_BYTES],
            });
        }
        GpuBatch {
            program: program.clone(),
            slots,
            id: NEXT_BATCH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            emitters,
            event_counts,
            event_data,
            ev_count_stride: ev_count_stride as u32,
            ev_data_stride: ev_data_stride as u32,
            scene,
            data,
            ray_tracing_scene: None,
            free: (0..slots).rev().collect(),
            ring: 0,
            staged: 0,
            frame_stride,
        }
    }

    /// Slots not in use.
    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    /// A new instance in a free slot (none if full). Its counts are cleared
    /// before its first step.
    pub fn alloc(&mut self, seed: u32) -> Option<GpuEffect> {
        let slot = self.free.pop()?;
        let program = &self.program;
        let system = &program.system;
        let region = |buffer: &wgpu::Buffer, words: u64, index: u64| Region {
            buffer: buffer.clone(),
            offset: words * 4 * index,
            size: words * 4,
        };
        let s = slot as u64;
        let emitters = system
            .emitters
            .iter()
            .zip(&program.emitters)
            .zip(&self.emitters)
            .map(|((e, ep), be)| {
                let rw = be.region_words as u64;
                GpuEmitter {
                    name: e.name.clone(),
                    capacity: ep.capacity,
                    words: ep.words,
                    particles: [region(&be.particles, rw, s * 2), region(&be.particles, rw, s * 2 + 1)],
                    grids: region(&be.grids, be.grid_words as u64, s),
                    state: region(&be.state, STATE_STRIDE, s),
                    region_words: be.region_words,
                    pipelines: ep.pipelines.clone(),
                    current: 0,
                    age: 0.0,
                    spawn_remainder: 0.0,
                    scale_remainder: 0.0,
                    shader_source: ep.shader_source.clone(),
                }
            })
            .collect();
        Some(GpuEffect {
            system: system.clone(),
            program: program.clone(),
            active: true,
            spawn_scale: 1.0,
            clear_pending: true,
            seed,
            nonce: fx_core::lifecycle::random_nonce(),
            warmed: false,
            frame: 0,
            time: 0.0,
            user: system.user_params.iter().map(|p| p.value).collect(),
            owner_velocity: [0.0; 3],
            owner_pose: OwnerPose::IDENTITY,
            previous_owner_pose: OwnerPose::IDENTITY,
            emitters,
            event_layout: program.event_layout.clone(),
            event_counts: region(&self.event_counts, self.ev_count_stride as u64, s),
            event_data: region(&self.event_data, self.ev_data_stride as u64, s),
            scene: Region {
                buffer: self.scene.scenes.clone(),
                offset: SCENE_STRIDE * s,
                size: SCENE_STRIDE,
            },
            slot,
            batch_id: self.id,
            own: None,
        })
    }

    /// Gives an instance's slot back (the instance must come from this batch).
    pub fn release(&mut self, effect: GpuEffect) {
        assert_eq!(effect.batch_id, self.id, "instance of another batch");
        self.free.push(effect.slot);
    }

    /// Whether `effect` lives in this batch.
    pub fn holds(&self, effect: &GpuEffect) -> bool {
        effect.batch_id == self.id
    }

    /// Stages one step of `effect` (runs its emitter stacks on the CPU and
    /// writes its frame); [`GpuBatch::record`] then steps every staged
    /// instance at once. A reset instance's buffers are cleared on `encoder`.
    pub fn stage(&mut self, fx: &mut GpuEffect, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        assert_eq!(fx.batch_id, self.id, "instance of another batch");
        assert!(
            self.ring < MAX_STEPS_PER_SUBMIT,
            "more than {MAX_STEPS_PER_SUBMIT} steps recorded without GpuBatch::submitted()"
        );
        if fx.clear_pending {
            // A reset instance: zero its counts, ids, events and grids (no allocation).
            for em in &fx.emitters {
                encoder.clear_buffer(&em.state.buffer, em.state.offset, Some(em.state.size));
                encoder.clear_buffer(&em.grids.buffer, em.grids.offset, Some(em.grids.size));
            }
            let ec = &fx.event_counts;
            encoder.clear_buffer(&ec.buffer, ec.offset, Some(ec.size));
            fx.clear_pending = false;
        }
        let j = self.staged as usize;
        let slot = fx.slot;
        for ei in 0..fx.emitters.len() {
            let ce = &fx.system.emitters[ei];
            let em = &mut fx.emitters[ei];
            let be = &mut self.emitters[ei];
            let seed = fx_core::lifecycle::emitter_seed(fx.seed, fx.nonce, ce.deterministic);
            let values = fx_core::sim::EmitterValues::at(&ce.lifecycle, em.age);
            let env = emitter_cpu::Env {
                user: &fx.user,
                dt,
                time: fx.time,
                emitter: values,
                owner_velocity: fx.owner_velocity,
                lifecycle: &ce.lifecycle,
            };
            let mut request = 0u32;
            if fx.frame == 0 {
                request = request.saturating_add(emitter_cpu::run_stack(
                    &ce.emitter_spawn,
                    emitter_cpu::STAGE_EMITTER_SPAWN,
                    &env,
                    &mut em.spawn_remainder,
                    seed,
                    ei as u32,
                    fx.frame,
                ));
            }
            request = request.saturating_add(emitter_cpu::run_stack(
                &ce.emitter_update,
                emitter_cpu::STAGE_EMITTER_UPDATE,
                &env,
                &mut em.spawn_remainder,
                seed,
                ei as u32,
                fx.frame,
            ));
            if !fx.active {
                request = 0;
            } else if fx.spawn_scale != 1.0 {
                let r = request as f32 * fx.spawn_scale.max(0.0) + em.scale_remainder;
                let whole = r.floor();
                em.scale_remainder = r - whole;
                request = whole as u32;
            }

            // Frame, in sim_common.wgsl's `Frame` layout.
            let s = &mut be.scratch[j * FRAME_BYTES..(j + 1) * FRAME_BYTES];
            let mut put = |w: usize, v: u32| s[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
            put(0, dt.to_bits());
            put(1, fx.time.to_bits());
            put(2, em.age.to_bits());
            put(3, fx.frame);
            put(4, request);
            put(5, seed);
            put(6, ei as u32);
            put(7, em.capacity);
            for (c, v) in fx.owner_velocity.iter().enumerate() {
                put(8 + c, v.to_bits());
            }
            put(11, 0);
            for i in 0..MAX_USER_PARAMS {
                let a = match fx.user.get(i) {
                    None => [0.0; 4],
                    Some(Value::Int(x)) => [f32::from_bits(*x as u32), 0.0, 0.0, 0.0],
                    Some(Value::Bool(b)) => [*b as u32 as f32, 0.0, 0.0, 0.0],
                    Some(v) => v.to_array(),
                };
                for (c, x) in a.iter().enumerate() {
                    put(12 + i * 4 + c, x.to_bits());
                }
            }
            // Emitter State: Emitter.LoopedAge, LoopCount, NormalizedLoopAge.
            put(FRAME_LOOP_WORD, values.looped_age.to_bits());
            put(FRAME_LOOP_WORD + 1, values.loop_count);
            put(FRAME_LOOP_WORD + 2, values.normalized_loop_age.to_bits());
            put(FRAME_LOOP_WORD + 3, 0);
            let rw = be.region_words;
            let cur = em.current as u32;
            put(FRAME_BASE_WORD, (slot * 2 + cur) * rw);
            put(FRAME_BASE_WORD + 1, (slot * 2 + 1 - cur) * rw);
            put(FRAME_BASE_WORD + 2, slot * STATE_STRIDE as u32);
            put(FRAME_BASE_WORD + 3, slot * self.ev_count_stride);
            put(FRAME_BASE_WORD + 4, slot * self.ev_data_stride);
            put(FRAME_BASE_WORD + 5, slot * be.grid_words);
            put(FRAME_BASE_WORD + 6, slot);
            put(FRAME_BASE_WORD + 7, 0);
            // Owner poses (world-space emitters): previous, then current.
            for (q, pose) in [fx.previous_owner_pose, fx.owner_pose].iter().enumerate() {
                let at = FRAME_POSE_WORD + q * 12;
                let words = [
                    pose.translation.extend(0.0).to_array(),
                    pose.rotation.to_array(),
                    pose.scale.extend(0.0).to_array(),
                ];
                for (r, v) in words.iter().flatten().enumerate() {
                    put(at + r, v.to_bits());
                }
            }

            em.current = 1 - em.current;
            em.age += dt;
        }
        fx.time += dt;
        fx.frame += 1;
        self.staged += 1;
    }

    /// Records one step of every instance staged since the last record: one
    /// dispatch per kernel per emitter. Writes the frames with
    /// `queue.write_buffer`, so the commands must be submitted on that queue
    /// before [`MAX_STEPS_PER_SUBMIT`] more records.
    pub fn record(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let n = self.staged;
        if n == 0 {
            return;
        }
        self.staged = 0;
        let offset = self.ring as u64 * self.frame_stride;
        self.ring += 1;
        let inst_groups = n.div_ceil(64);
        for be in &mut self.emitters {
            for j in 0..n as usize {
                let at = j * FRAME_BYTES + (FRAME_BASE_WORD + 7) * 4;
                be.scratch[at..at + 4].copy_from_slice(&n.to_le_bytes());
            }
            queue.write_buffer(&be.frames, offset, &be.scratch[..n as usize * FRAME_BYTES]);
        }
        for be in &self.emitters {
            let p = &be.pipelines;
            let bg = &be.bind_group;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            pass.set_bind_group(1, &be.args_bind_group, &[]);
            pass.set_pipeline(&p.begin);
            pass.dispatch_workgroups(inst_groups, 1, 1);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim update"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            match (&p.update_rt, &self.ray_tracing_scene) {
                (Some(update_rt), Some(rt)) => {
                    pass.set_bind_group(1, &self.data.bind_group, &[]);
                    pass.set_bind_group(2, &self.scene.bind_group, &[]);
                    pass.set_bind_group(3, rt, &[]);
                    pass.set_pipeline(update_rt);
                }
                _ => {
                    if p.uses_scene {
                        pass.set_bind_group(1, &self.data.bind_group, &[]);
                        pass.set_bind_group(2, &self.scene.bind_group, &[]);
                    }
                    pass.set_pipeline(&p.update);
                }
            }
            pass.dispatch_workgroups_indirect(&be.args, 0);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim spawn prep"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            pass.set_bind_group(1, &be.args_bind_group, &[]);
            pass.set_pipeline(&p.spawn_prep);
            pass.dispatch_workgroups(inst_groups, 1, 1);
            drop(pass);
            if let Some(rank) = &p.event_rank {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fx sim event rank"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bg, &[offset as u32, 0]);
                pass.set_pipeline(rank);
                pass.dispatch_workgroups_indirect(&be.args, 32);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim spawn"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            if p.uses_scene {
                pass.set_bind_group(1, &self.data.bind_group, &[]);
                pass.set_bind_group(2, &self.scene.bind_group, &[]);
            }
            pass.set_pipeline(&p.spawn);
            pass.dispatch_workgroups_indirect(&be.args, 16);
            pass.set_bind_group(1, &be.args_bind_group, &[]);
            pass.set_pipeline(&p.finish);
            pass.dispatch_workgroups(inst_groups, 1, 1);
            drop(pass);
            if !p.stages.is_empty() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fx sim stages"),
                    timestamp_writes: None,
                });
                if p.uses_scene {
                    pass.set_bind_group(1, &self.data.bind_group, &[]);
                    pass.set_bind_group(2, &self.scene.bind_group, &[]);
                }
                for (pipeline, k) in &p.stages {
                    pass.set_pipeline(pipeline);
                    for j in 0..k.iterations as u64 {
                        pass.set_bind_group(0, bg, &[offset as u32, (j * self.program.iteration_stride) as u32]);
                        match k.groups {
                            Some([x, y, z]) => pass.dispatch_workgroups(x, y, z * n),
                            None => pass.dispatch_workgroups_indirect(&be.args, 48),
                        }
                    }
                }
            }
        }
    }

    /// Call after each queue submission that contained records: frees the
    /// per-step frame slots.
    pub fn submitted(&mut self) {
        self.ring = 0;
    }

    /// Records left before [`GpuBatch::submitted`] must be called.
    pub fn steps_left_in_submit(&self) -> u32 {
        MAX_STEPS_PER_SUBMIT - self.ring
    }

    /// Runs `effect`'s warm-up alone (see [`GpuEffect::warm_up`]), submitting
    /// its own command buffers. Call it with nothing staged or recorded since
    /// the last [`GpuBatch::submitted`].
    pub fn warm_up(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, fx: &mut GpuEffect) {
        if !fx.needs_warm_up() {
            fx.warmed = true;
            return;
        }
        fx.warmed = true;
        let delta = fx.system.warmup_tick_delta;
        let mut left = fx_core::lifecycle::warmup_steps(&fx.system);
        // The owner stands where it starts during the warm-up (as on the CPU).
        let previous = fx.previous_owner_pose;
        fx.previous_owner_pose = fx.owner_pose;
        while left > 0 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fx warm-up"),
            });
            let n = left.min(self.steps_left_in_submit());
            for _ in 0..n {
                self.stage(fx, &mut enc, delta);
                self.record(queue, &mut enc);
            }
            queue.submit([enc.finish()]);
            self.submitted();
            left -= n;
        }
        fx.previous_owner_pose = previous;
    }

    /// The layout of the scene bind group (group 2), for hosts that build it.
    pub fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scene.layout
    }

    /// Binds the depth texture SceneDepth collisions read
    /// ([`SCENE_DEPTH_FORMAT`], reverse-Z), or none, for every instance.
    /// Call when the texture changes; [`GpuEffect::set_scene_view`] says
    /// which view it shows.
    pub fn set_scene_depth(&mut self, device: &wgpu::Device, depth: Option<&wgpu::TextureView>) {
        let s = &mut self.scene;
        s.bind_group = SceneBindings::bind(device, &s.layout, &s.scenes, depth.unwrap_or(&s.dummy_depth));
    }

    /// Binds the ray-tracing scene (a bind group of the layout given in
    /// [`GpuOptions::ray_tracing_layout`]), or none: RayTraced collisions
    /// then use the depth buffer.
    pub fn set_ray_tracing_scene(&mut self, bind_group: Option<wgpu::BindGroup>) {
        self.ray_tracing_scene = bind_group;
    }

    /// Binds mesh data interface `name` for every instance (see [`GpuEffect::set_mesh`]).
    pub fn set_mesh(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        mesh: &fx_core::data::MeshData,
        pose: fx_core::data::MeshPose,
    ) -> bool {
        self.data.set_mesh(device, &self.program.system, name, mesh, pose)
    }

    /// See [`GpuEffect::set_mesh_pose`].
    pub fn set_mesh_pose(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        matrices: &[bevy_math::Mat4],
        interval: f32,
    ) -> bool {
        self.data
            .set_mesh_pose(queue, &self.program.system, name, matrices, interval)
    }

    /// See [`GpuEffect::set_texture`].
    pub fn set_texture(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        texture: &fx_core::data::TextureData,
    ) -> bool {
        self.data.set_texture(device, &self.program.system, name, texture)
    }

    /// The particle buffer of emitter `emitter` (every slot's two copies;
    /// see [`GpuEmitter::particles`]).
    pub fn particles(&self, emitter: usize) -> &wgpu::Buffer {
        &self.emitters[emitter].particles
    }

    /// The state buffer of emitter `emitter` ([`STATE_STRIDE`] words per slot).
    pub fn state(&self, emitter: usize) -> &wgpu::Buffer {
        &self.emitters[emitter].state
    }

    /// Words between particle copies of emitter `emitter`.
    pub fn region_words(&self, emitter: usize) -> u32 {
        self.emitters[emitter].region_words
    }

    /// GPU bytes held by the batch's buffers.
    pub fn gpu_bytes(&self) -> u64 {
        self.emitters
            .iter()
            .map(|e| e.particles.size() + e.state.size() + e.frames.size() + e.grids.size() + e.args.size())
            .sum::<u64>()
            + self.event_counts.size()
            + self.event_data.size()
            + self.scene.scenes.size()
            + self.data.bytes()
    }
}

impl GpuEffect {
    /// Compiles every emitter and allocates its pooled buffers.
    pub fn new(device: &wgpu::Device, system: &CookedSystem, seed: u32) -> Result<Self, String> {
        Self::with_options(device, system, seed, GpuOptions::default())
    }

    /// [`GpuEffect::new`] with what the host offers (a ray-tracing scene).
    pub fn with_options(
        device: &wgpu::Device,
        system: &CookedSystem,
        seed: u32,
        options: GpuOptions,
    ) -> Result<Self, String> {
        let program = GpuProgram::new(device, Arc::new(system.clone()), options)?;
        Ok(Self::from_program(device, &Arc::new(program), seed))
    }

    /// A new instance of a compiled program with a batch of its own:
    /// allocates its buffers and bind groups only (no shader compilation).
    pub fn from_program(device: &wgpu::Device, program: &Arc<GpuProgram>, seed: u32) -> Self {
        let mut batch = GpuBatch::new(device, program, 1);
        let mut fx = batch.alloc(seed).expect("a new batch has a free slot");
        fx.clear_pending = false;
        fx.own = Some(Box::new(batch));
        fx
    }

    fn own(&mut self) -> &mut GpuBatch {
        self.own
            .as_deref_mut()
            .expect("an instance of a shared batch is stepped through its GpuBatch")
    }

    /// Whether [`GpuEffect::warm_up`] still has steps to run.
    pub fn needs_warm_up(&self) -> bool {
        !self.warmed && self.frame == 0 && fx_core::lifecycle::warmup_steps(&self.system) > 0
    }

    /// Runs the system's warm-up (`warmup_time` in steps of
    /// `warmup_tick_delta`, as the CPU simulation does before its first
    /// step), submitting its own command buffers. Call it before the first
    /// [`GpuEffect::step`] (and after a [`GpuEffect::reset`]), with no steps
    /// recorded since the last [`GpuEffect::submitted`]. Does nothing if it
    /// ran. Instances of a shared batch use [`GpuBatch::warm_up`].
    pub fn warm_up(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut batch = self.own.take().expect("an instance of a shared batch warms up through its GpuBatch");
        batch.warm_up(device, queue, self);
        self.own = Some(batch);
    }

    /// The owner moved to `world_from_local` over the next step (see
    /// [`fx_core::sim::Simulation::move_owner`]).
    pub fn move_owner(&mut self, world_from_local: bevy_math::Affine3A) {
        self.previous_owner_pose = self.owner_pose;
        self.owner_pose = OwnerPose::from_affine(world_from_local);
    }

    /// The owner jumped to `world_from_local`: no step interpolates from where it was.
    pub fn teleport_owner(&mut self, world_from_local: bevy_math::Affine3A) {
        self.owner_pose = OwnerPose::from_affine(world_from_local);
        self.previous_owner_pose = self.owner_pose;
    }

    /// Restarts the effect as if just created with `seed` (pooling: reuse an
    /// instance instead of building a new one). No allocation: the next
    /// step clears the GPU counts, events and grids with buffer clears before
    /// it runs. `User.` values go back to their defaults; data interface
    /// bindings are kept.
    pub fn reset(&mut self, seed: u32) {
        self.seed = seed;
        self.nonce = fx_core::lifecycle::random_nonce();
        self.warmed = false;
        self.frame = 0;
        self.time = 0.0;
        self.active = true;
        self.spawn_scale = 1.0;
        self.owner_velocity = [0.0; 3];
        self.previous_owner_pose = self.owner_pose;
        for (u, p) in self.user.iter_mut().zip(&self.system.user_params) {
            *u = p.value;
        }
        for em in &mut self.emitters {
            em.current = 0;
            em.age = 0.0;
            em.spawn_remainder = 0.0;
            em.scale_remainder = 0.0;
        }
        self.clear_pending = true;
    }

    /// Sets a `User.` parameter. Returns false if it is not declared or the type differs.
    pub fn set_user(&mut self, name: &str, value: Value) -> bool {
        match self.system.user_params.iter().position(|p| p.name == name) {
            Some(i) if self.user[i].ty() == value.ty() => {
                self.user[i] = value;
                true
            }
            _ => false,
        }
    }

    /// Binds mesh data interface `name` (SkeletalMesh or StaticMesh) to
    /// `mesh`, posed by `pose` (see `fx_core::data::MeshPose`). Uploads the
    /// mesh: call when the mesh changes, not every frame. Returns false if no
    /// such data interface is declared. Data interfaces belong to the batch:
    /// for an instance of a shared batch use [`GpuBatch::set_mesh`].
    pub fn set_mesh(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        mesh: &fx_core::data::MeshData,
        pose: fx_core::data::MeshPose,
    ) -> bool {
        self.own().set_mesh(device, name, mesh, pose)
    }

    /// Moves a bound mesh to a new pose, `interval` seconds after its last
    /// one: one matrix per joint (skinned) or one (static), from mesh space to
    /// the effect's simulation space. Writes in place; call every frame the
    /// mesh moves. Returns false if `name` is not a bound mesh.
    pub fn set_mesh_pose(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        matrices: &[bevy_math::Mat4],
        interval: f32,
    ) -> bool {
        self.own().set_mesh_pose(queue, name, matrices, interval)
    }

    /// Binds texture data interface `name`. Returns false if it is not declared.
    pub fn set_texture(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        texture: &fx_core::data::TextureData,
    ) -> bool {
        self.own().set_texture(device, name, texture)
    }

    /// Records one simulation step of every emitter of an instance made on
    /// its own ([`GpuEffect::new`]). Writes the step's frame with
    /// `queue.write_buffer`, so the commands must be submitted on that queue
    /// before the next [`MAX_STEPS_PER_SUBMIT`] steps.
    pub fn step(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        let mut batch = self.own.take().expect("an instance of a shared batch steps through its GpuBatch");
        batch.stage(self, encoder, dt);
        batch.record(queue, encoder);
        self.own = Some(batch);
    }

    /// Call after each queue submission that contained steps: frees the
    /// per-step uniform slots.
    pub fn submitted(&mut self) {
        if let Some(b) = &mut self.own {
            b.submitted();
        }
    }

    /// Steps left before [`GpuEffect::submitted`] must be called.
    pub fn steps_left_in_submit(&self) -> u32 {
        self.own.as_ref().map_or(MAX_STEPS_PER_SUBMIT, |b| b.steps_left_in_submit())
    }

    /// Whether an emitter's update reads the scene (SceneDepth or RayTraced collisions).
    pub fn uses_scene(&self) -> bool {
        self.emitters.iter().any(|e| e.pipelines.uses_scene)
    }

    /// Whether an emitter traces rays when a ray-tracing scene is bound.
    pub fn uses_ray_tracing(&self) -> bool {
        self.emitters.iter().any(|e| e.pipelines.update_rt.is_some())
    }

    /// The layout of the scene bind group (group 2), for hosts that build it.
    pub fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        &self.program.scene_layout
    }

    /// Binds the depth texture of an instance made on its own (see [`GpuBatch::set_scene_depth`]).
    pub fn set_scene_depth(&mut self, device: &wgpu::Device, depth: Option<&wgpu::TextureView>) {
        self.own().set_scene_depth(device, depth);
    }

    /// Sets where the effect is (`world_from_local`) and the camera of the
    /// bound depth texture (`None`: no depth, SceneDepth collisions find
    /// nothing). Writes the instance's scene entry; call before recording steps.
    pub fn set_scene_view(
        &mut self,
        queue: &wgpu::Queue,
        world_from_local: bevy_math::Mat4,
        view: Option<&SceneView>,
    ) {
        use bevy_math::{Mat4, Vec4};
        let local_from_world = world_from_local.inverse();
        let (clip_from_local, camera, size, flags, forward) = match view {
            Some(v) => {
                // The view direction: from the near plane's centre deeper in
                // (reverse-Z: 1 is near), for perspective and orthographic alike.
                let world_from_clip = v.clip_from_world.inverse();
                let near = world_from_clip.project_point3(bevy_math::Vec3::new(0.0, 0.0, 1.0));
                let deeper = world_from_clip.project_point3(bevy_math::Vec3::new(0.0, 0.0, 0.5));
                let forward = local_from_world
                    .transform_vector3(deeper - near)
                    .try_normalize()
                    .unwrap_or(bevy_math::Vec3::NEG_Z);
                (
                    v.clip_from_world * world_from_local,
                    local_from_world.transform_point3(v.camera_position),
                    [v.size[0] as f32, v.size[1] as f32],
                    1u32,
                    forward,
                )
            }
            None => (Mat4::IDENTITY, bevy_math::Vec3::ZERO, [1.0, 1.0], 0, bevy_math::Vec3::NEG_Z),
        };
        let mut b = [0u8; SCENE_BYTES];
        let mut w = 0usize;
        let mut put = |x: f32| {
            b[w * 4..w * 4 + 4].copy_from_slice(&x.to_le_bytes());
            w += 1;
        };
        for m in [
            clip_from_local,
            clip_from_local.inverse(),
            world_from_local,
            local_from_world,
        ] {
            m.to_cols_array().into_iter().for_each(&mut put);
        }
        Vec4::from((camera, 1.0)).to_array().into_iter().for_each(&mut put);
        put(size[0]);
        put(size[1]);
        put(f32::from_bits(flags));
        put(0.0);
        Vec4::from((forward, 0.0)).to_array().into_iter().for_each(&mut put);
        queue.write_buffer(&self.scene.buffer, self.scene.offset, &b);
    }

    /// Binds the ray-tracing scene of an instance made on its own (see [`GpuBatch::set_ray_tracing_scene`]).
    pub fn set_ray_tracing_scene(&mut self, bind_group: Option<wgpu::BindGroup>) {
        self.own().set_ray_tracing_scene(bind_group);
    }

    /// Events of `kind` that emitter `emitter` wrote in its last update, in
    /// sorted order (tests and tools only: waits for the GPU).
    pub fn read_events(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        emitter: usize,
        kind: EventKind,
    ) -> Vec<Event> {
        let Some(r) = self.event_layout.region(emitter, kind) else {
            return Vec::new();
        };
        let ec = &self.event_counts;
        let counts = read_buffer(device, queue, &ec.buffer, ec.offset, ec.size);
        let counts: &[u32] = bytemuck::cast_slice(&counts);
        let n = counts[r.index as usize].min(r.capacity) as usize;
        let ed = &self.event_data;
        let data = read_buffer(device, queue, &ed.buffer, ed.offset, ed.size);
        let d: &[u32] = bytemuck::cast_slice(&data);
        let f = |i: usize| f32::from_bits(d[i]);
        (0..n)
            .map(|rank| {
                let b = r.data as usize + d[r.sorted as usize + rank] as usize * EVENT_WORDS as usize;
                Event {
                    position: [f(b), f(b + 1), f(b + 2)],
                    velocity: [f(b + 3), f(b + 4), f(b + 5)],
                    normal: [f(b + 6), f(b + 7), f(b + 8)],
                    age: f(b + 9),
                    particle_id: d[b + 10],
                }
            })
            .collect()
    }

    /// Reads every emitter's living particles back (tests and tools only:
    /// allocates and waits for the GPU).
    pub fn read_particles(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec<Particle>> {
        let mut out = Vec::new();
        for em in &self.emitters {
            let state = read_buffer(device, queue, &em.state.buffer, em.state.offset, STATE_WORDS * 4);
            let words: &[u32] = bytemuck::cast_slice(&state);
            let n = words[STATE_COUNT as usize] as usize;
            let cap = em.capacity as usize;
            let data = read_buffer(
                device,
                queue,
                &em.particles[em.current].buffer,
                em.particles[em.current].offset,
                em.particles[em.current].size,
            );
            let d: &[f32] = bytemuck::cast_slice(&data);
            let w = |word: usize, i: usize| d[word * cap + i];
            let mut ps = Vec::with_capacity(n);
            for i in 0..n {
                let mut p = Particle::new(w(27, i).to_bits());
                p.position = [w(0, i), w(1, i), w(2, i)].into();
                p.velocity = [w(3, i), w(4, i), w(5, i)].into();
                p.color = [w(6, i), w(7, i), w(8, i), w(9, i)].into();
                p.sprite_size = [w(10, i), w(11, i)].into();
                p.sprite_rotation = w(12, i);
                p.mass = w(13, i);
                p.lifetime = w(14, i);
                p.age = w(15, i);
                p.normalized_age = w(16, i);
                p.initial_color = [w(17, i), w(18, i), w(19, i), w(20, i)].into();
                p.initial_sprite_size = [w(21, i), w(22, i)].into();
                p.force = [w(23, i), w(24, i), w(25, i)].into();
                p.drag = w(26, i);
                if em.words >= WORDS_PER_PARTICLE_EXTENDED {
                    p.scale = [w(28, i), w(29, i), w(30, i)].into();
                    p.mesh_orientation =
                        bevy_math::Quat::from_xyzw(w(31, i), w(32, i), w(33, i), w(34, i));
                    p.ribbon_id = w(35, i).to_bits() as i32;
                    p.ribbon_width = w(36, i);
                    p.ribbon_twist = w(37, i);
                    p.ribbon_link_order = w(38, i);
                    p.light_radius = w(39, i);
                    p.dynamic_material_parameter = [w(40, i), w(41, i), w(42, i), w(43, i)].into();
                    p.sub_image_index = w(44, i);
                    p.initial_scale = [w(45, i), w(46, i), w(47, i)].into();
                    p.initial_ribbon_width = w(48, i);
                    p.camera_offset = w(49, i);
                    p.sprite_facing = [w(50, i), w(51, i), w(52, i)].into();
                    p.sprite_alignment = [w(53, i), w(54, i), w(55, i)].into();
                    p.dynamic_material_parameter_1 = [w(56, i), w(57, i), w(58, i), w(59, i)].into();
                    p.rotational_velocity = [w(60, i), w(61, i), w(62, i)].into();
                    p.rotational_force = [w(63, i), w(64, i), w(65, i)].into();
                    p.rotational_drag = w(66, i);
                }
                if em.words >= WORDS_PER_PARTICLE_EXTENDED_2 {
                    p.dynamic_material_parameter_2 = [w(67, i), w(68, i), w(69, i), w(70, i)].into();
                    p.dynamic_material_parameter_3 = [w(71, i), w(72, i), w(73, i), w(74, i)].into();
                }
                ps.push(p);
            }
            out.push(ps);
        }
        out
    }

    /// `(living particles, next UniqueID)` of each emitter, read back (tests only).
    pub fn read_counts(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<(u32, u32)> {
        self.emitters
            .iter()
            .map(|em| {
                let state = read_buffer(device, queue, &em.state.buffer, em.state.offset, STATE_WORDS * 4);
                let words: &[u32] = bytemuck::cast_slice(&state);
                (words[STATE_COUNT as usize], words[STATE_NEXT_ID])
            })
            .collect()
    }

    /// GPU bytes of this instance's parts of its batch (its whole batch when made on its own).
    pub fn gpu_bytes(&self) -> u64 {
        if let Some(b) = &self.own {
            return b.gpu_bytes();
        }
        self.emitters
            .iter()
            .map(|e| e.particles[0].size * 2 + e.state.size + e.grids.size)
            .sum::<u64>()
            + self.event_counts.size
            + self.event_data.size
            + self.scene.size
    }
}

/// Copies a buffer range to the CPU and waits (tests and tools only).
pub fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fx readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll failed");
    let v = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    v
}
