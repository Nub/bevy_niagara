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
pub mod lights;
pub mod ribbon;
pub mod view;

use std::borrow::Cow;
use std::sync::Arc;

use fx_core::cook::CookedSystem;
use fx_core::events::{EVENT_WORDS, Event, EventKind};
use fx_core::sim::Particle;

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
/// CameraOffset (49).
pub const WORDS_PER_PARTICLE_EXTENDED: u32 = 50;
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

const FRAME_BYTES: usize = 48 + MAX_USER_PARAMS * 16;
const FRAME_SLOT: u64 = 512;
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

/// One emitter's GPU state.
pub struct GpuEmitter {
    pub name: String,
    pub capacity: u32,
    /// Words per particle: [`WORDS_PER_PARTICLE`] or [`WORDS_PER_PARTICLE_EXTENDED`].
    pub words: u32,
    /// Ping-pong particle buffers; `particles[current]` holds the living ones.
    pub particles: [wgpu::Buffer; 2],
    /// Grid collections ([`fx_core::custom`]): each grid twice, ping-pong.
    pub grids: wgpu::Buffer,
    /// Counts, ids and indirect dispatch arguments ([`STATE_WORDS`] u32s).
    pub state: wgpu::Buffer,
    frame_ubo: wgpu::Buffer,
    /// `[0]` reads buffer 0 and writes 1; `[1]` the reverse.
    bind_groups: [wgpu::BindGroup; 2],
    args: wgpu::Buffer,
    args_bind_group: wgpu::BindGroup,
    pipelines: Arc<Pipelines>,
    pub current: usize,
    age: f32,
    spawn_remainder: f32,
    /// Fraction of a particle carried over by the spawn-count scale.
    scale_remainder: f32,
    /// Generated WGSL, for inspection.
    pub shader_source: Arc<str>,
}

/// A cooked system simulated on the GPU.
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
    /// Steps taken so far.
    pub frame: u32,
    /// Simulated seconds (`Engine.Time`).
    pub time: f32,
    pub user: Vec<Value>,
    /// `Engine.Owner.Velocity` (m/s), read by Inherit Velocity; set by the host.
    pub owner_velocity: [f32; 3],
    pub emitters: Vec<GpuEmitter>,
    /// Where each emitter's events live.
    pub event_layout: EventLayout,
    /// One atomic counter per event region.
    pub event_counts: wgpu::Buffer,
    /// The events and their sorted order.
    pub event_data: wgpu::Buffer,
    scene: SceneBindings,
    data: data::DataBindings,
    /// The ray-tracing scene bind group (group 3), when RayTraced collisions trace rays.
    ray_tracing_scene: Option<wgpu::BindGroup>,
    slot: u32,
    scratch: [u8; FRAME_BYTES],
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
    uniform: wgpu::Buffer,
    dummy_depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    world_from_local: bevy_math::Mat4,
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
                        ty: wgpu::BufferBindingType::Uniform,
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

    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let layout = layout.clone();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx scene"),
            contents: &[0u8; SCENE_BYTES],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            uniform,
            dummy_depth,
            bind_group,
            world_from_local: bevy_math::Mat4::IDENTITY,
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(FRAME_BYTES as u64),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
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
            let words = if e.uses_extended_attributes() {
                WORDS_PER_PARTICLE_EXTENDED
            } else {
                WORDS_PER_PARTICLE
            };
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
        })
    }

    /// GPU bytes one instance of this program allocates (before data interfaces).
    pub fn instance_bytes(&self) -> u64 {
        self.emitters
            .iter()
            .map(|e| {
                e.capacity as u64 * e.words as u64 * 4 * 2
                    + STATE_WORDS * 4
                    + FRAME_SLOT * MAX_STEPS_PER_SUBMIT as u64
                    + e.grid_bytes
            })
            .sum::<u64>()
            + self.event_layout.regions.len().max(1) as u64 * 4
            + self.event_layout.data_words.max(1) * 4
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

    /// A new instance of a compiled program: allocates its buffers and bind
    /// groups only (no shader compilation).
    pub fn from_program(device: &wgpu::Device, program: &Arc<GpuProgram>, seed: u32) -> Self {
        let system = &program.system;
        let scene = SceneBindings::new(device, &program.scene_layout);
        let data = data::DataBindings::new(device, &program.data_layout, system);
        let event_layout = program.event_layout.clone();
        let event_counts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx event counts"),
            contents: &vec![0u8; event_layout.regions.len().max(1) * 4],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let event_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx event data"),
            size: event_layout.data_words.max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut emitters = Vec::new();
        for (e, ep) in system.emitters.iter().zip(&program.emitters) {
            let (capacity, words) = (ep.capacity, ep.words);
            let bytes = capacity as u64 * words as u64 * 4;
            let particle_buffer = |i: usize| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 {
                        "fx particles 0"
                    } else {
                        "fx particles 1"
                    }),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let particles = [particle_buffer(0), particle_buffer(1)];
            let grids = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fx grids"),
                contents: &vec![0u8; ep.grid_bytes as usize],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
            let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fx emitter state"),
                contents: &[0u8; STATE_WORDS as usize * 4],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
            let args = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx sim dispatch args"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx sim args"),
                layout: &program.args_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: args.as_entire_binding(),
                }],
            });
            let frame_ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx frame uniforms"),
                size: FRAME_SLOT * MAX_STEPS_PER_SUBMIT as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = |s: usize| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fx sim"),
                    layout: &program.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &frame_ubo,
                                offset: 0,
                                size: wgpu::BufferSize::new(FRAME_BYTES as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: particles[s].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: particles[1 - s].as_entire_binding(),
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
                })
            };
            let bind_groups = [bind_group(0), bind_group(1)];
            emitters.push(GpuEmitter {
                name: e.name.clone(),
                capacity,
                words,
                particles,
                grids,
                state,
                frame_ubo,
                bind_groups,
                args,
                args_bind_group,
                pipelines: ep.pipelines.clone(),
                current: 0,
                age: 0.0,
                spawn_remainder: 0.0,
                scale_remainder: 0.0,
                shader_source: ep.shader_source.clone(),
            });
        }
        GpuEffect {
            system: system.clone(),
            program: program.clone(),
            active: true,
            spawn_scale: 1.0,
            clear_pending: false,
            seed,
            frame: 0,
            time: 0.0,
            user: system.user_params.iter().map(|p| p.value).collect(),
            owner_velocity: [0.0; 3],
            emitters,
            event_layout,
            event_counts,
            event_data,
            scene,
            data,
            ray_tracing_scene: None,
            slot: 0,
            scratch: [0; FRAME_BYTES],
        }
    }

    /// Restarts the effect as if just created with `seed` (pooling: reuse an
    /// instance instead of building a new one). No allocation: the next
    /// [`GpuEffect::step`] clears the GPU counts, events and grids with
    /// buffer clears before it runs. `User.` values go back to their
    /// defaults; data interface bindings are kept.
    pub fn reset(&mut self, seed: u32) {
        self.seed = seed;
        self.frame = 0;
        self.time = 0.0;
        self.active = true;
        self.spawn_scale = 1.0;
        self.owner_velocity = [0.0; 3];
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
    /// such data interface is declared.
    pub fn set_mesh(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        mesh: &fx_core::data::MeshData,
        pose: fx_core::data::MeshPose,
    ) -> bool {
        self.data.set_mesh(device, &self.system, name, mesh, pose)
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
        self.data
            .set_mesh_pose(queue, &self.system, name, matrices, interval)
    }

    /// Binds texture data interface `name`. Returns false if it is not declared.
    pub fn set_texture(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        texture: &fx_core::data::TextureData,
    ) -> bool {
        self.data.set_texture(device, &self.system, name, texture)
    }

    /// Call after each queue submission that contained steps: frees the
    /// per-step uniform slots.
    pub fn submitted(&mut self) {
        self.slot = 0;
    }

    /// Steps left before [`GpuEffect::submitted`] must be called.
    pub fn steps_left_in_submit(&self) -> u32 {
        MAX_STEPS_PER_SUBMIT - self.slot
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
        &self.scene.layout
    }

    /// Binds the depth texture SceneDepth collisions read
    /// ([`SCENE_DEPTH_FORMAT`], reverse-Z), or none. Call when the texture
    /// changes; [`GpuEffect::set_scene_view`] says which view it shows.
    pub fn set_scene_depth(&mut self, device: &wgpu::Device, depth: Option<&wgpu::TextureView>) {
        let s = &mut self.scene;
        s.bind_group = SceneBindings::bind(
            device,
            &s.layout,
            &s.uniform,
            depth.unwrap_or(&s.dummy_depth),
        );
    }

    /// Sets where the effect is (`world_from_local`) and the camera of the
    /// bound depth texture (`None`: no depth, SceneDepth collisions find
    /// nothing). Writes the scene uniform; call before recording steps.
    pub fn set_scene_view(
        &mut self,
        queue: &wgpu::Queue,
        world_from_local: bevy_math::Mat4,
        view: Option<&SceneView>,
    ) {
        use bevy_math::{Mat4, Vec4};
        self.scene.world_from_local = world_from_local;
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
        queue.write_buffer(&self.scene.uniform, 0, &b);
    }

    /// Binds the ray-tracing scene (a bind group of the layout given in
    /// [`GpuOptions::ray_tracing_layout`]), or none: RayTraced collisions
    /// then use the depth buffer.
    pub fn set_ray_tracing_scene(&mut self, bind_group: Option<wgpu::BindGroup>) {
        self.ray_tracing_scene = bind_group;
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
        let counts = read_buffer(device, queue, &self.event_counts, 0, self.event_counts.size());
        let counts: &[u32] = bytemuck::cast_slice(&counts);
        let n = counts[r.index as usize].min(r.capacity) as usize;
        let data = read_buffer(device, queue, &self.event_data, 0, self.event_data.size());
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

    /// Records one simulation step of every emitter. Writes the step's
    /// uniforms with `queue.write_buffer`, so the commands must be submitted
    /// on that queue before the next [`MAX_STEPS_PER_SUBMIT`] steps.
    pub fn step(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        assert!(
            self.slot < MAX_STEPS_PER_SUBMIT,
            "more than {MAX_STEPS_PER_SUBMIT} steps recorded without GpuEffect::submitted()"
        );
        let offset = self.slot as u64 * FRAME_SLOT;
        if self.clear_pending {
            // A reset instance: zero its counts, ids, events and grids (no allocation).
            for em in &self.emitters {
                encoder.clear_buffer(&em.state, 0, None);
                encoder.clear_buffer(&em.grids, 0, None);
            }
            encoder.clear_buffer(&self.event_counts, 0, None);
            self.clear_pending = false;
        }
        for ei in 0..self.emitters.len() {
            let ce = &self.system.emitters[ei];
            let em = &mut self.emitters[ei];
            let env = emitter_cpu::Env {
                user: &self.user,
                dt,
                time: self.time,
                emitter_age: em.age,
                owner_velocity: self.owner_velocity,
            };
            let mut request = 0u32;
            if self.frame == 0 {
                request = request.saturating_add(emitter_cpu::run_stack(
                    &ce.emitter_spawn,
                    emitter_cpu::STAGE_EMITTER_SPAWN,
                    &env,
                    &mut em.spawn_remainder,
                    self.seed,
                    ei as u32,
                    self.frame,
                ));
            }
            request = request.saturating_add(emitter_cpu::run_stack(
                &ce.emitter_update,
                emitter_cpu::STAGE_EMITTER_UPDATE,
                &env,
                &mut em.spawn_remainder,
                self.seed,
                ei as u32,
                self.frame,
            ));
            if !self.active {
                request = 0;
            } else if self.spawn_scale != 1.0 {
                let r = request as f32 * self.spawn_scale.max(0.0) + em.scale_remainder;
                let whole = r.floor();
                em.scale_remainder = r - whole;
                request = whole as u32;
            }

            // Frame uniform, in sim_common.wgsl's `Frame` layout.
            let s = &mut self.scratch;
            let mut put = |w: usize, v: u32| s[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
            put(0, dt.to_bits());
            put(1, self.time.to_bits());
            put(2, em.age.to_bits());
            put(3, self.frame);
            put(4, request);
            put(5, self.seed);
            put(6, ei as u32);
            put(7, em.capacity);
            for (c, v) in self.owner_velocity.iter().enumerate() {
                put(8 + c, v.to_bits());
            }
            put(11, 0);
            for (i, v) in self.user.iter().enumerate() {
                let a = match *v {
                    Value::Int(x) => [f32::from_bits(x as u32), 0.0, 0.0, 0.0],
                    Value::Bool(b) => [b as u32 as f32, 0.0, 0.0, 0.0],
                    v => v.to_array(),
                };
                for (c, x) in a.iter().enumerate() {
                    put(12 + i * 4 + c, x.to_bits());
                }
            }
            queue.write_buffer(&em.frame_ubo, offset, &self.scratch);

            let p = &em.pipelines;
            let bg = &em.bind_groups[em.current];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            pass.set_bind_group(1, &em.args_bind_group, &[]);
            pass.set_pipeline(&p.begin);
            pass.dispatch_workgroups(1, 1, 1);
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
            pass.dispatch_workgroups_indirect(&em.args, 0);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim spawn prep"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32, 0]);
            pass.set_bind_group(1, &em.args_bind_group, &[]);
            pass.set_pipeline(&p.spawn_prep);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            if let Some(rank) = &p.event_rank {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fx sim event rank"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bg, &[offset as u32, 0]);
                pass.set_pipeline(rank);
                pass.dispatch_workgroups_indirect(&em.args, 32);
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
            pass.dispatch_workgroups_indirect(&em.args, 16);
            pass.set_bind_group(1, &em.args_bind_group, &[]);
            pass.set_pipeline(&p.finish);
            pass.dispatch_workgroups(1, 1, 1);
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
                            Some([x, y, z]) => pass.dispatch_workgroups(x, y, z),
                            None => pass.dispatch_workgroups_indirect(&em.args, 48),
                        }
                    }
                }
            }

            em.current = 1 - em.current;
            em.age += dt;
        }
        self.time += dt;
        self.frame += 1;
        self.slot += 1;
    }

    /// Reads every emitter's living particles back (tests and tools only:
    /// allocates and waits for the GPU).
    pub fn read_particles(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec<Particle>> {
        let mut out = Vec::new();
        for em in &self.emitters {
            let state = read_buffer(device, queue, &em.state, 0, STATE_WORDS * 4);
            let words: &[u32] = bytemuck::cast_slice(&state);
            let n = words[STATE_COUNT as usize] as usize;
            let cap = em.capacity as usize;
            let data = read_buffer(
                device,
                queue,
                &em.particles[em.current],
                0,
                em.particles[0].size(),
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
                if em.words == WORDS_PER_PARTICLE_EXTENDED {
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
                let state = read_buffer(device, queue, &em.state, 0, STATE_WORDS * 4);
                let words: &[u32] = bytemuck::cast_slice(&state);
                (words[STATE_COUNT as usize], words[STATE_NEXT_ID])
            })
            .collect()
    }

    /// GPU bytes held by this effect's buffers.
    pub fn gpu_bytes(&self) -> u64 {
        self.emitters
            .iter()
            .map(|e| e.particles[0].size() * 2 + e.state.size() + e.frame_ubo.size() + e.grids.size())
            .sum::<u64>()
            + self.event_counts.size()
            + self.event_data.size()
            + self.data.bytes()
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
