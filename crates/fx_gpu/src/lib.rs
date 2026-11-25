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
mod emitter_cpu;
pub mod lights;
pub mod ribbon;
pub mod view;

use std::borrow::Cow;

use fx_core::cook::CookedSystem;
use fx_core::sim::Particle;
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
/// SubImageIndex (44), Initial.Scale (45..47) and Initial.RibbonWidth (48).
pub const WORDS_PER_PARTICLE_EXTENDED: u32 = 49;
/// Word of `Particles.SubImageIndex` in the extended layout.
pub const WORD_SUB_IMAGE_INDEX: u32 = 44;
/// `User.` parameters a system may have on the GPU.
pub const MAX_USER_PARAMS: usize = 16;
/// Steps that can be recorded between two queue submissions.
pub const MAX_STEPS_PER_SUBMIT: u32 = 8;
/// Workgroup size of the per-particle kernels.
pub const WORKGROUP: u32 = 256;

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
    spawn_prep: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

/// One emitter's GPU state.
pub struct GpuEmitter {
    pub name: String,
    pub capacity: u32,
    /// Words per particle: [`WORDS_PER_PARTICLE`] or [`WORDS_PER_PARTICLE_EXTENDED`].
    pub words: u32,
    /// Ping-pong particle buffers; `particles[current]` holds the living ones.
    pub particles: [wgpu::Buffer; 2],
    /// Counts, ids and indirect dispatch arguments ([`STATE_WORDS`] u32s).
    pub state: wgpu::Buffer,
    frame_ubo: wgpu::Buffer,
    /// `[0]` reads buffer 0 and writes 1; `[1]` the reverse.
    bind_groups: [wgpu::BindGroup; 2],
    args: wgpu::Buffer,
    args_bind_group: wgpu::BindGroup,
    pipelines: Pipelines,
    pub current: usize,
    age: f32,
    spawn_remainder: f32,
    /// Generated WGSL, for inspection.
    pub shader_source: String,
}

/// A cooked system simulated on the GPU.
pub struct GpuEffect {
    pub system: CookedSystem,
    pub seed: u32,
    /// Steps taken so far.
    pub frame: u32,
    /// Simulated seconds (`Engine.Time`).
    pub time: f32,
    pub user: Vec<Value>,
    /// `Engine.Owner.Velocity` (m/s), read by Inherit Velocity; set by the host.
    pub owner_velocity: [f32; 3],
    pub emitters: Vec<GpuEmitter>,
    slot: u32,
    scratch: [u8; FRAME_BYTES],
}

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

impl GpuEffect {
    /// Compiles every emitter and allocates its pooled buffers.
    pub fn new(device: &wgpu::Device, system: &CookedSystem, seed: u32) -> Result<Self, String> {
        if system.user_params.len() > MAX_USER_PARAMS {
            return Err(format!(
                "{} user parameters; the GPU path supports {MAX_USER_PARAMS}",
                system.user_params.len()
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
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx_gpu sim args"),
            entries: &[storage_entry(0, false)],
        });
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
            let shader = codegen::emitter_shader(system, e);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&e.name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader.source)),
            });
            let pipe = |entry: &str| {
                let writes_args = entry == "begin" || entry == "spawn_prep";
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(if writes_args {
                        &args_pipeline_layout
                    } else {
                        &pipeline_layout
                    }),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let pipelines = Pipelines {
                begin: pipe("begin"),
                update: pipe("update"),
                spawn_prep: pipe("spawn_prep"),
                spawn: pipe("spawn"),
                finish: pipe("finish"),
            };
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
                size: 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx sim args"),
                layout: &args_layout,
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
                    layout: &layout,
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
                    ],
                })
            };
            let bind_groups = [bind_group(0), bind_group(1)];
            let _ = ei;
            emitters.push(GpuEmitter {
                name: e.name.clone(),
                capacity,
                words,
                particles,
                state,
                frame_ubo,
                bind_groups,
                args,
                args_bind_group,
                pipelines,
                current: 0,
                age: 0.0,
                spawn_remainder: 0.0,
                shader_source: shader.source,
            });
        }
        Ok(GpuEffect {
            system: system.clone(),
            seed,
            frame: 0,
            time: 0.0,
            user: system.user_params.iter().map(|p| p.value).collect(),
            owner_velocity: [0.0; 3],
            emitters,
            slot: 0,
            scratch: [0; FRAME_BYTES],
        })
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

    /// Call after each queue submission that contained steps: frees the
    /// per-step uniform slots.
    pub fn submitted(&mut self) {
        self.slot = 0;
    }

    /// Steps left before [`GpuEffect::submitted`] must be called.
    pub fn steps_left_in_submit(&self) -> u32 {
        MAX_STEPS_PER_SUBMIT - self.slot
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
            pass.set_bind_group(0, bg, &[offset as u32]);
            pass.set_bind_group(1, &em.args_bind_group, &[]);
            pass.set_pipeline(&p.begin);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim update"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32]);
            pass.set_pipeline(&p.update);
            pass.dispatch_workgroups_indirect(&em.args, 0);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim spawn prep"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32]);
            pass.set_bind_group(1, &em.args_bind_group, &[]);
            pass.set_pipeline(&p.spawn_prep);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fx sim spawn"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[offset as u32]);
            pass.set_pipeline(&p.spawn);
            pass.dispatch_workgroups_indirect(&em.args, 16);
            pass.set_pipeline(&p.finish);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);

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
            .map(|e| e.particles[0].size() * 2 + e.state.size() + e.frame_ubo.size())
            .sum()
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
