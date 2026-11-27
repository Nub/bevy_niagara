//! Per-view GPU culling, back-to-front sorting and indirect draw arguments.
//!
//! [`ViewDraw::record`] turns an emitter's living particles into
//! `indices[0..n]` (visible particle slots, sorted far to near when the
//! renderer sorts) and `draw_args` (`draw_indirect` or `draw_indexed_indirect`
//! arguments: [`ViewDrawConfig::vertex_count`] vertices or indices, `n`
//! instances). Sorting is a bitonic sort whose passes are all recorded
//! up front and sized on the GPU from the living count: passes beyond it get
//! zero workgroups, so the cost follows the particle count, not the pool.

use std::borrow::Cow;

use crate::GpuEmitter;

const VIEW_WGSL: &str = include_str!("wgsl/view.wgsl");
const RADIX_WGSL: &str = include_str!("wgsl/radix.wgsl");
const BLOCK: u32 = 512;
/// Views whose sort length can exceed view.wgsl's `RADIX_MIN` (16384) also
/// record the radix sort; the GPU picks bitonic or radix from the count.
const RADIX_MIN_POW2: u32 = 32768;
const RADIX_TILE: u64 = 2048;
/// Radix passes (4 bits each), two dispatch slots per pass.
const RADIX_PASSES: usize = 8;
/// Bytes per particle of the sorted sprite payload (four vec4<f32>).
pub const PAYLOAD_BYTES: u64 = 64;
const PARAMS_BYTES: usize = 6 * 16 + 16 + 16 + 16 + 16 + 16;
/// Bytes of `draw_args`: room for `draw_indexed_indirect`'s five words.
pub const DRAW_ARGS_BYTES: u64 = 20;
const PASS_STRIDE: u64 = 256;
/// Batch draws of emitters up to this capacity cull with a direct dispatch.
const DIRECT_CULL_MAX: u32 = 4096;

/// Niagara's sprite sort modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SortMode {
    #[default]
    None,
    ViewDepth,
    ViewDistance,
    /// By particle UniqueID: a draw order independent of GPU scheduling, for
    /// bit-identical images (headless preview). Costs a sort.
    UniqueId,
    /// Ribbons: by `Particles.RibbonID` (0..254), then by UniqueID (spawn order).
    RibbonSpawnOrder,
    /// Ribbons: by `Particles.RibbonID`, then by `Particles.RibbonLinkOrder` ascending.
    RibbonLinkOrder,
}

/// The bounding radius culling uses for a particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CullRadius {
    /// `max(|SpriteSize.x|, |SpriteSize.y|)` (covers rotation and pivots up to 0.5).
    SpriteSize,
    /// Particle word `word` (e.g. `Particles.LightRadius`) times `scale`.
    Word { word: u32, scale: f32 },
    /// The largest of `|Particles.Scale|` (words 28..30) times `scale` (the
    /// mesh's bounding radius).
    MeshScale { scale: f32 },
    /// Never culled (ribbons: a hidden particle would break its ribbon).
    Off,
}

/// How a [`ViewDraw`] culls, sorts and fills its draw arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewDrawConfig {
    pub sort_mode: SortMode,
    /// First draw argument: vertices (or indices, for an indexed draw) per instance.
    pub vertex_count: u32,
    pub cull: CullRadius,
}

impl ViewDrawConfig {
    /// Sprites: 6 vertices per particle, culled by sprite size.
    pub fn sprites(sort_mode: SortMode) -> Self {
        ViewDrawConfig {
            sort_mode,
            vertex_count: 6,
            cull: CullRadius::SpriteSize,
        }
    }
}

/// What a view needs for culling and sorting.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewParams {
    /// Frustum planes `(normal, d)`: a point is inside when `dot(normal, p) + d >= 0`.
    /// Unused planes can be `[0, 0, 0, 1]`.
    pub planes: [[f32; 4]; 6],
    pub camera_position: [f32; 3],
    pub camera_forward: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassKind {
    Local,
    MergeGlobal { k: u32, j: u32 },
    MergeLocal { k: u32 },
}

/// The sort passes for buffers up to `max_pow2` long, in the order
/// `cull_prep` in view.wgsl writes their dispatch arguments.
fn sort_passes(max_pow2: u32) -> Vec<PassKind> {
    let mut v = vec![PassKind::Local];
    let mut k = BLOCK * 2;
    while k <= max_pow2 {
        let mut j = k / 2;
        while j >= BLOCK {
            v.push(PassKind::MergeGlobal { k, j });
            j /= 2;
        }
        v.push(PassKind::MergeLocal { k });
        k *= 2;
    }
    v
}

/// Pipelines shared by every [`ViewDraw`] on a device.
pub struct ViewPipelines {
    layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    pass_layout: wgpu::BindGroupLayout,
    cull_prep: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
    sort_local: wgpu::ComputePipeline,
    sort_merge_local: wgpu::ComputePipeline,
    sort_merge_global: wgpu::ComputePipeline,
    batch_prep: wgpu::ComputePipeline,
    cull_batch: wgpu::ComputePipeline,
    sort_prep: wgpu::ComputePipeline,
    pad: wgpu::ComputePipeline,
    payload_layout: wgpu::BindGroupLayout,
    radix_layout: wgpu::BindGroupLayout,
    radix_hist: wgpu::ComputePipeline,
    radix_scan: wgpu::ComputePipeline,
    radix_scatter: wgpu::ComputePipeline,
    /// Bound to the batched-only bindings of unbatched draws.
    dummy: wgpu::Buffer,
}

/// A view's radix sort state: the ping-pong buffers, tile histograms and
/// the bind groups of the two directions.
struct Radix {
    /// First args slot of the radix passes (two per pass).
    base_slot: u32,
    /// Index of the first radix entry in the pass uniform.
    first_pass: usize,
    bind_groups: [wgpu::BindGroup; 2],
}

impl Radix {
    fn new(
        device: &wgpu::Device,
        pipelines: &ViewPipelines,
        keys: &wgpu::Buffer,
        vals: &wgpu::Buffer,
        args: &wgpu::Buffer,
        len: u64,
        base_slot: u32,
        first_pass: usize,
    ) -> Self {
        let buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let keys_alt = buf("fx view keys alt", len * 4);
        let vals_alt = buf("fx view indices alt", len * 4);
        let hist = buf("fx view radix hist", 4 * (16 + 16 * len.div_ceil(RADIX_TILE)));
        let group = |sk: &wgpu::Buffer, sv: &wgpu::Buffer, dk: &wgpu::Buffer, dv: &wgpu::Buffer| {
            fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding,
                    resource: b.as_entire_binding(),
                }
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx view radix"),
                layout: &pipelines.radix_layout,
                entries: &[e(0, sk), e(1, sv), e(2, dk), e(3, dv), e(4, &hist), e(5, args)],
            })
        };
        Radix {
            base_slot,
            first_pass,
            bind_groups: [
                group(keys, vals, &keys_alt, &vals_alt),
                group(&keys_alt, &vals_alt, keys, vals),
            ],
        }
    }

    /// Records the radix passes; each gets its dispatch sizes (zero when
    /// skipped) from the prep kernel's `set_sort_args`.
    fn record(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipelines: &ViewPipelines,
        pass_bind_group: &wgpu::BindGroup,
        args: &wgpu::Buffer,
    ) {
        for q in 0..RADIX_PASSES {
            let tiles = 16 * (self.base_slot as u64 + 2 * q as u64);
            pass.set_bind_group(0, &self.bind_groups[q % 2], &[]);
            pass.set_bind_group(
                1,
                pass_bind_group,
                &[((self.first_pass + q) as u64 * PASS_STRIDE) as u32],
            );
            pass.set_pipeline(&pipelines.radix_hist);
            pass.dispatch_workgroups_indirect(args, tiles);
            pass.set_pipeline(&pipelines.radix_scan);
            pass.dispatch_workgroups_indirect(args, tiles + 16);
            pass.set_pipeline(&pipelines.radix_scatter);
            pass.dispatch_workgroups_indirect(args, tiles);
        }
    }
}

/// The per-pass uniform (bitonic `(k, j)`, then radix `(shift, slot)`), one
/// dynamic-offset slot each, and the radix pass count to append.
fn pass_uniform(
    device: &wgpu::Device,
    label: &str,
    passes: &[PassKind],
    radix_base_slot: Option<u32>,
) -> wgpu::Buffer {
    let radix = if radix_base_slot.is_some() { RADIX_PASSES } else { 0 };
    let n = passes.len() + radix;
    let mut data = vec![0u8; (PASS_STRIDE as usize) * n.max(1)];
    let mut put = |i: usize, a: u32, b: u32| {
        let o = i * PASS_STRIDE as usize;
        data[o..o + 4].copy_from_slice(&a.to_le_bytes());
        data[o + 4..o + 8].copy_from_slice(&b.to_le_bytes());
    };
    for (i, p) in passes.iter().enumerate() {
        let (k, j) = match *p {
            PassKind::Local => (0, 0),
            PassKind::MergeGlobal { k, j } => (k, j),
            PassKind::MergeLocal { k } => (k, 0),
        };
        put(i, k, j);
    }
    if let Some(base) = radix_base_slot {
        for q in 0..RADIX_PASSES {
            put(passes.len() + q, 4 * q as u32, base + 2 * q as u32);
        }
    }
    wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &data,
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )
}

fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

impl ViewPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferBindingType::{Storage, Uniform};
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx view"),
            entries: &[
                buffer_entry(0, Uniform, false),
                buffer_entry(1, Storage { read_only: true }, false),
                buffer_entry(2, Storage { read_only: true }, false),
                buffer_entry(3, Storage { read_only: false }, false),
                buffer_entry(4, Storage { read_only: false }, false),
                buffer_entry(5, Storage { read_only: false }, false),
                buffer_entry(6, Storage { read_only: true }, false),
                buffer_entry(7, Storage { read_only: true }, false),
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx view args"),
            entries: &[buffer_entry(0, Storage { read_only: false }, false)],
        });
        let pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx view sort pass"),
            entries: &[buffer_entry(1, Uniform, true)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx view"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VIEW_WGSL)),
        });
        let make_layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fx view"),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        // The sprite payload (cull_batch only; 8 storage buffers with group 0,
        // WebGPU's guaranteed per-stage limit).
        let payload_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx view payload"),
            entries: &[buffer_entry(2, Storage { read_only: false }, false)],
        });
        let radix_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx view radix"),
            entries: &[
                buffer_entry(0, Storage { read_only: true }, false),
                buffer_entry(1, Storage { read_only: true }, false),
                buffer_entry(2, Storage { read_only: false }, false),
                buffer_entry(3, Storage { read_only: false }, false),
                buffer_entry(4, Storage { read_only: false }, false),
                buffer_entry(5, Storage { read_only: true }, false),
            ],
        });
        let radix_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx view radix"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RADIX_WGSL)),
        });
        let plain = make_layout(&[Some(&layout)]);
        let with_args = make_layout(&[Some(&layout), Some(&args_layout)]);
        let with_pass = make_layout(&[Some(&layout), Some(&pass_layout)]);
        let with_payload = make_layout(&[Some(&layout), Some(&payload_layout)]);
        let radix_pl = make_layout(&[Some(&radix_layout), Some(&pass_layout)]);
        let pipe_in = |module: &wgpu::ShaderModule, entry: &str, pl: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe = |entry: &str, pl: &wgpu::PipelineLayout| pipe_in(&module, entry, pl);
        ViewPipelines {
            cull_prep: pipe("cull_prep", &with_args),
            cull: pipe("cull", &plain),
            sort_local: pipe("sort_local", &plain),
            sort_merge_local: pipe("sort_merge_local", &with_pass),
            sort_merge_global: pipe("sort_merge_global", &with_pass),
            batch_prep: pipe("batch_prep", &with_args),
            cull_batch: pipe("cull_batch", &with_payload),
            sort_prep: pipe("sort_prep", &with_args),
            pad: pipe("pad", &plain),
            radix_hist: pipe_in(&radix_module, "radix_hist", &radix_pl),
            radix_scan: pipe_in(&radix_module, "radix_scan", &radix_pl),
            radix_scatter: pipe_in(&radix_module, "radix_scatter", &radix_pl),
            payload_layout,
            radix_layout,
            dummy: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx view dummy"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            layout,
            args_layout,
            pass_layout,
        }
    }
}

/// One emitter seen from one view.
pub struct ViewDraw {
    pub sort_mode: SortMode,
    pub config: ViewDrawConfig,
    capacity: u32,
    max_pow2: u32,
    passes: Vec<PassKind>,
    radix: Option<Radix>,
    params: wgpu::Buffer,
    /// Visible particle slots (`array<u32>`), sorted when sorting.
    pub indices: wgpu::Buffer,
    /// `draw_indirect` arguments at offset 0.
    pub draw_args: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    args_bind_group: wgpu::BindGroup,
    pass_bind_group: wgpu::BindGroup,
    args: wgpu::Buffer,
    scratch: [u8; PARAMS_BYTES],
}

impl ViewDraw {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ViewPipelines,
        emitter: &GpuEmitter,
        sort_mode: SortMode,
    ) -> Self {
        Self::with_config(
            device,
            pipelines,
            emitter,
            ViewDrawConfig::sprites(sort_mode),
        )
    }

    pub fn with_config(
        device: &wgpu::Device,
        pipelines: &ViewPipelines,
        emitter: &GpuEmitter,
        config: ViewDrawConfig,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let sort_mode = config.sort_mode;
        let capacity = emitter.capacity;
        let max_pow2 = capacity.next_power_of_two().max(BLOCK);
        let sorted = sort_mode != SortMode::None;
        let len = if sorted { max_pow2 } else { capacity } as u64;
        let buf = |label: &str, size: u64, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = buf(
            "fx view params",
            PARAMS_BYTES as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let keys = buf("fx view keys", if sorted { len * 4 } else { 4 }, U::STORAGE);
        let indices = buf("fx view indices", len * 4, U::STORAGE | U::COPY_SRC);
        let draw_args = buf(
            "fx view draw args",
            DRAW_ARGS_BYTES,
            U::STORAGE | U::INDIRECT | U::COPY_SRC,
        );
        let passes = if sorted {
            sort_passes(max_pow2)
        } else {
            Vec::new()
        };
        let radix_slot = (sorted && max_pow2 >= RADIX_MIN_POW2).then_some(1 + passes.len() as u32);
        let radix_slots = if radix_slot.is_some() { 2 * RADIX_PASSES as u64 } else { 0 };
        let args = buf(
            "fx view dispatch args",
            16 * (1 + passes.len() as u64 + radix_slots),
            U::STORAGE | U::INDIRECT,
        );
        let pass_ubo = pass_uniform(device, "fx view sort passes", &passes, radix_slot);
        let radix = radix_slot.map(|s| Radix::new(device, pipelines, &keys, &indices, &args, len, s, passes.len()));
        let bind_group = |s: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx view"),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: emitter.particles[s].binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: emitter.state.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: draw_args.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: pipelines.dummy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: pipelines.dummy.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups = [bind_group(0), bind_group(1)];
        let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx view args"),
            layout: &pipelines.args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: args.as_entire_binding(),
            }],
        });
        let pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx view sort pass"),
            layout: &pipelines.pass_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass_ubo,
                    offset: 0,
                    size: wgpu::BufferSize::new(8),
                }),
            }],
        });
        ViewDraw {
            sort_mode,
            config,
            capacity,
            max_pow2,
            passes,
            radix,
            params,
            indices,
            draw_args,
            bind_groups,
            args_bind_group,
            pass_bind_group,
            args,
            scratch: [0; PARAMS_BYTES],
        }
    }

    /// Records culling (and sorting) of `emitter`'s current particles for
    /// `view`. Allocates nothing.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ViewPipelines,
        emitter: &GpuEmitter,
        view: &ViewParams,
    ) {
        write_params(&mut self.scratch, view, self.capacity, self.sort_mode, self.max_pow2, &self.config, 0, 0);
        put_word(&mut self.scratch, 42, self.radix.is_some() as u32);
        queue.write_buffer(&self.params, 0, &self.scratch);

        let bg = &self.bind_groups[emitter.current];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx cull prep"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bg, &[]);
        pass.set_bind_group(1, &self.args_bind_group, &[]);
        pass.set_pipeline(&pipelines.cull_prep);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx cull and sort"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bg, &[]);
        pass.set_pipeline(&pipelines.cull);
        pass.dispatch_workgroups_indirect(&self.args, 0);
        for (i, p) in self.passes.iter().enumerate() {
            let args_offset = 16 * (1 + i as u64);
            match p {
                PassKind::Local => pass.set_pipeline(&pipelines.sort_local),
                PassKind::MergeGlobal { .. } => {
                    pass.set_bind_group(
                        1,
                        &self.pass_bind_group,
                        &[(i as u64 * PASS_STRIDE) as u32],
                    );
                    pass.set_pipeline(&pipelines.sort_merge_global);
                }
                PassKind::MergeLocal { .. } => {
                    pass.set_bind_group(
                        1,
                        &self.pass_bind_group,
                        &[(i as u64 * PASS_STRIDE) as u32],
                    );
                    pass.set_pipeline(&pipelines.sort_merge_local);
                }
            }
            pass.dispatch_workgroups_indirect(&self.args, args_offset);
        }
        if let Some(r) = &self.radix {
            r.record(&mut pass, pipelines, &self.pass_bind_group, &self.args);
        }
    }

    /// Number of recorded sort passes (0 when not sorting).
    pub fn sort_pass_count(&self) -> usize {
        self.passes.len()
    }
}

fn put_word(s: &mut [u8; PARAMS_BYTES], w: usize, v: u32) {
    s[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// The `ViewParams` uniform of view.wgsl.
fn write_params(
    s: &mut [u8; PARAMS_BYTES],
    view: &ViewParams,
    capacity: u32,
    sort_mode: SortMode,
    max_pow2: u32,
    c: &ViewDrawConfig,
    region_words: u32,
    entry_count: u32,
) {
    let mut put = |w: usize, v: u32| s[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    for (i, pl) in view.planes.iter().enumerate() {
        for c in 0..4 {
            put(i * 4 + c, pl[c].to_bits());
        }
    }
    for c in 0..3 {
        put(24 + c, view.camera_position[c].to_bits());
        put(28 + c, view.camera_forward[c].to_bits());
    }
    put(32, capacity);
    put(
        33,
        match sort_mode {
            SortMode::None => 0,
            SortMode::ViewDepth => 1,
            SortMode::ViewDistance => 2,
            SortMode::UniqueId => 3,
            SortMode::RibbonSpawnOrder => 4,
            SortMode::RibbonLinkOrder => 5,
        },
    );
    put(34, max_pow2);
    let (mode, word, scale) = match c.cull {
        CullRadius::SpriteSize => (0, 0, 1.0f32),
        CullRadius::Word { word, scale } => (1, word, scale),
        CullRadius::MeshScale { scale } => (2, 28, scale),
        CullRadius::Off => (3, 0, 0.0),
    };
    put(35, mode);
    put(36, c.vertex_count);
    put(37, word);
    put(38, scale.to_bits());
    put(39, region_words);
    put(40, entry_count);
}

/// The compute passes of a [`BatchDraw`]'s cull and sort, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullPhase {
    /// Draw-argument reset and dispatch sizes (and the cull of small emitters).
    Prep,
    /// The cull of larger emitters (indirect).
    Cull,
    /// Several instances sorted: sort length from the visible count.
    SortPrep,
    /// Tail padding, then the bitonic or radix passes.
    Sort,
}

impl CullPhase {
    pub const ALL: [CullPhase; 4] = [CullPhase::Prep, CullPhase::Cull, CullPhase::SortPrep, CullPhase::Sort];

    pub fn label(self) -> &'static str {
        match self {
            CullPhase::Prep => "fx batch cull prep",
            CullPhase::Cull => "fx batch cull",
            CullPhase::SortPrep => "fx batch sort prep",
            CullPhase::Sort => "fx batch sort",
        }
    }
}

/// One emitter of every instance of a [`GpuBatch`](crate::GpuBatch) seen
/// from one view: one cull over all the instances drawn, one sort, one
/// indirect draw. `indices` holds particle references `r * capacity + i`
/// (copy `r = 2 * slot + parity`, see [`GpuEmitter::particles`]), read at
/// word `r * region_words + w * capacity + i` of the batch's particle buffer;
/// planes, camera and sort keys are in world space, with each slot's
/// `world_from_local` from the `transforms` buffer given at creation.
pub struct BatchDraw {
    pub config: ViewDrawConfig,
    /// The emitter simulates in world space (`local_space: false`): its
    /// particles are culled and sorted without the slots' transforms.
    pub world_space: bool,
    capacity: u32,
    region_words: u32,
    max_pow2: u32,
    passes: Vec<PassKind>,
    radix: Option<Radix>,
    /// Sorted sprites: 64 bytes per particle position (see `payload_used`).
    pub payload: Option<wgpu::Buffer>,
    payload_bind_group: wgpu::BindGroup,
    /// Instances drawn this frame (set by `prepare`).
    copies: usize,
    params: wgpu::Buffer,
    entries: wgpu::Buffer,
    /// Visible particle references (`array<u32>`), sorted when sorting.
    pub indices: wgpu::Buffer,
    /// `draw_indirect` (or indexed) arguments at offset 0.
    pub draw_args: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    args_bind_group: wgpu::BindGroup,
    pass_bind_group: wgpu::BindGroup,
    args: wgpu::Buffer,
    scratch: [u8; PARAMS_BYTES],
    entry_bytes: Vec<u8>,
}

impl BatchDraw {
    /// `transforms` holds one `mat4x4<f32>` (world_from_local) per slot.
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ViewPipelines,
        batch: &crate::GpuBatch,
        emitter: usize,
        transforms: &wgpu::Buffer,
        config: ViewDrawConfig,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let capacity = batch.program.emitter_capacity(emitter);
        let total = capacity as u64 * batch.slots as u64;
        let sorted = config.sort_mode != SortMode::None;
        let max_pow2 = (total.next_power_of_two() as u32).max(BLOCK);
        let len = if sorted { max_pow2 as u64 } else { total };
        let buf = |label: &str, size: u64, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = buf("fx batch view params", PARAMS_BYTES as u64, U::UNIFORM | U::COPY_DST);
        let entries = buf("fx batch view entries", batch.slots as u64 * 4, U::STORAGE | U::COPY_DST);
        let keys = buf("fx batch view keys", if sorted { len * 4 } else { 4 }, U::STORAGE);
        let indices = buf("fx batch view indices", len * 4, U::STORAGE | U::COPY_SRC);
        let draw_args = buf("fx batch view draw args", DRAW_ARGS_BYTES, U::STORAGE | U::INDIRECT | U::COPY_SRC);
        let passes = if sorted { sort_passes(max_pow2) } else { Vec::new() };
        // Slot 0: pad; 1..: sort passes; last: the cull.
        // Slot 0: pad; 1..: sort passes; then the radix passes; last: the cull.
        let radix_slot = (sorted && max_pow2 >= RADIX_MIN_POW2).then_some(1 + passes.len() as u32);
        let radix_slots = if radix_slot.is_some() { 2 * RADIX_PASSES as u64 } else { 0 };
        let args = buf(
            "fx batch view dispatch args",
            16 * (2 + passes.len() as u64 + radix_slots),
            U::STORAGE | U::INDIRECT,
        );
        let pass_ubo = pass_uniform(device, "fx batch view sort passes", &passes, radix_slot);
        let radix = radix_slot.map(|s| Radix::new(device, pipelines, &keys, &indices, &args, len, s, passes.len()));
        // Sorted sprites: the cull writes what the sprite vertex shader reads
        // (one instance drawn) as 64 contiguous bytes per particle, so the
        // sorted draw reads two cache sectors per particle instead of
        // fourteen scattered words.
        let payload = (sorted && config.vertex_count == 6 && config.cull == CullRadius::SpriteSize)
            .then(|| buf("fx batch view sprite payload", capacity as u64 * PAYLOAD_BYTES, U::STORAGE));
        let payload_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx batch view payload"),
            layout: &pipelines.payload_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: payload.as_ref().unwrap_or(&pipelines.dummy).as_entire_binding(),
            }],
        });
        fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx batch view"),
            layout: &pipelines.layout,
            entries: &[
                e(0, &params),
                e(1, batch.particles(emitter)),
                e(2, batch.state(emitter)),
                e(3, &keys),
                e(4, &indices),
                e(5, &draw_args),
                e(6, &entries),
                e(7, transforms),
            ],
        });
        let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx batch view args"),
            layout: &pipelines.args_layout,
            entries: &[e(0, &args)],
        });
        let pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx batch view sort pass"),
            layout: &pipelines.pass_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass_ubo,
                    offset: 0,
                    size: wgpu::BufferSize::new(8),
                }),
            }],
        });
        BatchDraw {
            config,
            world_space: false,
            capacity,
            region_words: batch.region_words(emitter),
            max_pow2,
            passes,
            radix,
            payload,
            payload_bind_group,
            copies: 0,
            params,
            entries,
            indices,
            draw_args,
            bind_group,
            args_bind_group,
            pass_bind_group,
            args,
            scratch: [0; PARAMS_BYTES],
            entry_bytes: Vec::with_capacity(batch.slots as usize * 4),
        }
    }

    /// Records culling (and sorting) of the instances `copies` (each
    /// `2 * slot + current`, as [`GpuEmitter::particles`]) for `view` (world
    /// space). With no copies the draw is empty. Each [`CullPhase`] in a
    /// compute pass of its own; [`BatchDraw::record_phase`] records one phase
    /// of many draws in one pass instead.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ViewPipelines,
        copies: &[u32],
        view: &ViewParams,
    ) {
        self.prepare(queue, copies, view);
        for phase in CullPhase::ALL {
            if self.phase_used(phase) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(phase.label()),
                    timestamp_writes: None,
                });
                self.record_phase(phase, &mut pass, pipelines);
            }
        }
    }

    /// Writes the view and the instances drawn (`copies`) for this frame's
    /// phases. Call before [`BatchDraw::record_phase`].
    pub fn prepare(&mut self, queue: &wgpu::Queue, copies: &[u32], view: &ViewParams) {
        write_params(
            &mut self.scratch,
            view,
            self.capacity,
            self.config.sort_mode,
            self.max_pow2,
            &self.config,
            self.region_words,
            copies.len() as u32,
        );
        self.scratch[41 * 4..42 * 4].copy_from_slice(&(self.world_space as u32).to_le_bytes());
        let payload = self.payload_used(copies.len());
        put_word(&mut self.scratch, 42, self.radix.is_some() as u32);
        put_word(&mut self.scratch, 43, payload as u32);
        queue.write_buffer(&self.params, 0, &self.scratch);
        self.entry_bytes.clear();
        for c in copies {
            self.entry_bytes.extend_from_slice(&c.to_le_bytes());
        }
        if !copies.is_empty() {
            queue.write_buffer(&self.entries, 0, &self.entry_bytes);
        }
        self.copies = copies.len();
    }

    fn sorted(&self) -> bool {
        !self.passes.is_empty()
    }

    fn single_sorted(&self) -> bool {
        self.sorted() && self.copies == 1
    }

    /// Small emitters: the cull over the whole capacity, in the prep's pass.
    fn direct(&self) -> bool {
        !self.single_sorted() && self.capacity <= DIRECT_CULL_MAX
    }

    /// Whether `phase` records anything for the prepared frame.
    pub fn phase_used(&self, phase: CullPhase) -> bool {
        let drawn = self.copies > 0;
        match phase {
            CullPhase::Prep => true,
            CullPhase::Cull => drawn && !self.direct(),
            CullPhase::SortPrep => drawn && self.sorted() && !self.single_sorted(),
            CullPhase::Sort => drawn && self.sorted(),
        }
    }

    /// Records `phase` of the prepared frame into `pass` (nothing when
    /// unused). Phases of different draws can share a pass; each draw's
    /// phases must be recorded in [`CullPhase::ALL`] order, in passes of
    /// their own (a later phase reads what the earlier one wrote).
    pub fn record_phase(&self, phase: CullPhase, pass: &mut wgpu::ComputePass<'_>, pipelines: &ViewPipelines) {
        if !self.phase_used(phase) {
            return;
        }
        let drawn = self.copies > 0;
        match phase {
            CullPhase::Prep => {
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, &self.args_bind_group, &[]);
                pass.set_pipeline(&pipelines.batch_prep);
                pass.dispatch_workgroups(1, 1, 1);
                if drawn && self.direct() {
                    pass.set_bind_group(1, &self.payload_bind_group, &[]);
                    pass.set_pipeline(&pipelines.cull_batch);
                    pass.dispatch_workgroups(self.capacity.div_ceil(BLOCK / 2), self.copies as u32, 1);
                }
            }
            CullPhase::Cull => {
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, &self.payload_bind_group, &[]);
                pass.set_pipeline(&pipelines.cull_batch);
                pass.dispatch_workgroups_indirect(&self.args, self.args.size() - 16);
            }
            CullPhase::SortPrep => {
                // Several instances: sort length from the visible count, tail padded.
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, &self.args_bind_group, &[]);
                pass.set_pipeline(&pipelines.sort_prep);
                pass.dispatch_workgroups(1, 1, 1);
            }
            CullPhase::Sort => {
                pass.set_bind_group(0, &self.bind_group, &[]);
                if !self.single_sorted() {
                    pass.set_pipeline(&pipelines.pad);
                    pass.dispatch_workgroups_indirect(&self.args, 0);
                }
                // Only the passes the drawn instances can need (later ones would get no workgroups).
                let limit = (self.copies as u64 * self.capacity as u64).next_power_of_two().max(BLOCK as u64);
                for (i, p) in self.passes.iter().enumerate() {
                    match *p {
                        PassKind::MergeGlobal { k, .. } | PassKind::MergeLocal { k } if k as u64 > limit => break,
                        _ => {}
                    }
                    let args_offset = 16 * (1 + i as u64);
                    match p {
                        PassKind::Local => pass.set_pipeline(&pipelines.sort_local),
                        PassKind::MergeGlobal { .. } => {
                            pass.set_bind_group(1, &self.pass_bind_group, &[(i as u64 * PASS_STRIDE) as u32]);
                            pass.set_pipeline(&pipelines.sort_merge_global);
                        }
                        PassKind::MergeLocal { .. } => {
                            pass.set_bind_group(1, &self.pass_bind_group, &[(i as u64 * PASS_STRIDE) as u32]);
                            pass.set_pipeline(&pipelines.sort_merge_local);
                        }
                    }
                    pass.dispatch_workgroups_indirect(&self.args, args_offset);
                }
                if let Some(r) = &self.radix
                    && limit > RADIX_MIN_POW2 as u64 / 2
                {
                    r.record(pass, pipelines, &self.pass_bind_group, &self.args);
                    // Leave group 0 as the radix one: the next draw sets its own.
                }
            }
        }
    }

    /// Whether the cull writes the sprite payload for `copies` instances
    /// drawn (sorted sprites, one instance): then the sprite vertex shader
    /// reads particle `i`'s position, velocity, colour, size, rotation and
    /// age from `payload[i % capacity]`.
    pub fn payload_used(&self, copies: usize) -> bool {
        self.payload.is_some() && copies == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pass_list_counts() {
        assert_eq!(sort_passes(512), vec![PassKind::Local]);
        // 1024: one global j=512, one local merge.
        assert_eq!(sort_passes(1024).len(), 3);
        // 2^20: 1 + sum over k=2^10..2^20 of ((log2 k - 9) + 1).
        let n: usize = (10..=20).map(|s| s - 9 + 1).sum();
        assert_eq!(sort_passes(1 << 20).len(), 1 + n);
    }
}
