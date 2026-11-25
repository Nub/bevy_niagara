//! Component renderer support: chosen attributes of a capped number of
//! particles, gathered on the GPU and read back asynchronously.
//!
//! A [`ParticleGather`] reads a [`ViewDraw`]'s sorted index list (the
//! component renderer sorts by UniqueID, culls nothing), packs words
//! `words` of the first `max_rows` particles into a small buffer and copies
//! it to one of three staging buffers, mapped after submission
//! ([`ParticleGather::after_submit`]); [`ParticleGather::collect`] takes the
//! newest mapped rows. Only those rows (`16 + 4 * words * max_rows` bytes)
//! come back.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::GpuEmitter;
use crate::view::ViewDraw;

const GATHER_WGSL: &str = include_str!("wgsl/gather.wgsl");
const WG: u32 = 64;
const SLOTS: usize = 3;
/// Most words gathered per particle.
pub const MAX_WORDS: usize = 32;

const FREE: u8 = 0;
const RECORDED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;

pub struct GatherPipelines {
    layout: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
}

impl GatherPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        use wgpu::BufferBindingType::{Storage, Uniform};
        let ro = Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx gather"),
            entries: &[
                entry(0, Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, Storage { read_only: false }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx gather"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GATHER_WGSL)),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx gather"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let gather = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gather_rows"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("gather_rows"),
            compilation_options: Default::default(),
            cache: None,
        });
        GatherPipelines { layout, gather }
    }
}

/// Chosen words of the first particles of one [`ViewDraw`], read back.
pub struct ParticleGather {
    pub max_rows: u32,
    /// Words per row (`words.len()`).
    pub stride: usize,
    out: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    staging: [wgpu::Buffer; SLOTS],
    states: [Arc<AtomicU8>; SLOTS],
    tags: [u64; SLOTS],
    recorded: Option<usize>,
    /// The newest collected rows, `stride` words each.
    pub rows: Vec<f32>,
    /// Tag passed to [`ParticleGather::record`] for [`ParticleGather::rows`].
    pub tag: Option<u64>,
}

impl ParticleGather {
    /// `draw` lists `emitter`'s particles in the order rows are taken.
    /// `words`: particle words per row, at most [`MAX_WORDS`].
    pub fn new(
        device: &wgpu::Device,
        pipelines: &GatherPipelines,
        emitter: &GpuEmitter,
        draw: &ViewDraw,
        max_rows: u32,
        words: &[u32],
    ) -> Self {
        use wgpu::BufferUsages as U;
        assert!(!words.is_empty() && words.len() <= MAX_WORDS);
        let max_rows = max_rows.max(1);
        let stride = words.len();
        let bytes = 16 + 4 * (stride as u64) * max_rows as u64;
        let mut params = vec![emitter.capacity, max_rows, stride as u32, 0];
        let mut w = [0u32; MAX_WORDS];
        w[..stride].copy_from_slice(words);
        params.extend_from_slice(&w);
        let params = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fx gather params"),
                contents: bytemuck::cast_slice(&params),
                usage: U::UNIFORM,
            },
        );
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx gather"),
            size: bytes,
            usage: U::STORAGE | U::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = |s: usize| {
            let resources = [
                params.as_entire_binding(),
                emitter.particles[s].binding(),
                draw.indices.as_entire_binding(),
                draw.draw_args.as_entire_binding(),
                out.as_entire_binding(),
            ];
            let entries: Vec<_> = resources
                .into_iter()
                .enumerate()
                .map(|(i, resource)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource,
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx gather"),
                layout: &pipelines.layout,
                entries: &entries,
            })
        };
        let staging = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx gather readback"),
                size: bytes,
                usage: U::MAP_READ | U::COPY_DST,
                mapped_at_creation: false,
            })
        });
        ParticleGather {
            max_rows,
            stride,
            bind_groups: [bind_group(0), bind_group(1)],
            out,
            staging,
            states: std::array::from_fn(|_| Arc::new(AtomicU8::new(FREE))),
            tags: [0; SLOTS],
            recorded: None,
            rows: Vec::with_capacity(stride * max_rows as usize),
            tag: None,
        }
    }

    /// Records the gather and its copy to a free staging buffer (none free:
    /// this frame is skipped). Call after the draw's cull/sort was recorded.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &GatherPipelines,
        emitter: &GpuEmitter,
        tag: u64,
    ) {
        let Some(slot) = (0..SLOTS).find(|&s| self.states[s].load(Ordering::Acquire) == FREE)
        else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx gather rows"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_groups[emitter.current], &[]);
        pass.set_pipeline(&pipelines.gather);
        pass.dispatch_workgroups(self.max_rows.div_ceil(WG), 1, 1);
        drop(pass);
        encoder.copy_buffer_to_buffer(&self.out, 0, &self.staging[slot], 0, self.out.size());
        self.states[slot].store(RECORDED, Ordering::Release);
        self.tags[slot] = tag;
        self.recorded = Some(slot);
    }

    /// Call once the commands from [`ParticleGather::record`] were
    /// submitted: requests the mapping of the copy.
    pub fn after_submit(&mut self) {
        let Some(slot) = self.recorded.take() else {
            return;
        };
        self.states[slot].store(MAPPING, Ordering::Release);
        let state = self.states[slot].clone();
        self.staging[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                state.store(if r.is_ok() { MAPPED } else { FREE }, Ordering::Release);
            });
    }

    /// Takes the newest mapped rows into [`ParticleGather::rows`] (older
    /// mapped ones are dropped). Returns whether the rows changed.
    pub fn collect(&mut self) -> bool {
        let mut newest: Option<usize> = None;
        for s in 0..SLOTS {
            if self.states[s].load(Ordering::Acquire) == MAPPED
                && newest.is_none_or(|n| self.tags[s] > self.tags[n])
            {
                newest = Some(s);
            }
        }
        let Some(slot) = newest else {
            return false;
        };
        let stale = self.tag.is_some_and(|t| t > self.tags[slot]);
        if !stale {
            let data = self.staging[slot].slice(..).get_mapped_range();
            if let Ok(data) = data {
                let words: &[f32] = bytemuck::cast_slice(&data);
                let count = (words[0].to_bits() as usize).min(self.max_rows as usize);
                self.rows.clear();
                self.rows
                    .extend_from_slice(&words[4..4 + count * self.stride]);
            }
            self.tag = Some(self.tags[slot]);
        }
        for s in 0..SLOTS {
            if self.states[s].load(Ordering::Acquire) == MAPPED && self.tags[s] <= self.tags[slot] {
                self.staging[s].unmap();
                self.states[s].store(FREE, Ordering::Release);
            }
        }
        !stale
    }
}
