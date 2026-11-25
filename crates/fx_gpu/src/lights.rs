//! Light renderer support: per-particle point lights, capped and culled on
//! the GPU, read back asynchronously.
//!
//! A light renderer culls its emitter's particles against a view with
//! [`CullRadius::Word`](crate::view::CullRadius) on `Particles.LightRadius`
//! and sorts them nearest first ([`ViewDraw`] with `ViewDistance`).
//! [`LightGather`] packs the first `max_lights` into a small buffer and
//! copies it to one of three staging buffers, mapped after submission
//! ([`LightGather::after_submit`]); [`LightGather::collect`] takes the newest
//! mapped list. Only that list (at most `16 + 32 * max_lights` bytes) comes
//! back; particle counts and data stay on the GPU.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::GpuEmitter;
use crate::view::ViewDraw;

const LIGHTS_WGSL: &str = include_str!("wgsl/lights.wgsl");
const WG: u32 = 64;
const SLOTS: usize = 3;

const FREE: u8 = 0;
const RECORDED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;

/// One particle light, in the emitter's local space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuLight {
    pub position: [f32; 3],
    /// `Particles.LightRadius * radius_scale`, metres.
    pub radius: f32,
    /// `Particles.Color`.
    pub color: [f32; 4],
}

pub struct LightPipelines {
    layout: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
}

impl LightPipelines {
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
            label: Some("fx lights"),
            entries: &[
                entry(0, Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, Storage { read_only: false }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx lights"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LIGHTS_WGSL)),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx lights"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let gather = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gather_lights"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("gather_lights"),
            compilation_options: Default::default(),
            cache: None,
        });
        LightPipelines { layout, gather }
    }
}

/// One light renderer seen from one view.
pub struct LightGather {
    pub max_lights: u32,
    out: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    staging: [wgpu::Buffer; SLOTS],
    states: [Arc<AtomicU8>; SLOTS],
    tags: [u64; SLOTS],
    recorded: Option<usize>,
    /// The newest collected lights.
    pub lights: Vec<GpuLight>,
    /// Tag passed to [`LightGather::record`] for [`LightGather::lights`].
    pub tag: Option<u64>,
}

impl LightGather {
    /// `draw` must cull `emitter` for the lights (nearest first).
    pub fn new(
        device: &wgpu::Device,
        pipelines: &LightPipelines,
        emitter: &GpuEmitter,
        draw: &ViewDraw,
        max_lights: u32,
        radius_scale: f32,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let max_lights = max_lights.max(1);
        let bytes = 16 + 32 * max_lights as u64;
        let params = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fx lights params"),
                contents: &[
                    emitter.capacity.to_le_bytes(),
                    max_lights.to_le_bytes(),
                    radius_scale.to_le_bytes(),
                    [0; 4],
                ]
                .concat(),
                usage: U::UNIFORM,
            },
        );
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx lights"),
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
                label: Some("fx lights"),
                layout: &pipelines.layout,
                entries: &entries,
            })
        };
        let staging = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx lights readback"),
                size: bytes,
                usage: U::MAP_READ | U::COPY_DST,
                mapped_at_creation: false,
            })
        });
        LightGather {
            max_lights,
            bind_groups: [bind_group(0), bind_group(1)],
            out,
            staging,
            states: std::array::from_fn(|_| Arc::new(AtomicU8::new(FREE))),
            tags: [0; SLOTS],
            recorded: None,
            lights: Vec::with_capacity(max_lights as usize),
            tag: None,
        }
    }

    /// Records the gather and its copy to a free staging buffer (none free:
    /// this frame's lights are skipped). Call after the light cull was recorded.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &LightPipelines,
        emitter: &GpuEmitter,
        tag: u64,
    ) {
        let Some(slot) = (0..SLOTS).find(|&s| self.states[s].load(Ordering::Acquire) == FREE)
        else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx gather lights"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_groups[emitter.current], &[]);
        pass.set_pipeline(&pipelines.gather);
        pass.dispatch_workgroups(self.max_lights.div_ceil(WG), 1, 1);
        drop(pass);
        encoder.copy_buffer_to_buffer(&self.out, 0, &self.staging[slot], 0, self.out.size());
        self.states[slot].store(RECORDED, Ordering::Release);
        self.tags[slot] = tag;
        self.recorded = Some(slot);
    }

    /// Call once the commands from [`LightGather::record`] were submitted:
    /// requests the mapping of the copy.
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

    /// Whether a copy is waiting to be mapped or read.
    pub fn pending(&self) -> bool {
        self.states
            .iter()
            .any(|s| matches!(s.load(Ordering::Acquire), MAPPING | MAPPED))
    }

    /// Takes the newest mapped list into [`LightGather::lights`] (older
    /// mapped ones are dropped). Returns whether the lights changed.
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
                let count = (words[0].to_bits() as usize).min(self.max_lights as usize);
                self.lights.clear();
                for j in 0..count {
                    let w = &words[4 + j * 8..12 + j * 8];
                    self.lights.push(GpuLight {
                        position: [w[0], w[1], w[2]],
                        radius: w[3],
                        color: [w[4], w[5], w[6], w[7]],
                    });
                }
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
