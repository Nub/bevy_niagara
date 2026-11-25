//! GPU-built ribbon inputs (Niagara's Ribbon Renderer).
//!
//! A ribbon renderer sorts its emitter's particles by `(RibbonID, link
//! order)` with a [`ViewDraw`](crate::view::ViewDraw) (sort mode
//! `RibbonSpawnOrder` or `RibbonLinkOrder`, culling off). [`RibbonBuild`]
//! then computes, on the GPU, where each ribbon starts and ends in the sorted
//! list and the distance along it at every particle (a prefix sum of link
//! lengths), for the vertex shader's UVs. Nothing is read back: the draw is
//! the sort's `draw_indirect` with one instance per sorted particle (segment
//! k joins k and k + 1).

use std::borrow::Cow;

use crate::GpuEmitter;
use crate::view::ViewDraw;

const RIBBON_WGSL: &str = include_str!("wgsl/ribbon.wgsl");
const WG: u32 = 256;
/// Ribbon ids per renderer: `Particles.RibbonID` is clamped to `0..MAX_RIBBONS`.
pub const MAX_RIBBONS: u32 = 255;

pub struct RibbonPipelines {
    layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    prep: wgpu::ComputePipeline,
    link: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RibbonPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferBindingType::{Storage, Uniform};
        let ro = Storage { read_only: true };
        let rw = Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx ribbon"),
            entries: &[
                entry(0, Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx ribbon args"),
            entries: &[entry(0, rw)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx ribbon"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RIBBON_WGSL)),
        });
        let plain = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx ribbon"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let with_args = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx ribbon args"),
            bind_group_layouts: &[Some(&layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let pipe = |entry: &str, pl: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        RibbonPipelines {
            prep: pipe("ribbon_prep", &with_args),
            link: pipe("ribbon_link", &plain),
            scan: pipe("ribbon_scan", &plain),
            add: pipe("ribbon_add", &plain),
            layout,
            args_layout,
        }
    }
}

/// One ribbon renderer's GPU-built inputs.
pub struct RibbonBuild {
    /// `[n, blocks]`: sorted particles (`array<u32>`).
    pub meta: wgpu::Buffer,
    /// `(start, end)` sorted positions per ribbon id (`array<u32>`, 2 × [`MAX_RIBBONS`]).
    pub ribbons: wgpu::Buffer,
    /// Distance along the sorted polyline at each sorted position (`array<f32>`).
    pub dist: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    args_bind_group: wgpu::BindGroup,
    args: wgpu::Buffer,
}

impl RibbonBuild {
    /// `draw` must be the ribbon sort of `emitter` (it reads its indices and
    /// draw arguments).
    pub fn new(
        device: &wgpu::Device,
        pipelines: &RibbonPipelines,
        emitter: &GpuEmitter,
        draw: &ViewDraw,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let capacity = emitter.capacity;
        let blocks = capacity.div_ceil(WG).max(1);
        let buf = |label: &str, size: u64, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fx ribbon params"),
                contents: &[capacity.to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat(),
                usage: U::UNIFORM,
            },
        );
        let meta = buf("fx ribbon meta", 16, U::STORAGE);
        let ribbons = buf("fx ribbon table", MAX_RIBBONS as u64 * 8, U::STORAGE);
        let dist = buf("fx ribbon distances", capacity as u64 * 4, U::STORAGE);
        let block_sums = buf("fx ribbon block sums", blocks as u64 * 4, U::STORAGE);
        let args = buf("fx ribbon dispatch args", 16, U::STORAGE | U::INDIRECT);
        let bind_group = |s: usize| {
            let resources = [
                params.as_entire_binding(),
                emitter.particles[s].as_entire_binding(),
                draw.indices.as_entire_binding(),
                draw.draw_args.as_entire_binding(),
                meta.as_entire_binding(),
                ribbons.as_entire_binding(),
                dist.as_entire_binding(),
                block_sums.as_entire_binding(),
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
                label: Some("fx ribbon"),
                layout: &pipelines.layout,
                entries: &entries,
            })
        };
        let bind_groups = [bind_group(0), bind_group(1)];
        let args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx ribbon args"),
            layout: &pipelines.args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: args.as_entire_binding(),
            }],
        });
        RibbonBuild {
            meta,
            ribbons,
            dist,
            bind_groups,
            args_bind_group,
            args,
        }
    }

    /// Records the build; call after the ribbon sort was recorded. Allocates nothing.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &RibbonPipelines,
        emitter: &GpuEmitter,
    ) {
        let bg = &self.bind_groups[emitter.current];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx ribbon prep"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bg, &[]);
        pass.set_bind_group(1, &self.args_bind_group, &[]);
        pass.set_pipeline(&pipelines.prep);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx ribbon build"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bg, &[]);
        pass.set_pipeline(&pipelines.link);
        pass.dispatch_workgroups_indirect(&self.args, 0);
        pass.set_pipeline(&pipelines.scan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.add);
        pass.dispatch_workgroups_indirect(&self.args, 0);
    }
}
