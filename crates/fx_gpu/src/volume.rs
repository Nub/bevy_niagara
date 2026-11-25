//! Volumetric renderer support: particles splatted into a density grid.
//!
//! A [`VolumeSplat`] clears a fixed-point grid, splats every particle of a
//! [`ViewDraw`] (cull off) into it as a soft ball of radius
//! `|SpriteSize.x| * radius_scale` and density `Color.a * density_scale`,
//! and resolves it into an `rgba16float` 3D texture ([`VolumeSplat::texture`]):
//! rgb = density-weighted colour, a = extinction per metre. Renderers
//! ray-march the texture over the volume bounds.

use std::borrow::Cow;

use crate::GpuEmitter;
use crate::view::ViewDraw;

const VOLUME_WGSL: &str = include_str!("wgsl/volume.wgsl");
const WG: u32 = 64;

/// Format of [`VolumeSplat::texture`].
pub const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct VolumePipelines {
    layout: wgpu::BindGroupLayout,
    splat: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

impl VolumePipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
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
            label: Some("fx volume"),
            entries: &[
                buffer(0, Uniform),
                buffer(1, ro),
                buffer(2, ro),
                buffer(3, ro),
                buffer(4, Storage { read_only: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: VOLUME_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx volume"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VOLUME_WGSL)),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx volume"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        VolumePipelines {
            splat: pipeline("splat"),
            resolve: pipeline("resolve"),
            layout,
        }
    }
}

/// Where and how a volume splats, written each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeParams {
    /// `(min, max)` in the emitter's (effect) space.
    pub bounds: [[f32; 3]; 2],
    pub density_scale: f32,
    pub radius_scale: f32,
}

/// One volumetric renderer of one instance: its grid and texture.
pub struct VolumeSplat {
    pub resolution: u32,
    params: wgpu::Buffer,
    grid: wgpu::Buffer,
    /// rgb = colour, a = extinction per metre; `resolution`³ cells.
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    bind_groups: [wgpu::BindGroup; 2],
    capacity: u32,
}

impl VolumeSplat {
    /// `draw` lists `emitter`'s living particles (cull off, no sort needed).
    pub fn new(
        device: &wgpu::Device,
        pipelines: &VolumePipelines,
        emitter: &GpuEmitter,
        draw: &ViewDraw,
        resolution: u32,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let resolution = resolution.clamp(1, 256);
        let cells = (resolution as u64).pow(3);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx volume params"),
            size: 48,
            usage: U::UNIFORM | U::COPY_DST,
            mapped_at_creation: false,
        });
        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx volume grid"),
            size: cells * 16,
            usage: U::STORAGE | U::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fx volume"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: VOLUME_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = |s: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx volume"),
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
                        resource: draw.indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: draw.draw_args.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: grid.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            })
        };
        VolumeSplat {
            resolution,
            bind_groups: [bind_group(0), bind_group(1)],
            params,
            grid,
            texture,
            view,
            capacity: emitter.capacity,
        }
    }

    /// Writes `p` (call before the frame's commands are submitted).
    pub fn set_params(&self, queue: &wgpu::Queue, p: &VolumeParams) {
        let [lo, hi] = p.bounds;
        let w: [u32; 12] = [
            lo[0].to_bits(),
            lo[1].to_bits(),
            lo[2].to_bits(),
            self.resolution,
            hi[0].to_bits(),
            hi[1].to_bits(),
            hi[2].to_bits(),
            self.capacity,
            p.density_scale.to_bits(),
            p.radius_scale.to_bits(),
            0,
            0,
        ];
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&w));
    }

    /// Records clear, splat and resolve. Call after the draw's cull was
    /// recorded. The splat runs over the emitter's capacity (threads past
    /// the living count return at once).
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, pipelines: &VolumePipelines, emitter: &GpuEmitter) {
        encoder.clear_buffer(&self.grid, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx volume"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_groups[emitter.current], &[]);
        pass.set_pipeline(&pipelines.splat);
        pass.dispatch_workgroups(self.capacity.div_ceil(WG).max(1), 1, 1);
        pass.set_pipeline(&pipelines.resolve);
        let g = self.resolution.div_ceil(4);
        pass.dispatch_workgroups(g, g, g);
    }
}
