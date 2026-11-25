//! Per-view GPU culling, back-to-front sorting and indirect draw arguments.
//!
//! [`ViewDraw::record`] turns an emitter's living particles into
//! `indices[0..n]` (visible particle slots, sorted far to near when the
//! renderer sorts) and `draw_args` (`draw_indirect` arguments: 6 vertices,
//! `n` instances). Sorting is a bitonic sort whose passes are all recorded
//! up front and sized on the GPU from the living count: passes beyond it get
//! zero workgroups, so the cost follows the particle count, not the pool.

use std::borrow::Cow;

use crate::GpuEmitter;

const VIEW_WGSL: &str = include_str!("wgsl/view.wgsl");
const BLOCK: u32 = 512;
const PARAMS_BYTES: usize = 6 * 16 + 16 + 16 + 16;
const PASS_STRIDE: u64 = 256;

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
        let plain = make_layout(&[Some(&layout)]);
        let with_args = make_layout(&[Some(&layout), Some(&args_layout)]);
        let with_pass = make_layout(&[Some(&layout), Some(&pass_layout)]);
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
        ViewPipelines {
            cull_prep: pipe("cull_prep", &with_args),
            cull: pipe("cull", &plain),
            sort_local: pipe("sort_local", &plain),
            sort_merge_local: pipe("sort_merge_local", &with_pass),
            sort_merge_global: pipe("sort_merge_global", &with_pass),
            layout,
            args_layout,
            pass_layout,
        }
    }
}

/// One emitter seen from one view.
pub struct ViewDraw {
    pub sort_mode: SortMode,
    capacity: u32,
    max_pow2: u32,
    passes: Vec<PassKind>,
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
        use wgpu::BufferUsages as U;
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
            16,
            U::STORAGE | U::INDIRECT | U::COPY_SRC,
        );
        let passes = if sorted {
            sort_passes(max_pow2)
        } else {
            Vec::new()
        };
        let args = buf(
            "fx view dispatch args",
            16 * (1 + passes.len() as u64),
            U::STORAGE | U::INDIRECT,
        );
        // Static per-pass (k, j), one dynamic-offset slot each.
        let mut pass_data = vec![0u8; (PASS_STRIDE as usize) * passes.len().max(1)];
        for (i, p) in passes.iter().enumerate() {
            let (k, j) = match *p {
                PassKind::Local => (0, 0),
                PassKind::MergeGlobal { k, j } => (k, j),
                PassKind::MergeLocal { k } => (k, 0),
            };
            let o = i * PASS_STRIDE as usize;
            pass_data[o..o + 4].copy_from_slice(&k.to_le_bytes());
            pass_data[o + 4..o + 8].copy_from_slice(&j.to_le_bytes());
        }
        let pass_ubo = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("fx view sort passes"),
                contents: &pass_data,
                usage: U::UNIFORM,
            },
        );
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
                        resource: emitter.particles[s].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: emitter.state.as_entire_binding(),
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
            capacity,
            max_pow2,
            passes,
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
        let s = &mut self.scratch;
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
        put(32, self.capacity);
        put(
            33,
            match self.sort_mode {
                SortMode::None => 0,
                SortMode::ViewDepth => 1,
                SortMode::ViewDistance => 2,
                SortMode::UniqueId => 3,
            },
        );
        put(34, self.max_pow2);
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
    }

    /// Number of recorded sort passes (0 when not sorting).
    pub fn sort_pass_count(&self) -> usize {
        self.passes.len()
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
