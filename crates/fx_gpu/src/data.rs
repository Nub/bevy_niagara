//! Data interfaces on the GPU: the meshes and textures an effect samples,
//! packed into storage buffers (bind group 1, `wgsl/data.wgsl`).
//!
//! Binding data (a new mesh or texture) repacks and reallocates; that is
//! rare. Poses change every frame and are written in place with no
//! allocation.

use fx_core::cook::CookedSystem;
use fx_core::data::{DiType, MeshData, MeshPose, TextureData};
use wgpu::util::DeviceExt;

/// `di_info` rows per data interface.
const INFO_ROWS: usize = 3;
const FLAG_BOUND: u32 = 1;
const FLAG_SKINNED: u32 = 2;

/// One data interface's packed data.
#[derive(Default)]
struct Slot {
    ty: Option<DiType>,
    /// Packed `di_data` words (offsets relative to the slot's start).
    words: Vec<u32>,
    /// Relative offsets: vertices, skin, indices, area CDF, bones.
    offsets: [u32; 5],
    triangles: u32,
    bones: u32,
    /// Pose matrices (per set); 0 for textures.
    matrices: u32,
    width: u32,
    height: u32,
    flags: u32,
    pose: MeshPose,
    /// Where the pose starts in `di_pose` (vec4s).
    pose_offset: u32,
    /// Scratch for pose writes: (interval), current, previous as f32s.
    pose_scratch: Vec<f32>,
}

impl Slot {
    fn mesh(mesh: &MeshData, pose: MeshPose) -> Slot {
        let v = mesh.positions.len();
        let mut words = Vec::with_capacity(v * 14 + mesh.indices.len() * 2);
        let bits = |x: f32| x.to_bits();
        for i in 0..v {
            words.extend(mesh.positions[i].map(bits));
            words.extend(mesh.normals[i].map(bits));
        }
        let skin = words.len() as u32;
        if mesh.is_skinned() {
            for i in 0..v {
                words.extend(mesh.joints[i]);
                words.extend(mesh.weights[i].map(bits));
            }
        }
        let indices = words.len() as u32;
        words.extend(&mesh.indices);
        let cdf = words.len() as u32;
        words.extend(mesh.area_cdf.iter().map(|x| x.to_bits()));
        let bones = words.len() as u32;
        if mesh.is_skinned() {
            for b in &mesh.bone_positions {
                words.extend(b.map(bits));
            }
        }
        let matrices = mesh.pose_len() as u32;
        let mut slot = Slot {
            ty: None,
            words,
            offsets: [0, skin, indices, cdf, bones],
            triangles: mesh.triangle_count() as u32,
            bones: if mesh.is_skinned() {
                mesh.bone_positions.len() as u32
            } else {
                0
            },
            matrices,
            flags: FLAG_BOUND | if mesh.is_skinned() { FLAG_SKINNED } else { 0 },
            pose: MeshPose::default(),
            ..Default::default()
        };
        slot.set_pose_from(pose);
        slot
    }

    fn texture(t: &TextureData) -> Slot {
        Slot {
            words: t.texels.iter().flat_map(|c| c.map(f32::to_bits)).collect(),
            width: t.width,
            height: t.height,
            flags: FLAG_BOUND,
            ..Default::default()
        }
    }

    fn set_pose_from(&mut self, pose: MeshPose) {
        self.pose = pose;
        self.pose_scratch = vec![0.0; self.pose_vec4s() as usize * 4];
    }

    /// vec4s of `di_pose` the slot takes: the interval, then two sets.
    fn pose_vec4s(&self) -> u32 {
        if self.matrices == 0 {
            0
        } else {
            1 + 2 * self.matrices * 4
        }
    }

    /// Fills the pose scratch: matrices past the pose's length are identity,
    /// as `fx_core::data` reads them.
    fn fill_pose(&mut self) {
        let n = self.matrices as usize;
        let s = &mut self.pose_scratch;
        if s.is_empty() {
            return;
        }
        s[..4].copy_from_slice(&[self.pose.interval, 0.0, 0.0, 0.0]);
        for (set, mats) in [&self.pose.current, &self.pose.previous].into_iter().enumerate() {
            for i in 0..n {
                let m = mats.get(i).copied().unwrap_or(bevy_math::Mat4::IDENTITY);
                let at = 4 + (set * n + i) * 16;
                s[at..at + 16].copy_from_slice(&m.to_cols_array());
            }
        }
    }
}

/// Bind group 1 of the update and spawn kernels.
pub(crate) struct DataBindings {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    info: wgpu::Buffer,
    data: wgpu::Buffer,
    pose: wgpu::Buffer,
    slots: Vec<Slot>,
}

fn storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    let bytes: &[u8] = if words.is_empty() {
        &[0u8; 16]
    } else {
        bytemuck::cast_slice(words)
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

impl DataBindings {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx_gpu data interfaces"),
            entries: &[storage(0), storage(1), storage(2)],
        })
    }

    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, system: &CookedSystem) -> Self {
        let layout = layout.clone();
        let slots = system
            .data_interfaces
            .iter()
            .map(|d| Slot {
                ty: Some(d.ty),
                ..Default::default()
            })
            .collect();
        let empty = |l| buffer(device, l, &[]);
        let (info, data, pose) = (empty("fx di info"), empty("fx di data"), empty("fx di pose"));
        let bind_group = Self::bind(device, &layout, &info, &data, &pose);
        let mut b = DataBindings {
            layout,
            bind_group,
            info,
            data,
            pose,
            slots,
        };
        b.rebuild(device);
        b
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        info: &wgpu::Buffer,
        data: &wgpu::Buffer,
        pose: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx data interfaces"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pose.as_entire_binding(),
                },
            ],
        })
    }

    /// Repacks every slot into new buffers.
    fn rebuild(&mut self, device: &wgpu::Device) {
        let mut info = Vec::with_capacity(self.slots.len() * INFO_ROWS * 4);
        let mut data = Vec::new();
        let mut pose_vec4s = 0u32;
        for s in &mut self.slots {
            let base = data.len() as u32;
            data.extend_from_slice(&s.words);
            s.pose_offset = pose_vec4s;
            pose_vec4s += s.pose_vec4s();
            let o = s.offsets.map(|x| base + x);
            info.extend([o[0], o[1], o[2], o[3]]);
            info.extend([o[4], s.triangles, s.bones, s.pose_offset]);
            info.extend([s.width, s.height, s.matrices, s.flags]);
        }
        let mut pose = vec![0u32; pose_vec4s as usize * 4];
        for s in &mut self.slots {
            s.fill_pose();
            let at = s.pose_offset as usize * 4;
            for (i, x) in s.pose_scratch.iter().enumerate() {
                pose[at + i] = x.to_bits();
            }
        }
        self.info = buffer(device, "fx di info", &info);
        self.data = buffer(device, "fx di data", &data);
        self.pose = buffer(device, "fx di pose", &pose);
        self.bind_group = Self::bind(device, &self.layout, &self.info, &self.data, &self.pose);
    }

    fn index(&self, system: &CookedSystem, name: &str, ok: impl Fn(DiType) -> bool) -> Option<usize> {
        system
            .data_interfaces
            .iter()
            .position(|d| d.name == name && ok(d.ty))
    }

    pub fn set_mesh(
        &mut self,
        device: &wgpu::Device,
        system: &CookedSystem,
        name: &str,
        mesh: &MeshData,
        pose: MeshPose,
    ) -> bool {
        let Some(i) = self.index(system, name, DiType::is_mesh) else {
            return false;
        };
        let ty = self.slots[i].ty;
        self.slots[i] = Slot { ty, ..Slot::mesh(mesh, pose) };
        self.rebuild(device);
        true
    }

    pub fn set_mesh_pose(
        &mut self,
        queue: &wgpu::Queue,
        system: &CookedSystem,
        name: &str,
        matrices: &[bevy_math::Mat4],
        interval: f32,
    ) -> bool {
        let Some(i) = self.index(system, name, DiType::is_mesh) else {
            return false;
        };
        let s = &mut self.slots[i];
        if s.flags & FLAG_BOUND == 0 {
            return false;
        }
        s.pose.advance(matrices, interval);
        s.fill_pose();
        queue.write_buffer(
            &self.pose,
            s.pose_offset as u64 * 16,
            bytemuck::cast_slice(&s.pose_scratch),
        );
        true
    }

    pub fn set_texture(
        &mut self,
        device: &wgpu::Device,
        system: &CookedSystem,
        name: &str,
        texture: &TextureData,
    ) -> bool {
        let Some(i) = self.index(system, name, |t| t == DiType::Texture) else {
            return false;
        };
        let ty = self.slots[i].ty;
        self.slots[i] = Slot { ty, ..Slot::texture(texture) };
        self.rebuild(device);
        true
    }

    pub fn bytes(&self) -> u64 {
        self.info.size() + self.data.size() + self.pose.size()
    }
}
