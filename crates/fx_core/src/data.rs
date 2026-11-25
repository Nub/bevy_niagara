//! Data interfaces: scene and asset data modules read (Niagara's Data
//! Interfaces).
//!
//! A data interface is a typed `User.` parameter declared in the effect's
//! `user_parameters` beside the value parameters:
//!
//! ```ron
//! user_parameters: {
//!     "User.Character": SkeletalMesh("models/character.glb"),
//!     "User.Rock": StaticMesh("sphere"),
//!     "User.Pattern": Texture("textures/pattern.png"),
//!     "User.Fade": FloatCurve([(0.0, 1.0), (1.0, 0.0)]),
//!     "User.Tint": ColorCurve([(0.0, (1.0, 0.5, 0.2, 1.0)), (1.0, (0.2, 0.4, 1.0, 1.0))]),
//!     "User.Camera": CameraQuery,
//!     "User.Depth": SceneDepth,
//! }
//! ```
//!
//! Modules take one through an input of its type, written
//! `Link(SkeletalMesh, "User.Character")`. Meshes and textures name a default
//! asset (what previews and the viewer show); the host binds its own data at
//! run time ([`MeshData`], [`MeshPose`], [`TextureData`]). Curves are baked
//! into the effect. The camera and scene depth come from the view the effect
//! is drawn for.
//!
//! The sampling functions here are the CPU reference; `fx_gpu` mirrors them
//! in WGSL (`wgsl/data.wgsl`).

use bevy_math::{Mat4, Vec3, Vec4};
use serde::Serialize;

use crate::cook::CookedCurve;
use crate::model::Span;
use crate::types::{Curve, ValueType};

/// The kinds of data interface (Niagara's names in [`DiType::niagara_name`]).
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum DiType {
    SkeletalMesh,
    StaticMesh,
    Texture,
    FloatCurve,
    Vec2Curve,
    VectorCurve,
    Vec4Curve,
    ColorCurve,
    CameraQuery,
    SceneDepth,
}

impl DiType {
    pub const ALL: [DiType; 10] = [
        DiType::SkeletalMesh,
        DiType::StaticMesh,
        DiType::Texture,
        DiType::FloatCurve,
        DiType::Vec2Curve,
        DiType::VectorCurve,
        DiType::Vec4Curve,
        DiType::ColorCurve,
        DiType::CameraQuery,
        DiType::SceneDepth,
    ];

    /// The name written in source.
    pub fn name(self) -> &'static str {
        match self {
            DiType::SkeletalMesh => "SkeletalMesh",
            DiType::StaticMesh => "StaticMesh",
            DiType::Texture => "Texture",
            DiType::FloatCurve => "FloatCurve",
            DiType::Vec2Curve => "Vec2Curve",
            DiType::VectorCurve => "VectorCurve",
            DiType::Vec4Curve => "Vec4Curve",
            DiType::ColorCurve => "ColorCurve",
            DiType::CameraQuery => "CameraQuery",
            DiType::SceneDepth => "SceneDepth",
        }
    }

    /// Niagara's data interface name.
    pub fn niagara_name(self) -> &'static str {
        match self {
            DiType::SkeletalMesh => "Skeletal Mesh",
            DiType::StaticMesh => "Static Mesh",
            DiType::Texture => "Texture Sample",
            DiType::FloatCurve => "Curve for Floats",
            DiType::Vec2Curve => "Vector 2D Curve",
            DiType::VectorCurve => "Vector Curve",
            DiType::Vec4Curve => "Vector 4 Curve",
            DiType::ColorCurve => "Color Curve",
            DiType::CameraQuery => "Camera Query",
            DiType::SceneDepth => "Scene Depth (Collision Query)",
        }
    }

    pub fn from_name(name: &str) -> Option<DiType> {
        DiType::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The value type of a curve data interface.
    pub fn curve_type(self) -> Option<ValueType> {
        match self {
            DiType::FloatCurve => Some(ValueType::Float),
            DiType::Vec2Curve => Some(ValueType::Vec2),
            DiType::VectorCurve => Some(ValueType::Vec3),
            DiType::Vec4Curve => Some(ValueType::Vec4),
            DiType::ColorCurve => Some(ValueType::Color),
            _ => None,
        }
    }

    /// The curve data interface holding curves of `t`.
    pub fn for_curve(t: ValueType) -> Option<DiType> {
        DiType::ALL.into_iter().find(|d| d.curve_type() == Some(t))
    }

    /// Whether it names a default asset (`SkeletalMesh("path")`).
    pub fn has_asset(self) -> bool {
        matches!(
            self,
            DiType::SkeletalMesh | DiType::StaticMesh | DiType::Texture
        )
    }

    /// Whether it is a mesh (sampled by the mesh location modules).
    pub fn is_mesh(self) -> bool {
        matches!(self, DiType::SkeletalMesh | DiType::StaticMesh)
    }

    /// How it is declared, for help messages.
    pub fn example(self) -> &'static str {
        match self {
            DiType::SkeletalMesh => "SkeletalMesh(\"models/character.glb\")",
            DiType::StaticMesh => "StaticMesh(\"sphere\")",
            DiType::Texture => "Texture(\"textures/pattern.png\")",
            DiType::FloatCurve => "FloatCurve([(0.0, 1.0), (1.0, 0.0)])",
            DiType::Vec2Curve => "Vec2Curve([(0.0, (1.0, 1.0)), (1.0, (0.0, 0.0))])",
            DiType::VectorCurve => "VectorCurve([(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 1.0, 0.0))])",
            DiType::Vec4Curve => "Vec4Curve([(0.0, (0.0, 0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0, 1.0))])",
            DiType::ColorCurve => {
                "ColorCurve([(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0, 0.0))])"
            }
            DiType::CameraQuery => "CameraQuery",
            DiType::SceneDepth => "SceneDepth",
        }
    }
}

/// A data interface declared in `user_parameters`.
#[derive(Clone, Debug, PartialEq)]
pub struct DataInterfaceSource {
    /// Full name, `User.Character`.
    pub name: String,
    pub ty: DiType,
    /// Default asset of a mesh or texture: an asset path (or a built-in mesh
    /// name for static meshes); `None` until the host binds one.
    pub asset: Option<String>,
    /// The keys of a curve data interface.
    pub curve: Option<Curve>,
    pub span: Span,
}

impl Serialize for DataInterfaceSource {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let variant = self.ty.name();
        let index = self.ty as u32;
        match (&self.asset, &self.curve) {
            (Some(a), _) => s.serialize_newtype_variant("DataInterface", index, variant, a),
            (None, Some(c)) => match c {
                Curve::Float(k) => s.serialize_newtype_variant("DataInterface", index, variant, k),
                Curve::Vec2(k) => s.serialize_newtype_variant("DataInterface", index, variant, k),
                Curve::Vec3(k) => s.serialize_newtype_variant("DataInterface", index, variant, k),
                Curve::Vec4(k) | Curve::Color(k) => {
                    s.serialize_newtype_variant("DataInterface", index, variant, k)
                }
            },
            (None, None) => s.serialize_unit_variant("DataInterface", index, variant),
        }
    }
}

/// A data interface in the cooked effect; modules refer to it by index.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedDataInterface {
    pub name: String,
    pub ty: DiType,
    pub asset: Option<String>,
    pub curve: Option<CookedCurve>,
}

/// "Skeletal Mesh Location"'s Sampling Mode values.
pub const SAMPLE_SURFACE: i32 = 0;
pub const SAMPLE_BONES: i32 = 1;

/// What scene depth sampling gives where nothing is known (off screen, sky,
/// no depth bound): "nothing there".
pub const NO_SCENE_DEPTH: f32 = 1.0e30;

// ------------------------------------------------------------------ meshes

/// A mesh the mesh location modules sample, in its bind (rest) pose. A
/// static mesh has no joints; a skinned mesh gives four joints and weights
/// per vertex, indices into the [`MeshPose`] matrices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Per vertex, empty for a static mesh.
    pub joints: Vec<[u32; 4]>,
    /// Per vertex, empty for a static mesh.
    pub weights: Vec<[f32; 4]>,
    /// Triangle list.
    pub indices: Vec<u32>,
    /// Cumulative triangle area in the bind pose, normalised (last = 1):
    /// surface samples are spread evenly by area.
    pub area_cdf: Vec<f32>,
    /// Bind-pose origin of each joint (Bones sampling), in mesh space.
    pub bone_positions: Vec<[f32; 3]>,
}

impl MeshData {
    /// A static mesh. Normals may be empty (then flat normals are used).
    pub fn new(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        let mut m = MeshData {
            positions,
            normals,
            indices,
            ..Default::default()
        };
        m.finish();
        m
    }

    /// A skinned mesh: per-vertex joints and weights, and each joint's
    /// inverse bind matrix (the bind pose of joint `j` is its inverse).
    pub fn skinned(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        indices: Vec<u32>,
        joints: Vec<[u32; 4]>,
        weights: Vec<[f32; 4]>,
        inverse_bindposes: &[Mat4],
    ) -> Self {
        let mut m = MeshData {
            positions,
            normals,
            indices,
            joints,
            weights,
            bone_positions: inverse_bindposes
                .iter()
                .map(|ib| ib.inverse().w_axis.truncate().to_array())
                .collect(),
            ..Default::default()
        };
        m.finish();
        m
    }

    /// Checks the arrays agree and builds the area table; fills missing
    /// normals with face normals and normalises weights.
    fn finish(&mut self) {
        let n = self.positions.len();
        self.indices.truncate(self.indices.len() / 3 * 3);
        if self.indices.iter().any(|&i| i as usize >= n) {
            self.indices.clear();
        }
        if self.normals.len() != n {
            let mut acc = vec![Vec3::ZERO; n];
            for t in self.indices.chunks_exact(3) {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(self.positions[i as usize]));
                let f = (b - a).cross(c - a);
                for &i in t {
                    acc[i as usize] += f;
                }
            }
            self.normals = acc
                .iter()
                .map(|v| v.try_normalize().unwrap_or(Vec3::Y).to_array())
                .collect();
        }
        if self.joints.len() != n || self.weights.len() != n {
            self.joints.clear();
            self.weights.clear();
        }
        for w in &mut self.weights {
            let s: f32 = w.iter().sum();
            if s > 0.0 {
                w.iter_mut().for_each(|x| *x /= s);
            } else {
                *w = [1.0, 0.0, 0.0, 0.0];
            }
        }
        let mut total = 0.0f64;
        let areas: Vec<f64> = self
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(self.positions[i as usize]));
                (b - a).cross(c - a).length() as f64 * 0.5
            })
            .collect();
        let sum: f64 = areas.iter().sum();
        self.area_cdf = areas
            .iter()
            .map(|a| {
                total += if sum > 0.0 { *a / sum } else { 1.0 / areas.len() as f64 };
                total as f32
            })
            .collect();
        if let Some(last) = self.area_cdf.last_mut() {
            *last = 1.0;
        }
    }

    pub fn is_skinned(&self) -> bool {
        !self.joints.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Matrices a pose of this mesh holds: one per joint, or one for a static mesh.
    pub fn pose_len(&self) -> usize {
        if self.is_skinned() {
            self.bone_positions.len().max(1)
        } else {
            1
        }
    }
}

/// Where a mesh is this step and where it was the step before, as matrices
/// from mesh space to the effect's simulation space: one per joint of a
/// skinned mesh (joint world transform times inverse bind matrix, then the
/// effect's inverse transform), one for a static mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshPose {
    pub current: Vec<Mat4>,
    pub previous: Vec<Mat4>,
    /// Seconds between `previous` and `current`; 0 gives no velocity.
    pub interval: f32,
}

impl MeshPose {
    /// A pose that does not move.
    pub fn still(matrices: Vec<Mat4>) -> Self {
        MeshPose {
            previous: matrices.clone(),
            current: matrices,
            interval: 0.0,
        }
    }

    /// Moves to `matrices`, `interval` seconds after the current pose.
    pub fn advance(&mut self, matrices: &[Mat4], interval: f32) {
        if self.current.len() == matrices.len() {
            std::mem::swap(&mut self.previous, &mut self.current);
            self.current.clear();
            self.current.extend_from_slice(matrices);
            self.interval = interval;
        } else {
            *self = MeshPose::still(matrices.to_vec());
        }
    }
}

/// A point sampled on a mesh, in simulation space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshSample {
    pub position: Vec3,
    /// Unit surface normal (zero for bones).
    pub normal: Vec3,
    /// Velocity of the point from the previous pose (m/s).
    pub velocity: Vec3,
}

fn mat(pose: &[Mat4], i: u32) -> Mat4 {
    pose.get(i as usize).copied().unwrap_or(Mat4::IDENTITY)
}

/// The skinning matrix of vertex `v`: weights times joint matrices, summed
/// in joint order (static meshes: the one matrix).
fn skin(mesh: &MeshData, pose: &[Mat4], v: usize) -> Mat4 {
    if !mesh.is_skinned() {
        return mat(pose, 0);
    }
    let (j, w) = (mesh.joints[v], mesh.weights[v]);
    let mut m = mat(pose, j[0]) * w[0];
    for k in 1..4 {
        m += mat(pose, j[k]) * w[k];
    }
    m
}

fn transform_point(m: Mat4, p: Vec3) -> Vec3 {
    (m * Vec4::from((p, 1.0))).truncate()
}

/// Samples the surface of `mesh` in `pose` with three uniform random numbers
/// (`u[0]` picks the triangle by area, `u[1..3]` the point in it). `None`
/// when the mesh has no triangles.
pub fn sample_surface(mesh: &MeshData, pose: &MeshPose, u: [f32; 3]) -> Option<MeshSample> {
    let tris = mesh.triangle_count();
    if tris == 0 {
        return None;
    }
    let t = mesh.area_cdf.partition_point(|&c| c <= u[0]).min(tris - 1);
    let (mut r1, mut r2) = (u[1], u[2]);
    if r1 + r2 > 1.0 {
        r1 = 1.0 - r1;
        r2 = 1.0 - r2;
    }
    let w = [1.0 - r1 - r2, r1, r2];
    let (mut pos, mut prev, mut nrm) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    for (c, wc) in w.iter().enumerate() {
        let v = mesh.indices[t * 3 + c] as usize;
        let p = Vec3::from(mesh.positions[v]);
        let n = Vec3::from(mesh.normals[v]);
        let m = skin(mesh, &pose.current, v);
        let mp = skin(mesh, &pose.previous, v);
        pos += transform_point(m, p) * *wc;
        prev += transform_point(mp, p) * *wc;
        nrm += (m * Vec4::from((n, 0.0))).truncate() * *wc;
    }
    Some(MeshSample {
        position: pos,
        normal: unit_or_zero(nrm),
        velocity: velocity(pos, prev, pose.interval),
    })
}

/// A random joint's position (`u` picks it). `None` for a mesh without joints.
pub fn sample_bone(mesh: &MeshData, pose: &MeshPose, u: f32) -> Option<MeshSample> {
    let n = mesh.bone_positions.len();
    if n == 0 || !mesh.is_skinned() {
        return None;
    }
    let b = ((u * n as f32) as usize).min(n - 1);
    let p = Vec3::from(mesh.bone_positions[b]);
    let pos = transform_point(mat(&pose.current, b as u32), p);
    let prev = transform_point(mat(&pose.previous, b as u32), p);
    Some(MeshSample {
        position: pos,
        normal: Vec3::ZERO,
        velocity: velocity(pos, prev, pose.interval),
    })
}

fn velocity(pos: Vec3, prev: Vec3, interval: f32) -> Vec3 {
    if interval > 0.0 {
        (pos - prev) / interval
    } else {
        Vec3::ZERO
    }
}

fn unit_or_zero(v: Vec3) -> Vec3 {
    let l = v.length();
    if l > 1e-12 { v / l } else { Vec3::ZERO }
}

// ------------------------------------------------------------------ textures

/// A texture for Sample Texture: linear RGBA texels, row by row from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[f32; 4]>,
}

impl TextureData {
    pub fn new(width: u32, height: u32, texels: Vec<[f32; 4]>) -> Self {
        assert_eq!(texels.len(), width as usize * height as usize, "texel count");
        TextureData {
            width,
            height,
            texels,
        }
    }
}

/// Bilinear sample at `uv` (0..1 across the texture, (0, 0) the top-left
/// corner), wrapping at the edges as Niagara's default sampler does. An
/// empty texture gives white.
pub fn sample_texture(t: &TextureData, uv: [f32; 2]) -> [f32; 4] {
    if t.width == 0 || t.height == 0 {
        return [1.0; 4];
    }
    let (w, h) = (t.width as i32, t.height as i32);
    let x = uv[0] * w as f32 - 0.5;
    let y = uv[1] * h as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let wrap = |i: f32, n: i32| (i as i32).rem_euclid(n);
    let (xa, xb) = (wrap(x0, w), wrap(x0 + 1.0, w));
    let (ya, yb) = (wrap(y0, h), wrap(y0 + 1.0, h));
    let at = |x: i32, y: i32| Vec4::from(t.texels[(y * w + x) as usize]);
    let top = at(xa, ya).lerp(at(xb, ya), fx);
    let bottom = at(xa, yb).lerp(at(xb, yb), fx);
    top.lerp(bottom, fy).to_array()
}

// ------------------------------------------------------------------ camera

/// The camera an effect is seen from, in its simulation space (Camera Query).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraData {
    pub position: Vec3,
    /// Unit view direction.
    pub forward: Vec3,
}

/// What a bound data interface holds on the CPU.
#[derive(Clone, Debug, Default)]
pub enum DiBinding {
    #[default]
    None,
    Mesh(std::sync::Arc<MeshData>, MeshPose),
    Texture(std::sync::Arc<TextureData>),
}
