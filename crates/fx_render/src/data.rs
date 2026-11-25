//! Data interfaces in Bevy: what an effect's SkeletalMesh, StaticMesh and
//! Texture data interfaces read (see `fx_core::data`).
//!
//! Main world (`PostUpdate`, after transform propagation), every frame:
//! each mesh or texture data interface of every [`ParticleEffect`] is
//! resolved to its binding ([`ParticleEffect::bind_entity`],
//! [`ParticleEffect::bind_mesh`], [`ParticleEffect::bind_texture`]) or else to
//! its default asset; the Bevy `Mesh` / `Image` is converted once per asset
//! (cached in `Arc`s) to `fx_core`'s [`MeshData`] / [`TextureData`], and a
//! mesh's pose (matrices from mesh space to the effect's local space) is
//! computed. Extraction hands both to the render world, which uploads the
//! data when it changes and the pose every frame (`fx_gpu::GpuEffect::
//! set_mesh` / `set_mesh_pose` / `set_texture`).
//!
//! A SkeletalMesh's default asset (a `.glb` / `.gltf`) is spawned as a child
//! scene of the effect playing its first animation clip in a loop (a
//! `#AnimationN` label picks clip N instead: `"models/Fox.glb#Animation2"`),
//! and the first skinned mesh in it is bound. With [`FxClockMode::Manual`]
//! the clip is driven by the effect's simulation time, so previews are
//! deterministic.

use std::sync::Arc;

use bevy::asset::AssetPath;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use fx_core::data::{DiType, MeshData, TextureData};

use crate::{BuiltinMeshes, FxClock, FxClockMode, ParticleEffect};

/// What a data interface of a [`ParticleEffect`] is bound to.
#[derive(Clone, Debug, PartialEq)]
pub enum DataBinding {
    /// An entity with a `Mesh3d` (mesh data interfaces): skinned when it has a
    /// `SkinnedMesh` (posed by its joints), otherwise posed by its
    /// `GlobalTransform`.
    Entity(Entity),
    /// A mesh asset, still at the effect's origin (mesh data interfaces).
    Mesh(Handle<Mesh>),
    /// An image (Texture data interfaces).
    Texture(Handle<Image>),
}

/// A data interface's value for one frame, extracted to the render world.
#[derive(Clone)]
pub(crate) enum DiValue {
    /// Mesh data and its pose: one matrix per joint (skinned) or one.
    Mesh { data: Arc<MeshData>, pose: Vec<Mat4> },
    Texture(Arc<TextureData>),
}

/// One effect's data interface values this frame.
#[derive(Default)]
pub(crate) struct EffectData {
    pub values: Vec<(String, DiValue)>,
    /// A bound or default asset is still loading (the effect is not ready).
    pub pending: bool,
}

/// Main world: the data interfaces' state and this frame's values.
#[derive(Resource, Default)]
pub(crate) struct DataInterfaces {
    effects: HashMap<Entity, EffectState>,
    meshes: HashMap<MeshKey, Cached<MeshData>>,
    textures: HashMap<AssetId<Image>, Cached<TextureData>>,
    /// This frame's values per effect.
    pub frame: HashMap<Entity, EffectData>,
    /// Seconds since the previous frame's poses.
    pub interval: f32,
}

type MeshKey = (AssetId<Mesh>, Option<AssetId<SkinnedMeshInverseBindposes>>);

/// A converted asset (`None`: it cannot be converted; the error was logged).
struct Cached<T> {
    data: Option<Arc<T>>,
    used: bool,
}

/// Per effect: the system it was set up for and its data interfaces' defaults.
struct EffectState {
    system: usize,
    defaults: Vec<Fallback>,
}

/// What an unbound data interface falls back to.
enum Fallback {
    /// Not created yet (or nothing: no default asset).
    Unset,
    None,
    Mesh(Handle<Mesh>),
    Texture(Handle<Image>),
    Scene(SceneState),
}

/// A SkeletalMesh default: a glTF scene spawned under the effect.
struct SceneState {
    root: Entity,
    graph: Handle<BlendGraph>,
    node: BlendNodeIndex,
    clip: Handle<AnimationClip>,
    /// The scene's `AnimationPlayer`, once found and started.
    player: Option<Entity>,
    /// The skinned mesh it binds.
    mesh: Option<Entity>,
}

impl ParticleEffect {
    /// Binds data interface `name` (`"User.Character"`) to `binding`,
    /// replacing its default asset. Returns `self` for chaining.
    pub fn bind(&mut self, name: impl Into<String>, binding: DataBinding) -> &mut Self {
        let name = name.into();
        self.bindings.retain(|(n, _)| *n != name);
        self.bindings.push((name, binding));
        self
    }

    /// Binds mesh data interface `name` (SkeletalMesh or StaticMesh) to an
    /// entity with a `Mesh3d`. A skinned entity (`SkinnedMesh`) is sampled in
    /// its animated pose; another one where its `GlobalTransform` puts it.
    /// Particles follow it as it moves (the effect's own transform is
    /// accounted for).
    pub fn bind_entity(&mut self, name: impl Into<String>, entity: Entity) -> &mut Self {
        self.bind(name, DataBinding::Entity(entity))
    }

    /// Binds mesh data interface `name` to a mesh asset, placed at the
    /// effect's origin (in its local space).
    pub fn bind_mesh(&mut self, name: impl Into<String>, mesh: Handle<Mesh>) -> &mut Self {
        self.bind(name, DataBinding::Mesh(mesh))
    }

    /// Binds Texture data interface `name` to an image. Formats read:
    /// Rgba8UnormSrgb, Rgba8Unorm, Rgba16Float and Rgba32Float (others log an
    /// error); the image must keep its main-world data
    /// (`RenderAssetUsages::MAIN_WORLD`).
    pub fn bind_texture(&mut self, name: impl Into<String>, image: Handle<Image>) -> &mut Self {
        self.bind(name, DataBinding::Texture(image))
    }

    /// Removes the binding of `name`: it reads its default asset again.
    pub fn unbind(&mut self, name: &str) -> &mut Self {
        self.bindings.retain(|(n, _)| n != name);
        self
    }

    /// [`ParticleEffect::bind`] for building an effect.
    pub fn with_binding(mut self, name: impl Into<String>, binding: DataBinding) -> Self {
        self.bind(name, binding);
        self
    }
}

fn is_gltf(path: &AssetPath) -> bool {
    matches!(
        path.get_full_extension().as_deref(),
        Some("glb") | Some("gltf")
    )
}

/// Starts loading (or spawning) data interface `ty`'s default asset.
fn make_default(
    commands: &mut Commands,
    effect: Entity,
    ty: DiType,
    asset: Option<&String>,
    asset_server: &AssetServer,
    builtins: &BuiltinMeshes,
    graphs: &mut Assets<BlendGraph>,
) -> Fallback {
    let Some(asset) = asset else {
        return Fallback::None;
    };
    match ty {
        DiType::StaticMesh => {
            if let Some(h) = builtins.0.get(asset.as_str()) {
                return Fallback::Mesh(h.clone());
            }
            let path = AssetPath::parse(asset).into_owned();
            if path.label().is_none() && is_gltf(&path) {
                // A glTF file: its first primitive.
                Fallback::Mesh(asset_server.load(
                    GltfAssetLabel::Primitive {
                        mesh: 0,
                        primitive: 0,
                    }
                    .from_asset(path),
                ))
            } else {
                Fallback::Mesh(asset_server.load(path))
            }
        }
        DiType::Texture => Fallback::Texture(asset_server.load(asset.clone())),
        DiType::SkeletalMesh => {
            let path = AssetPath::parse(asset).into_owned();
            if !is_gltf(&path) {
                error!(
                    "skeletal mesh \"{asset}\": only glTF files (.glb, .gltf) can be a default skeletal mesh"
                );
                return Fallback::None;
            }
            let clip_index = match path.label() {
                None => 0,
                Some(l) => match l.strip_prefix("Animation").and_then(|n| n.parse().ok()) {
                    Some(n) => n,
                    None => {
                        error!("skeletal mesh \"{asset}\": the label must be #AnimationN (the clip to play)");
                        0
                    }
                },
            };
            let file = path.without_label().into_owned();
            let clip: Handle<AnimationClip> =
                asset_server.load(GltfAssetLabel::Animation(clip_index).from_asset(file.clone()));
            let (graph, node) = BlendGraph::from_clip(clip.clone());
            let root = commands
                .spawn((
                    WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(file))),
                    Transform::IDENTITY,
                    Visibility::Inherited,
                    ChildOf(effect),
                    Name::new(format!("fx skeletal mesh {asset}")),
                ))
                .id();
            Fallback::Scene(SceneState {
                root,
                graph: graphs.add(graph),
                node,
                clip,
                player: None,
                mesh: None,
            })
        }
        _ => Fallback::None,
    }
}

/// `PostUpdate`, before animation: drives the default skeletal meshes'
/// clips by the effect clock in manual mode (deterministic previews), and
/// pauses them with the clock otherwise.
pub(crate) fn drive_animations(
    state: Res<DataInterfaces>,
    clock: Res<FxClock>,
    clips: Res<Assets<AnimationClip>>,
    mut players: Query<&mut AnimationPlayer>,
) {
    for es in state.effects.values() {
        for d in &es.defaults {
            let Fallback::Scene(s) = d else { continue };
            let Some(mut player) = s.player.and_then(|p| players.get_mut(p).ok()) else {
                continue;
            };
            let Some(anim) = player.animation_mut(s.node) else {
                continue;
            };
            match clock.mode {
                FxClockMode::Manual => {
                    let duration = clips.get(&s.clip).map_or(0.0, |c| c.duration() as f64);
                    let t = clock.total_steps as f64 * clock.step as f64;
                    let t = if duration > 0.0 { t % duration } else { 0.0 };
                    anim.pause();
                    anim.set_seek_time(t as f32);
                }
                FxClockMode::Realtime => {
                    if clock.paused {
                        anim.pause();
                    } else {
                        anim.resume();
                    }
                }
            }
        }
    }
}

/// `PostUpdate`, after transform propagation: resolves every mesh and
/// texture data interface, converts new assets and computes the poses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_data_interfaces(
    mut commands: Commands,
    mut state: ResMut<DataInterfaces>,
    clock: Res<FxClock>,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
    builtins: Res<BuiltinMeshes>,
    (meshes, images, bindposes): (Res<Assets<Mesh>>, Res<Assets<Image>>, Res<Assets<SkinnedMeshInverseBindposes>>),
    (mut mesh_events, mut image_events): (MessageReader<AssetEvent<Mesh>>, MessageReader<AssetEvent<Image>>),
    mut graphs: ResMut<Assets<BlendGraph>>,
    effects: Query<(Entity, &ParticleEffect, &GlobalTransform)>,
    targets: Query<(&Mesh3d, Option<&SkinnedMesh>, &GlobalTransform)>,
    transforms: Query<&GlobalTransform>,
    children: Query<&Children>,
    skinned: Query<(), (With<SkinnedMesh>, With<Mesh3d>)>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let state = &mut *state;
    // Changed assets are converted again.
    for ev in mesh_events.read() {
        if let AssetEvent::Modified { id } | AssetEvent::Removed { id } = ev {
            state.meshes.retain(|k, _| k.0 != *id);
        }
    }
    for ev in image_events.read() {
        if let AssetEvent::Modified { id } | AssetEvent::Removed { id } = ev {
            state.textures.remove(id);
        }
    }
    state.interval = match clock.mode {
        FxClockMode::Manual => clock.steps_this_frame() as f32 * clock.step,
        FxClockMode::Realtime => time.delta_secs(),
    };
    state.frame.clear();
    let mut live = HashSet::new();

    for (entity, effect, transform) in &effects {
        live.insert(entity);
        let system = Arc::as_ptr(&effect.system) as usize;
        let dis = &effect.system.data_interfaces;
        let es = state.effects.entry(entity).or_insert_with(|| EffectState {
            system,
            defaults: Vec::new(),
        });
        if es.system != system || es.defaults.len() != dis.len() {
            for d in es.defaults.drain(..) {
                if let Fallback::Scene(s) = d {
                    commands.entity(s.root).try_despawn();
                }
            }
            es.system = system;
            es.defaults = dis.iter().map(|_| Fallback::Unset).collect();
        }
        let local_from_world = transform.to_matrix().inverse();
        let mut out = EffectData::default();

        for (i, di) in dis.iter().enumerate() {
            if !matches!(
                di.ty,
                DiType::SkeletalMesh | DiType::StaticMesh | DiType::Texture
            ) {
                continue;
            }
            let bound = effect
                .bindings
                .iter()
                .find(|(n, _)| *n == di.name)
                .map(|(_, b)| b.clone());
            let default = &mut es.defaults[i];
            if bound.is_some() {
                // A binding replaces the default: drop it.
                if let Fallback::Scene(s) = std::mem::replace(default, Fallback::Unset) {
                    commands.entity(s.root).try_despawn();
                }
            } else if matches!(default, Fallback::Unset) {
                *default = make_default(
                    &mut commands,
                    entity,
                    di.ty,
                    di.asset.as_ref(),
                    &asset_server,
                    &builtins,
                    &mut graphs,
                );
            }
            let binding = match (bound, &mut *default) {
                (Some(b), _) => b,
                (None, Fallback::Mesh(h)) => DataBinding::Mesh(h.clone()),
                (None, Fallback::Texture(h)) => DataBinding::Texture(h.clone()),
                (None, Fallback::Scene(s)) => {
                    if !scene_ready(s, &children, &skinned, &mut players, &mut commands, &clock) {
                        out.pending = true;
                        continue;
                    }
                    DataBinding::Entity(s.mesh.unwrap())
                }
                (None, _) => continue,
            };

            let value = match (di.ty, binding) {
                (DiType::Texture, DataBinding::Texture(h)) => {
                    let Some(image) = images.get(&h) else {
                        out.pending |= !asset_server.load_state(&h).is_failed();
                        continue;
                    };
                    let cached = state.textures.entry(h.id()).or_insert_with(|| Cached {
                        data: match texture_data(image) {
                            Ok(t) => Some(Arc::new(t)),
                            Err(e) => {
                                error!("texture for data interface \"{}\": {e}", di.name);
                                None
                            }
                        },
                        used: false,
                    });
                    cached.used = true;
                    let Some(t) = &cached.data else { continue };
                    DiValue::Texture(t.clone())
                }
                (DiType::SkeletalMesh | DiType::StaticMesh, DataBinding::Mesh(h)) => {
                    let Some(data) =
                        mesh_cached(&mut state.meshes, &meshes, &h, None, &di.name, &asset_server, &mut out)
                    else {
                        continue;
                    };
                    DiValue::Mesh {
                        data,
                        pose: vec![Mat4::IDENTITY],
                    }
                }
                (DiType::SkeletalMesh | DiType::StaticMesh, DataBinding::Entity(e)) => {
                    let Ok((mesh3d, skin, gt)) = targets.get(e) else {
                        warn_once!(
                            "data interface \"{}\" is bound to an entity without a Mesh3d and GlobalTransform",
                            di.name
                        );
                        continue;
                    };
                    let ibp = skin.and_then(|s| bindposes.get(&s.inverse_bindposes));
                    if skin.is_some() && ibp.is_none() {
                        out.pending = true;
                        continue;
                    }
                    let Some(data) = mesh_cached(
                        &mut state.meshes,
                        &meshes,
                        &mesh3d.0,
                        skin.zip(ibp).map(|(s, ibp)| (s.inverse_bindposes.id(), &ibp[..])),
                        &di.name,
                        &asset_server,
                        &mut out,
                    ) else {
                        continue;
                    };
                    let pose = match (skin, ibp) {
                        (Some(skin), Some(ibp)) if data.is_skinned() => skin
                            .joints
                            .iter()
                            .zip(ibp.iter())
                            .map(|(&j, ib)| {
                                let joint = transforms.get(j).map_or(Mat4::IDENTITY, |t| t.to_matrix());
                                local_from_world * joint * *ib
                            })
                            .collect(),
                        _ => vec![local_from_world * gt.to_matrix()],
                    };
                    DiValue::Mesh { data, pose }
                }
                (ty, b) => {
                    warn_once!(
                        "data interface \"{}\" ({}) cannot be bound to {b:?}",
                        di.name,
                        ty.name()
                    );
                    continue;
                }
            };
            out.values.push((di.name.clone(), value));
        }
        state.frame.insert(entity, out);
    }

    state.effects.retain(|e, _| live.contains(e));
    state.meshes.retain(|_, c| std::mem::take(&mut c.used));
    state.textures.retain(|_, c| std::mem::take(&mut c.used));
}

/// The default scene's skinned mesh, once spawned (and its animation started).
fn scene_ready(
    s: &mut SceneState,
    children: &Query<&Children>,
    skinned: &Query<(), (With<SkinnedMesh>, With<Mesh3d>)>,
    players: &mut Query<&mut AnimationPlayer>,
    commands: &mut Commands,
    clock: &FxClock,
) -> bool {
    if s.mesh.is_none() {
        s.mesh = children.iter_descendants(s.root).find(|&e| skinned.contains(e));
        if s.mesh.is_none() {
            return false;
        }
    }
    if s.player.is_none()
        && let Some(p) = children.iter_descendants(s.root).find(|&e| players.contains(e))
    {
        let mut player = players.get_mut(p).unwrap();
        let anim = player.play(s.node);
        anim.repeat();
        if clock.mode == FxClockMode::Manual || clock.paused {
            anim.pause();
        }
        commands.entity(p).insert(BlendGraphHandle(s.graph.clone()));
        s.player = Some(p);
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn mesh_cached(
    cache: &mut HashMap<MeshKey, Cached<MeshData>>,
    meshes: &Assets<Mesh>,
    handle: &Handle<Mesh>,
    skin: Option<(AssetId<SkinnedMeshInverseBindposes>, &[Mat4])>,
    name: &str,
    asset_server: &AssetServer,
    out: &mut EffectData,
) -> Option<Arc<MeshData>> {
    let key = (handle.id(), skin.map(|s| s.0));
    if let Some(c) = cache.get_mut(&key) {
        c.used = true;
        return c.data.clone();
    }
    let Some(mesh) = meshes.get(handle) else {
        out.pending |= !asset_server.load_state(handle).is_failed();
        return None;
    };
    let data = match mesh_data(mesh, skin.map(|s| s.1)) {
        Ok(m) => Some(Arc::new(m)),
        Err(e) => {
            error!("mesh for data interface \"{name}\": {e}");
            None
        }
    };
    cache.insert(
        key,
        Cached {
            data: data.clone(),
            used: true,
        },
    );
    data
}

/// Converts a Bevy triangle-list mesh: positions, normals, indices and, with
/// `inverse_bindposes`, its joints and weights.
pub fn mesh_data(mesh: &Mesh, inverse_bindposes: Option<&[Mat4]>) -> Result<MeshData, String> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(format!(
            "only triangle lists can be sampled, not {:?}",
            mesh.primitive_topology()
        ));
    }
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p.clone(),
        None => return Err("the mesh has no positions (or no main-world data)".into()),
        _ => return Err("the mesh has no Float32x3 positions".into()),
    };
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(n)) if n.len() == positions.len() => n.clone(),
        _ => Vec::new(),
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U16(i)) => i.iter().map(|&x| x as u32).collect(),
        Some(Indices::U32(i)) => i.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    if indices.len() < 3 {
        return Err("the mesh has no triangles".into());
    }
    let Some(ibp) = inverse_bindposes else {
        return Ok(MeshData::new(positions, normals, indices));
    };
    let joints: Vec<[u32; 4]> = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
        Some(VertexAttributeValues::Uint16x4(j)) => j.iter().map(|j| j.map(u32::from)).collect(),
        Some(VertexAttributeValues::Uint8x4(j)) => j.iter().map(|j| j.map(u32::from)).collect(),
        Some(VertexAttributeValues::Uint32x4(j)) => j.clone(),
        _ => return Err("the skinned mesh has no Uint16x4 joint indices".into()),
    };
    let weights: Vec<[f32; 4]> = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) {
        Some(VertexAttributeValues::Float32x4(w)) => w.clone(),
        Some(VertexAttributeValues::Unorm16x4(w)) => {
            w.iter().map(|w| w.map(|x| x as f32 / 65535.0)).collect()
        }
        Some(VertexAttributeValues::Unorm8x4(w)) => {
            w.iter().map(|w| w.map(|x| x as f32 / 255.0)).collect()
        }
        _ => return Err("the skinned mesh has no joint weights".into()),
    };
    if joints.len() != positions.len() || weights.len() != positions.len() {
        return Err("joint indices or weights do not match the vertices".into());
    }
    if joints.iter().flatten().any(|&j| j as usize >= ibp.len()) {
        return Err("a joint index is past the skin's joints".into());
    }
    Ok(MeshData::skinned(
        positions, normals, indices, joints, weights, ibp,
    ))
}

/// Converts the first mip level of a 2D image to linear RGBA texels.
pub fn texture_data(image: &Image) -> Result<TextureData, String> {
    let (w, h) = (image.width(), image.height());
    let Some(bytes) = &image.data else {
        return Err("the image has no main-world data (RenderAssetUsages::MAIN_WORLD)".into());
    };
    let n = w as usize * h as usize;
    let format = image.texture_descriptor.format;
    let texel_bytes = match format {
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => 4,
        TextureFormat::Rgba16Float => 8,
        TextureFormat::Rgba32Float => 16,
        f => {
            return Err(format!(
                "{f:?} cannot be sampled (Rgba8UnormSrgb, Rgba8Unorm, Rgba16Float or Rgba32Float)"
            ));
        }
    };
    if bytes.len() < n * texel_bytes {
        return Err("the image data is shorter than its size".into());
    }
    let bytes = &bytes[..n * texel_bytes];
    let texels: Vec<[f32; 4]> = match format {
        TextureFormat::Rgba8UnormSrgb => bytes
            .chunks_exact(4)
            .map(|c| {
                [
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                    c[3] as f32 / 255.0,
                ]
            })
            .collect(),
        TextureFormat::Rgba8Unorm => bytes
            .chunks_exact(4)
            .map(|c| [0, 1, 2, 3].map(|k| c[k] as f32 / 255.0))
            .collect(),
        TextureFormat::Rgba16Float => bytes
            .chunks_exact(8)
            .map(|c| [0, 1, 2, 3].map(|k| f16_to_f32(u16::from_le_bytes([c[2 * k], c[2 * k + 1]]))))
            .collect(),
        _ => bytes
            .chunks_exact(16)
            .map(|c| {
                [0, 1, 2, 3].map(|k| f32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]]))
            })
            .collect(),
    };
    Ok(TextureData::new(w, h, texels))
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// IEEE half to single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 if man == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension};

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn srgb_image_is_linearised() {
        let img = Image::new(
            Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![255, 0, 0, 255, 188, 188, 188, 128],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let t = texture_data(&img).unwrap();
        assert_eq!(t.texels[0], [1.0, 0.0, 0.0, 1.0]);
        assert!((t.texels[1][0] - 0.5).abs() < 0.01, "{:?}", t.texels[1]);
        assert!((t.texels[1][3] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn mesh_conversion() {
        let m = Cuboid::from_length(1.0).mesh().build();
        let d = mesh_data(&m, None).unwrap();
        assert_eq!(d.triangle_count(), 12);
        assert!(!d.is_skinned());
        let mut m = m;
        let n = d.positions.len();
        m.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0, 1, 0, 0]; n]),
        );
        m.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[0.5, 0.5, 0.0, 0.0]; n]),
        );
        let d = mesh_data(&m, Some(&[Mat4::IDENTITY, Mat4::from_translation(Vec3::X)])).unwrap();
        assert!(d.is_skinned());
        assert_eq!(d.pose_len(), 2);
        assert_eq!(d.bone_positions[1], [-1.0, 0.0, 0.0]);
    }
}
