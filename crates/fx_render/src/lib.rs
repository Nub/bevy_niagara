//! Bevy plugin for GPU-simulated effects drawn by Niagara's core renderers
//! in Bevy's raster pipeline:
//! - **Sprite**: one camera-facing quad per particle (`sprite.wesl`);
//! - **Ribbon**: particles linked by `Particles.RibbonID` and link order into
//!   strips whose geometry the GPU builds (`ribbon.wesl`, `fx_gpu::ribbon`);
//! - **Mesh**: any Bevy mesh instanced per particle with one indexed
//!   indirect draw (`mesh.wesl`);
//! - **Light**: per-particle point lights, culled and capped on the GPU, read
//!   back (only the capped list) and applied as Bevy `PointLight`s.
//! - **Decal**: a box per particle (`decal.wesl`) projecting the texture onto
//!   the scene surfaces the depth prepass shows inside it; drawn only on
//!   cameras with a `DepthPrepass` (Solari cameras have one);
//! - **Component**: the oldest particles drive pooled child entities
//!   ([`component`]): attributes read back, capped, like lights;
//! - **Volumetric**: particles splatted into a density texture per instance
//!   (`fx_gpu::volume`), ray-marched over the volume box (`volume.wesl`)
//!   against the scene depth.
//!
//! Sprites, ribbons and meshes draw in the Transparent3d phase, depth-tested
//! against the scene and sorted with other transparents by effect centre;
//! opaque meshes write depth there too. Custom materials replace a
//! renderer's fragment shader and read the particle attributes from
//! `fx_render::particle_io::FxVertexOutput` (see `particle_io.wesl`).
//!
//! Main world: spawn a [`ParticleEffect`] (a cooked system and a seed) with a
//! `Transform`; [`FxClock`] decides how many fixed steps run each frame.
//! Render world, each frame:
//! - `prepare_effects` builds an [`fx_gpu::GpuEffect`] per effect and the
//!   per-renderer GPU state (per view for sprites and meshes);
//! - `queue_particles` adds one Transparent3d item per drawn renderer and view
//!   (Transmissive3d for `distortion` sprites, on cameras with ScreenSpaceTransmission);
//! - `simulate` (in the `RenderGraph` schedule, before the cameras) records
//!   the sim steps, the per-view cull/sort, the ribbon builds and the light
//!   gathers into the frame's encoder;
//! - `DrawParticles` binds the renderer's buffers and issues the indirect draw;
//! - `read_lights` (after submission) maps the light lists.
//!
//! No particle counts are read back; the per-frame CPU→GPU traffic is a few
//! uniform writes from fixed-size scratch arrays.

pub mod asset;
pub mod component;
pub mod data;
pub mod game;
pub mod mesh_data;
pub mod prefab;
mod scene;
pub mod stats;
mod textures;

pub use stats::{EffectStats, EmitterStats, FxStats, FxStatsData, FxStatsPlugin};
pub use component::{
    FromParticle, FxComponentAppExt, ParticleAttributes, ParticleComponent,
    apply_particle_components,
};
pub use asset::{FxEffectAsset, FxEffectLoader};
pub use prefab::{FxPrefab, FxPrefabCamera, FxPrefabInstance, FxPrefabNode, FxPrefabPlayer, FxPrefabPlugin};
pub use data::DataBinding;
pub use game::{
    CullReason, EffectControl, EffectMode, FxCommandsExt, FxPoolStats, FxQuality, FxScalability,
    IntoParam, ParamError, PoolStats, find_joint,
};
pub use fx_core::scalability::QualityLevel;
pub use scene::{FxRayTracingScene, FxSceneDepth};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bevy::app::AnimationSystems;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d, TransparentSortingInfo3d};
use bevy::core_pipeline::schedule::camera_driver;
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::math::{Mat4, Vec3, Vec4};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, MeshPipelineSystems, ScreenSpaceTransmission, SetMeshViewBindGroup, Transmissive3d,
    ViewKeyCache,
};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, texture_3d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, RetainedViewEntity};
use bevy::render::{
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy::shader::{Shader, ShaderDefVal, load_shader_library};
use fx_core::cook::{
    CookedComponent, CookedDecal, CookedLight, CookedMesh, CookedRenderer, CookedRibbon,
    CookedSprite, CookedSystem, CookedVolumetric,
};
use fx_core::render::{
    BlendMode, FacingMode, MeshBlendMode, MeshFacingMode, RibbonFacingMode, RibbonLinkOrderMode,
    RibbonUvMode, SortMode, SpriteAlignment,
};
use fx_gpu::{GpuBatch, GpuEffect};
use fx_gpu::gather::{GatherPipelines, ParticleGather};
use fx_gpu::lights::{GpuLight, LightGather, LightPipelines};
use fx_gpu::volume::{VolumeParams, VolumePipelines, VolumeSplat};
use fx_gpu::ribbon::{RibbonBuild, RibbonPipelines};
use fx_gpu::view::{BatchDraw, CullPhase, CullRadius, ViewDraw, ViewDrawConfig, ViewParams, ViewPipelines};
use fx_gpu::wgpu;

/// A particle system instance. Its particles are simulated in the entity's
/// local space and drawn with the entity's `GlobalTransform`.
///
/// Data interfaces (meshes and textures the effect's modules read, declared
/// in its `user_parameters`) read their default asset unless bound with
/// [`ParticleEffect::bind_entity`], [`ParticleEffect::bind_mesh`] or
/// [`ParticleEffect::bind_texture`] (see the `data` module docs).
///
/// Instances made from the same `Arc<CookedSystem>` ([`ParticleEffect::shared`])
/// share one compiled program and one pool of GPU instances (see [`game`]).
#[derive(Component, Clone)]
#[require(Transform, Visibility, EffectControl)]
pub struct ParticleEffect {
    pub system: Arc<CookedSystem>,
    pub seed: u32,
    /// Meshes replacing a mesh renderer's `mesh` setting, by renderer index
    /// (all emitters' renderers in order): any Bevy mesh.
    pub meshes: Vec<(usize, Handle<Mesh>)>,
    /// Data interface bindings by full name (`"User.Character"`).
    pub bindings: Vec<(String, DataBinding)>,
    /// `User.` parameter values, in declaration order.
    user: Vec<fx_core::types::Value>,
}

impl ParticleEffect {
    pub fn new(system: CookedSystem, seed: u32) -> Self {
        Self::shared(Arc::new(system), seed)
    }

    /// An instance of a system loaded once and shared: spawns of the same
    /// `Arc` reuse its compiled program and pooled GPU instances.
    pub fn shared(system: Arc<CookedSystem>, seed: u32) -> Self {
        let user = system.user_params.iter().map(|p| p.value).collect();
        ParticleEffect {
            system,
            seed,
            meshes: Vec::new(),
            bindings: Vec::new(),
            user,
        }
    }

    /// Sets `User.` parameter `name` (`"User.Rate"` or `"Rate"`), checked
    /// against its declared type; applied from the next simulation step.
    /// No allocation on success.
    pub fn set_param(&mut self, name: &str, value: impl IntoParam) -> Result<(), ParamError> {
        let value = value.into_param();
        let Some(i) = game::user_index(&self.system, name) else {
            return Err(ParamError::Unknown {
                name: name.to_string(),
                system: self.system.name.clone(),
            });
        };
        let expected = self.system.user_params[i].value.ty();
        if expected != value.ty() {
            return Err(ParamError::WrongType {
                name: self.system.user_params[i].name.clone(),
                expected,
                found: value.ty(),
            });
        }
        self.user[i] = value;
        Ok(())
    }

    /// Builder form of [`Self::set_param`].
    pub fn with_param(mut self, name: &str, value: impl IntoParam) -> Result<Self, ParamError> {
        self.set_param(name, value)?;
        Ok(self)
    }

    /// The current value of `User.` parameter `name`.
    pub fn param(&self, name: &str) -> Option<fx_core::types::Value> {
        game::user_index(&self.system, name).map(|i| self.user[i])
    }

    /// Draws mesh renderer `renderer` (index among all emitters' renderers,
    /// in order) with `mesh`.
    pub fn with_mesh(mut self, renderer: usize, mesh: Handle<Mesh>) -> Self {
        self.meshes.push((renderer, mesh));
        self
    }
}

/// How the simulation advances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FxClockMode {
    /// Fixed steps of `step` seconds catching up with real time (at most
    /// [`fx_gpu::MAX_STEPS_PER_SUBMIT`] per frame).
    Realtime,
    /// Only the steps asked for with [`FxClock::request_steps`] (headless
    /// preview, scrubbing): frame-rate independent and deterministic.
    Manual,
}

#[derive(Resource, Clone, Debug)]
pub struct FxClock {
    /// Seconds per simulation step.
    pub step: f32,
    pub mode: FxClockMode,
    pub paused: bool,
    accumulator: f32,
    requested: u32,
    /// Steps taken this frame (read by extraction).
    steps_this_frame: u32,
    /// Steps taken since start.
    pub total_steps: u64,
}

impl Default for FxClock {
    fn default() -> Self {
        FxClock {
            step: 1.0 / 60.0,
            mode: FxClockMode::Realtime,
            paused: false,
            accumulator: 0.0,
            requested: 0,
            steps_this_frame: 0,
            total_steps: 0,
        }
    }
}

impl FxClock {
    pub fn manual(step: f32) -> Self {
        FxClock {
            step,
            mode: FxClockMode::Manual,
            ..default()
        }
    }
    /// In manual mode, runs `n` more steps (spread over frames, at most
    /// [`fx_gpu::MAX_STEPS_PER_SUBMIT`] per frame).
    pub fn request_steps(&mut self, n: u32) {
        self.requested += n;
    }
    pub fn steps_this_frame(&self) -> u32 {
        self.steps_this_frame
    }
}

/// Set by the render world once every effect is built, every mesh it draws
/// is loaded and every particle pipeline (and every other pipeline in the
/// cache) has compiled.
#[derive(Resource, Clone, Default)]
pub struct FxReady(pub Arc<AtomicBool>);

impl FxReady {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// `deterministic`: draw unsorted sprites and meshes in UniqueID order (a GPU
/// sort) so that blending rounds the same way every run, and wait for each
/// frame's particle lights so they always lag exactly one frame; for
/// headless previews whose images must match bit for bit.
#[derive(Default)]
pub struct FxPlugin {
    pub deterministic: bool,
}

#[derive(Resource, Clone, Copy)]
struct FxDeterministic(bool);

/// Light lists read back in the render world, waiting for the main world.
#[derive(Resource, Clone, Default)]
struct LightInbox(Arc<Mutex<Vec<LightUpdate>>>);

struct LightUpdate {
    effect: Entity,
    renderer: usize,
    settings: CookedLight,
    lights: Vec<GpuLight>,
    /// The lights are in world space (a `local_space: false` emitter), not the effect's.
    world_space: bool,
}

/// A point light entity driven by a light renderer (a child of its effect).
#[derive(Component, Clone, Copy, Debug)]
pub struct ParticleLight {
    pub renderer: usize,
    pub slot: usize,
}

/// Main world: point light entities per (effect, light renderer).
#[derive(Resource, Default)]
struct LightPools(HashMap<(Entity, usize), Vec<Entity>>);

/// Main world: the built-in meshes of the mesh renderer.
#[derive(Resource, Default)]
struct BuiltinMeshes(HashMap<&'static str, Handle<Mesh>>);

impl Plugin for FxPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "particle_surface.wesl");
        load_shader_library!(app, "particle_io.wesl");
        embedded_asset!(app, "sprite.wesl");
        embedded_asset!(app, "ribbon.wesl");
        embedded_asset!(app, "mesh.wesl");
        embedded_asset!(app, "decal.wesl");
        embedded_asset!(app, "volume.wesl");
        let ready = FxReady::default();
        let inbox = LightInbox::default();
        let components = component::ComponentInbox::default();
        let builtins = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            BuiltinMeshes(
                fx_core::render::MESH_BUILTINS
                    .iter()
                    .filter_map(|&n| Some((n, meshes.add(mesh_data::builtin(n)?))))
                    .collect(),
            )
        };
        let pool_stats = FxPoolStats::default();
        let done = game::DoneInbox::default();
        app.add_plugins(FxPrefabPlugin);
        app.init_asset::<FxEffectAsset>()
            .init_asset_loader::<FxEffectLoader>()
            .init_resource::<FxClock>()
            .init_resource::<FxQuality>()
            .init_resource::<FxScalability>()
            .insert_resource(pool_stats.clone())
            .insert_resource(done.clone())
            .add_systems(
                PostUpdate,
                (game::measure_owner_velocity, game::finish_effects, game::update_effects)
                    .chain()
                    .after(TransformSystems::Propagate)
                    .after(advance_clock),
            )
            .insert_resource(ready.clone())
            .insert_resource(inbox.clone())
            .insert_resource(builtins)
            .init_resource::<LightPools>()
            .insert_resource(components.clone())
            .init_resource::<component::ComponentPools>()
            .add_systems(
                PostUpdate,
                apply_particle_components.before(TransformSystems::Propagate),
            )
            .init_resource::<data::DataInterfaces>()
            .add_systems(
                PostUpdate,
                (advance_clock, data::drive_animations)
                    .chain()
                    .before(AnimationSystems),
            )
            .add_systems(
                PostUpdate,
                data::update_data_interfaces.after(TransformSystems::Propagate),
            )
            .add_systems(PostUpdate, scene::request_depth_binding)
            .add_systems(
                PostUpdate,
                apply_particle_lights.before(TransformSystems::Propagate),
            );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(ready)
            .insert_resource(inbox)
            .insert_resource(components)
            .insert_resource(FxDeterministic(self.deterministic))
            .insert_resource(pool_stats)
            .insert_resource(done)
            .init_resource::<DoneReadback>()
            .init_resource::<ExtractedEffects>()
            .init_resource::<FxSceneDepth>()
            .init_resource::<FxRayTracingScene>()
            .init_resource::<RendererAssetsCache>()
            .init_resource::<PendingMeshes>()
            .add_render_command::<Transparent3d, DrawParticlesCommands>()
            .add_render_command::<Transmissive3d, DrawParticlesCommands>()
            .init_gpu_resource::<SpecializedRenderPipelines<ParticlePipeline>>()
            .add_systems(RenderStartup, init_render.after(MeshPipelineSystems))
            .add_systems(ExtractSchedule, extract_effects)
            .add_systems(
                Render,
                prepare_effects
                    .in_set(RenderSystems::PrepareResources)
                    .in_set(FxRenderSystems::Prepare),
            )
            .add_systems(Render, queue_particles.in_set(RenderSystems::Queue))
            .add_systems(Render, read_lights.in_set(RenderSystems::Cleanup))
            .add_systems(Render, map_done_counts.in_set(RenderSystems::Cleanup))
            .add_systems(
                RenderGraph,
                copy_done_counts.after(simulate).before(camera_driver),
            )
            .add_systems(
                RenderGraph,
                simulate.in_set(FxRenderGraphSystems::Simulate).before(camera_driver),
            )
            .add_systems(RenderGraph, scene::capture_scene_depth.after(camera_driver));
    }
}

pub(crate) fn advance_clock(mut clock: ResMut<FxClock>, time: Res<Time>) {
    let max = fx_gpu::MAX_STEPS_PER_SUBMIT;
    let n = if clock.paused {
        0
    } else {
        match clock.mode {
            FxClockMode::Realtime => {
                clock.accumulator += time.delta_secs();
                let n = ((clock.accumulator / clock.step).floor() as u32).min(max);
                clock.accumulator =
                    (clock.accumulator - n as f32 * clock.step).clamp(0.0, clock.step);
                n
            }
            FxClockMode::Manual => {
                let n = clock.requested.min(max);
                clock.requested -= n;
                n
            }
        }
    };
    clock.steps_this_frame = n;
    clock.total_steps += n as u64;
}

/// Applies the newest particle light lists to pooled `PointLight` children
/// of their effects.
fn apply_particle_lights(
    mut commands: Commands,
    inbox: Res<LightInbox>,
    mut pools: ResMut<LightPools>,
    effects: Query<&GlobalTransform, With<ParticleEffect>>,
    mut lights: Query<(&mut PointLight, &mut Transform, &mut Visibility), With<ParticleLight>>,
) {
    let updates: Vec<LightUpdate> = std::mem::take(&mut *inbox.0.lock().unwrap());
    for u in updates {
        let Ok(gt) = effects.get(u.effect) else {
            continue;
        };
        // World-space lights: into the effect's space (the lights are its children).
        let local_from_world = u.world_space.then(|| gt.affine().inverse());
        let pool = pools.0.entry((u.effect, u.renderer)).or_default();
        let s = &u.settings;
        for (slot, l) in u.lights.iter().enumerate() {
            let add = Vec3::from_array(s.color_add);
            let rgb = (Vec3::new(l.color[0], l.color[1], l.color[2]) + add).max(Vec3::ZERO);
            let peak = rgb.max_element();
            let alpha = if s.alpha_scales_brightness {
                l.color[3].clamp(0.0, 1.0)
            } else {
                1.0
            };
            let light = PointLight {
                color: Color::linear_rgb(rgb.x, rgb.y, rgb.z) .with_alpha(1.0),
                intensity: 0.0,
                range: l.radius.max(0.0),
                radius: 0.0,
                shadow_maps_enabled: false,
                ..default()
            };
            let light = PointLight {
                color: if peak > 0.0 {
                    Color::linear_rgb(rgb.x / peak, rgb.y / peak, rgb.z / peak)
                } else {
                    light.color
                },
                intensity: s.intensity * peak * alpha,
                ..light
            };
            let position = Vec3::from_array(l.position);
            let position = local_from_world.map_or(position, |m| m.transform_point3(position));
            let transform = Transform::from_translation(position);
            match pool.get(slot).copied().map(|e| (e, lights.get_mut(e))) {
                Some((_, Ok((mut pl, mut t, mut v)))) => {
                    *pl = light;
                    *t = transform;
                    *v = Visibility::Inherited;
                }
                Some((e, Err(_))) => {
                    // Not spawned yet (commands from last frame) or despawned.
                    commands.entity(e).try_insert((light, transform, Visibility::Inherited));
                }
                None => {
                    let e = commands
                        .spawn((
                            light,
                            transform,
                            Visibility::Inherited,
                            ParticleLight {
                                renderer: u.renderer,
                                slot,
                            },
                            ChildOf(u.effect),
                        ))
                        .id();
                    pool.push(e);
                }
            }
        }
        for &e in pool.iter().skip(u.lights.len()) {
            if let Ok((_, _, mut v)) = lights.get_mut(e) {
                *v = Visibility::Hidden;
            }
        }
    }
    pools.0.retain(|(effect, _), _| effects.get(*effect).is_ok());
}

// ---------------------------------------------------------------- extraction

/// Assets a renderer draws with, loaded when its effect is first seen.
#[derive(Clone, Default)]
struct RendererAssets {
    texture: Option<Handle<Image>>,
    material: Option<Handle<Shader>>,
    mesh: Option<Handle<Mesh>>,
}

struct ExtractedEffect {
    main: Entity,
    system: Arc<CookedSystem>,
    seed: u32,
    world_from_local: Mat4,
    /// `world_from_local` at the start of this frame's steps.
    previous_world_from_local: Mat4,
    /// Engine.Owner.Velocity in effect space (m/s).
    owner_velocity: Vec3,
    assets: Arc<Vec<RendererAssets>>,
    /// Bound data interfaces' data and poses this frame.
    data: Vec<(String, data::DiValue)>,
    /// A data interface's asset is still loading.
    data_pending: bool,
    /// `User.` values (the first `system.user_params.len()`).
    user: [fx_core::types::Value; fx_gpu::MAX_USER_PARAMS],
    /// [`EffectControl`] generation: a change restarts the instance.
    generation: u32,
    spawning: bool,
    /// Culled by scalability: not simulated, not drawn.
    paused: bool,
    spawn_scale: f32,
    /// A one-shot waiting for its particles to die: read its counts back.
    draining: bool,
}

#[derive(Resource, Default)]
struct ExtractedEffects {
    steps: u32,
    dt: f32,
    /// Seconds between the previous frame's mesh poses and this frame's.
    pose_interval: f32,
    effects: Vec<ExtractedEffect>,
}

/// `(emitter index, renderer)` for every renderer of a system, in order.
fn renderers(sys: &CookedSystem) -> impl Iterator<Item = (usize, &CookedRenderer)> {
    sys.emitters
        .iter()
        .enumerate()
        .flat_map(|(ei, e)| e.renderers.iter().map(move |r| (ei, r)))
}

/// Renderer assets per (effect, system), held in the render world, which
/// keeps them loaded.
#[derive(Resource, Default)]
struct RendererAssetsCache(HashMap<(Entity, usize), Arc<Vec<RendererAssets>>>);

/// Mesh data converted from main-world mesh assets, waiting to be uploaded.
#[derive(Resource, Default)]
struct PendingMeshes {
    /// Converted (or failed) meshes, never converted twice.
    seen: HashMap<AssetId<Mesh>, ()>,
    ready: Vec<(AssetId<Mesh>, mesh_data::MeshData)>,
}

fn renderer_paths(r: &CookedRenderer) -> (Option<&String>, Option<&String>) {
    match r {
        CookedRenderer::Sprite(s) => (s.texture.as_ref(), s.material.as_ref()),
        CookedRenderer::Ribbon(s) => (s.texture.as_ref(), s.material.as_ref()),
        CookedRenderer::Mesh(s) => (s.texture.as_ref(), s.material.as_ref()),
        CookedRenderer::Decal(s) => (s.texture.as_ref(), None),
        CookedRenderer::Light(_) | CookedRenderer::Component(_) | CookedRenderer::Volumetric(_) => {
            (None, None)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_effects(
    mut out: ResMut<ExtractedEffects>,
    mut cache: ResMut<RendererAssetsCache>,
    mut pending: ResMut<PendingMeshes>,
    clock: Extract<Res<FxClock>>,
    effects: Extract<Query<(Entity, &ParticleEffect, &EffectControl, &GlobalTransform)>>,
    asset_server: Extract<Res<AssetServer>>,
    builtins: Extract<Res<BuiltinMeshes>>,
    meshes: Extract<Res<Assets<Mesh>>>,
    data_interfaces: Extract<Res<data::DataInterfaces>>,
) {
    out.steps = clock.steps_this_frame;
    out.dt = clock.step;
    out.pose_interval = data_interfaces.interval;
    out.effects.clear();
    for (entity, effect, control, transform) in &effects {
        let key = assets_key(entity, effect);
        let assets = cache
            .0
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    renderers(&effect.system)
                        .enumerate()
                        .map(|(ri, (_, r))| {
                            let (texture, material) = renderer_paths(r);
                            let mesh = match r {
                                CookedRenderer::Mesh(m) => Some(
                                    match effect.meshes.iter().find(|(i, _)| *i == ri) {
                                        Some((_, h)) => h.clone(),
                                        None => match builtins.0.get(m.mesh.as_str()) {
                                            Some(h) => h.clone(),
                                            None => load_mesh_asset(&asset_server, &m.mesh),
                                        },
                                    },
                                ),
                                _ => None,
                            };
                            RendererAssets {
                                texture: texture.map(|p| asset_server.load::<Image>(p.clone())),
                                material: material.map(|p| asset_server.load::<Shader>(p.clone())),
                                mesh,
                            }
                        })
                        .collect(),
                )
            })
            .clone();
        for a in assets.iter() {
            let Some(h) = &a.mesh else { continue };
            let id = h.id();
            if pending.seen.contains_key(&id) {
                continue;
            }
            if let Some(mesh) = meshes.get(id) {
                pending.seen.insert(id, ());
                match mesh_data::convert(mesh) {
                    Ok(data) => pending.ready.push((id, data)),
                    Err(e) => error!("particle mesh {id:?} cannot be drawn: {e}"),
                }
            }
        }
        let (data, data_pending) = match data_interfaces.frame.get(&entity) {
            Some(d) => (d.values.clone(), d.pending),
            // Not seen by the main world yet (spawned this frame).
            None => (
                Vec::new(),
                effect.system.data_interfaces.iter().any(|d| d.ty.has_asset() && d.asset.is_some()),
            ),
        };
        out.effects.push(ExtractedEffect {
            main: entity,
            system: effect.system.clone(),
            seed: effect.seed,
            world_from_local: transform.to_matrix(),
            previous_world_from_local: control
                .previous_world_from_local()
                .map_or_else(|| transform.to_matrix(), Mat4::from),
            owner_velocity: transform
                .affine()
                .inverse()
                .transform_vector3(control.owner_velocity()),
            assets,
            data,
            data_pending,
            user: {
                let mut u = [fx_core::types::Value::Float(0.0); fx_gpu::MAX_USER_PARAMS];
                let n = effect.user.len().min(fx_gpu::MAX_USER_PARAMS);
                u[..n].copy_from_slice(&effect.user[..n]);
                u
            },
            generation: control.generation(),
            spawning: control.spawning,
            paused: control.paused() || control.is_finished(),
            spawn_scale: control.spawn_scale(),
            draining: control.draining(),
        });
    }
    if cache.0.len() > out.effects.len() {
        let live: Vec<(Entity, usize)> = effects
            .iter()
            .map(|(e, effect, _, _)| assets_key(e, effect))
            .collect();
        cache.0.retain(|k, _| live.contains(k));
    }
}

// ------------------------------------------------------------------ pipeline

#[derive(Resource)]
struct ParticlePipeline {
    mesh_pipeline: MeshPipeline,
    layout: BindGroupLayoutDescriptor,
    sprite: Handle<Shader>,
    ribbon: Handle<Shader>,
    mesh: Handle<Shader>,
    decal: Handle<Shader>,
    volume: Handle<Shader>,
    /// Group 1 of the volumetric renderer: uniform, density texture, sampler.
    volume_layout: BindGroupLayoutDescriptor,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DrawKind {
    Sprite,
    Ribbon,
    Mesh,
    Decal,
    Volume,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Blend {
    Opaque,
    Alpha,
    Additive,
    Premultiplied,
}

impl From<BlendMode> for Blend {
    fn from(b: BlendMode) -> Self {
        match b {
            BlendMode::Alpha => Blend::Alpha,
            BlendMode::Additive => Blend::Additive,
            BlendMode::Premultiplied => Blend::Premultiplied,
        }
    }
}

impl From<MeshBlendMode> for Blend {
    fn from(b: MeshBlendMode) -> Self {
        match b {
            MeshBlendMode::Opaque => Blend::Opaque,
            MeshBlendMode::Alpha => Blend::Alpha,
            MeshBlendMode::Additive => Blend::Additive,
            MeshBlendMode::Premultiplied => Blend::Premultiplied,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ParticleKey {
    view_key: MeshPipelineKey,
    kind: DrawKind,
    blend: Blend,
    lit: bool,
    soft: bool,
    textured: bool,
    material: Option<Handle<Shader>>,
}

impl SpecializedRenderPipeline for ParticlePipeline {
    type Key = ParticleKey;

    fn specialize(&self, key: ParticleKey) -> RenderPipelineDescriptor {
        let mut defs: Vec<ShaderDefVal> = Vec::new();
        if key.view_key.msaa_samples() > 1 {
            defs.push("MULTISAMPLED".into());
        }
        if key.view_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            defs.push("DEPTH_PREPASS".into());
        }
        // The view layout then binds the scene copy (distortion materials).
        if key.view_key.contains(MeshPipelineKey::VIEW_TRANSMISSION_TEXTURE) {
            defs.push("VIEW_TRANSMISSION_TEXTURE".into());
        }
        if key.soft {
            defs.push("SOFT".into());
        }
        if key.lit {
            defs.push("LIT".into());
        }
        if key.textured {
            defs.push("TEXTURED".into());
        }
        let blend = match key.blend {
            Blend::Opaque => {
                defs.push("BLEND_OPAQUE".into());
                None
            }
            Blend::Alpha => {
                defs.push("BLEND_ALPHA".into());
                Some(BlendState::ALPHA_BLENDING)
            }
            Blend::Additive => {
                defs.push("BLEND_ADDITIVE".into());
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                })
            }
            Blend::Premultiplied => {
                defs.push("BLEND_PREMULTIPLIED".into());
                Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING)
            }
        };
        let (label, shader) = match key.kind {
            DrawKind::Sprite => ("fx sprite", &self.sprite),
            DrawKind::Ribbon => ("fx ribbon", &self.ribbon),
            DrawKind::Mesh => ("fx mesh", &self.mesh),
            DrawKind::Decal => ("fx decal", &self.decal),
            DrawKind::Volume => ("fx volume", &self.volume),
        };
        let fragment_shader = key.material.clone().unwrap_or_else(|| shader.clone());
        let view_layout = self.mesh_pipeline.get_view_layout(key.view_key.into());
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                view_layout.main_layout.clone(),
                if key.kind == DrawKind::Volume {
                    self.volume_layout.clone()
                } else {
                    self.layout.clone()
                },
            ],
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: fragment_shader,
                shader_defs: defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.view_key.target_format(),
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                cull_mode: match key.kind {
                    DrawKind::Mesh if key.blend == Blend::Opaque => Some(Face::Back),
                    // Boxes shade their back faces: the camera may be inside.
                    DrawKind::Decal | DrawKind::Volume => Some(Face::Front),
                    _ => None,
                },
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(key.blend == Blend::Opaque),
                // Decals and volumes test the scene depth themselves (prepass).
                depth_compare: Some(match key.kind {
                    DrawKind::Decal | DrawKind::Volume => CompareFunction::Always,
                    _ => CompareFunction::GreaterEqual,
                }),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.view_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            ..default()
        }
    }
}

/// The `FxRenderer` uniform of `particle_io.wesl`.
const UNIFORM_BYTES: usize = 44 * 4;
/// The `Volume` uniform of `volume.wesl`.
const VOLUME_UNIFORM_BYTES: u64 = 176;

/// A volumetric renderer's box in effect space: its `bounds`, else the
/// emitter's fixed bounds, else a cube of the system's bounds radius.
fn volume_bounds(v: &CookedVolumetric, sys: &CookedSystem, emitter: usize) -> [[f32; 3]; 2] {
    v.bounds
        .or_else(|| sys.emitters.get(emitter).and_then(|e| e.fixed_bounds))
        .unwrap_or_else(|| {
            let r = sys.scalability.bounds_radius.max(0.01);
            [[-r; 3], [r; 3]]
        })
}

// ---------------------------------------------------------------- GPU state

/// A culled (and sorted) draw for one view, with its bind groups.
struct ViewState {
    draw: ViewDraw,
    /// By particle buffer parity.
    bind_groups: [BindGroup; 2],
    draw_args: Buffer,
    /// Bind groups use the loaded texture (not the built-in fallback).
    uses_loaded_texture: bool,
}

struct RibbonState {
    view: ViewState,
    build: RibbonBuild,
}

struct LightState {
    view: RetainedViewEntity,
    draw: ViewDraw,
    gather: LightGather,
}

/// A component renderer: every particle by UniqueID, the first rows gathered.
struct ComponentState {
    draw: ViewDraw,
    gather: ParticleGather,
}

/// A volumetric renderer: its living particles splatted into a 3D texture.
struct VolumeState {
    draw: ViewDraw,
    splat: VolumeSplat,
    /// The `Volume` uniform of volume.wesl.
    uniform: wgpu::Buffer,
    bind_group: BindGroup,
}

enum Settings {
    Sprite(CookedSprite),
    Ribbon(CookedRibbon),
    Mesh(CookedMesh),
    Light(CookedLight),
    Decal(CookedDecal),
    Component(CookedComponent),
    Volumetric(CookedVolumetric),
}

/// A renderer of one instance: ribbons and lights (per instance, reading the
/// instance's part of its batch). Sprites and meshes draw per batch
/// ([`BatchRendererDraw`]); their entries here hold only the settings.
struct RendererDraw {
    emitter: usize,
    settings: Settings,
    uniform: wgpu::Buffer,
    assets: RendererAssets,
    ribbon: Option<RibbonState>,
    light: Option<LightState>,
    component: Option<ComponentState>,
    volume: Option<VolumeState>,
    scratch: [u8; UNIFORM_BYTES],
}

/// A sprite or mesh renderer of every instance of a batch.
struct BatchRendererDraw {
    /// Index among the system's renderers.
    renderer: usize,
    emitter: usize,
    settings: Settings,
    uniform: wgpu::Buffer,
    views: HashMap<RetainedViewEntity, BatchViewState>,
    scratch: [u8; UNIFORM_BYTES],
}

/// A batch renderer seen from one view: one cull and sort, one draw.
struct BatchViewState {
    draw: BatchDraw,
    bind_group: BindGroup,
    draw_args: Buffer,
    uses_loaded_texture: bool,
    seen: u64,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: Buffer,
    index_count: u32,
    radius: f32,
}

/// What instances must share to be simulated and drawn together: the
/// program, the renderers' assets, and (for systems with data interfaces,
/// which bind per batch) the instance itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct BatchKey {
    /// (system pointer, ray tracing): the program.
    program: (usize, bool),
    /// The renderer assets (a hash of their asset ids).
    assets: u64,
    /// Systems with data interfaces get a batch per instance.
    exclusive: bool,
}

/// Instances of one [`BatchKey`] simulated and drawn together.
struct Batch {
    key: BatchKey,
    gpu: GpuBatch,
    /// `world_from_local` of every slot (`array<mat4x4<f32>>`).
    transforms: wgpu::Buffer,
    transform_bytes: Vec<u8>,
    assets: Arc<Vec<RendererAssets>>,
    draws: Vec<BatchRendererDraw>,
    /// Instances simulated and drawn this frame (not paused).
    members: Vec<Entity>,
    /// Per emitter: the particle copy (2 * slot + current) of each member, after the steps.
    copies: Vec<Vec<u32>>,
    /// Slots held by instances (live or pooled).
    used: u32,
    /// [`FxSceneDepth::generation`] the scene bind group was made for.
    depth_generation: u64,
    /// Mean member position, for sorting against other transparents.
    center: Vec3,
}

struct EffectInstance {
    system: Arc<CookedSystem>,
    seed: u32,
    gpu: GpuEffect,
    /// Index of its batch in [`FxState::batches`].
    batch: usize,
    key: BatchKey,
    draws: Vec<RendererDraw>,
    world_from_local: Mat4,
    /// Owner poses at the start and end of this frame's steps (world-space emitters).
    owner_poses: (fx_core::sim::OwnerPose, fx_core::sim::OwnerPose),
    /// Data interface name → the data uploaded to its batch (held, so its address stays unique).
    uploaded: HashMap<String, Arc<dyn std::any::Any + Send + Sync>>,
    seen: u64,
    /// [`EffectControl`] generation it last ran.
    generation: u32,
    /// Culled this frame: no steps, no draws.
    paused: bool,
    /// A one-shot waiting for its particles to die.
    draining: bool,
    /// Spawning was off in every step since `draining` began (counts read now mean the end).
    drain_steps: u32,
}

/// Render world: the GPU state of every effect. Read it (after
/// [`FxRenderGraphSystems::Simulate`]) to feed particles to other passes.
#[derive(Resource)]
pub struct FxState {
    view_pipelines: ViewPipelines,
    ribbon_pipelines: RibbonPipelines,
    light_pipelines: LightPipelines,
    gather_pipelines: GatherPipelines,
    volume_pipelines: VolumePipelines,
    effects: HashMap<Entity, EffectInstance>,
    /// Batches by index (`None`: dropped).
    batches: Vec<Option<Batch>>,
    builtin: textures::Builtin,
    meshes: HashMap<AssetId<Mesh>, GpuMesh>,
    /// Bound where a renderer has no buffer of its own.
    dummy: wgpu::Buffer,
    frame: u64,
    /// Compiled systems by (system pointer, ray tracing).
    programs: HashMap<(usize, bool), Arc<fx_gpu::GpuProgram>>,
    /// Idle instances by batch key, ready for reuse (they keep their slots).
    pool: HashMap<BatchKey, Vec<EffectInstance>>,
    /// Scratch list of entities gone this frame.
    gone: Vec<Entity>,
    /// Pool counters changed this frame.
    pool_dirty: bool,
    /// Last frame each program key was used.
    used: HashMap<(usize, bool), u64>,
    stats_programs: u64,
    stats_created: u64,
    stats_reused: u64,
}

fn init_render(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    asset_server: Res<AssetServer>,
) {
    let stages = ShaderStages::VERTEX_FRAGMENT;
    // Fragment shaders only see interpolated particle values, so the storage
    // buffers are vertex-only: they do not count against the fragment stage's
    // storage buffer limit, which Bevy's view bindings mostly use (WebGPU
    // guarantees 8 per stage).
    let vertex = ShaderStages::VERTEX;
    let layout = BindGroupLayoutDescriptor::new(
        "fx particles",
        &BindGroupLayoutEntries::sequential(
            stages,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
                storage_buffer_read_only_sized(false, None).visibility(vertex),
            ),
        ),
    );
    let volume_layout = BindGroupLayoutDescriptor::new(
        "fx volume",
        &BindGroupLayoutEntries::sequential(
            stages,
            (
                uniform_buffer_sized(false, None),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    commands.insert_resource(ParticlePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        layout,
        sprite: load_embedded_asset!(asset_server.as_ref(), "sprite.wesl"),
        ribbon: load_embedded_asset!(asset_server.as_ref(), "ribbon.wesl"),
        mesh: load_embedded_asset!(asset_server.as_ref(), "mesh.wesl"),
        decal: load_embedded_asset!(asset_server.as_ref(), "decal.wesl"),
        volume: load_embedded_asset!(asset_server.as_ref(), "volume.wesl"),
        volume_layout,
    });
    let wd = device.wgpu_device();
    commands.insert_resource(FxState {
        view_pipelines: ViewPipelines::new(wd),
        ribbon_pipelines: RibbonPipelines::new(wd),
        light_pipelines: LightPipelines::new(wd),
        gather_pipelines: GatherPipelines::new(wd),
        volume_pipelines: VolumePipelines::new(wd),
        effects: HashMap::default(),
        batches: Vec::new(),
        builtin: textures::Builtin::new(wd),
        meshes: HashMap::default(),
        dummy: wd.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx dummy storage"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        frame: 0,
        programs: HashMap::default(),
        pool: HashMap::default(),
        gone: Vec::new(),
        pool_dirty: false,
        used: HashMap::default(),
        stats_programs: 0,
        stats_created: 0,
        stats_reused: 0,
    });
}

fn view_sort_mode(s: SortMode, deterministic: bool) -> fx_gpu::view::SortMode {
    match s {
        SortMode::None if deterministic => fx_gpu::view::SortMode::UniqueId,
        SortMode::None => fx_gpu::view::SortMode::None,
        SortMode::ViewDepth => fx_gpu::view::SortMode::ViewDepth,
        SortMode::ViewDistance => fx_gpu::view::SortMode::ViewDistance,
    }
}

/// Ribbon geometry does not depend on the view: one sort, no culling.
fn ribbon_config(r: &CookedRibbon) -> ViewDrawConfig {
    ViewDrawConfig {
        sort_mode: match r.link_order {
            RibbonLinkOrderMode::SpawnOrder => fx_gpu::view::SortMode::RibbonSpawnOrder,
            RibbonLinkOrderMode::Custom => fx_gpu::view::SortMode::RibbonLinkOrder,
        },
        vertex_count: 6 * r.tessellation_factor.clamp(1, 16),
        cull: CullRadius::Off,
    }
}

/// Group 1 bind groups (one per particle buffer parity) for a renderer.
#[allow(clippy::too_many_arguments)]
fn particle_bind_groups(
    wd: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    em: &fx_gpu::GpuEmitter,
    indices: &wgpu::Buffer,
    texture: (&wgpu::TextureView, &wgpu::Sampler),
    extra: [&wgpu::Buffer; 3],
    transforms: &wgpu::Buffer,
) -> [BindGroup; 2] {
    let bg = |s: usize| particle_bind_group(wd, layout, uniform, &em.particles[s], indices, texture, extra, transforms);
    [bg(0), bg(1)]
}

/// A group 1 bind group reading `particles`.
#[allow(clippy::too_many_arguments)]
fn particle_bind_group(
    wd: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    particles: &fx_gpu::Region,
    indices: &wgpu::Buffer,
    texture: (&wgpu::TextureView, &wgpu::Sampler),
    extra: [&wgpu::Buffer; 3],
    transforms: &wgpu::Buffer,
) -> BindGroup {
    {
        BindGroup::from(wd.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx particles"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(texture.0),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(texture.1),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: extra[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: extra[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: extra[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: transforms.as_entire_binding(),
                },
            ],
        }))
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_effects(
    mut state: ResMut<FxState>,
    extracted: Res<ExtractedEffects>,
    mut pending: ResMut<PendingMeshes>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<ParticlePipeline>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    views: Query<&ExtractedView>,
    deterministic: Res<FxDeterministic>,
    rt: Res<FxRayTracingScene>,
    pool_stats: Res<FxPoolStats>,
) {
    let state = &mut *state;
    state.frame += 1;
    let frame = state.frame;
    let wd = device.wgpu_device();
    let layout = cache.get_bind_group_layout(&pipeline.layout);
    let volume_layout = cache.get_bind_group_layout(&pipeline.volume_layout);

    for (id, data) in pending.ready.drain(..) {
        use wgpu::util::DeviceExt;
        let vertices = wd.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx particle mesh vertices"),
            contents: bytemuck_bytes(&data.vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let indices = wd.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx particle mesh indices"),
            contents: bytemuck_bytes(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        state.meshes.insert(
            id,
            GpuMesh {
                vertices,
                indices: Buffer::from(indices),
                index_count: data.indices.len() as u32,
                radius: data.radius,
            },
        );
    }

    for b in state.batches.iter_mut().flatten() {
        b.members.clear();
    }
    let ray_tracing = rt.layout.is_some();
    for ex in &extracted.effects {
        let program_key = (Arc::as_ptr(&ex.system) as usize, ray_tracing);
        let key = BatchKey {
            program: program_key,
            assets: assets_hash(&ex.assets),
            exclusive: !ex.system.data_interfaces.is_empty(),
        };
        state.used.insert(program_key, frame);
        // Another system (hot reload), scene layout or renderer assets: back to the pool, take a fitting one.
        if state.effects.get(&ex.main).is_some_and(|inst| inst.key != key) {
            let inst = state.effects.remove(&ex.main).unwrap();
            state.release(inst);
        }
        if !state.effects.contains_key(&ex.main) {
            let reused = state.pool.get_mut(&key).and_then(|idle| idle.pop());
            let inst = match reused {
                Some(mut inst) => {
                    inst.gpu.reset(ex.seed);
                    inst.seed = ex.seed;
                    inst.generation = ex.generation;
                    inst.draining = false;
                    state.stats_reused += 1;
                    inst
                }
                None => {
                    let program = match state.programs.get(&program_key) {
                        Some(p) => p.clone(),
                        None => {
                            let options = fx_gpu::GpuOptions {
                                ray_tracing_layout: rt.layout.as_ref(),
                            };
                            match fx_gpu::GpuProgram::new(wd, ex.system.clone(), options) {
                                Ok(p) => {
                                    let p = Arc::new(p);
                                    state.programs.insert(program_key, p.clone());
                                    state.stats_programs += 1;
                                    p
                                }
                                Err(e) => {
                                    error!(
                                        "effect \"{}\" cannot run on the GPU: {e}",
                                        ex.system.name
                                    );
                                    continue;
                                }
                            }
                        }
                    };
                    state.stats_created += 1;
                    let batch = state.batch_with_room(wd, &program, key, ex);
                    let b = state.batches[batch].as_mut().unwrap();
                    let gpu = b.gpu.alloc(ex.seed).expect("batch_with_room has a free slot");
                    b.used += 1;
                    new_instance(wd, gpu, batch, key, ex, frame)
                }
            };
            state.effects.insert(ex.main, inst);
            state.pool_dirty = true;
        }
        let inst = state.effects.get_mut(&ex.main).unwrap();
        if inst.seed != ex.seed || inst.generation != ex.generation {
            inst.gpu.reset(ex.seed);
            inst.seed = ex.seed;
            inst.generation = ex.generation;
            inst.draining = false;
        }
        inst.seen = frame;
        inst.world_from_local = ex.world_from_local;
        inst.owner_poses = (
            fx_core::sim::OwnerPose::from_mat4(ex.previous_world_from_local),
            fx_core::sim::OwnerPose::from_mat4(ex.world_from_local),
        );
        // Where a warm-up (before this frame's steps) runs.
        inst.gpu.owner_pose = inst.owner_poses.0;
        inst.gpu.previous_owner_pose = inst.owner_poses.0;
        inst.gpu.owner_velocity = ex.owner_velocity.to_array();
        inst.paused = ex.paused;
        inst.gpu.active = ex.spawning;
        inst.gpu.spawn_scale = ex.spawn_scale;
        let n = inst.gpu.user.len().min(ex.user.len());
        inst.gpu.user[..n].copy_from_slice(&ex.user[..n]);
        if !ex.draining {
            inst.draining = false;
        } else if !inst.draining {
            inst.draining = true;
            inst.drain_steps = 0;
        }
        let batch = state.batches[inst.batch].as_mut().unwrap();
        if !inst.paused {
            batch.members.push(ex.main);
        }
        let at = inst.gpu.slot as usize * 64;
        for (k, x) in inst.world_from_local.to_cols_array().iter().enumerate() {
            batch.transform_bytes[at + k * 4..at + k * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }

        // Data interfaces (bound per batch; such systems have a batch per instance):
        // upload new data, move meshes to this frame's pose.
        for (name, value) in &ex.data {
            match value {
                data::DiValue::Mesh { data, pose } => {
                    let ptr = Arc::as_ptr(data);
                    if !inst.uploaded.get(name).is_some_and(|u| std::ptr::addr_eq(Arc::as_ptr(u), ptr)) {
                        let still = fx_core::data::MeshPose::still(pose.clone());
                        batch.gpu.set_mesh(wd, name, data, still);
                        inst.uploaded.insert(name.clone(), data.clone());
                    } else {
                        batch
                            .gpu
                            .set_mesh_pose(&queue, name, pose, extracted.pose_interval);
                    }
                }
                data::DiValue::Texture(t) => {
                    let ptr = Arc::as_ptr(t);
                    if !inst.uploaded.get(name).is_some_and(|u| std::ptr::addr_eq(Arc::as_ptr(u), ptr)) {
                        batch.gpu.set_texture(wd, name, t);
                        inst.uploaded.insert(name.clone(), t.clone());
                    }
                }
            }
        }

        // Ribbons and lights: per instance.
        for rd in &mut inst.draws {
            if !matches!(
                rd.settings,
                Settings::Ribbon(_)
                    | Settings::Light(_)
                    | Settings::Component(_)
                    | Settings::Volumetric(_)
            ) {
                continue;
            }
            let em = &inst.gpu.emitters[rd.emitter];
            let sub_index = inst.gpu.system.emitters[rd.emitter].sets_sub_image_index();
            let world_space = !inst.gpu.system.emitters[rd.emitter].local_space;
            write_uniform(&rd.settings, &mut rd.scratch, &inst.world_from_local, em.capacity, fx_gpu::extended_level(em.words), 1.0, sub_index, None, world_space);
            queue.write_buffer(&rd.uniform, 0, &rd.scratch);
            let loaded = rd.assets.texture.as_ref().and_then(|h| images.get(h));
            match &rd.settings {
                Settings::Ribbon(r) => {
                    let stale = rd
                        .ribbon
                        .as_ref()
                        .is_some_and(|v| loaded.is_some() && !v.view.uses_loaded_texture);
                    if rd.ribbon.is_none() || stale {
                        let draw =
                            ViewDraw::with_config(wd, &state.view_pipelines, em, ribbon_config(r));
                        let build = RibbonBuild::new(wd, &state.ribbon_pipelines, em, &draw);
                        let (tv, ts) = renderer_texture(wd, &queue, &mut state.builtin, loaded, &rd.settings);
                        let bind_groups = particle_bind_groups(
                            wd,
                            &layout,
                            &rd.uniform,
                            em,
                            &draw.indices,
                            (&tv, &ts),
                            [&build.dist, &build.ribbons, &build.meta],
                            &state.dummy,
                        );
                        rd.ribbon = Some(RibbonState {
                            view: ViewState {
                                bind_groups,
                                draw_args: Buffer::from(draw.draw_args.clone()),
                                draw,
                                uses_loaded_texture: loaded.is_some(),
                            },
                            build,
                        });
                    }
                }
                Settings::Light(l) => {
                    // Lights are culled against one view (kept while it exists), nearest first.
                    let first = views.iter().map(|v| v.retained_view_entity).next();
                    let keep = rd.light.as_ref().is_some_and(|ls| {
                        views.iter().any(|v| v.retained_view_entity == ls.view)
                    });
                    if !keep {
                        rd.light = first.map(|view| {
                            let draw = ViewDraw::with_config(
                                wd,
                                &state.view_pipelines,
                                em,
                                ViewDrawConfig {
                                    sort_mode: fx_gpu::view::SortMode::ViewDistance,
                                    vertex_count: 0,
                                    cull: CullRadius::Word {
                                        word: 39,
                                        scale: l.radius_scale,
                                    },
                                },
                            );
                            let gather = LightGather::new(
                                wd,
                                &state.light_pipelines,
                                em,
                                &draw,
                                l.max_lights,
                                l.radius_scale,
                            );
                            LightState { view, draw, gather }
                        });
                    }
                }
                Settings::Component(c) => {
                    if rd.component.is_none() {
                        // Oldest first (by UniqueID), nothing culled.
                        let draw = ViewDraw::with_config(
                            wd,
                            &state.view_pipelines,
                            em,
                            ViewDrawConfig {
                                sort_mode: fx_gpu::view::SortMode::UniqueId,
                                vertex_count: 0,
                                cull: CullRadius::Off,
                            },
                        );
                        let gather = ParticleGather::new(
                            wd,
                            &state.gather_pipelines,
                            em,
                            &draw,
                            c.max_components,
                            &component::WORDS,
                        );
                        rd.component = Some(ComponentState { draw, gather });
                    }
                }
                Settings::Volumetric(v) => {
                    if rd.volume.is_none() {
                        let draw = ViewDraw::with_config(
                            wd,
                            &state.view_pipelines,
                            em,
                            ViewDrawConfig {
                                sort_mode: fx_gpu::view::SortMode::None,
                                vertex_count: 0,
                                cull: CullRadius::Off,
                            },
                        );
                        let splat = VolumeSplat::new(wd, &state.volume_pipelines, em, &draw, v.resolution);
                        let uniform = wd.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("fx volume uniform"),
                            size: VOLUME_UNIFORM_BYTES,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let sampler = wd.create_sampler(&wgpu::SamplerDescriptor {
                            label: Some("fx volume"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            ..Default::default()
                        });
                        let bind_group = BindGroup::from(wd.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("fx volume"),
                            layout: &volume_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: uniform.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(&splat.view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&sampler),
                                },
                            ],
                        }));
                        rd.volume = Some(VolumeState {
                            draw,
                            splat,
                            uniform,
                            bind_group,
                        });
                    }
                    let vs = rd.volume.as_ref().unwrap();
                    let bounds = volume_bounds(v, &inst.gpu.system, rd.emitter);
                    vs.splat.set_params(
                        &queue,
                        &VolumeParams {
                            bounds,
                            density_scale: v.density_scale,
                            radius_scale: v.radius_scale,
                        },
                    );
                    let mut u: Vec<u32> = Vec::with_capacity(VOLUME_UNIFORM_BYTES as usize / 4);
                    u.extend(inst.world_from_local.to_cols_array().map(f32::to_bits));
                    u.extend(inst.world_from_local.inverse().to_cols_array().map(f32::to_bits));
                    u.extend(bounds[0].map(f32::to_bits));
                    u.push(v.steps);
                    u.extend(bounds[1].map(f32::to_bits));
                    u.push(v.lit as u32);
                    u.extend([v.emissive.to_bits(), 0, 0, 0]);
                    queue.write_buffer(&vs.uniform, 0, bytemuck_bytes(&u));
                }
                _ => {}
            }
        }
    }
    // Instances whose entity is gone go back to the pool.
    let mut gone = std::mem::take(&mut state.gone);
    gone.clear();
    gone.extend(state.effects.iter().filter(|(_, e)| e.seen != frame).map(|(k, _)| *k));
    for e in gone.drain(..) {
        let inst = state.effects.remove(&e).unwrap();
        state.release(inst);
    }
    state.gone = gone;

    // Per batch: transforms, the sprite and mesh renderers' uniforms and per-view culls.
    for batch in state.batches.iter_mut().flatten() {
        if batch.members.is_empty() {
            continue;
        }
        queue.write_buffer(&batch.transforms, 0, &batch.transform_bytes);
        let mut center = Vec3::ZERO;
        for m in &batch.members {
            if let Some(inst) = state.effects.get(m) {
                center += inst.world_from_local.w_axis.truncate();
            }
        }
        batch.center = center / batch.members.len() as f32;
        let sys = batch.gpu.program.system.clone();
        for rd in &mut batch.draws {
            let a = &batch.assets[rd.renderer];
            let mesh = a.mesh.as_ref().and_then(|h| state.meshes.get(&h.id()));
            let program = &batch.gpu.program;
            let sub_index = sys.emitters[rd.emitter].sets_sub_image_index();
            write_uniform(
                &rd.settings,
                &mut rd.scratch,
                &Mat4::IDENTITY,
                program.emitter_capacity(rd.emitter),
                fx_gpu::extended_level(fx_gpu::particle_words(&program.system.emitters[rd.emitter])),
                mesh.map_or(1.0, |m| m.radius),
                sub_index,
                Some(batch.gpu.region_words(rd.emitter)),
                !sys.emitters[rd.emitter].local_space,
            );
            queue.write_buffer(&rd.uniform, 0, &rd.scratch);
            let loaded = a.texture.as_ref().and_then(|h| images.get(h));
            let (config, extra0) = match &rd.settings {
                Settings::Sprite(s) => (
                    ViewDrawConfig::sprites(view_sort_mode(s.sort_mode, deterministic.0)),
                    &state.dummy,
                ),
                Settings::Mesh(m) => {
                    let Some(mesh) = mesh else {
                        // Not loaded yet.
                        continue;
                    };
                    (
                        ViewDrawConfig {
                            sort_mode: view_sort_mode(m.sort_mode, deterministic.0),
                            vertex_count: mesh.index_count,
                            cull: CullRadius::MeshScale { scale: mesh.radius },
                        },
                        &mesh.vertices,
                    )
                }
                Settings::Decal(d) => (
                    ViewDrawConfig {
                        sort_mode: view_sort_mode(SortMode::None, deterministic.0),
                        vertex_count: 36,
                        // Half the box diagonal, times Particles.Scale.
                        cull: CullRadius::MeshScale {
                            scale: 0.5 * Vec3::from_array(d.size).length(),
                        },
                    },
                    &state.dummy,
                ),
                _ => continue,
            };
            for view in &views {
                let entry = rd.views.get(&view.retained_view_entity);
                let stale = entry.is_some_and(|v| loaded.is_some() && !v.uses_loaded_texture);
                if entry.is_none() || stale {
                    let mut draw = BatchDraw::new(
                        wd,
                        &state.view_pipelines,
                        &batch.gpu,
                        rd.emitter,
                        &batch.transforms,
                        config,
                    );
                    draw.world_space = !sys.emitters[rd.emitter].local_space;
                    let (tv, ts) = renderer_texture(wd, &queue, &mut state.builtin, loaded, &rd.settings);
                    let particles = batch.gpu.particles(rd.emitter);
                    // Sorted sprites read the cull's payload at binding 5.
                    let extra0 = draw.payload.as_ref().unwrap_or(extra0);
                    let bind_group = particle_bind_group(
                        wd,
                        &layout,
                        &rd.uniform,
                        &fx_gpu::Region {
                            buffer: particles.clone(),
                            offset: 0,
                            size: particles.size(),
                        },
                        &draw.indices,
                        (&tv, &ts),
                        [extra0, &state.dummy, &state.dummy],
                        &batch.transforms,
                    );
                    rd.views.insert(
                        view.retained_view_entity,
                        BatchViewState {
                            bind_group,
                            draw_args: Buffer::from(draw.draw_args.clone()),
                            draw,
                            uses_loaded_texture: loaded.is_some(),
                            seen: frame,
                        },
                    );
                }
                rd.views.get_mut(&view.retained_view_entity).unwrap().seen = frame;
            }
            rd.views.retain(|_, v| v.seen == frame);
        }
    }

    // Systems unused for a while (hot-reloaded away): drop their pools, batches and programs.
    if frame % 60 == 0 {
        let stale = |last: &u64| frame.saturating_sub(*last) > POOL_UNUSED_FRAMES;
        if state.used.values().any(stale) {
            let used = std::mem::take(&mut state.used);
            for (key, last) in &used {
                if stale(last) {
                    state.pool.retain(|k, _| k.program != *key);
                    for b in &mut state.batches {
                        if b.as_ref().is_some_and(|b| b.key.program == *key) {
                            *b = None;
                        }
                    }
                    state.programs.remove(key);
                    state.pool_dirty = true;
                }
            }
            state.used = used.into_iter().filter(|(_, l)| !stale(l)).collect();
        }
    }
    if state.pool_dirty {
        state.pool_dirty = false;
        let mut s = pool_stats.0.lock().unwrap();
        s.programs = state.stats_programs;
        s.instances_created = state.stats_created;
        s.instances_reused = state.stats_reused;
        s.live = state.effects.len() as u64;
        s.idle = state.pool.values().map(|v| v.len() as u64).sum();
        s.gpu_bytes = state
            .batches
            .iter()
            .flatten()
            .map(|b| b.gpu.gpu_bytes() + b.transforms.size())
            .sum();
    }
}

/// A renderer's texture: the loaded one, or the built-in sprite for its flipbook grid.
fn renderer_texture(
    wd: &wgpu::Device,
    queue: &RenderQueue,
    builtin: &mut textures::Builtin,
    loaded: Option<&GpuImage>,
    settings: &Settings,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let [cols, rows] = match settings {
        Settings::Sprite(s) => s.sub_image_size,
        _ => [1, 1],
    };
    match loaded {
        Some(img) => ((*img.texture_view).clone(), (*img.sampler).clone()),
        None => {
            let (v, s) = builtin.get(wd, queue, cols, rows);
            (v.clone(), s.clone())
        }
    }
}

/// First batch size of a system; each further batch doubles it.
const FIRST_BATCH_SLOTS: u32 = 16;
/// GPU bytes a system's first batch may take (fewer slots for big effects).
const FIRST_BATCH_BYTES: u64 = 16 << 20;

impl FxState {
    /// A batch of `key` with a free slot (made if none has one).
    fn batch_with_room(
        &mut self,
        wd: &wgpu::Device,
        program: &Arc<fx_gpu::GpuProgram>,
        key: BatchKey,
        ex: &ExtractedEffect,
    ) -> usize {
        if !key.exclusive
            && let Some(i) = self
                .batches
                .iter()
                .position(|b| b.as_ref().is_some_and(|b| b.key == key && b.gpu.free_slots() > 0))
        {
            return i;
        }
        let slots = if key.exclusive {
            1
        } else {
            let largest = self
                .batches
                .iter()
                .flatten()
                .filter(|b| b.key == key)
                .map(|b| b.gpu.slots)
                .max();
            largest.map_or_else(
                || (FIRST_BATCH_BYTES / program.instance_bytes().max(1)).clamp(1, FIRST_BATCH_SLOTS as u64) as u32,
                |s| s * 2,
            )
        };
        let gpu = GpuBatch::new(wd, program, slots);
        let slots = gpu.slots;
        let draws = renderers(&ex.system)
            .enumerate()
            .filter_map(|(i, (emitter, r))| {
                let settings = match r {
                    CookedRenderer::Sprite(s) => Settings::Sprite(s.clone()),
                    CookedRenderer::Mesh(s) => Settings::Mesh(s.clone()),
                    CookedRenderer::Decal(s) => Settings::Decal(s.clone()),
                    _ => return None,
                };
                Some(BatchRendererDraw {
                    renderer: i,
                    emitter,
                    settings,
                    uniform: wd.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fx batch renderer uniform"),
                        size: UNIFORM_BYTES as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    views: HashMap::default(),
                    scratch: [0; UNIFORM_BYTES],
                })
            })
            .collect();
        let batch = Batch {
            key,
            transforms: wd.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx batch transforms"),
                size: slots as u64 * 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            transform_bytes: vec![0; slots as usize * 64],
            assets: ex.assets.clone(),
            draws,
            members: Vec::new(),
            copies: vec![Vec::new(); ex.system.emitters.len()],
            used: 0,
            depth_generation: u64::MAX,
            center: Vec3::ZERO,
            gpu,
        };
        match self.batches.iter().position(Option::is_none) {
            Some(i) => {
                self.batches[i] = Some(batch);
                i
            }
            None => {
                self.batches.push(Some(batch));
                self.batches.len() - 1
            }
        }
    }
}

fn bytemuck_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: f32 and u32 slices have no padding and every byte is initialised.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// The `FxRenderer` uniform of particle_io.wesl; `batched`: the words
/// between particle copies of a batch draw (indices are batch references).
/// `world_space`: the emitter simulates in world space; drawn without the
/// effect transform (identity here and for every batch slot).
#[allow(clippy::too_many_arguments)]
fn write_uniform(
    settings: &Settings,
    b: &mut [u8; UNIFORM_BYTES],
    world_from_local: &Mat4,
    capacity: u32,
    extended: u32,
    mesh_radius: f32,
    sub_image_from_index: bool,
    batched: Option<u32>,
    world_space: bool,
) {
    let mut put = |w: usize, v: u32| b[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    let m = if world_space { Mat4::IDENTITY } else { *world_from_local };
    for (i, x) in m.to_cols_array().iter().enumerate() {
        put(i, x.to_bits());
    }
    put(42, world_space as u32);
    put(26, 1.0f32.to_bits());
    put(27, 1.0f32.to_bits());
    put(28, capacity);
    put(33, extended);
    put(38, mesh_radius.to_bits());
    put(40, batched.is_some() as u32);
    put(41, batched.unwrap_or(0));
    match settings {
        Settings::Sprite(s) => {
            for c in 0..3 {
                put(16 + c, s.custom_facing_vector[c].to_bits());
                put(20 + c, s.custom_alignment_vector[c].to_bits());
            }
            put(24, s.pivot_in_uv_space[0].to_bits());
            put(25, s.pivot_in_uv_space[1].to_bits());
            put(26, (s.sub_image_size[0].max(1) as f32).to_bits());
            put(27, (s.sub_image_size[1].max(1) as f32).to_bits());
            put(39, sub_image_from_index as u32);
            put(
                29,
                match s.facing_mode {
                    FacingMode::FaceCamera => 0,
                    FacingMode::FaceCameraPlane => 1,
                    FacingMode::FaceCameraPosition => 2,
                    FacingMode::CustomFacingVector => 3,
                },
            );
            put(
                30,
                match s.alignment {
                    SpriteAlignment::Unaligned => 0,
                    SpriteAlignment::VelocityAligned => 1,
                    SpriteAlignment::CustomAlignment => 2,
                },
            );
            put(31, s.sub_uv_blending as u32);
            put(32, s.soft_particle_fade_distance.max(1e-4).to_bits());
        }
        Settings::Ribbon(r) => {
            for c in 0..3 {
                put(16 + c, r.custom_facing_vector[c].to_bits());
            }
            put(
                29,
                match r.facing_mode {
                    RibbonFacingMode::Screen => 0,
                    RibbonFacingMode::CustomFacingVector => 1,
                },
            );
            put(
                34,
                match r.uv_mode {
                    RibbonUvMode::ScaledUniformly => 0,
                    RibbonUvMode::ScaledUsingRibbonSegmentLength => 1,
                    RibbonUvMode::TiledOverRibbonLength => 2,
                },
            );
            put(35, r.tessellation_factor.clamp(1, 16));
            put(36, r.tiling_length.max(1e-4).to_bits());
            put(37, (r.link_order == RibbonLinkOrderMode::SpawnOrder) as u32);
        }
        Settings::Mesh(m) => {
            put(
                29,
                match m.facing_mode {
                    MeshFacingMode::Default => 0,
                    MeshFacingMode::Velocity => 1,
                    MeshFacingMode::CameraPosition => 2,
                    MeshFacingMode::CameraPlane => 3,
                },
            );
        }
        Settings::Decal(d) => {
            for c in 0..3 {
                put(16 + c, d.size[c].to_bits());
            }
            put(32, d.fade_start.to_bits());
        }
        Settings::Light(_) | Settings::Component(_) | Settings::Volumetric(_) => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_particles(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    transmissive_draw_functions: Res<DrawFunctions<Transmissive3d>>,
    mut transmissive_phases: ResMut<ViewSortedRenderPhases<Transmissive3d>>,
    pipeline: Res<ParticlePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ParticlePipeline>>,
    cache: Res<PipelineCache>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, Option<&ScreenSpaceTransmission>)>,
    view_keys: Res<ViewKeyCache>,
    state: Res<FxState>,
    extracted: Res<ExtractedEffects>,
    ready: Res<FxReady>,
) {
    let draw = draw_functions.read().id::<DrawParticlesCommands>();
    let transmissive_draw = transmissive_draw_functions.read().id::<DrawParticlesCommands>();
    let mut all_ready = extracted
        .effects
        .iter()
        .all(|e| state.effects.contains_key(&e.main) && !e.data_pending);
    for (view, transmission) in &views {
        // Distortion sprites draw in the transmissive pass, which only runs
        // (and copies the scene for them) on cameras with
        // ScreenSpaceTransmission; elsewhere they are left out.
        let can_distort = transmission.is_some_and(|t| t.steps > 0);
        let mut transmissive = transmissive_phases.get_mut(&view.retained_view_entity);
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(&view_key) = view_keys.get(&view.retained_view_entity) else {
            all_ready = false;
            continue;
        };
        // Sprites and meshes: one item per batch renderer; ribbons: one per instance.
        let batch_items = state.batches.iter().enumerate().filter_map(|(b, batch)| {
            let batch = batch.as_ref().filter(|b| !b.members.is_empty())?;
            Some(batch.draws.iter().map(move |rd| {
                let shown = rd.views.contains_key(&view.retained_view_entity);
                (batch_entity(b), rd.renderer, batch.center, &rd.settings, &batch.assets[rd.renderer], shown)
            }))
        });
        let ribbon_items = state.effects.iter().filter(|(_, i)| !i.paused).map(|(main, inst)| {
            inst.draws.iter().enumerate().filter(|(_, rd)| matches!(rd.settings, Settings::Ribbon(_) | Settings::Volumetric(_))).map(|(i, rd)| {
                (*main, i, inst.world_from_local.w_axis.truncate(), &rd.settings, &rd.assets, true)
            })
        });
        let items: Vec<_> = batch_items.flatten().chain(ribbon_items.flatten()).collect();
        {
            for (main, i, center, settings, assets, shown) in items {
                let material = assets.material.clone();
                let textured = assets.texture.is_some();
                let key = match settings {
                    Settings::Sprite(s) => ParticleKey {
                        view_key,
                        kind: DrawKind::Sprite,
                        blend: s.blend_mode.into(),
                        lit: s.lit,
                        soft: s.soft_particle_fade_distance > 0.0
                            && view_key.contains(MeshPipelineKey::DEPTH_PREPASS),
                        textured: true,
                        material,
                    },
                    Settings::Ribbon(r) => ParticleKey {
                        view_key,
                        kind: DrawKind::Ribbon,
                        blend: r.blend_mode.into(),
                        lit: false,
                        soft: false,
                        textured,
                        material,
                    },
                    Settings::Mesh(m) => {
                        if !shown {
                            // Mesh not loaded yet.
                            all_ready = false;
                            continue;
                        }
                        ParticleKey {
                            view_key,
                            kind: DrawKind::Mesh,
                            blend: m.blend_mode.into(),
                            lit: m.lit,
                            soft: false,
                            textured,
                            material,
                        }
                    }
                    Settings::Decal(d) => {
                        // Decals project onto the depth prepass: none without one.
                        if !view_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
                            continue;
                        }
                        ParticleKey {
                            view_key,
                            kind: DrawKind::Decal,
                            blend: d.blend_mode.into(),
                            lit: d.lit,
                            soft: false,
                            textured: true,
                            material: None,
                        }
                    }
                    Settings::Volumetric(v) => ParticleKey {
                        view_key,
                        kind: DrawKind::Volume,
                        blend: Blend::Premultiplied,
                        lit: v.lit,
                        soft: false,
                        textured: false,
                        material: None,
                    },
                    Settings::Light(_) | Settings::Component(_) => continue,
                };
                let distortion = matches!(settings, Settings::Sprite(s) if s.distortion);
                if distortion && !(can_distort && transmissive.is_some()) {
                    continue;
                }
                let id = pipelines.specialize(&cache, &pipeline, key);
                if cache.get_render_pipeline(id).is_none() {
                    all_ready = false;
                }
                if distortion {
                    if let Some(t) = transmissive.as_deref_mut() {
                        t.add_transient(Transmissive3d {
                            sorting_info: TransparentSortingInfo3d::Sorted {
                                mesh_center: center,
                                depth_bias: 0.0,
                            },
                            distance: 0.0,
                            pipeline: id,
                            entity: (renderer_entity(i), MainEntity::from(main)),
                            draw_function: transmissive_draw,
                            batch_range: i as u32..i as u32 + 1,
                            extra_index: PhaseItemExtraIndex::None,
                            indexed: false,
                        });
                    }
                    continue;
                }
                phase.add_transient(Transparent3d {
                    sorting_info: TransparentSortingInfo3d::Sorted {
                        mesh_center: center,
                        depth_bias: 0.0,
                    },
                    distance: 0.0,
                    pipeline: id,
                    // Sorted phases key items by (entity, main entity): a distinct
                    // placeholder per renderer keeps an effect's renderers (and
                    // emitters) from replacing each other.
                    entity: (renderer_entity(i), MainEntity::from(main)),
                    draw_function: draw,
                    batch_range: i as u32..i as u32 + 1,
                    extra_index: PhaseItemExtraIndex::None,
                    indexed: false,
                });
            }
        }
    }
    if cache.waiting_pipelines().next().is_some() {
        all_ready = false;
    }
    ready.0.store(all_ready, Ordering::Release);
}

// ------------------------------------------------------- simulate and cull

/// Frustum planes of `clip_from_world`, moved into the effect's local space.
fn local_view(view: &ExtractedView, world_from_local: &Mat4) -> ViewParams {
    let clip_from_world = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
    let clip_from_local = clip_from_world * *world_from_local;
    let rows = [
        clip_from_local.row(0),
        clip_from_local.row(1),
        clip_from_local.row(2),
        clip_from_local.row(3),
    ];
    // Reverse-Z: near is z <= w; the far plane is at infinity (z >= 0 always holds).
    let raw = [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[3] - rows[2],
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    ];
    let mut planes = [[0.0f32; 4]; 6];
    for (i, p) in raw.iter().enumerate() {
        let l = p.truncate().length();
        planes[i] = if l > 0.0 {
            (*p / l).to_array()
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
    }
    let local_from_world = world_from_local.inverse();
    let world_from_view = view.world_from_view.to_matrix();
    let cam = local_from_world.transform_point3(world_from_view.w_axis.truncate());
    let fwd = local_from_world
        .transform_vector3(-world_from_view.z_axis.truncate())
        .normalize_or(Vec3::NEG_Z);
    ViewParams {
        planes,
        camera_position: cam.to_array(),
        camera_forward: fwd.to_array(),
    }
}

/// View parameters that cull nothing (ribbons).
pub const NO_VIEW: ViewParams = ViewParams {
    planes: [[0.0, 0.0, 0.0, 1.0]; 6],
    camera_position: [0.0; 3],
    camera_forward: [0.0, 0.0, -1.0],
};

fn simulate(
    mut ctx: RenderContext,
    queue: Res<RenderQueue>,
    device: Res<RenderDevice>,
    mut state: ResMut<FxState>,
    extracted: Res<ExtractedEffects>,
    views: Query<&ExtractedView>,
    depth: Res<FxSceneDepth>,
    rt: Res<FxRayTracingScene>,
) {
    let state = &mut *state;
    let wd = device.wgpu_device();
    // What scene collisions read this frame: the depth (per batch), the view (per instance).
    for batch in state.batches.iter_mut().flatten() {
        // The previous frame's commands were submitted: its frame slots are free.
        batch.gpu.submitted();
        if batch.members.is_empty() {
            continue;
        }
        let Some(first) = state.effects.get(&batch.members[0]) else { continue };
        if !first.gpu.uses_scene() {
            continue;
        }
        if batch.depth_generation != depth.generation {
            batch.gpu.set_scene_depth(wd, depth.texture_view());
            batch.depth_generation = depth.generation;
        }
        if first.gpu.uses_ray_tracing() {
            batch.gpu.set_ray_tracing_scene(rt.bind_group.clone());
        }
        for m in &batch.members {
            if let Some(inst) = state.effects.get_mut(m) {
                inst.gpu.set_scene_view(&queue, inst.world_from_local, depth.view.as_ref());
            }
        }
    }
    // The systems' warm-ups (Warmup Time) run once per instance, in their own submissions.
    for batch in state.batches.iter_mut().flatten() {
        for m in &batch.members {
            if let Some(inst) = state.effects.get_mut(m)
                && inst.gpu.needs_warm_up()
            {
                batch.gpu.warm_up(wd, &queue, &mut inst.gpu);
            }
        }
    }
    let recorder = ctx.diagnostic_recorder();
    let diagnostics = recorder.as_deref();
    let encoder = ctx.command_encoder();
    let span = diagnostics.time_span(encoder, "fx_sim");
    let steps = extracted.steps.min(fx_gpu::MAX_STEPS_PER_SUBMIT);
    for batch in state.batches.iter_mut().flatten() {
        if batch.members.is_empty() {
            continue;
        }
        // Every member steps in the same dispatches.
        for s in 0..steps {
            for m in &batch.members {
                if let Some(inst) = state.effects.get_mut(m) {
                    // The owner's path through the frame, step by step (world-space emitters).
                    let (from, to) = &inst.owner_poses;
                    inst.gpu.previous_owner_pose = from.lerp(to, s as f32 / steps as f32);
                    inst.gpu.owner_pose = from.lerp(to, (s + 1) as f32 / steps as f32);
                    batch.gpu.stage(&mut inst.gpu, encoder, extracted.dt);
                }
            }
            batch.gpu.record(&queue, encoder);
        }
        for (e, copies) in batch.copies.iter_mut().enumerate() {
            copies.clear();
            for m in &batch.members {
                if let Some(inst) = state.effects.get_mut(m) {
                    copies.push(inst.gpu.slot * 2 + inst.gpu.emitters[e].current as u32);
                    if e == 0 && inst.draining {
                        inst.drain_steps += steps;
                    }
                }
            }
        }
    }
    span.end(encoder);
    let span = diagnostics.time_span(encoder, "fx_cull_sort");
    for batch in state.batches.iter_mut().flatten() {
        if batch.members.is_empty() {
            continue;
        }
        for rd in &mut batch.draws {
            let mut payload = false;
            for view in &views {
                if let Some(vs) = rd.views.get_mut(&view.retained_view_entity) {
                    let vp = local_view(view, &Mat4::IDENTITY);
                    vs.draw.prepare(&queue, &batch.copies[rd.emitter], &vp);
                    payload |= vs.draw.payload_used(batch.copies[rd.emitter].len());
                }
            }
            if payload {
                // FxRenderer.payload (word 43; write_uniform leaves it 0).
                queue.write_buffer(&rd.uniform, 43 * 4, &1u32.to_le_bytes());
            }
        }
    }
    // Every draw's cull and sort, phase by phase in one compute pass each, so
    // the small dispatches of different emitters overlap on the GPU.
    for phase in CullPhase::ALL {
        let draws = || {
            state
                .batches
                .iter()
                .flatten()
                .filter(|b| !b.members.is_empty())
                .flat_map(|b| b.draws.iter())
                .flat_map(|rd| views.iter().filter_map(|v| rd.views.get(&v.retained_view_entity)))
        };
        if !draws().any(|vs| vs.draw.phase_used(phase)) {
            continue;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(phase.label()),
            timestamp_writes: None,
        });
        for vs in draws() {
            vs.draw.record_phase(phase, &mut pass, &state.view_pipelines);
        }
    }
    span.end(encoder);
    let span = diagnostics.time_span(encoder, "fx_ribbon_build");
    for inst in state.effects.values_mut() {
        if inst.paused {
            continue;
        }
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            if let Some(rs) = &mut rd.ribbon {
                rs.view
                    .draw
                    .record(&queue, encoder, &state.view_pipelines, em, &NO_VIEW);
                rs.build.record(encoder, &state.ribbon_pipelines, em);
            }
        }
    }
    span.end(encoder);
    // Volumes and components: spans only when some effect has them.
    let (any_volume, any_component) = state.effects.values().fold((false, false), |(v, c), i| {
        (
            v || i.draws.iter().any(|d| d.volume.is_some()),
            c || i.draws.iter().any(|d| d.component.is_some()),
        )
    });
    let span = any_volume.then(|| diagnostics.time_span(encoder, "fx_volumes"));
    for inst in state.effects.values_mut() {
        if inst.paused {
            continue;
        }
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            if let Some(vs) = &mut rd.volume {
                vs.draw.record(&queue, encoder, &state.view_pipelines, em, &NO_VIEW);
                vs.splat.record(encoder, &state.volume_pipelines, em);
            }
        }
    }
    if let Some(span) = span {
        span.end(encoder);
    }
    let span = any_component.then(|| diagnostics.time_span(encoder, "fx_components"));
    for inst in state.effects.values_mut() {
        if inst.paused {
            continue;
        }
        let tag = inst.gpu.frame as u64;
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            if let Some(cs) = &mut rd.component {
                cs.draw.record(&queue, encoder, &state.view_pipelines, em, &NO_VIEW);
                cs.gather.record(encoder, &state.gather_pipelines, em, tag);
            }
        }
    }
    if let Some(span) = span {
        span.end(encoder);
    }
    let span = diagnostics.time_span(encoder, "fx_lights");
    for inst in state.effects.values_mut() {
        if inst.paused {
            continue;
        }
        let tag = inst.gpu.frame as u64;
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            let Some(ls) = &mut rd.light else { continue };
            let Some(view) = views.iter().find(|v| v.retained_view_entity == ls.view) else {
                continue;
            };
            // World-space emitters: particles (and their lights) are in world space.
            let world_space = !inst.gpu.system.emitters[rd.emitter].local_space;
            let vp = local_view(view, if world_space { &Mat4::IDENTITY } else { &inst.world_from_local });
            ls.draw
                .record(&queue, encoder, &state.view_pipelines, em, &vp);
            ls.gather.record(encoder, &state.light_pipelines, em, tag);
        }
    }
    span.end(encoder);
}

/// After the frame's submission: maps the light lists recorded this frame
/// and hands the newest ones to the main world. Deterministic mode waits for
/// them, so lights lag the particles by exactly one frame.
fn read_lights(
    mut state: ResMut<FxState>,
    device: Res<RenderDevice>,
    inbox: Res<LightInbox>,
    components: Res<component::ComponentInbox>,
    deterministic: Res<FxDeterministic>,
) {
    let mut any = false;
    for inst in state.effects.values_mut() {
        for rd in &mut inst.draws {
            if let Some(ls) = &mut rd.light {
                ls.gather.after_submit();
                any = true;
            }
            if let Some(cs) = &mut rd.component {
                cs.gather.after_submit();
                any = true;
            }
        }
    }
    if !any {
        return;
    }
    let poll = if deterministic.0 {
        wgpu::PollType::wait_indefinitely()
    } else {
        wgpu::PollType::Poll
    };
    let _ = device.wgpu_device().poll(poll);
    let mut out = inbox.0.lock().unwrap();
    for (&effect, inst) in &mut state.effects {
        for (ri, rd) in inst.draws.iter_mut().enumerate() {
            let (Some(ls), Settings::Light(settings)) = (&mut rd.light, &rd.settings) else {
                continue;
            };
            if ls.gather.collect() {
                out.retain(|u| !(u.effect == effect && u.renderer == ri));
                out.push(LightUpdate {
                    effect,
                    renderer: ri,
                    settings: settings.clone(),
                    lights: ls.gather.lights.clone(),
                    world_space: !inst.gpu.system.emitters[rd.emitter].local_space,
                });
            }
        }
    }
    drop(out);
    let mut out = components.0.lock().unwrap();
    for (&effect, inst) in &mut state.effects {
        for (ri, rd) in inst.draws.iter_mut().enumerate() {
            let (Some(cs), Settings::Component(settings)) = (&mut rd.component, &rd.settings) else {
                continue;
            };
            if cs.gather.collect() {
                out.retain(|u| !(u.effect == effect && u.renderer == ri));
                out.push(component::ComponentUpdate {
                    effect,
                    renderer: ri,
                    settings: settings.clone(),
                    rows: cs.gather.rows.clone(),
                    world_space: !inst.gpu.system.emitters[rd.emitter].local_space,
                });
            }
        }
    }
}

// --------------------------------------------------------------------- draw

type DrawParticlesCommands = (SetItemPipeline, SetMeshViewBindGroup<0>, DrawParticles);

struct DrawParticles;

impl<P: PhaseItem> RenderCommand<P> for DrawParticles {
    type Param = SRes<FxState>;
    type ViewQuery = Read<ExtractedView>;
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<()>,
        state: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let state = state.into_inner();
        if let Some(b) = entity_batch(item.main_entity().id()) {
            let Some(batch) = state.batches.get(b).and_then(Option::as_ref) else {
                return RenderCommandResult::Skip;
            };
            let r = item.batch_range().start as usize;
            let Some(rd) = batch.draws.iter().find(|d| d.renderer == r) else {
                return RenderCommandResult::Skip;
            };
            let Some(vs) = rd.views.get(&view.retained_view_entity) else {
                return RenderCommandResult::Skip;
            };
            pass.set_bind_group(1, &vs.bind_group, &[]);
            match &rd.settings {
                Settings::Mesh(_) => {
                    let Some(mesh) = batch.assets[r].mesh.as_ref().and_then(|h| state.meshes.get(&h.id())) else {
                        return RenderCommandResult::Skip;
                    };
                    pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed_indirect(&vs.draw_args, 0);
                }
                _ => pass.draw_indirect(&vs.draw_args, 0),
            }
            return RenderCommandResult::Success;
        }
        let Some(inst) = state.effects.get(&item.main_entity().id()) else {
            return RenderCommandResult::Skip;
        };
        let Some(rd) = inst.draws.get(item.batch_range().start as usize) else {
            return RenderCommandResult::Skip;
        };
        let parity = inst.gpu.emitters[rd.emitter].current;
        let _ = view;
        match &rd.settings {
            Settings::Ribbon(_) => {
                let Some(rs) = &rd.ribbon else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(1, &rs.view.bind_groups[parity], &[]);
                pass.draw_indirect(&rs.view.draw_args, 0);
            }
            Settings::Volumetric(_) => {
                let Some(vs) = &rd.volume else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(1, &vs.bind_group, &[]);
                pass.draw(0..36, 0..1);
            }
            _ => return RenderCommandResult::Skip,
        }
        RenderCommandResult::Success
    }
}

/// The phase item entity of renderer `i` of an effect: not a real render
/// entity (the draw reads nothing from it), but distinct per renderer.
fn renderer_entity(i: usize) -> Entity {
    Entity::from_raw_u32(u32::MAX - 2 - i as u32).unwrap_or(Entity::PLACEHOLDER)
}

/// The phase item main entity of batch `b` (renderer draws of every instance
/// in it): not a real entity, distinct from real ones and per batch.
fn batch_entity(b: usize) -> Entity {
    Entity::from_raw_u32(BATCH_ENTITY_BASE - b as u32).unwrap_or(Entity::PLACEHOLDER)
}

const BATCH_ENTITY_BASE: u32 = u32::MAX - 4096;

/// The batch index of a [`batch_entity`].
fn entity_batch(e: Entity) -> Option<usize> {
    let i = e.index_u32();
    (i <= BATCH_ENTITY_BASE && i > BATCH_ENTITY_BASE - 1_000_000).then(|| (BATCH_ENTITY_BASE - i) as usize)
}

/// Systems of the `RenderGraph` schedule other plugins order against.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FxRenderGraphSystems {
    /// Records this frame's simulation steps, culls, ribbon builds and light
    /// gathers. Particle buffers are final for the frame after it.
    Simulate,
}

impl FxState {
    /// The simulated effect of main-world entity `main` and its
    /// world-from-local transform this frame.
    pub fn effect(&self, main: Entity) -> Option<(&GpuEffect, Mat4)> {
        self.effects
            .get(&main)
            .map(|inst| (&inst.gpu, inst.world_from_local))
    }

    /// The cull/sort pipelines, for an [`fx_gpu::view::ViewDraw`] of one's own.
    pub fn view_pipelines(&self) -> &ViewPipelines {
        &self.view_pipelines
    }

    /// The GPU-built ribbon of renderer `renderer` (index among all the
    /// effect's renderers) of effect `main`, once it was prepared: built in
    /// [`FxRenderGraphSystems::Simulate`], read it after.
    pub fn ribbon(&self, main: Entity, renderer: usize) -> Option<RibbonBuffers<'_>> {
        let rs = self.effects.get(&main)?.draws.get(renderer)?.ribbon.as_ref()?;
        Some(RibbonBuffers {
            indices: &rs.view.draw.indices,
            dist: &rs.build.dist,
            ribbons: &rs.build.ribbons,
            meta: &rs.build.meta,
        })
    }
}

/// A ribbon renderer's GPU-built inputs (see [`fx_gpu::ribbon::RibbonBuild`]).
pub struct RibbonBuffers<'a> {
    /// Particle indices sorted by (RibbonID, link order) (`array<u32>`).
    pub indices: &'a wgpu::Buffer,
    /// Distance along the sorted polyline at each sorted position (`array<f32>`).
    pub dist: &'a wgpu::Buffer,
    /// (first, last) sorted position per ribbon id (`array<u32>`).
    pub ribbons: &'a wgpu::Buffer,
    /// `[sorted particle count, ..]` (`array<u32>`).
    pub meta: &'a wgpu::Buffer,
}

/// The built-in sprite texture (soft round dot, or a puff per flipbook cell
/// for a `cols` × `rows` grid) as an image, for renderers that name none.
pub fn builtin_sprite_image(cols: u32, rows: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    let (cols, rows) = (cols.max(1), rows.max(1));
    Image::new(
        Extent3d {
            width: cols * textures::CELL,
            height: rows * textures::CELL,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        textures::pixels(cols, rows),
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Systems of the `Render` schedule other plugins order against.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FxRenderSystems {
    /// Builds and updates the effects' GPU state ([`FxState`]).
    Prepare,
}

/// Renderer assets are shared by every instance of a system unless the
/// instance replaces a mesh.
fn assets_key(entity: Entity, effect: &ParticleEffect) -> (Entity, usize) {
    let system = Arc::as_ptr(&effect.system) as usize;
    if effect.meshes.is_empty() {
        (Entity::PLACEHOLDER, system)
    } else {
        (entity, system)
    }
}

// ------------------------------------------------------------------ pooling

/// Idle instances kept per system.
const MAX_IDLE_PER_SYSTEM: usize = 1024;
/// Frames a system may go unused before its pool and program are dropped.
const POOL_UNUSED_FRAMES: u64 = 600;

impl FxState {
    /// Puts an instance whose entity is gone back in its batch key's pool
    /// (it keeps its slot), or frees its slot when the pool is full.
    fn release(&mut self, mut inst: EffectInstance) {
        self.pool_dirty = true;
        let idle = self.pool.entry(inst.key).or_default();
        if idle.len() < MAX_IDLE_PER_SYSTEM {
            inst.draining = false;
            inst.paused = false;
            idle.push(inst);
        } else if let Some(b) = self.batches.get_mut(inst.batch).and_then(Option::as_mut) {
            b.gpu.release(inst.gpu);
            b.used -= 1;
            if b.used == 0 {
                self.batches[inst.batch] = None;
            }
        }
    }
}

/// A new instance in slot `gpu` of batch `batch`, its ribbon and light state built lazily.
fn new_instance(
    wd: &wgpu::Device,
    gpu: GpuEffect,
    batch: usize,
    key: BatchKey,
    ex: &ExtractedEffect,
    frame: u64,
) -> EffectInstance {
    let draws = renderers(&ex.system)
        .enumerate()
        .map(|(i, (emitter, r))| RendererDraw {
            emitter,
            settings: match r {
                CookedRenderer::Sprite(s) => Settings::Sprite(s.clone()),
                CookedRenderer::Ribbon(s) => Settings::Ribbon(s.clone()),
                CookedRenderer::Mesh(s) => Settings::Mesh(s.clone()),
                CookedRenderer::Light(s) => Settings::Light(s.clone()),
                CookedRenderer::Decal(s) => Settings::Decal(s.clone()),
                CookedRenderer::Component(s) => Settings::Component(s.clone()),
                CookedRenderer::Volumetric(s) => Settings::Volumetric(s.clone()),
            },
            uniform: wd.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx renderer uniform"),
                size: UNIFORM_BYTES as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            assets: ex.assets.get(i).cloned().unwrap_or_default(),
            ribbon: None,
            light: None,
            component: None,
            volume: None,
            scratch: [0; UNIFORM_BYTES],
        })
        .collect();
    EffectInstance {
        system: ex.system.clone(),
        seed: ex.seed,
        gpu,
        batch,
        key,
        draws,
        world_from_local: ex.world_from_local,
        owner_poses: Default::default(),
        uploaded: HashMap::default(),
        seen: frame,
        generation: ex.generation,
        paused: false,
        draining: false,
        drain_steps: 0,
    }
}

// ------------------------------------------- one-shot completion readback

/// Count readback of draining one-shots: their emitters' living counts are
/// copied after the simulation into a staging buffer and mapped
/// asynchronously; an instance whose counts are all zero is reported to the
/// main world ([`game::DoneInbox`]). One readback in flight at a time; the
/// buffer and lists are reused (they grow only while warming up).
#[derive(Resource, Default)]
struct DoneReadback {
    buffer: Option<wgpu::Buffer>,
    /// `(entity, generation, emitters)` in copy order.
    layout: Arc<Mutex<Vec<(Entity, u32, u32)>>>,
    bytes: u64,
    copied: bool,
    busy: Arc<AtomicBool>,
}

fn copy_done_counts(
    mut ctx: RenderContext,
    state: Option<Res<FxState>>,
    device: Res<RenderDevice>,
    mut rb: ResMut<DoneReadback>,
) {
    let Some(state) = state else { return };
    if rb.busy.load(Ordering::Acquire) || rb.copied {
        return;
    }
    let draining = || {
        state
            .effects
            .iter()
            .filter(|(_, i)| i.draining && i.drain_steps > 0 && !i.paused)
    };
    let words: usize = draining().map(|(_, i)| i.gpu.emitters.len()).sum();
    if words == 0 {
        return;
    }
    let bytes = (words * 4) as u64;
    if rb.buffer.as_ref().is_none_or(|b| b.size() < bytes) {
        rb.buffer = Some(device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx one-shot counts"),
            size: bytes.next_power_of_two().max(256),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    let rb = &mut *rb;
    let buffer = rb.buffer.as_ref().unwrap();
    let encoder = ctx.command_encoder();
    let mut layout = rb.layout.lock().unwrap();
    layout.clear();
    let mut off = 0;
    for (entity, inst) in draining() {
        for em in &inst.gpu.emitters {
            encoder.copy_buffer_to_buffer(&em.state.buffer, em.state.offset + fx_gpu::STATE_COUNT * 4, buffer, off, 4);
            off += 4;
        }
        layout.push((*entity, inst.generation, inst.gpu.emitters.len() as u32));
    }
    rb.bytes = bytes;
    rb.copied = true;
}

/// After the frame's submit: map what `copy_done_counts` copied.
fn map_done_counts(mut rb: ResMut<DoneReadback>, inbox: Res<game::DoneInbox>) {
    if !rb.copied {
        return;
    }
    rb.copied = false;
    rb.busy.store(true, Ordering::Release);
    let buffer = rb.buffer.clone().unwrap();
    let layout = rb.layout.clone();
    let busy = rb.busy.clone();
    let done = inbox.0.clone();
    let bytes = rb.bytes;
    buffer.clone().slice(..bytes).map_async(wgpu::MapMode::Read, move |res| {
        if res.is_ok() {
            {
                let data = buffer.slice(..bytes).get_mapped_range().unwrap();
                let layout = layout.lock().unwrap();
                let mut done = done.lock().unwrap();
                let mut w = 0usize;
                for &(entity, generation, n) in layout.iter() {
                    let empty = (0..n as usize).all(|k| {
                        let at = (w + k) * 4;
                        u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) == 0
                    });
                    w += n as usize;
                    if empty {
                        done.push((entity, generation));
                    }
                }
            }
            buffer.unmap();
        }
        busy.store(false, Ordering::Release);
    });
}

/// Identity of a renderer asset list (instances drawn together share it).
fn assets_hash(assets: &[RendererAssets]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    for a in assets {
        a.texture.as_ref().map(|h| h.id()).hash(&mut h);
        a.material.as_ref().map(|h| h.id()).hash(&mut h);
        a.mesh.as_ref().map(|h| h.id()).hash(&mut h);
    }
    h.finish()
}

/// A mesh renderer's mesh asset: a glTF file without a label is its first
/// primitive (`"x.glb"` = `"x.glb#Mesh0/Primitive0"`).
pub fn load_mesh_asset(asset_server: &AssetServer, mesh: &str) -> Handle<Mesh> {
    let path = bevy::asset::AssetPath::parse(mesh).into_owned();
    let gltf = matches!(path.get_full_extension().as_deref(), Some("glb") | Some("gltf"));
    if path.label().is_none() && gltf {
        asset_server.load(GltfAssetLabel::Primitive { mesh: 0, primitive: 0 }.from_asset(path))
    } else {
        asset_server.load::<Mesh>(path)
    }
}
