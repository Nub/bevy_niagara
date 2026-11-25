//! Bevy plugin for GPU-simulated effects drawn by Niagara's four core
//! renderers in Bevy's raster pipeline:
//! - **Sprite**: one camera-facing quad per particle (`sprite.wesl`);
//! - **Ribbon**: particles linked by `Particles.RibbonID` and link order into
//!   strips whose geometry the GPU builds (`ribbon.wesl`, `fx_gpu::ribbon`);
//! - **Mesh**: any Bevy mesh instanced per particle with one indexed
//!   indirect draw (`mesh.wesl`);
//! - **Light**: per-particle point lights, culled and capped on the GPU, read
//!   back (only the capped list) and applied as Bevy `PointLight`s.
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
//! - `queue_particles` adds one Transparent3d item per drawn renderer and view;
//! - `simulate` (in the `RenderGraph` schedule, before the cameras) records
//!   the sim steps, the per-view cull/sort, the ribbon builds and the light
//!   gathers into the frame's encoder;
//! - `DrawParticles` binds the renderer's buffers and issues the indirect draw;
//! - `read_lights` (after submission) maps the light lists.
//!
//! No particle counts are read back; the per-frame CPU→GPU traffic is a few
//! uniform writes from fixed-size scratch arrays.

mod mesh_data;
mod scene;
pub mod stats;
mod textures;

pub use stats::{EffectStats, EmitterStats, FxStats, FxStatsData, FxStatsPlugin};
pub use scene::{FxRayTracingScene, FxSceneDepth};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d, TransparentSortingInfo3d};
use bevy::core_pipeline::schedule::camera_driver;
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::math::{Mat4, Vec3, Vec4};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, MeshPipelineSystems, SetMeshViewBindGroup, ViewKeyCache,
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
    sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer_sized,
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
use fx_core::cook::{CookedLight, CookedMesh, CookedRenderer, CookedRibbon, CookedSprite, CookedSystem};
use fx_core::render::{
    BlendMode, FacingMode, MeshBlendMode, MeshFacingMode, RibbonFacingMode, RibbonLinkOrderMode,
    RibbonUvMode, SortMode, SpriteAlignment,
};
use fx_gpu::GpuEffect;
use fx_gpu::lights::{GpuLight, LightGather, LightPipelines};
use fx_gpu::ribbon::{RibbonBuild, RibbonPipelines};
use fx_gpu::view::{CullRadius, ViewDraw, ViewDrawConfig, ViewParams, ViewPipelines};
use fx_gpu::wgpu;

/// A particle system instance. Its particles are simulated in the entity's
/// local space and drawn with the entity's `GlobalTransform`.
#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct ParticleEffect {
    pub system: Arc<CookedSystem>,
    pub seed: u32,
    /// Meshes replacing a mesh renderer's `mesh` setting, by renderer index
    /// (all emitters' renderers in order): any Bevy mesh.
    pub meshes: Vec<(usize, Handle<Mesh>)>,
}

impl ParticleEffect {
    pub fn new(system: CookedSystem, seed: u32) -> Self {
        ParticleEffect {
            system: Arc::new(system),
            seed,
            meshes: Vec::new(),
        }
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
        let ready = FxReady::default();
        let inbox = LightInbox::default();
        let builtins = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            BuiltinMeshes(
                fx_core::render::MESH_BUILTINS
                    .iter()
                    .filter_map(|&n| Some((n, meshes.add(mesh_data::builtin(n)?))))
                    .collect(),
            )
        };
        app.init_resource::<FxClock>()
            .insert_resource(ready.clone())
            .insert_resource(inbox.clone())
            .insert_resource(builtins)
            .init_resource::<LightPools>()
            .add_systems(PostUpdate, advance_clock)
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
            .insert_resource(FxDeterministic(self.deterministic))
            .init_resource::<ExtractedEffects>()
            .init_resource::<FxSceneDepth>()
            .init_resource::<FxRayTracingScene>()
            .init_resource::<RendererAssetsCache>()
            .init_resource::<PendingMeshes>()
            .add_render_command::<Transparent3d, DrawParticlesCommands>()
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
            .add_systems(
                RenderGraph,
                simulate.in_set(FxRenderGraphSystems::Simulate).before(camera_driver),
            )
            .add_systems(RenderGraph, scene::capture_scene_depth.after(camera_driver));
    }
}

fn advance_clock(mut clock: ResMut<FxClock>, time: Res<Time>) {
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
    effects: Query<(), With<ParticleEffect>>,
    mut lights: Query<(&mut PointLight, &mut Transform, &mut Visibility), With<ParticleLight>>,
) {
    let updates: Vec<LightUpdate> = std::mem::take(&mut *inbox.0.lock().unwrap());
    for u in updates {
        if effects.get(u.effect).is_err() {
            continue;
        }
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
            let transform = Transform::from_translation(Vec3::from_array(l.position));
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
    assets: Arc<Vec<RendererAssets>>,
}

#[derive(Resource, Default)]
struct ExtractedEffects {
    steps: u32,
    dt: f32,
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
        CookedRenderer::Light(_) => (None, None),
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_effects(
    mut out: ResMut<ExtractedEffects>,
    mut cache: ResMut<RendererAssetsCache>,
    mut pending: ResMut<PendingMeshes>,
    clock: Extract<Res<FxClock>>,
    effects: Extract<Query<(Entity, &ParticleEffect, &GlobalTransform)>>,
    asset_server: Extract<Res<AssetServer>>,
    builtins: Extract<Res<BuiltinMeshes>>,
    meshes: Extract<Res<Assets<Mesh>>>,
) {
    out.steps = clock.steps_this_frame;
    out.dt = clock.step;
    out.effects.clear();
    for (entity, effect, transform) in &effects {
        let key = (entity, Arc::as_ptr(&effect.system) as usize);
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
                                            None => asset_server.load::<Mesh>(m.mesh.clone()),
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
        out.effects.push(ExtractedEffect {
            main: entity,
            system: effect.system.clone(),
            seed: effect.seed,
            world_from_local: transform.to_matrix(),
            assets,
        });
    }
    if cache.0.len() > out.effects.len() {
        let live: Vec<(Entity, usize)> = out
            .effects
            .iter()
            .map(|e| (e.main, Arc::as_ptr(&e.system) as usize))
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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DrawKind {
    Sprite,
    Ribbon,
    Mesh,
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
        };
        let fragment_shader = key.material.clone().unwrap_or_else(|| shader.clone());
        let view_layout = self.mesh_pipeline.get_view_layout(key.view_key.into());
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![view_layout.main_layout.clone(), self.layout.clone()],
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
                cull_mode: if key.kind == DrawKind::Mesh && key.blend == Blend::Opaque {
                    Some(Face::Back)
                } else {
                    None
                },
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(key.blend == Blend::Opaque),
                depth_compare: Some(CompareFunction::GreaterEqual),
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
const UNIFORM_BYTES: usize = 40 * 4;

// ---------------------------------------------------------------- GPU state

/// A culled (and sorted) draw for one view, with its bind groups.
struct ViewState {
    draw: ViewDraw,
    /// By particle buffer parity.
    bind_groups: [BindGroup; 2],
    draw_args: Buffer,
    /// Bind groups use the loaded texture (not the built-in fallback).
    uses_loaded_texture: bool,
    seen: u64,
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

enum Settings {
    Sprite(CookedSprite),
    Ribbon(CookedRibbon),
    Mesh(CookedMesh),
    Light(CookedLight),
}

struct RendererDraw {
    emitter: usize,
    settings: Settings,
    uniform: wgpu::Buffer,
    assets: RendererAssets,
    /// Sprites and meshes: per view.
    views: HashMap<RetainedViewEntity, ViewState>,
    ribbon: Option<RibbonState>,
    light: Option<LightState>,
    scratch: [u8; UNIFORM_BYTES],
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: Buffer,
    index_count: u32,
    radius: f32,
}

struct EffectInstance {
    system: Arc<CookedSystem>,
    seed: u32,
    gpu: GpuEffect,
    /// Built with the ray-tracing scene layout.
    ray_tracing: bool,
    /// [`FxSceneDepth::generation`] the scene bind group was made for.
    depth_generation: u64,
    draws: Vec<RendererDraw>,
    world_from_local: Mat4,
    seen: u64,
}

/// Render world: the GPU state of every effect. Read it (after
/// [`FxRenderGraphSystems::Simulate`]) to feed particles to other passes.
#[derive(Resource)]
pub struct FxState {
    view_pipelines: ViewPipelines,
    ribbon_pipelines: RibbonPipelines,
    light_pipelines: LightPipelines,
    effects: HashMap<Entity, EffectInstance>,
    builtin: textures::Builtin,
    meshes: HashMap<AssetId<Mesh>, GpuMesh>,
    /// Bound where a renderer has no buffer of its own.
    dummy: wgpu::Buffer,
    frame: u64,
}

fn init_render(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    asset_server: Res<AssetServer>,
) {
    let stages = ShaderStages::VERTEX_FRAGMENT;
    let layout = BindGroupLayoutDescriptor::new(
        "fx particles",
        &BindGroupLayoutEntries::sequential(
            stages,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    commands.insert_resource(ParticlePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        layout,
        sprite: load_embedded_asset!(asset_server.as_ref(), "sprite.wesl"),
        ribbon: load_embedded_asset!(asset_server.as_ref(), "ribbon.wesl"),
        mesh: load_embedded_asset!(asset_server.as_ref(), "mesh.wesl"),
    });
    let wd = device.wgpu_device();
    commands.insert_resource(FxState {
        view_pipelines: ViewPipelines::new(wd),
        ribbon_pipelines: RibbonPipelines::new(wd),
        light_pipelines: LightPipelines::new(wd),
        effects: HashMap::default(),
        builtin: textures::Builtin::new(wd),
        meshes: HashMap::default(),
        dummy: wd.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx dummy storage"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        frame: 0,
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
) -> [BindGroup; 2] {
    let bg = |s: usize| {
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
                    resource: em.particles[s].as_entire_binding(),
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
            ],
        }))
    };
    [bg(0), bg(1)]
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
) {
    let state = &mut *state;
    state.frame += 1;
    let frame = state.frame;
    let wd = device.wgpu_device();
    let layout = cache.get_bind_group_layout(&pipeline.layout);

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

    for ex in &extracted.effects {
        let rebuild = match state.effects.get(&ex.main) {
            Some(inst) => {
                !Arc::ptr_eq(&inst.system, &ex.system)
                    || inst.seed != ex.seed
                    || inst.ray_tracing != rt.layout.is_some()
            }
            None => true,
        };
        if rebuild {
            let options = fx_gpu::GpuOptions {
                ray_tracing_layout: rt.layout.as_ref(),
            };
            let gpu = match GpuEffect::with_options(wd, &ex.system, ex.seed, options) {
                Ok(g) => g,
                Err(e) => {
                    error!("effect \"{}\" cannot run on the GPU: {e}", ex.system.name);
                    state.effects.remove(&ex.main);
                    continue;
                }
            };
            let draws = renderers(&ex.system)
                .enumerate()
                .map(|(i, (emitter, r))| RendererDraw {
                    emitter,
                    settings: match r {
                        CookedRenderer::Sprite(s) => Settings::Sprite(s.clone()),
                        CookedRenderer::Ribbon(s) => Settings::Ribbon(s.clone()),
                        CookedRenderer::Mesh(s) => Settings::Mesh(s.clone()),
                        CookedRenderer::Light(s) => Settings::Light(s.clone()),
                    },
                    uniform: wd.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fx renderer uniform"),
                        size: UNIFORM_BYTES as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    assets: ex.assets.get(i).cloned().unwrap_or_default(),
                    views: HashMap::default(),
                    ribbon: None,
                    light: None,
                    scratch: [0; UNIFORM_BYTES],
                })
                .collect();
            state.effects.insert(
                ex.main,
                EffectInstance {
                    system: ex.system.clone(),
                    seed: ex.seed,
                    gpu,
                    ray_tracing: rt.layout.is_some(),
                    depth_generation: u64::MAX,
                    draws,
                    world_from_local: ex.world_from_local,
                    seen: frame,
                },
            );
        }
        let inst = state.effects.get_mut(&ex.main).unwrap();
        inst.seen = frame;
        inst.world_from_local = ex.world_from_local;

        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            let mesh = rd
                .assets
                .mesh
                .as_ref()
                .and_then(|h| state.meshes.get(&h.id()));
            let sub_index = inst.gpu.system.emitters[rd.emitter].sets_sub_image_index();
            write_uniform(
                rd,
                em,
                &inst.world_from_local,
                mesh.map_or(1.0, |m| m.radius),
                sub_index,
            );
            queue.write_buffer(&rd.uniform, 0, &rd.scratch);
            let loaded = rd.assets.texture.as_ref().and_then(|h| images.get(h));
            let [cols, rows] = match &rd.settings {
                Settings::Sprite(s) => s.sub_image_size,
                _ => [1, 1],
            };
            let tex = |builtin: &mut textures::Builtin| -> (wgpu::TextureView, wgpu::Sampler) {
                match loaded {
                    Some(img) => ((*img.texture_view).clone(), (*img.sampler).clone()),
                    None => {
                        let (v, s) = builtin.get(wd, &queue, cols, rows);
                        (v.clone(), s.clone())
                    }
                }
            };
            match &rd.settings {
                Settings::Sprite(_) | Settings::Mesh(_) => {
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
                        _ => unreachable!(),
                    };
                    for view in &views {
                        let entry = rd.views.get(&view.retained_view_entity);
                        let stale =
                            entry.is_some_and(|v| loaded.is_some() && !v.uses_loaded_texture);
                        if entry.is_none() || stale {
                            let draw =
                                ViewDraw::with_config(wd, &state.view_pipelines, em, config);
                            let (tv, ts) = tex(&mut state.builtin);
                            let bind_groups = particle_bind_groups(
                                wd,
                                &layout,
                                &rd.uniform,
                                em,
                                &draw.indices,
                                (&tv, &ts),
                                [extra0, &state.dummy, &state.dummy],
                            );
                            rd.views.insert(
                                view.retained_view_entity,
                                ViewState {
                                    bind_groups,
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
                Settings::Ribbon(r) => {
                    let stale = rd
                        .ribbon
                        .as_ref()
                        .is_some_and(|v| loaded.is_some() && !v.view.uses_loaded_texture);
                    if rd.ribbon.is_none() || stale {
                        let draw =
                            ViewDraw::with_config(wd, &state.view_pipelines, em, ribbon_config(r));
                        let build = RibbonBuild::new(wd, &state.ribbon_pipelines, em, &draw);
                        let (tv, ts) = tex(&mut state.builtin);
                        let bind_groups = particle_bind_groups(
                            wd,
                            &layout,
                            &rd.uniform,
                            em,
                            &draw.indices,
                            (&tv, &ts),
                            [&build.dist, &build.ribbons, &build.meta],
                        );
                        rd.ribbon = Some(RibbonState {
                            view: ViewState {
                                bind_groups,
                                draw_args: Buffer::from(draw.draw_args.clone()),
                                draw,
                                uses_loaded_texture: loaded.is_some(),
                                seen: frame,
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
            }
        }
    }
    state.effects.retain(|_, e| e.seen == frame);
}

fn bytemuck_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: f32 and u32 slices have no padding and every byte is initialised.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn write_uniform(
    rd: &mut RendererDraw,
    em: &fx_gpu::GpuEmitter,
    world_from_local: &Mat4,
    mesh_radius: f32,
    sub_image_from_index: bool,
) {
    let b = &mut rd.scratch;
    let mut put = |w: usize, v: u32| b[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    for (i, x) in world_from_local.to_cols_array().iter().enumerate() {
        put(i, x.to_bits());
    }
    put(26, 1.0f32.to_bits());
    put(27, 1.0f32.to_bits());
    put(28, em.capacity);
    put(33, (em.words > fx_gpu::WORDS_PER_PARTICLE) as u32);
    put(38, mesh_radius.to_bits());
    match &rd.settings {
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
        Settings::Light(_) => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_particles(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<ParticlePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ParticlePipeline>>,
    cache: Res<PipelineCache>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView>,
    view_keys: Res<ViewKeyCache>,
    state: Res<FxState>,
    extracted: Res<ExtractedEffects>,
    ready: Res<FxReady>,
) {
    let draw = draw_functions.read().id::<DrawParticlesCommands>();
    let mut all_ready = extracted
        .effects
        .iter()
        .all(|e| state.effects.contains_key(&e.main));
    for view in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(&view_key) = view_keys.get(&view.retained_view_entity) else {
            all_ready = false;
            continue;
        };
        for (main, inst) in &state.effects {
            let center = inst.world_from_local.w_axis.truncate();
            for (i, rd) in inst.draws.iter().enumerate() {
                let material = rd.assets.material.clone();
                let textured = rd.assets.texture.is_some();
                let key = match &rd.settings {
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
                        if !rd.views.contains_key(&view.retained_view_entity) {
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
                    Settings::Light(_) => continue,
                };
                let id = pipelines.specialize(&cache, &pipeline, key);
                if cache.get_render_pipeline(id).is_none() {
                    all_ready = false;
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
                    entity: (renderer_entity(i), MainEntity::from(*main)),
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
    // What scene collisions read this frame.
    for inst in state.effects.values_mut() {
        if !inst.gpu.uses_scene() {
            continue;
        }
        if inst.depth_generation != depth.generation {
            inst.gpu
                .set_scene_depth(device.wgpu_device(), depth.texture_view());
            inst.depth_generation = depth.generation;
        }
        inst.gpu
            .set_scene_view(&queue, inst.world_from_local, depth.view.as_ref());
        if inst.gpu.uses_ray_tracing() {
            inst.gpu.set_ray_tracing_scene(rt.bind_group.clone());
        }
    }
    let recorder = ctx.diagnostic_recorder();
    let diagnostics = recorder.as_deref();
    let encoder = ctx.command_encoder();
    let span = diagnostics.time_span(encoder, "fx_sim");
    for inst in state.effects.values_mut() {
        // The previous frame's commands were submitted: its uniform slots are free.
        inst.gpu.submitted();
        for _ in 0..extracted.steps.min(fx_gpu::MAX_STEPS_PER_SUBMIT) {
            inst.gpu.step(&queue, encoder, extracted.dt);
        }
    }
    span.end(encoder);
    let span = diagnostics.time_span(encoder, "fx_cull_sort");
    for inst in state.effects.values_mut() {
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            for view in &views {
                if let Some(vs) = rd.views.get_mut(&view.retained_view_entity) {
                    let vp = local_view(view, &inst.world_from_local);
                    vs.draw
                        .record(&queue, encoder, &state.view_pipelines, em, &vp);
                }
            }
        }
    }
    span.end(encoder);
    let span = diagnostics.time_span(encoder, "fx_ribbon_build");
    for inst in state.effects.values_mut() {
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
    let span = diagnostics.time_span(encoder, "fx_lights");
    for inst in state.effects.values_mut() {
        let tag = inst.gpu.frame as u64;
        for rd in &mut inst.draws {
            let em = &inst.gpu.emitters[rd.emitter];
            let Some(ls) = &mut rd.light else { continue };
            let Some(view) = views.iter().find(|v| v.retained_view_entity == ls.view) else {
                continue;
            };
            let vp = local_view(view, &inst.world_from_local);
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
    deterministic: Res<FxDeterministic>,
) {
    let mut any = false;
    for inst in state.effects.values_mut() {
        for rd in &mut inst.draws {
            if let Some(ls) = &mut rd.light {
                ls.gather.after_submit();
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
        let Some(inst) = state.effects.get(&item.main_entity().id()) else {
            return RenderCommandResult::Skip;
        };
        let Some(rd) = inst.draws.get(item.batch_range().start as usize) else {
            return RenderCommandResult::Skip;
        };
        let parity = inst.gpu.emitters[rd.emitter].current;
        match &rd.settings {
            Settings::Sprite(_) => {
                let Some(vs) = rd.views.get(&view.retained_view_entity) else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(1, &vs.bind_groups[parity], &[]);
                pass.draw_indirect(&vs.draw_args, 0);
            }
            Settings::Ribbon(_) => {
                let Some(rs) = &rd.ribbon else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(1, &rs.view.bind_groups[parity], &[]);
                pass.draw_indirect(&rs.view.draw_args, 0);
            }
            Settings::Mesh(_) => {
                let (Some(vs), Some(mesh)) = (
                    rd.views.get(&view.retained_view_entity),
                    rd.assets.mesh.as_ref().and_then(|h| state.meshes.get(&h.id())),
                ) else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(1, &vs.bind_groups[parity], &[]);
                pass.set_index_buffer(mesh.indices.slice(..), IndexFormat::Uint32);
                pass.draw_indexed_indirect(&vs.draw_args, 0);
            }
            Settings::Light(_) => return RenderCommandResult::Skip,
        }
        RenderCommandResult::Success
    }
}

/// The phase item entity of renderer `i` of an effect: not a real render
/// entity (the draw reads nothing from it), but distinct per renderer.
fn renderer_entity(i: usize) -> Entity {
    Entity::from_raw_u32(u32::MAX - 2 - i as u32).unwrap_or(Entity::PLACEHOLDER)
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
