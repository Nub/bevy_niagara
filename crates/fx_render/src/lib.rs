//! Bevy plugin for GPU-simulated effects drawn by Niagara-style sprite
//! renderers in Bevy's raster pipeline (Transparent3d phase).
//!
//! Main world: spawn a [`ParticleEffect`] (a cooked system and a seed) with a
//! `Transform`; [`FxClock`] decides how many fixed steps run each frame.
//! Render world, each frame:
//! - `prepare_effects` builds an [`fx_gpu::GpuEffect`] per effect and one
//!   culling/sort set per (sprite renderer, view), writes the sprite uniforms;
//! - `queue_sprites` adds one Transparent3d item per sprite renderer and view;
//! - `simulate` (in the `RenderGraph` schedule, before the cameras) records
//!   the sim steps and the per-view cull/sort into the frame's encoder;
//! - `DrawSprites` binds the culled index list and issues `draw_indirect`.
//!
//! No counts are read back; the only per-frame CPU→GPU traffic is a few
//! uniform writes from fixed-size scratch arrays.

mod textures;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
use bevy::shader::{Shader, ShaderDefVal};
use fx_core::cook::{CookedRenderer, CookedSprite, CookedSystem};
use fx_core::model::{BlendMode, FacingMode, SortMode, SpriteAlignment};
use fx_gpu::GpuEffect;
use fx_gpu::view::{ViewDraw, ViewParams, ViewPipelines};

/// A particle system instance. Its particles are simulated in the entity's
/// local space and drawn with the entity's `GlobalTransform`.
#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct ParticleEffect {
    pub system: Arc<CookedSystem>,
    pub seed: u32,
}

impl ParticleEffect {
    pub fn new(system: CookedSystem, seed: u32) -> Self {
        ParticleEffect {
            system: Arc::new(system),
            seed,
        }
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

/// Set by the render world once every effect is built and every sprite
/// pipeline (and every other pipeline in the cache) has compiled.
#[derive(Resource, Clone, Default)]
pub struct FxReady(pub Arc<AtomicBool>);

impl FxReady {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// `deterministic`: draw unsorted sprites in UniqueID order (a GPU sort) so
/// that additive blending rounds the same way every run; for headless
/// previews whose images must match bit for bit.
#[derive(Default)]
pub struct FxPlugin {
    pub deterministic: bool,
}

#[derive(Resource, Clone, Copy)]
struct FxDeterministic(bool);

impl Plugin for FxPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "sprite.wesl");
        let ready = FxReady::default();
        app.init_resource::<FxClock>()
            .insert_resource(ready.clone())
            .add_systems(PostUpdate, advance_clock);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(ready)
            .insert_resource(FxDeterministic(self.deterministic))
            .init_resource::<ExtractedEffects>()
            .init_resource::<TextureHandles>()
            .add_render_command::<Transparent3d, DrawSpritesCommands>()
            .init_gpu_resource::<SpecializedRenderPipelines<SpritePipeline>>()
            .add_systems(RenderStartup, init_render.after(MeshPipelineSystems))
            .add_systems(ExtractSchedule, extract_effects)
            .add_systems(
                Render,
                prepare_effects.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(Render, queue_sprites.in_set(RenderSystems::Queue))
            .add_systems(RenderGraph, simulate.before(camera_driver));
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

// ---------------------------------------------------------------- extraction

struct ExtractedEffect {
    main: Entity,
    system: Arc<CookedSystem>,
    seed: u32,
    world_from_local: Mat4,
    textures: Arc<Vec<Option<Handle<Image>>>>,
}

#[derive(Resource, Default)]
struct ExtractedEffects {
    steps: u32,
    dt: f32,
    effects: Vec<ExtractedEffect>,
}

/// `(emitter index, sprite settings)` for every sprite renderer of a system.
fn sprite_renderers(sys: &CookedSystem) -> impl Iterator<Item = (usize, &CookedSprite)> {
    sys.emitters.iter().enumerate().flat_map(|(ei, e)| {
        e.renderers.iter().map(move |r| match r {
            CookedRenderer::Sprite(s) => (ei, s),
        })
    })
}

/// Sprite textures per effect (in [`sprite_renderers`] order), loaded from
/// their asset paths when the effect is first seen. Held in the render world,
/// which keeps them loaded.
#[derive(Resource, Default)]
struct TextureHandles(HashMap<(Entity, usize), Arc<Vec<Option<Handle<Image>>>>>);

fn extract_effects(
    mut out: ResMut<ExtractedEffects>,
    mut handles: ResMut<TextureHandles>,
    clock: Extract<Res<FxClock>>,
    effects: Extract<Query<(Entity, &ParticleEffect, &GlobalTransform)>>,
    asset_server: Extract<Res<AssetServer>>,
) {
    out.steps = clock.steps_this_frame;
    out.dt = clock.step;
    out.effects.clear();
    for (entity, effect, transform) in &effects {
        let key = (entity, Arc::as_ptr(&effect.system) as usize);
        let textures = handles
            .0
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    sprite_renderers(&effect.system)
                        .map(|(_, s)| {
                            s.texture
                                .as_ref()
                                .map(|p| asset_server.load::<Image>(p.clone()))
                        })
                        .collect(),
                )
            })
            .clone();
        out.effects.push(ExtractedEffect {
            main: entity,
            system: effect.system.clone(),
            seed: effect.seed,
            world_from_local: transform.to_matrix(),
            textures,
        });
    }
    if handles.0.len() > out.effects.len() {
        let live: Vec<(Entity, usize)> = out
            .effects
            .iter()
            .map(|e| (e.main, Arc::as_ptr(&e.system) as usize))
            .collect();
        handles.0.retain(|k, _| live.contains(k));
    }
}
// ------------------------------------------------------------------ pipeline

#[derive(Resource)]
struct SpritePipeline {
    mesh_pipeline: MeshPipeline,
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SpriteKey {
    view_key: MeshPipelineKey,
    blend: BlendMode,
    lit: bool,
    soft: bool,
}

impl SpecializedRenderPipeline for SpritePipeline {
    type Key = SpriteKey;

    fn specialize(&self, key: SpriteKey) -> RenderPipelineDescriptor {
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
        let blend = match key.blend {
            BlendMode::Alpha => {
                defs.push("BLEND_ALPHA".into());
                BlendState::ALPHA_BLENDING
            }
            BlendMode::Additive => {
                defs.push("BLEND_ADDITIVE".into());
                BlendState {
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
                }
            }
            BlendMode::Premultiplied => {
                defs.push("BLEND_PREMULTIPLIED".into());
                BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
        };
        let view_layout = self.mesh_pipeline.get_view_layout(key.view_key.into());
        RenderPipelineDescriptor {
            label: Some("fx sprite".into()),
            layout: vec![view_layout.main_layout.clone(), self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.view_key.target_format(),
                    blend: Some(blend),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
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

const SPRITE_UNIFORM_BYTES: usize = 64 + 16 + 16 + 8 + 8 + 4 * 4 + 4 * 4;

// ---------------------------------------------------------------- GPU state

struct SpriteView {
    draw: ViewDraw,
    /// By particle buffer parity.
    bind_groups: [BindGroup; 2],
    draw_args: Buffer,
    /// Bind groups use the loaded texture (not the built-in fallback).
    uses_loaded_texture: bool,
    seen: u64,
}

struct SpriteDraw {
    emitter: usize,
    settings: CookedSprite,
    uniform: fx_gpu::wgpu::Buffer,
    texture: Option<Handle<Image>>,
    views: HashMap<RetainedViewEntity, SpriteView>,
    scratch: [u8; SPRITE_UNIFORM_BYTES],
}

struct EffectInstance {
    system: Arc<CookedSystem>,
    seed: u32,
    gpu: GpuEffect,
    sprites: Vec<SpriteDraw>,
    world_from_local: Mat4,
    seen: u64,
}

#[derive(Resource)]
struct FxState {
    view_pipelines: ViewPipelines,
    effects: HashMap<Entity, EffectInstance>,
    builtin: textures::Builtin,
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
        "fx sprite",
        &BindGroupLayoutEntries::sequential(
            stages,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    commands.insert_resource(SpritePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "sprite.wesl"),
    });
    let wd = device.wgpu_device();
    commands.insert_resource(FxState {
        view_pipelines: ViewPipelines::new(wd),
        effects: HashMap::default(),
        builtin: textures::Builtin::new(wd),
        frame: 0,
    });
}

fn sort_mode(s: SortMode, deterministic: bool) -> fx_gpu::view::SortMode {
    match s {
        SortMode::None if deterministic => fx_gpu::view::SortMode::UniqueId,
        SortMode::None => fx_gpu::view::SortMode::None,
        SortMode::ViewDepth => fx_gpu::view::SortMode::ViewDepth,
        SortMode::ViewDistance => fx_gpu::view::SortMode::ViewDistance,
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_effects(
    mut state: ResMut<FxState>,
    extracted: Res<ExtractedEffects>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<SpritePipeline>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    views: Query<&ExtractedView>,
    deterministic: Res<FxDeterministic>,
) {
    let state = &mut *state;
    state.frame += 1;
    let frame = state.frame;
    let wd = device.wgpu_device();
    let layout = cache.get_bind_group_layout(&pipeline.layout);

    for ex in &extracted.effects {
        let rebuild = match state.effects.get(&ex.main) {
            Some(inst) => !Arc::ptr_eq(&inst.system, &ex.system) || inst.seed != ex.seed,
            None => true,
        };
        if rebuild {
            let gpu = match GpuEffect::new(wd, &ex.system, ex.seed) {
                Ok(g) => g,
                Err(e) => {
                    error!("effect \"{}\" cannot run on the GPU: {e}", ex.system.name);
                    state.effects.remove(&ex.main);
                    continue;
                }
            };
            let textures = ex.textures.clone();
            let sprites = sprite_renderers(&ex.system)
                .enumerate()
                .map(|(i, (emitter, s))| SpriteDraw {
                    emitter,
                    settings: s.clone(),
                    uniform: wd.create_buffer(&fx_gpu::wgpu::BufferDescriptor {
                        label: Some("fx sprite uniform"),
                        size: SPRITE_UNIFORM_BYTES as u64,
                        usage: fx_gpu::wgpu::BufferUsages::UNIFORM
                            | fx_gpu::wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    texture: textures.get(i).cloned().flatten(),
                    views: HashMap::default(),
                    scratch: [0; SPRITE_UNIFORM_BYTES],
                })
                .collect();
            state.effects.insert(
                ex.main,
                EffectInstance {
                    system: ex.system.clone(),
                    seed: ex.seed,
                    gpu,
                    sprites,
                    world_from_local: ex.world_from_local,
                    seen: frame,
                },
            );
        }
        let inst = state.effects.get_mut(&ex.main).unwrap();
        inst.seen = frame;
        inst.world_from_local = ex.world_from_local;

        for sd in &mut inst.sprites {
            let em = &inst.gpu.emitters[sd.emitter];
            write_sprite_uniform(sd, em.capacity, &inst.world_from_local);
            queue.write_buffer(&sd.uniform, 0, &sd.scratch);
            let loaded = sd.texture.as_ref().and_then(|h| images.get(h));
            for view in &views {
                let entry = sd.views.get_mut(&view.retained_view_entity);
                let stale = entry
                    .as_ref()
                    .is_some_and(|v| loaded.is_some() && !v.uses_loaded_texture);
                if entry.is_none() || stale {
                    let draw = ViewDraw::new(
                        wd,
                        &state.view_pipelines,
                        em,
                        sort_mode(sd.settings.sort_mode, deterministic.0),
                    );
                    let [cols, rows] = sd.settings.sub_image_size;
                    let (tex_view, samp): (&fx_gpu::wgpu::TextureView, &fx_gpu::wgpu::Sampler) =
                        match loaded {
                            Some(img) => (&img.texture_view, &img.sampler),
                            None => state.builtin.get(wd, &queue, cols, rows),
                        };
                    let bg = |s: usize| {
                        BindGroup::from(wd.create_bind_group(&fx_gpu::wgpu::BindGroupDescriptor {
                            label: Some("fx sprite"),
                            layout: &layout,
                            entries: &[
                                fx_gpu::wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: sd.uniform.as_entire_binding(),
                                },
                                fx_gpu::wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: em.particles[s].as_entire_binding(),
                                },
                                fx_gpu::wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: draw.indices.as_entire_binding(),
                                },
                                fx_gpu::wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: fx_gpu::wgpu::BindingResource::TextureView(tex_view),
                                },
                                fx_gpu::wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: fx_gpu::wgpu::BindingResource::Sampler(samp),
                                },
                            ],
                        }))
                    };
                    let sv = SpriteView {
                        bind_groups: [bg(0), bg(1)],
                        draw_args: Buffer::from(draw.draw_args.clone()),
                        draw,
                        uses_loaded_texture: loaded.is_some(),
                        seen: frame,
                    };
                    sd.views.insert(view.retained_view_entity, sv);
                }
                sd.views.get_mut(&view.retained_view_entity).unwrap().seen = frame;
            }
            sd.views.retain(|_, v| v.seen == frame);
        }
    }
    state.effects.retain(|_, e| e.seen == frame);
}

fn write_sprite_uniform(sd: &mut SpriteDraw, capacity: u32, world_from_local: &Mat4) {
    let s = &sd.settings;
    let b = &mut sd.scratch;
    let mut put = |w: usize, v: u32| b[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    for (i, x) in world_from_local.to_cols_array().iter().enumerate() {
        put(i, x.to_bits());
    }
    for c in 0..3 {
        put(16 + c, s.custom_facing_vector[c].to_bits());
        put(20 + c, s.custom_alignment_vector[c].to_bits());
    }
    put(24, s.pivot_in_uv_space[0].to_bits());
    put(25, s.pivot_in_uv_space[1].to_bits());
    put(26, (s.sub_image_size[0].max(1) as f32).to_bits());
    put(27, (s.sub_image_size[1].max(1) as f32).to_bits());
    put(28, capacity);
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

#[allow(clippy::too_many_arguments)]
fn queue_sprites(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<SpritePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SpritePipeline>>,
    cache: Res<PipelineCache>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView>,
    view_keys: Res<ViewKeyCache>,
    state: Res<FxState>,
    extracted: Res<ExtractedEffects>,
    ready: Res<FxReady>,
) {
    let draw = draw_functions.read().id::<DrawSpritesCommands>();
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
            for (i, sd) in inst.sprites.iter().enumerate() {
                let s = &sd.settings;
                let key = SpriteKey {
                    view_key,
                    blend: s.blend_mode,
                    lit: s.lit,
                    soft: s.soft_particle_fade_distance > 0.0
                        && view_key.contains(MeshPipelineKey::DEPTH_PREPASS),
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
                    entity: (Entity::PLACEHOLDER, MainEntity::from(*main)),
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

fn simulate(
    mut ctx: RenderContext,
    queue: Res<RenderQueue>,
    mut state: ResMut<FxState>,
    extracted: Res<ExtractedEffects>,
    views: Query<&ExtractedView>,
) {
    let state = &mut *state;
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
        for sd in &mut inst.sprites {
            let em = &inst.gpu.emitters[sd.emitter];
            for view in &views {
                if let Some(sv) = sd.views.get_mut(&view.retained_view_entity) {
                    let vp = local_view(view, &inst.world_from_local);
                    sv.draw
                        .record(&queue, encoder, &state.view_pipelines, em, &vp);
                }
            }
        }
    }
    span.end(encoder);
}

// --------------------------------------------------------------------- draw

type DrawSpritesCommands = (SetItemPipeline, SetMeshViewBindGroup<0>, DrawSprites);

struct DrawSprites;

impl<P: PhaseItem> RenderCommand<P> for DrawSprites {
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
        let Some(sd) = inst.sprites.get(item.batch_range().start as usize) else {
            return RenderCommandResult::Skip;
        };
        let Some(sv) = sd.views.get(&view.retained_view_entity) else {
            return RenderCommandResult::Skip;
        };
        let parity = inst.gpu.emitters[sd.emitter].current;
        pass.set_bind_group(1, &sv.bind_groups[parity], &[]);
        pass.draw_indirect(&sv.draw_args, 0);
        RenderCommandResult::Success
    }
}
