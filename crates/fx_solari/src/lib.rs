//! Particles in Solari ray tracing.
//!
//! [`FxSolariPlugin`] (add it after `FxPlugin` and `SolariPlugins`) puts every
//! sprite renderer with `ray_tracing: true` into Solari's ray-traced scene:
//!
//! - **One proxy per emitter, never one instance per particle.** The renderer
//!   gets a proxy entity: a [`RaytracingMesh3d`] whose mesh has two quads per
//!   particle slot (up to [`MAX_SLOTS`]). Each frame, after the simulation, a
//!   compute pass (`proxy.wesl`) writes the living particles' quads (the same
//!   facing as the raster sprites, via `fx_render::particle_surface`) into
//!   that mesh's vertices, and Solari's deform pass rebuilds the emitter's one
//!   BLAS from them; the instance sits in the partitioned TLAS like any other.
//! - **Same look in reflections.** The proxy's material ([`ParticleRtMaterial`])
//!   has a Solari hit function (`particle_hit.wesl`) computing the sprite's
//!   colour and opacity with the raster shader's WGSL from the particle's
//!   colour and age (a data texture the proxy pass writes) and its texture.
//! - **Emissive sprites light the scene.** Unlit and additive sprites are
//!   emissive surfaces, which Solari samples as lights. Light renderers'
//!   `PointLight`s are Solari lights already.
//! - **Shadows.** The sprite's opacity cuts the proxy (alpha mask at
//!   [`ALPHA_CUTOFF`]): shadow rays are blocked where the sprite is opaque.
//!
//! Seen directly, particles keep drawing in the raster transparent pass,
//! which runs after Solari's lighting and is depth tested against its
//! G-buffer: additive blending stays exact and nothing is stochastic. With
//! `ray_tracing` off everywhere no proxy exists and Solari does no extra work
//! beyond compiling the one hit function into its shaders.

use std::sync::Arc;

use bevy::asset::{RenderAssetUsages, embedded_asset, load_embedded_asset};
use bevy::camera::Exposure;
use bevy::image::ImageSampler;
use bevy::math::{Mat4, Vec4};
use bevy::mesh::{Indices, MeshRaytracingFlags, PrimitiveTopology, morph::MeshMorphWeights};
use bevy::pbr::{Material, MeshMaterial3d};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::mesh::RenderMesh;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue};
use bevy::render::sync_world::RenderEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::ExtractedView;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::load_shader_library;
use bevy::solari::prelude::{RaytracingMesh3d, SolariLighting};
use bevy::solari::scene::{RaytracingMaterial, RaytracingMaterialDescription, SolariMaterialPlugin};
use fx_core::cook::{CookedRenderer, CookedRibbon, CookedSprite, CookedSystem};
use fx_core::render::{BlendMode, FacingMode, RibbonFacingMode, RibbonLinkOrderMode, RibbonUvMode, SpriteAlignment};
use fx_gpu::view::{CullRadius, SortMode, ViewDraw, ViewDrawConfig};
use fx_gpu::wgpu;
use fx_render::{FxRenderGraphSystems, FxState, NO_VIEW, ParticleEffect};

/// Most particles of one emitter in ray tracing: Solari samples an emissive
/// mesh's triangles with a 16-bit index, and each particle has 4 triangles.
/// Past it, the first `MAX_SLOTS` living particles are traced.
pub const MAX_SLOTS: u32 = 16383;
/// Sprite opacity below which rays pass through the proxy.
pub const ALPHA_CUTOFF: f32 = 0.25;
/// Width of the particle data texture (two texels per particle).
const DATA_WIDTH: u32 = 512;
const PARAMS_BYTES: usize = 52 * 4;

/// Adds ray tracing to `fx_render`'s particles in Solari scenes.
pub struct FxSolariPlugin;

impl Plugin for FxSolariPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "particle_hit.wesl");
        embedded_asset!(app, "proxy.wesl");
        embedded_asset!(app, "ribbon_proxy.wesl");
        app.init_asset::<ParticleRtMaterial>()
            .add_plugins(SolariMaterialPlugin::<ParticleRtMaterial>::default())
            .init_resource::<Proxies>()
            .add_systems(PostUpdate, sync_proxies);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedProxies>()
            .add_systems(RenderStartup, init_proxy_state)
            .add_systems(ExtractSchedule, extract_proxies)
            .add_systems(
                Render,
                prepare_proxies
                    .in_set(RenderSystems::PrepareResources)
                    .after(fx_render::FxRenderSystems::Prepare),
            )
            .add_systems(
                RenderGraph,
                write_proxies
                    .after(FxRenderGraphSystems::Simulate)
                    .before(RenderGraphSystems::Begin),
            )
            .add_systems(
                RenderGraph,
                share_ray_tracing_scene.before(FxRenderGraphSystems::Simulate),
            );
    }
}

// ------------------------------------------------------------------ material

/// What Solari's rays see of a ray-traced sprite renderer: the fallback
/// properties, the sprite texture and the per-particle data texture, refined
/// per hit by `fx_solari::particle_hit`. Never drawn in raster (its proxy has
/// no `Mesh3d`).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ParticleRtMaterial {
    /// Sprite or ribbon texture (ribbons: `None` = the built-in soft edges).
    pub sprite: Option<Handle<Image>>,
    pub data: Handle<Image>,
    /// Unlit or additive: emits its colour.
    pub emissive: bool,
    pub additive: bool,
    pub sub_image: [u32; 2],
    pub sub_uv_blending: bool,
    /// Radiance per unit of raster colour: 1 / the Solari camera's exposure.
    pub radiance_scale: f32,
    /// A ribbon proxy (`ribbon_proxy.wesl`) instead of sprites.
    pub ribbon: bool,
}

impl Material for ParticleRtMaterial {}

impl RaytracingMaterial for ParticleRtMaterial {
    fn raytracing_hit_module() -> Option<&'static str> {
        Some("fx_solari::particle_hit")
    }

    fn raytracing_description(&self) -> RaytracingMaterialDescription {
        let flags = self.emissive as u32 | (self.additive as u32) << 1 | (self.ribbon as u32) << 2;
        RaytracingMaterialDescription {
            fallback: StandardMaterial {
                base_color: if self.emissive { Color::BLACK } else { Color::WHITE },
                // Non-zero so Solari lists it as a light; the hit function
                // gives the real emission.
                emissive: if self.emissive {
                    LinearRgba::rgb(0.1, 0.05, 0.02) * self.radiance_scale
                } else {
                    LinearRgba::BLACK
                },
                alpha_mode: AlphaMode::Mask(ALPHA_CUTOFF),
                double_sided: true,
                cull_mode: None,
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..default()
            },
            params: [
                Vec4::new(
                    self.sub_image[0].max(1) as f32,
                    self.sub_image[1].max(1) as f32,
                    self.sub_uv_blending as u32 as f32,
                    flags as f32,
                ),
                Vec4::new(self.radiance_scale, DATA_WIDTH as f32, 0.0, 0.0),
                Vec4::ZERO,
                Vec4::ZERO,
            ],
            textures: [self.sprite.clone(), Some(self.data.clone()), None, None],
        }
    }
}

// ---------------------------------------------------------------- main world

/// The renderer a proxy traces.
#[derive(Clone)]
pub enum ProxyRenderer {
    Sprite(CookedSprite),
    Ribbon(CookedRibbon),
}

/// The ray-tracing proxy of one sprite or ribbon renderer of a [`ParticleEffect`].
#[derive(Component, Clone)]
pub struct ParticleRtProxy {
    pub effect: Entity,
    /// Index among all the effect's renderers.
    pub renderer: usize,
    pub emitter: usize,
    /// Quad slots in the proxy mesh: one per particle (sprites) or per
    /// segment subdivision (ribbons).
    pub slots: u32,
    pub data: Handle<Image>,
    pub settings: ProxyRenderer,
}

/// Data texels per slot: sprites colour and age; ribbons the colour at both
/// ends of the sub-segment and its U range.
const SPRITE_TEXELS: u32 = 2;
const RIBBON_TEXELS: u32 = 3;

/// Main world: proxy entity and system of each (effect, renderer).
#[derive(Resource, Default)]
struct Proxies {
    by_renderer: HashMap<(Entity, usize), (Entity, usize)>,
    builtin_sprites: HashMap<(u32, u32), Handle<Image>>,
    radiance_scale: f32,
}

fn renderers(sys: &CookedSystem) -> impl Iterator<Item = (usize, &CookedRenderer)> {
    sys.emitters
        .iter()
        .enumerate()
        .flat_map(|(ei, e)| e.renderers.iter().map(move |r| (ei, r)))
}

/// Two quads (front and back) per slot; slot k's UVs are (2k + u, v).
fn proxy_mesh(slots: u32) -> Mesh {
    let n = slots as usize * 8;
    let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut uvs = Vec::with_capacity(n);
    let mut indices = Vec::with_capacity(slots as usize * 12);
    for k in 0..slots {
        for side in 0..2u32 {
            for c in corners {
                uvs.push([2.0 * k as f32 + c[0], c[1]]);
            }
            let b = 8 * k + 4 * side;
            // Counter-clockwise seen from the side the vertex normals face (uv v
            // runs down the sprite): the front quad faces the camera, the back
            // quad away. Solari keeps shading normals on the geometric side.
            if side == 0 {
                indices.extend([b, b + 2, b + 1, b, b + 3, b + 2]);
            } else {
                indices.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
            }
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; n])
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; n])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0f32, 0.0, 0.0, 1.0]; n])
        .with_inserted_indices(Indices::U32(indices));
    mesh.raytracing = MeshRaytracingFlags::NON_OPAQUE;
    mesh
}

fn data_image(texels: u32) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: DATA_WIDTH,
            height: texels.div_ceil(DATA_WIDTH).max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

/// Quad slots of a ribbon proxy: every segment subdivision of an emitter of
/// `max_particles` (at most [`MAX_SLOTS`]: past it, the first segments in
/// ribbon order are traced).
fn ribbon_slots(max_particles: u32, tessellation: u32) -> u32 {
    (max_particles.saturating_sub(1).max(1) * tessellation.clamp(1, 16)).min(MAX_SLOTS)
}

/// Spawns, replaces and despawns the proxies of ray-traced sprite and ribbon
/// renderers (only while a Solari camera is active),
/// and keeps their emission in step with the Solari camera's exposure.
#[allow(clippy::too_many_arguments)]
fn sync_proxies(
    mut commands: Commands,
    effects: Query<(Entity, &ParticleEffect)>,
    cameras: Query<(&Camera, Option<&Exposure>), With<SolariLighting>>,
    mut proxies: ResMut<Proxies>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ParticleRtMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let proxies = &mut *proxies;
    let solari_camera = cameras.iter().find(|(c, _)| c.is_active);
    let exposure = solari_camera
        .map(|(_, e)| e.copied().unwrap_or_default())
        .unwrap_or_default()
        .exposure();
    let radiance_scale = 1.0 / exposure.max(1e-12);
    if radiance_scale != proxies.radiance_scale {
        proxies.radiance_scale = radiance_scale;
        let ids: Vec<_> = materials.ids().collect();
        for id in ids {
            if let Some(mut m) = materials.get_mut(id) {
                m.radiance_scale = radiance_scale;
            }
        }
    }

    let mut live = HashSet::new();
    // Without an active Solari camera nothing traces rays: no proxies, no cost.
    for (effect, fx) in effects.iter().filter(|_| solari_camera.is_some()) {
        let system_ptr = Arc::as_ptr(&fx.system) as usize;
        for (ri, (ei, r)) in renderers(&fx.system).enumerate() {
            let traced = match r {
                CookedRenderer::Sprite(s) => s.ray_tracing,
                CookedRenderer::Ribbon(r) => r.ray_tracing,
                _ => false,
            };
            if !traced {
                continue;
            }
            live.insert((effect, ri));
            match proxies.by_renderer.get(&(effect, ri)) {
                Some(&(_, ptr)) if ptr == system_ptr => continue,
                Some(&(old, _)) => commands.entity(old).despawn(),
                None => {}
            }
            let max_particles = fx.system.emitters[ei].max_particles;
            let (slots, data, material, settings) = match r {
                CookedRenderer::Sprite(s) => {
                    let slots = max_particles.clamp(1, MAX_SLOTS);
                    let data = images.add(data_image(SPRITE_TEXELS * slots));
                    let sprite = match &s.texture {
                        Some(path) => asset_server.load::<Image>(path.clone()),
                        None => {
                            let key = (s.sub_image_size[0].max(1), s.sub_image_size[1].max(1));
                            proxies
                                .builtin_sprites
                                .entry(key)
                                .or_insert_with(|| images.add(fx_render::builtin_sprite_image(key.0, key.1)))
                                .clone()
                        }
                    };
                    let material = ParticleRtMaterial {
                        sprite: Some(sprite),
                        data: data.clone(),
                        emissive: !s.lit || s.blend_mode == BlendMode::Additive,
                        additive: s.blend_mode == BlendMode::Additive,
                        sub_image: s.sub_image_size,
                        sub_uv_blending: s.sub_uv_blending,
                        radiance_scale,
                        ribbon: false,
                    };
                    (slots, data, material, ProxyRenderer::Sprite(s.clone()))
                }
                CookedRenderer::Ribbon(rb) => {
                    let slots = ribbon_slots(max_particles, rb.tessellation_factor);
                    let data = images.add(data_image(RIBBON_TEXELS * slots));
                    let material = ParticleRtMaterial {
                        sprite: rb.texture.as_ref().map(|path| asset_server.load::<Image>(path.clone())),
                        data: data.clone(),
                        // Ribbons are unlit: they show (and emit) their colour.
                        emissive: true,
                        additive: rb.blend_mode == BlendMode::Additive,
                        sub_image: [1, 1],
                        sub_uv_blending: false,
                        radiance_scale,
                        ribbon: true,
                    };
                    (slots, data, material, ProxyRenderer::Ribbon(rb.clone()))
                }
                _ => unreachable!(),
            };
            let material = materials.add(material);
            let proxy = commands
                .spawn((
                    Name::new(format!("fx ray-tracing proxy {ri}")),
                    ParticleRtProxy {
                        effect,
                        renderer: ri,
                        emitter: ei,
                        slots,
                        data,
                        settings,
                    },
                    RaytracingMesh3d(meshes.add(proxy_mesh(slots))),
                    MeshMaterial3d(material),
                    // Particles are written in world space.
                    Transform::IDENTITY,
                    // Makes Solari re-read the mesh's vertices and rebuild its
                    // BLAS every frame (its deform path; no morph targets are
                    // used: the entity has no visibility, so Bevy's morph
                    // extraction skips it).
                    MeshMorphWeights::Value { weights: Vec::new() },
                ))
                .id();
            proxies.by_renderer.insert((effect, ri), (proxy, system_ptr));
        }
    }
    proxies.by_renderer.retain(|key, (proxy, _)| {
        let keep = live.contains(key);
        if !keep {
            commands.entity(*proxy).despawn();
        }
        keep
    });
}

// -------------------------------------------------------------- render world

struct ExtractedProxy {
    render: Entity,
    effect: Entity,
    renderer: usize,
    emitter: usize,
    slots: u32,
    mesh: AssetId<Mesh>,
    data: AssetId<Image>,
    settings: ProxyRenderer,
}

#[derive(Resource, Default)]
struct ExtractedProxies(Vec<ExtractedProxy>);

fn extract_proxies(
    mut out: ResMut<ExtractedProxies>,
    proxies: Extract<Query<(RenderEntity, &ParticleRtProxy, &RaytracingMesh3d)>>,
) {
    out.0.clear();
    for (render, p, mesh) in &proxies {
        out.0.push(ExtractedProxy {
            render,
            effect: p.effect,
            renderer: p.renderer,
            emitter: p.emitter,
            slots: p.slots,
            mesh: mesh.id(),
            data: p.data.id(),
            settings: p.settings.clone(),
        });
    }
}

struct ProxyGpu {
    /// The instance's first particle copy in its batch (a new instance rebuilds the proxy).
    particles: fx_gpu::Region,
    /// Sprites: every living particle, compacted. Ribbons use fx_render's sort.
    draw: Option<ViewDraw>,
    uniform: wgpu::Buffer,
    data_view: Option<(AssetId<Image>, TextureView)>,
    scratch: [u8; PARAMS_BYTES],
    seen: bool,
}

#[derive(Resource)]
struct ProxyState {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    ribbon_layout: BindGroupLayoutDescriptor,
    ribbon_pipeline: CachedComputePipelineId,
    entries: HashMap<Entity, ProxyGpu>,
    /// The camera sprites face: position, right, up, back (world).
    camera: [Vec4; 4],
}

fn init_proxy_state(mut commands: Commands, asset_server: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "fx rt proxy",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("fx rt proxy".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "proxy.wesl"),
        entry_point: Some("proxy".into()),
        ..default()
    });
    let ribbon_layout = BindGroupLayoutDescriptor::new(
        "fx rt ribbon proxy",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let ribbon_pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("fx rt ribbon proxy".into()),
        layout: vec![ribbon_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "ribbon_proxy.wesl"),
        entry_point: Some("ribbon_proxy".into()),
        ..default()
    });
    commands.insert_resource(ProxyState {
        layout,
        pipeline,
        ribbon_layout,
        ribbon_pipeline,
        entries: HashMap::default(),
        camera: [Vec4::ZERO, Vec4::X, Vec4::Y, Vec4::Z],
    });
}

/// Per proxy: the uniform and, for sprites, the cull (no frustum: every
/// living particle, compacted); the camera sprites face this frame.
fn prepare_proxies(
    mut state: ResMut<ProxyState>,
    extracted: Res<ExtractedProxies>,
    fx: Res<FxState>,
    device: Res<RenderDevice>,
    solari_views: Query<&ExtractedView, With<SolariLighting>>,
    views: Query<&ExtractedView>,
) {
    let state = &mut *state;
    if let Some(view) = solari_views.iter().next().or_else(|| views.iter().next()) {
        let m = view.world_from_view.to_matrix();
        state.camera = [m.w_axis, m.x_axis, m.y_axis, m.z_axis];
    }
    for e in state.entries.values_mut() {
        e.seen = false;
    }
    for p in &extracted.0 {
        let Some((gpu, _)) = fx.effect(p.effect) else { continue };
        let Some(em) = gpu.emitters.get(p.emitter) else { continue };
        let fresh = state
            .entries
            .get(&p.render)
            .is_some_and(|e| e.particles == em.particles[0]);
        if !fresh {
            let wd = device.wgpu_device();
            state.entries.insert(
                p.render,
                ProxyGpu {
                    particles: em.particles[0].clone(),
                    draw: matches!(p.settings, ProxyRenderer::Sprite(_)).then(|| {
                        ViewDraw::with_config(
                            wd,
                            fx.view_pipelines(),
                            em,
                            ViewDrawConfig {
                                sort_mode: SortMode::None,
                                vertex_count: 6,
                                cull: CullRadius::Off,
                            },
                        )
                    }),
                    uniform: wd.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fx rt proxy params"),
                        size: PARAMS_BYTES as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    data_view: None,
                    scratch: [0; PARAMS_BYTES],
                    seen: true,
                },
            );
        }
        state.entries.get_mut(&p.render).unwrap().seen = true;
    }
    state.entries.retain(|_, e| e.seen);
}

fn facing_index(f: FacingMode) -> u32 {
    match f {
        FacingMode::FaceCamera => 0,
        FacingMode::FaceCameraPlane => 1,
        FacingMode::FaceCameraPosition => 2,
        FacingMode::CustomFacingVector => 3,
    }
}

fn alignment_index(a: SpriteAlignment) -> u32 {
    match a {
        SpriteAlignment::Unaligned => 0,
        SpriteAlignment::VelocityAligned => 1,
        SpriteAlignment::CustomAlignment => 2,
    }
}

/// Word offsets of the normal and tangent in a mesh's interleaved vertices,
/// and words per vertex, if positions and normals are plain `Float32x3`.
fn vertex_layout(mesh: &RenderMesh) -> Option<(u32, u32, u32)> {
    let layout = mesh.layout.0.layout();
    if !layout.array_stride.is_multiple_of(4) {
        return None;
    }
    let (mut pos, mut normal, mut tangent) = (None, None, u32::MAX);
    for (id, a) in mesh.layout.0.attribute_ids().iter().zip(&layout.attributes) {
        if *id == Mesh::ATTRIBUTE_POSITION.id && a.format == VertexFormat::Float32x3 {
            pos = Some(a.offset as u32 / 4);
        } else if *id == Mesh::ATTRIBUTE_NORMAL.id && a.format == VertexFormat::Float32x3 {
            normal = Some(a.offset as u32 / 4);
        } else if *id == Mesh::ATTRIBUTE_TANGENT.id && a.format == VertexFormat::Float32x4 {
            tangent = a.offset as u32 / 4;
        }
    }
    (pos? == 0).then_some((normal?, tangent, layout.array_stride as u32 / 4))
}

/// Writes every proxy's geometry and particle data from this frame's
/// particles (after the simulation, before Solari's deform pass and TLAS
/// build, which run in `RenderGraphSystems::Begin`).
#[allow(clippy::too_many_arguments)]
fn write_proxies(
    mut ctx: RenderContext,
    queue: Res<RenderQueue>,
    fx: Res<FxState>,
    mut state: ResMut<ProxyState>,
    extracted: Res<ExtractedProxies>,
    mesh_allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    if extracted.0.is_empty() {
        return;
    }
    let (Some(pipeline), Some(ribbon_pipeline)) = (
        pipeline_cache.get_compute_pipeline(state.pipeline),
        pipeline_cache.get_compute_pipeline(state.ribbon_pipeline),
    ) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&state.layout);
    let ribbon_layout = pipeline_cache.get_bind_group_layout(&state.ribbon_layout);
    let state = &mut *state;
    let camera = state.camera;
    let recorder = ctx.diagnostic_recorder();
    let diagnostics = recorder.as_deref();
    let encoder = ctx.command_encoder();
    let span = diagnostics.time_span(encoder, "fx_rt_proxy");
    for p in &extracted.0 {
        let (Some(entry), Some((gpu, world_from_local))) =
            (state.entries.get_mut(&p.render), fx.effect(p.effect))
        else {
            continue;
        };
        let Some(em) = gpu.emitters.get(p.emitter) else { continue };
        let (Some(slice), Some(render_mesh), Some(image)) = (
            mesh_allocator.mesh_vertex_slice(&p.mesh),
            render_meshes.get(p.mesh),
            gpu_images.get(p.data),
        ) else {
            continue;
        };
        let Some((normal_offset, tangent_offset, stride)) = vertex_layout(render_mesh) else {
            continue;
        };
        if entry.data_view.as_ref().is_none_or(|(id, _)| *id != p.data) {
            entry.data_view = Some((p.data, image.texture.create_view(&TextureViewDescriptor::default())));
        }
        let Some((_, data_view)) = &entry.data_view else { continue };

        let mut words = [0u32; 52];
        let mut f = |i: usize, v: f32| words[i] = v.to_bits();
        let m: Mat4 = world_from_local;
        for (i, v) in m.to_cols_array().iter().enumerate() {
            f(i, *v);
        }
        for (j, c) in camera.iter().enumerate() {
            for (k, v) in c.to_array().iter().enumerate() {
                f(16 + 4 * j + k, *v);
            }
        }
        let texture = |binding: u32| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(data_view),
        };
        fn buffer(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let (bind_group, pipeline, threads) = match &p.settings {
            ProxyRenderer::Sprite(s) => {
                let Some(draw) = &mut entry.draw else { continue };
                // Uniform (see `ProxyParams` in proxy.wesl).
                for k in 0..3 {
                    f(32 + k, s.custom_facing_vector[k]);
                    f(36 + k, s.custom_alignment_vector[k]);
                }
                f(40, s.pivot_in_uv_space[0]);
                f(41, s.pivot_in_uv_space[1]);
                words[42] = em.capacity;
                words[43] = p.slots.min(em.capacity);
                words[44] = facing_index(s.facing_mode);
                words[45] = alignment_index(s.alignment);
                words[46] = slice.range.start * stride;
                words[47] = stride;
                words[48] = normal_offset;
                words[49] = tangent_offset;
                words[50] = DATA_WIDTH;
                words[51] = (em.words > fx_gpu::WORDS_PER_PARTICLE) as u32;
                draw.record(&queue, encoder, fx.view_pipelines(), em, &NO_VIEW);
                let bind_group = device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fx rt proxy"),
                    layout: &layout,
                    entries: &[
                        buffer(0, &entry.uniform),
                        wgpu::BindGroupEntry { binding: 1, resource: em.particles[em.current].binding() },
                        buffer(2, &draw.indices),
                        buffer(3, &draw.draw_args),
                        buffer(4, &slice.buffer),
                        texture(5),
                    ],
                });
                (bind_group, pipeline, words[43])
            }
            ProxyRenderer::Ribbon(r) => {
                // This frame's sort and ribbon build, recorded by fx_render.
                let Some(rb) = fx.ribbon(p.effect, p.renderer) else { continue };
                // Uniform (see `RibbonProxyParams` in ribbon_proxy.wesl).
                for k in 0..3 {
                    f(32 + k, r.custom_facing_vector[k]);
                }
                words[36] = em.capacity;
                words[37] = p.slots;
                words[38] = r.tessellation_factor.clamp(1, 16);
                words[39] = match r.facing_mode {
                    RibbonFacingMode::Screen => 0,
                    RibbonFacingMode::CustomFacingVector => 1,
                };
                words[40] = slice.range.start * stride;
                words[41] = stride;
                words[42] = normal_offset;
                words[43] = tangent_offset;
                words[44] = DATA_WIDTH;
                words[45] = match r.uv_mode {
                    RibbonUvMode::ScaledUniformly => 0,
                    RibbonUvMode::ScaledUsingRibbonSegmentLength => 1,
                    RibbonUvMode::TiledOverRibbonLength => 2,
                };
                // As fx_render's ribbon uniform: spawn order runs U from the newest end.
                words[46] = (r.link_order == RibbonLinkOrderMode::SpawnOrder) as u32;
                words[47] = r.tiling_length.max(1e-4).to_bits();
                let bind_group = device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fx rt ribbon proxy"),
                    layout: &ribbon_layout,
                    entries: &[
                        buffer(0, &entry.uniform),
                        wgpu::BindGroupEntry { binding: 1, resource: em.particles[em.current].binding() },
                        buffer(2, rb.indices),
                        buffer(3, rb.dist),
                        buffer(4, rb.ribbons),
                        buffer(5, rb.meta),
                        buffer(6, &slice.buffer),
                        texture(7),
                    ],
                });
                (bind_group, ribbon_pipeline, p.slots)
            }
        };
        entry.scratch.copy_from_slice(bytemuck::cast_slice(&words));
        queue.write_buffer(&entry.uniform, 0, &entry.scratch);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx rt proxy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
    }
    span.end(encoder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    #[test]
    fn proxy_mesh_has_two_quads_per_slot_and_slot_uvs() {
        let mesh = proxy_mesh(3);
        assert_eq!(mesh.count_vertices(), 24);
        let Some(Indices::U32(indices)) = mesh.indices() else { panic!("u32 indices") };
        assert_eq!(indices.len(), 3 * 12);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
            panic!("uvs")
        };
        // Slot k's vertices have u in [2k, 2k + 1]: floor(u / 2) = k in the hit function.
        for (v, uv) in uvs.iter().enumerate() {
            assert_eq!((uv[0] / 2.0).floor() as usize, v / 8);
        }
        // Each triangle stays within one quad.
        for tri in indices.chunks(3) {
            assert!(tri.iter().all(|&i| i / 4 == tri[0] / 4));
        }
        assert_eq!(mesh.raytracing, MeshRaytracingFlags::NON_OPAQUE);
    }

    #[test]
    fn data_texture_holds_two_texels_per_slot() {
        let image = data_image(RIBBON_TEXELS * MAX_SLOTS);
        let size = image.texture_descriptor.size;
        assert!(size.width * size.height >= RIBBON_TEXELS * MAX_SLOTS);
        assert!(image.texture_descriptor.usage.contains(TextureUsages::STORAGE_BINDING));
        // Solari indexes an emissive mesh's triangles with 16 bits.
        assert!(4 * MAX_SLOTS <= u16::MAX as u32);
    }

    #[test]
    fn ribbon_slots_cover_every_subsegment_up_to_the_cap() {
        assert_eq!(ribbon_slots(64, 4), 63 * 4);
        assert_eq!(ribbon_slots(1, 1), 1);
        assert_eq!(ribbon_slots(0, 0), 1);
        assert_eq!(ribbon_slots(100_000, 16), MAX_SLOTS);
    }
}

/// Gives fx_render Solari's scene bindings, so Collision modules of type
/// RayTraced trace rays against the scene (its previous frame's TLAS).
fn share_ray_tracing_scene(
    bindings: Option<Res<bevy::solari::scene::RaytracingSceneBindings>>,
    cache: Res<bevy::render::render_resource::PipelineCache>,
    mut rt: ResMut<fx_render::FxRayTracingScene>,
) {
    let Some(b) = bindings else { return };
    if rt.layout.is_none() {
        let layout = cache.get_bind_group_layout(&b.bind_group_layout);
        rt.layout = Some((*layout).clone());
    }
    rt.bind_group = b.bind_group.as_ref().map(|g| (**g).clone());
}
