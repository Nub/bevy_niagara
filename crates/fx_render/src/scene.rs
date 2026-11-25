//! What scene collisions read: a copy of a camera's depth buffer
//! (SceneDepth) and the ray-tracing scene (RayTraced, filled in by fx_solari).
//!
//! After the cameras render, the depth of the first 3D camera (lowest
//! `order`) is copied to an `R32Float` texture with its view, so the next
//! frame's simulation (which runs before the cameras) collides with what
//! the camera saw. The camera's depth texture needs `TEXTURE_BINDING`,
//! which [`request_depth_binding`] adds while an effect collides with the
//! scene.

use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::render_resource::TextureUsages;
use bevy::render::view::{ExtractedView, ViewDepthStencilTexture};
use fx_gpu::wgpu;

use crate::{FxState, ParticleEffect};

/// Main world: adds `TEXTURE_BINDING` to every 3D camera's depth texture
/// while an effect has SceneDepth or RayTraced collisions.
pub(crate) fn request_depth_binding(
    effects: Query<&ParticleEffect>,
    mut cameras: Query<&mut Camera3d>,
) {
    let wanted = effects
        .iter()
        .any(|e| e.system.emitters.iter().any(|em| em.uses_scene_collision()));
    if !wanted {
        return;
    }
    let bit = TextureUsages::TEXTURE_BINDING.bits();
    for mut cam in &mut cameras {
        if cam.depth_texture_usages.0 & bit == 0 {
            cam.depth_texture_usages.0 |= bit;
        }
    }
}

/// The ray-tracing scene RayTraced collisions trace rays in (render world).
/// fx_solari fills it with Solari's scene bindings; without it, RayTraced
/// collisions use the depth buffer.
#[derive(Resource, Default)]
pub struct FxRayTracingScene {
    /// Layout of the bind group (Solari's scene bindings).
    pub layout: Option<wgpu::BindGroupLayout>,
    /// This frame's bind group, once Solari has built it.
    pub bind_group: Option<wgpu::BindGroup>,
}

/// Render world: the depth copy SceneDepth collisions read.
#[derive(Resource, Default)]
pub struct FxSceneDepth {
    /// `[single-sampled, multisampled]` copy pipelines, made on first use.
    pipelines: [Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>; 2],
    texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Bumped when the copy texture is replaced (effects rebind it).
    pub(crate) generation: u64,
    /// The camera the copy shows; `None` until one was copied.
    pub(crate) view: Option<fx_gpu::SceneView>,
}

impl FxSceneDepth {
    pub(crate) fn texture_view(&self) -> Option<&wgpu::TextureView> {
        self.texture.as_ref().map(|(_, v)| v)
    }
}

const COPY: &str = r#"
@group(0) @binding(0) var depth: DEPTH;
@group(0) @binding(1) var out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out);
    if (any(id.xy >= size)) {
        return;
    }
    let d = textureLoad(depth, vec2<i32>(id.xy), 0);
    textureStore(out, vec2<i32>(id.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}
"#;

fn copy_pipeline(device: &wgpu::Device, multisampled: bool) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let ty = if multisampled {
        "texture_depth_multisampled_2d"
    } else {
        "texture_depth_2d"
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fx scene depth copy"),
        source: wgpu::ShaderSource::Wgsl(COPY.replace("DEPTH", ty).into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fx scene depth copy"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: fx_gpu::SCENE_DEPTH_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fx scene depth copy"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fx scene depth copy"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}

/// After the cameras: copies the first 3D camera's depth for the next
/// frame's scene collisions (only while an effect uses them).
pub(crate) fn capture_scene_depth(
    mut ctx: RenderContext,
    device: Res<RenderDevice>,
    mut scene: ResMut<FxSceneDepth>,
    state: Option<Res<FxState>>,
    views: Query<(&ExtractedCamera, &ExtractedView, &ViewDepthStencilTexture)>,
) {
    let Some(state) = state else { return };
    if !state.effects.values().any(|i| i.gpu.uses_scene()) {
        return;
    }
    let Some((_, view, depth)) = views
        .iter()
        .filter(|(c, _, d)| {
            c.physical_target_size.is_some()
                && d.texture().usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        })
        .min_by_key(|(c, _, _)| c.order)
    else {
        scene.view = None;
        return;
    };
    let wd = device.wgpu_device();
    let src: &wgpu::Texture = depth.texture();
    let size = src.size();
    let resized = scene
        .texture
        .as_ref()
        .is_none_or(|(t, _)| t.width() != size.width || t.height() != size.height);
    if resized {
        let tex = wd.create_texture(&wgpu::TextureDescriptor {
            label: Some("fx scene depth"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fx_gpu::SCENE_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let v = tex.create_view(&Default::default());
        scene.texture = Some((tex, v));
        scene.generation += 1;
    }
    let ms = src.sample_count() > 1;
    let (pipeline, layout) = scene.pipelines[ms as usize]
        .get_or_insert_with(|| copy_pipeline(wd, ms))
        .clone();
    let src_view = src.create_view(&wgpu::TextureViewDescriptor {
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    let (_, out) = scene.texture.as_ref().unwrap();
    let bg = wd.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fx scene depth copy"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(out),
            },
        ],
    });
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("fx scene depth copy"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
    drop(pass);
    let clip_from_world = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
    scene.view = Some(fx_gpu::SceneView {
        clip_from_world,
        camera_position: view.world_from_view.translation(),
        size: [size.width, size.height],
    });
}
