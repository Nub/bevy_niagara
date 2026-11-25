// Shared by every particle renderer (sprite, ribbon, mesh) and by custom
// particle materials: the renderer uniform, the particle buffer, the texture,
// the vertex → fragment interface with the particle's attributes, and helpers.
//
// A custom material is a WESL file whose `fragment` entry point takes
// `FxVertexOutput` and returns the colour through `finish` (which applies the
// renderer's blend mode):
//
//     import fx_render::particle_io::{FxVertexOutput, sample_particle_texture, finish};
//
//     @fragment
//     fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
//         let tex = sample_particle_texture(in);   // flipbook-aware
//         let heat = in.dynamic_param.x;           // Particles.DynamicMaterialParameter
//         let glow = in.dynamic_param_1.x;         // Particles.DynamicMaterialParameter1
//         return finish(in.color.rgb * tex.rgb * (1.0 + heat), in.color.a * tex.a);
//     }

import bevy_pbr::render::mesh_view_bindings::{view, lights};
import fx_render::particle_surface::{flipbook, flipbook_cell_uv, flipbook_frame};

struct FxRenderer {
    world_from_local: mat4x4<f32>,
    custom_facing: vec4<f32>,
    custom_alignment: vec4<f32>,
    pivot: vec2<f32>,
    sub_image: vec2<f32>,       // flipbook columns, rows
    capacity: u32,
    facing_mode: u32,
    alignment: u32,
    sub_uv_blending: u32,
    fade_distance: f32,
    extended: u32,              // the emitter stores Scale, MeshOrientation, Ribbon*, LightRadius, DynamicMaterialParameter
    uv_mode: u32,               // ribbon: 0 scaled uniformly, 1 by segment length, 2 tiled over length
    tessellation: u32,          // ribbon: subdivisions per segment
    tiling_length: f32,         // ribbon: metres per U for uv_mode 2
    link_from_end: u32,         // ribbon: U starts at the last sorted particle (the newest)
    mesh_radius: f32,
    sub_image_from_index: u32,  // sprite: show Particles.SubImageIndex (SubUV Animation), not the flipbook over life
    batched: u32,               // indices are batch references r * capacity + i (fx_gpu::view::BatchDraw)
    region_words: u32,          // batched: words between particle copies
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<uniform> fx: FxRenderer;
@group(1) @binding(1) var<storage, read> particles: array<f32>;
@group(1) @binding(2) var<storage, read> indices: array<u32>;
@group(1) @binding(3) var particle_texture: texture_2d<f32>;
@group(1) @binding(4) var particle_sampler: sampler;
// Batched draws: world_from_local of every batch slot.
@group(1) @binding(8) var<storage, read> transforms: array<mat4x4<f32>>;

// Word `w` of particle slot `i` (fx_gpu's structure-of-arrays layout).
fn pw(w: u32, i: u32) -> f32 {
    if (fx.batched != 0u) {
        // Batch reference r * capacity + i: copy r of the batch buffer.
        let r = i / fx.capacity;
        return particles[r * fx.region_words + w * fx.capacity + (i - r * fx.capacity)];
    }
    return particles[w * fx.capacity + i];
}

// The effect transform of particle `i` (per instance when batched).
fn particle_world_from_local(i: u32) -> mat4x4<f32> {
    if (fx.batched != 0u) {
        return transforms[i / fx.capacity / 2u];
    }
    return fx.world_from_local;
}

// What every particle renderer passes to its fragment shader.
struct FxVertexOutput {
    @builtin(position) position: vec4<f32>,
    // Particles.Color (linear RGBA).
    @location(0) color: vec4<f32>,
    // Flipbook cell UVs of the current and next frame, and the blend between them.
    @location(1) uv0: vec2<f32>,
    @location(2) uv1: vec2<f32>,
    @location(3) sub_uv_blend: f32,
    // World-space normal (towards the viewer for sprites and ribbons).
    @location(4) normal: vec3<f32>,
    @location(5) world_position: vec3<f32>,
    // Particles.DynamicMaterialParameter (0 when the emitter never sets it).
    @location(6) dynamic_param: vec4<f32>,
    @location(7) normalized_age: f32,
    // The quad, ribbon or mesh UV before flipbook mapping.
    @location(8) uv: vec2<f32>,
    // Flipbook frame (fractional when blending): over the particle's life, or
    // Particles.SubImageIndex for sprites with SubUV Animation.
    @location(9) sub_image_index: f32,
    // Particles.DynamicMaterialParameter1 (Dynamic Material Parameters 2; 0 when never set).
    @location(10) dynamic_param_1: vec4<f32>,
}

// Particle attributes for vertex shaders.
fn particle_color(i: u32) -> vec4<f32> {
    return vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i));
}

fn particle_dynamic_param(i: u32) -> vec4<f32> {
    if (fx.extended == 0u) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(pw(40u, i), pw(41u, i), pw(42u, i), pw(43u, i));
}

fn particle_dynamic_param_1(i: u32) -> vec4<f32> {
    if (fx.extended == 0u) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(pw(56u, i), pw(57u, i), pw(58u, i), pw(59u, i));
}

// Particles.SpriteFacing (words 50..52) when the emitter sets it, else the
// renderer's custom facing vector; in effect space.
fn particle_custom_facing(i: u32) -> vec3<f32> {
    if (fx.extended != 0u) {
        let v = vec3<f32>(pw(50u, i), pw(51u, i), pw(52u, i));
        if (dot(v, v) > 0.0) {
            return v;
        }
    }
    return fx.custom_facing.xyz;
}

// Particles.SpriteAlignment (words 53..55) when set, else the renderer's custom alignment vector.
fn particle_custom_alignment(i: u32) -> vec3<f32> {
    if (fx.extended != 0u) {
        let v = vec3<f32>(pw(53u, i), pw(54u, i), pw(55u, i));
        if (dot(v, v) > 0.0) {
            return v;
        }
    }
    return fx.custom_alignment.xyz;
}

// Particles.CameraOffset (word 49; 0 when the emitter never sets it).
fn particle_camera_offset(i: u32) -> f32 {
    if (fx.extended == 0u) {
        return 0.0;
    }
    return pw(49u, i);
}

// World-space particle centre `center` moved Particles.CameraOffset metres
// towards the camera (Niagara's Camera Offset: drawn in front, simulated
// where it is).
fn camera_offset(center: vec3<f32>, i: u32) -> vec3<f32> {
    let d = particle_camera_offset(i);
    if (d == 0.0) {
        return center;
    }
    let to_camera = view.world_position - center;
    let l = length(to_camera);
    if (l < 1e-6) {
        return center;
    }
    return center + to_camera * (d / l);
}

// Fills the flipbook fields of `out` for normalized age `nage` and quad UV `uv`.
fn set_sub_uv(out: ptr<function, FxVertexOutput>, nage: f32, uv: vec2<f32>) {
    let f = flipbook(nage, uv, fx.sub_image, fx.sub_uv_blending != 0u);
    (*out).uv = uv;
    (*out).uv0 = f.uv0;
    (*out).uv1 = f.uv1;
    (*out).sub_uv_blend = f.blend;
    (*out).sub_image_index = f.index;
    (*out).normalized_age = nage;
}

// Sprites: the flipbook frame is Particles.SubImageIndex when SubUV Animation
// sets it, otherwise played over the particle's life.
fn set_sprite_sub_uv(out: ptr<function, FxVertexOutput>, i: u32, nage: f32, uv: vec2<f32>) {
    if (fx.sub_image_from_index == 0u) {
        set_sub_uv(out, nage, uv);
        return;
    }
    let f = flipbook_frame(pw(44u, i), uv, fx.sub_image, fx.sub_uv_blending != 0u);
    (*out).uv = uv;
    (*out).uv0 = f.uv0;
    (*out).uv1 = f.uv1;
    (*out).sub_uv_blend = f.blend;
    (*out).sub_image_index = f.index;
    (*out).normalized_age = nage;
}

// UV of `uv` inside flipbook cell `frame`.
fn cell_uv(frame: f32, uv: vec2<f32>) -> vec2<f32> {
    return flipbook_cell_uv(frame, uv, fx.sub_image);
}

// Unit vector, or `fallback` when `v` is (nearly) zero.
fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(fallback, v / l, l > 1e-6);
}

// The renderer's texture (or the built-in sprite), flipbook frames blended.
fn sample_particle_texture(in: FxVertexOutput) -> vec4<f32> {
    let t0 = textureSample(particle_texture, particle_sampler, in.uv0);
    let t1 = textureSample(particle_texture, particle_sampler, in.uv1);
    return mix(t0, t1, in.sub_uv_blend);
}

// Ambient plus directional light at `normal`. `wrap`: half-Lambert, for thin
// volumes lit from both sides (sprites); otherwise Lambert (meshes).
fn scene_light(normal: vec3<f32>, wrap: bool) -> vec3<f32> {
    var light = lights.ambient_color.rgb;
    for (var k = 0u; k < lights.n_directional_lights; k++) {
        let l = lights.directional_lights[k];
        var ndl = dot(normal, l.direction_to_light);
        ndl = select(max(ndl, 0.0), ndl * 0.5 + 0.5, wrap);
        light += l.color.rgb * ndl * (1.0 / 3.14159265);
    }
    return light * view.exposure;
}

// Final colour for the renderer's blend mode (for Premultiplied, `rgb` is
// already multiplied by `a`).
fn finish(rgb: vec3<f32>, a: f32) -> vec4<f32> {
    return finish_faded(rgb, a, 1.0);
}

// `finish`, faded out by `fade` (soft particles).
fn finish_faded(rgb: vec3<f32>, a: f32, fade: f32) -> vec4<f32> {
    @if(BLEND_ADDITIVE) {
        return vec4<f32>(rgb * a * fade, 0.0);
    }
    @if(BLEND_PREMULTIPLIED) {
        return vec4<f32>(rgb, a) * fade;
    }
    @if(BLEND_OPAQUE) {
        return vec4<f32>(rgb, 1.0);
    }
    return vec4<f32>(rgb, a * fade);
}
