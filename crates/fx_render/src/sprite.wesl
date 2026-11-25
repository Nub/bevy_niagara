// Niagara-style sprite renderer: one instanced quad per visible particle,
// read straight from fx_gpu's structure-of-arrays particle buffer through
// the per-view (culled, optionally sorted) index list.

import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::prepass::utils;

struct Sprite {
    world_from_local: mat4x4<f32>,
    custom_facing: vec4<f32>,
    custom_alignment: vec4<f32>,
    pivot: vec2<f32>,
    sub_image: vec2<f32>,       // columns, rows
    capacity: u32,
    facing_mode: u32,           // 0 FaceCamera, 1 FaceCameraPlane, 2 FaceCameraPosition, 3 CustomFacingVector
    alignment: u32,             // 0 Unaligned, 1 VelocityAligned, 2 CustomAlignment
    sub_uv_blending: u32,
    fade_distance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0) var<uniform> sprite: Sprite;
@group(1) @binding(1) var<storage, read> particles: array<f32>;
@group(1) @binding(2) var<storage, read> indices: array<u32>;
@group(1) @binding(3) var sprite_texture: texture_2d<f32>;
@group(1) @binding(4) var sprite_sampler: sampler;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * sprite.capacity + i];
}

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv0: vec2<f32>,
    @location(2) uv1: vec2<f32>,
    @location(3) blend: f32,
    @location(4) normal: vec3<f32>,
}

// Unit vector, or `fallback` when `v` is (nearly) zero.
fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(fallback, v / l, l > 1e-6);
}

fn cell_uv(frame: f32, uv: vec2<f32>) -> vec2<f32> {
    let cols = sprite.sub_image.x;
    let col = frame % cols;
    let row = floor(frame / cols);
    return (vec2<f32>(col, row) + uv) / sprite.sub_image;
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let i = indices[ii];
    let m = sprite.world_from_local;
    let local = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
    let pos = (m * vec4<f32>(local, 1.0)).xyz;
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
    let color = vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i));
    let size = vec2<f32>(pw(10u, i), pw(11u, i));
    let rotation = radians(pw(12u, i));
    let nage = pw(16u, i);

    // Facing: the sprite plane's normal, pointing at the viewer.
    let cam_right = view.world_from_view[0].xyz;
    let cam_up = view.world_from_view[1].xyz;
    let cam_back = view.world_from_view[2].xyz;
    // FaceCamera faces the camera position keeping the camera's roll;
    // FaceCameraPlane faces the view plane; FaceCameraPosition faces the
    // camera position with world up; CustomFacingVector faces a local vector.
    var n = cam_back;
    var up_ref = cam_up;
    switch sprite.facing_mode {
        case 0u: { n = safe_normalize(view.world_position - pos, cam_back); }
        case 2u: {
            n = safe_normalize(view.world_position - pos, cam_back);
            up_ref = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 3u: { n = safe_normalize((m * vec4<f32>(sprite.custom_facing.xyz, 0.0)).xyz, cam_back); }
        default: {}
    }
    // Alignment: the sprite's up axis follows velocity or a custom vector.
    if (sprite.alignment == 1u) {
        up_ref = safe_normalize(vel, up_ref);
    } else if (sprite.alignment == 2u) {
        up_ref = safe_normalize((m * vec4<f32>(sprite.custom_alignment.xyz, 0.0)).xyz, up_ref);
    }
    var right = cross(up_ref, n);
    if (length(right) < 1e-6) {
        right = cam_right;
    }
    right = normalize(right);
    let up = cross(n, right);

    // Quad corner: two triangles, uv (0,0) top left.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0),
    );
    let uv = corners[vi];
    let q = (vec2<f32>(uv.x, 1.0 - uv.y) - vec2<f32>(sprite.pivot.x, 1.0 - sprite.pivot.y)) * size;
    let c = cos(rotation);
    let s = sin(rotation);
    let r = vec2<f32>(q.x * c - q.y * s, q.x * s + q.y * c);
    let world = pos + right * r.x + up * r.y;

    // Sub-UV flipbook over the particle's life.
    let frames = sprite.sub_image.x * sprite.sub_image.y;
    let f = clamp(nage, 0.0, 0.99999) * frames;
    let f0 = floor(f);
    let f1 = min(f0 + 1.0, frames - 1.0);

    var out: VOut;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.color = color;
    out.uv0 = cell_uv(f0, uv);
    out.uv1 = cell_uv(f1, uv);
    out.blend = select(0.0, f - f0, sprite.sub_uv_blending != 0u);
    out.normal = n;
    return out;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: VOut,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;
    let t0 = textureSample(sprite_texture, sprite_sampler, in.uv0);
    let t1 = textureSample(sprite_texture, sprite_sampler, in.uv1);
    let tex = mix(t0, t1, in.blend);
    var rgb = in.color.rgb * tex.rgb;
    var a = in.color.a * tex.a;

    @if(LIT) {
        var light = lights.ambient_color.rgb;
        for (var k = 0u; k < lights.n_directional_lights; k++) {
            let l = lights.directional_lights[k];
            // Half-Lambert wrap: sprites are thin volumes, lit from both sides.
            let ndl = dot(in.normal, l.direction_to_light) * 0.5 + 0.5;
            light += l.color.rgb * ndl * (1.0 / 3.14159265);
        }
        rgb *= light * view.exposure;
    }

    var fade = 1.0;
    @if(SOFT) {
        // Depth fade against the scene (reverse-Z perspective: view z = -near / ndc).
        let scene_ndc = utils::prepass_depth(in.position, sample_index);
        let near = view.clip_from_view[3][2];
        let scene_z = near / max(scene_ndc, 1e-7);
        let frag_z = near / max(in.position.z, 1e-7);
        fade = clamp((scene_z - frag_z) / sprite.fade_distance, 0.0, 1.0);
    }

    @if(BLEND_ADDITIVE) {
        return vec4<f32>(rgb * a * fade, 0.0);
    }
    @if(BLEND_PREMULTIPLIED) {
        return vec4<f32>(rgb, a) * fade;
    }
    return vec4<f32>(rgb, a * fade);
}
