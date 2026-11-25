// Volumetric renderer: the volume's box (back faces, 36 vertices, no vertex
// buffer) ray-marches the density texture splatted by fx_gpu::volume from
// the camera (or where the ray enters the box) to where it leaves the box
// or hits the scene (depth prepass). Absorption plus single in-scattering
// of the ambient and directional lights (no shadows), plus emission;
// premultiplied output.

import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::prepass::utils;

struct Volume {
    world_from_local: mat4x4<f32>,
    local_from_world: mat4x4<f32>,
    bounds_min: vec3<f32>,
    steps: u32,
    bounds_max: vec3<f32>,
    lit: u32,
    emissive: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0) var<uniform> volume: Volume;
@group(1) @binding(1) var density: texture_3d<f32>;
@group(1) @binding(2) var density_sampler: sampler;

struct VolumeOutput {
    @builtin(position) position: vec4<f32>,
}

fn cube_corner(vi: u32) -> vec3<f32> {
    let face = vi / 6u;
    let k = vi % 6u;
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = uv[k];
    let s = select(0.0, 1.0, face % 2u == 0u);
    let a = select(c, c.yx, face % 2u == 1u);
    switch face / 2u {
        case 0u: { return vec3<f32>(s, a.x, a.y); }
        case 1u: { return vec3<f32>(a.y, s, a.x); }
        default: { return vec3<f32>(a.x, a.y, s); }
    }
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VolumeOutput {
    let local = mix(volume.bounds_min, volume.bounds_max, cube_corner(vi));
    var out: VolumeOutput;
    out.position = view.clip_from_world * (volume.world_from_local * vec4<f32>(local, 1.0));
    return out;
}

// Light scattered toward the camera per unit colour: ambient plus the
// directional lights with an isotropic phase.
fn in_scatter() -> vec3<f32> {
    var light = lights.ambient_color.rgb;
    for (var k = 0u; k < lights.n_directional_lights; k++) {
        light += lights.directional_lights[k].color.rgb * (0.5 / 3.14159265);
    }
    return light * view.exposure;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: VolumeOutput,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;
    let uv = (in.position.xy - view.viewport.xy) / view.viewport.zw;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far = view.world_from_clip * vec4<f32>(ndc_xy, in.position.z, 1.0);
    let cam = view.world_position;
    let dir = normalize(far.xyz / far.w - cam);

    // Scene hit distance along the ray (infinite for the sky or without a prepass).
    var t_scene = 1e30;
    @if(DEPTH_PREPASS) {
        let depth = utils::prepass_depth(in.position, sample_index);
        if (depth > 0.0) {
            let s = view.world_from_clip * vec4<f32>(ndc_xy, depth, 1.0);
            t_scene = dot(s.xyz / s.w - cam, dir);
        }
    }

    // Slab test in the volume's space (t stays in world metres: the map is linear).
    let o = (volume.local_from_world * vec4<f32>(cam, 1.0)).xyz;
    let d = (volume.local_from_world * vec4<f32>(dir, 0.0)).xyz;
    let inv = 1.0 / select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    let ta = (volume.bounds_min - o) * inv;
    let tb = (volume.bounds_max - o) * inv;
    let t_near = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
    let t_far = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    let t0 = max(t_near, 0.0);
    let t1 = min(t_far, t_scene);
    if (t1 <= t0) {
        discard;
    }

    let steps = f32(volume.steps);
    let dt = (t_far - t0) / steps;
    // Per-pixel jitter (interleaved gradient noise) hides step banding.
    let jitter = fract(52.9829189 * fract(dot(in.position.xy, vec2<f32>(0.06711056, 0.00583715))));
    let light = select(vec3<f32>(0.0), in_scatter(), volume.lit != 0u);
    let source = light + vec3<f32>(volume.emissive);
    let size = volume.bounds_max - volume.bounds_min;
    var transmittance = 1.0;
    var rgb = vec3<f32>(0.0);
    var t = t0 + dt * jitter;
    for (var k = 0u; k < volume.steps; k++) {
        if (t >= t1 || transmittance < 0.003) {
            break;
        }
        let step = min(dt, t1 - t);
        let p = o + d * t;
        let s = textureSampleLevel(density, density_sampler, (p - volume.bounds_min) / size, 0.0);
        let absorbed = 1.0 - exp(-s.a * step);
        rgb += transmittance * absorbed * s.rgb * source;
        transmittance *= 1.0 - absorbed;
        t += dt;
    }
    return vec4<f32>(rgb, 1.0 - transmittance);
}
