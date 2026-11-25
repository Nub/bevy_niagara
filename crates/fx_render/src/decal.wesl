// Niagara-style decal renderer: per visible particle, a box (36 vertices, no
// vertex buffer) whose back faces are drawn; each pixel reconstructs the
// scene surface from the depth prepass and, when it lies inside the box,
// shows the texture projected along the box's Y axis (UV = box X, Z).
//
// Box: centre Particles.Position, size = renderer size * Particles.Scale,
// turned by Particles.MeshOrientation. Fades out from fade_start to the end
// of Particles.NormalizedAge, and near the box's top and bottom.

import bevy_pbr::render::mesh_view_bindings::view;
import bevy_pbr::prepass::utils;
import fx_render::particle_io::{
    fx, indices, pw, particle_world_from_local, particle_color, particle_texture, particle_sampler,
    scene_light, finish_faded, safe_normalize,
};

struct DecalOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) center: vec3<f32>,
    // Box edges (world, full length) along its X, Y (projection) and Z axes.
    @location(2) @interpolate(flat) ax: vec3<f32>,
    @location(3) @interpolate(flat) ay: vec3<f32>,
    @location(4) @interpolate(flat) az: vec3<f32>,
    @location(5) @interpolate(flat) normalized_age: f32,
}

// q * v * q⁻¹ for unit quaternion q = (x, y, z, w).
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Corner of triangle vertex `vi` of a unit cube centred on 0, faces outward (CCW).
fn cube_corner(vi: u32) -> vec3<f32> {
    // 6 faces × 2 triangles; per face: axis, sign.
    let face = vi / 6u;
    let k = vi % 6u;
    // Quad corners (u, v) in triangle order.
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5),
    );
    let c = uv[k];
    let s = select(-0.5, 0.5, face % 2u == 0u);
    // Swapping u and v on negative faces keeps the winding outward.
    let a = select(c, c.yx, face % 2u == 1u);
    switch face / 2u {
        case 0u: { return vec3<f32>(s, a.x, a.y); }
        case 1u: { return vec3<f32>(a.y, s, a.x); }
        default: { return vec3<f32>(a.x, a.y, s); }
    }
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DecalOutput {
    let i = indices[ii];
    let m = particle_world_from_local(i);
    let center = (m * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz;
    let size = fx.custom_facing.xyz * vec3<f32>(pw(28u, i), pw(29u, i), pw(30u, i));
    var q = vec4<f32>(pw(31u, i), pw(32u, i), pw(33u, i), pw(34u, i));
    if (dot(q, q) < 1e-12) {
        q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let ax = (m * vec4<f32>(rotate(q, vec3<f32>(size.x, 0.0, 0.0)), 0.0)).xyz;
    let ay = (m * vec4<f32>(rotate(q, vec3<f32>(0.0, size.y, 0.0)), 0.0)).xyz;
    let az = (m * vec4<f32>(rotate(q, vec3<f32>(0.0, 0.0, size.z)), 0.0)).xyz;
    let c = cube_corner(vi);
    let world = center + ax * c.x + ay * c.y + az * c.z;

    var out: DecalOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.color = particle_color(i);
    out.center = center;
    out.ax = ax;
    out.ay = ay;
    out.az = az;
    out.normalized_age = pw(16u, i);
    return out;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: DecalOutput,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;
    var depth = 0.0;
    @if(DEPTH_PREPASS) {
        depth = utils::prepass_depth(in.position, sample_index);
    }
    if (depth <= 0.0) {
        // Sky (or no depth prepass): no surface to project onto.
        discard;
    }
    let uv = (in.position.xy - view.viewport.xy) / view.viewport.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = view.world_from_clip * ndc;
    let surface = w.xyz / w.w;
    let d = surface - in.center;
    let local = vec3<f32>(
        dot(d, in.ax) / max(dot(in.ax, in.ax), 1e-12),
        dot(d, in.ay) / max(dot(in.ay, in.ay), 1e-12),
        dot(d, in.az) / max(dot(in.az, in.az), 1e-12),
    );
    if (any(abs(local) > vec3<f32>(0.5))) {
        discard;
    }
    let tex = textureSampleLevel(particle_texture, particle_sampler, vec2<f32>(local.x + 0.5, local.z + 0.5), 0.0);
    var rgb = in.color.rgb * tex.rgb;
    @if(LIT) {
        rgb *= scene_light(safe_normalize(in.ay, vec3<f32>(0.0, 1.0, 0.0)), false);
    }
    // fade_distance carries the renderer's fade_start.
    let start = fx.fade_distance;
    let age_fade = select(1.0, 1.0 - smoothstep(start, 1.0, in.normalized_age), start < 1.0);
    let edge_fade = 1.0 - smoothstep(0.35, 0.5, abs(local.y));
    return finish_faded(rgb, in.color.a * tex.a, age_fade * edge_fade);
}
