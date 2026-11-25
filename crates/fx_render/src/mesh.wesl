// Niagara-style mesh renderer: the renderer's mesh drawn once per visible
// particle with one indexed indirect draw (instance count from the GPU
// cull). Vertices are pulled from a storage buffer (position, normal, uv).

import bevy_pbr::render::mesh_view_bindings::view;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_color, particle_dynamic_param, set_sub_uv,
    safe_normalize, sample_particle_texture, scene_light, finish, camera_offset,
};

// 8 floats per vertex: position, normal, uv.
@group(1) @binding(5) var<storage, read> vertices: array<f32>;

// q * v * q⁻¹ for unit quaternion q = (x, y, z, w).
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Rotation taking the mesh's X axis to `x` (world), keeping Y as close to
// world up as it can.
fn facing_basis(x: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(x, up)) > 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let z = normalize(cross(x, up));
    let y = cross(z, x);
    return mat3x3<f32>(x, y, z);
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxVertexOutput {
    let i = indices[ii];
    let m = fx.world_from_local;
    let b = vi * 8u;
    let p = vec3<f32>(vertices[b], vertices[b + 1u], vertices[b + 2u]);
    let nrm = vec3<f32>(vertices[b + 3u], vertices[b + 4u], vertices[b + 5u]);
    let uv = vec2<f32>(vertices[b + 6u], vertices[b + 7u]);

    let scale = vec3<f32>(pw(28u, i), pw(29u, i), pw(30u, i));
    let q = vec4<f32>(pw(31u, i), pw(32u, i), pw(33u, i), pw(34u, i));
    let center = camera_offset((m * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz, i);
    var offset = rotate(q, p * scale);
    // Normals transform by the inverse scale.
    var normal = rotate(q, nrm / select(scale, vec3<f32>(1.0), scale == vec3<f32>(0.0)));

    // Facing: Default keeps MeshOrientation in the effect's space; the others
    // turn the oriented mesh's X axis to velocity or the camera.
    var x = vec3<f32>(0.0);
    switch fx.facing_mode {
        case 1u: { x = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz; }
        case 2u: { x = view.world_position - center; }
        case 3u: { x = -view.world_from_view[2].xyz; }
        default: {}
    }
    if (fx.facing_mode != 0u && length(x) > 1e-6) {
        let f = facing_basis(normalize(x));
        offset = f * offset;
        normal = f * normal;
    } else {
        offset = (m * vec4<f32>(offset, 0.0)).xyz;
        normal = (m * vec4<f32>(normal, 0.0)).xyz;
    }
    let world = center + offset;

    var out: FxVertexOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.normal = safe_normalize(normal, vec3<f32>(0.0, 1.0, 0.0));
    out.color = particle_color(i);
    out.dynamic_param = particle_dynamic_param(i);
    set_sub_uv(&out, pw(16u, i), uv);
    return out;
}

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    var tex = vec4<f32>(1.0);
    @if(TEXTURED) {
        tex = sample_particle_texture(in);
    }
    var rgb = in.color.rgb * tex.rgb;
    @if(LIT) {
        rgb *= scene_light(normalize(in.normal), false);
    }
    return finish(rgb, in.color.a * tex.a);
}
