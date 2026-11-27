// Niagara-style mesh renderer: the renderer's mesh drawn once per visible
// particle with one indexed indirect draw (instance count from the GPU
// cull). Vertices are pulled from a storage buffer (position, normal, uv).

import bevy_pbr::render::mesh_view_bindings::view;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_world_from_local, particle_color, particle_dynamic_param, particle_dynamic_param_1, particle_dynamic_param_2, particle_dynamic_param_3, set_sub_uv,
    safe_normalize, sample_particle_texture, scene_light, finish, camera_offset,
};
import fx_render::particle_surface::mesh_point;

// 8 floats per vertex: position, normal, uv.
@group(1) @binding(5) var<storage, read> vertices: array<f32>;

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxVertexOutput {
    let i = indices[ii];
    let m = particle_world_from_local(i);
    let b = vi * 8u;
    let p = vec3<f32>(vertices[b], vertices[b + 1u], vertices[b + 2u]);
    let nrm = vec3<f32>(vertices[b + 3u], vertices[b + 4u], vertices[b + 5u]);
    let uv = vec2<f32>(vertices[b + 6u], vertices[b + 7u]);

    let scale = vec3<f32>(pw(28u, i), pw(29u, i), pw(30u, i));
    let q = vec4<f32>(pw(31u, i), pw(32u, i), pw(33u, i), pw(34u, i));
    let center = camera_offset((m * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz, i);
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
    // Facing: Default keeps MeshOrientation in the effect's space; the others
    // turn the oriented mesh's X axis to velocity or the camera.
    let point = mesh_point(
        m, center, vel, scale, q, p, nrm, fx.facing_mode, view.world_position, view.world_from_view[2].xyz,
    );
    let world = point.world;
    let normal = point.normal;
    var out: FxVertexOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.normal = safe_normalize(normal, vec3<f32>(0.0, 1.0, 0.0));
    out.color = particle_color(i);
    out.dynamic_param = particle_dynamic_param(i);
    out.dynamic_param_1 = particle_dynamic_param_1(i);
    out.dynamic_param_2 = particle_dynamic_param_2(i);
    out.dynamic_param_3 = particle_dynamic_param_3(i);
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
