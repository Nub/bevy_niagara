// Niagara-style sprite renderer: one instanced quad per visible particle,
// read straight from fx_gpu's structure-of-arrays particle buffer through
// the per-view (culled, optionally sorted) index list.

import bevy_pbr::render::mesh_view_bindings::view;
import bevy_pbr::prepass::utils;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_world_from_local, particle_color, particle_dynamic_param, particle_dynamic_param_1, set_sprite_sub_uv,
    particle_custom_facing, particle_custom_alignment,
    sample_particle_texture, scene_light, finish_faded, camera_offset,
};
import fx_render::particle_surface::{sprite_corner, sprite_texel};

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxVertexOutput {
    let i = indices[ii];
    let m = particle_world_from_local(i);
    let local = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
    let pos = camera_offset((m * vec4<f32>(local, 1.0)).xyz, i);
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
    let size = vec2<f32>(pw(10u, i), pw(11u, i));
    let rotation = radians(pw(12u, i));
    let nage = pw(16u, i);

    // Quad corner: two triangles, uv (0,0) top left.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0),
    );
    let uv = corners[vi];
    // Facing and alignment (shared with fx_solari's ray-traced proxies).
    let corner = sprite_corner(
        pos, vel, size, rotation, uv, fx.pivot, fx.facing_mode, fx.alignment,
        (m * vec4<f32>(particle_custom_facing(i), 0.0)).xyz,
        (m * vec4<f32>(particle_custom_alignment(i), 0.0)).xyz,
        view.world_position,
        view.world_from_view[0].xyz, view.world_from_view[1].xyz, view.world_from_view[2].xyz,
    );

    var out: FxVertexOutput;
    out.position = view.clip_from_world * vec4<f32>(corner.world, 1.0);
    out.color = particle_color(i);
    out.normal = corner.normal;
    out.world_position = corner.world;
    out.dynamic_param = particle_dynamic_param(i);
    out.dynamic_param_1 = particle_dynamic_param_1(i);
    set_sprite_sub_uv(&out, i, nage, uv);
    return out;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: FxVertexOutput,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;
    let texel = sprite_texel(in.color, sample_particle_texture(in));
    var rgb = texel.rgb;
    let a = texel.a;

    @if(LIT) {
        rgb *= scene_light(in.normal, true);
    }

    var fade = 1.0;
    @if(SOFT) {
        // Depth fade against the scene (reverse-Z perspective: view z = -near / ndc).
        let scene_ndc = utils::prepass_depth(in.position, sample_index);
        let near = view.clip_from_view[3][2];
        let scene_z = near / max(scene_ndc, 1e-7);
        let frag_z = near / max(in.position.z, 1e-7);
        fade = clamp((scene_z - frag_z) / fx.fade_distance, 0.0, 1.0);
    }
    return finish_faded(rgb, a, fade);
}
