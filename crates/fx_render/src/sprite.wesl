// Niagara-style sprite renderer: one instanced quad per visible particle,
// read straight from fx_gpu's structure-of-arrays particle buffer through
// the per-view (culled, optionally sorted) index list.

import bevy_pbr::render::mesh_view_bindings::view;
import bevy_pbr::prepass::utils;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_world_from_local, particle_color, particle_dynamic_param, particle_dynamic_param_1, particle_dynamic_param_2, particle_dynamic_param_3, set_sprite_sub_uv,
    particle_custom_facing, particle_custom_alignment,
    sample_particle_texture, scene_light, finish_faded, camera_offset,
};
import fx_render::particle_surface::{sprite_corner, sprite_texel};

// Sorted sprites of one instance: the cull's contiguous copy of what this
// shader reads (fx_gpu::view::BatchDraw::payload), indexed by slot. Sprites
// leave binding 5 (mesh vertices) free otherwise.
@group(1) @binding(5) var<storage, read> payload: array<vec4<f32>>;

struct SpriteAttrs {
    local: vec3<f32>,
    nage: f32,
    vel: vec3<f32>,
    rotation: f32,
    size: vec2<f32>,
    color: vec4<f32>,
}

fn sprite_attrs(i: u32) -> SpriteAttrs {
    var a: SpriteAttrs;
    if (fx.payload != 0u) {
        let p = (i % fx.capacity) * 4u;
        let v0 = payload[p];
        let v1 = payload[p + 1u];
        let v2 = payload[p + 2u];
        let v3 = payload[p + 3u];
        a.local = v0.xyz;
        a.nage = v0.w;
        a.vel = v1.xyz;
        a.rotation = v1.w;
        a.size = v2.xy;
        a.color = vec4<f32>(v2.zw, v3.xy);
        return a;
    }
    a.local = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
    a.nage = pw(16u, i);
    a.vel = vec3<f32>(pw(3u, i), pw(4u, i), pw(5u, i));
    a.rotation = pw(12u, i);
    a.size = vec2<f32>(pw(10u, i), pw(11u, i));
    a.color = particle_color(i);
    return a;
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxVertexOutput {
    let i = indices[ii];
    let m = particle_world_from_local(i);
    let a = sprite_attrs(i);
    let pos = camera_offset((m * vec4<f32>(a.local, 1.0)).xyz, i);
    let vel = (m * vec4<f32>(a.vel, 0.0)).xyz;
    let size = a.size;
    let rotation = radians(a.rotation);
    let nage = a.nage;

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
    out.color = a.color;
    out.normal = corner.normal;
    out.world_position = corner.world;
    out.dynamic_param = particle_dynamic_param(i);
    out.dynamic_param_1 = particle_dynamic_param_1(i);
    out.dynamic_param_2 = particle_dynamic_param_2(i);
    out.dynamic_param_3 = particle_dynamic_param_3(i);
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
