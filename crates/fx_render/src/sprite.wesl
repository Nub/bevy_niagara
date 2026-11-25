// Niagara-style sprite renderer: one instanced quad per visible particle,
// read straight from fx_gpu's structure-of-arrays particle buffer through
// the per-view (culled, optionally sorted) index list.

import bevy_pbr::render::mesh_view_bindings::view;
import bevy_pbr::prepass::utils;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_color, particle_dynamic_param, set_sub_uv,
    safe_normalize, sample_particle_texture, scene_light, finish_faded,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxVertexOutput {
    let i = indices[ii];
    let m = fx.world_from_local;
    let local = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
    let pos = (m * vec4<f32>(local, 1.0)).xyz;
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
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
    switch fx.facing_mode {
        case 0u: { n = safe_normalize(view.world_position - pos, cam_back); }
        case 2u: {
            n = safe_normalize(view.world_position - pos, cam_back);
            up_ref = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 3u: { n = safe_normalize((m * vec4<f32>(fx.custom_facing.xyz, 0.0)).xyz, cam_back); }
        default: {}
    }
    // Alignment: the sprite's up axis follows velocity or a custom vector.
    if (fx.alignment == 1u) {
        up_ref = safe_normalize(vel, up_ref);
    } else if (fx.alignment == 2u) {
        up_ref = safe_normalize((m * vec4<f32>(fx.custom_alignment.xyz, 0.0)).xyz, up_ref);
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
    let q = (vec2<f32>(uv.x, 1.0 - uv.y) - vec2<f32>(fx.pivot.x, 1.0 - fx.pivot.y)) * size;
    let c = cos(rotation);
    let s = sin(rotation);
    let r = vec2<f32>(q.x * c - q.y * s, q.x * s + q.y * c);
    let world = pos + right * r.x + up * r.y;

    var out: FxVertexOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.color = particle_color(i);
    out.normal = n;
    out.world_position = world;
    out.dynamic_param = particle_dynamic_param(i);
    set_sub_uv(&out, nage, uv);
    return out;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: FxVertexOutput,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;
    let tex = sample_particle_texture(in);
    var rgb = in.color.rgb * tex.rgb;
    let a = in.color.a * tex.a;

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
