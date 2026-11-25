// Niagara-style ribbon renderer. The particles are sorted on the GPU by
// (RibbonID, link order) into `indices`; fx_gpu's ribbon build gives each
// ribbon's first and last sorted position and the distance along it. One
// instance per sorted particle k draws the segment k → k + 1 (nothing when
// k + 1 is in another ribbon) as `tessellation` Catmull-Rom sub-segments.
// The geometry and look come from `fx_render::particle_surface`, which
// fx_solari's ray-traced ribbon proxy shares.

import bevy_pbr::render::mesh_view_bindings::view;
import fx_render::particle_io::{
    FxVertexOutput, fx, indices, pw, particle_color, particle_dynamic_param, particle_dynamic_param_1, particle_dynamic_param_2, particle_dynamic_param_3, set_sub_uv,
    sample_particle_texture, scene_light, finish,
};
import fx_render::particle_surface::{ribbon_u, ribbon_point, ribbon_builtin_texel};

// Distance along the sorted polyline at each sorted position.
@group(1) @binding(5) var<storage, read> dist: array<f32>;
// (first, last) sorted position of each ribbon id.
@group(1) @binding(6) var<storage, read> ribbons: array<u32>;
// [0] = sorted particle count.
@group(1) @binding(7) var<storage, read> ribbon_meta: array<u32>;

fn ribbon_of(i: u32) -> u32 {
    return u32(clamp(bitcast<i32>(pw(35u, i)), 0, 254));
}

fn world_position(i: u32) -> vec3<f32> {
    return (fx.world_from_local * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz;
}

fn u_at(k: u32, s: u32, e: u32) -> f32 {
    return ribbon_u(k, s, e, dist[k], dist[s], dist[e], fx.uv_mode, fx.link_from_end != 0u, fx.tiling_length);
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> FxVertexOutput {
    var out: FxVertexOutput;
    let n = ribbon_meta[0];
    let i1 = indices[k];
    let r = ribbon_of(i1);
    let s = ribbons[r * 2u];
    let e = ribbons[r * 2u + 1u];
    if (k + 1u >= n || k >= e) {
        // Last particle of its ribbon: no segment (off-screen, zero area).
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return out;
    }
    let i2 = indices[k + 1u];
    let i0 = indices[select(k, k - 1u, k > s)];
    let i3 = indices[select(k + 1u, k + 2u, k + 2u <= e)];
    let p1 = world_position(i1);
    let p2 = world_position(i2);
    var p0 = world_position(i0);
    var p3 = world_position(i3);
    if (i0 == i1) {
        p0 = 2.0 * p1 - p2;
    }
    if (i3 == i2) {
        p3 = 2.0 * p2 - p1;
    }

    // Sub-segment `sub` of `tessellation`; two triangles of (t, side) corners.
    let sub = vi / 6u;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[vi % 6u];
    let t = (f32(sub) + c.x) / f32(max(fx.tessellation, 1u));
    let side_sign = c.y;

    let point = ribbon_point(
        p0, p1, p2, p3, t, side_sign,
        mix(pw(36u, i1), pw(36u, i2), t),
        radians(mix(pw(37u, i1), pw(37u, i2), t)),
        fx.facing_mode,
        (fx.world_from_local * vec4<f32>(fx.custom_facing.xyz, 0.0)).xyz,
        view.world_position, view.world_from_view[0].xyz, view.world_from_view[1].xyz,
    );
    let world = point.world;
    let normal = select(point.normal, -point.normal, dot(point.normal, view.world_position - world) < 0.0);

    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.normal = normal;
    out.color = mix(particle_color(i1), particle_color(i2), t);
    out.dynamic_param = mix(particle_dynamic_param(i1), particle_dynamic_param(i2), t);
    out.dynamic_param_1 = mix(particle_dynamic_param_1(i1), particle_dynamic_param_1(i2), t);
    out.dynamic_param_2 = mix(particle_dynamic_param_2(i1), particle_dynamic_param_2(i2), t);
    out.dynamic_param_3 = mix(particle_dynamic_param_3(i1), particle_dynamic_param_3(i2), t);
    let u = mix(u_at(k, s, e), u_at(k + 1u, s, e), t);
    let nage = mix(pw(16u, i1), pw(16u, i2), t);
    set_sub_uv(&out, nage, vec2<f32>(u, side_sign * 0.5 + 0.5));
    return out;
}

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    var tex = vec4<f32>(1.0);
    @if(TEXTURED) {
        // U may run past 1 (tiled); the flipbook does not apply to ribbons.
        var q = in;
        q.uv0 = fract(in.uv);
        q.uv1 = q.uv0;
        tex = sample_particle_texture(q);
    }
    @if(!TEXTURED) {
        tex = ribbon_builtin_texel(in.uv.y);
    }
    var rgb = in.color.rgb * tex.rgb;
    @if(LIT) {
        rgb *= scene_light(in.normal, true);
    }
    return finish(rgb, in.color.a * tex.a);
}
