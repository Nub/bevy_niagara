// Writes one ribbon renderer's ray-traced proxy geometry from its GPU-built
// ribbon (fx_render's sort and ribbon build of this frame): quad slot
// q = k * tessellation + sub is sub-segment `sub` of the segment from sorted
// particle k to k + 1, as two quads (one per side, 2 mm apart, so the emissive
// ribbon lights both sides) in the proxy mesh's vertices, through the same
// `fx_render::particle_surface::ribbon_point` the raster ribbon uses. Per quad
// three data texels for the hit function: the colour at both ends and
// (U at both ends). Slots with no segment collapse to a point at the
// effect's origin (zero area: never hit, never emit).

import fx_render::particle_surface::{ribbon_u, ribbon_point, surface_safe_normalize};

struct RibbonProxyParams {
    world_from_local: mat4x4<f32>,
    cam_position: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_back: vec4<f32>,
    custom_facing: vec4<f32>,
    capacity: u32,
    slots: u32,
    tessellation: u32,
    facing_mode: u32,
    // Mesh vertices in the slab: first word, words per vertex, attribute offsets.
    base: u32,
    stride: u32,
    normal_offset: u32,
    tangent_offset: u32,
    data_width: u32,
    uv_mode: u32,
    link_from_end: u32,
    tiling_length: f32,
}

const NONE = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> p: RibbonProxyParams;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> dist: array<f32>;
@group(0) @binding(4) var<storage, read> ribbons: array<u32>;
@group(0) @binding(5) var<storage, read> ribbon_meta: array<u32>;
@group(0) @binding(6) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(7) var data: texture_storage_2d<rgba16float, write>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * p.capacity + i];
}

fn world_position(i: u32) -> vec3<f32> {
    return (p.world_from_local * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz;
}

fn color(i: u32) -> vec4<f32> {
    return vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i));
}

fn u_at(k: u32, s: u32, e: u32) -> f32 {
    return ribbon_u(k, s, e, dist[k], dist[s], dist[e], p.uv_mode, p.link_from_end != 0u, p.tiling_length);
}

fn write_vertex(v: u32, pos: vec3<f32>, n: vec3<f32>, t: vec3<f32>) {
    let b = p.base + v * p.stride;
    vertices[b] = pos.x;
    vertices[b + 1u] = pos.y;
    vertices[b + 2u] = pos.z;
    let nb = b + p.normal_offset;
    vertices[nb] = n.x;
    vertices[nb + 1u] = n.y;
    vertices[nb + 2u] = n.z;
    if p.tangent_offset != NONE {
        let tb = b + p.tangent_offset;
        vertices[tb] = t.x;
        vertices[tb + 1u] = t.y;
        vertices[tb + 2u] = t.z;
        vertices[tb + 3u] = 1.0;
    }
}

fn write_texel(t: u32, value: vec4<f32>) {
    textureStore(data, vec2<u32>(t % p.data_width, t / p.data_width), value);
}

fn collapse(q: u32) {
    let anchor = p.world_from_local[3].xyz;
    for (var v = 0u; v < 8u; v++) {
        write_vertex(8u * q + v, anchor, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
    }
    for (var j = 0u; j < 3u; j++) {
        write_texel(3u * q + j, vec4<f32>(0.0));
    }
}

fn ribbon_of(i: u32) -> u32 {
    return u32(clamp(bitcast<i32>(pw(35u, i)), 0, 254));
}

@compute @workgroup_size(64, 1, 1)
fn ribbon_proxy(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = id.x;
    if q >= p.slots { return; }
    let tess = max(p.tessellation, 1u);
    let k = q / tess;
    let sub = q % tess;
    let n = ribbon_meta[0];
    if k + 1u >= n {
        collapse(q);
        return;
    }
    let i1 = indices[k];
    let r = ribbon_of(i1);
    let s = ribbons[r * 2u];
    let e = ribbons[r * 2u + 1u];
    if k >= e {
        collapse(q);
        return;
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
    let facing = (p.world_from_local * vec4<f32>(p.custom_facing.xyz, 0.0)).xyz;
    let t0 = f32(sub) / f32(tess);
    let t1 = f32(sub + 1u) / f32(tess);
    // Corners in the order of the proxy mesh's vertices: quad UV (along, across).
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    for (var c = 0u; c < 4u; c++) {
        let t = mix(t0, t1, corners[c].x);
        let point = ribbon_point(
            p0, p1, p2, p3, t, corners[c].y * 2.0 - 1.0,
            mix(pw(36u, i1), pw(36u, i2), t),
            radians(mix(pw(37u, i1), pw(37u, i2), t)),
            p.facing_mode, facing,
            p.cam_position.xyz, p.cam_right.xyz, p.cam_up.xyz,
        );
        // The proxy mesh winds its first quad counter-clockwise seen from
        // cross(side, tangent) for these corners: that side is its front.
        let nrm = point.normal;
        let along = surface_safe_normalize(p2 - p1, vec3<f32>(1.0, 0.0, 0.0));
        write_vertex(8u * q + c, point.world, nrm, along);
        write_vertex(8u * q + 4u + c, point.world - nrm * 0.002, -nrm, along);
    }
    let u1 = u_at(k, s, e);
    let u2 = u_at(k + 1u, s, e);
    write_texel(3u * q, mix(color(i1), color(i2), t0));
    write_texel(3u * q + 1u, mix(color(i1), color(i2), t1));
    write_texel(3u * q + 2u, vec4<f32>(mix(u1, u2, t0), mix(u1, u2, t1), 0.0, 0.0));
}
