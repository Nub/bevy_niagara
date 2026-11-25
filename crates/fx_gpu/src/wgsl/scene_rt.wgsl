
// ---- scene collisions: ray tracing (bind group 3, Solari's scene layout) ----
// Only the previous frame's TLAS is read: the simulation runs before Solari
// builds this frame's (which then holds the particles' own proxies). Opaque
// geometry only, so ray-traced particle proxies (alpha-masked, non-opaque)
// and alpha-masked scene geometry are not hit.

@group(3) @binding(6) var tlas: acceleration_structure;

const RT_CULL_NO_OPAQUE: u32 = 0x80u;

// World-space distance to the first opaque hit from `origin` along unit
// `dir`, within `t_max`; negative on a miss.
fn scene_ray(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RT_CULL_NO_OPAQUE, 0xFFu, 0.0, t_max, origin, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return -1.0;
    }
    return hit.t;
}

// The surface a particle crossed moving from `from` to `to` this step (or
// within `radius` of `to`). The normal comes from two more rays beside the
// first; without them, it faces back along the move.
fn scene_ray_contact(from: vec3<f32>, to: vec3<f32>, radius: f32, _thickness: f32) -> Contact {
    let a = (scene.world_from_local * vec4<f32>(from, 1.0)).xyz;
    let b = (scene.world_from_local * vec4<f32>(to, 1.0)).xyz;
    let scale = length(scene.world_from_local[0].xyz);
    let d = b - a;
    let len = length(d);
    if (!(len > 0.0)) {
        return no_contact();
    }
    let dir = d / len;
    let back = 0.001;
    let t_max = len + radius * scale + back;
    let start = a - dir * back;
    let t = scene_ray(start, dir, t_max);
    if (t < 0.0) {
        return no_contact();
    }
    let p0 = start + dir * t;
    let e = max(0.01, 0.01 * t);
    let t1 = basis_t1(dir);
    let t2 = basis_t2(dir);
    let s1 = scene_ray(start + t1 * e, dir, t_max * 2.0);
    let s2 = scene_ray(start + t2 * e, dir, t_max * 2.0);
    var n = -dir;
    if (s1 >= 0.0 && s2 >= 0.0) {
        let c = cross(start + t1 * e + dir * s1 - p0, start + t2 * e + dir * s2 - p0);
        let l = length(c);
        if (l > 0.0) {
            n = c / l;
            if (dot(n, dir) > 0.0) {
                n = -n;
            }
        }
    }
    let m = scene.world_from_local;
    let nl = transpose(mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz)) * n;
    let o = (scene.local_from_world * vec4<f32>(p0, 1.0)).xyz;
    return Contact(true, unit_or_y(nl), o);
}
