
// ---- data interfaces (bind group 1): meshes and textures ----
// Mirrors fx_core::data. Per data interface, three vec4<u32> of `di_info`:
//   [vertex offset, skin offset, index offset, area-CDF offset]
//   [bone offset, triangles, bones, pose offset]
//   [texture width, texture height, pose matrices, flags (1 bound, 2 skinned)]
// Offsets index `di_data` (f32 bits; joints and indices are u32 bits).
// A mesh vertex is 6 words (position, normal); a skinned vertex has 8 more
// (4 joints, 4 weights) at the skin offset. A texture is width * height RGBA
// texels from the top row at the vertex offset. `di_pose` holds, from the
// pose offset: (interval, 0, 0, 0), the current matrices, the previous ones.

@group(1) @binding(0) var<storage, read> di_info: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> di_data: array<u32>;
@group(1) @binding(2) var<storage, read> di_pose: array<vec4<f32>>;

fn dif(i: u32) -> f32 {
    return bitcast<f32>(di_data[i]);
}

fn di_vec3(i: u32) -> vec3<f32> {
    return vec3<f32>(dif(i), dif(i + 1u), dif(i + 2u));
}

struct MeshSample {
    ok: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    velocity: vec3<f32>,
}

fn no_mesh_sample() -> MeshSample {
    return MeshSample(false, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
}

// Matrix `i` of a pose (`which` 0 current, 1 previous); identity past the end.
fn di_mat(pose: u32, count: u32, which: u32, i: u32) -> mat4x4<f32> {
    if (i >= count) {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }
    let b = pose + 1u + (which * count + i) * 4u;
    return mat4x4<f32>(di_pose[b], di_pose[b + 1u], di_pose[b + 2u], di_pose[b + 3u]);
}

// The skinning matrix of vertex `v` (fx_core::data::skin).
fn di_skin(a: vec4<u32>, pose: u32, c: vec4<u32>, which: u32, v: u32) -> mat4x4<f32> {
    if ((c.w & 2u) == 0u) {
        return di_mat(pose, c.z, which, 0u);
    }
    let s = a.y + v * 8u;
    var m = di_mat(pose, c.z, which, di_data[s]) * dif(s + 4u);
    for (var k = 1u; k < 4u; k++) {
        m += di_mat(pose, c.z, which, di_data[s + k]) * dif(s + 4u + k);
    }
    return m;
}

fn di_velocity(pos: vec3<f32>, prev: vec3<f32>, interval: f32) -> vec3<f32> {
    if (interval > 0.0) {
        return (pos - prev) / interval;
    }
    return vec3<f32>(0.0);
}

fn di_unit_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 1e-12) {
        return v / l;
    }
    return vec3<f32>(0.0);
}

// fx_core::data::sample_surface.
fn mesh_sample_surface(di: u32, u: vec3<f32>) -> MeshSample {
    let a = di_info[di * 3u];
    let b = di_info[di * 3u + 1u];
    let c = di_info[di * 3u + 2u];
    let tris = b.y;
    if ((c.w & 1u) == 0u || tris == 0u) {
        return no_mesh_sample();
    }
    // First triangle whose cumulative area is past u.x.
    var lo = 0u;
    var hi = tris;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (dif(a.w + mid) <= u.x) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let t = min(lo, tris - 1u);
    var r1 = u.y;
    var r2 = u.z;
    if (r1 + r2 > 1.0) {
        r1 = 1.0 - r1;
        r2 = 1.0 - r2;
    }
    let w = array<f32, 3>(1.0 - r1 - r2, r1, r2);
    var pos = vec3<f32>(0.0);
    var prev = vec3<f32>(0.0);
    var nrm = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        let v = di_data[a.z + t * 3u + k];
        let p = di_vec3(a.x + v * 6u);
        let n = di_vec3(a.x + v * 6u + 3u);
        let m = di_skin(a, b.w, c, 0u, v);
        let mp = di_skin(a, b.w, c, 1u, v);
        pos += (m * vec4<f32>(p, 1.0)).xyz * w[k];
        prev += (mp * vec4<f32>(p, 1.0)).xyz * w[k];
        nrm += (m * vec4<f32>(n, 0.0)).xyz * w[k];
    }
    let interval = di_pose[b.w].x;
    return MeshSample(true, pos, di_unit_or_zero(nrm), di_velocity(pos, prev, interval));
}

// fx_core::data::sample_bone.
fn mesh_sample_bone(di: u32, u: f32) -> MeshSample {
    let b = di_info[di * 3u + 1u];
    let c = di_info[di * 3u + 2u];
    let n = b.z;
    if ((c.w & 1u) == 0u || (c.w & 2u) == 0u || n == 0u) {
        return no_mesh_sample();
    }
    let i = min(u32(u * f32(n)), n - 1u);
    let p = di_vec3(b.x + i * 3u);
    let pos = (di_mat(b.w, c.z, 0u, i) * vec4<f32>(p, 1.0)).xyz;
    let prev = (di_mat(b.w, c.z, 1u, i) * vec4<f32>(p, 1.0)).xyz;
    return MeshSample(true, pos, vec3<f32>(0.0), di_velocity(pos, prev, di_pose[b.w].x));
}

fn di_texel(off: u32, w: i32, x: i32, y: i32) -> vec4<f32> {
    let i = off + u32(y * w + x) * 4u;
    return vec4<f32>(dif(i), dif(i + 1u), dif(i + 2u), dif(i + 3u));
}

fn di_wrap(i: f32, n: i32) -> i32 {
    let k = i32(i);
    return ((k % n) + n) % n;
}

// fx_core::data::sample_texture: bilinear, wrapping; white when unbound.
fn texture_sample(di: u32, uv: vec2<f32>) -> vec4<f32> {
    let a = di_info[di * 3u];
    let c = di_info[di * 3u + 2u];
    if ((c.w & 1u) == 0u || c.x == 0u || c.y == 0u) {
        return vec4<f32>(1.0);
    }
    let w = i32(c.x);
    let h = i32(c.y);
    let x = uv.x * f32(w) - 0.5;
    let y = uv.y * f32(h) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let xa = di_wrap(x0, w);
    let xb = di_wrap(x0 + 1.0, w);
    let ya = di_wrap(y0, h);
    let yb = di_wrap(y0 + 1.0, h);
    let t00 = di_texel(a.x, w, xa, ya);
    let t10 = di_texel(a.x, w, xb, ya);
    let t01 = di_texel(a.x, w, xa, yb);
    let t11 = di_texel(a.x, w, xb, yb);
    let top = t00 + (t10 - t00) * fx;
    let bottom = t01 + (t11 - t01) * fx;
    return top + (bottom - top) * fy;
}
