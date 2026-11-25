// Light renderer: the first `max_lights` visible particles of a view's
// cull/sort (nearest first) packed into a small list read back by the CPU.
//
// out[0] = count, out[4 + 8 j ..] = (position.xyz, radius, color.rgba) of light j,
// in the emitter's local space.

const WG: u32 = 64u;

struct Params {
    capacity: u32,
    max_lights: u32,
    radius_scale: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> draw: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * params.capacity + i];
}

@compute @workgroup_size(WG)
fn gather_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let count = min(draw[1], params.max_lights);
    if (j == 0u) {
        out[0] = bitcast<f32>(count);
    }
    if (j >= count) {
        return;
    }
    let i = vals[j];
    let o = 4u + j * 8u;
    out[o] = pw(0u, i);
    out[o + 1u] = pw(1u, i);
    out[o + 2u] = pw(2u, i);
    out[o + 3u] = pw(39u, i) * params.radius_scale;
    out[o + 4u] = pw(6u, i);
    out[o + 5u] = pw(7u, i);
    out[o + 6u] = pw(8u, i);
    out[o + 7u] = pw(9u, i);
}
