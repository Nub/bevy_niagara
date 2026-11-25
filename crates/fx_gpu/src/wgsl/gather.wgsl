// Component renderer: chosen particle words of the first `max_rows` particles
// of a cull/sort (oldest first) packed into a small list read back by the CPU.
//
// out[0] = row count, out[4 + stride * j + k] = word words[k] of particle j.

const WG: u32 = 64u;

struct Params {
    capacity: u32,
    max_rows: u32,
    stride: u32,
    _pad: u32,
    words: array<vec4<u32>, 8>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> draw: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(WG)
fn gather_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let count = min(draw[1], params.max_rows);
    if (j == 0u) {
        out[0] = bitcast<f32>(count);
    }
    if (j >= count) {
        return;
    }
    let i = vals[j];
    let o = 4u + j * params.stride;
    for (var k = 0u; k < params.stride; k++) {
        let w = params.words[k / 4u][k % 4u];
        out[o + k] = particles[w * params.capacity + i];
    }
}
