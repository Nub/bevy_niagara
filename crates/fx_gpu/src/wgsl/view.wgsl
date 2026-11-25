// Per-view culling, depth sort and indirect draw arguments for one emitter.
//
// cull_prep (1 thread) → cull (indirect) → [bitonic sort passes (indirect)]
// Output: `vals[0..instance_count]` = indices of the visible particles, in
// back-to-front order when sorting; `draw` = draw_indirect arguments.

const WG: u32 = 256u;
const BLOCK: u32 = 512u;     // elements sorted in workgroup memory
const NONE: u32 = 0xFFFFFFFFu;

struct ViewParams {
    planes: array<vec4<f32>, 6>,   // inside when dot(xyz, p) + w >= -radius
    camera_position: vec4<f32>,
    camera_forward: vec4<f32>,
    capacity: u32,
    sort_mode: u32,               // 0 none, 1 view depth, 2 view distance, 3 UniqueID
    max_pow2: u32,                // sort buffer length
    cull_mode: u32,               // 0 sprite size, 1 word * scale, 2 max |Scale| * scale, 3 off
    vertex_count: u32,            // draw argument 0: vertices or indices per instance
    radius_word: u32,
    radius_scale: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> vp: ViewParams;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw: array<atomic<u32>>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * vp.capacity + i];
}

fn count() -> u32 {
    return state[0];
}

// Sort size for the living particles: a power of two, at least one block.
fn sort_len() -> u32 {
    let n = max(count(), 1u);
    let p = 1u << (32u - countLeadingZeros(n - 1u));
    return clamp(p, BLOCK, vp.max_pow2);
}

// ---- cull_prep: dispatch args for cull and every sort pass ----

@group(1) @binding(0) var<storage, read_write> args: array<u32>;

fn set_args(slot: u32, x: u32) {
    args[slot * 4u] = x;
    args[slot * 4u + 1u] = 1u;
    args[slot * 4u + 2u] = 1u;
}

@compute @workgroup_size(1)
fn cull_prep() {
    atomicStore(&draw[0], vp.vertex_count);
    atomicStore(&draw[1], 0u);
    atomicStore(&draw[2], 0u);
    atomicStore(&draw[3], 0u);
    atomicStore(&draw[4], 0u);
    if (vp.sort_mode == 0u) {
        set_args(0u, (count() + WG - 1u) / WG);
        return;
    }
    let p = sort_len();
    set_args(0u, p / WG);
    // Same pass order as fx_gpu::view::sort_passes.
    var slot = 1u;
    set_args(slot, p / BLOCK);   // local sort
    slot += 1u;
    for (var k = BLOCK * 2u; k <= vp.max_pow2; k = k * 2u) {
        for (var j = k / 2u; j >= BLOCK; j = j / 2u) {
            set_args(slot, select(0u, p / BLOCK, k <= p));
            slot += 1u;
        }
        set_args(slot, select(0u, p / BLOCK, k <= p));   // local merge
        slot += 1u;
    }
}

// ---- cull: visibility and sort keys ----

fn visible(i: u32) -> bool {
    if (vp.cull_mode == 3u) {
        return true;
    }
    let pos = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
    var radius = 0.0;
    switch vp.cull_mode {
        case 1u: { radius = abs(pw(vp.radius_word, i)) * vp.radius_scale; }
        case 2u: {
            radius = max(abs(pw(28u, i)), max(abs(pw(29u, i)), abs(pw(30u, i)))) * vp.radius_scale;
        }
        default: {
            // Covers rotation and pivot offsets up to 0.5.
            radius = max(abs(pw(10u, i)), abs(pw(11u, i)));
        }
    }
    for (var k = 0u; k < 6u; k++) {
        let pl = vp.planes[k];
        if (dot(pl.xyz, pos) + pl.w < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(WG)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = count();
    if (vp.sort_mode == 0u) {
        if (i < n && visible(i)) {
            vals[atomicAdd(&draw[1], 1u)] = i;
        }
        return;
    }
    // Sorting: every slot below the sort length gets a key; hidden ones sort last.
    var key = NONE;
    if (i < n && visible(i)) {
        let d = vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i)) - vp.camera_position.xyz;
        var dist = dot(d, vp.camera_forward.xyz);
        if (vp.sort_mode == 2u) {
            dist = length(d);
        }
        // Far first: larger distance → smaller key. Non-negative floats order as their bits.
        key = 0xFFFFFFFEu - bitcast<u32>(max(dist, 0.0));
        if (vp.sort_mode == 3u) {
            // Deterministic order (headless preview): by UniqueID, independent
            // of the slot order atomic compaction produced.
            key = min(bitcast<u32>(pw(27u, i)), 0xFFFFFFFEu);
        }
        if (vp.sort_mode >= 4u) {
            // Ribbons: ribbon id in the top 8 bits, then the link order.
            let ribbon = u32(clamp(bitcast<i32>(pw(35u, i)), 0, 254));
            var order = bitcast<u32>(pw(27u, i)) & 0xFFFFFFu;
            if (vp.sort_mode == 5u) {
                // Float to an order-preserving uint, top 24 bits.
                let b = bitcast<u32>(pw(38u, i));
                order = select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u) >> 8u;
            }
            key = min((ribbon << 24u) | order, 0xFFFFFFFEu);
        }
        atomicAdd(&draw[1], 1u);
    }
    keys[i] = key;
    vals[i] = i;
}

// ---- bitonic sort of (keys, vals), ascending by key ----

struct Pass {
    k: u32,
    j: u32,
}
@group(1) @binding(1) var<uniform> pass_params: Pass;

var<workgroup> sk: array<u32, BLOCK>;
var<workgroup> sv: array<u32, BLOCK>;

fn local_compare(t: u32, base: u32, k: u32, j: u32) {
    let i = 2u * j * (t / j) + (t % j);
    let l = i + j;
    let ascending = ((base + i) & k) == 0u;
    let a = sk[i];
    let b = sk[l];
    if ((a > b) == ascending) {
        sk[i] = b;
        sk[l] = a;
        let v = sv[i];
        sv[i] = sv[l];
        sv[l] = v;
    }
}

fn load_block(t: u32, base: u32) {
    sk[t] = keys[base + t];
    sv[t] = vals[base + t];
    sk[t + WG] = keys[base + t + WG];
    sv[t + WG] = vals[base + t + WG];
    workgroupBarrier();
}

fn store_block(t: u32, base: u32) {
    workgroupBarrier();
    keys[base + t] = sk[t];
    vals[base + t] = sv[t];
    keys[base + t + WG] = sk[t + WG];
    vals[base + t + WG] = sv[t + WG];
}

// Fully sorts each block (stages k = 2..BLOCK).
@compute @workgroup_size(WG)
fn sort_local(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let base = wid.x * BLOCK;
    load_block(t, base);
    for (var k = 2u; k <= BLOCK; k = k * 2u) {
        for (var j = k / 2u; j > 0u; j = j / 2u) {
            local_compare(t, base, k, j);
            workgroupBarrier();
        }
    }
    store_block(t, base);
}

// Stage k, strides j < BLOCK, in workgroup memory.
@compute @workgroup_size(WG)
fn sort_merge_local(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let base = wid.x * BLOCK;
    let k = pass_params.k;
    load_block(t, base);
    for (var j = BLOCK / 2u; j > 0u; j = j / 2u) {
        local_compare(t, base, k, j);
        workgroupBarrier();
    }
    store_block(t, base);
}

// Stage k, one stride j >= BLOCK, across the whole buffer.
@compute @workgroup_size(WG)
fn sort_merge_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let k = pass_params.k;
    let j = pass_params.j;
    let i = 2u * j * (t / j) + (t % j);
    let l = i + j;
    let ascending = (i & k) == 0u;
    let a = keys[i];
    let b = keys[l];
    if ((a > b) == ascending) {
        keys[i] = b;
        keys[l] = a;
        let v = vals[i];
        vals[i] = vals[l];
        vals[l] = v;
    }
}
