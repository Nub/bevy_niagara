// Stable LSD radix sort of (keys, vals) for large views (fx_gpu::view).
//
// 4-bit digits, up to 8 passes (32-bit keys; depth keys skip the low 8 bits,
// see view.wgsl set_sort_args). Each pass q is three dispatches, sized on the
// GPU from the sort length n (word 3 of the pass's tiles slot in `args`):
//   radix_hist    (tiles)  digit counts per tile of TILE keys
//   radix_scan    (16)     per digit, exclusive prefix over the tiles + digit total
//   radix_scatter (tiles)  stable ranks, keys and vals written to the other buffer
// Passes ping-pong between (keys, vals) and (keys_alt, vals_alt); an even pass
// count ends in the original buffers. Portable WebGPU: no subgroups, 8 KiB of
// workgroup memory, no forward-progress assumptions.

const RWG: u32 = 256u;
const ITEMS: u32 = 8u;            // consecutive keys per thread (stability)
const TILE: u32 = 2048u;          // RWG * ITEMS
const DIGITS: u32 = 16u;

@group(0) @binding(0) var<storage, read> src_keys: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> src_vals: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_vals: array<u32>;
// [0..16): digit totals; [16 + d * tiles + tile]: counts, then exclusive prefixes.
@group(0) @binding(4) var<storage, read_write> hist: array<u32>;
@group(0) @binding(5) var<storage, read> args: array<u32>;

struct RadixPass {
    shift: u32,
    slot: u32,     // args slot of this pass's tile dispatch; word 3 holds n
}
@group(1) @binding(1) var<uniform> rp: RadixPass;

// Per thread: 16 digit counts in 16-bit lanes (8 words), scanned over threads.
var<workgroup> lanes: array<u32, 2048>;
var<workgroup> tile_base: array<u32, 16>;

fn sort_n() -> u32 {
    return args[rp.slot * 4u + 3u];
}

fn tiles_of(n: u32) -> u32 {
    return (n + TILE - 1u) / TILE;
}

// The thread's ITEMS keys (digit 16 = past the end), counted per digit in
// 4-bit lanes (at most 8 each).
struct Items {
    k0: vec4<u32>,
    k1: vec4<u32>,
    count: u32,
}

fn digit(k: u32) -> u32 {
    return (k >> rp.shift) & 15u;
}

fn load_items(t: u32, tile: u32, n: u32) -> Items {
    let first = tile * TILE + t * ITEMS;
    var it: Items;
    it.k0 = src_keys[first / 4u];
    it.k1 = src_keys[first / 4u + 1u];
    it.count = min(ITEMS, select(0u, n - first, n > first));
    return it;
}

fn item_key(it: Items, k: u32) -> u32 {
    switch k {
        case 0u: { return it.k0.x; }
        case 1u: { return it.k0.y; }
        case 2u: { return it.k0.z; }
        case 3u: { return it.k0.w; }
        case 4u: { return it.k1.x; }
        case 5u: { return it.k1.y; }
        case 6u: { return it.k1.z; }
        default: { return it.k1.w; }
    }
}

// Writes the thread's counts to `lanes` and turns them into an inclusive
// prefix over threads (Hillis-Steele; 16-bit lanes never carry: a tile has
// at most 2048 keys).
fn count_and_scan(t: u32, it: Items) {
    var c = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    for (var k = 0u; k < it.count; k++) {
        let d = digit(item_key(it, k));
        c[d >> 1u] += 1u << ((d & 1u) * 16u);
    }
    for (var w = 0u; w < 8u; w++) {
        lanes[t * 8u + w] = c[w];
    }
    workgroupBarrier();
    for (var off = 1u; off < RWG; off = off * 2u) {
        var add = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        if (t >= off) {
            for (var w = 0u; w < 8u; w++) {
                add[w] = lanes[(t - off) * 8u + w];
            }
        }
        workgroupBarrier();
        if (t >= off) {
            for (var w = 0u; w < 8u; w++) {
                lanes[t * 8u + w] += add[w];
            }
        }
        workgroupBarrier();
    }
}

// Inclusive count of digit d over threads 0..=t.
fn lane(t: u32, d: u32) -> u32 {
    return (lanes[t * 8u + (d >> 1u)] >> ((d & 1u) * 16u)) & 0xFFFFu;
}

@compute @workgroup_size(RWG)
fn radix_hist(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = sort_n();
    count_and_scan(t, load_items(t, wid.x, n));
    if (t < DIGITS) {
        hist[DIGITS + t * tiles_of(n) + wid.x] = lane(RWG - 1u, t);
    }
}

var<workgroup> scan_sums: array<u32, 256>;

// One workgroup per digit: exclusive prefix of its tile counts, in place.
@compute @workgroup_size(RWG)
fn radix_scan(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = wid.x;
    let tiles = tiles_of(sort_n());
    let per = (tiles + RWG - 1u) / RWG;
    let base = DIGITS + d * tiles;
    let lo = min(t * per, tiles);
    let hi = min(lo + per, tiles);
    var s = 0u;
    for (var i = lo; i < hi; i++) {
        s += hist[base + i];
    }
    scan_sums[t] = s;
    workgroupBarrier();
    for (var off = 1u; off < RWG; off = off * 2u) {
        var add = 0u;
        if (t >= off) {
            add = scan_sums[t - off];
        }
        workgroupBarrier();
        scan_sums[t] += add;
        workgroupBarrier();
    }
    var run = scan_sums[t] - s;
    for (var i = lo; i < hi; i++) {
        let v = hist[base + i];
        hist[base + i] = run;
        run += v;
    }
    if (t == RWG - 1u) {
        hist[d] = scan_sums[t];
    }
}

@compute @workgroup_size(RWG)
fn radix_scatter(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = sort_n();
    let tiles = tiles_of(n);
    let tile = wid.x;
    if (t < DIGITS) {
        // Keys of smaller digits, plus this digit's keys in earlier tiles.
        var below = 0u;
        for (var e = 0u; e < t; e++) {
            below += hist[e];
        }
        tile_base[t] = below + hist[DIGITS + t * tiles + tile];
    }
    let it = load_items(t, tile, n);
    count_and_scan(t, it);
    let first = tile * TILE + t * ITEMS;
    let v0 = src_vals[first / 4u];
    let v1 = src_vals[first / 4u + 1u];
    var mine = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    for (var k = 0u; k < it.count; k++) {
        let key = item_key(it, k);
        let d = digit(key);
        var before = 0u;
        if (t > 0u) {
            before = lane(t - 1u, d);
        }
        let shift = (d & 1u) * 16u;
        let rank = (mine[d >> 1u] >> shift) & 0xFFFFu;
        mine[d >> 1u] += 1u << shift;
        let at = tile_base[d] + before + rank;
        var val = 0u;
        switch k {
            case 0u: { val = v0.x; }
            case 1u: { val = v0.y; }
            case 2u: { val = v0.z; }
            case 3u: { val = v0.w; }
            case 4u: { val = v1.x; }
            case 5u: { val = v1.y; }
            case 6u: { val = v1.z; }
            default: { val = v1.w; }
        }
        dst_keys[at] = key;
        dst_vals[at] = val;
    }
}
