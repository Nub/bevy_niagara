// Ribbon geometry inputs, built on the GPU after the ribbon sort.
//
// The ribbon renderer's ViewDraw sorts the living particles by
// (RibbonID, link order) into `vals[0..n]`. These kernels then compute, per
// sorted position k:
// - `ribbons[id]`: the first and last sorted positions of each ribbon;
// - `dist[k]`: the length of the polyline from sorted position 0 to k,
//   counting only links within a ribbon (a global prefix sum), so the
//   distance along ribbon r is `dist[k] - dist[ribbons[r].start]`.
// The vertex shader turns these into UVs; segment k joins k and k + 1 when
// both are in the same ribbon.
//
// ribbon_prep (1 group) → ribbon_link (indirect) → ribbon_scan (1 group)
// → ribbon_add (indirect)

const WG: u32 = 256u;
const MAX_RIBBONS: u32 = 255u;

struct Params {
    capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> draw: array<u32>;
// rrmeta[0] = n (sorted particles), rmeta[1] = blocks of WG.
@group(0) @binding(4) var<storage, read_write> rmeta: array<u32>;
// (start, end) per ribbon id.
@group(0) @binding(5) var<storage, read_write> ribbons: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> dist: array<f32>;
@group(0) @binding(7) var<storage, read_write> block_sums: array<f32>;

@group(1) @binding(0) var<storage, read_write> args: array<u32>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * params.capacity + i];
}

fn ribbon_of(i: u32) -> u32 {
    return u32(clamp(bitcast<i32>(pw(35u, i)), 0, 254));
}

fn position_of(i: u32) -> vec3<f32> {
    return vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i));
}

@compute @workgroup_size(WG)
fn ribbon_prep(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t < MAX_RIBBONS) {
        atomicStore(&ribbons[t * 2u], 0xFFFFFFFFu);
        atomicStore(&ribbons[t * 2u + 1u], 0u);
    }
    if (t == 0u) {
        let n = draw[1];
        let blocks = (n + WG - 1u) / WG;
        rmeta[0] = n;
        rmeta[1] = blocks;
        args[0] = blocks;
        args[1] = 1u;
        args[2] = 1u;
    }
}

var<workgroup> scan: array<f32, WG>;
var<workgroup> wg_blocks: u32;

@compute @workgroup_size(WG)
fn ribbon_link(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let k = gid.x;
    let t = lid.x;
    let n = rmeta[0];
    var seg = 0.0;
    if (k < n) {
        let i = vals[k];
        let r = ribbon_of(i);
        atomicMin(&ribbons[r * 2u], k);
        atomicMax(&ribbons[r * 2u + 1u], k);
        if (k + 1u < n) {
            let j = vals[k + 1u];
            if (ribbon_of(j) == r) {
                seg = distance(position_of(i), position_of(j));
            }
        }
    }
    // Inclusive Hillis-Steele scan of the link lengths within the block.
    scan[t] = seg;
    workgroupBarrier();
    for (var o = 1u; o < WG; o = o * 2u) {
        var v = scan[t];
        if (t >= o) {
            v += scan[t - o];
        }
        workgroupBarrier();
        scan[t] = v;
        workgroupBarrier();
    }
    if (k < n) {
        // Exclusive: the distance up to k, not past it.
        dist[k] = scan[t] - seg;
    }
    if (t == WG - 1u) {
        block_sums[wid.x] = scan[t];
    }
}

// Exclusive scan of the block sums, in one workgroup.
@compute @workgroup_size(WG)
fn ribbon_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t == 0u) {
        wg_blocks = rmeta[1];
    }
    // Uniform for the barriers in the loop.
    let blocks = workgroupUniformLoad(&wg_blocks);
    var carry = 0.0;
    for (var base = 0u; base < blocks; base += WG) {
        let b = base + t;
        var v = 0.0;
        if (b < blocks) {
            v = block_sums[b];
        }
        scan[t] = v;
        workgroupBarrier();
        for (var o = 1u; o < WG; o = o * 2u) {
            var s = scan[t];
            if (t >= o) {
                s += scan[t - o];
            }
            workgroupBarrier();
            scan[t] = s;
            workgroupBarrier();
        }
        if (b < blocks) {
            block_sums[b] = carry + scan[t] - v;
        }
        carry += scan[WG - 1u];
        workgroupBarrier();
    }
}

@compute @workgroup_size(WG)
fn ribbon_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let k = gid.x;
    if (k < rmeta[0]) {
        dist[k] += block_sums[wid.x];
    }
}
