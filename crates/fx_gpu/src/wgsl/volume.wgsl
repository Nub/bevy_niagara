// Volumetric renderer: particles splat density and colour into a grid over
// the volume bounds (fixed-point atomics), then the grid resolves into an
// rgba16float 3D texture: rgb = density-weighted colour, a = extinction (1/m).

const WG: u32 = 64u;
// Fixed-point scale of the accumulated values.
const FIX: f32 = 1024.0;
// Largest splat radius, in cells.
const MAX_CELLS: f32 = 3.0;

struct Params {
    bounds_min: vec3<f32>,
    resolution: u32,
    bounds_max: vec3<f32>,
    capacity: u32,
    density_scale: f32,
    radius_scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> draw: array<u32>;
@group(0) @binding(4) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(5) var volume: texture_storage_3d<rgba16float, write>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * params.capacity + i];
}

fn add(cell: u32, v: f32, k: u32) {
    if (v > 0.0) {
        atomicAdd(&grid[cell * 4u + k], u32(min(v * FIX, 4.0e9)));
    }
}

// One thread per living particle: a soft ball, (1 - d²/r²)² falloff.
@compute @workgroup_size(WG)
fn splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= draw[1]) {
        return;
    }
    let i = vals[j];
    let res = params.resolution;
    let size = params.bounds_max - params.bounds_min;
    let cell = size / f32(res);
    let p = (vec3<f32>(pw(0u, i), pw(1u, i), pw(2u, i)) - params.bounds_min) / cell - 0.5;
    let color = vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i));
    let density = max(color.a, 0.0) * params.density_scale;
    if (density <= 0.0) {
        return;
    }
    let radius = abs(pw(10u, i)) * params.radius_scale;
    // Radius in cells per axis, at least half a cell, at most MAX_CELLS.
    let rc = clamp(vec3<f32>(radius) / cell, vec3<f32>(0.75), vec3<f32>(MAX_CELLS));
    let lo = vec3<i32>(max(floor(p - rc), vec3<f32>(0.0)));
    let hi = vec3<i32>(min(ceil(p + rc), vec3<f32>(f32(res) - 1.0)));
    let rgb = max(color.rgb, vec3<f32>(0.0));
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let d = (vec3<f32>(f32(x), f32(y), f32(z)) - p) / rc;
                let q = dot(d, d);
                if (q >= 1.0) {
                    continue;
                }
                let w = (1.0 - q) * (1.0 - q) * density;
                let c = (u32(z) * res + u32(y)) * res + u32(x);
                add(c, w, 0u);
                add(c, w * rgb.r, 1u);
                add(c, w * rgb.g, 2u);
                add(c, w * rgb.b, 3u);
            }
        }
    }
}

// One thread per cell: grid → texture.
@compute @workgroup_size(4, 4, 4)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.resolution;
    if (any(gid >= vec3<u32>(res))) {
        return;
    }
    let c = (gid.z * res + gid.y) * res + gid.x;
    let d = f32(atomicLoad(&grid[c * 4u])) / FIX;
    var rgb = vec3<f32>(0.0);
    if (d > 0.0) {
        rgb = vec3<f32>(
            f32(atomicLoad(&grid[c * 4u + 1u])),
            f32(atomicLoad(&grid[c * 4u + 2u])),
            f32(atomicLoad(&grid[c * 4u + 3u])),
        ) / FIX / d;
    }
    textureStore(volume, gid, vec4<f32>(rgb, d));
}
