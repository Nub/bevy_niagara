// Shared simulation library, prepended to every generated emitter shader.
// Must match fx_core::sim bit for bit where it can (integer hashing) and
// within float tolerance elsewhere.

const WG: u32 = 256u;

// Words of the emitter state buffer (see fx_gpu::layout).
const S_COUNT: u32 = 0u;      // living particles in the current buffer
const S_NEW: u32 = 1u;        // append counter while writing the next buffer
const S_NEXT_ID: u32 = 2u;    // next UniqueID
const S_SPAWN_N: u32 = 3u;    // particles to spawn this step
const S_BASE_ID: u32 = 4u;    // UniqueID of the first one

struct Frame {
    dt: f32,
    time: f32,
    emitter_age: f32,
    frame: u32,
    spawn_request: u32,
    seed: u32,
    emitter: u32,
    capacity: u32,
    user: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> fr: Frame;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;
// Indirect dispatch args: update at words 0..3, spawn at 4..7. Bound only to the
// kernels that write them, never to the indirect dispatches that read them.
@group(1) @binding(0) var<storage, read_write> args: array<u32>;

// ---- random numbers (fx_core::sim::rng) ----

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

struct Key {
    particle: u32,
    frame: u32,
    module_id: u32,
}

fn rand_unit(k: Key, slot: u32) -> f32 {
    var h = 0x9E3779B9u;
    h = pcg(h ^ fr.seed);
    h = pcg(h ^ fr.emitter);
    h = pcg(h ^ k.particle);
    h = pcg(h ^ k.frame);
    h = pcg(h ^ k.module_id);
    h = pcg(h ^ slot);
    return to_unit(h);
}

// ---- noise (fx_core::sim::noise) ----

var<private> GRADIENTS: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(-1.0, 1.0, 0.0), vec3<f32>(1.0, -1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, -1.0), vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, -1.0, 1.0), vec3<f32>(0.0, 1.0, -1.0), vec3<f32>(0.0, -1.0, -1.0),
);

fn lattice_gradient(x: i32, y: i32, z: i32, channel: u32) -> vec3<f32> {
    let h = pcg(pcg(pcg(pcg(channel) ^ bitcast<u32>(x)) ^ bitcast<u32>(y)) ^ bitcast<u32>(z));
    return GRADIENTS[h % 12u];
}

// Gradient of Perlin noise at p (the value is not needed by the curl).
fn noise_grad(p: vec3<f32>, channel: u32) -> vec3<f32> {
    let i = floor(p);
    let f = p - i;
    let ix = i32(i.x);
    let iy = i32(i.y);
    let iz = i32(i.z);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let ga = lattice_gradient(ix, iy, iz, channel);
    let gb = lattice_gradient(ix + 1, iy, iz, channel);
    let gc = lattice_gradient(ix, iy + 1, iz, channel);
    let gd = lattice_gradient(ix + 1, iy + 1, iz, channel);
    let ge = lattice_gradient(ix, iy, iz + 1, channel);
    let gf = lattice_gradient(ix + 1, iy, iz + 1, channel);
    let gg = lattice_gradient(ix, iy + 1, iz + 1, channel);
    let gh = lattice_gradient(ix + 1, iy + 1, iz + 1, channel);
    let va = dot(ga, f);
    let vb = dot(gb, f - vec3<f32>(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3<f32>(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3<f32>(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3<f32>(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3<f32>(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3<f32>(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3<f32>(1.0, 1.0, 1.0));

    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = ve - va;
    let k4 = va - vb - vc + vd;
    let k5 = va - vc - ve + vg;
    let k6 = va - vb - ve + vf;
    let k7 = -va + vb + vc - vd + ve - vf - vg + vh;

    let gk1 = gb - ga;
    let gk2 = gc - ga;
    let gk3 = ge - ga;
    let gk4 = ga - gb - gc + gd;
    let gk5 = ga - gc - ge + gg;
    let gk6 = ga - gb - ge + gf;
    let gk7 = -ga + gb + gc - gd + ge - gf - gg + gh;
    var grad = ga + u.x * gk1 + u.y * gk2 + u.z * gk3 + u.x * u.y * gk4 + u.y * u.z * gk5
        + u.z * u.x * gk6 + u.x * u.y * u.z * gk7;
    grad += du * vec3<f32>(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y,
    );
    return grad;
}

fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let g0 = noise_grad(p, 0u);
    let g1 = noise_grad(p, 1u);
    let g2 = noise_grad(p, 2u);
    return vec3<f32>(g2.y - g1.z, g0.z - g2.x, g1.x - g0.y);
}

// Orthonormal basis around unit n (Duff et al. 2017); copysign keeps -0.0's sign.
fn basis_t1(n: vec3<f32>) -> vec3<f32> {
    let sign = select(1.0, -1.0, (bitcast<u32>(n.z) & 0x80000000u) != 0u);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}
fn basis_t2(n: vec3<f32>) -> vec3<f32> {
    let sign = select(1.0, -1.0, (bitcast<u32>(n.z) & 0x80000000u) != 0u);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(b, sign + n.y * n.y * a, -n.y);
}

// ---- particles: structure of arrays, word w of particle i at w * capacity + i ----

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    color: vec4<f32>,
    sprite_size: vec2<f32>,
    sprite_rotation: f32,
    mass: f32,
    lifetime: f32,
    age: f32,
    normalized_age: f32,
    initial_color: vec4<f32>,
    initial_sprite_size: vec2<f32>,
    force: vec3<f32>,
    drag: f32,
    id: u32,
    alive: bool,
}

fn new_particle(id: u32) -> Particle {
    var p: Particle;
    p.position = vec3<f32>(0.0);
    p.velocity = vec3<f32>(0.0);
    p.color = vec4<f32>(1.0);
    p.sprite_size = vec2<f32>(0.1);
    p.sprite_rotation = 0.0;
    p.mass = 1.0;
    p.lifetime = 2.0;
    p.age = 0.0;
    p.normalized_age = 0.0;
    p.initial_color = vec4<f32>(1.0);
    p.initial_sprite_size = vec2<f32>(0.1);
    p.force = vec3<f32>(0.0);
    p.drag = 0.0;
    p.id = id;
    p.alive = true;
    return p;
}

fn ld(w: u32, i: u32) -> f32 {
    return src[w * fr.capacity + i];
}

fn load_particle(i: u32) -> Particle {
    var p: Particle;
    p.position = vec3<f32>(ld(0u, i), ld(1u, i), ld(2u, i));
    p.velocity = vec3<f32>(ld(3u, i), ld(4u, i), ld(5u, i));
    p.color = vec4<f32>(ld(6u, i), ld(7u, i), ld(8u, i), ld(9u, i));
    p.sprite_size = vec2<f32>(ld(10u, i), ld(11u, i));
    p.sprite_rotation = ld(12u, i);
    p.mass = ld(13u, i);
    p.lifetime = ld(14u, i);
    p.age = ld(15u, i);
    p.normalized_age = ld(16u, i);
    p.initial_color = vec4<f32>(ld(17u, i), ld(18u, i), ld(19u, i), ld(20u, i));
    p.initial_sprite_size = vec2<f32>(ld(21u, i), ld(22u, i));
    p.force = vec3<f32>(ld(23u, i), ld(24u, i), ld(25u, i));
    p.drag = ld(26u, i);
    p.id = bitcast<u32>(ld(27u, i));
    p.alive = true;
    return p;
}

fn st(w: u32, i: u32, v: f32) {
    dst[w * fr.capacity + i] = v;
}

fn store_particle(i: u32, p: Particle) {
    st(0u, i, p.position.x); st(1u, i, p.position.y); st(2u, i, p.position.z);
    st(3u, i, p.velocity.x); st(4u, i, p.velocity.y); st(5u, i, p.velocity.z);
    st(6u, i, p.color.x); st(7u, i, p.color.y); st(8u, i, p.color.z); st(9u, i, p.color.w);
    st(10u, i, p.sprite_size.x); st(11u, i, p.sprite_size.y);
    st(12u, i, p.sprite_rotation);
    st(13u, i, p.mass);
    st(14u, i, p.lifetime);
    st(15u, i, p.age);
    st(16u, i, p.normalized_age);
    st(17u, i, p.initial_color.x); st(18u, i, p.initial_color.y); st(19u, i, p.initial_color.z); st(20u, i, p.initial_color.w);
    st(21u, i, p.initial_sprite_size.x); st(22u, i, p.initial_sprite_size.y);
    st(23u, i, p.force.x); st(24u, i, p.force.y); st(25u, i, p.force.z);
    st(26u, i, p.drag);
    st(27u, i, bitcast<f32>(p.id));
}

fn groups(n: u32) -> u32 {
    return (n + WG - 1u) / WG;
}

// ---- the step: begin → update (indirect) → spawn_prep → spawn (indirect) → finish ----

@compute @workgroup_size(1)
fn begin() {
    let n = atomicLoad(&state[S_COUNT]);
    args[0] = groups(n);
    args[1] = 1u;
    args[2] = 1u;
    atomicStore(&state[S_NEW], 0u);
}

@compute @workgroup_size(WG)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&state[S_COUNT])) {
        return;
    }
    var p = load_particle(i);
    update_particle(&p);
    if (p.alive) {
        store_particle(atomicAdd(&state[S_NEW], 1u), p);
    }
}

@compute @workgroup_size(1)
fn spawn_prep() {
    let alive = atomicLoad(&state[S_NEW]);
    let room = fr.capacity - min(alive, fr.capacity);
    let n = min(fr.spawn_request, room);
    let base = atomicLoad(&state[S_NEXT_ID]);
    atomicStore(&state[S_BASE_ID], base);
    atomicStore(&state[S_NEXT_ID], base + n);
    atomicStore(&state[S_SPAWN_N], n);
    args[4] = groups(n);
    args[5] = 1u;
    args[6] = 1u;
}

@compute @workgroup_size(WG)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= atomicLoad(&state[S_SPAWN_N])) {
        return;
    }
    var p = new_particle(atomicLoad(&state[S_BASE_ID]) + k);
    spawn_particle(&p);
    if (p.alive) {
        store_particle(atomicAdd(&state[S_NEW], 1u), p);
    }
}

@compute @workgroup_size(1)
fn finish() {
    atomicStore(&state[S_COUNT], atomicLoad(&state[S_NEW]));
}
