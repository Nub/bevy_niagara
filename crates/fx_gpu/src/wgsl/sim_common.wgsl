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
    owner_velocity: vec3<f32>,  // Engine.Owner.Velocity; padded to 16 bytes
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

// ---- quaternions (x, y, z, w), as fx_core::sim::{euler_quat, quat_mul, quat_normalize} ----

const HALF_RADIANS_PER_DEGREE: f32 = 0.008726646;

// Rotation by Euler angles in degrees, X applied first: qz * qy * qx, written out.
fn euler_quat(deg: vec3<f32>) -> vec4<f32> {
    let h = deg * HALF_RADIANS_PER_DEGREE;
    let sx = sin(h.x);
    let cx = cos(h.x);
    let sy = sin(h.y);
    let cy = cos(h.y);
    let sz = sin(h.z);
    let cz = cos(h.z);
    return vec4<f32>(
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    );
}

// Hamilton product a * b (b's rotation first).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// q at unit length, or the identity if its length is 0 or not finite.
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let rl = 1.0 / length(q);
    if (rl > 0.0 && rl <= 3.4028235e38) {
        return q * rl;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// ---- common modules (fx_core::sim helpers of the same names) ----

const SUB_UV_LAST: f32 = 0.99999;
const TAU: f32 = 6.2831855;
const DEGREES: f32 = 0.017453292;

// (1 - dist / radius) ^ exponent, for 0 <= dist < radius.
fn falloff(dist: f32, radius: f32, exponent: f32) -> f32 {
    return pow(1.0 - dist / radius, exponent);
}

// v normalised, or +Y if it has no direction.
fn unit_or_y(v: vec3<f32>) -> vec3<f32> {
    let rl = 1.0 / length(v);
    if (rl > 0.0 && rl <= 3.4028235e38) {
        return v * rl;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// Unit direction around unit `axis` (counter-clockwise from its tip) and
// towards it, for offset r from a point on it; zero on the axis.
struct VortexDirs {
    tangent: vec3<f32>,
    inward: vec3<f32>,
}

fn vortex_dirs(axis: vec3<f32>, r: vec3<f32>) -> VortexDirs {
    let radial = r - axis * dot(axis, r);
    let len = length(radial);
    if (len > 0.0) {
        let o = radial / len;
        return VortexDirs(cross(axis, o), -o);
    }
    return VortexDirs(vec3<f32>(0.0), vec3<f32>(0.0));
}

// clamp(|velocity| / threshold, 0, 1); 1 when the threshold is not positive.
fn speed_fraction(velocity: vec3<f32>, threshold: f32) -> f32 {
    if (threshold > 0.0) {
        return clamp(length(velocity) / threshold, 0.0, 1.0);
    }
    return 1.0;
}

fn around(r: f32, phi: f32, y: f32) -> vec3<f32> {
    return vec3<f32>(r * cos(phi), y, r * sin(phi));
}

// Shape Location, one function per shape (fx_core::sim::shape_point); u are
// five uniform draws, radial ones used as 1 - u so they are never 0.
fn shape_sphere(radius: f32, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    let y = 1.0 - 2.0 * u[0];
    let phi = TAU * u[1];
    let ring = sqrt(max(1.0 - y * y, 0.0));
    let r = radius * select(pow(1.0 - u[2], 1.0 / 3.0), 1.0, surface);
    return vec3<f32>(ring * cos(phi), y, ring * sin(phi)) * r;
}

fn shape_box(size: vec3<f32>, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    var p = vec3<f32>(u[0] - 0.5, u[1] - 0.5, u[2] - 0.5) * size;
    if (surface) {
        let ax = size.y * size.z;
        let ay = size.x * size.z;
        let t = u[3] * (ax + ay + size.x * size.y);
        let side = select(0.5, -0.5, u[4] < 0.5);
        if (t < ax) {
            p.x = side * size.x;
        } else if (t < ax + ay) {
            p.y = side * size.y;
        } else {
            p.z = side * size.z;
        }
    }
    return p;
}

fn shape_cylinder(height: f32, radius: f32, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    let r = select(sqrt(1.0 - u[1]), 1.0, surface);
    return around(radius * r, TAU * u[0], (u[2] - 0.5) * height);
}

fn shape_torus(large: f32, handle: f32, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    let theta = TAU * u[0];
    let psi = TAU * u[1];
    let rho = handle * select(sqrt(1.0 - u[2]), 1.0, surface);
    return around(large + rho * cos(psi), theta, rho * sin(psi));
}

fn shape_cone(len: f32, angle: f32, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    let h = len * select(pow(1.0 - u[0], 1.0 / 3.0), sqrt(1.0 - u[0]), surface);
    let r = h * tan(clamp(angle, 0.0, 89.9) * DEGREES) * select(sqrt(1.0 - u[1]), 1.0, surface);
    return around(r, TAU * u[2], h);
}

fn shape_ring(radius: f32, coverage: f32, u: array<f32, 5>) -> vec3<f32> {
    return around(radius * sqrt(1.0 - clamp(coverage, 0.0, 1.0) * u[1]), TAU * u[0], 0.0);
}

// ---- particles: structure of arrays, word w of particle i at w * capacity + i ----
// Words 0..27 are the base attributes every emitter stores. Words 28..48 (Scale,
// MeshOrientation, the Ribbon attributes, LightRadius, DynamicMaterialParameter,
// SubImageIndex, Initial.Scale, Initial.RibbonWidth)
// exist only when the generated `EXT` is true; otherwise they stay at their
// defaults in registers and the buffers end at word 27.

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
    scale: vec3<f32>,
    mesh_orientation: vec4<f32>,
    ribbon_id: i32,
    ribbon_width: f32,
    ribbon_twist: f32,
    ribbon_link_order: f32,
    light_radius: f32,
    dynamic_material_parameter: vec4<f32>,
    sub_image_index: f32,
    initial_scale: vec3<f32>,
    initial_ribbon_width: f32,
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
    p.scale = vec3<f32>(1.0);
    p.mesh_orientation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    p.ribbon_id = 0;
    p.ribbon_width = 0.1;
    p.ribbon_twist = 0.0;
    p.ribbon_link_order = 0.0;
    p.light_radius = 1.0;
    p.dynamic_material_parameter = vec4<f32>(0.0);
    p.sub_image_index = 0.0;
    p.initial_scale = vec3<f32>(1.0);
    p.initial_ribbon_width = 0.1;
    p.id = id;
    p.alive = true;
    return p;
}

fn ld(w: u32, i: u32) -> f32 {
    return src[w * fr.capacity + i];
}

fn load_particle(i: u32) -> Particle {
    var p = new_particle(bitcast<u32>(ld(27u, i)));
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
    if (EXT) {
        p.scale = vec3<f32>(ld(28u, i), ld(29u, i), ld(30u, i));
        p.mesh_orientation = vec4<f32>(ld(31u, i), ld(32u, i), ld(33u, i), ld(34u, i));
        p.ribbon_id = bitcast<i32>(ld(35u, i));
        p.ribbon_width = ld(36u, i);
        p.ribbon_twist = ld(37u, i);
        p.ribbon_link_order = ld(38u, i);
        p.light_radius = ld(39u, i);
        p.dynamic_material_parameter = vec4<f32>(ld(40u, i), ld(41u, i), ld(42u, i), ld(43u, i));
        p.sub_image_index = ld(44u, i);
        p.initial_scale = vec3<f32>(ld(45u, i), ld(46u, i), ld(47u, i));
        p.initial_ribbon_width = ld(48u, i);
    }
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
    if (EXT) {
        st(28u, i, p.scale.x); st(29u, i, p.scale.y); st(30u, i, p.scale.z);
        st(31u, i, p.mesh_orientation.x); st(32u, i, p.mesh_orientation.y);
        st(33u, i, p.mesh_orientation.z); st(34u, i, p.mesh_orientation.w);
        st(35u, i, bitcast<f32>(p.ribbon_id));
        st(36u, i, p.ribbon_width);
        st(37u, i, p.ribbon_twist);
        st(38u, i, p.ribbon_link_order);
        st(39u, i, p.light_radius);
        st(40u, i, p.dynamic_material_parameter.x); st(41u, i, p.dynamic_material_parameter.y);
        st(42u, i, p.dynamic_material_parameter.z); st(43u, i, p.dynamic_material_parameter.w);
        st(44u, i, p.sub_image_index);
        st(45u, i, p.initial_scale.x); st(46u, i, p.initial_scale.y); st(47u, i, p.initial_scale.z);
        st(48u, i, p.initial_ribbon_width);
    }
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
