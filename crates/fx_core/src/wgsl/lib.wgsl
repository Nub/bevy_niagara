// Pure WGSL helpers shared by the generated GPU simulation (fx_gpu) and the
// validation of custom WGSL modules (fx_core::custom). No bindings here.

// ---- random numbers (fx_core::sim::rng) ----

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
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
    if (rl > 0.0 && rl <= 0x1.fffffep+127f) {
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
    if (rl > 0.0 && rl <= 0x1.fffffep+127f) {
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
    return shape_box_mid(size, vec3<f32>(0.5), surface, u);
}

// The box with Offset at `mid` (fractions of `size` per axis; 0.5 = centred).
fn shape_box_mid(size: vec3<f32>, mid: vec3<f32>, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    var p = (vec3<f32>(u[0], u[1], u[2]) - mid) * size;
    if (surface) {
        let ax = size.y * size.z;
        let ay = size.x * size.z;
        let t = u[3] * (ax + ay + size.x * size.y);
        let face = select(1.0, 0.0, u[4] < 0.5);
        if (t < ax) {
            p.x = (face - mid.x) * size.x;
        } else if (t < ax + ay) {
            p.y = (face - mid.y) * size.y;
        } else {
            p.z = (face - mid.z) * size.z;
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

// The cone with a hollow core (`inner` half-angle, degrees), a sweep of
// `radial` degrees and the fraction `empty` of it left empty at the apex.
fn shape_cone_ex(len: f32, angle: f32, inner: f32, radial: f32, empty: f32, surface: bool, u: array<f32, 5>) -> vec3<f32> {
    let w = 1.0 - u[0] * (1.0 - clamp(empty, 0.0, 1.0));
    let h = len * select(pow(w, 1.0 / 3.0), sqrt(w), surface);
    let outer = clamp(angle, 0.0, 89.9);
    let t = tan(outer * DEGREES);
    let core = clamp(inner, 0.0, 89.9);
    var r = h * t;
    if (!surface) {
        if (core > 0.0) {
            let ti = tan(min(core, outer) * DEGREES);
            r = h * sqrt(ti * ti + (t * t - ti * ti) * (1.0 - u[1]));
        } else {
            r = h * t * sqrt(1.0 - u[1]);
        }
    }
    return around(r, TAU * u[2] * (clamp(radial, 0.0, 360.0) / 360.0), h);
}

// Niagara's HSV colour adjustment (fx_core::sim::hsv_shift): a YIQ hue
// rotation of `hue` degrees scaled by saturation and value; alpha kept.
fn hsv_shift(c: vec4<f32>, hue: f32, saturation: f32, value: f32) -> vec4<f32> {
    let s = max(saturation, 0.0);
    let v = max(value, 0.0);
    let a = hue * DEGREES;
    let vc = v * s * cos(a);
    let vs = v * s * sin(a);
    let r = (0.299 * v + 0.701 * vc + 0.168 * vs) * c.x
        + (0.587 * v - 0.587 * vc + 0.330 * vs) * c.y
        + (0.114 * v - 0.114 * vc - 0.497 * vs) * c.z;
    let g = (0.299 * v - 0.299 * vc - 0.328 * vs) * c.x
        + (0.587 * v + 0.413 * vc + 0.035 * vs) * c.y
        + (0.114 * v - 0.114 * vc + 0.292 * vs) * c.z;
    let b = (0.299 * v - 0.300 * vc + 1.25 * vs) * c.x
        + (0.587 * v - 0.588 * vc - 1.05 * vs) * c.y
        + (0.114 * v + 0.886 * vc - 0.203 * vs) * c.z;
    return vec4<f32>(r, g, b, c.w);
}

// Add Velocity in Cone's speed factor (fx_core::sim::cone_speed_falloff).
fn cone_speed_falloff(cos_t: f32, falloff: f32) -> f32 {
    return 1.0 + (pow(clamp(cos_t, 0.0, 1.0), 10.0 * falloff) - 1.0) * falloff;
}

fn shape_ring(radius: f32, coverage: f32, u: array<f32, 5>) -> vec3<f32> {
    return around(radius * sqrt(1.0 - clamp(coverage, 0.0, 1.0) * u[1]), TAU * u[0], 0.0);
}


// ---- batch 2 modules (fx_core::sim helpers of the same names) ----

// v turned by unit quaternion q.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = cross(q.xyz, v) * 2.0;
    return v + t * q.w + cross(q.xyz, t);
}

// The rotation by rotation vector v (axis times angle in degrees).
fn rotation_vector_quat(v: vec3<f32>) -> vec4<f32> {
    let len = length(v);
    if (!(len > 0.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let half = len * HALF_RADIANS_PER_DEGREE;
    let s = sin(half) / len;
    return vec4<f32>(v * s, cos(half));
}

// v * (1 / length(v)) if that is finite and positive, else `fallback`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let rl = 1.0 / length(v);
    if (rl > 0.0 && rl <= 0x1.fffffep+127f) {
        return v * rl;
    }
    return fallback;
}

// Shortest rotation taking direction `from` (+X when zero) onto `to`.
fn shortest_arc(from_dir: vec3<f32>, to_dir: vec3<f32>) -> vec4<f32> {
    let a = normalize_or(from_dir, vec3<f32>(1.0, 0.0, 0.0));
    let b = normalize_or(to_dir, vec3<f32>(0.0));
    let d = dot(a, b);
    if (d < -0.999999) {
        return vec4<f32>(basis_t1(a), 0.0);
    }
    return quat_normalize(vec4<f32>(cross(a, b), 1.0 + d));
}

// Grid Location's lattice point for particle `id` on an n grid, -0.5..0.5 per axis.
fn grid_point(id: u32, n: vec3<u32>) -> vec3<f32> {
    let total = max(n.x * n.y * n.z, 1u);
    let i = id % total;
    let c = vec3<u32>(i % n.x, (i / n.x) % n.y, i / (n.x * n.y));
    var out = vec3<f32>(0.0);
    for (var k = 0; k < 3; k++) {
        if (n[k] > 1u) {
            out[k] = f32(c[k]) / f32(n[k] - 1u) - 0.5;
        }
    }
    return out;
}

// clamp((d - near) / (far - near), 0, 1); a step at far when far <= near.
fn distance_fraction(d: f32, near: f32, far: f32) -> f32 {
    if (far > near) {
        return clamp((d - near) / (far - near), 0.0, 1.0);
    }
    return select(0.0, 1.0, d >= far);
}

// Noise Force's field: gradient noise, octaves of double frequency and half amplitude.
fn noise_force(q: vec3<f32>, octaves: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var f = 1.0;
    var a = 1.0;
    for (var o = 0u; o < octaves; o++) {
        sum += noise_grad(q * f, 3u + o) * a;
        f *= 2.0;
        a *= 0.5;
    }
    return sum;
}
