//! Gradient noise with analytic derivatives, and its curl.

use bevy_math::Vec3;

use super::rng::pcg;

/// The 12 edge-centre gradients of a cube (Perlin's improved-noise set).
const GRADIENTS: [Vec3; 12] = [
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, -1.0),
];

/// Gradient at integer lattice point `(x, y, z)` for noise channel `channel`.
pub fn lattice_gradient(x: i32, y: i32, z: i32, channel: u32) -> Vec3 {
    let h = pcg(pcg(pcg(pcg(channel) ^ x as u32) ^ y as u32) ^ z as u32);
    GRADIENTS[(h % 12) as usize]
}

/// Perlin gradient noise at `p` with its gradient: `(value, d value / d p)`.
/// Value is 0 at every lattice point, where the gradient equals the lattice gradient.
pub fn gradient_noise(p: Vec3, channel: u32) -> (f32, Vec3) {
    let i = p.floor();
    let f = p - i;
    let (ix, iy, iz) = (i.x as i32, i.y as i32, i.z as i32);
    // Quintic fade and its derivative.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let g = |dx: i32, dy: i32, dz: i32| lattice_gradient(ix + dx, iy + dy, iz + dz, channel);
    let (ga, gb, gc, gd) = (g(0, 0, 0), g(1, 0, 0), g(0, 1, 0), g(1, 1, 0));
    let (ge, gf, gg, gh) = (g(0, 0, 1), g(1, 0, 1), g(0, 1, 1), g(1, 1, 1));
    let va = ga.dot(f);
    let vb = gb.dot(f - Vec3::new(1.0, 0.0, 0.0));
    let vc = gc.dot(f - Vec3::new(0.0, 1.0, 0.0));
    let vd = gd.dot(f - Vec3::new(1.0, 1.0, 0.0));
    let ve = ge.dot(f - Vec3::new(0.0, 0.0, 1.0));
    let vf = gf.dot(f - Vec3::new(1.0, 0.0, 1.0));
    let vg = gg.dot(f - Vec3::new(0.0, 1.0, 1.0));
    let vh = gh.dot(f - Vec3::new(1.0, 1.0, 1.0));

    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = ve - va;
    let k4 = va - vb - vc + vd;
    let k5 = va - vc - ve + vg;
    let k6 = va - vb - ve + vf;
    let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
    let value = va
        + u.x * k1
        + u.y * k2
        + u.z * k3
        + u.x * u.y * k4
        + u.y * u.z * k5
        + u.z * u.x * k6
        + u.x * u.y * u.z * k7;

    let gk1 = gb - ga;
    let gk2 = gc - ga;
    let gk3 = ge - ga;
    let gk4 = ga - gb - gc + gd;
    let gk5 = ga - gc - ge + gg;
    let gk6 = ga - gb - ge + gf;
    let gk7 = -ga + gb + gc - gd + ge - gf - gg + gh;
    let mut grad = ga
        + u.x * gk1
        + u.y * gk2
        + u.z * gk3
        + u.x * u.y * gk4
        + u.y * u.z * gk5
        + u.z * u.x * gk6
        + u.x * u.y * u.z * gk7;
    grad += du
        * Vec3::new(
            k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
            k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
            k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y,
        );
    (value, grad)
}

/// Curl of the vector potential `(N0(p), N1(p), N2(p))` built from three
/// independent noise channels. Divergence-free by construction.
pub fn curl_noise(p: Vec3) -> Vec3 {
    let (_, g0) = gradient_noise(p, 0);
    let (_, g1) = gradient_noise(p, 1);
    let (_, g2) = gradient_noise(p, 2);
    Vec3::new(g2.y - g1.z, g0.z - g2.x, g1.x - g0.y)
}
