//! World-space emitters (`local_space: false`, Niagara's Local Space off):
//! particles spawn in effect space, move to world space with the owner's
//! transform at their birth, and stay there when the owner moves on
//! (trails behind a flying arrow). Local-space particles follow the owner.

mod common;

use bevy_math::{Affine3A, Quat, Vec3};
use common::*;
use fx_core::sim::{OwnerPose, Simulation};

const DT: f32 = 1.0 / 30.0;
/// Owner speed along +X (m/s).
const SPEED: f32 = 6.0;

fn effect(local_space: bool, interpolated: bool) -> String {
    format!(
        r#"(
    name: "Trail",
    emitters: [(
        name: "E",
        max_particles: 1000,
        local_space: {local_space},
        interpolated_spawning: {interpolated},
        emitter_update: [(module: "Spawn Rate", inputs: {{ "Spawn Rate": Float(120.0) }})],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": Float(10.0), "Position": Vec3((0.0, 1.0, 0.0)) }}),
            (module: "Add Velocity", inputs: {{ "Velocity": Vec3((0.0, 0.0, 1.0)) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Solve Forces and Velocity"),
        ],
    )],
)"#
    )
}

/// Runs 1 s with the owner flying along +X from the origin, yawed 90° about
/// +Y; returns the particles' positions and velocities as the renderer
/// places them in the world (local-space ones through the owner transform).
fn fly(local_space: bool, interpolated: bool) -> Vec<(u32, Vec3, Vec3)> {
    let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let at = |t: f32| Affine3A::from_rotation_translation(rotation, Vec3::new(SPEED * t, 0.0, 0.0));
    let mut sim = Simulation::new(compile(&effect(local_space, interpolated)), 3);
    sim.teleport_owner(at(0.0));
    for f in 0..30 {
        sim.move_owner(at((f + 1) as f32 * DT));
        sim.step(DT);
    }
    let owner = sim.owner_pose;
    sim.particles(0)
        .iter()
        .map(|p| {
            if local_space {
                (p.unique_id, owner.transform_point(p.position), owner.transform_vector(p.velocity))
            } else {
                (p.unique_id, p.position, p.velocity)
            }
        })
        .collect()
}

#[test]
fn world_space_particles_stay_behind_a_moving_owner() {
    let parts = fly(false, false);
    assert!(parts.len() >= 100, "{}", parts.len());
    let xs: Vec<f32> = parts.iter().map(|p| p.1.x).collect();
    let (lo, hi) = xs.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    // Spread along the 6 m path (each particle drifts 1 m/s along world -X
    // after the 90° yaw of its +Z velocity, so the spread is 6 - 1 m).
    assert!(hi - lo > 4.0, "spread {lo}..{hi}");
    for (_, pos, vel) in &parts {
        // Velocity rotated with the owner: (0, 0, 1) yawed 90° about +Y is (1, 0, 0).
        assert!((*vel - Vec3::X).length() < 1e-4, "velocity {vel}");
        assert!((pos.y - 1.0).abs() < 1e-4 && pos.z.abs() < 1e-3, "{pos}");
        assert!(pos.x <= SPEED + 1.0 + 1e-3, "ahead of the owner: {pos}");
    }
    // The oldest particle is furthest behind the owner.
    let oldest = parts.iter().min_by_key(|p| p.0).unwrap();
    let newest = parts.iter().max_by_key(|p| p.0).unwrap();
    assert!(newest.1.x - oldest.1.x > 4.0, "{oldest:?} {newest:?}");
}

#[test]
fn local_space_particles_follow_the_owner() {
    let parts = fly(true, false);
    assert!(parts.len() >= 100);
    // Every particle is where the owner is (plus its own drift along the
    // yawed +Z, i.e. world +X, at most 1 m after 1 s).
    for (_, pos, _) in &parts {
        assert!(pos.x >= SPEED - 1e-3 && pos.x <= SPEED + 1.0 + 1e-3, "{pos}");
    }
}

#[test]
fn interpolated_spawning_spreads_world_space_particles_evenly() {
    // 120 particles/s at 30 steps/s: 4 per step. Without interpolation the
    // ones of a step are born at the same place (clumps 0.2 m apart); with
    // it they are spread over the step's 0.2 m of path.
    let gaps = |interpolated: bool| {
        let mut xs: Vec<f32> = fly(false, interpolated).iter().map(|p| p.1.x).collect();
        xs.sort_by(f32::total_cmp);
        xs.windows(2).map(|w| w[1] - w[0]).fold((f32::MAX, 0.0f32), |(a, b), g| (a.min(g), b.max(g)))
    };
    let (min, max) = gaps(false);
    assert!(min < 1e-3 && max > 0.15, "clumped: {min} {max}");
    let (min, max) = gaps(true);
    // Each particle also drifts at 1 m/s, so neighbours are (6 - 1) / 120 m apart.
    let even = (SPEED - 1.0) / 120.0;
    assert!((min - even).abs() < 2e-3 && (max - even).abs() < 2e-3, "even: {min} {max} want {even}");
}

#[test]
fn owner_pose_lerps_and_round_trips() {
    let a = Affine3A::from_scale_rotation_translation(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(0.3), Vec3::new(4.0, 5.0, 6.0));
    let p = OwnerPose::from_affine(a);
    let x = Vec3::new(0.5, -1.0, 2.0);
    assert!((p.transform_point(x) - a.transform_point3(x)).length() < 1e-5);
    assert!((p.transform_vector(x) - a.transform_vector3(x)).length() < 1e-5);
    let b = OwnerPose {
        translation: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::from_rotation_z(1.3),
        scale: Vec3::ONE,
    };
    let mid = p.lerp(&b, 0.5);
    assert!((mid.translation - Vec3::new(7.0, 2.5, 3.0)).length() < 1e-5);
    assert!(mid.rotation.angle_between(Quat::from_rotation_z(0.8)) < 1e-4);
    assert_eq!(p.lerp(&b, 0.0), p);
    assert_eq!(p.lerp(&b, 1.0), b);
}

#[test]
fn local_space_defaults_to_true_and_survives_cooking() {
    let sys = compile(&effect(true, false).replace("        local_space: true,\n", ""));
    assert!(sys.emitters[0].local_space);
    let sys = compile(&effect(false, false));
    assert!(!sys.emitters[0].local_space);
    let bytes = fx_core::cook::to_bytes(&sys);
    assert!(!fx_core::cook::from_bytes(&bytes).unwrap().emitters[0].local_space);
}
