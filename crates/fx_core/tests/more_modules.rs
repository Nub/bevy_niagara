//! The second batch of Niagara modules (Set modules, rotation, distance and
//! speed scaling, kills, noise), the new spawn modules, Emitter State loops
//! and the system / emitter options (warm-up, determinism, fixed bounds),
//! each against values worked out by hand. dt is 0.25 and the inputs powers
//! of two, so most expected values are exact.

mod common;

use bevy_math::{Quat, Vec2, Vec3, Vec4};
use common::*;
use fx_core::lifecycle::Lifecycle;
use fx_core::sim::{Particle, Simulation, euler_quat, noise_force, quat_rotate};

const DT: f32 = 0.25;
const STATE: &str = r#"(module: "Particle State")"#;
const SOLVE: &str = r#"(module: "Solve Forces and Velocity")"#;

fn v3(v: Vec3) -> String {
    format!("Vec3(({:?}, {:?}, {:?}))", v.x, v.y, v.z)
}

/// Initialize Particle at `pos`.
fn at(pos: Vec3) -> String {
    format!(r#"(module: "Initialize Particle", inputs: {{ "Position": {} }})"#, v3(pos))
}

fn set_velocity(v: Vec3) -> String {
    format!(r#"(module: "Set Particle Velocity", inputs: {{ "Velocity": {} }})"#, v3(v))
}

/// The particle right after Particle Spawn (step 1).
fn spawned(spawn: &str) -> Particle {
    let mut sim = single_particle("", spawn, "");
    sim.step(DT);
    sim.particles(0)[0]
}

/// Spawns one particle (step 1) and runs `update` on it `n` times.
fn after_updates(spawn: &str, update: &str, n: usize) -> Particle {
    let mut sim = single_particle("", spawn, update);
    for _ in 0..=n {
        sim.step(DT);
    }
    sim.particles(0)[0]
}

fn after_one_update(spawn: &str, update: &str) -> Particle {
    after_updates(spawn, update, 1)
}

fn alive_after_one_update(spawn: &str, update: &str) -> bool {
    let mut sim = single_particle("", spawn, update);
    sim.step(DT);
    sim.step(DT);
    !sim.particles(0).is_empty()
}

fn assert_v3(a: Vec3, b: Vec3) {
    assert_vec_close(&a.to_array(), &b.to_array());
}

/// One-emitter effect with extra system and emitter fields.
fn effect(
    system_fields: &str,
    emitter_fields: &str,
    emitter_update: &str,
    particle_spawn: &str,
    particle_update: &str,
) -> String {
    format!(
        "(
    name: \"Test\",
    {system_fields}
    user_parameters: {{ }},
    emitters: [(
        name: \"E\",
        max_particles: 1000,
        {emitter_fields}
        emitter_update: [ {emitter_update} ],
        particle_spawn: [ {particle_spawn} ],
        particle_update: [ {particle_update} ],
    )],
)"
    )
}

/// Particles spawned in each of `steps` steps by `emitter_update`.
fn spawn_counts(emitter_update: &str, steps: usize) -> Vec<u32> {
    let mut sim = Simulation::new(compile(&one_emitter("", emitter_update, "", "")), 1);
    (0..steps)
        .map(|_| {
            sim.step(DT);
            sim.emitters[0].spawned_last_step
        })
        .collect()
}

/// Error messages of an effect that must not compile.
fn cook_errors(src: &str) -> Vec<String> {
    let c = fx_core::check_source("test.fx.ron", src);
    let errors: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    assert!(!errors.is_empty(), "expected errors for:\n{src}");
    assert!(c.cooked.is_none());
    errors
}

fn has_error(errors: &[String], needle: &str) -> bool {
    errors.iter().any(|e| e.contains(needle))
}

// ---- Set modules ----

#[test]
fn set_modules_write_their_attribute() {
    let p = spawned(
        r#"(module: "Set Particle Lifetime", inputs: { "Lifetime": Float(4.0) }),
           (module: "Set Particle Mass", inputs: { "Mass": Float(2.0) }),
           (module: "Set Particle Sprite Rotation", inputs: { "Sprite Rotation": Float(45.0) }),
           (module: "Set Particle Velocity", inputs: { "Velocity": Vec3((1.0, 2.0, 3.0)) }),
           (module: "Set Particle Position", inputs: { "Position": Vec3((4.0, 5.0, 6.0)) })"#,
    );
    assert_eq!(p.lifetime, 4.0);
    assert_eq!(p.mass, 2.0);
    assert_eq!(p.sprite_rotation, 45.0);
    assert_eq!(p.velocity, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(p.position, Vec3::new(4.0, 5.0, 6.0));
}

#[test]
fn set_velocity_replaces_and_set_position_overrides_initialize() {
    let p = spawned(&format!(
        r#"{}, (module: "Add Velocity", inputs: {{ "Velocity": Vec3((8.0, 8.0, 8.0)) }}), {},
           (module: "Set Particle Position", inputs: {{ "Position": Vec3((0.5, 0.0, 0.0)) }})"#,
        at(Vec3::new(2.0, 2.0, 2.0)),
        set_velocity(Vec3::new(0.0, 1.0, 0.0))
    ));
    assert_eq!(p.velocity, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(p.position, Vec3::new(0.5, 0.0, 0.0));
}

#[test]
fn set_sprite_size_with_and_without_initial() {
    let size = |initial: bool| {
        spawned(&format!(
            r#"(module: "Set Particle Sprite Size", inputs: {{ "Sprite Size": Vec2((0.5, 0.25)), "Set Initial": Bool({initial}) }})"#
        ))
    };
    let p = size(false);
    assert_eq!(p.sprite_size, Vec2::new(0.5, 0.25));
    assert_eq!(p.initial_sprite_size, Vec2::splat(0.1));
    let p = size(true);
    assert_eq!(p.sprite_size, Vec2::new(0.5, 0.25));
    assert_eq!(p.initial_sprite_size, Vec2::new(0.5, 0.25));
    // Scale Sprite Size multiplies the initial size.
    let p = after_one_update(
        r#"(module: "Set Particle Sprite Size", inputs: { "Sprite Size": Vec2((0.5, 0.25)), "Set Initial": Bool(true) })"#,
        r#"(module: "Scale Sprite Size", inputs: { "Scale Factor": Vec2((2.0, 4.0)) })"#,
    );
    assert_eq!(p.sprite_size, Vec2::new(1.0, 1.0));
}

#[test]
fn set_scale_with_and_without_initial() {
    let scale = |initial: bool| {
        spawned(&format!(
            r#"(module: "Set Particle Scale", inputs: {{ "Scale": Vec3((2.0, 4.0, 0.5)), "Set Initial": Bool({initial}) }})"#
        ))
    };
    let p = scale(false);
    assert_eq!(p.scale, Vec3::new(2.0, 4.0, 0.5));
    assert_eq!(p.initial_scale, Vec3::ONE);
    let p = scale(true);
    assert_eq!(p.scale, Vec3::new(2.0, 4.0, 0.5));
    assert_eq!(p.initial_scale, Vec3::new(2.0, 4.0, 0.5));
}

#[test]
fn set_lifetime_drives_particle_state() {
    // Lifetime 0.5: NormalizedAge 0.5 after one update, dead after the third (age 0.75 > 0.5).
    let spawn = r#"(module: "Set Particle Lifetime", inputs: { "Lifetime": Float(0.5) })"#;
    let p = after_one_update(spawn, STATE);
    assert_eq!(p.normalized_age, 0.5);
    let mut sim = single_particle("", spawn, STATE);
    for _ in 0..3 {
        sim.step(DT);
    }
    assert_eq!(sim.particles(0).len(), 1);
    sim.step(DT);
    assert!(sim.particles(0).is_empty());
}

// ---- velocity ----

#[test]
fn add_velocity_from_point_pushes_away() {
    // From (1, 0, 0) to (2, 0, 0): +X, times 2, times scale (0.5, 1, 1).
    let from_point = r#"(module: "Add Velocity from Point", inputs: {
        "Velocity Strength": Float(2.0), "Velocity Origin": Vec3((1.0, 0.0, 0.0)), "Velocity Scale": Vec3((0.5, 1.0, 1.0)) })"#;
    let p = spawned(&format!("{}, {from_point}", at(Vec3::new(2.0, 0.0, 0.0))));
    assert_eq!(p.velocity, Vec3::new(1.0, 0.0, 0.0));
    // Below the origin: -Y.
    let p = spawned(&format!("{}, {from_point}", at(Vec3::new(1.0, -3.0, 0.0))));
    assert_eq!(p.velocity, Vec3::new(0.0, -2.0, 0.0));
    // At the origin: +Y.
    let p = spawned(&format!("{}, {from_point}", at(Vec3::new(1.0, 0.0, 0.0))));
    assert_eq!(p.velocity, Vec3::new(0.0, 2.0, 0.0));
    // Adds to what is there.
    let p = spawned(&format!("{}, {}, {from_point}", at(Vec3::new(2.0, 0.0, 0.0)), set_velocity(Vec3::new(0.0, 0.0, 1.0))));
    assert_eq!(p.velocity, Vec3::new(1.0, 0.0, 1.0));
}

#[test]
fn limit_velocity_caps_speed() {
    let limit = |l: f32| format!(r#"(module: "Limit Velocity", inputs: {{ "Velocity Limit": Float({l:?}) }})"#);
    let spawn = set_velocity(Vec3::new(3.0, 4.0, 0.0));
    // Speed 5 limited to 2.5: halved.
    assert_eq!(after_one_update(&spawn, &limit(2.5)).velocity, Vec3::new(1.5, 2.0, 0.0));
    assert_eq!(after_one_update(&spawn, &limit(10.0)).velocity, Vec3::new(3.0, 4.0, 0.0));
    assert_eq!(after_one_update(&spawn, &limit(0.0)).velocity, Vec3::ZERO);
}

#[test]
fn scale_velocity_multiplies_per_axis() {
    let scale = r#"(module: "Scale Velocity", inputs: { "Velocity Scale": Vec3((2.0, 0.5, -1.0)) })"#;
    let p = spawned(&format!("{}, {scale}", set_velocity(Vec3::new(1.0, 2.0, 3.0))));
    assert_eq!(p.velocity, Vec3::new(2.0, 1.0, -3.0));
    // In Particle Update it compounds every step.
    let p = after_updates(&set_velocity(Vec3::new(1.0, 2.0, 3.0)), scale, 2);
    assert_eq!(p.velocity, Vec3::new(4.0, 0.5, 3.0));
}

// ---- Grid Location ----

#[test]
fn grid_location_fills_the_lattice_and_wraps() {
    // 2 x 3 x 1 points over 2 x 4 x 6 metres, moved by (1, 0, 0); ids 6 and 7 wrap to 0 and 1.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(8) })"#,
        r#"(module: "Grid Location", inputs: { "Grid Size": Vec3((2.0, 4.0, 6.0)),
            "X Count": Int(2), "Y Count": Int(3), "Z Count": Int(1), "Offset": Vec3((1.0, 0.0, 0.0)) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 1);
    sim.step(DT);
    let ps = sim.particles(0);
    assert_eq!(ps.len(), 8);
    let expected = [
        (0.0, -2.0),
        (2.0, -2.0),
        (0.0, 0.0),
        (2.0, 0.0),
        (0.0, 2.0),
        (2.0, 2.0),
        (0.0, -2.0),
        (2.0, -2.0),
    ];
    for p in ps {
        let (x, y) = expected[p.unique_id as usize];
        assert_eq!(p.position, Vec3::new(x, y, 0.0), "id {}", p.unique_id);
    }
}

#[test]
fn grid_point_by_hand() {
    use fx_core::sim::grid_point;
    // 3 x 2 x 2: X fastest, then Y, then Z.
    let n = [3, 2, 2];
    assert_eq!(grid_point(0, n), Vec3::new(-0.5, -0.5, -0.5));
    assert_eq!(grid_point(1, n), Vec3::new(0.0, -0.5, -0.5));
    assert_eq!(grid_point(2, n), Vec3::new(0.5, -0.5, -0.5));
    assert_eq!(grid_point(3, n), Vec3::new(-0.5, 0.5, -0.5));
    assert_eq!(grid_point(6, n), Vec3::new(-0.5, -0.5, 0.5));
    assert_eq!(grid_point(11, n), Vec3::new(0.5, 0.5, 0.5));
    assert_eq!(grid_point(12, n), grid_point(0, n));
    assert_eq!(grid_point(13, n), grid_point(1, n));
    // One point on every axis: the centre.
    assert_eq!(grid_point(5, [1, 1, 1]), Vec3::ZERO);
}

// ---- rotation ----

#[test]
fn rotational_velocity_turns_mesh_in_solve() {
    // 90°/s around Y for 1 s in four steps: X goes to -Z.
    let spawn = r#"(module: "Add Rotational Velocity", inputs: { "Rotational Velocity": Vec3((0.0, 90.0, 0.0)) })"#;
    let p = after_updates(spawn, SOLVE, 4);
    assert_eq!(p.rotational_velocity, Vec3::new(0.0, 90.0, 0.0));
    let q = p.mesh_orientation;
    assert_v3(quat_rotate(q, Vec3::X), Vec3::new(0.0, 0.0, -1.0));
    assert_v3(quat_rotate(q, Vec3::Y), Vec3::Y);
    let want = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    assert_vec_close(&q.to_array(), &want.to_array());
    // After one step: 22.5°.
    let p = after_updates(spawn, SOLVE, 1);
    let want = Quat::from_rotation_y(22.5f32.to_radians());
    assert_vec_close(&p.mesh_orientation.to_array(), &want.to_array());
    // Rotation vectors add.
    let p = spawned(&format!("{spawn}, {spawn}"));
    assert_eq!(p.rotational_velocity, Vec3::new(0.0, 180.0, 0.0));
}

#[test]
fn solve_without_rotation_keeps_orientation_bits() {
    let spawn = r#"(module: "Initialize Mesh", inputs: { "Mesh Rotation": Vec3((30.0, 40.0, 50.0)) })"#;
    let update = format!(r#"(module: "Linear Force", inputs: {{ "Linear Force": Vec3((1.0, 0.0, 0.0)) }}), {SOLVE}"#);
    let mut sim = single_particle("", spawn, &update);
    sim.step(DT);
    let before = sim.particles(0)[0].mesh_orientation;
    assert_eq!(before.to_array(), euler_quat(Vec3::new(30.0, 40.0, 50.0)).to_array());
    for _ in 0..4 {
        sim.step(DT);
    }
    let p = sim.particles(0)[0];
    assert!(p.position.x > 0.0);
    let bits = |q: Quat| q.to_array().map(f32::to_bits);
    assert_eq!(bits(p.mesh_orientation), bits(before));
}

#[test]
fn mesh_rotation_force_divides_by_mass() {
    // 8 °/s² on 2 kg for 0.25 s: +1 °/s; the force is cleared.
    let p = after_one_update(
        r#"(module: "Set Particle Mass", inputs: { "Mass": Float(2.0) })"#,
        &format!(r#"(module: "Mesh Rotation Force", inputs: {{ "Rotation Force": Vec3((0.0, 8.0, 0.0)) }}), {SOLVE}"#),
    );
    assert_eq!(p.rotational_velocity, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(p.rotational_force, Vec3::ZERO);
    let want = Quat::from_rotation_y(0.25f32.to_radians());
    assert_vec_close(&p.mesh_orientation.to_array(), &want.to_array());
    // Without the Solve the torque only accumulates.
    let p = after_one_update(
        "",
        r#"(module: "Mesh Rotation Force", inputs: { "Rotation Force": Vec3((0.0, 8.0, 0.0)) }),
           (module: "Mesh Rotation Force", inputs: { "Rotation Force": Vec3((1.0, 0.0, 0.0)) })"#,
    );
    assert_eq!(p.rotational_force, Vec3::new(1.0, 8.0, 0.0));
    assert_eq!(p.rotational_velocity, Vec3::ZERO);
}

#[test]
fn rotational_drag_decays_exponentially() {
    let spawn = r#"(module: "Add Rotational Velocity", inputs: { "Rotational Velocity": Vec3((0.0, 8.0, 0.0)) })"#;
    let update = format!(r#"(module: "Rotational Drag", inputs: {{ "Rotational Drag": Float(2.0) }}), {SOLVE}"#);
    let p = after_updates(spawn, &update, 1);
    assert_close(p.rotational_velocity.y, 8.0 * (-0.5f32).exp());
    assert_eq!(p.rotational_drag, 0.0);
    let p = after_updates(spawn, &update, 2);
    assert_close(p.rotational_velocity.y, 8.0 * (-1.0f32).exp());
    // Drag accumulates within a step.
    let p = after_one_update(spawn, r#"(module: "Rotational Drag", inputs: { "Rotational Drag": Float(2.0) }), (module: "Rotational Drag")"#);
    assert_eq!(p.rotational_drag, 3.0);
}

#[test]
fn orient_mesh_to_vector() {
    let orient = |v: Vec3, axis: Vec3| {
        format!(
            r#"(module: "Orient Mesh to Vector", inputs: {{ "Orientation Vector": {}, "Mesh Axis": {} }})"#,
            v3(v),
            v3(axis)
        )
    };
    // X onto Y: 90° around Z.
    let p = spawned(&orient(Vec3::new(0.0, 2.0, 0.0), Vec3::X));
    let h = std::f32::consts::FRAC_1_SQRT_2;
    assert_vec_close(&p.mesh_orientation.to_array(), &[0.0, 0.0, h, h]);
    assert_v3(quat_rotate(p.mesh_orientation, Vec3::X), Vec3::Y);
    // Opposite vectors: a half turn that takes X to -X.
    let p = spawned(&orient(Vec3::new(-4.0, 0.0, 0.0), Vec3::X));
    assert_close(p.mesh_orientation.w, 0.0);
    assert_v3(quat_rotate(p.mesh_orientation, Vec3::X), -Vec3::X);
    // Mesh axis Z onto the velocity (1, 1, 0) / √2.
    let p = spawned(&orient(Vec3::new(1.0, 1.0, 0.0), Vec3::Z));
    assert_v3(quat_rotate(p.mesh_orientation, Vec3::Z), Vec3::new(h, h, 0.0));
    // Zero vector: the orientation stays.
    let p = spawned(&format!(
        r#"(module: "Initialize Mesh", inputs: {{ "Mesh Rotation": Vec3((0.0, 90.0, 0.0)) }}), {}"#,
        orient(Vec3::ZERO, Vec3::X)
    ));
    assert_eq!(p.mesh_orientation, euler_quat(Vec3::new(0.0, 90.0, 0.0)));
}

#[test]
fn align_sprite_to_mesh_orientation() {
    // 90° around X: Y -> Z, Z -> -Y.
    let p = spawned(
        r#"(module: "Initialize Mesh", inputs: { "Mesh Rotation": Vec3((90.0, 0.0, 0.0)) }),
           (module: "Align Sprite to Mesh Orientation")"#,
    );
    assert_v3(p.sprite_facing, Vec3::new(0.0, -1.0, 0.0));
    assert_v3(p.sprite_alignment, Vec3::new(0.0, 0.0, 1.0));
    // Identity orientation: the axes as given.
    let p = spawned(
        r#"(module: "Align Sprite to Mesh Orientation", inputs: { "Facing Axis": Vec3((1.0, 0.0, 0.0)), "Alignment Axis": Vec3((0.0, 0.0, 2.0)) })"#,
    );
    assert_eq!(p.sprite_facing, Vec3::X);
    assert_eq!(p.sprite_alignment, Vec3::new(0.0, 0.0, 2.0));
}

#[test]
fn set_sprite_facing_and_alignment() {
    let p = spawned("");
    assert_eq!(p.sprite_facing, Vec3::ZERO);
    assert_eq!(p.sprite_alignment, Vec3::ZERO);
    let p = spawned(
        r#"(module: "Set Sprite Facing", inputs: { "Sprite Facing": Vec3((1.0, 0.0, 0.0)) }),
           (module: "Set Sprite Alignment", inputs: { "Sprite Alignment": Vec3((0.0, 0.0, -1.0)) })"#,
    );
    assert_eq!(p.sprite_facing, Vec3::X);
    assert_eq!(p.sprite_alignment, Vec3::new(0.0, 0.0, -1.0));
    // Defaults.
    let p = spawned(r#"(module: "Set Sprite Facing"), (module: "Set Sprite Alignment")"#);
    assert_eq!(p.sprite_facing, Vec3::Z);
    assert_eq!(p.sprite_alignment, Vec3::Y);
}

// ---- scaling by speed and distance ----

#[test]
fn scale_mesh_size_by_speed() {
    let set_scale = r#"(module: "Set Particle Scale", inputs: { "Scale": Vec3((2.0, 2.0, 2.0)), "Set Initial": Bool(true) })"#;
    let by_speed = r#"(module: "Scale Mesh Size by Speed", inputs: {
        "Velocity Threshold": Float(1.0), "Min Scale Factor": Vec3((1.0, 1.0, 1.0)), "Max Scale Factor": Vec3((1.0, 1.0, 2.0)) })"#;
    // Half the threshold: halfway, times the initial 2.
    let p = spawned(&format!("{set_scale}, {}, {by_speed}", set_velocity(Vec3::new(0.0, 0.5, 0.0))));
    assert_eq!(p.scale, Vec3::new(2.0, 2.0, 3.0));
    // At rest: Min.
    let p = spawned(&format!("{set_scale}, {by_speed}"));
    assert_eq!(p.scale, Vec3::new(2.0, 2.0, 2.0));
    // Faster than the threshold: Max.
    let p = spawned(&format!("{set_scale}, {}, {by_speed}", set_velocity(Vec3::new(0.0, 0.0, 8.0))));
    assert_eq!(p.scale, Vec3::new(2.0, 2.0, 4.0));
}

#[test]
fn scale_color_by_distance() {
    let init = |pos: Vec3| {
        format!(
            r#"(module: "Initialize Particle", inputs: {{ "Position": {}, "Color": Color((1.0, 0.5, 0.25, 1.0)) }})"#,
            v3(pos)
        )
    };
    let by_distance = r#"(module: "Scale Color by Distance", inputs: {
        "Reference Position": Vec3((0.0, 1.0, 0.0)), "Near Distance": Float(0.0), "Far Distance": Float(8.0) })"#;
    // Distance 4 of 8: alpha halfway from 1 to 0.
    let p = spawned(&format!("{}, {by_distance}", init(Vec3::new(0.0, 5.0, 0.0))));
    assert_eq!(p.color, Vec4::new(1.0, 0.5, 0.25, 0.5));
    // Beyond Far: the far factor.
    let p = spawned(&format!("{}, {by_distance}", init(Vec3::new(16.0, 1.0, 0.0))));
    assert_eq!(p.color, Vec4::new(1.0, 0.5, 0.25, 0.0));
    // At the reference: the near factor.
    let p = spawned(&format!("{}, {by_distance}", init(Vec3::new(0.0, 1.0, 0.0))));
    assert_eq!(p.color, Vec4::new(1.0, 0.5, 0.25, 1.0));
}

#[test]
fn scale_sprite_size_by_distance() {
    let init = |pos: Vec3| {
        format!(
            r#"(module: "Initialize Particle", inputs: {{ "Position": {}, "Sprite Size": Vec2((0.5, 0.25)) }})"#,
            v3(pos)
        )
    };
    let by_distance = r#"(module: "Scale Sprite Size by Distance", inputs: { "Near Distance": Float(0.0), "Far Distance": Float(8.0) })"#;
    // Distance 2 of 8: a quarter from 1 to 2.
    let p = spawned(&format!("{}, {by_distance}", init(Vec3::new(0.0, 0.0, 2.0))));
    assert_eq!(p.sprite_size, Vec2::new(0.625, 0.3125));
    let p = spawned(&format!("{}, {by_distance}", init(Vec3::new(0.0, 0.0, 100.0))));
    assert_eq!(p.sprite_size, Vec2::new(1.0, 0.5));
    // Near beyond the particle: Near Scale Factor.
    let near = r#"(module: "Scale Sprite Size by Distance", inputs: { "Near Distance": Float(4.0), "Far Distance": Float(8.0), "Near Scale Factor": Vec2((0.5, 0.5)) })"#;
    let p = spawned(&format!("{}, {near}", init(Vec3::new(0.0, 0.0, 2.0))));
    assert_eq!(p.sprite_size, Vec2::new(0.25, 0.125));
}

#[test]
fn scale_ribbon_width_by_speed() {
    let spawn = |speed: f32| {
        format!(
            r#"(module: "Initialize Ribbon", inputs: {{ "Ribbon Width": Float(0.5) }}), {}"#,
            set_velocity(Vec3::new(speed, 0.0, 0.0))
        )
    };
    let by_speed = r#"(module: "Scale Ribbon Width by Speed", inputs: {
        "Velocity Threshold": Float(1.0), "Min Scale Factor": Float(1.0), "Max Scale Factor": Float(0.25) })"#;
    // Speed 0.5: factor 0.625 of the initial 0.5.
    assert_eq!(after_one_update(&spawn(0.5), by_speed).ribbon_width, 0.3125);
    assert_eq!(after_one_update(&spawn(0.0), by_speed).ribbon_width, 0.5);
    assert_eq!(after_one_update(&spawn(4.0), by_speed).ribbon_width, 0.125);
}

#[test]
fn ribbon_twist_rate_adds_per_second() {
    let spawn = r#"(module: "Initialize Ribbon", inputs: { "Ribbon Twist": Float(10.0) })"#;
    let twist = r#"(module: "Ribbon Twist Rate", inputs: { "Twist Rate": Float(90.0) })"#;
    assert_eq!(after_updates(spawn, twist, 1).ribbon_twist, 32.5);
    assert_eq!(after_updates(spawn, twist, 2).ribbon_twist, 55.0);
}

#[test]
fn dynamic_material_parameters_2() {
    let p = spawned(
        r#"(module: "Dynamic Material Parameters 2", inputs: { "Dynamic Material Parameter 1": Vec4((1.0, 2.0, 3.0, 4.0)) })"#,
    );
    assert_eq!(p.dynamic_material_parameter_1, Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(p.dynamic_material_parameter, Vec4::ZERO);
}

// ---- jitter, kills, orbits, noise ----

#[test]
fn jitter_position_stays_within_amount() {
    let amount = Vec3::new(0.5, 1.0, 2.0);
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })"#,
        &at(Vec3::new(1.0, 1.0, 1.0)),
        r#"(module: "Jitter Position", inputs: { "Jitter Amount": Vec3((0.5, 1.0, 2.0)) })"#,
    );
    let mut sim = Simulation::new(compile(&src), 1);
    sim.step(DT);
    sim.step(DT);
    let ps = sim.particles(0);
    assert_eq!(ps.len(), 200);
    let mut moved_far = [false; 3];
    for p in ps {
        let d = p.position - Vec3::ONE;
        for c in 0..3 {
            assert!(d[c].abs() <= amount[c], "{d:?}");
            moved_far[c] |= d[c].abs() > amount[c] * 0.5;
        }
    }
    assert_eq!(moved_far, [true; 3], "offsets should cover the range");
    // Every step moves again.
    let before: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
    sim.step(DT);
    assert!(sim.particles(0).iter().zip(&before).all(|(p, b)| p.position != *b));
}

#[test]
fn jitter_delay_fires_once_per_delay() {
    // Particle State first: ages 0.25, 0.5, 0.75, 1.0; with a 0.5 s delay it jitters at 0.5 and 1.0.
    let update = format!(
        r#"{STATE}, (module: "Jitter Position", inputs: {{ "Jitter Amount": Vec3((1.0, 1.0, 1.0)), "Jitter Delay": Float(0.5) }})"#
    );
    let mut sim = single_particle("", "", &update);
    sim.step(DT);
    let mut positions = vec![sim.particles(0)[0].position];
    for _ in 0..4 {
        sim.step(DT);
        positions.push(sim.particles(0)[0].position);
    }
    let moved: Vec<bool> = positions.windows(2).map(|w| w[0] != w[1]).collect();
    assert_eq!(moved, [false, true, false, true]);
}

#[test]
fn kill_particles_by_speed() {
    let spawn = set_velocity(Vec3::new(2.0, 0.0, 0.0));
    let kill = |min: f32, max: f32| {
        format!(
            r#"(module: "Kill Particles by Speed", inputs: {{ "Minimum Speed": Float({min:?}), "Maximum Speed": Float({max:?}) }})"#
        )
    };
    assert!(alive_after_one_update(&spawn, &kill(1.0, 4.0)));
    assert!(alive_after_one_update(&spawn, &kill(2.0, 2.0)));
    assert!(!alive_after_one_update(&spawn, &kill(3.0, 4.0)));
    assert!(!alive_after_one_update(&spawn, &kill(0.0, 1.0)));
    // In Particle Spawn the particle never appears.
    let mut sim = single_particle("", &format!("{spawn}, {}", kill(3.0, 4.0)), "");
    sim.step(DT);
    assert!(sim.particles(0).is_empty());
}

#[test]
fn kill_particles_by_distance() {
    let spawn = at(Vec3::new(1.0, 3.0, 0.0));
    let kill = |d: f32, closer: bool| {
        format!(
            r#"(module: "Kill Particles by Distance", inputs: {{
                "Reference Position": Vec3((1.0, 0.0, 0.0)), "Kill Distance": Float({d:?}), "Kill Closer": Bool({closer}) }})"#
        )
    };
    // Distance 3.
    assert!(!alive_after_one_update(&spawn, &kill(2.0, false)));
    assert!(alive_after_one_update(&spawn, &kill(4.0, false)));
    assert!(!alive_after_one_update(&spawn, &kill(4.0, true)));
    assert!(alive_after_one_update(&spawn, &kill(2.0, true)));
}

#[test]
fn rotate_around_point() {
    // Around +Y (given as (0, 2, 0)) the circle starts at +X and turns towards -Z
    // (the basis of fx_core::sim::basis(Y)). 360°/s after 0.25 s: 90°.
    let rotate = |phase: f32| {
        format!(
            r#"(module: "Rotate Around Point", inputs: {{
                "Rotation Center": Vec3((1.0, 2.0, 0.0)), "Rotation Axis": Vec3((0.0, 2.0, 0.0)),
                "Rotation Radius": Float(2.0), "Rotation Rate": Float(360.0), "Phase": Float({phase:?}) }})"#
        )
    };
    let p = spawned(&rotate(0.0));
    assert_v3(p.position, Vec3::new(3.0, 2.0, 0.0));
    let p = spawned(&rotate(90.0));
    assert_v3(p.position, Vec3::new(1.0, 2.0, -2.0));
    let p = after_one_update("", &format!("{STATE}, {}", rotate(0.0)));
    assert_v3(p.position, Vec3::new(1.0, 2.0, -2.0));
    let p = after_updates("", &format!("{STATE}, {}", rotate(0.0)), 2);
    assert_v3(p.position, Vec3::new(-1.0, 2.0, 0.0));
    // Forces do not move it: the module sets the position.
    let p = after_updates(
        &set_velocity(Vec3::new(5.0, 5.0, 5.0)),
        &format!("{STATE}, {SOLVE}, {}", rotate(0.0)),
        2,
    );
    assert_v3(p.position, Vec3::new(-1.0, 2.0, 0.0));
}

#[test]
fn noise_force_matches_the_field() {
    let pos = Vec3::new(0.3, 0.7, 0.2);
    let noise = |octaves: i32, pan: Vec3| {
        format!(
            r#"(module: "Noise Force", inputs: {{ "Noise Strength": Float(2.0), "Noise Frequency": Float(0.5),
                "Noise Octaves": Int({octaves}), "Pan Noise Field": {} }})"#,
            v3(pan)
        )
    };
    let force = |octaves: i32, pan: Vec3| after_one_update(&at(pos), &noise(octaves, pan)).force;
    let f1 = force(1, Vec3::ZERO);
    assert!(f1.length() > 0.0, "{f1}");
    assert_eq!(f1, noise_force(pos * 0.5, 1) * 2.0);
    // Deterministic.
    assert_eq!(force(1, Vec3::ZERO), f1);
    // More octaves change it.
    let f3 = force(3, Vec3::ZERO);
    assert_ne!(f3, f1);
    assert_eq!(f3, noise_force(pos * 0.5, 3) * 2.0);
    // The pan moves the field with System.Age (0.25 in the second step).
    let pan = Vec3::new(4.0, 0.0, 0.0);
    let fp = force(1, pan);
    assert_eq!(fp, noise_force(pos * 0.5 + pan * 0.25, 1) * 2.0);
    assert_ne!(fp, f1);
    // Different places, different forces.
    assert_ne!(noise_force(Vec3::new(1.3, 0.1, 0.9), 1), noise_force(pos, 1));
}

// ---- emitter spawn modules ----

#[test]
fn spawn_per_unit_follows_owner_speed() {
    let per_unit = |extra: &str| {
        format!(r#"(module: "Spawn Per Unit", inputs: {{ "Spawn Per Unit": Float(10.0) {extra} }})"#)
    };
    let counts = |extra: &str, velocity: Vec3, steps: usize| {
        let mut sim = Simulation::new(compile(&one_emitter("", &per_unit(extra), "", "")), 1);
        sim.owner_velocity = velocity;
        (0..steps)
            .map(|_| {
                sim.step(DT);
                sim.emitters[0].spawned_last_step
            })
            .collect::<Vec<_>>()
    };
    // 10 per metre at 4 m/s over 0.25 s: 10 per step.
    assert_eq!(counts("", Vec3::new(0.0, 0.0, 4.0), 3), [10, 10, 10]);
    // 2.5 per step: fractions carry over.
    assert_eq!(counts("", Vec3::new(1.0, 0.0, 0.0), 4), [2, 3, 2, 3]);
    // Not moving: nothing.
    assert_eq!(counts("", Vec3::ZERO, 2), [0, 0]);
    // Movement Tolerance: at or below it nothing.
    assert_eq!(counts(r#", "Movement Tolerance": Float(4.0)"#, Vec3::new(4.0, 0.0, 0.0), 2), [0, 0]);
    assert_eq!(counts(r#", "Movement Tolerance": Float(3.0)"#, Vec3::new(4.0, 0.0, 0.0), 2), [10, 10]);
    // Max Movement Threshold: above it (a teleport) nothing.
    assert_eq!(counts(r#", "Max Movement Threshold": Float(3.0)"#, Vec3::new(4.0, 0.0, 0.0), 2), [0, 0]);
    assert_eq!(counts(r#", "Max Movement Threshold": Float(4.0)"#, Vec3::new(4.0, 0.0, 0.0), 2), [10, 10]);
}

#[test]
fn spawn_per_frame_spawns_every_step() {
    assert_eq!(
        spawn_counts(r#"(module: "Spawn Per Frame", inputs: { "Spawn Count": Int(3) })"#, 4),
        [3, 3, 3, 3]
    );
    assert_eq!(spawn_counts(r#"(module: "Spawn Per Frame")"#, 2), [1, 1]);
}

#[test]
fn spawn_burst_probability() {
    let burst = |p: f32| {
        format!(
            r#"(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int(5), "Spawn Probability": Float({p:?}) }})"#
        )
    };
    assert_eq!(spawn_counts(&burst(0.0), 3), [0, 0, 0]);
    assert_eq!(spawn_counts(&burst(1.0), 3), [5, 0, 0]);
    // Probability 0.5 over many seeds: some fire, some do not.
    let src = one_emitter("", &burst(0.5), "", "");
    let cooked = compile(&src);
    let fired: Vec<bool> = (0..64)
        .map(|seed| {
            let mut sim = Simulation::new(cooked.clone(), seed);
            sim.step(DT);
            let n = sim.particles(0).len();
            assert!(n == 0 || n == 5);
            n == 5
        })
        .collect();
    assert!(fired.iter().any(|f| *f) && fired.iter().any(|f| !*f));
}

#[test]
fn spawn_burst_interval() {
    let interval = |extra: &str| {
        format!(
            r#"(module: "Spawn Burst Interval", inputs: {{ "Spawn Count": Int(2), "Spawn Interval": Float(0.5) {extra} }})"#
        )
    };
    // Bursts at 0, 0.5, 1.0, 1.5 over ages 0..1.75.
    assert_eq!(spawn_counts(&interval(""), 8), [2, 0, 2, 0, 2, 0, 2, 0]);
    // Burst Count 3.
    assert_eq!(spawn_counts(&interval(r#", "Burst Count": Int(3)"#), 8), [2, 0, 2, 0, 2, 0, 0, 0]);
    // First at Spawn Time 0.25.
    assert_eq!(spawn_counts(&interval(r#", "Spawn Time": Float(0.25)"#), 6), [0, 2, 0, 2, 0, 2]);
    // Probability 0.
    assert_eq!(spawn_counts(&interval(r#", "Spawn Probability": Float(0.0)"#), 4), [0, 0, 0, 0]);
    // Shorter than a step: several bursts in one.
    assert_eq!(
        spawn_counts(r#"(module: "Spawn Burst Interval", inputs: { "Spawn Count": Int(1), "Spawn Interval": Float(0.125) })"#, 2),
        [2, 2]
    );
}

// ---- Emitter State ----

fn emitter_state(inputs: &str) -> String {
    format!(r#"(module: "Emitter State", inputs: {{ {inputs} }})"#)
}

const BURST: &str = r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(1) })"#;

#[test]
fn burst_repeats_every_loop() {
    let counts = |state: &str| spawn_counts(&format!("{}, {BURST}", emitter_state(state)), 14);
    // Infinite loops of 1 s: a burst at 0, 1, 2, 3.
    assert_eq!(
        counts(r#""Loop Behavior": Infinite, "Loop Duration": Float(1.0)"#),
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    );
    // Multiple, 2 loops.
    assert_eq!(
        counts(r#""Loop Behavior": Multiple, "Loop Count": Int(2), "Loop Duration": Float(1.0)"#),
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    // Once.
    assert_eq!(
        counts(r#""Loop Behavior": Once, "Loop Duration": Float(1.0)"#),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    // Without an Emitter State: one endless loop, one burst.
    assert_eq!(spawn_counts(BURST, 8), [1, 0, 0, 0, 0, 0, 0, 0]);
    // Loop Delay 0.5 before every loop of 1 s: bursts at 0.5 and 2.0.
    assert_eq!(
        counts(r#""Loop Behavior": Infinite, "Loop Duration": Float(1.0), "Loop Delay": Float(0.5)"#),
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    );
    // Only the first loop waits: 0.5, 1.5, 2.5.
    assert_eq!(
        counts(r#""Loop Behavior": Infinite, "Loop Duration": Float(1.0), "Loop Delay": Float(0.5), "Delay First Loop Only": Bool(true)"#),
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    );
}

#[test]
fn spawn_rate_waits_out_delay_and_stops_after_last_loop() {
    // Two loops of 1 s, each after 0.5 s: active 0.5..1.5 and 2.0..3.0; 4/s = 1 per step.
    let rate = r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Float(4.0) })"#;
    let state = emitter_state(
        r#""Loop Behavior": Multiple, "Loop Count": Int(2), "Loop Duration": Float(1.0), "Loop Delay": Float(0.5)"#,
    );
    assert_eq!(
        spawn_counts(&format!("{state}, {rate}"), 16),
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    );
    // Delay First Loop Only: one stretch 0.5..2.5.
    let state = emitter_state(
        r#""Loop Behavior": Multiple, "Loop Count": Int(2), "Loop Duration": Float(1.0), "Loop Delay": Float(0.5), "Delay First Loop Only": Bool(true)"#,
    );
    assert_eq!(
        spawn_counts(&format!("{state}, {rate}"), 12),
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    );
    // Spawn Per Frame and Spawn Per Unit stop too.
    let once = emitter_state(r#""Loop Behavior": Once, "Loop Duration": Float(0.5)"#);
    assert_eq!(
        spawn_counts(&format!(r#"{once}, (module: "Spawn Per Frame")"#), 4),
        [1, 1, 0, 0]
    );
    let mut sim = Simulation::new(
        compile(&one_emitter("", &format!(r#"{once}, (module: "Spawn Per Unit")"#), "", "")),
        1,
    );
    sim.owner_velocity = Vec3::new(4.0, 0.0, 0.0);
    let counts: Vec<u32> = (0..4)
        .map(|_| {
            sim.step(DT);
            sim.emitters[0].spawned_last_step
        })
        .collect();
    assert_eq!(counts, [10, 10, 0, 0]);
}

#[test]
fn loop_age_and_count_are_readable() {
    // One particle per step records Emitter.LoopedAge (as its lifetime) and
    // Emitter.LoopCount (as its ribbon id) at the start of the step.
    let src = one_emitter(
        "",
        &format!(
            r#"{}, (module: "Spawn Per Frame")"#,
            emitter_state(r#""Loop Behavior": Infinite, "Loop Duration": Float(1.0)"#)
        ),
        r#"(module: "Set Particle Lifetime", inputs: { "Lifetime": Link(Float, "Emitter.LoopedAge") }),
           (module: "Initialize Ribbon", inputs: { "Ribbon ID": Link(Int, "Emitter.LoopCount") })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 1);
    for _ in 0..10 {
        sim.step(DT);
    }
    let mut ps: Vec<Particle> = sim.particles(0).to_vec();
    ps.sort_by_key(|p| p.unique_id);
    let looped: Vec<f32> = ps.iter().map(|p| p.lifetime).collect();
    let count: Vec<i32> = ps.iter().map(|p| p.ribbon_id).collect();
    assert_eq!(looped, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
    assert_eq!(count, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    // Without Emitter State, LoopedAge is Emitter.Age.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Per Frame")"#,
        r#"(module: "Set Particle Lifetime", inputs: { "Lifetime": Link(Float, "Emitter.LoopedAge") }),
           (module: "Set Particle Mass", inputs: { "Mass": Link(Float, "Emitter.Age") })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 1);
    for _ in 0..6 {
        sim.step(DT);
    }
    for p in sim.particles(0) {
        assert_eq!(p.lifetime, p.mass);
        assert_eq!(p.lifetime, p.unique_id as f32 * DT);
    }
}

#[test]
fn emitter_state_cook_errors() {
    let src = |state: &str, user: &str| {
        one_emitter(user, &format!("{}, {BURST}", emitter_state(state)), "", "")
    };
    // Not a constant.
    let errors = cook_errors(&src(r#""Loop Duration": Link(Float, "User.D")"#, r#""User.D": Float(1.0)"#));
    assert!(has_error(&errors, "must be a constant"), "{errors:?}");
    assert!(has_error(&errors, "Loop Duration"), "{errors:?}");
    let errors = cook_errors(&src(
        r#""Loop Delay": Dynamic("Uniform Ranged Float", { "Minimum": Float(0.0), "Maximum": Float(1.0) })"#,
        "",
    ));
    assert!(has_error(&errors, "must be a constant"), "{errors:?}");
    // Loop Count 0 with Multiple.
    let errors = cook_errors(&src(r#""Loop Behavior": Multiple, "Loop Count": Int(0)"#, ""));
    assert!(has_error(&errors, "Loop Count"), "{errors:?}");
    // Loop Duration 0 (Fixed).
    let errors = cook_errors(&src(r#""Loop Duration": Float(0.0)"#, ""));
    assert!(has_error(&errors, "Loop Duration"), "{errors:?}");
    // Negative delay.
    let errors = cook_errors(&src(r#""Loop Delay": Float(-1.0)"#, ""));
    assert!(has_error(&errors, "Loop Delay"), "{errors:?}");
    // Two Emitter States.
    let two = one_emitter("", &format!("{}, {}, {BURST}", emitter_state(""), emitter_state("")), "", "");
    let errors = cook_errors(&two);
    assert!(has_error(&errors, "one \"Emitter State\""), "{errors:?}");
    // Loop Count 0 is fine for Infinite (unused), and Loop Duration 0 with an Infinite mode.
    compile(&src(r#""Loop Behavior": Infinite, "Loop Count": Int(0)"#, ""));
    compile(&src(r#""Loop Duration Mode": Infinite, "Loop Duration": Float(0.0)"#, ""));
}

#[test]
fn loops_set_duration_and_spawn_end() {
    let cooked = |state: &str, spawn: &str| compile(&one_emitter("", &format!("{}, {spawn}", emitter_state(state)), "", ""));
    // Once, 1 s.
    let c = cooked(r#""Loop Behavior": Once, "Loop Duration": Float(1.0)"#, BURST);
    assert_eq!(c.emitters[0].spawn_end, Some(1.0));
    assert_eq!(c.duration, Some(1.0));
    assert_eq!(
        c.emitters[0].lifecycle,
        Lifecycle { loops: 1, duration: 1.0, delay: 0.0, delay_first_only: false, explicit: true }
    );
    // Three loops of 1 s after 0.5 s each: 3 * 1.5.
    let c = cooked(
        r#""Loop Behavior": Multiple, "Loop Count": Int(3), "Loop Duration": Float(1.0), "Loop Delay": Float(0.5)"#,
        r#"(module: "Spawn Rate")"#,
    );
    assert_eq!(c.emitters[0].spawn_end, Some(4.5));
    assert_eq!(c.duration, Some(4.5));
    // Delay first loop only: 0.5 + 3.
    let c = cooked(
        r#""Loop Behavior": Multiple, "Loop Count": Int(3), "Loop Duration": Float(1.0), "Loop Delay": Float(0.5), "Delay First Loop Only": Bool(true)"#,
        BURST,
    );
    assert_eq!(c.emitters[0].spawn_end, Some(3.5));
    // Infinite loops: no end.
    let c = cooked(r#""Loop Behavior": Infinite, "Loop Duration": Float(1.0)"#, BURST);
    assert_eq!(c.emitters[0].spawn_end, None);
    assert_eq!(c.duration, None);
    // One endless loop after a delay: the last burst time plus the delay.
    let c = cooked(
        r#""Loop Behavior": Once, "Loop Duration Mode": Infinite, "Loop Delay": Float(0.25)"#,
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Time": Float(0.5) })"#,
    );
    assert_eq!(c.emitters[0].spawn_end, Some(0.75));
    // Without Emitter State: the default life cycle.
    let c = compile(&one_emitter("", BURST, "", ""));
    assert_eq!(c.emitters[0].lifecycle, Lifecycle::default());
    assert_eq!(c.emitters[0].spawn_end, Some(0.0));
}

fn lc(loops: u32, duration: f32, delay: f32, delay_first_only: bool) -> Lifecycle {
    Lifecycle { loops, duration, delay, delay_first_only, explicit: true }
}

#[test]
fn lifecycle_phase() {
    // Two loops of 1 s, each after 0.5 s: active 0.5..1.5 and 2.0..3.0.
    let l = lc(2, 1.0, 0.5, false);
    let ph = |age: f32| {
        let p = l.phase(age);
        (p.index, p.looped_age, p.active)
    };
    assert_eq!(ph(0.0), (0, -0.5, false));
    assert_eq!(ph(0.25), (0, -0.25, false));
    assert_eq!(ph(0.5), (0, 0.0, true));
    assert_eq!(ph(1.25), (0, 0.75, true));
    assert_eq!(ph(1.5), (1, -0.5, false));
    assert_eq!(ph(1.75), (1, -0.25, false));
    assert_eq!(ph(2.0), (1, 0.0, true));
    assert_eq!(ph(2.75), (1, 0.75, true));
    assert_eq!(ph(3.0), (1, 1.0, false));
    assert_eq!(ph(10.0), (1, 8.0, false));
    assert_eq!(l.end(), Some(3.0));
    assert_eq!(l.loop_start(0), 0.5);
    assert_eq!(l.loop_start(1), 2.0);
    assert_eq!(l.phase(2.5).normalized(&l), 0.5);
    assert_eq!(l.phase(0.25).normalized(&l), 0.0);
    // Delay first loop only.
    let l = lc(0, 1.0, 0.5, true);
    assert_eq!(l.loop_start(3), 3.5);
    let p = l.phase(2.75);
    assert_eq!((p.index, p.looped_age, p.active), (2, 0.25, true));
    assert_eq!(l.end(), None);
    // Default: one endless loop, LoopedAge = Age.
    let d = Lifecycle::default();
    let p = d.phase(5.0);
    assert_eq!((p.index, p.looped_age, p.active), (0, 5.0, true));
    assert_eq!(p.normalized(&d), 0.0);
    assert_eq!(d.end(), None);
    // Endless loop after a delay.
    let l = lc(1, f32::INFINITY, 1.0, false);
    assert!(!l.phase(0.5).active);
    assert!(l.phase(100.0).active);
    assert_eq!(l.end(), None);
}

#[test]
fn lifecycle_fires() {
    let l = lc(2, 1.0, 0.5, false);
    // t = 0.5 into each loop: ages 1.0 and 2.5.
    assert_eq!(l.fires(0.0, 0.25, 0.5), 0);
    assert_eq!(l.fires(1.0, 0.25, 0.5), 1);
    assert_eq!(l.fires(0.75, 0.25, 0.5), 0);
    assert_eq!(l.fires(2.5, 0.25, 0.5), 1);
    assert_eq!(l.fires(4.0, 0.25, 0.5), 0);
    // A long step sees both loops.
    assert_eq!(l.fires(0.0, 10.0, 0.5), 2);
    // Outside the active part: never.
    assert_eq!(l.fires(0.0, 10.0, 1.0), 0);
    assert_eq!(l.fires(0.0, 10.0, -0.25), 0);
    // Endless loops of 0.25 s, one step of 1 s: four times.
    let l = lc(0, 0.25, 0.0, false);
    assert_eq!(l.fires(0.0, 1.0, 0.0), 4);
    // The default fires once, at t.
    let d = Lifecycle::default();
    assert_eq!(d.fires(0.0, 0.25, 0.0), 1);
    assert_eq!(d.fires(0.25, 0.25, 0.0), 0);
    assert_eq!(d.fires(0.25, 0.25, 0.25), 1);
}

#[test]
fn lifecycle_fires_every() {
    let l = lc(0, 1.0, 0.0, false);
    // 0, 0.25, 0.5, 0.75 in the first loop.
    assert_eq!(l.fires_every(0.0, 1.0, 0.0, 0.25, 0), 4);
    assert_eq!(l.fires_every(0.0, 1.0, 0.0, 0.25, 2), 2);
    // Two loops, two each.
    assert_eq!(l.fires_every(0.0, 2.0, 0.0, 0.25, 2), 4);
    // Step by step with an interval of 0.5.
    let steps: Vec<u32> = (0..8).map(|i| l.fires_every(i as f32 * 0.25, 0.25, 0.0, 0.5, 0)).collect();
    assert_eq!(steps, [1, 0, 1, 0, 1, 0, 1, 0]);
    // Starting at 0.25 into each loop.
    let steps: Vec<u32> = (0..8).map(|i| l.fires_every(i as f32 * 0.25, 0.25, 0.25, 0.5, 0)).collect();
    assert_eq!(steps, [0, 1, 0, 1, 0, 1, 0, 1]);
    // No interval: like fires.
    assert_eq!(l.fires_every(0.0, 0.25, 0.0, 0.0, 0), l.fires(0.0, 0.25, 0.0));
    // Finite loops stop.
    let l = lc(1, 1.0, 0.0, false);
    assert_eq!(l.fires_every(0.0, 10.0, 0.0, 0.25, 0), 4);
    assert_eq!(l.fires_every(1.0, 1.0, 0.0, 0.25, 0), 0);
}

// ---- system and emitter options ----

fn warmup_effect(system_fields: &str) -> String {
    effect(system_fields, "", r#"(module: "Spawn Per Frame")"#, "", STATE)
}

#[test]
fn warmup_runs_before_the_first_step() {
    let c = compile(&warmup_effect("warmup_time: 1.0, warmup_tick_delta: 0.25,"));
    assert_eq!(c.warmup_time, 1.0);
    assert_eq!(c.warmup_tick_delta, 0.25);
    assert_eq!(fx_core::lifecycle::warmup_steps(&c), 4);
    let mut warm = Simulation::new(c, 1);
    warm.step(DT);
    // Four warm-up steps and the visible one.
    assert_eq!(warm.frame, 5);
    assert_eq!(warm.time, 1.25);
    assert_eq!(warm.emitters[0].age, 1.25);
    let mut ages: Vec<f32> = warm.particles(0).iter().map(|p| p.age).collect();
    ages.sort_by(f32::total_cmp);
    assert_eq!(ages, [0.0, 0.25, 0.5, 0.75, 1.0]);
    // The same as stepping five times by hand.
    let plain = compile(&warmup_effect(""));
    assert_eq!(plain.warmup_time, 0.0);
    let mut hand = Simulation::new(plain, 1);
    for _ in 0..5 {
        hand.step(DT);
    }
    let bits = |s: &Simulation| s.particles(0).iter().map(Particle::bits).collect::<Vec<_>>();
    assert_eq!(bits(&warm), bits(&hand));
    // Later steps add no more warm-up.
    warm.step(DT);
    assert_eq!(warm.frame, 6);
}

#[test]
fn no_warmup_without_warmup_time() {
    let mut sim = Simulation::new(compile(&warmup_effect("")), 1);
    sim.step(DT);
    assert_eq!(sim.frame, 1);
    assert_eq!(sim.particles(0).len(), 1);
    // A tick delta alone does nothing.
    let mut sim = Simulation::new(compile(&warmup_effect("warmup_tick_delta: 0.25,")), 1);
    sim.step(DT);
    assert_eq!(sim.frame, 1);
    // A warm-up that does not divide evenly rounds up: 0.3 / 0.25 = 2 steps.
    let c = compile(&warmup_effect("warmup_time: 0.3, warmup_tick_delta: 0.25,"));
    assert_eq!(fx_core::lifecycle::warmup_steps(&c), 2);
}

fn random_positions(emitter_fields: &str, seed: u32) -> Vec<Vec3> {
    let src = effect(
        "",
        emitter_fields,
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(8) })"#,
        r#"(module: "Initialize Particle", inputs: { "Position": Dynamic("Random Range Vec3 by Uniform", {
            "Minimum": Vec3((-1.0, -1.0, -1.0)), "Maximum": Vec3((1.0, 1.0, 1.0)) }) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), seed);
    sim.step(DT);
    sim.particles(0).iter().map(|p| p.position).collect()
}

#[test]
fn determinism_option() {
    // Default and `deterministic: true`: the same seed gives the same particles.
    assert_eq!(random_positions("", 7), random_positions("", 7));
    assert_eq!(random_positions("deterministic: true,", 7), random_positions("", 7));
    assert_ne!(random_positions("", 7), random_positions("", 8));
    // `deterministic: false`: copies differ.
    let a = random_positions("deterministic: false,", 7);
    let b = random_positions("deterministic: false,", 7);
    assert_eq!(a.len(), 8);
    assert_ne!(a, b);
    let c = compile(&effect("", "deterministic: false,", "", "", ""));
    assert!(!c.emitters[0].deterministic);
    let c = compile(&effect("", "", "", "", ""));
    assert!(c.emitters[0].deterministic);
}

#[test]
fn emitter_seed_mixes_the_nonce_only_when_not_deterministic() {
    use fx_core::lifecycle::emitter_seed;
    assert_eq!(emitter_seed(5, 1, true), 5);
    assert_eq!(emitter_seed(5, 2, true), 5);
    assert_ne!(emitter_seed(5, 1, false), emitter_seed(5, 2, false));
    assert_eq!(emitter_seed(5, 1, false), emitter_seed(5, 1, false));
}

#[test]
fn fixed_bounds_option() {
    let c = compile(&effect("", "fixed_bounds: ((-1.0, 0.0, -1.0), (1.0, 2.0, 1.0)),", "", "", ""));
    assert_eq!(c.emitters[0].fixed_bounds, Some([[-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]]));
    let c = compile(&effect("", "", "", "", ""));
    assert_eq!(c.emitters[0].fixed_bounds, None);
    // A flat box is fine.
    let c = compile(&effect("", "fixed_bounds: ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),", "", "", ""));
    assert_eq!(c.emitters[0].fixed_bounds, Some([[0.0; 3], [1.0, 0.0, 1.0]]));
    // Min above max.
    let errors = cook_errors(&effect("", "fixed_bounds: ((1.0, 0.0, -1.0), (-1.0, 2.0, 1.0)),", "", "", ""));
    assert!(has_error(&errors, "fixed_bounds"), "{errors:?}");
    // Not a pair.
    let errors = cook_errors(&effect("", "fixed_bounds: ((1.0, 0.0, -1.0),),", "", "", ""));
    assert!(has_error(&errors, "fixed_bounds"), "{errors:?}");
}
