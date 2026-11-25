//! One test (or more) per module, against values worked out by hand.
//! Step sizes are powers of two so most expected values are exact.

mod common;

use bevy_math::{EulerRot, Quat, Vec3, Vec4};
use common::*;
use fx_core::sim::noise::{curl_noise, gradient_noise, lattice_gradient};
use fx_core::sim::{Simulation, euler_quat, quat_mul, quat_normalize};
use fx_core::types::Value;

const STATE: &str = r#"(module: "Particle State")"#;
const SOLVE: &str = r#"(module: "Solve Forces and Velocity")"#;

fn spawned_per_step(sim: &mut Simulation, steps: usize, dt: f32) -> Vec<u32> {
    (0..steps)
        .map(|_| {
            sim.step(dt);
            sim.emitters[0].spawned_last_step
        })
        .collect()
}

#[test]
fn spawn_rate_accumulates_fractions() {
    // 10/s at dt 0.25 = 2.5 per step: 2, 3, 2, 3 (the half carries over).
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Float(10.0) })"#,
        "",
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    assert_eq!(spawned_per_step(&mut sim, 4, 0.25), [2, 3, 2, 3]);
    assert_eq!(sim.particle_count(), 10);
    // UniqueIDs count up from 0.
    let ids: Vec<u32> = sim.particles(0).iter().map(|p| p.unique_id).collect();
    assert_eq!(ids, (0..10).collect::<Vec<_>>());
}

#[test]
fn spawn_burst_fires_once_at_its_time() {
    // Steps cover emitter ages [0, .25), [.25, .5), [.5, .75), [.75, 1): the burst at 0.5 fires in the third.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(5), "Spawn Time": Float(0.5) })"#,
        "",
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    assert_eq!(spawned_per_step(&mut sim, 4, 0.25), [0, 0, 5, 0]);
}

#[test]
fn spawn_is_capped_by_max_particles() {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(5000) })"#,
        "",
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(0.25);
    assert_eq!(sim.particle_count(), 1000);
}

#[test]
fn initialize_particle_sets_attributes() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: {
            "Lifetime": Float(3.0), "Position": Vec3((1.0, 2.0, 3.0)), "Color": Color((1.0, 0.5, 0.25, 1.0)),
            "Sprite Size": Vec2((0.2, 0.4)), "Sprite Rotation": Float(90.0), "Mass": Float(2.0) })"#,
        "",
    );
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(p.lifetime, 3.0);
    assert_eq!(p.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(p.color.to_array(), [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(p.initial_color, p.color);
    assert_eq!(p.sprite_size.to_array(), [0.2, 0.4]);
    assert_eq!(p.initial_sprite_size, p.sprite_size);
    assert_eq!(p.sprite_rotation, 90.0);
    assert_eq!(p.mass, 2.0);
    assert_eq!(p.age, 0.0);
}

#[test]
fn random_range_stays_in_range_and_varies() {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })"#,
        r#"(module: "Initialize Particle", inputs: {
            "Lifetime": RandomRange(Float(1.0), Float(2.0)),
            "Position": RandomRange(Vec3((-1.0, 0.0, 5.0)), Vec3((1.0, 0.0, 6.0))) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 7);
    sim.step(0.25);
    let ps = sim.particles(0);
    assert!(ps.iter().all(|p| (1.0..2.0).contains(&p.lifetime)));
    assert!(ps.iter().all(|p| (-1.0..1.0).contains(&p.position.x)
        && p.position.y == 0.0
        && (5.0..6.0).contains(&p.position.z)));
    let mean = ps.iter().map(|p| p.lifetime).sum::<f32>() / ps.len() as f32;
    assert!((mean - 1.5).abs() < 0.08, "mean lifetime {mean}");
    // Components draw independently.
    assert!(
        ps.iter()
            .any(|p| (p.position.x + 1.0) / 2.0 - (p.position.z - 5.0) > 0.1)
    );
}

#[test]
fn add_velocity_adds() {
    let add = r#"(module: "Add Velocity", inputs: { "Velocity": Vec3((1.0, 2.0, 3.0)) })"#;
    let mut sim = single_particle("", &format!("{add}, {add}"), "");
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(2.0, 4.0, 6.0));
}

#[test]
fn add_velocity_in_cone_zero_angle_follows_axis() {
    // Axis (0, 0, 2) normalises to +Z; a 0° cone gives exactly the axis.
    let mut sim = single_particle(
        "",
        r#"(module: "Add Velocity in Cone", inputs: { "Cone Axis": Vec3((0.0, 0.0, 2.0)), "Cone Angle": Float(0.0), "Velocity Strength": Float(5.0) })"#,
        "",
    );
    sim.step(0.25);
    assert_vec_close(&sim.particles(0)[0].velocity.to_array(), &[0.0, 0.0, 5.0]);
}

#[test]
fn add_velocity_in_cone_stays_in_cone() {
    // 60° half-angle around +X: every direction has x >= cos 60° = 0.5 and |v| = strength.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(500) })"#,
        r#"(module: "Add Velocity in Cone", inputs: { "Cone Axis": Vec3((1.0, 0.0, 0.0)), "Cone Angle": Float(60.0), "Velocity Strength": Float(3.0) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 3);
    sim.step(0.25);
    let ps = sim.particles(0);
    for p in ps {
        let v = p.velocity;
        assert_close(v.length(), 3.0);
        assert!(v.x / 3.0 >= 0.5 - 1e-5, "{v}");
    }
    // Uniform over the cap: mean cos θ = (1 + cos 60°) / 2 = 0.75; the sideways mean is ~0.
    let n = ps.len() as f32;
    let mean = ps.iter().map(|p| p.velocity / 3.0).sum::<Vec3>() / n;
    assert!(
        (mean.x - 0.75).abs() < 0.02 && mean.y.abs() < 0.05 && mean.z.abs() < 0.05,
        "{mean}"
    );
}

#[test]
fn gravity_then_solve_integrates_semi_implicit_euler() {
    // Mass 2, gravity -10, dt 0.5: Force = -20, v = -20/2*0.5 = -5, x = -5*0.5 = -2.5.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Mass": Float(2.0) })"#,
        &format!(
            r#"(module: "Gravity Force", inputs: {{ "Gravity": Vec3((0.0, -10.0, 0.0)) }}), {SOLVE}"#
        ),
    );
    sim.step(0.5); // spawn
    sim.step(0.5); // first update
    let p = sim.particles(0)[0];
    assert_eq!(p.velocity, Vec3::new(0.0, -5.0, 0.0));
    assert_eq!(p.position, Vec3::new(0.0, -2.5, 0.0));
    assert_eq!(p.force, Vec3::ZERO, "Solve clears the force");
    sim.step(0.5); // v = -10, x = -2.5 - 5 = -7.5
    let p = sim.particles(0)[0];
    assert_eq!(p.velocity.y, -10.0);
    assert_eq!(p.position.y, -7.5);
}

#[test]
fn drag_decays_velocity_exponentially() {
    // v0 = 10, Drag 2, dt 0.5: v = 10 e^-1, x = v * 0.5.
    let mut sim = single_particle(
        "",
        r#"(module: "Add Velocity", inputs: { "Velocity": Vec3((10.0, 0.0, 0.0)) })"#,
        &format!(r#"(module: "Drag", inputs: {{ "Drag": Float(2.0) }}), {SOLVE}"#),
    );
    sim.step(0.5);
    sim.step(0.5);
    let p = sim.particles(0)[0];
    let v = 10.0 * (-1.0f32).exp();
    assert_close(p.velocity.x, v);
    assert_close(p.position.x, v * 0.5);
    assert_eq!(p.drag, 0.0, "Solve clears drag");
}

#[test]
fn noise_is_zero_at_lattice_points_with_lattice_gradient() {
    for (x, y, z) in [(0, 0, 0), (1, 2, 3), (-4, 5, -6)] {
        let (v, g) = gradient_noise(Vec3::new(x as f32, y as f32, z as f32), 1);
        assert_eq!(v, 0.0);
        assert_eq!(g, lattice_gradient(x, y, z, 1));
    }
}

#[test]
fn noise_gradient_matches_finite_differences() {
    let p = Vec3::new(0.37, 1.61, -2.23);
    let (_, g) = gradient_noise(p, 2);
    let h = 1e-3;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let fd =
            (gradient_noise(p + axis * h, 2).0 - gradient_noise(p - axis * h, 2).0) / (2.0 * h);
        assert!((fd - g.dot(axis)).abs() < 2e-3, "{fd} vs {}", g.dot(axis));
    }
}

#[test]
fn curl_noise_is_divergence_free() {
    let h = 1e-3;
    for p in [Vec3::new(0.3, 0.7, 0.1), Vec3::new(2.4, -1.3, 5.9)] {
        let div = (curl_noise(p + Vec3::X * h).x - curl_noise(p - Vec3::X * h).x
            + curl_noise(p + Vec3::Y * h).y
            - curl_noise(p - Vec3::Y * h).y
            + curl_noise(p + Vec3::Z * h).z
            - curl_noise(p - Vec3::Z * h).z)
            / (2.0 * h);
        assert!(div.abs() < 1e-2, "divergence {div} at {p}");
    }
}

#[test]
fn curl_noise_force_at_lattice_point() {
    // At the origin the potential's gradients are the lattice gradients g0, g1, g2, so
    // curl = (g2.y - g1.z, g0.z - g2.x, g1.x - g0.y). Strength 2, mass 1, dt 0.5: v = curl * 2 * 0.5.
    let (g0, g1, g2) = (
        lattice_gradient(0, 0, 0, 0),
        lattice_gradient(0, 0, 0, 1),
        lattice_gradient(0, 0, 0, 2),
    );
    let curl = Vec3::new(g2.y - g1.z, g0.z - g2.x, g1.x - g0.y);
    assert_eq!(curl_noise(Vec3::ZERO), curl);
    let mut sim = single_particle(
        "",
        "",
        &format!(
            r#"(module: "Curl Noise Force", inputs: {{ "Noise Strength": Float(2.0), "Noise Frequency": Float(3.0) }}), {SOLVE}"#
        ),
    );
    sim.step(0.5);
    sim.step(0.5);
    assert_eq!(sim.particles(0)[0].velocity, curl);
}

#[test]
fn scale_color_multiplies_initial_color() {
    let init =
        r#"(module: "Initialize Particle", inputs: { "Color": Color((1.0, 0.5, 0.25, 1.0)) })"#;
    let scale = r#"(module: "Scale Color", inputs: { "Scale RGB": Vec3((2.0, 2.0, 2.0)), "Scale Alpha": Float(0.5) })"#;
    let mut sim = single_particle("", init, scale);
    sim.step(0.25);
    sim.step(0.25);
    sim.step(0.25);
    // Applied to the initial colour every step, so it does not compound.
    assert_eq!(sim.particles(0)[0].color.to_array(), [2.0, 1.0, 0.5, 0.5]);
}

#[test]
fn scale_color_alpha_curve_over_age() {
    // Lifetime 1, dt 0.25: after one update NormalizedAge = 0.25, so alpha = 1 - 0.25.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Lifetime": Float(1.0) })"#,
        &format!(
            r#"{STATE}, (module: "Scale Color", inputs: {{ "Scale Alpha": CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)])) }})"#
        ),
    );
    sim.step(0.25);
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].color.w, 0.75);
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].color.w, 0.5);
}

#[test]
fn scale_sprite_size_multiplies_initial_size() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Sprite Size": Vec2((0.25, 0.5)) })"#,
        r#"(module: "Scale Sprite Size", inputs: { "Scale Factor": Vec2((2.0, 0.5)) })"#,
    );
    sim.step(0.25);
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].sprite_size.to_array(), [0.5, 0.25]);
}

#[test]
fn particle_state_ages_and_kills_after_lifetime() {
    // Lifetime 1 at dt 0.25: ages 0.25, 0.5, 0.75, 1.0 survive (not > 1); 1.25 dies.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Lifetime": Float(1.0) })"#,
        STATE,
    );
    sim.step(0.25); // spawn
    for expected_age in [0.25, 0.5, 0.75, 1.0] {
        sim.step(0.25);
        let p = sim.particles(0)[0];
        assert_eq!(p.age, expected_age);
        assert_eq!(p.normalized_age, expected_age);
    }
    sim.step(0.25);
    assert_eq!(sim.particle_count(), 0);
}

#[test]
fn particle_state_can_keep_particles_alive() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Lifetime": Float(0.25) })"#,
        r#"(module: "Particle State", inputs: { "Kill Particles When Lifetime Has Elapsed": Bool(false) })"#,
    );
    for _ in 0..10 {
        sim.step(0.25);
    }
    assert_eq!(sim.particles(0)[0].normalized_age, 9.0);
}

#[test]
fn kill_particles_kills() {
    let kill = r#"(module: "Kill Particles", inputs: { "Kill Particles": Bool(true) })"#;
    // In Particle Update: alive after spawning, gone after its first update.
    let mut sim = single_particle("", "", kill);
    sim.step(0.25);
    assert_eq!(sim.particle_count(), 1);
    sim.step(0.25);
    assert_eq!(sim.particle_count(), 0);
    // In Particle Spawn: never added.
    let mut sim = single_particle("", kill, "");
    sim.step(0.25);
    assert_eq!(sim.particle_count(), 0);
}

#[test]
fn links_read_engine_and_user_parameters() {
    let mut sim = single_particle(
        r#""User.Push": Vec3((0.0, 4.0, 0.0))"#,
        r#"(module: "Initialize Particle", inputs: { "Lifetime": Link(Float, "Engine.DeltaTime") }),
           (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "User.Push") })"#,
        "",
    );
    sim.step(0.125);
    let p = sim.particles(0)[0];
    assert_eq!(p.lifetime, 0.125);
    assert_eq!(p.velocity, Vec3::new(0.0, 4.0, 0.0));
    // User parameters can be changed at runtime.
    assert!(sim.set_user("User.Push", Value::Vec3([1.0, 0.0, 0.0])));
    assert!(
        !sim.set_user("User.Push", Value::Float(1.0)),
        "type must match"
    );
    assert!(!sim.set_user("User.Missing", Value::Float(1.0)));
}

#[test]
fn particle_links_read_attributes() {
    // Add Velocity linked to Particles.Position doubles as "move away from origin".
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Position": Vec3((1.0, 0.0, 0.0)) }),
           (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Position") })"#,
        "",
    );
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(1.0, 0.0, 0.0));
}

#[test]
fn random_int_range_is_inclusive() {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": RandomRange(Int(2), Int(4)) })"#,
        "",
        "",
    );
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let mut sim = Simulation::new(compile(&src), seed);
        sim.step(0.25);
        seen.insert(sim.particle_count());
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), [2, 3, 4]);
}

#[test]
fn disabled_modules_are_skipped() {
    let mut sim = single_particle(
        "",
        r#"(module: "Add Velocity", enabled: false, inputs: { "Velocity": Vec3((1.0, 0.0, 0.0)) })"#,
        "",
    );
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::ZERO);
}

fn assert_quat_close(a: Quat, b: Quat) {
    assert_vec_close(&a.to_array(), &b.to_array());
}

#[test]
fn initialize_ribbon_sets_ribbon_attributes() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Ribbon", inputs: {
            "Ribbon Width": Float(0.25), "Ribbon Twist": Float(45.0), "Ribbon ID": Int(7), "Ribbon Link Order": Float(3.5) })"#,
        "",
    );
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(p.ribbon_width, 0.25);
    assert_eq!(p.ribbon_twist, 45.0);
    assert_eq!(p.ribbon_id, 7);
    assert_eq!(p.ribbon_link_order, 3.5);
    // Defaults without the module.
    let mut sim = single_particle("", "", "");
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(
        (
            p.ribbon_width,
            p.ribbon_twist,
            p.ribbon_id,
            p.ribbon_link_order
        ),
        (0.1, 0.0, 0, 0.0)
    );
    assert_eq!(p.light_radius, 1.0);
    assert_eq!(p.scale, Vec3::ONE);
    assert_eq!(p.mesh_orientation, Quat::IDENTITY);
    assert_eq!(p.dynamic_material_parameter, Vec4::ZERO);
}

#[test]
fn random_ribbon_ids_cover_the_range() {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })"#,
        r#"(module: "Initialize Ribbon", inputs: { "Ribbon ID": RandomRange(Int(0), Int(3)) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 5);
    sim.step(0.25);
    let ids: std::collections::BTreeSet<i32> =
        sim.particles(0).iter().map(|p| p.ribbon_id).collect();
    assert_eq!(ids.into_iter().collect::<Vec<_>>(), [0, 1, 2, 3]);
}

#[test]
fn euler_quat_applies_x_then_y_then_z() {
    for deg in [
        Vec3::new(90.0, 0.0, 0.0),
        Vec3::new(0.0, 90.0, 0.0),
        Vec3::new(0.0, 0.0, 90.0),
        Vec3::new(30.0, -45.0, 120.0),
    ] {
        let r = deg * (std::f32::consts::PI / 180.0);
        let want = Quat::from_euler(EulerRot::ZYX, r.z, r.y, r.x);
        assert_quat_close(euler_quat(deg), want);
    }
    // X first: (90, 0, 90) takes +Y to +Z (X turn), which the Z turn keeps; +X goes to +Y.
    let q = euler_quat(Vec3::new(90.0, 0.0, 90.0));
    assert_vec_close(&(q * Vec3::Y).to_array(), &[0.0, 0.0, 1.0]);
    assert_vec_close(&(q * Vec3::X).to_array(), &[0.0, 1.0, 0.0]);
    // quat_mul is the Hamilton product.
    let (a, b) = (
        euler_quat(Vec3::new(10.0, 20.0, 30.0)),
        euler_quat(Vec3::new(-50.0, 5.0, 70.0)),
    );
    assert_quat_close(quat_mul(a, b), a * b);
    assert_eq!(
        quat_normalize(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        Quat::IDENTITY
    );
    assert_quat_close(
        quat_normalize(Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        Quat::IDENTITY,
    );
}

#[test]
fn initialize_mesh_sets_scale_and_orientation() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Mesh", inputs: { "Mesh Scale": Vec3((1.0, 2.0, 3.0)), "Mesh Rotation": Vec3((0.0, 90.0, 0.0)) })"#,
        "",
    );
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(p.scale, Vec3::new(1.0, 2.0, 3.0));
    let h = std::f32::consts::FRAC_PI_4;
    assert_quat_close(
        p.mesh_orientation,
        Quat::from_xyzw(0.0, h.sin(), 0.0, h.cos()),
    );
}

#[test]
fn update_mesh_orientation_integrates_the_rate() {
    // 90°/s around Y for 4 steps of 0.25 s = 90° around Y; then the orientation stays unit length.
    let mut sim = single_particle(
        "",
        "",
        r#"(module: "Update Mesh Orientation", inputs: { "Rotation Rate": Vec3((0.0, 90.0, 0.0)) })"#,
    );
    sim.step(0.25); // spawn
    for _ in 0..4 {
        sim.step(0.25);
    }
    let q = sim.particles(0)[0].mesh_orientation;
    assert_quat_close(q, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
    assert_close(q.length(), 1.0);
    // Composed on the right: starting from 90° about X, a Y rate turns about the mesh's own Y.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Mesh", inputs: { "Mesh Rotation": Vec3((90.0, 0.0, 0.0)) })"#,
        r#"(module: "Update Mesh Orientation", inputs: { "Rotation Rate": Vec3((0.0, 180.0, 0.0)) })"#,
    );
    sim.step(0.5);
    sim.step(0.5);
    let want = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
        * Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    assert_quat_close(sim.particles(0)[0].mesh_orientation, want);
}

#[test]
fn initialize_light_and_dynamic_material_parameters() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Light", inputs: { "Light Radius": Float(2.5) }),
           (module: "Initialize Particle", inputs: { "Lifetime": Float(1.0) })"#,
        &format!(
            r#"{STATE}, (module: "Dynamic Material Parameters", inputs: {{ "Dynamic Material Parameter": CurveOverAge(Vec4([(0.0, (0.0, 0.0, 0.0, 0.0)), (1.0, (4.0, 8.0, 12.0, 16.0))])) }}),
               (module: "Initialize Light", inputs: {{ "Light Radius": Link(Float, "Particles.NormalizedAge") }})"#
        ),
    );
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(p.light_radius, 2.5);
    assert_eq!(p.dynamic_material_parameter, Vec4::ZERO);
    sim.step(0.25);
    let p = sim.particles(0)[0];
    assert_eq!(p.dynamic_material_parameter, Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(p.light_radius, 0.25);
}

#[test]
fn extended_attributes_are_linkable() {
    // Add Velocity linked to Particles.Scale.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Mesh", inputs: { "Mesh Scale": Vec3((1.0, 2.0, 3.0)) }),
           (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Scale") })"#,
        "",
    );
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(1.0, 2.0, 3.0));
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate")"#,
        r#"(module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Scale") })"#,
        "",
    );
    assert!(compile(&src).emitters[0].uses_extended_attributes());
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate")"#,
        r#"(module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Position") })"#,
        "",
    );
    assert!(!compile(&src).emitters[0].uses_extended_attributes());
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate")"#,
        "",
        r#"(module: "Initialize Light")"#,
    );
    assert!(compile(&src).emitters[0].uses_extended_attributes());
}

#[test]
fn curl_noise_pan_moves_the_field_with_time() {
    // Pan (1, 0, 0) per second: the force at step n samples the field at x = Engine.Time.
    // At t = 1.0 (start of the 5th step, dt 0.25) the sample point is the lattice point (1, 0, 0).
    let mut sim = single_particle(
        "",
        "",
        r#"(module: "Curl Noise Force", inputs: { "Pan Noise Field": Vec3((1.0, 0.0, 0.0)) })"#,
    );
    for _ in 0..4 {
        sim.step(0.25);
    }
    assert_eq!(sim.time, 1.0);
    sim.step(0.25);
    assert_eq!(sim.particles(0)[0].force.to_array(), {
        let mut want = Vec3::ZERO;
        for t in [0.25f32, 0.5, 0.75, 1.0] {
            want += curl_noise(Vec3::new(t, 0.0, 0.0));
        }
        want.to_array()
    });
    let at_one = curl_noise(Vec3::X);
    let (g0, g1, g2) = (
        lattice_gradient(1, 0, 0, 0),
        lattice_gradient(1, 0, 0, 1),
        lattice_gradient(1, 0, 0, 2),
    );
    assert_eq!(at_one, Vec3::new(g2.y - g1.z, g0.z - g2.x, g1.x - g0.y));
}
