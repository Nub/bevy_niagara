//! Niagara's common modules (forces, shapes, rotation, sub-UV, scaling),
//! one test or more each against values worked out by hand. dt is 0.25 and
//! the inputs powers of two, so most expected values are exact.

mod common;

use bevy_math::{Quat, Vec2, Vec3, Vec4};
use common::*;
use fx_core::sim::{Particle, ShapeInputs, Simulation, shape_point};

const DT: f32 = 0.25;
const STATE: &str = r#"(module: "Particle State")"#;

/// Position (and optionally more Initialize Particle inputs) as spawn source.
fn at(pos: Vec3) -> String {
    format!(
        r#"(module: "Initialize Particle", inputs: {{ "Position": Vec3(({}, {}, {})) }})"#,
        pos.x, pos.y, pos.z
    )
}

/// Spawns one particle (step 1) and runs `update` on it once (step 2).
fn after_one_update(spawn: &str, update: &str) -> Particle {
    let mut sim = single_particle("", spawn, update);
    sim.step(DT);
    sim.step(DT);
    sim.particles(0)[0]
}

fn alive_after_one_update(spawn: &str, update: &str) -> bool {
    let mut sim = single_particle("", spawn, update);
    sim.step(DT);
    sim.step(DT);
    !sim.particles(0).is_empty()
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-5
}

// ---- forces ----

#[test]
fn point_attraction_force_pulls_with_falloff() {
    // Distance 2 of radius 4, exponent 2: (1 - 0.5)^2 = 0.25 of strength 4 = 1, towards the origin.
    let attract = r#"(module: "Point Attraction Force", inputs: {
        "Attraction Strength": Float(4.0), "Attraction Radius": Float(4.0), "Falloff Exponent": Float(2.0) })"#;
    let p = after_one_update(&at(Vec3::new(2.0, 0.0, 0.0)), attract);
    assert_eq!(p.force, Vec3::new(-1.0, 0.0, 0.0));
    // Beyond the radius: nothing.
    let p = after_one_update(&at(Vec3::new(0.0, 5.0, 0.0)), attract);
    assert_eq!(p.force, Vec3::ZERO);
}

#[test]
fn point_attraction_force_kills_inside_kill_radius() {
    let attract = |kill: f32| {
        format!(
            r#"(module: "Point Attraction Force", inputs: {{ "Attractor Position": Vec3((1.0, 0.0, 0.0)), "Kill Radius": Float({kill}) }})"#
        )
    };
    let spawn = at(Vec3::new(3.0, 0.0, 0.0));
    assert!(!alive_after_one_update(&spawn, &attract(2.5)));
    assert!(alive_after_one_update(&spawn, &attract(2.0)));
}

#[test]
fn vortex_velocity_adds_tangential_velocity() {
    // At +X around +Y: Y × X = -Z.
    let vortex = r#"(module: "Vortex Velocity", inputs: { "Vortex Axis": Vec3((0.0, 2.0, 0.0)), "Velocity Amount": Float(3.0) })"#;
    let p = after_one_update(&at(Vec3::new(1.0, 7.0, 0.0)), vortex);
    assert_eq!(p.velocity, Vec3::new(0.0, 0.0, -3.0));
    // On the axis: no direction, no velocity.
    let p = after_one_update(&at(Vec3::new(0.0, 5.0, 0.0)), vortex);
    assert_eq!(p.velocity, Vec3::ZERO);
    // In Particle Spawn too, around a moved origin.
    let mut sim = single_particle(
        "",
        &format!(
            r#"{}, (module: "Vortex Velocity", inputs: {{ "Vortex Origin": Vec3((1.0, 0.0, 1.0)) }})"#,
            at(Vec3::new(1.0, 0.0, 3.0))
        ),
        "",
    );
    sim.step(DT);
    // Radial +Z: Y × Z = +X.
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(1.0, 0.0, 0.0));
}

#[test]
fn vortex_force_swirls_and_pulls_to_the_axis() {
    // At (0, 1, 2): radial +Z, tangent Y × Z = +X, inward -Z.
    let vortex = r#"(module: "Vortex Force", inputs: { "Vortex Force Amount": Float(2.0), "Origin Pull Amount": Float(3.0) })"#;
    let p = after_one_update(&at(Vec3::new(0.0, 1.0, 2.0)), vortex);
    assert_eq!(p.force, Vec3::new(2.0, 0.0, -3.0));
}

#[test]
fn wind_force_pulls_towards_wind_velocity() {
    // ((0, 0, 2) * 2 - (1, 0, 0)) * 0.5.
    let p = after_one_update(
        r#"(module: "Add Velocity", inputs: { "Velocity": Vec3((1.0, 0.0, 0.0)) })"#,
        r#"(module: "Wind Force", inputs: { "Wind Velocity": Vec3((0.0, 0.0, 2.0)), "Wind Speed": Float(2.0), "Air Resistance": Float(0.5) })"#,
    );
    assert_eq!(p.force, Vec3::new(-0.5, 0.0, 2.0));
}

#[test]
fn linear_force_adds() {
    let f = r#"(module: "Linear Force", inputs: { "Linear Force": Vec3((1.0, 2.0, 3.0)) })"#;
    let p = after_one_update("", &format!("{f}, {f}"));
    assert_eq!(p.force, Vec3::new(2.0, 4.0, 6.0));
}

#[test]
fn limit_force_clamps_length() {
    let f = r#"(module: "Linear Force", inputs: { "Linear Force": Vec3((3.0, 4.0, 0.0)) })"#;
    let limit = |l: f32| format!(r#"(module: "Limit Force", inputs: {{ "Force Limit": Float({l:.1}) }})"#);
    // Length 5 limited to 2.5: halved.
    let p = after_one_update("", &format!("{f}, {}", limit(2.5)));
    assert_eq!(p.force, Vec3::new(1.5, 2.0, 0.0));
    let p = after_one_update("", &format!("{f}, {}", limit(10.0)));
    assert_eq!(p.force, Vec3::new(3.0, 4.0, 0.0));
}

#[test]
fn acceleration_force_scales_with_mass() {
    let p = after_one_update(
        r#"(module: "Initialize Particle", inputs: { "Mass": Float(2.0) })"#,
        r#"(module: "Acceleration Force", inputs: { "Acceleration": Vec3((0.0, 1.0, 0.5)) })"#,
    );
    assert_eq!(p.force, Vec3::new(0.0, 2.0, 1.0));
}

#[test]
fn point_force_pushes_away_with_falloff() {
    // Distance 3 of radius 4, exponent 1: 0.25 of 8 = 2, away from the centre.
    let push = r#"(module: "Point Force", inputs: {
        "Point Position": Vec3((1.0, 0.0, 0.0)), "Point Force Strength": Float(8.0), "Falloff Radius": Float(4.0) })"#;
    let p = after_one_update(&at(Vec3::new(1.0, 3.0, 0.0)), push);
    assert_eq!(p.force, Vec3::new(0.0, 2.0, 0.0));
    let p = after_one_update(&at(Vec3::new(1.0, -5.0, 0.0)), push);
    assert_eq!(p.force, Vec3::ZERO);
}

#[test]
fn inherit_velocity_scales_owner_velocity() {
    let mut sim = single_particle(
        "",
        r#"(module: "Inherit Velocity", inputs: { "Velocity Scale": Float(0.5) })"#,
        "",
    );
    sim.owner_velocity = Vec3::new(1.0, 2.0, 3.0);
    sim.step(DT);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(0.5, 1.0, 1.5));
    // And Engine.Owner.Velocity links like any engine value.
    let mut sim = single_particle(
        "",
        r#"(module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Engine.Owner.Velocity") })"#,
        "",
    );
    sim.owner_velocity = Vec3::new(4.0, 0.0, 0.0);
    sim.step(DT);
    assert_eq!(sim.particles(0)[0].velocity, Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn forces_move_particles_through_solve() {
    // Linear Force 2 N on 1 kg for 0.25 s: v = 0.5, x = 0.125 (semi-implicit Euler).
    let p = after_one_update(
        "",
        r#"(module: "Linear Force", inputs: { "Linear Force": Vec3((2.0, 0.0, 0.0)) }), (module: "Solve Forces and Velocity")"#,
    );
    assert_eq!(p.velocity, Vec3::new(0.5, 0.0, 0.0));
    assert_eq!(p.position, Vec3::new(0.125, 0.0, 0.0));
}

// ---- shapes ----

fn shape() -> ShapeInputs {
    ShapeInputs {
        surface: false,
        sphere_radius: 2.0,
        box_size: Vec3::new(2.0, 4.0, 6.0),
        cylinder_height: 4.0,
        cylinder_radius: 2.0,
        large_radius: 2.0,
        handle_radius: 0.5,
        cone_length: 2.0,
        cone_angle: 45.0,
        ring_radius: 2.0,
        disc_coverage: 0.0,
        ..ShapeInputs::default()
    }
}

#[test]
fn shape_points_by_hand() {
    let s = shape();
    let surf = ShapeInputs {
        surface: true,
        ..shape()
    };
    // Sphere: y = 1 - 2u0 = 0, phi = 0, radius * (1 - 0)^(1/3) = 2.
    assert_eq!(shape_point(0, &s, [0.5, 0.0, 0.0, 0.0, 0.0]), Vec3::new(2.0, 0.0, 0.0));
    // Sphere: u0 = 0 is the pole.
    assert!(close(shape_point(0, &surf, [0.0, 0.3, 0.9, 0.0, 0.0]), Vec3::new(0.0, 2.0, 0.0)));
    // Box volume: (u - 0.5) * size.
    assert_eq!(
        shape_point(1, &s, [0.75, 0.25, 1.0, 0.0, 0.0]),
        Vec3::new(0.5, -1.0, 3.0)
    );
    // Box surface: face areas 24 (x), 12 (y), 8 (z) of 44; t = 0.5 * 44 = 22 < 24 picks
    // an X face, u4 >= 0.5 its + side.
    assert_eq!(
        shape_point(1, &surf, [0.75, 0.25, 0.5, 0.5, 0.75]),
        Vec3::new(1.0, -1.0, 0.0)
    );
    // t = 0.75 * 44 = 33 in [24, 36): a Y face, - side.
    assert_eq!(
        shape_point(1, &surf, [0.75, 0.25, 0.5, 0.75, 0.25]),
        Vec3::new(0.5, -2.0, 0.0)
    );
    // Cylinder: phi 0, radius 2 * sqrt(1 - 0.75) = 1, y = (0.75 - 0.5) * 4 = 1.
    assert_eq!(shape_point(2, &s, [0.0, 0.75, 0.75, 0.0, 0.0]), Vec3::new(1.0, 1.0, 0.0));
    assert_eq!(shape_point(2, &surf, [0.0, 0.75, 0.75, 0.0, 0.0]), Vec3::new(2.0, 1.0, 0.0));
    // Torus: theta 0, psi 0, rho = 0.5 * sqrt(1 - 0.75) = 0.25: x = 2 + 0.25.
    assert_eq!(shape_point(3, &s, [0.0, 0.0, 0.75, 0.0, 0.0]), Vec3::new(2.25, 0.0, 0.0));
    // Torus surface, psi = 90°: top of the tube.
    assert!(close(shape_point(3, &surf, [0.0, 0.25, 0.0, 0.0, 0.0]), Vec3::new(2.0, 0.5, 0.0)));
    // Cone surface, u0 = 0: the rim, h = 2, r = 2 * tan 45° = 2.
    assert!(close(shape_point(4, &surf, [0.0, 0.0, 0.0, 0.0, 0.0]), Vec3::new(2.0, 2.0, 0.0)));
    // Cone volume, u0 = 7/8: h = 2 * (1/8)^(1/3) = 1, r = 1 * tan 45° * sqrt(1 - 0.75) = 0.5.
    assert!(close(shape_point(4, &s, [0.875, 0.75, 0.0, 0.0, 0.0]), Vec3::new(0.5, 1.0, 0.0)));
    // Ring: on the circle with coverage 0, half-way in by area with coverage 1.
    assert_eq!(shape_point(5, &s, [0.0, 0.75, 0.0, 0.0, 0.0]), Vec3::new(2.0, 0.0, 0.0));
    let disc = ShapeInputs {
        disc_coverage: 1.0,
        ..shape()
    };
    assert_eq!(shape_point(5, &disc, [0.0, 0.75, 0.0, 0.0, 0.0]), Vec3::new(1.0, 0.0, 0.0));
}

/// 400 particles from a Shape Location with `inputs` (RON map body).
fn shape_particles(inputs: &str) -> Vec<Particle> {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(400) })"#,
        &format!(r#"(module: "Shape Location", inputs: {{ {inputs} }})"#),
        "",
    );
    let mut sim = Simulation::new(compile(&src), 3);
    sim.step(DT);
    sim.particles(0).to_vec()
}

#[test]
fn shape_location_fills_each_shape() {
    let off = Vec3::new(10.0, 0.0, 0.0);
    // Sphere volume: inside, mean distance 3/4 of the radius.
    let ps = shape_particles(r#""Sphere Radius": Float(2.0), "Offset": Vec3((10.0, 0.0, 0.0))"#);
    let d: Vec<f32> = ps.iter().map(|p| (p.position - off).length()).collect();
    assert!(d.iter().all(|&r| r <= 2.0 + 1e-5));
    let mean = d.iter().sum::<f32>() / d.len() as f32;
    assert!((mean - 1.5).abs() < 0.08, "mean radius {mean}");
    // Sphere surface.
    let ps = shape_particles(r#""Sphere Radius": Float(2.0), "Surface Only": Bool(true)"#);
    assert!(ps.iter().all(|p| (p.position.length() - 2.0).abs() < 1e-5));
    // Box surface: every point on a face, all six faces used.
    let ps = shape_particles(r#""Shape Primitive": Box, "Box Size": Vec3((2.0, 4.0, 6.0)), "Surface Only": Bool(true)"#);
    let h = Vec3::new(1.0, 2.0, 3.0);
    let mut faces = [0; 6];
    for p in &ps {
        let q = p.position;
        assert!(q.abs().cmple(h + 1e-5).all());
        let on: Vec<usize> = (0..3).filter(|&a| (q[a].abs() - h[a]).abs() < 1e-5).collect();
        assert!(!on.is_empty(), "{q} is not on a face");
        faces[on[0] * 2 + (q[on[0]] > 0.0) as usize] += 1;
    }
    assert!(faces.iter().all(|&n| n > 0), "faces {faces:?}");
    // Cylinder: within radius and height, about Y.
    let ps = shape_particles(r#""Shape Primitive": Cylinder, "Cylinder Radius": Float(1.0), "Cylinder Height": Float(4.0)"#);
    assert!(ps.iter().all(|p| Vec2::new(p.position.x, p.position.z).length() <= 1.0 + 1e-5
        && p.position.y.abs() <= 2.0));
    // Torus: within the tube.
    let ps = shape_particles(r#""Shape Primitive": Torus, "Large Radius": Float(2.0), "Handle Radius": Float(0.5)"#);
    assert!(ps.iter().all(|p| {
        let ring = Vec2::new(p.position.x, p.position.z).length() - 2.0;
        Vec2::new(ring, p.position.y).length() <= 0.5 + 1e-5
    }));
    // Cone: inside the 30° half-angle, up to its length.
    let ps = shape_particles(r#""Shape Primitive": Cone, "Cone Length": Float(3.0), "Cone Angle": Float(30.0)"#);
    let tan = 30f32.to_radians().tan();
    assert!(ps.iter().all(|p| (0.0..=3.0 + 1e-5).contains(&p.position.y)
        && Vec2::new(p.position.x, p.position.z).length() <= p.position.y * tan + 1e-5));
    // Ring: on the circle in the XZ plane.
    let ps = shape_particles(r#""Shape Primitive": Ring, "Ring Radius": Float(1.5)"#);
    assert!(ps.iter().all(|p| p.position.y == 0.0
        && (Vec2::new(p.position.x, p.position.z).length() - 1.5).abs() < 1e-5));
}

#[test]
fn static_switch_inputs_take_names() {
    let bad = |inputs: &str| {
        let src = one_emitter(
            "",
            r#"(module: "Spawn Rate")"#,
            &format!(r#"(module: "Shape Location", inputs: {{ {inputs} }})"#),
            "",
        );
        let c = fx_core::check_source("t.fx.ron", &src);
        c.errors().map(|d| d.to_string()).collect::<Vec<_>>().join("\n")
    };
    let e = bad(r#""Shape Primitive": Bxo"#);
    assert!(e.contains("has no choice `Bxo`") && e.contains("did you mean `Box`"), "{e}");
    let e = bad(r#""Shape Primitive": Link(Int, "User.Shape")"#);
    assert!(e.contains("is a static switch"), "{e}");
    let e = bad(r#""Sphere Radius": Big"#);
    assert!(e.contains("is Float, but the value given is the bare name `Big`"), "{e}");
    // The index is accepted too.
    assert!(bad(r#""Shape Primitive": Int(2)"#).is_empty());
    assert!(bad(r#""Shape Primitive": Ring"#).is_empty());
}

// ---- rotation ----

#[test]
fn initial_mesh_orientation_sets_quaternion() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initial Mesh Orientation", inputs: { "Rotation": Vec3((0.0, 90.0, 0.0)) })"#,
        "",
    );
    sim.step(DT);
    let q = sim.particles(0)[0].mesh_orientation;
    let want = Quat::from_rotation_y(90f32.to_radians());
    assert!(q.abs_diff_eq(want, 1e-6), "{q} vs {want}");
}

#[test]
fn sprite_rotation_rate_turns_sprites() {
    // 10° + 8°/s * 0.25 s * 2 updates.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Sprite Rotation": Float(10.0) })"#,
        r#"(module: "Sprite Rotation Rate", inputs: { "Rotation Rate": Float(8.0) })"#,
    );
    sim.step(DT);
    sim.step(DT);
    sim.step(DT);
    assert_eq!(sim.particles(0)[0].sprite_rotation, 14.0);
}

// ---- sub-UV ----

/// One particle of lifetime 1 with `update`, drawn by `renderer`; after
/// `updates` updates of 0.25 s.
fn sub_uv(update: &str, renderer: &str, updates: usize) -> Particle {
    let src = format!(
        r#"(
    name: "T",
    emitters: [(
        name: "E",
        max_particles: 10,
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int(1) }})],
        particle_spawn: [(module: "Initialize Particle", inputs: {{ "Lifetime": Float(1.0) }})],
        particle_update: [{STATE}, {update}],
        renderers: [{renderer}],
    )],
)"#
    );
    let mut sim = Simulation::new(compile(&src), 5);
    for _ in 0..=updates {
        sim.step(DT);
    }
    sim.particles(0)[0]
}

#[test]
fn sub_uv_linear_plays_over_life() {
    // NormalizedAge 0.25 of 4 frames from frame 2: 2 + 1.
    let m = r#"(module: "SubUV Animation", inputs: { "Number of Frames": Int(4), "Start Frame": Int(2) })"#;
    assert_eq!(sub_uv(m, "Sprite", 1).sub_image_index, 3.0);
    // Number of Frames 0 reads the renderer's grid: 4 x 2 = 8, so 0.5 → frame 4.
    let m = r#"(module: "SubUV Animation")"#;
    assert_eq!(sub_uv(m, "Sprite((sub_image_size: (4, 2)))", 2).sub_image_index, 4.0);
}

#[test]
fn sub_uv_infinite_loop_wraps() {
    // Age 0.75 * 8 fps = 6 frames into a loop of 4: frame 2.
    let m = r#"(module: "SubUV Animation", inputs: { "SubUV Animation Mode": InfiniteLoop,
        "Number of Frames": Int(4), "Frames Per Second": Float(8.0) })"#;
    assert_eq!(sub_uv(m, "Sprite", 3).sub_image_index, 2.0);
}

#[test]
fn sub_uv_random_picks_one_whole_frame_per_particle() {
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })"#,
        "",
        r#"(module: "SubUV Animation", inputs: { "SubUV Animation Mode": Random, "Number of Frames": Int(4), "Start Frame": Int(1) })"#,
    );
    let mut sim = Simulation::new(compile(&src), 9);
    sim.step(DT);
    sim.step(DT);
    let first: Vec<f32> = sim.particles(0).iter().map(|p| p.sub_image_index).collect();
    sim.step(DT);
    let second: Vec<f32> = sim.particles(0).iter().map(|p| p.sub_image_index).collect();
    assert_eq!(first, second, "the frame changes between steps");
    for f in 1..=4 {
        assert!(first.contains(&(f as f32)), "frame {f} never chosen");
    }
    assert!(first.iter().all(|&f| f.fract() == 0.0 && (1.0..=4.0).contains(&f)));
}

// ---- scaling and colour ----

#[test]
fn scale_mesh_size_multiplies_initial_scale() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Mesh", inputs: { "Mesh Scale": Vec3((1.0, 2.0, 3.0)) })"#,
        r#"(module: "Scale Mesh Size", inputs: { "Scale Factor": Vec3((2.0, 2.0, 0.5)) })"#,
    );
    for _ in 0..3 {
        sim.step(DT);
    }
    let p = sim.particles(0)[0];
    assert_eq!(p.initial_scale, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(p.scale, Vec3::new(2.0, 4.0, 1.5));
}

#[test]
fn scale_ribbon_width_multiplies_initial_width() {
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Ribbon", inputs: { "Ribbon Width": Float(0.5) })"#,
        r#"(module: "Scale Ribbon Width", inputs: { "Ribbon Width Scale": Float(3.0) })"#,
    );
    for _ in 0..3 {
        sim.step(DT);
    }
    assert_eq!(sim.particles(0)[0].ribbon_width, 1.5);
}

#[test]
fn color_sets_color_from_curve() {
    // NormalizedAge 0.25 along red → transparent blue.
    let mut sim = single_particle(
        "",
        r#"(module: "Initialize Particle", inputs: { "Lifetime": Float(1.0), "Color": Color((0.0, 1.0, 0.0, 1.0)) })"#,
        &format!(
            r#"{STATE}, (module: "Color", inputs: {{ "Color": CurveOverAge(Color([(0.0, (1.0, 0.0, 0.0, 1.0)), (1.0, (0.0, 0.0, 1.0, 0.0))])) }})"#
        ),
    );
    sim.step(DT);
    sim.step(DT);
    let p = sim.particles(0)[0];
    assert_eq!(p.color, Vec4::new(0.75, 0.0, 0.25, 0.75));
    assert_eq!(p.initial_color, Vec4::new(0.0, 1.0, 0.0, 1.0));
}

#[test]
fn scale_color_by_speed_lerps_by_speed() {
    // Speed 2 of threshold 4: half-way from (0, 0, 0, 1) to (1, 1, 1, 1).
    let p = after_one_update(
        r#"(module: "Initialize Particle", inputs: { "Color": Color((1.0, 0.5, 0.25, 1.0)) }),
           (module: "Add Velocity", inputs: { "Velocity": Vec3((0.0, 2.0, 0.0)) })"#,
        r#"(module: "Scale Color by Speed", inputs: { "Velocity Threshold": Float(4.0),
            "Min Scale Factor": Color((0.0, 0.0, 0.0, 1.0)), "Max Scale Factor": Color((1.0, 1.0, 1.0, 1.0)) })"#,
    );
    assert_eq!(p.color, Vec4::new(0.5, 0.25, 0.125, 1.0));
}

#[test]
fn scale_sprite_size_by_speed_lerps_and_clamps() {
    let run = |threshold: f32| {
        after_one_update(
            r#"(module: "Initialize Particle", inputs: { "Sprite Size": Vec2((0.25, 0.5)) }),
               (module: "Add Velocity", inputs: { "Velocity": Vec3((3.0, 4.0, 0.0)) })"#,
            &format!(
                r#"(module: "Scale Sprite Size by Speed", inputs: {{ "Velocity Threshold": Float({threshold:.1}) }})"#
            ),
        )
        .sprite_size
    };
    // Speed 5 over threshold 2.5: clamped to Max (1, 2).
    assert_eq!(run(2.5), Vec2::new(0.25, 1.0));
    // Speed 5 of 10: (1, 1.5).
    assert_eq!(run(10.0), Vec2::new(0.25, 0.75));
}

// ---- kill ----

#[test]
fn kill_particles_in_volume() {
    let kill = |shape: &str, invert: bool| {
        format!(
            r#"(module: "Kill Particles in Volume", inputs: {{ "Volume Shape": {shape},
                "Volume Center": Vec3((1.0, 0.0, 0.0)), "Sphere Radius": Float(1.0),
                "Box Size": Vec3((2.0, 2.0, 2.0)), "Invert Volume": Bool({invert}) }})"#
        )
    };
    let inside_sphere = at(Vec3::new(1.5, 0.0, 0.0));
    let in_box_corner = at(Vec3::new(1.9, 0.9, -0.9));
    let outside = at(Vec3::new(2.1, 0.0, 0.0));
    assert!(!alive_after_one_update(&inside_sphere, &kill("Sphere", false)));
    assert!(alive_after_one_update(&in_box_corner, &kill("Sphere", false)));
    assert!(!alive_after_one_update(&in_box_corner, &kill("Box", false)));
    assert!(alive_after_one_update(&outside, &kill("Box", false)));
    assert!(!alive_after_one_update(&outside, &kill("Box", true)));
    assert!(alive_after_one_update(&inside_sphere, &kill("Sphere", true)));
}
