//! Inputs added for Niagara imports: Scale Color's Scale RGBA, Initialize
//! Particle's HSV adjustment, Add Velocity in Cone's inner angle,
//! distribution and speed falloff, Shape Location's box midpoint and cone
//! options, Wind Force turbulence, Point Attraction Force's attracted offset
//! and Dynamic Material Parameters 3 and 4. Their defaults leave results as
//! they were.

mod common;

use bevy_math::{Vec3, Vec4};
use common::*;
use fx_core::sim::{Particle, ShapeInputs, Simulation, cone_speed_falloff, hsv_shift, input_salt, shape_point};

const DT: f32 = 0.25;

/// The particles right after the spawn of `n` particles.
fn spawn_many(n: u32, spawn: &str) -> Vec<Particle> {
    let src = one_emitter(
        "",
        &format!(r#"(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int({n}) }})"#),
        spawn,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 7);
    sim.step(DT);
    sim.particles(0).to_vec()
}

fn after_one_update(spawn: &str, update: &str) -> Particle {
    let mut sim = single_particle("", spawn, update);
    sim.step(DT);
    sim.step(DT);
    sim.particles(0)[0]
}

#[test]
fn scale_rgba_multiplies_rgb_and_alpha() {
    let init = r#"(module: "Initialize Particle", inputs: { "Color": Color((1.0, 0.5, 0.25, 1.0)) })"#;
    let p = after_one_update(init, r#"(module: "Scale Color", inputs: { "Scale RGBA": Vec4((0.5, 0.5, 0.5, 0.25)) })"#);
    assert_eq!(p.color, Vec4::new(0.5, 0.25, 0.125, 0.25));
    // With Scale RGB / Alpha too, all three multiply.
    let p = after_one_update(
        init,
        r#"(module: "Scale Color", inputs: { "Scale RGB": Vec3((2.0, 2.0, 2.0)), "Scale Alpha": Float(0.5), "Scale RGBA": Vec4((0.5, 1.0, 1.0, 0.5)) })"#,
    );
    assert_eq!(p.color, Vec4::new(1.0, 1.0, 0.5, 0.25));
    // The default RGBA is the identity.
    let p = after_one_update(init, r#"(module: "Scale Color", inputs: { "Scale RGB": Vec3((0.3, 0.7, 0.9)), "Scale Alpha": Float(0.6) })"#);
    assert_eq!(p.color, Vec4::new(1.0, 0.5, 0.25, 1.0) * Vec4::new(0.3, 0.7, 0.9, 0.6));
}

#[test]
fn hsv_shift_matches_niagara() {
    let c = Vec4::new(0.8, 0.4, 0.2, 0.7);
    // No change: hue 0, saturation and value 1 (the YIQ matrix is the identity
    // to its 3-digit constants).
    let same = hsv_shift(c, 0.0, 1.0, 1.0);
    assert!((same - c).abs().max_element() < 2e-3, "{same}");
    // Value scales the colour; saturation 0 leaves grey of the same luma.
    assert!((hsv_shift(Vec4::splat(0.5), 0.0, 1.0, 0.5) - Vec4::new(0.25, 0.25, 0.25, 0.5)).abs().max_element() < 1e-3);
    let luma = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let grey = hsv_shift(c, 0.0, 0.0, 1.0);
    for k in 0..3 {
        assert_close(grey[k], luma);
    }
    assert_eq!(grey.w, c.w);
    // A hue turn keeps luma and moves the colour; a full turn comes back.
    let turned = hsv_shift(c, 120.0, 1.0, 1.0);
    assert!((turned - c).abs().max_element() > 0.1);
    assert!((0.299 * turned.x + 0.587 * turned.y + 0.114 * turned.z - luma).abs() < 2e-3);
    assert!((hsv_shift(c, 360.0, 1.0, 1.0) - c).abs().max_element() < 2e-3);
    // Negative saturation and value count as 0.
    assert_eq!(hsv_shift(c, 30.0, -1.0, -1.0).truncate(), Vec3::ZERO);
}

#[test]
fn initialize_particle_hsv_and_alpha() {
    let base = r#""Color": Color((0.8, 0.4, 0.2, 0.8))"#;
    let get = |extra: &str| after_one_update(&format!(r#"(module: "Initialize Particle", inputs: {{ {base}{extra} }})"#), "");
    // Defaults: the colour exactly as given.
    assert_eq!(get("").color, Vec4::new(0.8, 0.4, 0.2, 0.8));
    assert_eq!(get(r#", "Hue Shift": Float(0.0), "Saturation Scale": Float(1.0), "Value Scale": Float(1.0)"#).color, Vec4::new(0.8, 0.4, 0.2, 0.8));
    // Hue, saturation and value as hsv_shift; Alpha Scale multiplies alpha.
    let p = get(r#", "Hue Shift": Float(90.0), "Saturation Scale": Float(0.5), "Value Scale": Float(2.0), "Alpha Scale": Float(0.5)"#);
    let want = hsv_shift(Vec4::new(0.8, 0.4, 0.2, 0.8), 90.0, 0.5, 2.0);
    assert_vec_close(&p.color.truncate().to_array(), &want.truncate().to_array());
    assert_close(p.color.w, 0.4);
    assert_eq!(p.initial_color, p.color);
    // A random hue range varies per particle.
    let ps = spawn_many(64, &format!(r#"(module: "Initialize Particle", inputs: {{ {base}, "Hue Shift": RandomRange(Float(-36.0), Float(36.0)) }})"#));
    let distinct = ps.iter().filter(|p| (p.color.x - ps[0].color.x).abs() > 1e-4).count();
    assert!(distinct > 50, "{distinct}");
}

fn angle_deg(v: Vec3) -> f32 {
    v.normalize().dot(Vec3::Y).clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn cone_inner_angle_distribution_and_falloff() {
    let cone = |extra: &str| {
        spawn_many(
            2000,
            &format!(r#"(module: "Add Velocity in Cone", inputs: {{ "Cone Angle": Float(60.0), "Velocity Strength": Float(2.0){extra} }})"#),
        )
    };
    // Inner Cone Angle: every direction between 30° and 60° from the axis.
    for p in cone(r#", "Inner Cone Angle": Float(30.0)"#) {
        let a = angle_deg(p.velocity);
        assert!((29.9..=60.1).contains(&a), "{a}");
        assert_close(p.velocity.length(), 2.0);
    }
    // Distribution 1: the cone angle scales by saturate(random -0.5..1), so
    // a third of the particles go along the axis.
    let ps = cone(r#", "Distribution Along Cone Axis": Float(1.0)"#);
    let on_axis = ps.iter().filter(|p| angle_deg(p.velocity) < 0.1).count() as f32 / ps.len() as f32;
    assert!((0.28..0.39).contains(&on_axis), "{on_axis}");
    // Without it directions are uniform over the cap: hardly any on the axis.
    let ps = cone("");
    assert!(ps.iter().filter(|p| angle_deg(p.velocity) < 0.1).count() < 5);
    // Speed Falloff 1: speed * cos(angle)^10; 0.5: lerp(1, cos^5, 0.5).
    for (f, ps) in [(1.0f32, cone(r#", "Speed Falloff": Float(1.0)"#)), (0.5, cone(r#", "Speed Falloff": Float(0.5)"#))] {
        for p in ps {
            let c = p.velocity.normalize().y;
            let want = 2.0 * (1.0 + (c.max(0.0).powf(10.0 * f) - 1.0) * f);
            assert!((p.velocity.length() - want).abs() < 1e-4, "{} {want}", p.velocity.length());
        }
    }
    assert_eq!(cone_speed_falloff(0.3, 0.0), 1.0);
    assert_eq!(cone_speed_falloff(1.0, 1.0), 1.0);
}

#[test]
fn cone_defaults_leave_velocities_unchanged() {
    let a = spawn_many(200, r#"(module: "Add Velocity in Cone", inputs: { "Cone Angle": Float(40.0) })"#);
    let b = spawn_many(
        200,
        r#"(module: "Add Velocity in Cone", inputs: { "Cone Angle": Float(40.0), "Inner Cone Angle": Float(0.0), "Distribution Along Cone Axis": Float(0.0), "Speed Falloff": Float(0.0) })"#,
    );
    for (p, q) in a.iter().zip(&b) {
        assert_eq!(p.velocity, q.velocity);
    }
}

#[test]
fn box_midpoint_moves_the_box() {
    let s = ShapeInputs { box_size: Vec3::new(2.0, 4.0, 6.0), box_midpoint: Vec3::ZERO, ..ShapeInputs::default() };
    assert_eq!(shape_point(1, &s, [0.5, 0.25, 1.0, 0.0, 0.0]), Vec3::new(1.0, 1.0, 6.0));
    // Surface: the faces are at 0 and the full size.
    let surf = ShapeInputs { surface: true, ..s };
    assert_eq!(shape_point(1, &surf, [0.75, 0.25, 0.5, 0.5, 0.25]), Vec3::new(0.0, 1.0, 3.0));
    assert_eq!(shape_point(1, &surf, [0.75, 0.25, 0.5, 0.5, 0.75]), Vec3::new(2.0, 1.0, 3.0));
    // In a simulation, with a random midpoint per particle.
    let ps = spawn_many(
        300,
        r#"(module: "Shape Location", inputs: { "Shape Primitive": Box, "Box Size": Vec3((1.0, 1.0, 1.0)), "Box Midpoint": Vec3((0.0, 1.0, 0.5)) })"#,
    );
    for p in ps {
        assert!(p.position.x >= 0.0 && p.position.y <= 0.0 && p.position.z.abs() <= 0.5, "{}", p.position);
    }
}

#[test]
fn cone_shape_options() {
    let base = ShapeInputs { cone_length: 2.0, cone_angle: 45.0, ..ShapeInputs::default() };
    // Inner angle 30°: the point is at least tan 30° * h from the axis.
    let hollow = ShapeInputs { cone_inner_angle: 30.0, ..base };
    for k in 0..50 {
        let u = [k as f32 / 50.0, (k * 7 % 50) as f32 / 50.0, (k * 3 % 50) as f32 / 50.0, 0.0, 0.0];
        let p = shape_point(4, &hollow, u);
        let r = (p.x * p.x + p.z * p.z).sqrt();
        assert!(r >= p.y * 30f32.to_radians().tan() - 1e-5 && r <= p.y + 1e-5, "{p}");
    }
    // u1 = 0 is the outer wall, u1 -> 1 the inner one.
    let p = shape_point(4, &hollow, [0.875, 0.0, 0.0, 0.0, 0.0]);
    assert!((p - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5, "{p}");
    // Radial angle 90°: a quarter turn.
    let quarter = ShapeInputs { cone_radial_angle: 90.0, ..base };
    let p = shape_point(4, &quarter, [0.875, 0.75, 1.0, 0.0, 0.0]);
    assert!(p.x.abs() < 1e-5 && (p.z - 0.5).abs() < 1e-5, "{p}");
    // Surface distribution 1: only the base (h = length).
    let base_only = ShapeInputs { cone_surface_distribution: 1.0, ..base };
    for u0 in [0.0, 0.3, 0.99] {
        assert_close(shape_point(4, &base_only, [u0, 0.5, 0.5, 0.0, 0.0]).y, 2.0);
    }
    // 0.5: the half of the volume nearest the apex stays empty.
    let half = ShapeInputs { cone_surface_distribution: 0.5, ..base };
    assert_close(shape_point(4, &half, [0.999_999, 0.5, 0.5, 0.0, 0.0]).y, 2.0 * 0.5f32.powf(1.0 / 3.0));
    // Defaults: the same points as before these inputs.
    for u in [[0.875f32, 0.75, 0.3, 0.0, 0.0], [0.1, 0.2, 0.9, 0.0, 0.0]] {
        let h = 2.0 * (1.0 - u[0]).powf(1.0 / 3.0);
        let r = h * (45.0 * std::f32::consts::PI / 180.0).tan() * (1.0 - u[1]).sqrt();
        let phi = std::f32::consts::TAU * u[2];
        assert_eq!(shape_point(4, &base, u), Vec3::new(r * phi.cos(), h, r * phi.sin()));
    }
}

#[test]
fn wind_turbulence() {
    let init = r#"(module: "Initialize Particle", inputs: { "Position": Vec3((0.3, 0.7, -0.2)) })"#;
    let wind = r#""Wind Velocity": Vec3((2.0, 0.0, 0.0)), "Air Resistance": Float(1.0)"#;
    let force = |extra: &str| after_one_update(init, &format!(r#"(module: "Wind Force", inputs: {{ {wind}{extra} }})"#)).force;
    // No turbulence: (wind - velocity) * resistance.
    assert_eq!(force(""), Vec3::new(2.0, 0.0, 0.0));
    assert_eq!(force(r#", "Turbulence Frequency": Float(3.0), "Turbulence Offset By Time": Vec3((1.0, 1.0, 1.0))"#), Vec3::new(2.0, 0.0, 0.0));
    // Turbulence: curl noise at (Position + Offset * time) * Frequency, times |wind| * scale.
    let f = force(r#", "Turbulence Scale": Float(0.5), "Turbulence Frequency": Float(2.0), "Turbulence Offset By Time": Vec3((0.0, -1.0, 0.0)), "Turbulence Position": Link(Vec3, "Particles.Position")"#);
    let time = DT;
    let at = Vec3::new(0.3, 0.7, -0.2) + Vec3::new(0.0, -1.0, 0.0) * time;
    let want = Vec3::new(2.0, 0.0, 0.0) + fx_core::sim::noise::curl_noise(at * 2.0) * (2.0 * 0.5);
    assert_vec_close(&f.to_array(), &want.to_array());
    assert!((f - Vec3::new(2.0, 0.0, 0.0)).length() > 1e-3);
}

#[test]
fn point_attraction_measures_from_the_offset_position() {
    let attract = |pos: &str, off: &str| {
        after_one_update(
            &format!(r#"(module: "Initialize Particle", inputs: {{ "Position": {pos} }})"#),
            &format!(r#"(module: "Point Attraction Force", inputs: {{ "Attractor Position": Vec3((2.0, 0.0, 0.0)), "Attraction Radius": Float(4.0), "Attraction Strength": Float(1.0){off} }})"#),
        )
        .force
    };
    let moved = attract("Vec3((0.0, 0.0, 0.0))", r#", "Attracted Position Offset": Vec3((1.0, 0.0, 0.0))"#);
    let at = attract("Vec3((1.0, 0.0, 0.0))", "");
    assert_eq!(moved, at);
    assert!(moved.x > 0.0);
    assert_ne!(moved, attract("Vec3((0.0, 0.0, 0.0))", ""));
}

#[test]
fn material_parameters_2_and_3() {
    let update = r#"(module: "Dynamic Material Parameters 3", inputs: { "Dynamic Material Parameter 2": Vec4((1.0, 2.0, 3.0, 4.0)) }),
        (module: "Dynamic Material Parameters 4", inputs: { "Dynamic Material Parameter 3": Link(Vec4, "Particles.DynamicMaterialParameter2") })"#;
    let p = after_one_update(r#"(module: "Initialize Particle")"#, update);
    assert_eq!(p.dynamic_material_parameter_2, Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(p.dynamic_material_parameter_3, Vec4::new(1.0, 2.0, 3.0, 4.0));
    let uses = |update: &str| {
        let e = &compile(&one_emitter("", "", r#"(module: "Initialize Particle")"#, update)).emitters[0];
        (e.uses_extended_attributes(), e.uses_material_parameters_2_3())
    };
    assert_eq!(uses(update), (true, true));
    assert_eq!(uses(r#"(module: "Dynamic Material Parameters 2")"#), (true, false));
    assert_eq!(uses(r#"(module: "Drag", inputs: { "Drag": Dynamic("Break Vec4", { "Value": Link(Vec4, "Particles.DynamicMaterialParameter3"), "Component": X }) })"#), (true, true));
    assert_eq!(uses(""), (false, false));
}

#[test]
fn input_salts_skip_the_module_draws() {
    assert_eq!(input_salt(0), 0);
    assert_eq!(input_salt(15), 120);
    assert_eq!(input_salt(16), 256);
    assert_eq!(input_salt(17), 264);
    // Shape Location's 17th input (Cone Surface Distribution) draws apart from the shape's own draws.
    let ps = spawn_many(
        200,
        r#"(module: "Shape Location", inputs: { "Shape Primitive": Cone, "Cone Length": Float(1.0), "Cone Surface Distribution": RandomRange(Float(0.0), Float(1.0)) })"#,
    );
    for p in &ps {
        assert!(p.position.y <= 1.0 + 1e-5 && p.position.y >= 0.0);
    }
}
