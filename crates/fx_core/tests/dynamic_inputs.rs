//! Dynamic inputs (Niagara's Uniform Ranged Float, Multiply Vector, Float
//! from Curve, ...): one test per kind on the CPU, nesting, and the
//! validator's errors.

mod common;

use bevy_math::{Vec2, Vec3, Vec4};
use common::*;
use fx_core::cook;
use fx_core::dynamic::DYNAMIC_INPUTS;
use fx_core::sim::{Particle, Simulation};

const DT: f32 = 0.25;

/// The particle after one step whose Initialize Particle sets `inputs`.
fn spawn(inputs: &str) -> Particle {
    spawn_with("", inputs, "")
}

fn spawn_with(user: &str, inputs: &str, more_spawn: &str) -> Particle {
    let mut sim = single_particle(
        user,
        &format!(r#"(module: "Initialize Particle", inputs: {{ {inputs} }}), {more_spawn}"#),
        "",
    );
    sim.step(DT);
    assert_eq!(sim.particle_count(), 1);
    sim.particles(0)[0]
}

fn float(e: &str) -> f32 {
    spawn(&format!(r#""Lifetime": {e}"#)).lifetime
}
fn vec2(e: &str) -> Vec2 {
    spawn(&format!(r#""Sprite Size": {e}"#)).sprite_size
}
fn vec3(e: &str) -> Vec3 {
    spawn(&format!(r#""Position": {e}"#)).position
}
fn color(e: &str) -> Vec4 {
    spawn(&format!(r#""Color": {e}"#)).color
}
fn vec4(e: &str) -> Vec4 {
    spawn_with(
        "",
        "",
        &format!(r#"(module: "Dynamic Material Parameters", inputs: {{ "Dynamic Material Parameter": {e} }})"#),
    )
    .dynamic_material_parameter
}
fn int(e: &str) -> i32 {
    spawn_with("", "", &format!(r#"(module: "Initialize Ribbon", inputs: {{ "Ribbon ID": {e} }})"#)).ribbon_id
}
/// Whether a Bool input given to Kill Particles kills the new particle.
fn bool_(e: &str) -> bool {
    let mut sim = single_particle(
        "",
        &format!(r#"(module: "Kill Particles", inputs: {{ "Kill Particles": {e} }})"#),
        "",
    );
    sim.step(DT);
    sim.particle_count() == 0
}

/// Many particles from one spawn stack, after one step.
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

fn error(src: &str) -> String {
    let c = fx_core::check_source("test.fx.ron", src);
    assert!(c.cooked.is_none(), "expected errors:\n{src}");
    c.errors().map(|d| d.to_string()).collect::<Vec<_>>().join("\n")
}

fn spawn_error(inputs: &str) -> String {
    error(&one_emitter(
        "",
        r#"(module: "Spawn Rate")"#,
        &format!(r#"(module: "Initialize Particle", inputs: {{ {inputs} }})"#),
        "",
    ))
}

#[test]
fn uniform_ranged_draws_each_component_in_range() {
    let ps = spawn_many(
        200,
        r#"(module: "Initialize Particle", inputs: {
            "Lifetime": Dynamic("Uniform Ranged Float", { "Minimum": Float(1.0), "Maximum": Float(3.0) }),
            "Position": Dynamic("Uniform Ranged Vec3", { "Minimum": Vec3((0.0, 0.0, 0.0)), "Maximum": Vec3((1.0, 2.0, 4.0)) }),
            "Color": Dynamic("Uniform Ranged Color", { "Minimum": Color((0.0, 0.0, 0.0, 1.0)), "Maximum": Color((1.0, 1.0, 1.0, 1.0)) }),
            "Sprite Size": Dynamic("Uniform Ranged Vec2", { "Minimum": Vec2((0.1, 0.1)), "Maximum": Vec2((0.2, 0.2)) }),
        }),
        (module: "Initialize Ribbon", inputs: { "Ribbon ID": Dynamic("Uniform Ranged Int", { "Minimum": Int(2), "Maximum": Int(4) }) }),
        (module: "Dynamic Material Parameters", inputs: { "Dynamic Material Parameter": Dynamic("Uniform Ranged Vec4", { "Maximum": Vec4((1.0, 1.0, 1.0, 1.0)) }) })"#,
    );
    assert_eq!(ps.len(), 200);
    let mut ids = [0; 5];
    for p in &ps {
        assert!((1.0..3.0).contains(&p.lifetime));
        let q = p.position;
        assert!(q.x < 1.0 && q.y < 2.0 && q.z < 4.0 && q.min_element() >= 0.0);
        assert_eq!(p.color.w, 1.0);
        assert!(p.sprite_size.min_element() >= 0.1 && p.sprite_size.max_element() < 0.2);
        ids[p.ribbon_id as usize] += 1;
        assert!(p.dynamic_material_parameter.min_element() >= 0.0);
    }
    // Ints are whole numbers from 2 to 4 inclusive, all of them hit.
    assert_eq!(ids[0] + ids[1], 0);
    assert!(ids[2..].iter().all(|&n| n > 30), "{ids:?}");
    // Components are independent: x/1 and y/2 differ.
    assert!(ps.iter().any(|p| (p.position.x - p.position.y / 2.0).abs() > 0.1));
    // Spread over the range.
    let mean = ps.iter().map(|p| p.lifetime).sum::<f32>() / 200.0;
    assert!((mean - 2.0).abs() < 0.15, "{mean}");
}

#[test]
fn random_range_by_uniform_shares_one_draw() {
    let ps = spawn_many(
        50,
        r#"(module: "Initialize Particle", inputs: {
            "Color": Dynamic("Random Range Color by Uniform", { "Minimum": Color((0.2, 0.2, 0.2, 1.0)), "Maximum": Color((0.6, 0.4, 0.2, 1.0)) }),
            "Position": Dynamic("Random Range Vec3 by Uniform", { "Minimum": Vec3((0.0, 0.0, 0.0)), "Maximum": Vec3((1.0, 2.0, 3.0)) }),
        })"#,
    );
    for p in &ps {
        // Grey to brown: on the line, never tinted.
        let u = (p.color.x - 0.2) / 0.4;
        assert_close(p.color.y, 0.2 + 0.2 * u);
        assert_close(p.color.z, 0.2);
        assert_close(p.position.y, 2.0 * p.position.x);
        assert_close(p.position.z, 3.0 * p.position.x);
    }
    assert!(ps.iter().any(|p| p.position.x > 0.5) && ps.iter().any(|p| p.position.x < 0.5));
}

#[test]
fn from_curve_samples_at_any_index() {
    // Default index: normalised age (0 at spawn).
    assert_eq!(float(r#"Dynamic("Float from Curve", { "Curve": Float([(0.0, 2.0), (1.0, 4.0)]) })"#), 2.0);
    let e = r#"Dynamic("Float from Curve", { "Curve": Float([(0.0, 2.0), (1.0, 4.0)]), "Curve Index": Float(0.25) })"#;
    assert_eq!(float(e), 2.5);
    let v = vec3(
        r#"Dynamic("Vec3 from Curve", { "Curve": Vec3([(0.0, (0.0, 0.0, 0.0)), (1.0, (2.0, 4.0, 8.0))]), "Curve Index": Float(0.5) })"#,
    );
    assert_eq!(v, Vec3::new(1.0, 2.0, 4.0));
    let c = color(
        r#"Dynamic("Color from Curve", { "Curve": Color([(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (0.0, 0.0, 0.0, 0.0))]), "Curve Index": Float(0.75) })"#,
    );
    assert_eq!(c, Vec4::splat(0.25));
    let s = vec2(r#"Dynamic("Vec2 from Curve", { "Curve": Vec2([(0.0, (1.0, 2.0))]), "Curve Index": Float(9.0) })"#);
    assert_eq!(s, Vec2::new(1.0, 2.0));
    let d = vec4(r#"Dynamic("Vec4 from Curve", { "Curve": Vec4([(0.0, (0.0, 0.0, 0.0, 0.0)), (2.0, (2.0, 2.0, 2.0, 2.0))]), "Curve Index": Float(1.0) })"#);
    assert_eq!(d, Vec4::ONE);
}

#[test]
fn curve_by_speed_in_update() {
    // Speed 5 of max 10 → index 0.5 → size 1.5 (curve 1..2).
    let mut sim = single_particle(
        "",
        r#"(module: "Add Velocity", inputs: { "Velocity": Vec3((3.0, 4.0, 0.0)) })"#,
        r#"(module: "Scale Sprite Size", inputs: { "Scale Factor": Dynamic("Vec2 from Curve", {
            "Curve": Vec2([(0.0, (1.0, 1.0)), (1.0, (2.0, 2.0))]),
            "Curve Index": Dynamic("Divide Float", {
                "A": Dynamic("Vec3 Length", { "Vector": Link(Vec3, "Particles.Velocity") }),
                "B": Float(10.0),
            }),
        }) })"#,
    );
    sim.step(DT);
    sim.step(DT);
    assert_vec_close(&sim.particles(0)[0].sprite_size.to_array(), &[0.15, 0.15]);
}

#[test]
fn arithmetic() {
    let ab = |name: &str, a: &str, b: &str| format!(r#"Dynamic("{name}", {{ "A": {a}, "B": {b} }})"#);
    assert_eq!(float(&ab("Add Float", "Float(1.5)", "Float(2.0)")), 3.5);
    assert_eq!(float(&ab("Subtract Float", "Float(1.5)", "Float(2.0)")), -0.5);
    assert_eq!(float(&ab("Multiply Float", "Float(1.5)", "Float(2.0)")), 3.0);
    assert_eq!(float(&ab("Divide Float", "Float(3.0)", "Float(2.0)")), 1.5);
    assert_eq!(float(&ab("Divide Float", "Float(3.0)", "Float(0.0)")), 0.0);
    assert_eq!(
        vec3(&ab("Add Vec3", "Vec3((1.0, 2.0, 3.0))", "Vec3((1.0, 1.0, 1.0))")),
        Vec3::new(2.0, 3.0, 4.0)
    );
    assert_eq!(
        vec3(&ab("Multiply Vec3", "Vec3((1.0, 2.0, 3.0))", "Vec3((2.0, 0.5, 0.0))")),
        Vec3::new(2.0, 1.0, 0.0)
    );
    assert_eq!(
        vec3(&ab("Multiply Vec3 by Float", "Vec3((1.0, 2.0, 3.0))", "Float(2.0)")),
        Vec3::new(2.0, 4.0, 6.0)
    );
    assert_eq!(
        vec2(&ab("Subtract Vec2", "Vec2((1.0, 2.0))", "Vec2((0.5, 0.5))")),
        Vec2::new(0.5, 1.5)
    );
    assert_eq!(
        color(&ab("Multiply Color", "Color((1.0, 0.5, 0.25, 1.0))", "Color((2.0, 2.0, 2.0, 0.5))")),
        Vec4::new(2.0, 1.0, 0.5, 0.5)
    );
    assert_eq!(
        color(&ab("Multiply Color by Float", "Color((1.0, 0.5, 0.25, 1.0))", "Float(2.0)")),
        Vec4::new(2.0, 1.0, 0.5, 2.0)
    );
    assert_eq!(
        vec4(&ab("Add Vec4", "Vec4((1.0, 2.0, 3.0, 4.0))", "Vec4((1.0, 1.0, 1.0, 1.0))")),
        Vec4::new(2.0, 3.0, 4.0, 5.0)
    );
    let lerp = r#"Dynamic("Lerp Vec3", { "A": Vec3((0.0, 0.0, 0.0)), "B": Vec3((4.0, 8.0, -4.0)), "Alpha": Float(0.25) })"#;
    assert_eq!(vec3(lerp), Vec3::new(1.0, 2.0, -1.0));
    assert_eq!(float(r#"Dynamic("Lerp Float", { "A": Float(1.0), "B": Float(3.0) })"#), 2.0);
    let clamp = |v: f32| float(&format!(r#"Dynamic("Clamp Float", {{ "Value": Float({v:?}), "Minimum": Float(1.0), "Maximum": Float(2.0) }})"#));
    assert_eq!([clamp(0.0), clamp(1.5), clamp(9.0)], [1.0, 1.5, 2.0]);
}

#[test]
fn make_and_break_vectors() {
    let v = vec3(r#"Dynamic("Make Vec3 from Floats", { "X": Float(1.0), "Y": Float(2.0), "Z": Float(3.0) })"#);
    assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));
    let s = vec2(r#"Dynamic("Make Vec2 from Floats", { "X": Float(0.5), "Y": Float(0.25) })"#);
    assert_eq!(s, Vec2::new(0.5, 0.25));
    let c = color(r#"Dynamic("Make Color from Floats", { "R": Float(1.0), "G": Float(0.5), "B": Float(0.0), "A": Float(0.5) })"#);
    assert_eq!(c, Vec4::new(1.0, 0.5, 0.0, 0.5));
    let c = color(r#"Dynamic("Make Color from Vec3 and Float", { "RGB": Vec3((0.1, 0.2, 0.3)), "Alpha": Float(0.4) })"#);
    assert_eq!(c, Vec4::new(0.1, 0.2, 0.3, 0.4));
    let d = vec4(r#"Dynamic("Make Vec4 from Floats", { "W": Float(7.0) })"#);
    assert_eq!(d, Vec4::new(0.0, 0.0, 0.0, 7.0));
    let b = |ty: &str, v: &str, c: &str| {
        float(&format!(r#"Dynamic("Break {ty}", {{ "Value": {v}, "Component": {c} }})"#))
    };
    assert_eq!(b("Vec3", "Vec3((1.0, 2.0, 3.0))", "Z"), 3.0);
    assert_eq!(b("Vec2", "Vec2((1.0, 2.0))", "Y"), 2.0);
    assert_eq!(b("Color", "Color((1.0, 2.0, 3.0, 4.0))", "A"), 4.0);
    assert_eq!(b("Vec4", "Vec4((1.0, 2.0, 3.0, 4.0))", "X"), 1.0);
}

#[test]
fn vector_math() {
    assert_eq!(vec3(r#"Dynamic("Normalize Vec3", { "Vector": Vec3((0.0, 3.0, 4.0)) })"#), Vec3::new(0.0, 0.6, 0.8));
    assert_eq!(vec3(r#"Dynamic("Normalize Vec3", { "Vector": Vec3((0.0, 0.0, 0.0)) })"#), Vec3::ZERO);
    assert_eq!(float(r#"Dynamic("Vec3 Length", { "Vector": Vec3((0.0, 3.0, 4.0)) })"#), 5.0);
    assert_eq!(
        float(r#"Dynamic("Vec3 Distance", { "A": Vec3((1.0, 1.0, 1.0)), "B": Vec3((1.0, 4.0, 5.0)) })"#),
        5.0
    );
    // A defaults to the particle's position (set before by Initialize Particle).
    let p = spawn_with(
        "",
        r#""Position": Vec3((3.0, 4.0, 0.0))"#,
        r#"(module: "Initialize Ribbon", inputs: { "Ribbon Width": Dynamic("Vec3 Distance", {}) })"#,
    );
    assert_eq!(p.ribbon_width, 5.0);
    let ps = spawn_many(100, r#"(module: "Initialize Particle", inputs: { "Position": Dynamic("Random Unit Vec3") })"#);
    for p in &ps {
        assert_close(p.position.length(), 1.0);
    }
    let mean = ps.iter().map(|p| p.position).sum::<Vec3>() / 100.0;
    assert!(mean.length() < 0.25, "{mean}");
}

#[test]
fn sine_and_cosine_over_time() {
    // At t = Engine.Time = 0 (first step): sin(2π·phase), cos(2π·phase).
    let s = float(r#"Dynamic("Sine over Time", { "Amplitude": Float(2.0), "Phase": Float(0.25), "Offset": Float(3.0) })"#);
    assert_close(s, 5.0);
    let c = float(r#"Dynamic("Cosine over Time", { "Amplitude": Float(2.0), "Offset": Float(3.0) })"#);
    assert_close(c, 5.0);
    // Time from a link: 0.25 s at 1 Hz is a quarter cycle.
    let s = float(r#"Dynamic("Sine over Time", { "Time": Float(0.25), "Offset": Float(3.0) })"#);
    assert_close(s, 4.0);
    // Over Engine.Time in an emitter stage: Spawn Rate 100 + 100 sin(2π·t), dt 0.25.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Sine over Time", { "Amplitude": Float(100.0), "Offset": Float(100.0) }) })"#,
        "",
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    let mut spawned = Vec::new();
    for _ in 0..4 {
        sim.step(DT);
        spawned.push(sim.emitters[0].spawned_last_step);
    }
    // Rates 100, 200, 100, 0 → 25, 50, 25, 0 (within rounding of sin).
    assert!(spawned[0] == 25 && spawned[1] == 50 && (24..=25).contains(&spawned[2]) && spawned[3] <= 1, "{spawned:?}");
}

#[test]
fn select_by_bool_and_int() {
    assert_eq!(float(r#"Dynamic("Select Float by Bool", { "Condition": Bool(true), "If True": Float(1.0), "If False": Float(2.0) })"#), 1.0);
    assert_eq!(float(r#"Dynamic("Select Float by Bool", { "Condition": Bool(false), "If True": Float(1.0), "If False": Float(2.0) })"#), 2.0);
    let sel = |i: i32| {
        vec3(&format!(
            r#"Dynamic("Select Vec3 by Int", {{ "Index": Int({i}), "Values": [Vec3((1.0, 0.0, 0.0)), Vec3((0.0, 2.0, 0.0)), Vec3((0.0, 0.0, 3.0))] }})"#
        ))
    };
    assert_eq!(sel(1), Vec3::new(0.0, 2.0, 0.0));
    assert_eq!(sel(-5), Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(sel(9), Vec3::new(0.0, 0.0, 3.0));
    assert_eq!(int(r#"Dynamic("Select Int by Bool", { "Condition": Bool(true), "If True": Int(7) })"#), 7);
    assert_eq!(int(r#"Dynamic("Select Int by Int", { "Index": Int(1), "Values": [Int(3), Int(4)] })"#), 4);
}

#[test]
fn compare_floats() {
    let cmp = |a: f32, op: &str, b: f32| {
        bool_(&format!(r#"Dynamic("Compare Floats", {{ "A": Float({a:?}), "B": Float({b:?}), "Operation": {op} }})"#))
    };
    assert!(cmp(1.0, "LessThan", 2.0) && !cmp(2.0, "LessThan", 2.0));
    assert!(cmp(2.0, "LessOrEqual", 2.0) && !cmp(3.0, "LessOrEqual", 2.0));
    assert!(cmp(3.0, "GreaterThan", 2.0) && !cmp(2.0, "GreaterThan", 2.0));
    assert!(cmp(2.0, "GreaterOrEqual", 2.0) && !cmp(1.0, "GreaterOrEqual", 2.0));
    assert!(cmp(2.0, "Equal", 2.0) && !cmp(1.0, "Equal", 2.0));
    assert!(cmp(1.0, "NotEqual", 2.0) && !cmp(2.0, "NotEqual", 2.0));
}

#[test]
fn random_from_list_picks_every_value() {
    let ps = spawn_many(
        300,
        r#"(module: "Initialize Particle", inputs: {
            "Lifetime": Dynamic("Random Float from List", { "Values": [Float(1.0), Float(2.0), Float(3.0)] }),
            "Color": Dynamic("Random Color from List", { "Values": [Color((1.0, 0.0, 0.0, 1.0)), Color((0.0, 0.0, 1.0, 1.0))] }),
        })"#,
    );
    let mut n = [0; 3];
    for p in &ps {
        n[p.lifetime as usize - 1] += 1;
        assert!(p.color == Vec4::new(1.0, 0.0, 0.0, 1.0) || p.color == Vec4::new(0.0, 0.0, 1.0, 1.0));
    }
    assert!(n.iter().all(|&k| k > 70), "{n:?}");
    // The two lists draw independently.
    assert!(ps.iter().any(|p| p.lifetime == 1.0 && p.color.x == 1.0));
    assert!(ps.iter().any(|p| p.lifetime == 1.0 && p.color.z == 1.0));
}

#[test]
fn normalized_ages() {
    // The burst particle spawns in step one, at emitter and system age 0.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Float from Curve", {
            "Curve": Float([(0.0, 0.0), (1.0, 400.0)]),
            "Curve Index": Dynamic("Normalized Emitter Age", { "Duration": Float(1.0) }),
        }) })"#,
        "",
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    let mut spawned = Vec::new();
    for _ in 0..6 {
        sim.step(DT);
        spawned.push(sim.emitters[0].spawned_last_step);
    }
    // Rate 400·age for ages 0, .25, .5, .75, then clamped at 1.
    assert_eq!(spawned, [0, 25, 50, 75, 100, 100]);
    let x = float(r#"Dynamic("Normalized System Age", { "Duration": Float(0.0) })"#);
    assert_eq!(x, 1.0);
    let x = float(r#"Dynamic("Normalized System Age", { "Age": Float(3.0), "Duration": Float(4.0) })"#);
    assert_eq!(x, 0.75);
}

/// Nesting: a curve indexed by a division of a length, lerping towards a
/// random-by-uniform colour, inside Initialize Particle; plus a user link.
const NESTED: &str = r#"(module: "Initialize Particle", inputs: {
    "Color": Dynamic("Lerp Color", {
        "A": Link(Color, "User.Tint"),
        "B": Dynamic("Random Range Color by Uniform", {
            "Minimum": Color((0.0, 0.0, 0.0, 1.0)),
            "Maximum": Dynamic("Make Color from Vec3 and Float", {
                "RGB": Dynamic("Multiply Vec3 by Float", { "A": Vec3((1.0, 1.0, 1.0)), "B": RandomRange(Float(0.5), Float(1.0)) }),
                "Alpha": Float(1.0),
            }),
        }),
        "Alpha": Dynamic("Float from Curve", {
            "Curve": Float([(0.0, 0.0), (1.0, 1.0)]),
            "Curve Index": Dynamic("Divide Float", {
                "A": Dynamic("Vec3 Length", { "Vector": Link(Vec3, "Particles.Position") }),
                "B": Float(10.0),
            }),
        }),
    }),
    "Position": Dynamic("Multiply Vec3 by Float", {
        "A": Dynamic("Random Unit Vec3"),
        "B": Dynamic("Uniform Ranged Float", { "Minimum": Float(2.0), "Maximum": Float(8.0) }),
    }),
})"#;

#[test]
fn nested_dynamic_inputs_validate_cook_and_simulate() {
    let src = one_emitter(
        r#""User.Tint": Color((1.0, 0.0, 0.0, 1.0))"#,
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(50) })"#,
        NESTED,
        "",
    );
    let c = fx_core::check_source("nested.fx.ron", &src);
    assert!(c.diagnostics.iter().all(|d| !d.is_error()), "{:?}", c.diagnostics);
    let sys = c.cooked.unwrap();
    // Two top-level dynamic inputs with 7 + 3 nodes.
    assert_eq!(sys.emitters[0].particle_spawn[0].dynamics.len(), 10);
    // Survives the cooked binary.
    assert_eq!(cook::from_bytes(&cook::to_bytes(&sys)).unwrap(), sys);

    let mut sim = Simulation::new(sys, 3);
    sim.step(DT);
    let ps = sim.particles(0);
    assert_eq!(ps.len(), 50);
    for p in ps {
        // Position: unit vector times 2..8. Inputs run in slot order, so Color
        // (slot 2) reads the Position (slot 1) just set.
        let r = p.position.length();
        assert!((2.0..8.0).contains(&(r + 1e-5)) && r < 8.0 + 1e-5, "{r}");
        let alpha = r / 10.0;
        // B is on the grey line from black to (s, s, s), s in 0.5..1.
        let b = (p.color - Vec4::new(1.0, 0.0, 0.0, 1.0) * (1.0 - alpha)) / alpha;
        assert!((b.x - b.y).abs() < 1e-4 && (b.y - b.z).abs() < 1e-4, "{b}");
        assert!(b.x >= -1e-4 && b.x <= 1.0 + 1e-4, "{b}");
        assert!((p.color.w - 1.0).abs() < 1e-5);
    }
    // Deterministic: the same seed gives the same particles.
    let again = {
        let mut s = Simulation::new(compile(&src), 3);
        s.step(DT);
        s.state_hash()
    };
    assert_eq!(again, sim.state_hash());
}

#[test]
fn nested_dynamic_inputs_in_update_and_emitter_stages() {
    // Update: Scale Color alpha from a curve over the particle's own sine.
    let src = one_emitter(
        "",
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Dynamic("Select Int by Bool", {
            "Condition": Dynamic("Compare Floats", { "A": Link(Float, "Emitter.Age"), "B": Float(0.1), "Operation": LessThan }),
            "If True": Dynamic("Uniform Ranged Int", { "Minimum": Int(3), "Maximum": Int(3) }),
            "If False": Int(0),
        }) })"#,
        "",
        r#"(module: "Particle State"),
        (module: "Scale Color", inputs: { "Scale Alpha": Dynamic("Float from Curve", {
            "Curve": Float([(0.0, 0.0), (1.0, 1.0)]),
            "Curve Index": Dynamic("Sine over Time", { "Time": Link(Float, "Particles.Age"), "Frequency": Float(0.5) }),
        }) })"#,
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(0.5);
    sim.step(0.5);
    assert_eq!(sim.particle_count(), 3);
    // Age 0.5 s at 0.5 Hz: sin(π/2) = 1.
    for p in sim.particles(0) {
        assert_close(p.color.w, 1.0);
    }
}

#[test]
fn every_dynamic_input_is_covered_by_a_test() {
    let me = [include_str!("dynamic_inputs.rs"), include_str!("data_interfaces.rs")].concat();
    for d in DYNAMIC_INPUTS.iter() {
        let n = &d.name;
        let covered = me.contains(&format!("\"{n}\""))
            || me.contains(&n.replace(d.ty.name(), "{ty}"))
            || n.starts_with("Break ")
            || (n.starts_with("Uniform Ranged ") || n.starts_with("Random Range "))
            || ["Add ", "Subtract ", "Multiply ", "Lerp ", "Select ", "Random ", "Make "]
                .iter()
                .any(|p| n.starts_with(p))
            || n.ends_with(" from Curve");
        assert!(covered, "no test names \"{n}\"");
    }
}

#[test]
fn generic_dynamic_inputs_agree_across_types() {
    // Families written once per type: run each type once on a simple case.
    for t in ["Float", "Vec2", "Vec3", "Vec4", "Color"] {
        let lit = |x: f32| match t {
            "Float" => format!("Float({x:?})"),
            "Vec2" => format!("Vec2(({x:?}, {x:?}))"),
            "Vec3" => format!("Vec3(({x:?}, {x:?}, {x:?}))"),
            _ => format!("{t}(({x:?}, {x:?}, {x:?}, {x:?}))"),
        };
        let first = |e: String| -> f32 {
            match t {
                "Float" => float(&e),
                "Vec2" => vec2(&e).x,
                "Vec3" => vec3(&e).x,
                "Vec4" => vec4(&e).x,
                _ => color(&e).x,
            }
        };
        let ab = |op: &str| format!(r#"Dynamic("{op} {t}", {{ "A": {}, "B": {} }})"#, lit(3.0), lit(2.0));
        assert_eq!(first(ab("Add")), 5.0, "{t}");
        assert_eq!(first(ab("Subtract")), 1.0, "{t}");
        assert_eq!(first(ab("Multiply")), 6.0, "{t}");
        assert_eq!(first(format!(r#"Dynamic("Lerp {t}", {{ "A": {}, "B": {}, "Alpha": Float(0.5) }})"#, lit(3.0), lit(2.0))), 2.5, "{t}");
        assert_eq!(first(format!(r#"Dynamic("Select {t} by Bool", {{ "Condition": Bool(false), "If False": {} }})"#, lit(4.0))), 4.0, "{t}");
        assert_eq!(first(format!(r#"Dynamic("Select {t} by Int", {{ "Index": Int(1), "Values": [{}, {}] }})"#, lit(4.0), lit(5.0))), 5.0, "{t}");
        assert_eq!(first(format!(r#"Dynamic("Random {t} from List", {{ "Values": [{}] }})"#, lit(4.0))), 4.0, "{t}");
        let x = first(format!(r#"Dynamic("Uniform Ranged {t}", {{ "Minimum": {}, "Maximum": {} }})"#, lit(4.0), lit(5.0)));
        assert!((4.0..5.0).contains(&x), "{t}");
        if t != "Float" {
            let x = first(format!(r#"Dynamic("Random Range {t} by Uniform", {{ "Minimum": {}, "Maximum": {} }})"#, lit(4.0), lit(5.0)));
            assert!((4.0..5.0).contains(&x), "{t}");
            assert_eq!(first(format!(r#"Dynamic("Multiply {t} by Float", {{ "A": {}, "B": Float(2.0) }})"#, lit(4.0))), 8.0, "{t}");
        }
    }
}

#[test]
fn errors_name_the_line_and_suggest_fixes() {
    // Unknown name.
    let e = spawn_error(r#""Lifetime": Dynamic("Uniform Range Float", { })"#);
    assert!(e.contains("unknown dynamic input \"Uniform Range Float\""), "{e}");
    assert!(e.contains("did you mean \"Uniform Ranged Float\"?"), "{e}");
    // Wrong result type, with the same op suggested.
    let e = spawn_error(r#""Lifetime": Dynamic("Multiply Vec3", { })"#);
    assert!(e.contains("input \"Lifetime\" of \"Initialize Particle\" is Float, but dynamic input \"Multiply Vec3\" gives a Vec3"), "{e}");
    assert!(e.contains("use \"Multiply Float\""), "{e}");
    // A nested input of the wrong type, at its own line.
    let e = spawn_error(
        "\"Position\": Dynamic(\"Multiply Vec3 by Float\", {\n    \"A\": Vec3((1.0, 0.0, 0.0)),\n    \"B\": Vec3((1.0, 0.0, 0.0)),\n})",
    );
    assert!(e.contains("test.fx.ron:10:5"), "{e}");
    assert!(e.contains("input \"B\" of \"Multiply Vec3 by Float\" is Float, but the value given is Vec3"), "{e}");
    // Unknown input of a dynamic input.
    let e = spawn_error(r#""Lifetime": Dynamic("Add Float", { "C": Float(1.0) })"#);
    assert!(e.contains("dynamic input \"Add Float\" has no input \"C\""), "{e}");
    assert!(e.contains("\"A\" (Float), \"B\" (Float)"), "{e}");
    // Missing required curve; wrong curve type; bare curve elsewhere.
    let e = spawn_error(r#""Lifetime": Dynamic("Float from Curve", { })"#);
    assert!(e.contains("needs its input \"Curve\" (Float curve)"), "{e}");
    let e = spawn_error(r#""Lifetime": Dynamic("Float from Curve", { "Curve": Vec2([(0.0, (1.0, 1.0))]) })"#);
    assert!(e.contains("is a Float curve, but a Vec2 curve is given"), "{e}");
    let e = spawn_error(r#""Lifetime": Float([(0.0, 1.0)])"#);
    assert!(e.contains("bare curve") && e.contains("CurveOverAge(Float(...))"), "{e}");
    // Lists.
    let e = spawn_error(r#""Lifetime": Dynamic("Random Float from List", { "Values": [] })"#);
    assert!(e.contains("is an empty list"), "{e}");
    let e = spawn_error(r#""Lifetime": Dynamic("Random Float from List", { "Values": [Float(1.0), Int(2)] })"#);
    assert!(e.contains("input \"Values[1]\" of \"Random Float from List\" is Float, but the value given is Int"), "{e}");
    let e = spawn_error(r#""Lifetime": Dynamic("Random Float from List", { "Values": Float(1.0) })"#);
    assert!(e.contains("takes a list of Float"), "{e}");
    let e = spawn_error(r#""Lifetime": [Float(1.0)]"#);
    assert!(e.contains("not a list"), "{e}");
    // Static switch inside a dynamic input.
    let e = spawn_error(r#""Lifetime": Dynamic("Break Vec3", { "Value": Vec3((1.0, 2.0, 3.0)), "Component": W })"#);
    assert!(e.contains("\"Component\" of \"Break Vec3\" has no choice `W`"), "{e}");
    // A default reading Particles. in an emitter stage.
    let e = error(&one_emitter(
        "",
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Float from Curve", { "Curve": Float([(0.0, 1.0)]) }) })"#,
        "",
        "",
    ));
    assert!(e.contains("reads Particles.NormalizedAge by default, which `emitter_update` does not have"), "{e}");
    // An explicit Particles. link nested in an emitter stage.
    let e = error(&one_emitter(
        "",
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Add Float", { "A": Link(Float, "Particles.Age") }) })"#,
        "",
        "",
    ));
    assert!(e.contains("`emitter_update` cannot read Particles. parameters"), "{e}");
    // Malformed Dynamic.
    let e = spawn_error(r#""Lifetime": Dynamic(Float(1.0))"#);
    assert!(e.contains("must be `Dynamic(\"Name\", { inputs })`"), "{e}");
}

#[test]
fn source_round_trips_through_ron() {
    // The model serialises back to the source form.
    let src = one_emitter("", r#"(module: "Spawn Rate")"#, NESTED.replace("User.Tint", "Particles.Color").replace("Link(Color, \"Particles.Color\")", "Color((1.0, 0.0, 0.0, 1.0))").as_str(), "");
    let c = fx_core::check_source("a.fx.ron", &src);
    let model = c.source.unwrap();
    let text = ron::ser::to_string(&model).unwrap();
    assert!(text.contains("Dynamic(\"Lerp Color\""), "{text}");
    let again = fx_core::check_source("b.fx.ron", &text);
    assert_eq!(again.cooked.unwrap(), c.cooked.unwrap());
}
