#![allow(dead_code)]

use std::path::PathBuf;

use fx_core::cook::CookedSystem;
use fx_core::sim::Simulation;

pub fn effects_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../effects")
}

/// Compiles effect source, panicking with the diagnostics on errors.
pub fn compile(src: &str) -> CookedSystem {
    let c = fx_core::check_source("test.fx.ron", src);
    let msgs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked
        .unwrap_or_else(|| panic!("effect failed to compile:\n{}\n---\n{src}", msgs.join("\n")))
}

/// One-emitter effect from its stacks (RON list bodies, without brackets).
pub fn one_emitter(
    user: &str,
    emitter_update: &str,
    particle_spawn: &str,
    particle_update: &str,
) -> String {
    format!(
        "(
    name: \"Test\",
    user_parameters: {{ {user} }},
    emitters: [(
        name: \"E\",
        max_particles: 1000,
        emitter_update: [ {emitter_update} ],
        particle_spawn: [ {particle_spawn} ],
        particle_update: [ {particle_update} ],
    )],
)"
    )
}

/// A sim whose emitter spawns exactly one particle on the first step.
pub fn single_particle(user: &str, particle_spawn: &str, particle_update: &str) -> Simulation {
    let src = one_emitter(
        user,
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(1) })"#,
        particle_spawn,
        particle_update,
    );
    Simulation::new(compile(&src), 1)
}

pub fn assert_close(a: f32, b: f32) {
    assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} != {b}");
}

pub fn assert_vec_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-5 * (1.0 + y.abs()), "{a:?} != {b:?}");
    }
}
