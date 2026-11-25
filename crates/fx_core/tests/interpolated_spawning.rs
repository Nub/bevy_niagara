//! Interpolated spawning (Niagara's emitter option, set by every imported
//! Niagara emitter that has it): particles spawned in a step run Particle
//! Update in that step, so none is drawn with its spawn values only. Without
//! it a new particle shows one frame of its spawn colour and size before its
//! update curves apply: the flashing and snapping of the imported
//! NS_FireSmokeOnly (white, full-alpha sprites) and NS_Fish (full-size fish).

mod common;

use std::collections::HashMap;

use common::*;
use fx_core::sim::{Particle, Simulation, spawn_step_dt};

const DT: f32 = 1.0 / 30.0;

fn effect(interpolated: bool) -> String {
    format!(
        r#"(
    name: "Test",
    emitters: [(
        name: "E",
        max_particles: 1000,
        interpolated_spawning: {interpolated},
        emitter_update: [(module: "Spawn Rate", inputs: {{ "Spawn Rate": Float(100.0) }})],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": RandomRange(Float(1.0), Float(2.0)), "Color": Color((1.0, 1.0, 1.0, 1.0)), "Sprite Size": Vec2((0.5, 0.5)) }}),
            (module: "Initialize Mesh", inputs: {{ "Mesh Scale": Vec3((2.0, 2.0, 2.0)) }}),
            (module: "Add Velocity", inputs: {{ "Velocity": Vec3((0.0, 1.0, 0.0)) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Scale Color", inputs: {{ "Scale Alpha": CurveOverAge(Float([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])) }}),
            (module: "Scale Sprite Size", inputs: {{ "Scale Factor": CurveOverAge(Vec2([(0.0, (0.0, 0.0)), (1.0, (1.0, 1.0))])) }}),
            (module: "Scale Mesh Size", inputs: {{ "Scale Factor": CurveOverAge(Vec3([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])) }}),
            (module: "Solve Forces and Velocity"),
        ],
    )],
)"#
    )
}

/// The largest change of alpha, sprite size and mesh scale of one particle
/// between consecutive frames, and of a particle's first frame from 0 (what
/// the update curves start at), over 3 s.
fn largest_jumps(interpolated: bool) -> (f32, f32) {
    let mut sim = Simulation::new(compile(&effect(interpolated)), 5);
    let mut prev: HashMap<u32, Particle> = HashMap::new();
    let (mut between, mut first) = (0.0f32, 0.0f32);
    for _ in 0..90 {
        sim.step(DT);
        let mut cur = HashMap::new();
        for p in sim.particles(0) {
            let v = [p.color.w, p.sprite_size.x, p.scale.x];
            match prev.get(&p.unique_id) {
                Some(q) => {
                    let u = [q.color.w, q.sprite_size.x, q.scale.x];
                    for k in 0..3 {
                        between = between.max((v[k] - u[k]).abs());
                    }
                }
                None => first = first.max(v.into_iter().fold(0.0, f32::max)),
            }
            cur.insert(p.unique_id, *p);
        }
        prev = cur;
    }
    (between, first)
}

#[test]
fn spawned_particles_never_show_spawn_values() {
    // Without interpolated spawning, a new particle is drawn once with
    // alpha 1, size 0.5 and scale 2 before the curves bring them to ~0.
    let (between, first) = largest_jumps(false);
    assert!(first >= 1.0 && between >= 1.0, "{first} {between}");
    // With it, its first frame already went through the update curves, and
    // nothing jumps between frames (curves move < 0.1 per 1/30 s here).
    let (between, first) = largest_jumps(true);
    assert!(first < 0.1, "first frame {first}");
    assert!(between < 0.1, "between frames {between}");
}

#[test]
fn spawns_spread_over_the_step() {
    let mut sim = Simulation::new(compile(&effect(true)), 5);
    sim.step(DT);
    let ps = sim.particles(0);
    let n = ps.len() as u32;
    assert_eq!(n, 3, "100/s over 1/30 s");
    // The k-th spawn lived (n - k) / n of the step: ages DT, 2/3 DT, 1/3 DT,
    // and moved that far at 1 m/s.
    for (k, p) in ps.iter().enumerate() {
        let t = spawn_step_dt(DT, k as u32, n);
        assert_close(p.age, t);
        assert_close(p.position.y, t);
    }
    assert_close(spawn_step_dt(DT, 0, n), DT);
    // A particle whose lifetime ends inside its spawn step is never drawn.
    let src = effect(true).replace("RandomRange(Float(1.0), Float(2.0))", "Float(0.001)");
    let mut sim = Simulation::new(compile(&src), 5);
    sim.step(DT);
    assert_eq!(sim.particles(0).len(), 0);
}
