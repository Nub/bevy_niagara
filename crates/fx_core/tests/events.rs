//! Collision response and event delivery on the CPU reference sim, against
//! values worked out by hand (dt = 0.25 so most values are exact).

mod common;

use bevy_math::Vec3;
use common::*;
use fx_core::events::EventKind;
use fx_core::sim::{Collider, Simulation};

const DT: f32 = 0.25;

/// One particle at `pos` moving at `vel`, updated by Solve then `collision`.
fn falling(pos: &str, vel: &str, collision: &str, extra_update: &str) -> String {
    format!(
        r#"(
    name: "Test",
    emitters: [(
        name: "E",
        max_particles: 10,
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int(1) }})],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": Float(100.0), "Position": Vec3({pos}) }}),
            (module: "Add Velocity", inputs: {{ "Velocity": Vec3({vel}) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Solve Forces and Velocity"),
            (module: "Collision", inputs: {{ {collision} }}),
            (module: "Generate Collision Event"),
            {extra_update}
        ],
    ), (
        name: "Hits",
        max_particles: 100,
        event_handlers: [(source: "E", event: Collision, spawn_number: 1, modules: [(module: "Receive Collision Event")])],
        particle_update: [(module: "Particle State")],
    )],
)"#
    )
}

fn one(sim: &Simulation) -> fx_core::sim::Particle {
    assert_eq!(sim.particles(0).len(), 1);
    sim.particles(0)[0]
}

#[test]
fn bounce_with_restitution_and_friction() {
    // Step 1 moves (0, 0.5, 0) by (2, -4, 0) * 0.25 to (0.5, -0.5, 0): back onto
    // y = 0; normal speed 4 → 4 * 0.5 = 2 out; sliding 2 loses 0.25 * 4 = 1.
    let src = falling(
        "(0.0, 0.5, 0.0)",
        "(2.0, -4.0, 0.0)",
        r#""Restitution": Float(0.5), "Friction": Float(0.25)"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(DT);
    sim.step(DT);
    let p = one(&sim);
    assert_eq!(p.position, Vec3::new(0.5, 0.0, 0.0));
    assert_eq!(p.velocity, Vec3::new(1.0, 2.0, 0.0));
    let ev = sim.events(0, EventKind::Collision);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].normal, [0.0, 1.0, 0.0]);
    assert_eq!(ev[0].position, [0.5, 0.0, 0.0]);
    assert_eq!(ev[0].velocity, [1.0, 2.0, 0.0]);
    // The handler spawned one particle at the hit.
    assert_eq!(sim.particles(1).len(), 1);
    assert_eq!(sim.particles(1)[0].position, Vec3::new(0.5, 0.0, 0.0));
    // Moving away: no more collisions.
    sim.step(DT);
    assert!(sim.events(0, EventKind::Collision).is_empty());
}

#[test]
fn particle_radius_and_tilted_plane() {
    // Plane x = 1 facing -X, radius 0.25: a particle reaching x = 1.5 is put at 0.75.
    let src = falling(
        "(0.5, 0.0, 0.0)",
        "(4.0, 0.0, 0.0)",
        r#""Plane Normal": Vec3((-2.0, 0.0, 0.0)), "Plane Origin": Vec3((1.0, 5.0, 0.0)), "Particle Radius": Float(0.25), "Restitution": Float(1.0)"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(DT);
    sim.step(DT);
    let p = one(&sim);
    assert_eq!(p.position, Vec3::new(0.75, 0.0, 0.0));
    assert_eq!(p.velocity, Vec3::new(-4.0, 0.0, 0.0));
}

#[test]
fn stop_and_kill_responses() {
    let stop = falling(
        "(0.0, 0.5, 0.0)",
        "(2.0, -4.0, 0.0)",
        r#""Collision Response": Stop"#,
        "",
    );
    let mut sim = Simulation::new(compile(&stop), 0);
    for _ in 0..4 {
        sim.step(DT);
    }
    let p = one(&sim);
    assert_eq!(p.position, Vec3::new(0.5, 0.0, 0.0));
    assert_eq!(p.velocity, Vec3::ZERO);

    let kill = falling(
        "(0.0, 0.5, 0.0)",
        "(2.0, -4.0, 0.0)",
        r#""Collision Response": Kill"#,
        r#"(module: "Generate Death Event"),"#,
    );
    let src = kill.replace(
        "event_handlers: [(source: \"E\", event: Collision",
        "event_handlers: [(source: \"E\", event: Death, spawn_number: 2), (source: \"E\", event: Collision",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(DT);
    sim.step(DT);
    assert!(sim.particles(0).is_empty());
    assert_eq!(sim.events(0, EventKind::Death).len(), 1);
    assert_eq!(sim.events(0, EventKind::Collision).len(), 1);
    // Two from the death, one from the collision.
    assert_eq!(sim.particles(1).len(), 3);
}

#[test]
fn slow_hits_rest_without_events() {
    // 0.2 m/s into the floor with Rest Speed 0.5: no bounce, no event, sliding kept.
    let src = falling(
        "(0.0, 0.01, 0.0)",
        "(1.0, -0.2, 0.0)",
        r#""Rest Speed": Float(0.5), "Restitution": Float(1.0)"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(DT);
    sim.step(DT);
    let p = one(&sim);
    assert_eq!(p.position.y, 0.0);
    assert_eq!(p.velocity, Vec3::new(1.0, 0.0, 0.0));
    assert!(sim.events(0, EventKind::Collision).is_empty());
    assert!(sim.particles(1).is_empty());
}

#[test]
fn cpu_scene_collisions_use_the_host_colliders() {
    let src = falling(
        "(0.0, 3.0, 0.0)",
        "(0.0, -4.0, 0.0)",
        r#""Collision Type": SceneDepth, "Restitution": Float(1.0)"#,
        "",
    );
    // No colliders: falls through.
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(DT);
    sim.step(DT);
    assert_eq!(one(&sim).position, Vec3::new(0.0, 2.0, 0.0));
    // A unit sphere at y = 1.5: the step to y = 2 ends inside it, pushed out to its top at y = 2.5.
    let mut sim = Simulation::new(compile(&src), 0);
    sim.colliders = vec![Collider::Sphere {
        center: Vec3::new(0.0, 1.5, 0.0),
        radius: 1.0,
    }];
    sim.step(DT);
    sim.step(DT);
    let p = one(&sim);
    assert_eq!(p.position, Vec3::new(0.0, 2.5, 0.0));
    assert_eq!(p.velocity, Vec3::new(0.0, 4.0, 0.0));
    // A box: out through the nearest face.
    let mut sim = Simulation::new(compile(&src), 0);
    sim.colliders = vec![Collider::Box {
        center: Vec3::new(0.0, 0.0, 0.0),
        half_extents: Vec3::new(1.0, 1.5, 1.0),
    }];
    for _ in 0..3 {
        sim.step(DT);
    }
    assert_eq!(one(&sim).position, Vec3::new(0.0, 1.5, 0.0));
}

/// A "Shell" emitter of `n` particles dying at the fourth step, and a
/// receiving "Burst" emitter, `burst_first` putting it before the source.
fn fireworks(n: u32, handler: &str, burst_first: bool) -> String {
    let shell = format!(
        r#"(
        name: "Shell",
        max_particles: 100,
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int({n}) }})],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": Float(0.6) }}),
            (module: "Add Velocity", inputs: {{ "Velocity": RandomRange(Vec3((-1.0, 2.0, -1.0)), Vec3((1.0, 4.0, 1.0))) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Generate Death Event"),
            (module: "Generate Location Event"),
            (module: "Solve Forces and Velocity"),
        ],
    )"#
    );
    let burst = format!(
        r#"(
        name: "Burst",
        max_particles: 40,
        event_handlers: [{handler}],
        particle_spawn: [(module: "Initialize Particle", inputs: {{ "Lifetime": Float(10.0) }})],
        particle_update: [(module: "Particle State")],
    )"#
    );
    let (a, b) = if burst_first { (&burst, &shell) } else { (&shell, &burst) };
    format!("(name: \"Fireworks\", emitters: [{a}, {b}])")
}

#[test]
fn death_events_spawn_particles_at_the_shells() {
    let src = fireworks(
        5,
        r#"(source: "Shell", event: Death, spawn_number: 3, modules: [(module: "Receive Death Event", inputs: { "Position Offset": Vec3((0.0, 1.0, 0.0)) })])"#,
        false,
    );
    let mut sim = Simulation::new(compile(&src), 7);
    for _ in 0..3 {
        sim.step(DT);
        assert!(sim.particles(1).is_empty());
    }
    let shells = sim.particles(0).to_vec();
    sim.step(DT);
    assert!(sim.particles(0).is_empty());
    let deaths = sim.events(0, EventKind::Death);
    assert_eq!(deaths.len(), 5);
    let ids: Vec<u32> = deaths.iter().map(|e| e.particle_id).collect();
    assert_eq!(ids, [0, 1, 2, 3, 4]);
    let burst = sim.particles(1);
    assert_eq!(burst.len(), 15);
    assert_eq!(sim.emitters[1].spawned_last_step, 15);
    for (k, p) in burst.iter().enumerate() {
        let shell = &shells[k / 3];
        // The shell moved once more in the step it died.
        let at = shell.position + shell.velocity * DT;
        assert_eq!(p.position, at + Vec3::Y, "burst particle {k}");
        assert_eq!(p.unique_id, k as u32);
    }
}

#[test]
fn a_receiver_before_its_source_gets_the_events_one_step_later() {
    let src = fireworks(5, r#"(source: "Shell", event: Death, spawn_number: 2)"#, true);
    let mut sim = Simulation::new(compile(&src), 7);
    for _ in 0..4 {
        sim.step(DT);
    }
    assert!(sim.particles(0).is_empty(), "the burst is emitter 0");
    assert_eq!(sim.events(1, EventKind::Death).len(), 5);
    sim.step(DT);
    assert_eq!(sim.particles(0).len(), 10);
}

#[test]
fn max_events_and_capacity_cap_the_spawns() {
    // Five deaths, at most two handled, lowest UniqueIDs first.
    let src = fireworks(
        5,
        r#"(source: "Shell", event: Death, spawn_number: 3, max_events_per_frame: 2, modules: [(module: "Receive Death Event")])"#,
        false,
    );
    let mut sim = Simulation::new(compile(&src), 7);
    let mut shells = Vec::new();
    for _ in 0..4 {
        shells = sim.particles(0).to_vec();
        sim.step(DT);
    }
    assert_eq!(sim.particles(1).len(), 6);
    let at = |i: usize| shells[i].position + shells[i].velocity * DT;
    assert_eq!(sim.particles(1)[5].position, at(1));
    // 20 deaths * 3 = 60 asked for, 40 room: the first 13 events fill it, one particle of the 14th.
    let src = fireworks(20, r#"(source: "Shell", event: Death, spawn_number: 3)"#, false);
    let mut sim = Simulation::new(compile(&src), 7);
    for _ in 0..4 {
        sim.step(DT);
    }
    assert_eq!(sim.particles(1).len(), 40);
}

#[test]
fn location_events_every_step() {
    let src = fireworks(
        4,
        r#"(source: "Shell", event: Location, spawn_number: 1, modules: [(module: "Receive Location Event", inputs: { "Inherit Velocity Scale": Float(0.5) })])"#,
        false,
    );
    let mut sim = Simulation::new(compile(&src), 3);
    sim.step(DT);
    assert!(sim.events(0, EventKind::Location).is_empty(), "spawned, not yet updated");
    for step in 1..=2 {
        sim.step(DT);
        assert_eq!(sim.events(0, EventKind::Location).len(), 4);
        assert_eq!(sim.particles(1).len(), 4 * step);
    }
    let shell = sim.particles(0)[3];
    let trail = sim.particles(1).last().unwrap();
    assert_eq!(trail.velocity, shell.velocity * 0.5);
}

#[test]
fn every_particle_handlers_run_once_per_event() {
    // Each receiver particle gains 1 m/s of X velocity per handled death event.
    let src = r#"(
    name: "T",
    emitters: [(
        name: "Ticks",
        max_particles: 10,
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(3) })],
        particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(0.3) })],
        particle_update: [(module: "Particle State"), (module: "Generate Death Event")],
    ), (
        name: "Listeners",
        max_particles: 10,
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(2) })],
        particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(10.0) })],
        particle_update: [(module: "Particle State")],
        event_handlers: [(source: "Ticks", event: Death, execution_mode: EveryParticle, max_events_per_frame: 2, modules: [
            (module: "Add Velocity", inputs: { "Velocity": Vec3((1.0, 0.0, 0.0)) }),
            (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Event.Position") }),
        ])],
    )],
)"#;
    let mut sim = Simulation::new(compile(src), 0);
    for _ in 0..3 {
        sim.step(DT);
    }
    // Ticks died in step 2 (age 0.5 > 0.3); Listeners follow Ticks, so they saw it then.
    assert_eq!(sim.events(0, EventKind::Death).len(), 3);
    for p in sim.particles(1) {
        assert_eq!(p.velocity, Vec3::new(2.0, 0.0, 0.0));
    }
}

#[test]
fn event_errors() {
    let check = |src: &str| {
        let c = fx_core::check_source("t.fx.ron", src);
        c.errors().map(|d| d.to_string()).collect::<Vec<_>>().join("\n")
    };
    let errs = check(&fireworks(1, r#"(source: "Shel", event: Death)"#, false));
    assert!(errs.contains("is not an emitter") && errs.contains("did you mean \"Shell\""), "{errs}");
    let errs = check(&fireworks(1, r#"(source: "Shell", event: Collision)"#, false));
    assert!(errs.contains("writes no Collision events") && errs.contains("Generate Collision Event"), "{errs}");
    let errs = check(&fireworks(1, r#"(source: "Burst", event: Death, execution_mode: EveryParticle)"#, false));
    assert!(errs.contains("writes no Death events"), "{errs}");
    let errs = check(&fireworks(1, r#"(source: "Shell", event: Death, execution_mode: Sometimes)"#, false));
    assert!(errs.contains("`execution_mode` must be"), "{errs}");
    let errs = check(&fireworks(1, r#"(source: "Shell", event: Death, modules: [(module: "Collision")])"#, false));
    assert!(errs.contains("cannot be used in"), "{errs}");
    let errs = check(&fireworks(1, r#"(source: "Shell", event: Death, modules: [(module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Event.Speed") })])"#, false));
    assert!(errs.contains("Event.Velocity"), "{errs}");
    // Event. parameters only in handlers.
    let errs = check(&falling("(0.0, 0.0, 0.0)", "(0.0, 0.0, 0.0)", "", r#"(module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Event.Position") }),"#));
    assert!(errs.contains("cannot read Event."), "{errs}");
}

#[test]
fn event_warnings() {
    let c = fx_core::check_source(
        "t.fx.ron",
        &falling("(0.0, 0.0, 0.0)", "(0.0, 0.0, 0.0)", "", r#"(module: "Generate Death Event"),"#)
            .replace(
                r#"(module: "Solve Forces and Velocity"),
            (module: "Collision", inputs: {  }),
            (module: "Generate Collision Event"),"#,
                r#"(module: "Collision", inputs: {  }), (module: "Solve Forces and Velocity"), (module: "Generate Collision Event"),"#,
            ),
    );
    let w: Vec<String> = c.diagnostics.iter().map(|d| d.to_string()).collect();
    let w = w.join("\n");
    assert!(w.contains("runs before \"Solve Forces and Velocity\""), "{w}");
    assert!(w.contains("writes Death events but no emitter handles them"), "{w}");
}

#[test]
fn events_are_deterministic() {
    let src = fireworks(
        20,
        r#"(source: "Shell", event: Location, spawn_number: 1, modules: [(module: "Receive Location Event")])"#,
        false,
    );
    let run = || {
        let mut sim = Simulation::new(compile(&src), 11);
        for _ in 0..8 {
            sim.step(DT);
        }
        sim.state_hash()
    };
    assert_eq!(run(), run());
}
