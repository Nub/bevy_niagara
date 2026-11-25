//! Same seed → bit-identical particle state; the cooked binary simulates the
//! same as the in-memory compile.

mod common;

use common::*;
use fx_core::cook;
use fx_core::sim::Simulation;

fn run(sys: fx_core::CookedSystem, seed: u32, steps: usize) -> Simulation {
    let mut sim = Simulation::new(sys, seed);
    for _ in 0..steps {
        sim.step(1.0 / 60.0);
    }
    sim
}

fn sample(name: &str) -> fx_core::CookedSystem {
    let c = fx_core::check_file(&effects_dir().join("samples").join(name)).unwrap();
    c.cooked.unwrap()
}

#[test]
fn same_seed_gives_identical_state() {
    for name in ["fountain.fx.ron", "burst.fx.ron", "smoke_curl.fx.ron"] {
        let sys = sample(name);
        let a = run(sys.clone(), 42, 60);
        let b = run(sys.clone(), 42, 60);
        assert!(a.particle_count() > 0, "{name}");
        assert_eq!(a.emitters, b.emitters, "{name}");
        for (ea, eb) in a.emitters.iter().zip(&b.emitters) {
            for (pa, pb) in ea.particles.iter().zip(&eb.particles) {
                assert_eq!(pa.bits(), pb.bits(), "{name}");
            }
        }
        assert_eq!(a.state_hash(), b.state_hash(), "{name}");

        // A different seed gives a different state.
        let c = run(sys, 43, 60);
        assert_ne!(a.state_hash(), c.state_hash(), "{name}");
    }
}

#[test]
fn cooked_bytes_simulate_identically() {
    let sys = sample("smoke_curl.fx.ron");
    let loaded = cook::from_bytes(&cook::to_bytes(&sys)).unwrap();
    assert_eq!(
        run(sys, 9, 120).state_hash(),
        run(loaded, 9, 120).state_hash()
    );
}

#[test]
fn split_runs_match_one_run() {
    // Stepping is pure: a clone taken mid-run continues exactly like the original.
    let sys = sample("fountain.fx.ron");
    let mut a = run(sys, 5, 50);
    let mut b = a.clone();
    for _ in 0..50 {
        a.step(1.0 / 60.0);
        b.step(1.0 / 60.0);
    }
    assert_eq!(a.state_hash(), b.state_hash());
}
