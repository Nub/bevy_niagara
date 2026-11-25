//! The free Niagara test packages in testdata/niagara (fetched by
//! `sh testdata/niagara_assets.sh`; skipped when absent), converted and run
//! on the CPU simulation: no particle flashes or snaps between frames.
//!
//! Every one of their emitters has Interpolated Spawning, so Niagara runs
//! Particle Update on a particle in the tick it is spawned in. Before the
//! converter carried that over, each new particle was drawn once with its
//! spawn values (alpha 1, full size or mesh scale) before its update curves
//! brought it to ~0: the flashing of NS_FireSmokeOnly and the snapping fish
//! of NS_Fish.

use std::collections::HashMap;
use std::path::PathBuf;

use fx_core::sim::{Particle, Simulation};
use fx_import::{Package, convert, niagara};

const DT: f32 = 1.0 / 30.0;

fn files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/niagara");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: {} is absent (sh testdata/niagara_assets.sh)", dir.display());
        return Vec::new();
    };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "uasset"))
        .collect();
    v.sort();
    v
}

/// Alpha, sprite size and mesh scale: what a curve over age usually drives.
fn shown(p: &Particle) -> [f32; 3] {
    [p.color.w, p.sprite_size.max_element(), p.scale.max_element()]
}

#[test]
fn imported_particles_change_smoothly_between_frames() {
    for f in files() {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let pkg = Package::read(&std::fs::read(&f).unwrap()).unwrap();
        let sys = niagara::read_system(&pkg).unwrap();
        let conv = convert::convert(&sys, &format!("{name}.uasset"));
        for e in sys.emitters.iter().filter(|e| e.enabled && e.interpolated_spawning) {
            assert!(conv.ron.contains("interpolated_spawning: true"), "{name}: {} lost Interpolated Spawning", e.name);
        }
        let c = fx_core::check_source(&format!("{name}.fx.ron"), &conv.ron);
        let Some(cooked) = c.cooked else { panic!("{name} does not cook") };
        let mut sim = Simulation::new(cooked, 11);
        let emitters = sim.emitters.len();
        // Per emitter and attribute: largest value seen, largest change of
        // one particle between frames, and the first frame of the particle
        // that changed most after it.
        let mut most = vec![[0.0f32; 3]; emitters];
        let mut jump: Vec<[(f32, String); 3]> = (0..emitters).map(|_| Default::default()).collect();
        let mut prev: Vec<HashMap<u32, Particle>> = vec![HashMap::new(); emitters];
        for step in 0..120 {
            sim.step(DT);
            for ei in 0..emitters {
                let mut cur = HashMap::new();
                for p in sim.particles(ei) {
                    let v = shown(p);
                    for k in 0..3 {
                        most[ei][k] = most[ei][k].max(v[k].abs());
                    }
                    if let Some(q) = prev[ei].get(&p.unique_id) {
                        let u = shown(q);
                        for k in 0..3 {
                            let d = (v[k] - u[k]).abs();
                            if d > jump[ei][k].0 {
                                jump[ei][k] = (d, format!("step {step}, normalized age {} -> {}", q.normalized_age, p.normalized_age));
                            }
                        }
                    }
                    cur.insert(p.unique_id, *p);
                }
                prev[ei] = cur;
            }
        }
        for ei in 0..emitters {
            for (k, attr) in ["alpha", "sprite size", "mesh scale"].iter().enumerate() {
                let (d, at) = &jump[ei][k];
                let rel = if most[ei][k] > 0.0 { d / most[ei][k] } else { 0.0 };
                eprintln!("{name} emitter {ei} {attr}: largest change {d:.4} ({:.0}% of its largest value) at {at}", rel * 100.0);
                assert!(rel < 0.35, "{name} emitter {ei}: {attr} jumps {d} of {} at {at}", most[ei][k]);
            }
        }
    }
}
