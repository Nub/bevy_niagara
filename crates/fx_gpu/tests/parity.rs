//! CPU/GPU parity: the same sample effect and seed give the same particles
//! on the GPU as fx_core's CPU reference, within float tolerance.
//!
//! Integer work (random hashes, spawn counts, UniqueIDs, kills) must match
//! exactly; float attributes may differ by the GPU's FMA contraction and its
//! exp/sin/cos approximations. One inline system covers the extended
//! attributes (Ribbon, Mesh, Light, material parameters).

use std::path::PathBuf;

use fx_core::sim::{Particle, Simulation};
use fx_gpu::{GpuEffect, WORDS_PER_PARTICLE, WORDS_PER_PARTICLE_EXTENDED};

const DT: f32 = 1.0 / 60.0;

fn sample(name: &str) -> fx_core::CookedSystem {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../effects/samples")
        .join(name);
    let c = fx_core::check_file(&path).unwrap();
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("{name}: {errs:?}"))
}

fn fields(p: &Particle) -> Vec<(&'static str, f32)> {
    let mut v = Vec::new();
    let mut put = |n: &'static str, xs: &[f32]| xs.iter().for_each(|x| v.push((n, *x)));
    put("position", &p.position.to_array());
    put("velocity", &p.velocity.to_array());
    put("color", &p.color.to_array());
    put("sprite_size", &p.sprite_size.to_array());
    put("sprite_rotation", &[p.sprite_rotation]);
    put("mass", &[p.mass]);
    put("lifetime", &[p.lifetime]);
    put("age", &[p.age]);
    put("normalized_age", &[p.normalized_age]);
    put("initial_color", &p.initial_color.to_array());
    put("initial_sprite_size", &p.initial_sprite_size.to_array());
    put("force", &p.force.to_array());
    put("drag", &[p.drag]);
    put("scale", &p.scale.to_array());
    put("mesh_orientation", &p.mesh_orientation.to_array());
    put("ribbon_width", &[p.ribbon_width]);
    put("ribbon_twist", &[p.ribbon_twist]);
    put("ribbon_link_order", &[p.ribbon_link_order]);
    put("light_radius", &[p.light_radius]);
    put(
        "dynamic_material_parameter",
        &p.dynamic_material_parameter.to_array(),
    );
    v
}

/// Largest absolute and relative difference seen.
#[derive(Default, Debug)]
struct Diff {
    max_abs: f32,
    worst: String,
}

fn compare(name: &str, step: u32, cpu: &[Particle], gpu: &[Particle], diff: &mut Diff) {
    let mut cpu: Vec<_> = cpu.to_vec();
    let mut gpu: Vec<_> = gpu.to_vec();
    cpu.sort_by_key(|p| p.unique_id);
    gpu.sort_by_key(|p| p.unique_id);
    let ids = |v: &[Particle]| v.iter().map(|p| p.unique_id).collect::<Vec<_>>();
    assert_eq!(
        ids(&cpu),
        ids(&gpu),
        "{name} step {step}: living particle ids differ ({} on CPU, {} on GPU)",
        cpu.len(),
        gpu.len()
    );
    for (a, b) in cpu.iter().zip(&gpu) {
        assert_eq!(
            a.ribbon_id, b.ribbon_id,
            "{name} step {step} particle {}: ribbon_id",
            a.unique_id
        );
        for ((field, x), (_, y)) in fields(a).into_iter().zip(fields(b)) {
            let d = (x - y).abs();
            let tol = 1e-3 * (1.0 + x.abs());
            assert!(
                d <= tol,
                "{name} step {step} particle {}: {field} CPU {x} GPU {y}",
                a.unique_id
            );
            if d > diff.max_abs {
                diff.max_abs = d;
                diff.worst = format!("{field} of particle {} at step {step}", a.unique_id);
            }
        }
    }
}

/// Every Ribbon/Mesh/Light/material attribute and the modules that write
/// them, plus a panning curl field; drawn by Mesh and Light renderers, so the
/// emitter stores the extended attributes.
const EXTENDED: &str = r#"(
    name: "Extended",
    emitters: [(
        name: "E",
        max_particles: 2000,
        emitter_update: [
            (module: "Spawn Rate", inputs: { "Spawn Rate": Float(300.0) }),
            (module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(50) }),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {
                "Lifetime": RandomRange(Float(0.5), Float(1.5)),
                "Position": RandomRange(Vec3((-1.0, -1.0, -1.0)), Vec3((1.0, 1.0, 1.0))),
            }),
            (module: "Initialize Ribbon", inputs: {
                "Ribbon ID": RandomRange(Int(0), Int(7)),
                "Ribbon Width": RandomRange(Float(0.05), Float(0.2)),
                "Ribbon Twist": Float(30.0),
                "Ribbon Link Order": Link(Float, "Engine.Time"),
            }),
            (module: "Initialize Mesh", inputs: {
                "Mesh Scale": RandomRange(Vec3((0.5, 0.5, 0.5)), Vec3((2.0, 2.0, 2.0))),
                "Mesh Rotation": RandomRange(Vec3((-180.0, -180.0, -180.0)), Vec3((180.0, 180.0, 180.0))),
            }),
            (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Scale") }),
            (module: "Initialize Light", inputs: { "Light Radius": RandomRange(Float(0.5), Float(3.0)) }),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Update Mesh Orientation", inputs: { "Rotation Rate": Vec3((90.0, -45.0, 200.0)) }),
            (module: "Dynamic Material Parameters", inputs: {
                "Dynamic Material Parameter": CurveOverAge(Vec4([(0.0, (0.0, 0.0, 0.0, 1.0)), (1.0, (1.0, 2.0, 3.0, 0.0))])),
            }),
            (module: "Curl Noise Force", inputs: {
                "Noise Strength": Float(2.0),
                "Pan Noise Field": Vec3((0.5, 1.0, -0.25)),
            }),
            (module: "Initialize Light", inputs: { "Light Radius": Link(Float, "Particles.NormalizedAge") }),
            (module: "Solve Forces and Velocity"),
        ],
        renderers: [Mesh((mesh: "sphere")), Light],
    )],
)"#;

fn extended_system() -> fx_core::CookedSystem {
    let c = fx_core::check_source("extended.fx.ron", EXTENDED);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("extended: {errs:?}"))
}

#[test]
fn gpu_matches_cpu_reference() {
    let (device, queue, info) = match fx_gpu::headless_device() {
        Ok(d) => d,
        Err(e) => panic!("parity test needs a GPU: {e}"),
    };
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    let mut systems: Vec<_> = ["fountain.fx.ron", "burst.fx.ron", "smoke_curl.fx.ron"]
        .into_iter()
        .map(|n| (n, sample(n), WORDS_PER_PARTICLE))
        .collect();
    systems.push(("extended", extended_system(), WORDS_PER_PARTICLE_EXTENDED));
    for (name, sys, words) in systems {
        let seed = 42;
        let mut cpu = Simulation::new(sys.clone(), seed);
        let mut gpu = GpuEffect::new(&device, &sys, seed).unwrap();
        assert!(
            gpu.emitters.iter().all(|e| e.words == words),
            "{name}: words per particle"
        );
        let mut diff = Diff::default();
        let mut most_alive = 0usize;
        // 3 seconds: bursts, steady spawning, deaths and the curl field.
        for step in 1..=180u32 {
            cpu.step(DT);
            let mut enc = device.create_command_encoder(&Default::default());
            gpu.step(&queue, &mut enc, DT);
            queue.submit([enc.finish()]);
            gpu.submitted();
            if step % 15 == 0 {
                let g = gpu.read_particles(&device, &queue);
                for (ei, ps) in g.iter().enumerate() {
                    most_alive = most_alive.max(ps.len());
                    compare(name, step, cpu.particles(ei), ps, &mut diff);
                }
                let counts = gpu.read_counts(&device, &queue);
                for (ei, (_, next_id)) in counts.iter().enumerate() {
                    assert_eq!(*next_id, cpu.emitters[ei].next_id, "{name}: next UniqueID");
                }
            }
        }
        assert!(most_alive > 0, "{name}: nothing alive to compare");
        eprintln!(
            "{name}: up to {} particles compared at 12 checkpoints; largest difference {:.2e} ({})",
            most_alive, diff.max_abs, diff.worst
        );
    }
}

#[test]
fn several_steps_per_submit_match_one_per_submit() {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let sys = sample("fountain.fx.ron");
    let mut a = GpuEffect::new(&device, &sys, 7).unwrap();
    let mut b = GpuEffect::new(&device, &sys, 7).unwrap();
    for _ in 0..8 {
        let mut enc = device.create_command_encoder(&Default::default());
        for _ in 0..4 {
            a.step(&queue, &mut enc, DT);
        }
        queue.submit([enc.finish()]);
        a.submitted();
        for _ in 0..4 {
            let mut enc = device.create_command_encoder(&Default::default());
            b.step(&queue, &mut enc, DT);
            queue.submit([enc.finish()]);
            b.submitted();
        }
    }
    let mut pa = a.read_particles(&device, &queue).remove(0);
    let mut pb = b.read_particles(&device, &queue).remove(0);
    pa.sort_by_key(|p| p.unique_id);
    pb.sort_by_key(|p| p.unique_id);
    assert!(!pa.is_empty());
    assert_eq!(pa.len(), pb.len());
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(
            x.bits(),
            y.bits(),
            "GPU results depend on how steps are submitted"
        );
    }
}
