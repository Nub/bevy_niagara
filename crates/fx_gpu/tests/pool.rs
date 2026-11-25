//! Instances sharing a compiled program, reset for reuse, deactivated and
//! with scaled spawn counts.

use std::sync::Arc;

use fx_gpu::{GpuEffect, GpuProgram};

const DT: f32 = 1.0 / 60.0;

fn sample(name: &str) -> fx_core::CookedSystem {
    let path = format!("{}/../../effects/samples/{name}", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).unwrap();
    let c = fx_core::check_source(&path, &src);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("{errs:?}"))
}

fn run(device: &fx_gpu::wgpu::Device, queue: &fx_gpu::wgpu::Queue, fx: &mut GpuEffect, steps: u32) {
    for _ in 0..steps {
        let mut enc = device.create_command_encoder(&Default::default());
        fx.step(queue, &mut enc, DT);
        queue.submit([enc.finish()]);
        fx.submitted();
    }
}

fn sorted(
    device: &fx_gpu::wgpu::Device,
    queue: &fx_gpu::wgpu::Queue,
    fx: &GpuEffect,
) -> Vec<Vec<fx_core::sim::Particle>> {
    let mut ps = fx.read_particles(device, queue);
    for e in &mut ps {
        e.sort_by_key(|p| p.unique_id);
    }
    ps
}

#[test]
fn reset_instance_matches_a_fresh_one() {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    for name in ["fountain.fx.ron", "burst.fx.ron", "grid_smoke.fx.ron"] {
        let program = Arc::new(GpuProgram::new(&device, Arc::new(sample(name)), Default::default()).unwrap());
        let mut used = GpuEffect::from_program(&device, &program, 1);
        run(&device, &queue, &mut used, 45);
        used.reset(9);
        run(&device, &queue, &mut used, 30);
        let mut fresh = GpuEffect::from_program(&device, &program, 9);
        run(&device, &queue, &mut fresh, 30);
        assert!(Arc::ptr_eq(&used.program, &fresh.program));
        let (a, b) = (sorted(&device, &queue, &used), sorted(&device, &queue, &fresh));
        for (ea, eb) in a.iter().zip(&b) {
            assert_eq!(ea.len(), eb.len(), "{name}: count after reset");
            for (pa, pb) in ea.iter().zip(eb) {
                assert_eq!(pa.unique_id, pb.unique_id, "{name}");
                assert!(
                    (pa.position - pb.position).length() < 1e-5,
                    "{name}: particle {} at {} vs {}",
                    pa.unique_id,
                    pa.position,
                    pb.position
                );
            }
        }
        assert!(a.iter().any(|e| !e.is_empty()), "{name}: nothing alive");
    }
}

#[test]
fn inactive_spawns_nothing_and_scale_cuts_spawns() {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let sys = sample("fountain.fx.ron");
    let program = Arc::new(GpuProgram::new(&device, Arc::new(sys), Default::default()).unwrap());
    let mut full = GpuEffect::from_program(&device, &program, 3);
    let mut half = GpuEffect::from_program(&device, &program, 3);
    half.spawn_scale = 0.5;
    run(&device, &queue, &mut full, 60);
    run(&device, &queue, &mut half, 60);
    let ids = |fx: &GpuEffect| fx.read_counts(&device, &queue).iter().map(|c| c.1).sum::<u32>();
    let (nf, nh) = (ids(&full), ids(&half));
    eprintln!("spawned: full {nf}, half {nh}");
    assert!(nf > 0 && (nh as f32 - nf as f32 * 0.5).abs() <= 2.0 + nf as f32 * 0.02);
    // Deactivated: no new particles, the living ones keep simulating and die.
    full.active = false;
    run(&device, &queue, &mut full, 1);
    let before = ids(&full);
    run(&device, &queue, &mut full, 600);
    assert_eq!(ids(&full), before, "spawned while inactive");
    let alive: u32 = full.read_counts(&device, &queue).iter().map(|c| c.0).sum();
    assert_eq!(alive, 0, "particles still alive 10 s after deactivation");
}
