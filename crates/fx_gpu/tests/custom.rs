//! Custom WGSL modules and simulation stages on the GPU.

use std::path::PathBuf;

use fx_gpu::GpuEffect;
use fx_gpu::codegen::{ShaderOptions, emitter_shader};
use fx_gpu::events::EventLayout;

const DT: f32 = 1.0 / 60.0;

fn effects(dir: &str) -> Vec<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../effects").join(dir);
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".fx.ron"))
        .collect();
    v.sort();
    v
}

fn cook(p: &PathBuf) -> fx_core::CookedSystem {
    let c = fx_core::check_file(p).unwrap();
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("{}: {errs:?}", p.display()))
}

fn sample(name: &str) -> fx_core::CookedSystem {
    cook(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../effects/samples").join(name))
}

/// Every generated emitter shader (raster and ray-tracing variants) passes
/// naga's validation: the generator and fx_core's custom-module checks agree.
/// Needs no GPU.
#[test]
fn every_generated_shader_validates() {
    for p in effects("samples").iter().chain(&effects("library")) {
        let sys = cook(p);
        let layout = EventLayout::new(&sys);
        for ei in 0..sys.emitters.len() {
            for ray_tracing in [false, true] {
                let sh = emitter_shader(&sys, ei, &layout, ShaderOptions { ray_tracing });
                if let Err(e) = fx_core::custom::validate_wgsl(&sh.source) {
                    let line = sh.source.lines().nth(e.line.saturating_sub(1) as usize).unwrap_or("");
                    panic!("{} emitter {ei} (rt {ray_tracing}): {}:{}: {}\n  {line}", p.display(), e.line, e.col, e.message);
                }
            }
        }
    }
}

fn run(sys: &fx_core::CookedSystem, steps: u32, seed: u32) -> (GpuEffect, wgpu::Device, wgpu::Queue) {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let mut gpu = GpuEffect::new(&device, sys, seed).unwrap();
    for _ in 0..steps {
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.step(&queue, &mut enc, DT);
        queue.submit([enc.finish()]);
        gpu.submitted();
    }
    (gpu, device, queue)
}

#[test]
fn wgsl_swirl_turns_and_tints_the_particles() {
    let sys = sample("wgsl_swirl.fx.ron");
    let (gpu, device, queue) = run(&sys, 120, 5);
    let ps = &gpu.read_particles(&device, &queue)[0];
    assert!(ps.len() > 1000, "{} particles", ps.len());
    // Counter-clockwise round +Y seen from above: velocity along (-z, 0, x).
    let mut turning = 0.0f32;
    let mut tinted = 0;
    for p in ps {
        let r = p.position;
        let t = bevy_math::Vec3::new(-r.z, 0.0, r.x).normalize_or_zero();
        turning += p.velocity.dot(t);
        if (p.color.x - p.color.z).abs() > 0.05 {
            tinted += 1;
        }
        assert!(p.position.is_finite() && p.velocity.is_finite());
    }
    assert!(turning / ps.len() as f32 > 0.5, "mean tangential speed {}", turning / ps.len() as f32);
    assert!(tinted > ps.len() / 2, "{tinted} of {} tinted by the module's Color output", ps.len());
}

/// Grid words of the smoke sample's grid, current copy.
fn smoke_grid(gpu: &GpuEffect, device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<f32>, u32) {
    let g = &gpu.system.emitters[0].grids[0];
    let per_step: u32 = gpu.system.emitters[0].stages.iter().map(|s| s.iterations).sum();
    let half = ((gpu.frame * per_step) & 1) as u64;
    let bytes = fx_gpu::read_buffer(device, queue, &gpu.emitters[0].grids, half * g.half_words() as u64 * 4, g.half_words() as u64 * 4);
    (bytemuck::cast_slice(&bytes).to_vec(), g.cell_count())
}

#[test]
fn grid_smoke_rises_and_carries_the_particles() {
    let sys = sample("grid_smoke.fx.ron");
    let (gpu, device, queue) = run(&sys, 120, 9);
    let (grid, cells) = smoke_grid(&gpu, &device, &queue);
    assert!(grid.iter().all(|x| x.is_finite()), "non-finite grid values");
    let density: f32 = grid[2 * cells as usize..3 * cells as usize].iter().sum();
    let up: f32 = grid[cells as usize..2 * cells as usize].iter().sum();
    assert!(density > 1.0, "density {density}");
    assert!(up > 0.0, "net vertical velocity {up}");
    let ps = &gpu.read_particles(&device, &queue)[0];
    let mean_y = ps.iter().map(|p| p.position.y).sum::<f32>() / ps.len() as f32;
    assert!(mean_y > 0.6, "particles at mean height {mean_y}; they spawn below 0.45");
    // Same seed, same result.
    let (again, d2, q2) = run(&sys, 120, 9);
    assert_eq!(smoke_grid(&again, &d2, &q2).0, grid, "the grid simulation is not deterministic");
}

/// A particle simulation stage runs after the spawn, in place, and sees the
/// grids; three iterations apply it three times.
#[test]
fn particle_stage_iterations_apply_in_place() {
    let src = r##"(
    name: "T",
    custom_modules: [(name: "Nudge", outputs: ["Particles.Mass"], wgsl: r#"Mass += 1.0;"#)],
    emitters: [(
        name: "E",
        emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(100) })],
        particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(10.0), "Mass": Float(1.0) })],
        particle_update: [(module: "Particle State")],
        simulation_stages: [(name: "N", iterate: Particles, iterations: 3, modules: [(module: "Nudge")])],
    )],
)"##;
    let c = fx_core::check_source("t.fx.ron", src);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    let sys = c.cooked.unwrap_or_else(|| panic!("{errs:?}"));
    let (gpu, device, queue) = run(&sys, 2, 1);
    let ps = &gpu.read_particles(&device, &queue)[0];
    assert_eq!(ps.len(), 100);
    // Spawned at 1, +3 in each of the two steps.
    assert!(ps.iter().all(|p| p.mass == 7.0), "masses {:?}", ps.iter().map(|p| p.mass).take(5).collect::<Vec<_>>());
}
