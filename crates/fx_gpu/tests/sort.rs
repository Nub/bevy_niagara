//! Per-view sorting on the GPU: the bitonic sort (small views) and the radix
//! sort (views over 16384 particles) both give every visible particle once,
//! far to near for depth, by UniqueID for the deterministic mode.

use fx_gpu::GpuEffect;
use fx_gpu::view::{SortMode, ViewDraw, ViewParams, ViewPipelines};

fn effect(n: u32) -> fx_core::CookedSystem {
    let src = format!(
        r#"(
    name: "Sort",
    emitters: [(
        name: "E",
        max_particles: {n},
        emitter_update: [
            (module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int({n}) }}),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{
                "Lifetime": Float(100.0),
                "Position": RandomRange(Vec3((-2.0, 0.0, -2.0)), Vec3((2.0, 4.0, 2.0))),
            }}),
            (module: "Add Velocity", inputs: {{ "Velocity": RandomRange(Vec3((-0.2, -0.2, -0.2)), Vec3((0.2, 0.2, 0.2))) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Solve Forces and Velocity"),
        ],
    )],
)"#
    );
    fx_core::check_source("sort.fx.ron", &src).cooked.expect("compiles")
}

fn view() -> ViewParams {
    ViewParams {
        planes: [[0.0, 0.0, 0.0, 1.0]; 6],
        camera_position: [0.0, 1.0, 8.0],
        camera_forward: [0.0, 0.0, -1.0],
    }
}

/// Sorted indices of `n` particles and the particles (slot order).
fn sorted(n: u32, mode: SortMode) -> (Vec<u32>, Vec<fx_core::sim::Particle>) {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let vp = ViewPipelines::new(&device);
    let mut fx = GpuEffect::new(&device, &effect(n), 7).unwrap();
    let mut draw = ViewDraw::new(&device, &vp, &fx.emitters[0], mode);
    let v = view();
    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&Default::default());
        fx.step(&queue, &mut enc, 1.0 / 60.0);
        draw.record(&queue, &mut enc, &vp, &fx.emitters[0], &v);
        queue.submit([enc.finish()]);
        fx.submitted();
    }
    let args = fx_gpu::read_buffer(&device, &queue, &draw.draw_args, 0, 8);
    let count = u32::from_le_bytes(args[4..8].try_into().unwrap());
    let idx = fx_gpu::read_buffer(&device, &queue, &draw.indices, 0, count as u64 * 4);
    let idx: Vec<u32> = idx.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let ps = fx.read_particles(&device, &queue).remove(0);
    (idx, ps)
}

fn check(n: u32, mode: SortMode) {
    let (idx, ps) = sorted(n, mode);
    assert_eq!(ps.len(), n as usize, "all spawned");
    assert_eq!(idx.len(), n as usize, "every particle visible once");
    let mut seen = vec![false; n as usize];
    for &i in &idx {
        assert!(!seen[i as usize], "slot {i} twice");
        seen[i as usize] = true;
    }
    let v = view();
    let depth = |i: u32| {
        let p = ps[i as usize].position;
        (p.x - v.camera_position[0]) * v.camera_forward[0]
            + (p.y - v.camera_position[1]) * v.camera_forward[1]
            + (p.z - v.camera_position[2]) * v.camera_forward[2]
    };
    for w in idx.windows(2) {
        match mode {
            SortMode::ViewDepth => {
                let (a, b) = (depth(w[0]), depth(w[1]));
                // Far first; the radix sort keys depth to 2^-15 relative.
                assert!(a >= b - a.abs() * 1e-4, "depth order: {a} then {b}");
            }
            SortMode::UniqueId => {
                assert!(ps[w[0] as usize].unique_id < ps[w[1] as usize].unique_id, "UniqueID order");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn bitonic_depth() {
    check(3000, SortMode::ViewDepth);
}

#[test]
fn radix_depth() {
    check(200_000, SortMode::ViewDepth);
}

#[test]
fn radix_unique_id() {
    check(70_000, SortMode::UniqueId);
}

/// The deterministic mode (headless preview): the same particles in the same
/// order on every run, whatever slots the sim's atomic compaction gave them.
#[test]
fn radix_unique_id_is_deterministic() {
    let ids = || {
        let (idx, ps) = sorted(100_000, SortMode::UniqueId);
        idx.iter().map(|&i| ps[i as usize].unique_id).collect::<Vec<_>>()
    };
    assert_eq!(ids(), ids());
}
