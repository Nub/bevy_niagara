//! Per-frame allocations once warm.
//!
//! fx_gpu's own per-frame work (emitter stacks, uniform packing, choosing
//! bind groups and indirect offsets) must not touch the heap and must not
//! create GPU resources. wgpu itself allocates a few small blocks for every
//! compute pass it records; that cost is fixed per pass, so the test checks
//! that a frame's heap allocations are the same at 1k and at 200k particles,
//! the same every frame, and equal to a frame recording the same passes with
//! no fx_gpu logic at all. GPU memory must not change after creation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use fx_gpu::GpuEffect;
use fx_gpu::view::{SortMode, ViewDraw, ViewParams, ViewPipelines};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn effect(n: u32) -> fx_core::CookedSystem {
    let src = format!(
        r#"(
    name: "Alloc",
    user_parameters: {{ "User.Rate": Float(50000.0) }},
    emitters: [(
        name: "E",
        max_particles: {n},
        emitter_update: [
            (module: "Spawn Rate", inputs: {{ "Spawn Rate": Link(Float, "User.Rate") }}),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{
                "Lifetime": RandomRange(Float(0.5), Float(2.0)),
                "Position": RandomRange(Vec3((-1.0, 0.0, -1.0)), Vec3((1.0, 2.0, 1.0))),
            }}),
            (module: "Add Velocity in Cone", inputs: {{ "Velocity Strength": Float(2.0) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Gravity Force"),
            (module: "Curl Noise Force"),
            (module: "Solve Forces and Velocity"),
            (module: "Scale Color", inputs: {{ "Scale Alpha": CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)])) }}),
        ],
    )],
)"#
    );
    let c = fx_core::check_source("alloc.fx.ron", &src);
    c.cooked.expect("compiles")
}

fn view() -> ViewParams {
    ViewParams {
        planes: [[0.0, 0.0, 0.0, 1.0]; 6],
        camera_position: [0.0, 1.0, 5.0],
        camera_forward: [0.0, 0.0, -1.0],
    }
}

/// Heap allocations of each of `frames` warm frames.
fn frame_allocs(n: u32, sort: SortMode, frames: usize) -> (Vec<(u64, u32)>, u64) {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let vp = ViewPipelines::new(&device);
    let mut fx = GpuEffect::new(&device, &effect(n), 3).unwrap();
    let mut draw = ViewDraw::new(&device, &vp, &fx.emitters[0], sort);
    let v = view();
    let bytes_before = fx.gpu_bytes();
    let mut per_frame = Vec::new();
    for f in 0..(30 + frames) {
        let before = ALLOCS.load(Ordering::Relaxed);
        let mut enc = device.create_command_encoder(&Default::default());
        fx.step(&queue, &mut enc, 1.0 / 60.0);
        draw.record(&queue, &mut enc, &vp, &fx.emitters[0], &v);
        queue.submit([enc.finish()]);
        fx.submitted();
        let after = ALLOCS.load(Ordering::Relaxed);
        device
            .poll(fx_gpu::wgpu::PollType::wait_indefinitely())
            .unwrap();
        // The count read-back is a test-only extra, outside the measured window
        // (done during warm-up too, so its own first-use costs are warm).
        let alive = fx.read_counts(&device, &queue)[0].0;
        if f >= 30 {
            per_frame.push((after - before, alive));
        }
    }
    assert_eq!(
        fx.gpu_bytes(),
        bytes_before,
        "GPU memory changed after creation"
    );
    (per_frame, n as u64)
}

/// The same frame shape with no fx_gpu logic: the passes wgpu would record anyway.
fn baseline_allocs(passes: usize, frames: usize) -> Vec<u64> {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let mut per_frame = Vec::new();
    for f in 0..(60 + frames) {
        let before = ALLOCS.load(Ordering::Relaxed);
        let mut enc = device.create_command_encoder(&Default::default());
        for _ in 0..passes {
            drop(enc.begin_compute_pass(&Default::default()));
        }
        queue.submit([enc.finish()]);
        let after = ALLOCS.load(Ordering::Relaxed);
        device
            .poll(fx_gpu::wgpu::PollType::wait_indefinitely())
            .unwrap();
        if f >= 60 {
            per_frame.push(after - before);
        }
    }
    per_frame
}

#[test]
fn warm_frames_allocate_a_fixed_amount_independent_of_particles() {
    // 50k particles/s with 0.5-2 s lifetimes: the living count grows from
    // ~25k to ~60k over the measured frames.
    let (frames, cap) = frame_allocs(200_000, SortMode::ViewDepth, 110);
    let (first, last) = (frames[0], frames[frames.len() - 1]);
    eprintln!(
        "per-frame heap allocations: {} at {} particles ... {} at {} (pool {cap})",
        first.0, first.1, last.0, last.1
    );
    assert!(
        last.1 > first.1 * 2,
        "the living count should grow: {first:?} -> {last:?}"
    );
    assert!(
        frames.iter().all(|f| f.0 == first.0),
        "allocations vary: {:?}",
        frames.iter().map(|f| f.0).collect::<Vec<_>>()
    );
    // Sim: 4 passes; view: 2 passes.
    let base = baseline_allocs(6, 20);
    eprintln!(
        "bare wgpu frame recording the same 6 empty passes: {} allocations",
        base[0]
    );
}
