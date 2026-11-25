//! One benchmark scene, run headless in its own process.
//!
//! Renders the scene's effect to a 1080p offscreen target with the real
//! renderer (fx_render in Bevy's raster pipeline), one fixed 1/60 s sim step
//! per frame. Waits for every pipeline to compile, warms up so spawn-rate
//! effects reach steady state, then records over the measured frames:
//! - every render diagnostic span (GPU timestamp queries and CPU encode time),
//! - the wall time of each frame (CPU frame time; the GPU is not awaited
//!   beyond what wgpu's frame pacing does),
//! - heap allocations per frame (whole process: Bevy, wgpu and fx).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderAdapterInfo;
use bevy::window::ExitCondition;
use fx_render::{FxClock, FxPlugin, FxReady, ParticleEffect};

use crate::{PassTiming, SceneDef, SceneResult, median};

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

#[derive(Default)]
struct Samples {
    /// Diagnostic path (without the elapsed_* suffix) -> (gpu ms, cpu ms).
    spans: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
    frame_ms: Vec<f64>,
    allocs: Vec<u64>,
    adapter: Option<String>,
}

#[derive(Resource)]
struct Bench {
    warm_frames: u32,
    frames: u32,
    warm: u32,
    measured: u32,
    last: Option<(Instant, u64)>,
    /// When the scene started waiting to be ready.
    started: Option<Instant>,
    samples: Arc<Mutex<Samples>>,
}

pub fn run(def: &SceneDef, frames: u32, warm_frames: u32, (w, h): (u32, u32)) -> SceneResult {
    let path = crate::bench_dir().join("..").join(def.file);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let checked = fx_core::check_source(def.file, &src);
    for d in &checked.diagnostics {
        eprintln!("{d}");
    }
    let system = checked.cooked.expect("bench scene compiles");
    let max_particles: u64 = system
        .emitters
        .iter()
        .map(|e| e.max_particles as u64)
        .sum();

    let samples = Arc::new(Mutex::new(Samples::default()));
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            // Textures and materials load relative to the effect file, as in fxpreview.
            .set(bevy::asset::AssetPlugin {
                file_path: std::fs::canonicalize(path.parent().unwrap_or(&path))
                    .unwrap_or_else(|_| path.parent().unwrap_or(&path).to_path_buf())
                    .to_string_lossy()
                    .into_owned(),
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::winit::WinitPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        RenderDiagnosticsPlugin,
        FxPlugin::default(),
    ))
    .insert_resource(FxClock::manual(1.0 / 60.0))
    .insert_resource(Bench {
        warm_frames,
        frames,
        warm: 0,
        measured: 0,
        last: None,
        started: None,
        samples: samples.clone(),
    });
    let (camera, target) = (def.camera, def.target);
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut images: ResMut<Assets<Image>>,
              mut meshes: ResMut<Assets<Mesh>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            // Scenes whose particles collide with the scene get a floor to hit.
            if system.emitters.iter().any(|e| e.uses_scene_collision()) {
                commands.spawn((
                    Mesh3d(meshes.add(Plane3d::default().mesh().size(30.0, 30.0))),
                    MeshMaterial3d(materials.add(StandardMaterial::default())),
                ));
            }
            let mut img = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
            img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            commands.spawn((
                Camera3d::default(),
                RenderTarget::from(images.add(img)),
                Hdr,
                DepthPrepass,
                Msaa::Off,
                Transform::from_translation(camera.into()).looking_at(target.into(), Vec3::Y),
            ));
            commands.spawn((ParticleEffect::new(system.clone(), 1), Transform::IDENTITY));
        },
    )
    .add_systems(Update, measure);
    app.run();

    let s = samples.lock().unwrap();
    let passes: BTreeMap<String, PassTiming> = s
        .spans
        .iter()
        .map(|(p, (g, c))| {
            (
                p.clone(),
                PassTiming {
                    gpu_ms: median(g),
                    cpu_ms: median(c),
                },
            )
        })
        .collect();
    let allocs: Vec<f64> = s.allocs.iter().map(|&a| a as f64).collect();
    SceneResult::new(
        def,
        max_particles,
        s.adapter.clone(),
        passes,
        median(&s.frame_ms).unwrap_or(f64::NAN),
        median(&allocs).unwrap_or(f64::NAN),
        s.frame_ms.len() as u32,
    )
}

fn measure(
    mut bench: ResMut<Bench>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    store: Res<DiagnosticsStore>,
    adapter: Option<Res<RenderAdapterInfo>>,
    mut exit: MessageWriter<AppExit>,
) {
    if !ready.get() {
        // A texture, material or pipeline that never loads would hang the run.
        let started = *bench.started.get_or_insert_with(Instant::now);
        if started.elapsed() > Duration::from_secs(120) {
            panic!("scene never became ready (an asset failed to load?)");
        }
        return;
    }
    clock.request_steps(1);
    // Let spawn-rate effects reach steady state and timings settle.
    if bench.warm < bench.warm_frames {
        bench.warm += 1;
        return;
    }
    let now = (Instant::now(), ALLOCS.load(Ordering::Relaxed));
    let last = bench.last.replace(now);
    let Some((t0, a0)) = last else {
        return;
    };
    let mut s = bench.samples.lock().unwrap();
    if s.adapter.is_none() {
        s.adapter = adapter.map(|a| a.name.clone());
    }
    s.frame_ms.push((now.0 - t0).as_secs_f64() * 1000.0);
    s.allocs.push(now.1 - a0);
    for d in store.iter() {
        let path = d.path().as_str();
        let (span, gpu) = if let Some(p) = path.strip_suffix("/elapsed_gpu") {
            (p, true)
        } else if let Some(p) = path.strip_suffix("/elapsed_cpu") {
            (p, false)
        } else {
            continue;
        };
        let Some(v) = d.measurement().map(|m| m.value) else {
            continue;
        };
        let e = s.spans.entry(span.to_string()).or_default();
        if gpu { e.0.push(v) } else { e.1.push(v) }
    }
    drop(s);
    bench.measured += 1;
    if bench.measured >= bench.frames {
        exit.write(AppExit::Success);
    }
}
