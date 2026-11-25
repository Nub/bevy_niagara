//! fxsolari: the particle ray-tracing test scene, headless.
//!
//! A dim room: a campfire (`scenes/fire_rt.fx.ron`) in front of a white back
//! wall, a mirror on the back wall, and a shower of sparks
//! (`scenes/sparks_rt.fx.ron`) behind the camera: seen only in the mirror. Rendered with Solari (or the raster pipeline), fixed 1/60 s sim
//! steps, the same particles every run.
//!
//! fxsolari shot  --out FILE.png [--mode raster|off|on] [--size WxH] [--frames N] [--accumulate N]
//!                [--effect FILE]   (another effect instead of the fire and sparks)
//! fxsolari bench [--mode none|off|on] [--size WxH] [--frames N] [--copies N]
//!
//! Modes: `raster` no Solari; `none` Solari without `FxSolariPlugin`; `off`
//! Solari with the plugin and every renderer's `ray_tracing` forced off; `on`
//! Solari with the effects' ray tracing on.
//! `shot` simulates `--frames` steps (one per frame), then pauses the
//! particles and averages `--accumulate` rendered frames (linear light), so
//! Solari's per-frame noise (no denoiser without DLSS) averages out; shots
//! use ReSTIR.
//! `bench` prints the median GPU ms of every render span over the measured
//! frames (and the median CPU frame time); `--copies` places N fire+sparks
//! pairs (or N copies of `--effect`) side by side.
//!
//! `scenes/sword_trail_rt.fx.ron` (with `--effect`) is the ribbon test: a
//! glowing sword trail lighting the back wall, seen in the mirror.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::{CameraMainTextureUsages, Exposure, Hdr, RenderTarget};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins};
use bevy::window::ExitCondition;
use fx_core::cook::{CookedRenderer, CookedSystem};
use fx_render::{FxClock, FxPlugin, FxReady, ParticleEffect};
use fx_solari::FxSolariPlugin;

const USAGE: &str = "usage: fxsolari shot --out FILE.png [--mode raster|off|on] [--size WxH] [--frames N] [--accumulate N] [--effect FILE]\n       fxsolari bench [--mode none|off|on] [--size WxH] [--frames N] [--copies N]";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Raster,
    None,
    Off,
    On,
}

#[derive(Clone, Debug)]
struct Options {
    bench: bool,
    out: PathBuf,
    mode: Mode,
    width: u32,
    height: u32,
    frames: u32,
    accumulate: u32,
    copies: u32,
    /// Another effect instead of the fire and sparks.
    effect: Option<PathBuf>,
}

fn parse() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let bench = match args.next().as_deref() {
        Some("shot") => false,
        Some("bench") => true,
        _ => return Err(String::new()),
    };
    let mut o = Options {
        bench,
        out: PathBuf::from("fxsolari.png"),
        mode: Mode::On,
        width: if bench { 1920 } else { 1280 },
        height: if bench { 1080 } else { 720 },
        frames: if bench { 300 } else { 150 },
        accumulate: 1,
        copies: 1,
        effect: None,
    };
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--out" => o.out = PathBuf::from(val()?),
            "--mode" => {
                o.mode = match val()?.as_str() {
                    "raster" => Mode::Raster,
                    "none" => Mode::None,
                    "off" => Mode::Off,
                    "on" => Mode::On,
                    m => return Err(format!("unknown mode {m}")),
                }
            }
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--size is WxH")?;
                o.width = w.parse().map_err(|e| format!("--size: {e}"))?;
                o.height = h.parse().map_err(|e| format!("--size: {e}"))?;
            }
            "--frames" => o.frames = val()?.parse().map_err(|e| format!("--frames: {e}"))?,
            "--accumulate" => o.accumulate = val()?.parse::<u32>().map_err(|e| format!("--accumulate: {e}"))?.max(1),
            "--effect" => o.effect = Some(PathBuf::from(val()?)),
            "--copies" => o.copies = val()?.parse().map_err(|e| format!("--copies: {e}"))?,
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown option {a}")),
        }
    }
    Ok(o)
}

fn scenes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes")
}

fn load(name: &str, ray_tracing: bool) -> Result<CookedSystem, String> {
    load_path(&scenes_dir().join(name), ray_tracing)
}

fn load_path(path: &Path, ray_tracing: bool) -> Result<CookedSystem, String> {
    let checked = fx_core::check_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    for d in &checked.diagnostics {
        eprintln!("{d}");
    }
    let mut sys = checked
        .cooked
        .ok_or_else(|| format!("{} has errors", path.display()))?;
    for e in &mut sys.emitters {
        for r in &mut e.renderers {
            match r {
                CookedRenderer::Sprite(s) => s.ray_tracing &= ray_tracing,
                CookedRenderer::Ribbon(s) => s.ray_tracing &= ray_tracing,
                _ => {}
            }
        }
    }
    Ok(sys)
}

#[derive(Default)]
struct Samples {
    spans: BTreeMap<String, Vec<f64>>,
    frame_ms: Vec<f64>,
    /// Captured frames (padded rows).
    images: Vec<Vec<u8>>,
}

#[derive(Resource)]
struct Run {
    o: Options,
    target: Handle<Image>,
    frame: u32,
    requested: bool,
    last: Option<Instant>,
    samples: Arc<Mutex<Samples>>,
}

fn main() -> ExitCode {
    let o = match parse() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("fxsolari: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let rt = o.mode == Mode::On;
    let extra = match o.effect.as_deref().map(|p| load_path(p, rt)).transpose() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("fxsolari: {e}");
            return ExitCode::from(1);
        }
    };
    let (fire, sparks) = match (load("fire_rt.fx.ron", rt), load("sparks_rt.fx.ron", rt)) {
        (Ok(f), Ok(s)) => (f, s),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("fxsolari: {e}");
            return ExitCode::from(1);
        }
    };

    let samples = Arc::new(Mutex::new(Samples::default()));
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::winit::WinitPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        FxPlugin {
            deterministic: !o.bench,
        },
    ));
    if o.mode != Mode::Raster {
        app.add_plugins(SolariPlugins);
    }
    if matches!(o.mode, Mode::Off | Mode::On) {
        app.add_plugins(FxSolariPlugin);
    }
    if o.bench {
        app.add_plugins(RenderDiagnosticsPlugin);
    }
    app.insert_resource(FxClock::manual(1.0 / 60.0))
        .insert_resource(Run {
            o: o.clone(),
            target: Handle::default(),
            frame: 0,
            requested: false,
            last: None,
            samples: samples.clone(),
        })
        .add_systems(
            Startup,
            move |commands: Commands,
                  run: ResMut<Run>,
                  images: ResMut<Assets<Image>>,
                  meshes: ResMut<Assets<Mesh>>,
                  materials: ResMut<Assets<StandardMaterial>>| {
                setup(commands, run, images, meshes, materials, &fire, &sparks, extra.as_ref())
            },
        )
        .add_systems(Update, drive);
    app.run();

    let s = samples.lock().unwrap();
    if o.bench {
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        println!("mode {:?}, {}x{}, {} copies, {} frames", o.mode, o.width, o.height, o.copies, s.frame_ms.len());
        println!("frame_cpu_ms {:.3}", median(&s.frame_ms));
        for (span, v) in &s.spans {
            println!("{span} {:.4}", median(v));
        }
        ExitCode::SUCCESS
    } else {
        if s.images.is_empty() {
            eprintln!("fxsolari: no image captured");
            return ExitCode::from(1);
        }
        let frames: Vec<Vec<u8>> = s
            .images
            .iter()
            .take(o.accumulate as usize)
            .map(|d| unpad(d, o.width, o.height))
            .collect();
        match write_png(&o.out, o.width, o.height, &average(&frames)) {
            Ok(()) => {
                println!("{}", o.out.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("fxsolari: {e}");
                ExitCode::from(1)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn setup(
    mut commands: Commands,
    mut run: ResMut<Run>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    fire: &CookedSystem,
    sparks: &CookedSystem,
    extra: Option<&CookedSystem>,
) {
    let o = run.o.clone();
    let solari = o.mode != Mode::Raster;
    let mut img = Image::new_target_texture(o.width, o.height, TextureFormat::Rgba8UnormSrgb, None);
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(img);
    run.target = target.clone();

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::from(target),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        Exposure { ev100: 6.0 },
        Transform::from_xyz(0.3, 1.5, 4.6).looking_at(Vec3::new(0.2, 0.9, -0.6), Vec3::Y),
    ));
    if solari {
        camera.insert((
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
            SolariLighting {
                // Solari's default (no ReSTIR); FXSOLARI_RESTIR=1 turns it on.
                restir: std::env::var_os("FXSOLARI_RESTIR").is_some(),
                ..default()
            },
        ));
    }

    let mut surface = |commands: &mut Commands, mesh: Mesh, material: StandardMaterial, t: Transform| {
        let mesh = meshes.add(mesh.with_generated_tangents().unwrap());
        let mut e = commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(materials.add(material)), t));
        if solari {
            e.insert(RaytracingMesh3d(mesh));
        }
    };
    let diffuse = |c: Color| StandardMaterial {
        base_color: c,
        perceptual_roughness: 0.9,
        ..default()
    };
    let span = 2 * o.copies as i32 - 1;
    let width = 7.0 + 2.5 * (o.copies - 1) as f32;
    // Floor, back wall (behind the fire), left wall.
    surface(
        &mut commands,
        Cuboid::new(width, 0.1, 8.0).mesh().build(),
        diffuse(Color::srgb(0.45, 0.45, 0.47)),
        Transform::from_xyz(0.0, -0.05, 0.5),
    );
    surface(
        &mut commands,
        Cuboid::new(width, 3.2, 0.1).mesh().build(),
        diffuse(Color::srgb(0.85, 0.85, 0.85)),
        Transform::from_xyz(0.0, 1.6, -1.65),
    );
    surface(
        &mut commands,
        Cuboid::new(0.1, 3.2, 8.0).mesh().build(),
        diffuse(Color::srgb(0.6, 0.62, 0.7)),
        Transform::from_xyz(-width / 2.0, 1.6, 0.5),
    );
    // The mirror, on the back wall right of the fire, turned a little to
    // show the sparks behind the camera.
    surface(
        &mut commands,
        Cuboid::new(1.5, 1.9, 0.04).mesh().build(),
        StandardMaterial {
            base_color: Color::srgb(0.95, 0.95, 0.95),
            metallic: 1.0,
            perceptual_roughness: 0.0,
            ..default()
        },
        Transform::from_xyz(1.35, 1.3, -1.57).with_rotation(Quat::from_rotation_y(-0.12)),
    );
    // A dim ceiling light so the room reads.
    commands.spawn((
        PointLight {
            intensity: 3000.0,
            range: 30.0,
            shadow_maps_enabled: !solari,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 2.5),
    ));
    let _ = span;
    if let Some(extra) = extra {
        // `--copies` side by side (bench).
        for c in 0..o.copies {
            let dx = 2.5 * c as f32 - 1.25 * (o.copies - 1) as f32;
            commands.spawn((ParticleEffect::new(extra.clone(), 1 + c), Transform::from_xyz(dx, 0.0, -0.4)));
        }
        // Shots: one more behind the camera, turned around, seen in the mirror.
        if !o.bench {
            commands.spawn((
                ParticleEffect::new(extra.clone(), 2),
                Transform::from_xyz(2.2, 0.0, 6.0).with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
            ));
        }
        return;
    }
    for c in 0..o.copies {
        let dx = 2.5 * c as f32 - 1.25 * (o.copies - 1) as f32;
        commands.spawn((
            ParticleEffect::new(fire.clone(), 1 + c),
            Transform::from_xyz(-0.9 + dx, 0.0, -1.0),
        ));
        commands.spawn((
            ParticleEffect::new(sparks.clone(), 101 + c),
            // Behind the camera, spraying left.
            Transform::from_xyz(2.2 + dx, 0.0, 6.0).with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
        ));
    }
}

fn drive(
    mut commands: Commands,
    mut run: ResMut<Run>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    store: Option<Res<DiagnosticsStore>>,
    mut exit: MessageWriter<AppExit>,
) {
    if !ready.get() {
        return;
    }
    run.frame += 1;
    let frames = run.o.frames;
    if run.o.bench {
        clock.request_steps(1);
        // First 120 frames: steady state and settled timings.
        let now = Instant::now();
        let last = run.last.replace(now);
        if run.frame <= 120 {
            return;
        }
        let mut s = run.samples.lock().unwrap();
        if let Some(t0) = last {
            s.frame_ms.push((now - t0).as_secs_f64() * 1000.0);
        }
        if let Some(store) = store {
            for d in store.iter() {
                let Some(span) = d.path().as_str().strip_suffix("/elapsed_gpu") else {
                    continue;
                };
                if let Some(v) = d.measurement().map(|m| m.value) {
                    s.spans.entry(span.to_string()).or_default().push(v);
                }
            }
        }
        drop(s);
        if run.frame >= 120 + frames {
            exit.write(AppExit::Success);
        }
        return;
    }
    // Shot: simulate `frames` steps, then pause the particles and read back
    // `accumulate` frames (the readback entity reads every frame).
    if run.frame < frames {
        clock.request_steps(1);
        return;
    }
    if !run.requested {
        run.requested = true;
        let samples = run.samples.clone();
        commands
            .spawn(Readback::texture(run.target.clone()))
            .observe(move |ev: On<ReadbackComplete>| {
                samples.lock().unwrap().images.push(ev.data.clone());
            });
    }
    if run.samples.lock().unwrap().images.len() >= run.o.accumulate as usize {
        exit.write(AppExit::Success);
    }
}

/// Mean of sRGB8 frames, averaged in linear light.
fn average(frames: &[Vec<u8>]) -> Vec<u8> {
    let to_linear = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let to_srgb = |l: f32| {
        let c = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (c.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    let lut: Vec<f32> = (0..=255u8).map(to_linear).collect();
    let n = frames.len() as f32;
    (0..frames[0].len())
        .map(|i| {
            if i % 4 == 3 {
                return 255;
            }
            to_srgb(frames.iter().map(|f| lut[f[i] as usize]).sum::<f32>() / n)
        })
        .collect()
}

/// Removes the 256-byte row padding of a texture readback.
fn unpad(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4) as usize;
    let padded = row.div_ceil(256) * 256;
    if data.len() == row * h as usize {
        return data.to_vec();
    }
    (0..h as usize)
        .flat_map(|y| data[y * padded..y * padded + row].iter().copied())
        .collect()
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())
}
