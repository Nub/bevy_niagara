//! `fxpreview`: renders an effect headless with the real renderer (fx_render
//! in Bevy's raster pipeline) to a PNG frame strip at fixed times and an MP4.
//!
//! Deterministic: fixed camera and backdrop, fixed seed, fixed 1/60 s
//! simulation steps driven frame by frame (never by wall-clock time), and
//! capture only starts once every pipeline has compiled.
//!
//! ```text
//! fxpreview <effect.fx.ron> [--out DIR] [--times 0,0.25,0.5,1,2] [--duration S]
//!           [--fps N] [--size WxH] [--seed N] [--backdrop dark|grey]
//!           [--camera X,Y,Z] [--target X,Y,Z] [--no-mp4] [--test-scene]
//! ```
//!
//! `--test-scene` adds an opaque pillar through the effect and a translucent
//! glass pane in front of part of it, to check depth and ordering against
//! scene geometry and other transparents.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use fx_render::{FxClock, FxPlugin, FxReady, ParticleEffect};

/// Simulation step: fixed, whatever the output frame rate.
const SIM_HZ: u32 = 60;

#[derive(Clone, Debug)]
struct Options {
    effect: PathBuf,
    out: PathBuf,
    times: Vec<f32>,
    duration: f32,
    fps: u32,
    width: u32,
    height: u32,
    seed: u32,
    backdrop: String,
    camera: Vec3,
    target: Vec3,
    mp4: bool,
    test_scene: bool,
}

fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse::<f32>().map_err(|e| format!("{s}: {e}")))
        .collect::<Result<_, _>>()?;
    match v[..] {
        [x, y, z] => Ok(Vec3::new(x, y, z)),
        _ => Err(format!("expected X,Y,Z, got {s}")),
    }
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        effect: PathBuf::new(),
        out: PathBuf::from("preview"),
        times: vec![0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        duration: 3.0,
        fps: 30,
        width: 480,
        height: 360,
        seed: 1,
        backdrop: "dark".into(),
        camera: Vec3::new(0.0, 1.6, 6.5),
        target: Vec3::new(0.0, 1.3, 0.0),
        mp4: true,
        test_scene: false,
    };
    let mut effect = None;
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--out" => o.out = PathBuf::from(val()?),
            "--times" => {
                o.times = val()?
                    .split(',')
                    .map(|t| t.trim().parse::<f32>().map_err(|e| format!("--times: {e}")))
                    .collect::<Result<_, _>>()?
            }
            "--duration" => o.duration = val()?.parse().map_err(|e| format!("--duration: {e}"))?,
            "--fps" => o.fps = val()?.parse().map_err(|e| format!("--fps: {e}"))?,
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--size is WxH")?;
                o.width = w.parse().map_err(|e| format!("--size: {e}"))?;
                o.height = h.parse().map_err(|e| format!("--size: {e}"))?;
            }
            "--seed" => o.seed = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--backdrop" => o.backdrop = val()?,
            "--camera" => o.camera = parse_vec3(&val()?)?,
            "--target" => o.target = parse_vec3(&val()?)?,
            "--no-mp4" => o.mp4 = false,
            "--test-scene" => o.test_scene = true,
            "-h" | "--help" => return Err(String::new()),
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => effect = Some(PathBuf::from(a)),
        }
    }
    o.effect = effect.ok_or("no effect file given")?;
    if !(1..=SIM_HZ).contains(&o.fps) || !SIM_HZ.is_multiple_of(o.fps) {
        return Err(format!("--fps must divide {SIM_HZ}"));
    }
    if !["dark", "grey"].contains(&o.backdrop.as_str()) {
        return Err("--backdrop is dark or grey".into());
    }
    let last = o.times.iter().cloned().fold(0.0, f32::max);
    o.duration = o.duration.max(last);
    Ok(o)
}

const USAGE: &str = "usage: fxpreview <effect.fx.ron> [--out DIR] [--times 0,0.5,1] [--duration S] [--fps N] [--size WxH] [--seed N] [--backdrop dark|grey] [--camera X,Y,Z] [--target X,Y,Z] [--no-mp4] [--test-scene]";

fn main() -> ExitCode {
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("fxpreview: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let checked = match fx_core::check_file(&o.effect) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fxpreview: {}: {e}", o.effect.display());
            return ExitCode::from(1);
        }
    };
    for d in &checked.diagnostics {
        eprintln!("{d}");
    }
    let Some(system) = checked.cooked else {
        eprintln!(
            "fxpreview: {} has errors; nothing rendered",
            o.effect.display()
        );
        return ExitCode::from(1);
    };

    let frames = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let result = Arc::new(Mutex::new(None));
    let n_frames = (o.duration * o.fps as f32).round() as usize + 1;
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(AssetPlugin {
                // Textures and materials resolve next to the effect file.
                file_path: asset_root(&o.effect),
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        FxPlugin {
            deterministic: true,
        },
    ))
    .insert_resource(FxClock::manual(1.0 / SIM_HZ as f32))
    .insert_resource(Preview {
        o: o.clone(),
        effect: Some(ParticleEffect::new(system, o.seed)),
        image: Handle::default(),
        frames: frames.clone(),
        n_frames,
        phase: Phase::Warmup(0),
        result: result.clone(),
    })
    .add_systems(Startup, setup)
    .add_systems(Update, drive);
    app.run();

    let result = result.lock().unwrap().take();
    match result {
        Some(Ok(msg)) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Some(Err(e)) => {
            eprintln!("fxpreview: {e}");
            ExitCode::from(1)
        }
        None => ExitCode::from(1),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Frames seen with everything ready.
    Warmup(u32),
    /// Frames requested so far.
    Capturing(usize),
    Draining,
}

#[derive(Resource)]
struct Preview {
    o: Options,
    effect: Option<ParticleEffect>,
    image: Handle<Image>,
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
    n_frames: usize,
    phase: Phase,
    result: Arc<Mutex<Option<Result<String, String>>>>,
}

fn setup(
    mut commands: Commands,
    mut preview: ResMut<Preview>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let o = preview.o.clone();
    let mut img = Image::new_target_texture(o.width, o.height, TextureFormat::Rgba8UnormSrgb, None);
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image = images.add(img);
    preview.image = image.clone();

    let (clear, ground) = match o.backdrop.as_str() {
        "grey" => (Color::srgb(0.32, 0.33, 0.35), Color::srgb(0.4, 0.4, 0.42)),
        _ => (
            Color::srgb(0.015, 0.017, 0.022),
            Color::srgb(0.06, 0.06, 0.065),
        ),
    };
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(clear),
            ..default()
        },
        RenderTarget::from(image),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        Transform::from_translation(o.camera).looking_at(o.target, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(30.0, 30.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: ground,
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 3000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 6.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 300.0,
        ..default()
    });
    if o.test_scene {
        // An opaque pillar through the effect's centre, and a glass pane
        // (another transparent) in front of its left half.
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.35, 3.0, 0.35))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.55, 0.55, 0.6),
                perceptual_roughness: 0.6,
                ..default()
            })),
            Transform::from_xyz(0.0, 1.5, 0.0),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(1.6, 2.2, 0.02))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.2, 0.5, 0.35, 0.45),
                alpha_mode: AlphaMode::Blend,
                perceptual_roughness: 0.2,
                ..default()
            })),
            Transform::from_xyz(-0.9, 1.2, 1.4),
        ));
    }
    let effect = preview.effect.take().unwrap();
    commands.spawn((effect, Transform::IDENTITY));
}

fn drive(
    mut commands: Commands,
    mut preview: ResMut<Preview>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    mut exit: MessageWriter<AppExit>,
) {
    let steps_per_frame = SIM_HZ / preview.o.fps;
    match preview.phase {
        Phase::Warmup(n) => {
            if !ready.get() {
                preview.phase = Phase::Warmup(0);
            } else if n < 10 {
                preview.phase = Phase::Warmup(n + 1);
            } else {
                // Frame 0 is t = 0: no steps this frame.
                let frames = preview.frames.clone();
                commands
                    .spawn(Readback::texture(preview.image.clone()))
                    .observe(move |ev: On<ReadbackComplete>| {
                        frames.lock().unwrap().push(ev.data.clone());
                    });
                preview.phase = Phase::Capturing(1);
            }
        }
        Phase::Capturing(k) => {
            if k < preview.n_frames {
                clock.request_steps(steps_per_frame);
                preview.phase = Phase::Capturing(k + 1);
            } else {
                preview.phase = Phase::Draining;
            }
        }
        Phase::Draining => {
            // Readbacks arrive in order a few frames late; extra ones past
            // the last requested frame are ignored.
            let got = preview.frames.lock().unwrap().len();
            if got >= preview.n_frames {
                let frames: Vec<Vec<u8>> =
                    preview.frames.lock().unwrap()[..preview.n_frames].to_vec();
                *preview.result.lock().unwrap() = Some(write_outputs(&preview.o, &frames));
                exit.write(AppExit::Success);
            }
        }
    }
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
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())
}

fn write_outputs(o: &Options, frames: &[Vec<u8>]) -> Result<String, String> {
    let (w, h) = (o.width, o.height);
    let frames: Vec<Vec<u8>> = frames.iter().map(|f| unpad(f, w, h)).collect();
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    let stem = o
        .effect
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".fx.ron").to_string())
        .unwrap_or_else(|| "effect".into());

    // Frame strip: the frames at the requested times, left to right, 4 px apart.
    let gap = 4u32;
    let picks: Vec<usize> = o
        .times
        .iter()
        .map(|t| ((t * o.fps as f32).round() as usize).min(frames.len() - 1))
        .collect();
    let sw = picks.len() as u32 * (w + gap) - gap;
    let mut strip = vec![255u8; (sw * h * 4) as usize];
    for (i, &f) in picks.iter().enumerate() {
        let x0 = i as u32 * (w + gap);
        for y in 0..h {
            let src = &frames[f][(y * w * 4) as usize..((y + 1) * w * 4) as usize];
            let d = ((y * sw + x0) * 4) as usize;
            strip[d..d + src.len()].copy_from_slice(src);
        }
    }
    let strip_path = o.out.join(format!("{stem}_strip.png"));
    write_png(&strip_path, sw, h, &strip)?;
    let mut msg = format!(
        "{}: {} frames at {} fps; strip at t = {:?} s",
        strip_path.display(),
        frames.len(),
        o.fps,
        o.times
    );

    if o.mp4 {
        let mp4 = o.out.join(format!("{stem}.mp4"));
        let mut child = Command::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
            ])
            .arg(format!("{w}x{h}"))
            .args(["-r", &o.fps.to_string(), "-i", "-"])
            .args([
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
            ])
            .arg(&mp4)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e} (is ffmpeg on PATH? pass --no-mp4 to skip)"))?;
        {
            let stdin = child.stdin.as_mut().unwrap();
            for f in &frames {
                stdin.write_all(f).map_err(|e| format!("ffmpeg: {e}"))?;
            }
        }
        let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
        if !status.success() {
            return Err(format!("ffmpeg failed ({status})"));
        }
        msg.push_str(&format!("\n{}", mp4.display()));
    }
    Ok(msg)
}

/// The folder holding the effect file, absolute: its `texture` and
/// `material` paths are relative to it.
fn asset_root(effect: &Path) -> String {
    let dir = effect.parent().unwrap_or(Path::new("."));
    let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
    std::fs::canonicalize(dir)
        .unwrap_or_else(|_| dir.to_path_buf())
        .to_string_lossy()
        .into_owned()
}
