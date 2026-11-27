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
//!           [--camera X,Y,Z] [--target X,Y,Z] [--fov DEG] [--ground Y] [--no-mp4] [--test-scene]
//!           [--orbit RADIUS,SECONDS] [--fly VX,VY,VZ] [--look default|ue] [--exposure EV] [--bloom I] [--floor grid|proto]
//! ```
//!
//! `--look ue` matches Unreal's default look for side-by-side comparisons
//! with Unreal footage: ACES filmic tonemapping (Unreal's tone curve is ACES
//! based), bloom, and a dark floor with a 1 m grid (like the editor's
//! showcase floors). `--floor proto` swaps it for the blue-grey 2 m prototype grid of the
//! arrow trail demo map. `--exposure` shifts everything by stops (Unreal's
//! exposure compensation), `--bloom` sets the bloom intensity.
//!
//! `--fly` moves the effect in a straight line, turned to face its velocity
//! (local +X forward, Unreal's forward axis): projectile trails.
//!
//! `--also FILE@X,Y,Z` (repeatable) adds more effects to the scene at fixed
//! positions: a lineup of a whole pack, as in a marketplace showcase. `--at`
//! places the main effect.
//!
//! `--orbit` moves the effect on a horizontal circle around the origin (one
//! turn per SECONDS of simulated time), so effects reading
//! Engine.Owner.Velocity (Spawn Per Unit, Inherit Velocity) see it move.
//!
//! A `.prefab.ron` (from `fximport blueprint`) can stand in for any effect
//! file (main or `--also`): its effects, lights and animation play, Bevy
//! time (what materials read) then follows the frames. `--look-through NODE`
//! puts the camera at a prefab camera node; `--play-timeline NAME[:RATE]`
//! (repeatable) plays a prefab timeline from the first frame.
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
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::bloom::Bloom;
use bevy::render::view::{ColorGrading, ColorGradingGlobal};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use fx_render::{FxClock, FxPlugin, FxPrefab, FxPrefabCamera, FxPrefabInstance, FxPrefabNode, FxPrefabPlayer, FxReady, ParticleAttributes, ParticleComponent, ParticleEffect};

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
    /// Moves the effect on a horizontal circle: (radius m, seconds per turn).
    orbit: Option<(f32, f32)>,
    /// Flies the effect in a straight line at this velocity (m/s), turned to
    /// face it (local +X forward), passing the origin halfway through.
    fly: Option<Vec3>,
    /// Look through this prefab camera node (`--look-through`).
    look_through: Option<String>,
    /// Timelines to play from the first frame (`--play-timeline NAME[:RATE]`).
    play: Vec<(String, Option<f32>)>,
    /// More effects drawn in the same scene at fixed positions (`--also`).
    also: Vec<(PathBuf, Vec3)>,
    /// Where the main effect sits (`--at`), added to `--orbit` and `--fly`.
    at: Vec3,
    /// Vertical field of view in degrees (Bevy's default 45; Unreal's editor
    /// viewport is 90 horizontal).
    fov: f32,
    /// Height of the ground plane (metres).
    ground: f32,
    /// `default` (Bevy's look) or `ue` (Unreal-like: ACES tonemap, bloom,
    /// dark grid floor), for fair comparisons with Unreal footage.
    look: String,
    /// `--look ue` floor: `grid` (dark, 1 m cells, a grid wall behind) or
    /// `proto` (blue-grey prototype grid: 2 m cells of 4 x 4, no wall; the
    /// arrow trail demo map).
    floor: String,
    /// Exposure in stops (ColorGrading), applied to everything drawn.
    exposure: f32,
    /// Bloom intensity (`--look ue`; 0 turns it off).
    bloom: f32,
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
        orbit: None,
        fly: None,
        look_through: None,
        play: Vec::new(),
        also: Vec::new(),
        at: Vec3::ZERO,
        fov: 45.0,
        ground: 0.0,
        look: "default".into(),
        floor: "grid".into(),
        exposure: 0.0,
        bloom: 0.2,
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
            "--ground" => o.ground = val()?.parse::<f32>().map_err(|e| format!("--ground: {e}"))?,
            "--fov" => o.fov = val()?.parse::<f32>().map_err(|e| format!("--fov: {e}"))?,
            "--target" => o.target = parse_vec3(&val()?)?,
            "--look" => o.look = val()?,
            "--floor" => o.floor = val()?,
            "--exposure" => o.exposure = val()?.parse::<f32>().map_err(|e| format!("--exposure: {e}"))?,
            "--bloom" => o.bloom = val()?.parse::<f32>().map_err(|e| format!("--bloom: {e}"))?,
            "--no-mp4" => o.mp4 = false,
            "--test-scene" => o.test_scene = true,
            "--at" => o.at = parse_vec3(&val()?)?,
            "--also" => {
                let v = val()?;
                let (f, p) = v.rsplit_once('@').ok_or("--also is FILE@X,Y,Z")?;
                o.also.push((PathBuf::from(f), parse_vec3(p)?));
            }
            "--look-through" => o.look_through = Some(val()?),
            "--play-timeline" => {
                let v = val()?;
                o.play.push(match v.split_once(':') {
                    Some((n, r)) => (n.to_string(), Some(r.trim().parse::<f32>().map_err(|e| format!("--play-timeline: {e}"))?)),
                    None => (v, None),
                });
            }
            "--fly" => o.fly = Some(parse_vec3(&val()?)?),
            "--orbit" => {
                let v = val()?;
                let (r, t) = v.split_once(',').ok_or("--orbit is RADIUS,SECONDS")?;
                let r: f32 = r.trim().parse().map_err(|e| format!("--orbit: {e}"))?;
                let t: f32 = t.trim().parse().map_err(|e| format!("--orbit: {e}"))?;
                if !(t > 0.0) {
                    return Err("--orbit: SECONDS must be more than 0".into());
                }
                o.orbit = Some((r, t));
            }
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
    if !["default", "ue"].contains(&o.look.as_str()) {
        return Err("--look is default or ue".into());
    }
    let last = o.times.iter().cloned().fold(0.0, f32::max);
    o.duration = o.duration.max(last);
    Ok(o)
}

const USAGE: &str = "usage: fxpreview <effect.fx.ron> [--out DIR] [--times 0,0.5,1] [--duration S] [--fps N] [--size WxH] [--seed N] [--backdrop dark|grey] [--camera X,Y,Z] [--target X,Y,Z] [--fov DEG] [--ground Y] [--look default|ue] [--exposure EV] [--bloom I] [--floor grid|proto] [--no-mp4] [--test-scene] [--orbit RADIUS,SECONDS] [--fly VX,VY,VZ] [--at X,Y,Z] [--also FILE@X,Y,Z ...] [--look-through NODE] [--play-timeline NAME[:RATE]]";

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
    // Textures and materials resolve next to each effect file: with `--also`,
    // from the folder holding them all.
    let files: Vec<&Path> = std::iter::once(o.effect.as_path()).chain(o.also.iter().map(|(f, _)| f.as_path())).collect();
    let root = asset_root(&files);
    let mut effects = Vec::new();
    let mut prefabs = Vec::new();
    for (k, f) in files.iter().enumerate() {
        if f.to_string_lossy().ends_with(".prefab.ron") {
            // Loaded by the asset server (its effects resolve beside it).
            let full = std::fs::canonicalize(f).unwrap_or_else(|_| f.to_path_buf());
            let rel = full.strip_prefix(&root).map(|p| p.to_path_buf()).unwrap_or(full.clone());
            let at = if k == 0 { None } else { Some(o.also[k - 1].1) };
            prefabs.push((rel.to_string_lossy().into_owned(), at, o.seed.wrapping_add(1000 * k as u32)));
            continue;
        }
        match cook(f, &root) {
            Ok(system) => {
                let at = if k == 0 { None } else { Some(o.also[k - 1].1) };
                effects.push((ParticleEffect::new(system, o.seed.wrapping_add(k as u32)), at));
            }
            Err(e) => {
                eprintln!("fxpreview: {e}");
                return ExitCode::from(1);
            }
        }
    }

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
                file_path: root.to_string_lossy().into_owned(),
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::winit::WinitPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        FxPlugin {
            deterministic: true,
        },
    ))
    .insert_resource(FxClock::manual(1.0 / SIM_HZ as f32))
    .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(Duration::ZERO))
    .insert_resource(Preview {
        o: o.clone(),
        effects,
        prefab_time: !prefabs.is_empty(),
        prefabs,
        image: Handle::default(),
        frames: frames.clone(),
        n_frames,
        phase: Phase::Warmup(0),
        result: result.clone(),
    })
    .add_systems(Startup, setup)
    .add_systems(Update, drive)
    .add_systems(PostUpdate, show_particle_components.after(fx_render::apply_particle_components));
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
    /// The effects to spawn: the main one (None: at `effect_transform`), then `--also`.
    effects: Vec<(ParticleEffect, Option<Vec3>)>,
    /// Prefabs to spawn (asset path, `--also` position, seed).
    prefabs: Vec<(String, Option<Vec3>, u32)>,
    /// Prefabs were given: Bevy time (what materials read) follows the frames.
    prefab_time: bool,
    image: Handle<Image>,
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
    n_frames: usize,
    phase: Phase,
    result: Arc<Mutex<Option<Result<String, String>>>>,
}

fn setup(
    mut commands: Commands,
    assets: Res<AssetServer>,
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
    let ue = o.look == "ue";
    let proto = ue && o.floor == "proto";
    // `--floor proto` brightness: the floor (and backdrop) shown the same
    // whatever `--exposure`, which then only sets the effects against it.
    let proto_k = PROTO_GAIN * 2f32.powf(-o.exposure);
    let clear = if proto {
        Color::LinearRgba(Color::srgb_u8(16, 22, 32).to_linear() * proto_k)
    } else if ue {
        Color::srgb(0.008, 0.009, 0.012)
    } else {
        clear
    };
    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(clear),
            ..default()
        },
        RenderTarget::from(image),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        // Gives `distortion` sprites (heat haze) a copy of the scene to refract.
        bevy::pbr::ScreenSpaceTransmission::default(),
        Projection::Perspective(PerspectiveProjection { fov: o.fov.to_radians(), ..default() }),
        Transform::from_translation(o.camera).looking_at(o.target, Vec3::Y),
    ));
    if o.exposure != 0.0 {
        camera.insert(ColorGrading {
            global: ColorGradingGlobal {
                exposure: o.exposure,
                ..default()
            },
            ..default()
        });
    }
    if ue {
        // Unreal's filmic tone curve is fitted to ACES; its default bloom is
        // a wide, soft glow around anything brighter than white.
        camera.insert(Tonemapping::AcesFitted);
        if o.bloom > 0.0 {
            camera.insert(Bloom {
                intensity: o.bloom,
                ..Bloom::NATURAL
            });
        }
    }
    if proto {
        // The demo map's height fog: the floor fades to the backdrop with distance.
        camera.insert(DistanceFog {
            color: clear,
            falloff: FogFalloff::Linear { start: PROTO_FOG.0, end: PROTO_FOG.1 },
            ..default()
        });
    }
    let floor = if proto {
        // Unlit: the demo map's grid reads as a flat blue-grey under its
        // sky light and fog, whatever the view angle.
        StandardMaterial {
            base_color_texture: Some(images.add(proto_grid_image())),
            base_color: Color::linear_rgb(proto_k, proto_k, proto_k),
            uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(PROTO_FLOOR_SIZE / PROTO_CELL)),
            unlit: true,
            ..default()
        }
    } else if ue {
        StandardMaterial {
            base_color_texture: Some(images.add(grid_image())),
            uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(FLOOR_SIZE)),
            perceptual_roughness: 0.85,
            reflectance: 0.3,
            ..default()
        }
    } else {
        StandardMaterial {
            base_color: ground,
            perceptual_roughness: 0.9,
            ..default()
        }
    };
    let size = if proto {
        PROTO_FLOOR_SIZE
    } else if ue {
        FLOOR_SIZE
    } else {
        30.0
    };
    let floor = materials.add(floor);
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
        MeshMaterial3d(floor.clone()),
        Transform::from_xyz(0.0, o.ground, 0.0),
    ));
    if ue && !proto {
        // A grid wall far behind, as in showcase levels: something for heat
        // haze to bend and for bloom to glow against.
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(FLOOR_SIZE * 0.5)).mesh())),
            MeshMaterial3d(floor),
            Transform::from_xyz(0.0, o.ground, -WALL_DISTANCE),
        ));
    }
    commands.spawn((
        DirectionalLight {
            illuminance: if ue { 800.0 } else { 3000.0 },
            ..default()
        },
        Transform::from_xyz(3.0, 6.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: if ue { 700.0 } else { 300.0 },
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
    for (path, at, seed) in std::mem::take(&mut preview.prefabs) {
        let p = at.unwrap_or(o.at);
        commands.spawn((FxPrefabInstance { prefab: assets.load(path), seed }, PreviewEffect(p), offset(effect_transform(&o, 0.0), p)));
    }
    for (effect, at) in std::mem::take(&mut preview.effects) {
        match at {
            None => commands.spawn((effect, PreviewEffect(o.at), offset(effect_transform(&o, 0.0), o.at))),
            Some(p) => commands.spawn((effect, PreviewEffect(p), offset(effect_transform(&o, 0.0), p))),
        };
    }
}

/// Side of the `--look ue` floor (metres): one grid cell per metre.
const FLOOR_SIZE: f32 = 80.0;
/// How far behind the origin the `--look ue` grid wall stands (metres).
const WALL_DISTANCE: f32 = 12.0;

/// One 1 m cell of the `--look ue` floor: dark tile, light border lines,
/// with a full mip chain (box filtered) so the grid does not shimmer or
/// moiré towards the horizon.
fn grid_image() -> Image {
    const N: usize = 256;
    let tile = [0.15f32, 0.155, 0.17];
    let line = [0.7f32, 0.72, 0.78];
    mipped_tile(N, |x, y| {
        // Distance to the nearest cell edge in texels; ~5 texel (2 cm) lines.
        let d = x.min(N as f32 - x).min(y).min(N as f32 - y);
        let w = (1.0 - (d - 2.5).max(0.0)).clamp(0.0, 1.0);
        [0, 1, 2].map(|k| srgb_to_linear(tile[k]) * (1.0 - w) + srgb_to_linear(line[k]) * w)
    })
}

/// Side of the `--floor proto` floor (metres) and of one major cell.
const PROTO_FLOOR_SIZE: f32 = 160.0;
const PROTO_CELL: f32 = 2.0;
/// Linear gain of the `--floor proto` floor before ACES, matched to the
/// listing shots (ACES darkens dark values).
const PROTO_GAIN: f32 = 3.0;
/// Start and end distance (metres) of the `--floor proto` fog.
const PROTO_FOG: (f32, f32) = (8.0, 45.0);

/// One 2 m cell of the `--floor proto` floor, after the arrow trail demo's
/// prototype grid (MI_PrototypeGrid: 200 cm cells): blue-grey tile, 4 x 4
/// minor cells with thin lines, lighter major lines. Colours matched to
/// the listing shots (the map's sky light and fog tint its grey grid blue).
fn proto_grid_image() -> Image {
    const N: usize = 512;
    let tile = [40.0f32 / 255.0, 47.0 / 255.0, 58.0 / 255.0];
    let minor = [62.0f32 / 255.0, 72.0 / 255.0, 88.0 / 255.0];
    let major = [84.0f32 / 255.0, 96.0 / 255.0, 114.0 / 255.0];
    let sub = N as f32 / 4.0;
    mipped_tile(N, |x, y| {
        let edge = |v: f32, n: f32| v.min(n - v);
        let d_major = edge(x, N as f32).min(edge(y, N as f32));
        let d_minor = edge(x % sub, sub).min(edge(y % sub, sub));
        // ~4 cm major and ~1.5 cm minor lines.
        let wm = (1.0 - (d_major - 4.5).max(0.0)).clamp(0.0, 1.0);
        let wn = (1.0 - (d_minor - 1.5).max(0.0)).clamp(0.0, 1.0);
        [0, 1, 2].map(|k| {
            let c = srgb_to_linear(tile[k]) * (1.0 - wn) + srgb_to_linear(minor[k]) * wn;
            c * (1.0 - wm) + srgb_to_linear(major[k]) * wm
        })
    })
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// A repeating N x N sRGB texture from linear texel colours at texel
/// centres, with a full box-filtered mip chain.
#[allow(non_snake_case)]
fn mipped_tile(n0: usize, texel: impl Fn(f32, f32) -> [f32; 3]) -> Image {
    use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
    use bevy::render::render_resource::{Extent3d, TextureDimension};
    let N = n0;
    // Linear values, encoded to sRGB per mip.
    let mut level: Vec<[f32; 3]> = (0..N * N).map(|i| texel((i % N) as f32 + 0.5, (i / N) as f32 + 0.5)).collect();
    let enc = |c: f32| {
        let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let mut data = Vec::new();
    let mut n = N;
    let mut mips = 0;
    loop {
        for p in &level {
            data.extend([enc(p[0]), enc(p[1]), enc(p[2]), 255]);
        }
        mips += 1;
        if n == 1 {
            break;
        }
        let h = n / 2;
        level = (0..h * h)
            .map(|i| {
                let (x, y) = (i % h * 2, i / h * 2);
                let at = |dx: usize, dy: usize| level[(y + dy) * n + x + dx];
                [0, 1, 2].map(|k| (at(0, 0)[k] + at(1, 0)[k] + at(0, 1)[k] + at(1, 1)[k]) * 0.25)
            })
            .collect();
        n = h;
    }
    let mut img = Image::new(
        Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data[..N * N * 4].to_vec(),
        TextureFormat::Rgba8UnormSrgb,
        default(),
    );
    // Every mip, largest first (Image::new checks for the first only).
    img.data = Some(data);
    img.texture_descriptor.mip_level_count = mips;
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..default()
    });
    img
}

/// Draws each entity a Component renderer drives as a small cube in its
/// particle's colour, so previews show which particles drive entities.
fn show_particle_components(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cube: Local<Option<Handle<Mesh>>>,
    added: Query<(Entity, &ParticleAttributes), (Added<ParticleComponent>, Without<Mesh3d>)>,
    changed: Query<(&ParticleAttributes, &MeshMaterial3d<StandardMaterial>), Changed<ParticleAttributes>>,
) {
    let cube = cube.get_or_insert_with(|| meshes.add(Cuboid::new(0.12, 0.12, 0.12))).clone();
    let color = |a: &ParticleAttributes| Color::linear_rgb(a.color.x, a.color.y, a.color.z);
    for (e, a) in &added {
        let material = materials.add(StandardMaterial {
            base_color: color(a),
            perceptual_roughness: 0.5,
            ..default()
        });
        commands.entity(e).insert((Mesh3d(cube.clone()), MeshMaterial3d(material)));
    }
    for (a, m) in &changed {
        if let Some(mut mat) = materials.get_mut(&m.0) {
            mat.base_color = color(a);
        }
    }
}

/// An effect fxpreview renders, at its `--also` offset (zero for the main one).
#[derive(Component)]
struct PreviewEffect(Vec3);

/// `tr` moved by `p` (an `--also` position): extra effects move with `--orbit`
/// and `--fly` too, keeping their place in the lineup.
fn offset(tr: Transform, p: Vec3) -> Transform {
    Transform { translation: tr.translation + p, ..tr }
}

/// Where `--orbit` or `--fly` puts the effect at simulated time `t`.
fn effect_transform(o: &Options, t: f32) -> Transform {
    if let Some(v) = o.fly {
        let start = -v * o.duration * 0.5;
        let dir = v.normalize_or(Vec3::X);
        return Transform::from_translation(start + v * t)
            .with_rotation(Quat::from_rotation_arc(Vec3::X, dir));
    }
    match o.orbit {
        Some((r, period)) => {
            let a = std::f32::consts::TAU * t / period;
            Transform::from_xyz(r * a.cos(), 0.0, r * a.sin())
        }
        None => Transform::IDENTITY,
    }
}

fn drive(
    mut commands: Commands,
    mut preview: ResMut<Preview>,
    mut effect: Query<(&mut Transform, &PreviewEffect)>,
    pending: Query<(), (With<FxPrefabInstance>, Without<FxPrefabPlayer>)>,
    mut players: Query<&mut FxPrefabPlayer>,
    prefab_assets: Res<Assets<FxPrefab>>,
    mut camera: Query<(&mut Transform, &mut Projection), (With<Camera3d>, Without<PreviewEffect>, Without<FxPrefabNode>)>,
    prefab_cameras: Query<(&FxPrefabNode, &GlobalTransform, &FxPrefabCamera)>,
    mut strategy: ResMut<bevy::time::TimeUpdateStrategy>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    mut exit: MessageWriter<AppExit>,
) {
    let steps_per_frame = SIM_HZ / preview.o.fps;
    // `--look-through`: the camera at a prefab camera node (Unreal cameras look along their +X).
    if let Some(name) = &preview.o.look_through {
        if let Some((_, g, c)) = prefab_cameras.iter().find(|(n, _, _)| &n.name == name) {
            for (mut tf, mut proj) in &mut camera {
                let (_, r, t) = g.to_scale_rotation_translation();
                *tf = Transform { translation: t, rotation: r * Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2), scale: Vec3::ONE };
                if let Projection::Perspective(p) = &mut *proj {
                    p.fov = c.fov.to_radians();
                }
            }
        }
    }
    match preview.phase {
        Phase::Warmup(n) => {
            if !ready.get() || !pending.is_empty() {
                preview.phase = Phase::Warmup(0);
            } else if n < 10 {
                preview.phase = Phase::Warmup(n + 1);
            } else {
                // Frame 0 is t = 0: no steps this frame.
                for mut pl in &mut players {
                    let Some(p) = prefab_assets.get(pl.prefab()) else { continue };
                    for (name, rate) in &preview.o.play {
                        pl.play(&p.prefab, name, true);
                        if let Some(r) = rate {
                            pl.set_rate(&p.prefab, name, *r);
                        }
                    }
                }
                if preview.prefab_time {
                    *strategy = bevy::time::TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(1.0 / preview.o.fps as f64));
                }
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
                // The effect's position at the end of this frame's steps.
                let t = (k * steps_per_frame as usize) as f32 / SIM_HZ as f32;
                for (mut tr, p) in &mut effect {
                    *tr = offset(effect_transform(&preview.o, t), p.0);
                }
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
        .map(|n| n.trim_end_matches(".fx.ron").trim_end_matches(".prefab.ron").to_string())
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

/// The deepest folder holding every effect file, absolute (with one effect,
/// its folder: its `texture` and `material` paths are relative to it).
fn asset_root(effects: &[&Path]) -> PathBuf {
    let dir = |e: &Path| {
        let d = e.parent().unwrap_or(Path::new("."));
        let d = if d.as_os_str().is_empty() { Path::new(".") } else { d };
        std::fs::canonicalize(d).unwrap_or_else(|_| d.to_path_buf())
    };
    let mut root = dir(effects[0]);
    for e in &effects[1..] {
        let d = dir(e);
        while !d.starts_with(&root) {
            if !root.pop() {
                break;
            }
        }
    }
    root
}

/// Checks and cooks an effect file. When it is not directly in `root`
/// (`--also`), the paths of the assets `fximport convert` writes beside it
/// (`"<stem>_assets/..."`) are made relative to `root`.
fn cook(file: &Path, root: &Path) -> Result<fx_core::CookedSystem, String> {
    let mut text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let dir = std::fs::canonicalize(file.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))
        .map_err(|e| format!("{}: {e}", file.display()))?;
    if let Ok(rel) = dir.strip_prefix(root)
        && !rel.as_os_str().is_empty()
    {
        let stem = file.file_name().and_then(|n| n.to_str()).unwrap_or("").trim_end_matches(".fx.ron");
        text = text.replace(&format!("\"{stem}_assets/"), &format!("\"{}/{stem}_assets/", rel.display()));
    }
    let checked = fx_core::check_source(&file.display().to_string(), &text);
    for d in &checked.diagnostics {
        eprintln!("{d}");
    }
    checked.cooked.ok_or_else(|| format!("{} has errors; nothing rendered", file.display()))
}
