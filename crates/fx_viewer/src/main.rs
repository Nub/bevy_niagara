//! `fxviewer`: interactive viewer for `.fx.ron` effects, drawn by the same
//! renderer as `fxpreview` (fx_render in Bevy's raster pipeline).
//!
//! ```text
//! fxviewer [effect.fx.ron] [--backdrop dark|grey|room] [--duration S] [--speed X]
//!          [--size WxH] [--seed N] [--paused] [--at S]
//! ```
//!
//! - Hot reload: the file is re-checked when it changes on disk. A valid edit
//!   replaces the effect and fast-forwards to the same time; errors are shown
//!   in the window and the last good version keeps running.
//! - Mouse: left drag orbits, right/middle drag pans, wheel zooms. Click or
//!   drag the timeline to scrub.
//! - Keys: Space play/pause, R restart, ←/→ scrub 0.1 s (Shift: 1 s), ↑/↓
//!   speed ×2/÷2, B backdrop, L loop, F reset camera, H help, F12 screenshot.
//!
//! Remote control (used by the `fx_mcp` server): the viewer writes its pid to
//! `<dir>/fxviewer.pid`, reads commands appended as lines to
//! `<dir>/fxviewer.cmd` (`open PATH`, `screenshot PATH`, `play`, `pause`,
//! `restart`, `seek S`, `speed X`, `backdrop NAME`, `quit`) and writes its
//! state to `<dir>/fxviewer.status.json` twice a second. `<dir>` is
//! `$FXVIEWER_DIR`, else `$XDG_RUNTIME_DIR`, else the temp dir.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::text::FontSize;
use bevy::window::{PrimaryWindow, WindowResolution};
use fx_core::CookedSystem;
use fx_render::{FxClock, FxPlugin, FxReady, FxStats, FxStatsPlugin, ParticleEffect};

const SIM_HZ: f32 = 60.0;
const BACKDROPS: [&str; 3] = ["dark", "grey", "room"];

// ------------------------------------------------------------------ control

fn control_dir() -> PathBuf {
    std::env::var_os("FXVIEWER_DIR")
        .or_else(|| std::env::var_os("XDG_RUNTIME_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

struct Control {
    dir: PathBuf,
    read: u64,
    partial: String,
}

impl Control {
    fn new() -> Self {
        let dir = control_dir();
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("fxviewer.cmd"), "");
        let _ = std::fs::write(dir.join("fxviewer.pid"), std::process::id().to_string());
        Control {
            dir,
            read: 0,
            partial: String::new(),
        }
    }

    /// Complete command lines appended since the last call.
    fn poll(&mut self) -> Vec<String> {
        let Ok(mut f) = std::fs::File::open(self.dir.join("fxviewer.cmd")) else {
            return Vec::new();
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < self.read {
            self.read = 0; // truncated by someone
        }
        if len == self.read {
            return Vec::new();
        }
        let mut buf = String::new();
        if f.seek(SeekFrom::Start(self.read)).is_err() || f.read_to_string(&mut buf).is_err() {
            return Vec::new();
        }
        self.read += buf.len() as u64;
        self.partial.push_str(&buf);
        let mut out = Vec::new();
        while let Some(i) = self.partial.find('\n') {
            let line: String = self.partial.drain(..=i).collect();
            let line = line.trim();
            if !line.is_empty() {
                out.push(line.to_string());
            }
        }
        out
    }
}

impl Drop for Control {
    fn drop(&mut self) {
        // Only remove the pid file if it is still ours.
        let pid = self.dir.join("fxviewer.pid");
        if std::fs::read_to_string(&pid).ok().as_deref() == Some(&std::process::id().to_string()) {
            let _ = std::fs::remove_file(pid);
        }
    }
}

// ------------------------------------------------------------------- state

#[derive(Resource)]
struct Viewer {
    path: Option<PathBuf>,
    mtime: Option<SystemTime>,
    check_timer: Timer,
    /// Last version that compiled.
    system: Option<Arc<CookedSystem>>,
    /// Diagnostics of the last check (errors and warnings).
    diagnostics: Vec<String>,
    has_errors: bool,
    reloads: u32,
    effect: Option<Entity>,
    seed: u32,
    /// Simulated seconds since the effect (re)started.
    time: f32,
    /// Where the simulation should be.
    target: f32,
    playing: bool,
    speed: f32,
    duration: f32,
    looping: bool,
    backdrop: usize,
    backdrop_dirty: bool,
    help: bool,
    status_timer: Timer,
    control: Control,
    screenshots: u32,
    scrubbing: bool,
}

/// Set to despawn the effect and spawn it again from `Viewer::system` at time 0.
#[derive(Resource, Default)]
struct Respawn(bool);

struct Options {
    path: Option<PathBuf>,
    backdrop: usize,
    duration: f32,
    speed: f32,
    width: u32,
    height: u32,
    seed: u32,
    paused: bool,
    at: f32,
}

const USAGE: &str = "usage: fxviewer [effect.fx.ron] [--backdrop dark|grey|room] [--duration S] [--speed X] [--size WxH] [--seed N] [--paused] [--at S]";

fn parse_args() -> Result<Options, String> {
    let mut o = Options {
        path: None,
        backdrop: 0,
        duration: 4.0,
        speed: 1.0,
        width: 1280,
        height: 720,
        seed: 1,
        paused: false,
        at: 0.0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--backdrop" => {
                let v = val()?;
                o.backdrop = BACKDROPS
                    .iter()
                    .position(|b| *b == v)
                    .ok_or("--backdrop is dark, grey or room")?;
            }
            "--duration" => o.duration = val()?.parse().map_err(|e| format!("--duration: {e}"))?,
            "--speed" => o.speed = val()?.parse().map_err(|e| format!("--speed: {e}"))?,
            "--seed" => o.seed = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--at" => o.at = val()?.parse().map_err(|e| format!("--at: {e}"))?,
            "--paused" => o.paused = true,
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--size is WxH")?;
                o.width = w.parse().map_err(|e| format!("--size: {e}"))?;
                o.height = h.parse().map_err(|e| format!("--size: {e}"))?;
            }
            "-h" | "--help" => return Err(String::new()),
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => o.path = Some(PathBuf::from(a)),
        }
    }
    o.duration = o.duration.max(0.1);
    Ok(o)
}

fn main() -> std::process::ExitCode {
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("fxviewer: {e}");
            }
            eprintln!("{USAGE}");
            return std::process::ExitCode::from(2);
        }
    };
    let path = o.path.as_ref().map(|p| std::path::absolute(p).unwrap_or(p.clone()));
    let viewer = Viewer {
        path: path.clone(),
        mtime: None,
        check_timer: Timer::from_seconds(0.2, TimerMode::Repeating),
        system: None,
        diagnostics: Vec::new(),
        has_errors: false,
        reloads: 0,
        effect: None,
        seed: o.seed,
        time: 0.0,
        target: o.at,
        playing: !o.paused,
        speed: o.speed,
        duration: o.duration,
        looping: true,
        backdrop: o.backdrop,
        backdrop_dirty: true,
        help: true,
        status_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        control: Control::new(),
        screenshots: 0,
        scrubbing: false,
    };
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    // Textures and materials resolve next to the first effect opened.
                    file_path: asset_root(path.as_deref()),
                    ..default()
                })
                .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "fxviewer".into(),
                    resolution: WindowResolution::new(o.width, o.height),
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
            FxPlugin::default(),
            FxStatsPlugin,
        ))
        .insert_resource(FxClock::manual(1.0 / SIM_HZ))
        .insert_resource(viewer)
        .insert_resource(Respawn(true))
        .insert_resource(OrbitCamera::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (remote_control, watch_file, keyboard, timeline_mouse),
                apply_effect,
                advance_time,
                (orbit_camera, apply_backdrop, update_overlay, write_status),
            )
                .chain(),
        )
        .run();
    std::process::ExitCode::SUCCESS
}

/// The folder holding the effect file: its `texture` and `material` paths are relative to it.
fn asset_root(effect: Option<&std::path::Path>) -> String {
    let dir = effect
        .and_then(|p| p.parent())
        .map(|d| d.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::canonicalize(&dir)
        .unwrap_or(dir)
        .to_string_lossy()
        .into_owned()
}

// ------------------------------------------------------------------- setup

#[derive(Component)]
struct ViewCamera;
#[derive(Component)]
struct BackdropPart;
#[derive(Component)]
struct StatsText;
#[derive(Component)]
struct HelpText;
#[derive(Component)]
struct ErrorPanel;
#[derive(Component)]
struct ErrorText;
#[derive(Component)]
struct Timeline;
#[derive(Component)]
struct TimelineFill;
#[derive(Component)]
struct TimelineLabel;

const HELP: &str = "Space play/pause   R restart   Left/Right scrub (Shift: 1 s)\nUp/Down speed   B backdrop   L loop   F reset camera\nLeft drag orbit   Right drag pan   Wheel zoom\nF12 screenshot   H hide help";

fn mono(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        ViewCamera,
        Transform::default(),
    ));
    // Stats overlay, top left.
    commands.spawn((
        Text::new(""),
        mono(15.0),
        TextColor(Color::srgb(0.85, 0.9, 0.95)),
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            left: px(12),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
        StatsText,
    ));
    // Help, top right.
    commands.spawn((
        Text::new(HELP),
        mono(13.0),
        TextColor(Color::srgb(0.75, 0.78, 0.82)),
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            right: px(12),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
        HelpText,
    ));
    // Validation errors, above the timeline.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(12),
                right: px(12),
                bottom: px(52),
                padding: UiRect::all(px(8)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.35, 0.02, 0.02, 0.88)),
            ErrorPanel,
        ))
        .with_child((
            Text::new(""),
            mono(14.0),
            TextColor(Color::srgb(1.0, 0.88, 0.85)),
            ErrorText,
        ));
    // Timeline, bottom.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(12),
                right: px(12),
                bottom: px(12),
                height: px(28),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.11, 0.13, 0.85)),
            Timeline,
        ))
        .with_children(|p| {
            p.spawn((
                Node {
                    width: percent(0),
                    height: percent(100),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.25, 0.55, 0.95, 0.8)),
                TimelineFill,
            ));
            p.spawn((
                Text::new(""),
                mono(14.0),
                TextColor(Color::WHITE),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(8),
                    top: px(5),
                    ..default()
                },
                TimelineLabel,
            ));
        });
}

// ---------------------------------------------------------------- loading

fn load(v: &mut Viewer, respawn: &mut Respawn) {
    let Some(path) = v.path.clone() else { return };
    v.mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    match fx_core::check_file(&path) {
        Ok(c) => {
            v.diagnostics = c.diagnostics.iter().map(|d| d.to_string()).collect();
            v.has_errors = !c.ok();
            if let Some(sys) = c.cooked {
                v.system = Some(Arc::new(sys));
                v.reloads += 1;
                // Same moment of the new version: restart and fast-forward.
                v.target = v.target.max(v.time);
                v.time = 0.0;
                respawn.0 = true;
            }
        }
        Err(e) => {
            v.diagnostics = vec![format!("{}: cannot read: {e}", path.display())];
            v.has_errors = true;
        }
    }
    for d in &v.diagnostics {
        info!("{d}");
    }
}

fn watch_file(mut v: ResMut<Viewer>, mut respawn: ResMut<Respawn>, time: Res<Time>) {
    if !v.check_timer.tick(time.delta()).just_finished() {
        return;
    }
    let Some(path) = v.path.clone() else { return };
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    if v.system.is_none() && v.diagnostics.is_empty() || mtime != v.mtime {
        load(&mut v, &mut respawn);
    }
}

fn apply_effect(
    mut commands: Commands,
    mut v: ResMut<Viewer>,
    mut respawn: ResMut<Respawn>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if !respawn.0 {
        return;
    }
    respawn.0 = false;
    if let Some(e) = v.effect.take() {
        commands.entity(e).despawn();
    }
    v.time = 0.0;
    if let Some(sys) = v.system.clone() {
        let e = commands
            .spawn((
                ParticleEffect {
                    system: sys,
                    seed: v.seed,
                    meshes: Vec::new(),
                },
                Transform::IDENTITY,
            ))
            .id();
        v.effect = Some(e);
    }
    if let Ok(mut w) = windows.single_mut() {
        let name = v
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "no effect".into());
        w.title = format!("fxviewer - {name}");
    }
}

/// Moves the simulation towards `target` in fixed steps, at most
/// `MAX_STEPS_PER_SUBMIT` a frame (fast-forward after a reload or scrub).
fn advance_time(
    mut v: ResMut<Viewer>,
    mut clock: ResMut<FxClock>,
    mut respawn: ResMut<Respawn>,
    ready: Res<FxReady>,
    time: Res<Time>,
) {
    if v.playing && !v.scrubbing {
        v.target += time.delta_secs() * v.speed;
    }
    if v.target >= v.duration {
        if v.looping && v.playing {
            v.target = 0.0;
            respawn.0 = true;
            return;
        }
        v.target = v.duration;
    }
    if v.target + 0.5 / SIM_HZ < v.time {
        // Backwards: restart and catch up.
        respawn.0 = true;
        return;
    }
    if v.effect.is_none() || !ready.get() {
        return;
    }
    let step = 1.0 / SIM_HZ;
    let want = ((v.target - v.time) / step + 1e-3).floor().max(0.0) as u32;
    let n = want.min(fx_gpu_max_steps());
    if n > 0 {
        clock.request_steps(n);
        v.time += n as f32 * step;
    }
}

fn fx_gpu_max_steps() -> u32 {
    8
}

// ------------------------------------------------------------------ input

fn keyboard(
    keys: Res<ButtonInput<KeyCode>>,
    mut v: ResMut<Viewer>,
    mut respawn: ResMut<Respawn>,
    mut orbit: ResMut<OrbitCamera>,
    mut commands: Commands,
) {
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    if keys.just_pressed(KeyCode::Space) {
        v.playing = !v.playing;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        v.target = 0.0;
        respawn.0 = true;
    }
    let jump = if shift { 1.0 } else { 0.1 };
    if keys.just_pressed(KeyCode::ArrowRight) {
        v.target = (v.target + jump).min(v.duration);
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        v.target = (v.target - jump).max(0.0);
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        v.speed = (v.speed * 2.0).min(8.0);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        v.speed = (v.speed * 0.5).max(1.0 / 16.0);
    }
    if keys.just_pressed(KeyCode::KeyB) {
        v.backdrop = (v.backdrop + 1) % BACKDROPS.len();
        v.backdrop_dirty = true;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        v.looping = !v.looping;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        *orbit = OrbitCamera::default();
    }
    if keys.just_pressed(KeyCode::KeyH) {
        v.help = !v.help;
    }
    if keys.just_pressed(KeyCode::F12) {
        v.screenshots += 1;
        let path = format!("fxviewer_{}.png", v.screenshots);
        info!("screenshot to {path}");
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}

fn timeline_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    bar: Query<(&ComputedNode, &UiGlobalTransform), With<Timeline>>,
    mut v: ResMut<Viewer>,
) {
    let (Ok(w), Ok((node, xf))) = (windows.single(), bar.single()) else {
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        v.scrubbing = false;
        return;
    }
    let Some(cursor) = w.physical_cursor_position() else {
        return;
    };
    let size = node.size();
    let center = xf.translation;
    let min = center - size / 2.0;
    let inside = cursor.x >= min.x
        && cursor.x <= min.x + size.x
        && cursor.y >= min.y
        && cursor.y <= min.y + size.y;
    if buttons.just_pressed(MouseButton::Left) && inside {
        v.scrubbing = true;
    }
    if v.scrubbing {
        let f = ((cursor.x - min.x) / size.x).clamp(0.0, 1.0);
        v.target = f * v.duration;
    }
}

#[derive(Resource)]
struct OrbitCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // Same view as fxpreview's default camera.
        let eye = Vec3::new(0.0, 1.6, 6.5);
        let focus = Vec3::new(0.0, 1.3, 0.0);
        let d = eye - focus;
        OrbitCamera {
            focus,
            yaw: d.x.atan2(d.z),
            pitch: (d.y / d.length()).asin(),
            distance: d.length(),
        }
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    v: Res<Viewer>,
    mut orbit: ResMut<OrbitCamera>,
    mut cam: Query<&mut Transform, With<ViewCamera>>,
) {
    let delta = motion.delta;
    if !v.scrubbing && buttons.pressed(MouseButton::Left) {
        orbit.yaw -= delta.x * 0.006;
        orbit.pitch = (orbit.pitch + delta.y * 0.006).clamp(-1.5, 1.5);
    }
    if buttons.pressed(MouseButton::Right) || buttons.pressed(MouseButton::Middle) {
        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        let k = orbit.distance * 0.0015;
        let pan = rot * Vec3::new(-delta.x * k, delta.y * k, 0.0);
        orbit.focus += pan;
    }
    let lines = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if lines != 0.0 {
        orbit.distance = (orbit.distance * 0.9f32.powf(lines)).clamp(0.2, 200.0);
    }
    let Ok(mut t) = cam.single_mut() else { return };
    let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
    let eye = orbit.focus + rot * Vec3::new(0.0, 0.0, orbit.distance);
    *t = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Y);
}

// ---------------------------------------------------------------- control

fn remote_control(
    mut v: ResMut<Viewer>,
    mut respawn: ResMut<Respawn>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    for line in v.control.poll() {
        let (cmd, arg) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        let arg = arg.trim();
        info!("control: {line}");
        match cmd {
            "open" => {
                v.path = Some(PathBuf::from(arg));
                v.system = None;
                v.diagnostics.clear();
                v.target = 0.0;
                v.time = 0.0;
                load(&mut v, &mut respawn);
                respawn.0 = true;
            }
            "screenshot" => {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(PathBuf::from(arg)));
            }
            "play" => v.playing = true,
            "pause" => v.playing = false,
            "restart" => {
                v.target = 0.0;
                respawn.0 = true;
            }
            "seek" => {
                if let Ok(t) = arg.parse::<f32>() {
                    v.target = t.clamp(0.0, v.duration);
                }
            }
            "speed" => {
                if let Ok(s) = arg.parse::<f32>() {
                    v.speed = s.clamp(1.0 / 16.0, 8.0);
                }
            }
            "backdrop" => {
                if let Some(i) = BACKDROPS.iter().position(|b| *b == arg) {
                    v.backdrop = i;
                    v.backdrop_dirty = true;
                }
            }
            "help" => v.help = arg != "off",
            "quit" => {
                exit.write(AppExit::Success);
            }
            _ => warn!("unknown control command: {line}"),
        }
    }
}

// --------------------------------------------------------------- backdrop

fn apply_backdrop(
    mut commands: Commands,
    mut v: ResMut<Viewer>,
    parts: Query<Entity, With<BackdropPart>>,
    mut cam: Query<&mut Camera, With<ViewCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !v.backdrop_dirty {
        return;
    }
    v.backdrop_dirty = false;
    for e in &parts {
        commands.entity(e).despawn();
    }
    let mut mat = |c: Color, rough: f32| {
        materials.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let name = BACKDROPS[v.backdrop];
    let (clear, ground) = match name {
        "grey" => (Color::srgb(0.32, 0.33, 0.35), Color::srgb(0.4, 0.4, 0.42)),
        "room" => (Color::srgb(0.05, 0.05, 0.06), Color::srgb(0.45, 0.43, 0.4)),
        _ => (
            Color::srgb(0.015, 0.017, 0.022),
            Color::srgb(0.06, 0.06, 0.065),
        ),
    };
    if let Ok(mut c) = cam.single_mut() {
        c.clear_color = ClearColorConfig::Custom(clear);
    }
    let size = if name == "room" { 12.0 } else { 30.0 };
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
        MeshMaterial3d(mat(ground, 0.9)),
        BackdropPart,
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: if name == "room" { 1500.0 } else { 3000.0 },
            shadow_maps_enabled: name == "room",
            ..default()
        },
        Transform::from_xyz(3.0, 6.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        BackdropPart,
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 300.0,
        ..default()
    });
    if name == "room" {
        // A lit test room: back and side walls, a few reference objects and
        // a warm point light, to judge lighting, soft particles and scale.
        let wall = mat(Color::srgb(0.55, 0.55, 0.58), 0.85);
        let wall_mesh = meshes.add(Cuboid::new(12.0, 6.0, 0.2));
        for (pos, rot) in [
            (Vec3::new(0.0, 3.0, -6.0), 0.0),
            (Vec3::new(-6.0, 3.0, 0.0), std::f32::consts::FRAC_PI_2),
            (Vec3::new(6.0, 3.0, 0.0), std::f32::consts::FRAC_PI_2),
        ] {
            commands.spawn((
                Mesh3d(wall_mesh.clone()),
                MeshMaterial3d(wall.clone()),
                Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(rot)),
                BackdropPart,
            ));
        }
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
            MeshMaterial3d(mat(Color::srgb(0.7, 0.25, 0.2), 0.6)),
            Transform::from_xyz(-2.2, 0.5, -1.5),
            BackdropPart,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.6))),
            MeshMaterial3d(mat(Color::srgb(0.2, 0.45, 0.75), 0.3)),
            Transform::from_xyz(2.2, 0.6, -1.2),
            BackdropPart,
        ));
        commands.spawn((
            PointLight {
                intensity: 400_000.0,
                color: Color::srgb(1.0, 0.85, 0.65),
                range: 20.0,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_xyz(0.0, 4.5, 2.0),
            BackdropPart,
        ));
    }
}

// ---------------------------------------------------------------- overlay

fn gpu_passes(store: &DiagnosticsStore) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = store
        .iter()
        .filter_map(|d| {
            let p = d.path().as_str();
            let name = p.strip_suffix("/elapsed_gpu")?;
            let name = name.strip_prefix("render/").unwrap_or(name);
            Some((name.to_string(), d.smoothed()?))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[allow(clippy::too_many_arguments)]
fn update_overlay(
    v: Res<Viewer>,
    store: Res<DiagnosticsStore>,
    stats: Res<FxStats>,
    mut stats_text: Query<&mut Text, (With<StatsText>, Without<ErrorText>, Without<TimelineLabel>)>,
    mut help: Query<&mut Node, (With<HelpText>, Without<ErrorPanel>, Without<TimelineFill>)>,
    mut panel: Query<&mut Node, (With<ErrorPanel>, Without<HelpText>, Without<TimelineFill>)>,
    mut err_text: Query<&mut Text, (With<ErrorText>, Without<StatsText>, Without<TimelineLabel>)>,
    mut fill: Query<&mut Node, (With<TimelineFill>, Without<HelpText>, Without<ErrorPanel>)>,
    mut label: Query<&mut Text, (With<TimelineLabel>, Without<StatsText>, Without<ErrorText>)>,
) {
    let fps = store
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let mut s = String::new();
    let file = v
        .path
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "(no file: fxviewer <effect.fx.ron>)".into());
    let name = v.system.as_ref().map(|s| s.name.as_str()).unwrap_or("-");
    s.push_str(&format!("{file}  \"{name}\"  reloads {}\n", v.reloads.saturating_sub(1)));
    s.push_str(&format!(
        "Raster  |  backdrop {}  |  {:.0} fps\n",
        BACKDROPS[v.backdrop], fps
    ));
    let snap = stats.snapshot();
    let mut total = 0;
    if let Some(es) = v.effect.and_then(|e| snap.effects.get(&e)) {
        for em in &es.emitters {
            total += em.alive;
            s.push_str(&format!(
                "  {:<14} {:>8} / {:<8}\n",
                em.name, em.alive, em.capacity
            ));
        }
        s.push_str(&format!(
            "  particles {total}   GPU mem {:.1} MB\n",
            es.gpu_bytes as f64 / 1048576.0
        ));
    } else {
        s.push_str("  (no particles)\n");
    }
    s.push_str("GPU ms per pass\n");
    for (pass, ms) in gpu_passes(&store) {
        s.push_str(&format!("  {pass:<26} {ms:>6.3}\n"));
    }
    if let Ok(mut t) = stats_text.single_mut() {
        t.0 = s.trim_end().to_string();
    }
    if let Ok(mut n) = help.single_mut() {
        n.display = if v.help { Display::Flex } else { Display::None };
    }
    let errs = v.has_errors && !v.diagnostics.is_empty();
    if let Ok(mut n) = panel.single_mut() {
        n.display = if errs { Display::Flex } else { Display::None };
    }
    if let Ok(mut t) = err_text.single_mut() {
        let keep = if v.system.is_some() {
            "  (still running the last good version)"
        } else {
            ""
        };
        t.0 = format!("Effect has errors{keep}\n{}", v.diagnostics.join("\n"));
    }
    if let Ok(mut n) = fill.single_mut() {
        n.width = percent(100.0 * (v.time / v.duration).clamp(0.0, 1.0));
    }
    if let Ok(mut t) = label.single_mut() {
        t.0 = format!(
            "{} {:5.2} / {:.2} s   speed {}x{}",
            if v.playing { "PLAY " } else { "PAUSE" },
            v.time,
            v.duration,
            v.speed,
            if v.looping { "   loop" } else { "" }
        );
    }
}

fn write_status(
    mut v: ResMut<Viewer>,
    time: Res<Time>,
    store: Res<DiagnosticsStore>,
    stats: Res<FxStats>,
) {
    if !v.status_timer.tick(time.delta()).just_finished() {
        return;
    }
    let snap = stats.snapshot();
    let emitters: Vec<serde_json::Value> = v
        .effect
        .and_then(|e| snap.effects.get(&e))
        .map(|es| {
            es.emitters
                .iter()
                .map(|em| {
                    serde_json::json!({"name": em.name, "alive": em.alive, "capacity": em.capacity})
                })
                .collect()
        })
        .unwrap_or_default();
    let passes: serde_json::Map<String, serde_json::Value> = gpu_passes(&store)
        .into_iter()
        .map(|(k, ms)| (k, serde_json::json!((ms * 1000.0).round() / 1000.0)))
        .collect();
    let status = serde_json::json!({
        "pid": std::process::id(),
        "file": v.path.as_ref().map(|p| p.display().to_string()),
        "effect": v.system.as_ref().map(|s| s.name.clone()),
        "ok": !v.has_errors,
        "diagnostics": v.diagnostics,
        "reloads": v.reloads.saturating_sub(1),
        "time": v.time,
        "playing": v.playing,
        "speed": v.speed,
        "backdrop": BACKDROPS[v.backdrop],
        "emitters": emitters,
        "gpu_ms": passes,
    });
    let dir = v.control.dir.clone();
    let tmp = dir.join("fxviewer.status.json.tmp");
    if std::fs::write(&tmp, status.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join("fxviewer.status.json"));
    }
}
