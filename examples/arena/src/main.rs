//! Arena: an example game scene for the fx game API.
//!
//! A caster walks a circle in the middle of a walled arena, a looping wisp
//! attached to its staff tip (brighter while casting: a `User.` parameter set
//! every frame). It and four corner turrets fire projectiles, each with an
//! attached looping orb effect; where a projectile lands, a pooled one-shot
//! impact (sparks colliding with the scene depth, flash, dust; tinted per
//! shooter) spawns and despawns itself when done. Every few seconds a
//! barrage drops 120 impacts at once. Instances come from the pool after the
//! first seconds: no compilation or GPU allocation per spawn.
//!
//! ```text
//! arena                                   window, realtime
//! arena --record DIR [--seconds 8] [--fps 30] [--size 1280x720]
//!                                         headless: DIR/arena.png (middle frame) and DIR/arena.mp4
//! arena --bench [--count 500] [--out FILE.json]
//!                                         headless: GPU cost of COUNT concurrent one-shots,
//!                                         scalability off/on, camera at several distances
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use fx_core::CookedSystem;
use fx_render::{
    EffectControl, EffectMode, FxClock, FxCommandsExt, FxPlugin, FxPoolStats, FxReady,
    FxScalability, ParticleEffect,
};

const SIM_HZ: u32 = 60;
const ARENA: f32 = 15.0;
const WALL_H: f32 = 3.0;
const PILLARS: [Vec2; 4] = [
    Vec2::new(-6.0, -6.0),
    Vec2::new(6.0, -6.0),
    Vec2::new(-6.0, 6.0),
    Vec2::new(6.0, 6.0),
];
const PILLAR_HALF: f32 = 0.8;

#[derive(Clone, Debug)]
enum Mode {
    Window,
    Record {
        out: PathBuf,
        seconds: f32,
        fps: u32,
        size: (u32, u32),
    },
    Bench {
        count: usize,
        out: Option<PathBuf>,
    },
}

fn parse_args() -> Result<Mode, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = Mode::Window;
    let mut i = 0;
    let next = |i: &mut usize| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{} needs a value", args[*i - 1]))
    };
    let (mut seconds, mut fps, mut size, mut count, mut out) =
        (8.0f32, 30u32, (1280u32, 720u32), 500usize, None::<PathBuf>);
    let mut record = None;
    let mut bench = false;
    while i < args.len() {
        match args[i].as_str() {
            "--record" => record = Some(PathBuf::from(next(&mut i)?)),
            "--bench" => bench = true,
            "--seconds" => seconds = next(&mut i)?.parse().map_err(|e| format!("--seconds: {e}"))?,
            "--fps" => fps = next(&mut i)?.parse().map_err(|e| format!("--fps: {e}"))?,
            "--count" => count = next(&mut i)?.parse().map_err(|e| format!("--count: {e}"))?,
            "--out" => out = Some(PathBuf::from(next(&mut i)?)),
            "--size" => {
                let v = next(&mut i)?;
                let (w, h) = v.split_once('x').ok_or("--size WxH")?;
                size = (
                    w.parse().map_err(|_| "--size WxH")?,
                    h.parse().map_err(|_| "--size WxH")?,
                );
            }
            a => return Err(format!("unknown argument {a}")),
        }
        i += 1;
    }
    if let Some(out) = record {
        mode = Mode::Record {
            out,
            seconds,
            fps,
            size,
        };
    } else if bench {
        mode = Mode::Bench { count, out };
    }
    Ok(mode)
}

fn load(name: &str) -> Arc<CookedSystem> {
    let path = format!("{}/effects/{name}", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let c = fx_core::check_source(&path, &src);
    for d in &c.diagnostics {
        eprintln!("{d}");
    }
    Arc::new(c.cooked.unwrap_or_else(|| panic!("{path} does not compile")))
}

/// The effects, each loaded once: every spawn shares the `Arc`, hence one
/// compiled program and one pool per effect.
#[derive(Resource, Clone)]
struct Effects {
    impact: Arc<CookedSystem>,
    orb: Arc<CookedSystem>,
    staff: Arc<CookedSystem>,
}

/// Deterministic xorshift, so recordings repeat.
#[derive(Resource)]
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
    fn seed(&mut self) -> u32 {
        (self.next() * u32::MAX as f32) as u32
    }
}

#[derive(Resource, Default)]
struct Game {
    /// Simulated seconds (follows the effects clock).
    time: f32,
    next_cast: f32,
    next_turret: f32,
    next_barrage: f32,
    cast_glow: f32,
    shots: u64,
}

#[derive(Component)]
struct Caster;

#[derive(Component)]
struct StaffTip;

#[derive(Component)]
struct StaffGlow;

#[derive(Component)]
struct Turret(Color);

#[derive(Component)]
struct Projectile {
    target: Vec3,
    normal: Vec3,
    speed: f32,
    tint: Color,
}

#[derive(Component)]
struct OrbitCamera;

fn main() {
    let mode = match parse_args() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("arena: {e}");
            std::process::exit(2);
        }
    };
    let effects = Effects {
        impact: load("impact.fx.ron"),
        orb: load("orb.fx.ron"),
        staff: load("staff.fx.ron"),
    };
    let mut app = App::new();
    match &mode {
        Mode::Window => {
            app.add_plugins((
                DefaultPlugins.set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "fx arena".into(),
                        resolution: (1600, 900).into(),
                        ..default()
                    }),
                    ..default()
                }),
                FxPlugin::default(),
            ));
        }
        Mode::Record { .. } | Mode::Bench { .. } => {
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
                RenderDiagnosticsPlugin,
                FxPlugin::default(),
            ))
            .insert_resource(FxClock::manual(1.0 / SIM_HZ as f32));
        }
    }
    app.insert_resource(effects)
        .insert_resource(Rng(0x9e37_79b9))
        .init_resource::<Game>()
        .insert_resource(RunMode(mode.clone()))
        .add_systems(Startup, (setup_scene, setup_camera));
    match mode {
        Mode::Bench { .. } => {
            app.init_resource::<BenchState>().add_systems(Update, bench);
        }
        _ => {
            app.add_systems(
                Update,
                (
                    game_clock,
                    (move_caster, cast, turrets, barrage, fly, staff_glow, orbit_camera),
                )
                    .chain(),
            );
            if let Mode::Record { .. } = mode {
                app.init_resource::<Recorder>().add_systems(Update, record);
            } else {
                app.add_systems(Update, print_stats);
            }
        }
    }
    app.run();
}

#[derive(Resource)]
struct RunMode(Mode);

// ------------------------------------------------------------------- scene

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    effects: Res<Effects>,
    mode: Res<RunMode>,
) {
    let stone = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.31, 0.33),
        perceptual_roughness: 0.85,
        ..default()
    });
    let floor = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.16, 0.18),
        perceptual_roughness: 0.95,
        ..default()
    });
    let size = if matches!(mode.0, Mode::Bench { .. }) { 240.0 } else { 2.0 * ARENA + 10.0 };
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
        MeshMaterial3d(floor),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 120.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 1500.0,
            ..default()
        },
        Transform::from_xyz(6.0, 12.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    if matches!(mode.0, Mode::Bench { .. }) {
        return;
    }
    // Walls and pillars.
    let wall = meshes.add(Cuboid::new(2.0 * ARENA + 1.0, WALL_H, 1.0));
    for (pos, rot) in [
        (Vec3::new(0.0, WALL_H / 2.0, -ARENA - 0.5), 0.0),
        (Vec3::new(0.0, WALL_H / 2.0, ARENA + 0.5), 0.0),
        (Vec3::new(-ARENA - 0.5, WALL_H / 2.0, 0.0), std::f32::consts::FRAC_PI_2),
        (Vec3::new(ARENA + 0.5, WALL_H / 2.0, 0.0), std::f32::consts::FRAC_PI_2),
    ] {
        commands.spawn((
            Mesh3d(wall.clone()),
            MeshMaterial3d(stone.clone()),
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(rot)),
        ));
    }
    let pillar = meshes.add(Cuboid::new(2.0 * PILLAR_HALF, 4.0, 2.0 * PILLAR_HALF));
    for p in PILLARS {
        commands.spawn((
            Mesh3d(pillar.clone()),
            MeshMaterial3d(stone.clone()),
            Transform::from_xyz(p.x, 2.0, p.y),
        ));
    }
    // The caster: a robed body, a head and a staff whose tip carries a looping wisp.
    let robe = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.22, 0.45),
        perceptual_roughness: 0.7,
        ..default()
    });
    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.22, 0.12),
        ..default()
    });
    let caster = commands
        .spawn((
            Caster,
            Transform::from_xyz(3.0, 0.0, 0.0),
            Visibility::default(),
            children![
                (
                    Mesh3d(meshes.add(Capsule3d::new(0.35, 1.0))),
                    MeshMaterial3d(robe.clone()),
                    Transform::from_xyz(0.0, 0.85, 0.0),
                ),
                (
                    Mesh3d(meshes.add(Sphere::new(0.25))),
                    MeshMaterial3d(robe),
                    Transform::from_xyz(0.0, 1.75, 0.0),
                ),
            ],
        ))
        .id();
    let staff = commands
        .spawn((
            Mesh3d(meshes.add(Cylinder::new(0.04, 2.0))),
            MeshMaterial3d(wood),
            Transform::from_xyz(0.45, 1.0, -0.2),
            ChildOf(caster),
        ))
        .id();
    let tip = commands
        .spawn((StaffTip, Transform::from_xyz(0.0, 1.05, 0.0), ChildOf(staff)))
        .id();
    commands
        .attach_effect(
            tip,
            ParticleEffect::shared(effects.staff.clone(), 7),
            EffectMode::Looping,
            Transform::IDENTITY,
        )
        .insert(StaffGlow);
    // Turrets in the corners, one tint each.
    let metal = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.25, 0.28),
        metallic: 0.6,
        perceptual_roughness: 0.4,
        ..default()
    });
    let tints = [
        Color::linear_rgb(1.0, 0.45, 0.12),
        Color::linear_rgb(0.25, 1.0, 0.35),
        Color::linear_rgb(1.0, 0.2, 0.6),
        Color::linear_rgb(0.3, 0.6, 1.0),
    ];
    for (i, (x, z)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].into_iter().enumerate() {
        commands.spawn((
            Turret(tints[i]),
            Mesh3d(meshes.add(Cuboid::new(0.8, 1.2, 0.8))),
            MeshMaterial3d(metal.clone()),
            Transform::from_xyz(x * (ARENA - 1.2), 0.6, z * (ARENA - 1.2)),
        ));
    }
}

fn setup_camera(mut commands: Commands, mode: Res<RunMode>, mut images: ResMut<Assets<Image>>) {
    let size = match &mode.0 {
        Mode::Record { size, .. } => Some(*size),
        Mode::Bench { .. } => Some((1280, 720)),
        Mode::Window => None,
    };
    let mut cam = commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        bevy::core_pipeline::tonemapping::Tonemapping::TonyMcMapface,
        bevy::post_process::bloom::Bloom::NATURAL,
        OrbitCamera,
        Transform::from_xyz(0.0, 11.0, 22.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
    if let Some((w, h)) = size {
        let mut img = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
        img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let handle = images.add(img);
        cam.insert(RenderTarget::from(handle.clone()));
        commands.insert_resource(Target(handle));
    }
}

#[derive(Resource)]
struct Target(Handle<Image>);

// -------------------------------------------------------------------- game

/// Seconds of simulation this frame: the game follows the effects clock.
fn game_clock(
    mut game: ResMut<Game>,
    mut clock: ResMut<FxClock>,
    mode: Res<RunMode>,
    ready: Res<FxReady>,
    mut started: Local<bool>,
) {
    if let Mode::Record { fps, .. } = mode.0 {
        // Two steps per 30 fps frame; nothing moves until everything is loaded.
        *started |= ready.get();
        if *started {
            clock.request_steps(SIM_HZ / fps);
        }
    }
    game.time += clock.steps_this_frame() as f32 * clock.step;
}

fn move_caster(game: Res<Game>, mut caster: Query<&mut Transform, With<Caster>>) {
    let t = game.time * 0.35;
    for mut tr in &mut caster {
        tr.translation = Vec3::new(3.0 * t.cos(), 0.0, 3.0 * t.sin());
        // Facing along the circle.
        tr.rotation = Quat::from_rotation_y(-t);
    }
}

/// A point on a surface of the arena and its normal: a wall's inner face, a
/// pillar face or the floor.
fn random_target(rng: &mut Rng) -> (Vec3, Vec3) {
    let pick = rng.next();
    if pick < 0.5 {
        let side = (rng.next() * 4.0) as u32;
        let along = rng.range(-ARENA + 0.5, ARENA - 0.5);
        let y = rng.range(0.3, WALL_H - 0.3);
        match side {
            0 => (Vec3::new(along, y, -ARENA), Vec3::Z),
            1 => (Vec3::new(along, y, ARENA), Vec3::NEG_Z),
            2 => (Vec3::new(-ARENA, y, along), Vec3::X),
            _ => (Vec3::new(ARENA, y, along), Vec3::NEG_X),
        }
    } else if pick < 0.7 {
        let p = PILLARS[(rng.next() * 4.0) as usize % 4];
        let face = (rng.next() * 4.0) as u32;
        let y = rng.range(0.3, 3.5);
        let along = rng.range(-PILLAR_HALF, PILLAR_HALF);
        let (off, n) = match face {
            0 => (Vec3::new(along, 0.0, PILLAR_HALF), Vec3::Z),
            1 => (Vec3::new(along, 0.0, -PILLAR_HALF), Vec3::NEG_Z),
            2 => (Vec3::new(PILLAR_HALF, 0.0, along), Vec3::X),
            _ => (Vec3::new(-PILLAR_HALF, 0.0, along), Vec3::NEG_X),
        };
        (Vec3::new(p.x, y, p.y) + off, n)
    } else {
        let r = ARENA - 1.0;
        (Vec3::new(rng.range(-r, r), 0.0, rng.range(-r, r)), Vec3::Y)
    }
}

fn fire(
    commands: &mut Commands,
    effects: &Effects,
    rng: &mut Rng,
    from: Vec3,
    tint: Color,
    speed: f32,
) {
    let (target, normal) = random_target(rng);
    let orb = ParticleEffect::shared(effects.orb.clone(), rng.seed())
        .with_param("User.Tint", tint)
        .expect("orb has a Color User.Tint");
    commands.spawn((
        Projectile {
            target,
            normal,
            speed,
            tint,
        },
        orb,
        EffectControl::looping(),
        Transform::from_translation(from),
    ));
}

fn cast(
    mut commands: Commands,
    mut game: ResMut<Game>,
    effects: Res<Effects>,
    mut rng: ResMut<Rng>,
    tip: Query<&GlobalTransform, With<StaffTip>>,
) {
    let Ok(tip) = tip.single() else { return };
    game.cast_glow = (game.cast_glow - 0.05).max(0.0);
    while game.time >= game.next_cast {
        game.next_cast += 0.35;
        game.cast_glow = 1.0;
        game.shots += 1;
        fire(
            &mut commands,
            &effects,
            &mut rng,
            tip.translation(),
            Color::linear_rgb(0.35, 0.6, 1.0),
            16.0,
        );
    }
}

fn turrets(
    mut commands: Commands,
    mut game: ResMut<Game>,
    effects: Res<Effects>,
    mut rng: ResMut<Rng>,
    turrets: Query<(&Turret, &GlobalTransform)>,
) {
    while game.time >= game.next_turret {
        game.next_turret += 0.08;
        let n = turrets.iter().count().max(1);
        let k = (rng.next() * n as f32) as usize % n;
        if let Some((t, gt)) = turrets.iter().nth(k) {
            game.shots += 1;
            fire(&mut commands, &effects, &mut rng, gt.translation() + Vec3::Y * 0.8, t.0, 22.0);
        }
    }
}

/// A burst of 120 impacts on the floor at once (a meteor shower).
fn barrage(
    mut commands: Commands,
    mut game: ResMut<Game>,
    effects: Res<Effects>,
    mut rng: ResMut<Rng>,
) {
    if game.time < game.next_barrage {
        return;
    }
    if game.next_barrage == 0.0 {
        game.next_barrage = 2.5;
        return;
    }
    game.next_barrage += 4.0;
    let center = Vec2::new(rng.range(-6.0, 6.0), rng.range(-6.0, 6.0));
    for _ in 0..120 {
        let a = rng.range(0.0, std::f32::consts::TAU);
        let r = 4.0 * rng.next().sqrt();
        let p = center + Vec2::new(a.cos(), a.sin()) * r;
        let effect = ParticleEffect::shared(effects.impact.clone(), rng.seed())
            .with_param("Tint", Color::linear_rgb(1.0, 0.75, 0.3))
            .unwrap();
        commands.spawn_one_shot(effect, Transform::from_xyz(p.x, 0.0, p.y));
    }
}

fn fly(
    mut commands: Commands,
    clock: Res<FxClock>,
    effects: Res<Effects>,
    mut rng: ResMut<Rng>,
    mut projectiles: Query<(Entity, &Projectile, &mut Transform)>,
) {
    let dt = clock.steps_this_frame() as f32 * clock.step;
    for (e, p, mut tr) in &mut projectiles {
        let to = p.target - tr.translation;
        let step = p.speed * dt;
        if to.length() <= step {
            // Impact: a pooled one-shot, +Y along the surface normal.
            let impact = ParticleEffect::shared(effects.impact.clone(), rng.seed())
                .with_param("Tint", p.tint)
                .unwrap();
            commands.spawn_one_shot(
                impact,
                Transform::from_translation(p.target)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, p.normal)),
            );
            commands.entity(e).despawn();
        } else {
            tr.translation += to.normalize() * step;
        }
    }
}

fn staff_glow(game: Res<Game>, mut glow: Query<&mut ParticleEffect, With<StaffGlow>>) {
    for mut fx in &mut glow {
        fx.set_param("User.Intensity", 1.0 + 3.0 * game.cast_glow).unwrap();
    }
}

fn orbit_camera(game: Res<Game>, mut cam: Query<&mut Transform, With<OrbitCamera>>) {
    let a = 0.6 + game.time * 0.08;
    for mut tr in &mut cam {
        *tr = Transform::from_xyz(21.0 * a.sin(), 10.5, 21.0 * a.cos())
            .looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y);
    }
}

fn print_stats(
    time: Res<Time>,
    mut last: Local<f32>,
    pool: Res<FxPoolStats>,
    game: Res<Game>,
    effects: Query<&EffectControl>,
) {
    if time.elapsed_secs() - *last < 2.0 {
        return;
    }
    *last = time.elapsed_secs();
    let s = pool.get();
    let culled = effects.iter().filter(|c| c.culled().is_some()).count();
    println!(
        "{:.1} s: {} effects live ({} culled), {} idle in pool, {} instances built, {} reused, {} shots, {:.1} MB GPU",
        game.time,
        s.live,
        culled,
        s.idle,
        s.instances_created,
        s.instances_reused,
        game.shots,
        s.gpu_bytes as f64 / 1e6
    );
}

// ------------------------------------------------------------------ record

#[derive(Resource, Default)]
struct Recorder {
    /// Frames seen since the game started moving.
    frame: u32,
    requested: u32,
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
    max_live: u64,
}

/// Starts capturing after 3 s of play (pools warm, the arena busy), then one
/// readback per frame.
fn record(
    mut commands: Commands,
    mut rec: ResMut<Recorder>,
    mode: Res<RunMode>,
    target: Res<Target>,
    ready: Res<FxReady>,
    pool: Res<FxPoolStats>,
    clock: Res<FxClock>,
    mut exit: MessageWriter<AppExit>,
    mut started: Local<bool>,
) {
    let Mode::Record {
        out,
        seconds,
        fps,
        size,
    } = &mode.0
    else {
        return;
    };
    *started |= ready.get() && clock.total_steps > 0;
    if !*started {
        return;
    }
    rec.frame += 1;
    rec.max_live = rec.max_live.max(pool.get().live);
    let warm = 3 * fps;
    let n = (*seconds * *fps as f32).round() as u32;
    if rec.frame > warm && rec.requested < n {
        let frames = rec.frames.clone();
        commands
            .spawn(Readback::texture(target.0.clone()))
            .observe(move |ev: On<ReadbackComplete>| {
                frames.lock().unwrap().push(ev.data.clone());
            });
        rec.requested += 1;
    }
    let got = rec.frames.lock().unwrap().len();
    if got >= n as usize {
        let frames: Vec<Vec<u8>> = rec.frames.lock().unwrap()[..n as usize].to_vec();
        match write_video(out, *fps, *size, &frames) {
            Ok(msg) => {
                let s = pool.get();
                println!(
                    "{msg}\npool: {} instances built, {} reused, up to {} live",
                    s.instances_created, s.instances_reused, rec.max_live
                );
                exit.write(AppExit::Success);
            }
            Err(e) => {
                eprintln!("arena: {e}");
                exit.write(AppExit::error());
            }
        }
    }
}

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

fn write_video(out: &PathBuf, fps: u32, (w, h): (u32, u32), frames: &[Vec<u8>]) -> Result<String, String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let frames: Vec<Vec<u8>> = frames.iter().map(|f| unpad(f, w, h)).collect();
    let png_path = out.join("arena.png");
    {
        let file = std::fs::File::create(&png_path).map_err(|e| e.to_string())?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(&frames[frames.len() / 2]).map_err(|e| e.to_string())?;
    }
    let mp4 = out.join("arena.mp4");
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
        .arg(format!("{w}x{h}"))
        .args(["-r", &fps.to_string(), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"])
        .arg(&mp4)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    {
        let stdin = child.stdin.as_mut().unwrap();
        for f in &frames {
            stdin.write_all(f).map_err(|e| e.to_string())?;
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("ffmpeg failed: {status}"));
    }
    Ok(format!(
        "{} and {}: {} frames at {fps} fps",
        png_path.display(),
        mp4.display(),
        frames.len()
    ))
}

// ------------------------------------------------------------------- bench

/// Camera distances (m) from the centre of the effects' field.
const BENCH_DISTANCES: [f32; 5] = [8.0, 20.0, 40.0, 60.0, 90.0];
const BENCH_WARM: u32 = 90;
const BENCH_FRAMES: u32 = 120;

#[derive(Resource, Default)]
struct BenchState {
    /// Index into the (scalability, distance) configurations.
    config: usize,
    frame: u32,
    samples: Vec<(String, Vec<f64>)>,
    live: Vec<f64>,
    culled: Vec<f64>,
    results: Vec<String>,
    started: bool,
}

fn bench_configs() -> Vec<(bool, f32)> {
    let mut v = Vec::new();
    for on in [false, true] {
        for d in BENCH_DISTANCES {
            v.push((on, d));
        }
    }
    v
}

/// Keeps `count` one-shot impacts alive, spread over a 24 m square, and
/// measures each configuration's GPU time.
#[allow(clippy::too_many_arguments)]
fn bench(
    mut commands: Commands,
    mut st: ResMut<BenchState>,
    mode: Res<RunMode>,
    effects: Res<Effects>,
    mut rng: ResMut<Rng>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    store: Res<DiagnosticsStore>,
    mut scal: ResMut<FxScalability>,
    pool: Res<FxPoolStats>,
    live: Query<&EffectControl>,
    mut cam: Query<&mut Transform, With<OrbitCamera>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Mode::Bench { count, out } = &mode.0 else {
        return;
    };
    let configs = bench_configs();
    let (on, dist) = configs[st.config];
    scal.enabled = on;
    for mut tr in &mut cam {
        *tr = Transform::from_xyz(0.0, dist * 0.45, dist).looking_at(Vec3::ZERO, Vec3::Y);
    }
    // Top up to `count` instances.
    let n = live.iter().count();
    for _ in n..*count {
        let p = Vec2::new(rng.range(-12.0, 12.0), rng.range(-12.0, 12.0));
        let effect = ParticleEffect::shared(effects.impact.clone(), rng.seed());
        commands.spawn_one_shot(effect, Transform::from_xyz(p.x, 0.0, p.y));
    }
    clock.request_steps(1);
    st.started |= ready.get();
    if !st.started {
        return;
    }
    st.frame += 1;
    if st.frame <= BENCH_WARM {
        return;
    }
    let culled = live.iter().filter(|c| c.culled().is_some()).count();
    st.live.push(n as f64);
    st.culled.push(culled as f64);
    for d in store.iter() {
        let path = d.path().as_str();
        if !path.ends_with("elapsed_gpu") {
            continue;
        }
        let Some(v) = d.measurement().map(|m| m.value) else {
            continue;
        };
        let name = path.trim_end_matches("/elapsed_gpu").rsplit('/').next().unwrap_or(path).to_string();
        match st.samples.iter_mut().find(|(p, _)| *p == name) {
            Some((_, l)) => l.push(v),
            None => st.samples.push((name, vec![v])),
        }
    }
    if st.frame < BENCH_WARM + BENCH_FRAMES {
        return;
    }
    let median = |v: &[f64]| -> f64 {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let get = |st: &BenchState, name: &str| {
        st.samples.iter().find(|(p, _)| p == name).map_or(0.0, |(_, v)| median(v))
    };
    let (sim, cull, ribbon, draw) = (
        get(&st, "fx_sim"),
        get(&st, "fx_cull_sort"),
        get(&st, "fx_ribbon_build"),
        get(&st, "main_transparent_pass_3d"),
    );
    let total = sim + cull + ribbon + draw;
    let line = format!(
        "{{\"scalability\": {on}, \"distance\": {dist}, \"effects\": {:.0}, \"culled\": {:.0}, \"fx_sim_ms\": {sim:.3}, \"fx_cull_sort_ms\": {cull:.3}, \"transparent_pass_ms\": {draw:.3}, \"total_ms\": {total:.3}}}",
        median(&st.live),
        median(&st.culled)
    );
    println!("{line}");
    st.results.push(line);
    st.samples.clear();
    st.live.clear();
    st.culled.clear();
    st.frame = 0;
    st.config += 1;
    if st.config == configs.len() {
        let s = pool.get();
        println!(
            "pool: {} programs, {} instances built, {} reused, {:.1} MB GPU",
            s.programs,
            s.instances_created,
            s.instances_reused,
            s.gpu_bytes as f64 / 1e6
        );
        if let Some(out) = out {
            let json = format!("[\n  {}\n]\n", st.results.join(",\n  "));
            if let Err(e) = std::fs::write(out, json) {
                eprintln!("arena: {}: {e}", out.display());
            }
        }
        exit.write(AppExit::Success);
    }
}
