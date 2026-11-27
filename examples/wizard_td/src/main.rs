//! Wizard TD: a first-person wizard defends a tower against waves of foxes
//! walking a path. Every spell is an fx effect spawned through the game API:
//!
//! - **Fireball**: a projectile carrying a looping effect (core, flames and a
//!   world-space ribbon trail); a pooled one-shot explosion where it lands.
//! - **Lightning**: an instant beam (one-shot, `User.Length` set to the
//!   distance) with sparks at the target, chaining to two more foxes.
//! - **Frost nova**: an expanding ring and lingering mist (one-shot,
//!   `User.Radius`) that damages and slows.
//! - **Turret**: a crystal tower that alternates fireballs and lightning on
//!   its own.
//!
//! Dying foxes spawn debris whose collisions spawn dust (particle events).
//! One-shots come from the pool after their first use. Quality adapts to the
//! frame rate (render scale, effects quality, bloom).
//!
//! ```text
//! wizard_td                                window (desktop) or the browser (web)
//! wizard_td --scripted                     window, a bot plays
//! wizard_td --scripted --out DIR [--seconds 60] [--fps 30] [--size 1280x720]
//!                                          headless: a bot plays; DIR/wizard_td.mp4 + stills
//! ```
//!
//! Desktop: WASD move (Shift run), mouse look (click to capture), left click
//! casts the selected spell, 1-4 select, R restarts. Touch: left stick moves,
//! drag on the right looks, the spell buttons cast at the crosshair.

mod controls;
mod hud;
#[cfg(not(target_arch = "wasm32"))]
mod record;
mod world;

use std::path::PathBuf;
use std::sync::Arc;

use bevy::camera::Hdr;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::world_serialization::WorldInstanceReady;
use fx_core::CookedSystem;
use fx_render::{
    EffectControl, FxClock, FxCommandsExt, FxEffectAsset, FxPlugin, FxQuality, FxReady,
    ParticleEffect, QualityLevel,
};

use world::{TOWER, path_distance, path_length, path_point};

pub const SIM_HZ: u32 = 60;
pub const EYE: f32 = 1.7;
pub const BOUNDS: f32 = 38.0;
pub const TOWER_RADIUS: f32 = 2.6;
pub const MAX_MANA: f32 = 100.0;
pub const MANA_REGEN: f32 = 14.0;
pub const MAX_TURRETS: usize = 6;

// ------------------------------------------------------------------ modes

#[derive(Clone, Debug)]
pub enum Mode {
    /// A window (or the browser canvas); `bot` plays when true.
    Window { bot: bool },
    /// Headless: the bot plays `seconds`, frames go to `out`.
    Record {
        out: PathBuf,
        seconds: f32,
        fps: u32,
        size: (u32, u32),
    },
}

#[derive(Resource, Clone)]
pub struct RunMode(pub Mode);

impl RunMode {
    pub fn bot(&self) -> bool {
        matches!(self.0, Mode::Window { bot: true } | Mode::Record { .. })
    }
    pub fn headless(&self) -> bool {
        matches!(self.0, Mode::Record { .. })
    }
}

fn parse_args() -> Result<Mode, String> {
    #[cfg(target_arch = "wasm32")]
    {
        return Ok(Mode::Window {
            bot: query().contains("bot"),
        });
    }
    #[allow(unreachable_code)]
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (mut scripted, mut out) = (false, None::<PathBuf>);
        let (mut seconds, mut fps, mut size) = (60.0f32, 30u32, (1280u32, 720u32));
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].clone();
            let mut val = || {
                i += 1;
                args.get(i).cloned().ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--scripted" => scripted = true,
                "--out" => out = Some(PathBuf::from(val()?)),
                "--seconds" => seconds = val()?.parse().map_err(|e| format!("--seconds: {e}"))?,
                "--fps" => fps = val()?.parse().map_err(|e| format!("--fps: {e}"))?,
                "--size" => {
                    let v = val()?;
                    let (w, h) = v.split_once('x').ok_or("--size WxH")?;
                    size = (w.parse().map_err(|_| "--size WxH")?, h.parse().map_err(|_| "--size WxH")?);
                }
                a => return Err(format!("unknown argument {a}")),
            }
            i += 1;
        }
        Ok(match out {
            Some(out) => Mode::Record { out, seconds, fps, size },
            None => Mode::Window { bot: scripted },
        })
    }
}

/// The page's query string in the browser (`?limits=default`, `?bot`).
pub fn query() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("FX_WEB_QUERY").unwrap_or_default()
    }
}

// ---------------------------------------------------------------- effects

#[derive(Resource)]
struct FxHandles {
    fireball: Handle<FxEffectAsset>,
    explosion: Handle<FxEffectAsset>,
    lightning: Handle<FxEffectAsset>,
    frost: Handle<FxEffectAsset>,
    death: Handle<FxEffectAsset>,
    fox: Handle<Gltf>,
}

/// The loaded effects: every spawn shares one `Arc` per effect, hence one
/// compiled program and one pool each.
#[derive(Resource, Clone)]
pub struct Fx {
    pub fireball: Arc<CookedSystem>,
    pub explosion: Arc<CookedSystem>,
    pub lightning: Arc<CookedSystem>,
    pub frost: Arc<CookedSystem>,
    pub death: Arc<CookedSystem>,
}

/// An effect with `User.` parameter `name` set (a warning if the effect
/// does not have it).
pub fn with(mut fx: ParticleEffect, name: &str, v: impl fx_render::IntoParam) -> ParticleEffect {
    if let Err(e) = fx.set_param(name, v) {
        warn_once!("{name}: {e:?}");
    }
    fx
}

#[derive(Resource)]
pub struct FoxAnim {
    graph: Handle<BlendGraph>,
    run: BlendNodeIndex,
    scene: Handle<WorldAsset>,
}

// ------------------------------------------------------------------- game

/// Deterministic xorshift, so scripted runs repeat.
#[derive(Resource)]
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
    pub fn seed(&mut self) -> u32 {
        (self.next() * u32::MAX as f32) as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Loading,
    Playing,
    GameOver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spell {
    Fireball,
    Lightning,
    Frost,
    Turret,
}

impl Spell {
    pub const ALL: [Spell; 4] = [Spell::Fireball, Spell::Lightning, Spell::Frost, Spell::Turret];
    pub fn cost(self) -> f32 {
        match self {
            Spell::Fireball => 10.0,
            Spell::Lightning => 18.0,
            Spell::Frost => 35.0,
            Spell::Turret => 50.0,
        }
    }
    pub fn cooldown(self) -> f32 {
        match self {
            Spell::Fireball => 0.3,
            Spell::Lightning => 0.6,
            Spell::Frost => 2.0,
            Spell::Turret => 0.8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Spell::Fireball => "Fireball",
            Spell::Lightning => "Lightning",
            Spell::Frost => "Frost Nova",
            Spell::Turret => "Turret",
        }
    }
    pub fn index(self) -> usize {
        Spell::ALL.iter().position(|&s| s == self).unwrap()
    }
}

/// The round: wave, tower, mana, score.
#[derive(Resource)]
pub struct Run {
    pub phase: Phase,
    /// Simulated seconds since the round started.
    pub time: f32,
    pub wave: u32,
    pub to_spawn: u32,
    pub spawn_timer: f32,
    /// Seconds until the next wave (while no fox is left).
    pub next_wave: f32,
    pub tower_hp: f32,
    pub mana: f32,
    pub score: u32,
    pub kills: u32,
    pub cooldown: [f32; 4],
    /// A short message (centre of the screen) and its remaining seconds.
    pub toast: Option<(String, f32)>,
    pub casts: u32,
}

impl Default for Run {
    fn default() -> Self {
        Run {
            phase: Phase::Loading,
            time: 0.0,
            wave: 0,
            to_spawn: 0,
            spawn_timer: 0.0,
            next_wave: 2.0,
            tower_hp: 100.0,
            mana: MAX_MANA,
            score: 0,
            kills: 0,
            cooldown: [0.0; 4],
            toast: None,
            casts: 0,
        }
    }
}

impl Run {
    pub fn toast(&mut self, s: impl Into<String>) {
        self.toast = Some((s.into(), 1.6));
    }
}

/// The wizard (the camera follows it).
#[derive(Resource)]
pub struct Player {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub selected: Spell,
}

impl Default for Player {
    fn default() -> Self {
        Player {
            pos: Vec3::new(0.0, 0.0, -19.0),
            yaw: 0.9,
            pitch: -0.12,
            selected: Spell::Fireball,
        }
    }
}

impl Player {
    pub fn rotation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }
    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::Y * EYE
    }
    pub fn forward(&self) -> Vec3 {
        self.rotation() * Vec3::NEG_Z
    }
}

/// What the player (keyboard, mouse, touch or the bot) asks for this frame.
#[derive(Resource, Default)]
pub struct Intent {
    /// x: strafe right, y: forward; length up to 1.
    pub mv: Vec2,
    pub run: bool,
    /// Radians: yaw (x) and pitch (y) to add.
    pub look: Vec2,
    pub select: Option<Spell>,
    pub cast: Option<Spell>,
    /// Cast the selected spell (left click).
    pub cast_selected: bool,
    pub restart: bool,
}

/// Everything that belongs to a round (despawned on restart).
#[derive(Component)]
pub struct Round;

#[derive(Component)]
pub struct Enemy {
    pub hp: f32,
    pub max_hp: f32,
    pub speed: f32,
    /// Metres walked along the path.
    pub dist: f32,
    pub slow: f32,
    pub big: bool,
}

impl Enemy {
    pub fn radius(&self) -> f32 {
        if self.big { 1.0 } else { 0.75 }
    }
}

pub fn enemy_center(t: &Transform, e: &Enemy) -> Vec3 {
    t.translation + Vec3::Y * if e.big { 0.8 } else { 0.55 }
}

#[derive(Component)]
pub struct Projectile {
    pub vel: Vec3,
    /// Homing target (turret shots).
    pub target: Option<Entity>,
    pub damage: f32,
    pub radius: f32,
    pub life: f32,
    pub tint: Color,
}

#[derive(Component)]
pub struct Turret {
    pub cooldown: f32,
    pub shots: u32,
}

#[derive(Component)]
pub struct TurretCrystal(pub f32);

#[derive(Component)]
pub struct StaffTip;

#[derive(Component)]
pub struct MainCamera;

/// Warm-up instances (pipelines compiled before play starts).
#[derive(Component)]
struct Warmup;

pub const FIRE_TINT: Color = Color::linear_rgb(1.0, 0.45, 0.12);
pub const TURRET_TINT: Color = Color::linear_rgb(0.75, 0.3, 1.0);

// ----------------------------------------------------------------- quality

/// Adaptive quality (window mode): drops a level while the frame rate is
/// under ~52 fps, raises one after a long stretch at 60.
#[derive(Resource)]
pub struct Quality {
    pub level: usize,
    pub base_scale: Option<f32>,
    pub frames: u32,
    pub acc: f32,
    pub fps: f32,
    pub calm: f32,
    pub applied: Option<usize>,
}

/// (render scale cap, effects quality, bloom) per level, best first.
pub const LEVELS: [(f32, QualityLevel, bool); 4] = [
    (2.0, QualityLevel::High, true),
    (1.5, QualityLevel::High, true),
    (1.0, QualityLevel::Medium, true),
    (0.75, QualityLevel::Low, false),
];

// ------------------------------------------------------------------- main

fn main() {
    let mode = match parse_args() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("wizard_td: {e}");
            std::process::exit(2);
        }
    };
    let mut wgpu = WgpuSettings::default();
    if query().contains("limits=default") {
        wgpu.constrained_limits = Some(bevy::render::settings::WgpuLimits::default());
    }
    let mut app = App::new();
    let assets = AssetPlugin {
        meta_check: bevy::asset::AssetMetaCheck::Never,
        ..default()
    };
    match &mode {
        Mode::Window { .. } => {
            app.add_plugins((
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "animatey: Wizard TD".into(),
                            resolution: (1600, 900).into(),
                            canvas: Some("#fx-canvas".into()),
                            fit_canvas_to_parent: true,
                            prevent_default_event_handling: true,
                            ..default()
                        }),
                        ..default()
                    })
                    .set(RenderPlugin {
                        render_creation: RenderCreation::Automatic(Box::new(wgpu)),
                        ..default()
                    })
                    .set(assets),
                FxPlugin::default(),
            ));
        }
        Mode::Record { .. } => {
            #[cfg(not(target_arch = "wasm32"))]
            record::headless_plugins(&mut app, assets);
        }
    }
    let run_mode = RunMode(mode.clone());
    app.insert_resource(ClearColor(Color::srgb(0.55, 0.62, 0.72)))
        .insert_resource(run_mode)
        .insert_resource(Rng(0x2545_f491))
        .init_resource::<Run>()
        .init_resource::<Player>()
        .init_resource::<Intent>()
        .init_resource::<controls::TouchState>()
        .insert_resource(Quality {
            level: 0,
            base_scale: None,
            frames: 0,
            acc: 0.0,
            fps: 60.0,
            calm: 0.0,
            applied: None,
        })
        .add_systems(Startup, (load, world::build_world, setup_camera, hud::build_hud))
        .add_systems(
            Update,
            (
                (loading, clear_intent),
                (controls::keyboard_mouse, controls::touch, controls::bot),
                (
                    restart,
                    tick,
                    move_player,
                    cast,
                    turrets,
                    waves,
                    walk,
                    projectiles,
                    deaths,
                ),
                (hud::update_hud, hud::update_buttons, hud::update_stick, adapt_quality, spin_crystals),
            )
                .chain(),
        );
    #[cfg(not(target_arch = "wasm32"))]
    if let Mode::Record { .. } = mode {
        record::add(&mut app);
    }
    app.add_observer(play_fox_run);
    app.run();
}

fn load(mut commands: Commands, assets: Res<AssetServer>, mut graphs: ResMut<Assets<BlendGraph>>) {
    let fox_path = "models/Fox.glb";
    let (graph, run) = BlendGraph::from_clip(assets.load(GltfAssetLabel::Animation(2).from_asset(fox_path)));
    commands.insert_resource(FoxAnim {
        graph: graphs.add(graph),
        run,
        scene: assets.load(GltfAssetLabel::Scene(0).from_asset(fox_path)),
    });
    commands.insert_resource(FxHandles {
        fireball: assets.load("effects/fireball.fx.ron"),
        explosion: assets.load("effects/explosion.fx.ron"),
        lightning: assets.load("effects/lightning.fx.ron"),
        frost: assets.load("effects/frost_nova.fx.ron"),
        death: assets.load("effects/death.fx.ron"),
        fox: assets.load(fox_path),
    });
}

fn setup_camera(mut commands: Commands, mode: Res<RunMode>, mut images: ResMut<Assets<Image>>) {
    let player = Player::default();
    let cam = commands.spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        bevy::core_pipeline::tonemapping::Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        Projection::Perspective(PerspectiveProjection {
            fov: 72f32.to_radians(),
            near: 0.05,
            ..default()
        }),
        bevy::pbr::DistanceFog {
            color: Color::srgb(0.55, 0.62, 0.72),
            falloff: bevy::pbr::FogFalloff::Linear { start: 35.0, end: 95.0 },
            ..default()
        },
        IsDefaultUiCamera,
        MainCamera,
        Transform::from_translation(player.eye()).with_rotation(player.rotation()),
    ));
    let cam_id = cam.id();
    #[cfg(not(target_arch = "wasm32"))]
    if let Mode::Record { size, .. } = &mode.0 {
        record::attach_target(&mut commands, cam_id, *size, &mut images);
    }
    let _ = (&mode, &mut images);
    // The staff, held at the lower right of the view.
    let staff = commands
        .spawn((
            Transform::from_xyz(0.42, -0.45, -0.8).with_rotation(Quat::from_euler(EulerRot::XYZ, -0.35, 0.0, -0.18)),
            Visibility::default(),
            ChildOf(cam_id),
        ))
        .id();
    commands.queue(move |world: &mut World| {
        let (shaft, gem) = {
            let mut meshes = world.resource_mut::<Assets<Mesh>>();
            (
                meshes.add(Cylinder::new(0.025, 1.0).mesh().resolution(6)),
                meshes.add(world::flat(Sphere::new(0.035).mesh().ico(0).unwrap())),
            )
        };
        let (wood, glow) = {
            let mut mats = world.resource_mut::<Assets<StandardMaterial>>();
            (
                mats.add(StandardMaterial {
                    base_color: Color::srgb(0.32, 0.2, 0.11),
                    perceptual_roughness: 0.8,
                    ..default()
                }),
                mats.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.7, 0.3),
                    emissive: LinearRgba::rgb(3.0, 1.0, 0.25),
                    ..default()
                }),
            )
        };
        world.spawn((Mesh3d(shaft), MeshMaterial3d(wood), Transform::from_xyz(0.0, -0.3, 0.0), ChildOf(staff)));
        world.spawn((
            Mesh3d(gem),
            MeshMaterial3d(glow),
            StaffTip,
            Transform::from_xyz(0.0, 0.24, 0.0),
            ChildOf(staff),
        ));
    });
}

// ---------------------------------------------------------------- systems

/// Waits for the effects and the fox, spawns one warm-up instance of every
/// effect (pipelines compile before play), then starts the round.
#[allow(clippy::too_many_arguments)]
fn loading(
    mut commands: Commands,
    handles: Option<Res<FxHandles>>,
    effects: Res<Assets<FxEffectAsset>>,
    assets: Res<AssetServer>,
    ready: Res<FxReady>,
    mut run: ResMut<Run>,
    fx: Option<Res<Fx>>,
    warm: Query<Entity, With<Warmup>>,
    mut frames: Local<u32>,
    mode: Res<RunMode>,
) {
    if run.phase != Phase::Loading {
        return;
    }
    let Some(h) = handles else { return };
    let Some(fx) = fx else {
        let get = |h: &Handle<FxEffectAsset>| effects.get(h).map(|e| e.system.clone());
        let (Some(fireball), Some(explosion), Some(lightning), Some(frost), Some(death)) =
            (get(&h.fireball), get(&h.explosion), get(&h.lightning), get(&h.frost), get(&h.death))
        else {
            return;
        };
        if !assets.is_loaded_with_dependencies(&h.fox) {
            return;
        }
        let fx = Fx { fireball, explosion, lightning, frost, death };
        // In front of the starting view, behind the loading screen.
        let at = |x: f32| Transform::from_xyz(x, 0.5, 0.0);
        commands.spawn((ParticleEffect::shared(fx.fireball.clone(), 1), EffectControl::looping(), at(-3.0), Warmup));
        commands.spawn_one_shot(ParticleEffect::shared(fx.explosion.clone(), 1), at(-1.5)).insert(Warmup);
        commands.spawn_one_shot(ParticleEffect::shared(fx.lightning.clone(), 1), at(0.0)).insert(Warmup);
        commands.spawn_one_shot(ParticleEffect::shared(fx.frost.clone(), 1), at(1.5)).insert(Warmup);
        commands.spawn_one_shot(ParticleEffect::shared(fx.death.clone(), 1), at(3.0)).insert(Warmup);
        commands.insert_resource(fx);
        return;
    };
    let _ = fx;
    *frames += 1;
    if !ready.get() || *frames < 10 {
        return;
    }
    for e in &warm {
        commands.entity(e).despawn();
    }
    if !page_started(&mode) {
        return;
    }
    run.phase = Phase::Playing;
    run.toast("Defend the tower!");
}

/// In the browser the round starts when the page's Play button is pressed
/// (it sets `data-playing` on the body); elsewhere at once.
fn page_started(mode: &RunMode) -> bool {
    #[cfg(target_arch = "wasm32")]
    if !mode.bot() {
        return web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.body())
            .is_some_and(|b| b.has_attribute("data-playing"));
    }
    let _ = mode;
    true
}

fn clear_intent(mut intent: ResMut<Intent>) {
    *intent = Intent::default();
}

/// Seconds of simulation this frame: the game follows the effects clock
/// (fixed steps; manual and deterministic when recording).
pub fn game_dt(clock: &FxClock) -> f32 {
    clock.steps_this_frame() as f32 * clock.step
}

fn tick(clock: Res<FxClock>, mut run: ResMut<Run>) {
    let dt = game_dt(&clock);
    if let Some((_, t)) = &mut run.toast {
        *t -= dt;
        if *t <= 0.0 {
            run.toast = None;
        }
    }
    if run.phase != Phase::Playing {
        return;
    }
    run.time += dt;
    run.mana = (run.mana + MANA_REGEN * dt).min(MAX_MANA);
    for c in &mut run.cooldown {
        *c = (*c - dt).max(0.0);
    }
}

fn restart(
    mut commands: Commands,
    intent: Res<Intent>,
    mut run: ResMut<Run>,
    mut player: ResMut<Player>,
    round: Query<Entity, With<Round>>,
) {
    if !intent.restart || run.phase == Phase::Loading {
        return;
    }
    for e in &round {
        commands.entity(e).despawn();
    }
    *run = Run {
        phase: Phase::Playing,
        ..default()
    };
    *player = Player::default();
    run.toast("Defend the tower!");
}

fn move_player(
    clock: Res<FxClock>,
    intent: Res<Intent>,
    run: Res<Run>,
    mut player: ResMut<Player>,
    mut cam: Query<&mut Transform, With<MainCamera>>,
) {
    let dt = game_dt(&clock);
    player.yaw -= intent.look.x;
    player.pitch = (player.pitch - intent.look.y).clamp(-1.3, 1.3);
    if let Some(s) = intent.select {
        player.selected = s;
    }
    if run.phase == Phase::Playing {
        let (s, c) = player.yaw.sin_cos();
        let fwd = Vec3::new(-s, 0.0, -c);
        let right = Vec3::new(c, 0.0, -s);
        let speed = if intent.run { 9.0 } else { 5.5 };
        let mv = intent.mv.clamp_length_max(1.0);
        let mut p = player.pos + (right * mv.x + fwd * mv.y) * speed * dt;
        p.x = p.x.clamp(-BOUNDS, BOUNDS);
        p.z = p.z.clamp(-BOUNDS, BOUNDS);
        // Not through the tower.
        let to = Vec2::new(p.x - TOWER.x, p.z - TOWER.z);
        if to.length() < TOWER_RADIUS + 0.6 {
            let n = to.normalize_or(Vec2::Y) * (TOWER_RADIUS + 0.6);
            p.x = TOWER.x + n.x;
            p.z = TOWER.z + n.y;
        }
        player.pos = p;
    }
    for mut t in &mut cam {
        t.translation = player.eye();
        t.rotation = player.rotation();
    }
}

/// Where the crosshair meets the ground, at most `max` metres away
/// (horizontally) from the player.
pub fn aim_ground(player: &Player, max: f32) -> Vec3 {
    let o = player.eye();
    let d = player.forward();
    let hit = if d.y < -0.01 { o + d * (-o.y / d.y) } else { o + Vec3::new(d.x, 0.0, d.z).normalize_or(Vec3::NEG_Z) * max };
    let mut off = Vec3::new(hit.x - player.pos.x, 0.0, hit.z - player.pos.z);
    if off.length() > max {
        off = off.normalize() * max;
    }
    Vec3::new(player.pos.x + off.x, 0.0, player.pos.z + off.z)
}

/// The fox closest to the crosshair (within `cone` radians, `range` m).
fn aimed_enemy(
    from: Vec3,
    dir: Vec3,
    cone: f32,
    range: f32,
    enemies: &Query<(Entity, &Transform, &mut Enemy)>,
) -> Option<(Entity, Vec3)> {
    enemies
        .iter()
        .filter_map(|(e, t, en)| {
            let c = enemy_center(t, en);
            let v = c - from;
            let d = v.length();
            let ang = v.angle_between(dir);
            (d < range && ang < cone).then_some((e, c, ang + d * 0.002))
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(e, c, _)| (e, c))
}

#[allow(clippy::too_many_arguments)]
fn cast(
    mut commands: Commands,
    intent: Res<Intent>,
    mut run: ResMut<Run>,
    player: Res<Player>,
    fx: Option<Res<Fx>>,
    mut rng: ResMut<Rng>,
    tip: Query<&GlobalTransform, With<StaffTip>>,
    mut enemies: Query<(Entity, &Transform, &mut Enemy)>,
    turrets: Query<&Transform, (With<Turret>, Without<Enemy>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    let spell = intent.cast.or(intent.cast_selected.then_some(player.selected));
    let (Some(spell), Some(fx)) = (spell, fx) else { return };
    if run.phase != Phase::Playing {
        return;
    }
    let i = spell.index();
    if run.cooldown[i] > 0.0 {
        return;
    }
    if run.mana < spell.cost() {
        run.toast("Not enough mana");
        return;
    }
    let from = tip.single().map(|t| t.translation()).unwrap_or(player.eye());
    let dir = player.forward();
    match spell {
        Spell::Fireball => {
            // Towards what the crosshair points at, from the staff.
            let aim = player.eye() + dir * 40.0;
            let v = (aim - from).normalize() * 24.0;
            spawn_fireball(&mut commands, &fx, &mut rng, from, v, None, 50.0, FIRE_TINT);
        }
        Spell::Lightning => {
            let (target, end) = match aimed_enemy(player.eye(), dir, 0.16, 45.0, &enemies) {
                Some((e, c)) => (Some(e), c),
                None => (None, aim_ground(&player, 30.0)),
            };
            strike(&mut commands, &fx, &mut rng, from, end);
            if let Some(t) = target {
                chain_lightning(&mut commands, &fx, &mut rng, t, end, 55.0, &mut enemies);
            }
        }
        Spell::Frost => {
            let at = aim_ground(&player, 28.0);
            frost_nova(&mut commands, &fx, &mut rng, at, &mut enemies);
        }
        Spell::Turret => {
            let at = aim_ground(&player, 22.0);
            let n = turrets.iter().count();
            if n >= MAX_TURRETS {
                run.toast(format!("At most {MAX_TURRETS} turrets"));
                return;
            }
            if path_distance(at) < 2.4
                || Vec2::new(at.x - TOWER.x, at.z - TOWER.z).length() < TOWER_RADIUS + 1.5
                || turrets.iter().any(|t| t.translation.distance(at) < 2.5)
            {
                run.toast("Can't build there");
                return;
            }
            world::spawn_turret(&mut commands, &mut meshes, &mut mats, at);
        }
    }
    run.mana -= spell.cost();
    run.cooldown[i] = spell.cooldown();
    run.casts += 1;
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_fireball(
    commands: &mut Commands,
    fx: &Fx,
    rng: &mut Rng,
    from: Vec3,
    vel: Vec3,
    target: Option<Entity>,
    damage: f32,
    tint: Color,
) {
    let effect = with(ParticleEffect::shared(fx.fireball.clone(), rng.seed()), "User.Tint", tint);
    commands.spawn((
        Projectile { vel, target, damage, radius: 2.6, life: 3.0, tint },
        effect,
        EffectControl::looping(),
        Transform::from_translation(from),
        Round,
    ));
}

/// A lightning beam from `from` to `to` (+Y of the effect along the beam).
pub fn strike(commands: &mut Commands, fx: &Fx, rng: &mut Rng, from: Vec3, to: Vec3) {
    let d = to - from;
    let len = d.length().max(0.1);
    let effect = with(ParticleEffect::shared(fx.lightning.clone(), rng.seed()), "User.Length", len);
    commands.spawn_one_shot(
        effect,
        Transform::from_translation(from).with_rotation(Quat::from_rotation_arc(Vec3::Y, d / len)),
    );
}

/// Damages `first` and jumps to up to two more foxes within 6 m.
fn chain_lightning(
    commands: &mut Commands,
    fx: &Fx,
    rng: &mut Rng,
    first: Entity,
    at: Vec3,
    damage: f32,
    enemies: &mut Query<(Entity, &Transform, &mut Enemy)>,
) {
    let mut hit = vec![first];
    if let Ok((_, _, mut e)) = enemies.get_mut(first) {
        e.hp -= damage;
    }
    let mut from = at;
    for k in 0..2 {
        let next = enemies
            .iter()
            .filter(|(e, _, _)| !hit.contains(e))
            .map(|(e, t, en)| (e, enemy_center(t, en)))
            .filter(|(_, c)| c.distance(from) < 6.0)
            .min_by(|a, b| a.1.distance(from).total_cmp(&b.1.distance(from)));
        let Some((e, c)) = next else { break };
        strike(commands, fx, rng, from, c);
        if let Ok((_, _, mut en)) = enemies.get_mut(e) {
            en.hp -= damage * (0.7 - 0.2 * k as f32);
        }
        hit.push(e);
        from = c;
    }
}

fn frost_nova(
    commands: &mut Commands,
    fx: &Fx,
    rng: &mut Rng,
    at: Vec3,
    enemies: &mut Query<(Entity, &Transform, &mut Enemy)>,
) {
    let radius = 6.0;
    let effect = with(ParticleEffect::shared(fx.frost.clone(), rng.seed()), "User.Radius", radius);
    commands.spawn_one_shot(effect, Transform::from_translation(at + Vec3::Y * 0.02));
    for (_, t, mut e) in enemies.iter_mut() {
        if Vec2::new(t.translation.x - at.x, t.translation.z - at.z).length() < radius + e.radius() * 0.5 {
            e.hp -= 30.0;
            e.slow = 3.5;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn turrets(
    mut commands: Commands,
    clock: Res<FxClock>,
    run: Res<Run>,
    fx: Option<Res<Fx>>,
    mut rng: ResMut<Rng>,
    mut turrets: Query<(&mut Turret, &Transform)>,
    mut enemies: Query<(Entity, &Transform, &mut Enemy), Without<Turret>>,
) {
    let Some(fx) = fx else { return };
    if run.phase != Phase::Playing {
        return;
    }
    let dt = game_dt(&clock);
    for (mut tu, t) in &mut turrets {
        tu.cooldown -= dt;
        if tu.cooldown > 0.0 {
            continue;
        }
        let top = t.translation + Vec3::Y * 3.1;
        let target = enemies
            .iter()
            .map(|(e, tr, en)| (e, enemy_center(tr, en)))
            .filter(|(_, c)| c.distance(top) < 17.0)
            .min_by(|a, b| a.1.distance(top).total_cmp(&b.1.distance(top)));
        let Some((e, c)) = target else { continue };
        tu.cooldown = 1.1;
        tu.shots += 1;
        if tu.shots % 2 == 0 {
            strike(&mut commands, &fx, &mut rng, top, c);
            if let Ok((_, _, mut en)) = enemies.get_mut(e) {
                en.hp -= 22.0;
            }
        } else {
            let v = (c - top).normalize() * 16.0;
            spawn_fireball(&mut commands, &fx, &mut rng, top, v, Some(e), 30.0, TURRET_TINT);
        }
    }
}

fn spin_crystals(time: Res<Time>, mut q: Query<(&mut Transform, &TurretCrystal)>) {
    let t = time.elapsed_secs();
    for (mut tr, c) in &mut q {
        tr.rotation = Quat::from_rotation_y(t * 1.5);
        tr.translation.y = c.0 + 0.12 * (t * 2.0).sin();
    }
}

fn waves(
    mut commands: Commands,
    clock: Res<FxClock>,
    mut run: ResMut<Run>,
    anim: Res<FoxAnim>,
    enemies: Query<(), With<Enemy>>,
) {
    if run.phase != Phase::Playing {
        return;
    }
    let dt = game_dt(&clock);
    if run.to_spawn == 0 && enemies.is_empty() {
        run.next_wave -= dt;
        if run.next_wave <= 0.0 {
            run.wave += 1;
            run.to_spawn = 5 + 3 * run.wave;
            run.spawn_timer = 0.0;
            run.next_wave = 5.0;
            let w = run.wave;
            run.toast(format!("Wave {w}"));
        }
        return;
    }
    if run.to_spawn == 0 {
        return;
    }
    run.spawn_timer -= dt;
    if run.spawn_timer > 0.0 {
        return;
    }
    run.spawn_timer = (1.5 - 0.1 * run.wave as f32).max(0.55);
    run.to_spawn -= 1;
    let big = run.wave >= 2 && run.to_spawn % 4 == 0;
    let hp = (55.0 + 22.0 * run.wave as f32) * if big { 3.0 } else { 1.0 };
    let speed = (3.0 + 0.15 * run.wave as f32).min(5.0) * if big { 0.7 } else { 1.0 };
    let (p, _) = path_point(0.0);
    let scale = if big { 0.022 } else { 0.015 };
    commands
        .spawn((
            Enemy { hp, max_hp: hp, speed, dist: 0.0, slow: 0.0, big },
            Transform::from_translation(p),
            Visibility::default(),
            Round,
        ))
        .with_child((
            WorldAssetRoot(anim.scene.clone()),
            Transform::from_scale(Vec3::splat(scale)),
        ));
}

/// Starts the fox's run animation once its glTF scene is spawned.
fn play_fox_run(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    anim: Option<Res<FoxAnim>>,
    children: Query<&Children>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Some(anim) = anim else { return };
    for child in children.iter_descendants(ready.entity) {
        if let Ok(mut player) = players.get_mut(child) {
            player.play(anim.run).repeat().set_speed(1.3);
            commands.entity(child).insert(BlendGraphHandle(anim.graph.clone()));
        }
    }
}

fn walk(
    mut commands: Commands,
    clock: Res<FxClock>,
    mut run: ResMut<Run>,
    fx: Option<Res<Fx>>,
    mut rng: ResMut<Rng>,
    mut enemies: Query<(Entity, &mut Transform, &mut Enemy)>,
) {
    if run.phase != Phase::Playing {
        return;
    }
    let dt = game_dt(&clock);
    let end = path_length();
    for (e, mut t, mut en) in &mut enemies {
        en.slow = (en.slow - dt).max(0.0);
        let k = if en.slow > 0.0 { 0.4 } else { 1.0 };
        en.dist += en.speed * k * dt;
        if en.dist >= end {
            // At the tower: it hurts.
            run.tower_hp -= if en.big { 20.0 } else { 8.0 };
            if let Some(fx) = &fx {
                let at = t.translation + Vec3::Y * 0.5;
                commands.spawn_one_shot(
                    with(ParticleEffect::shared(fx.explosion.clone(), rng.seed()), "User.Tint", Color::linear_rgb(1.0, 0.25, 0.1)),
                    Transform::from_translation(at),
                );
            }
            commands.entity(e).despawn();
            if run.tower_hp <= 0.0 {
                run.tower_hp = 0.0;
                run.phase = Phase::GameOver;
            }
            continue;
        }
        let (p, dir) = path_point(en.dist);
        t.translation = p;
        // The fox model faces +Z.
        let want = Quat::from_rotation_y(dir.x.atan2(dir.z));
        t.rotation = t.rotation.slerp(want, (dt * 10.0).min(1.0));
    }
}

#[allow(clippy::too_many_arguments)]
fn projectiles(
    mut commands: Commands,
    clock: Res<FxClock>,
    fx: Option<Res<Fx>>,
    mut rng: ResMut<Rng>,
    mut shots: Query<(Entity, &mut Projectile, &mut Transform), Without<Enemy>>,
    mut enemies: Query<(Entity, &Transform, &mut Enemy)>,
) {
    let Some(fx) = fx else { return };
    let dt = game_dt(&clock);
    for (e, mut p, mut t) in &mut shots {
        if let Some(target) = p.target {
            match enemies.get(target) {
                Ok((_, et, en)) => {
                    let to = enemy_center(et, en) - t.translation;
                    let speed = p.vel.length();
                    p.vel = p.vel.lerp(to.normalize_or_zero() * speed, (dt * 6.0).min(1.0)).normalize_or_zero() * speed;
                }
                Err(_) => p.target = None,
            }
        }
        p.life -= dt;
        let step = p.vel * dt;
        let next = t.translation + step;
        let hit_enemy = enemies.iter().any(|(_, et, en)| {
            let c = enemy_center(et, en);
            // Closest point of this step's segment to the fox.
            let s = (c - t.translation).dot(step) / step.length_squared().max(1e-6);
            let q = t.translation + step * s.clamp(0.0, 1.0);
            q.distance(c) < en.radius()
        });
        let hit_tower = Vec2::new(next.x - TOWER.x, next.z - TOWER.z).length() < TOWER_RADIUS && next.y < 9.0;
        if hit_enemy || next.y <= 0.05 || p.life <= 0.0 || hit_tower {
            let at = Vec3::new(next.x, next.y.max(0.15), next.z);
            commands.spawn_one_shot(
                with(ParticleEffect::shared(fx.explosion.clone(), rng.seed()), "User.Tint", p.tint),
                Transform::from_translation(at),
            );
            for (_, et, mut en) in enemies.iter_mut() {
                let d = enemy_center(et, &en).distance(at);
                if d < p.radius + en.radius() * 0.5 {
                    en.hp -= p.damage * (1.0 - 0.5 * (d / p.radius).min(1.0));
                }
            }
            commands.entity(e).despawn();
        } else {
            t.translation = next;
        }
    }
}

fn deaths(
    mut commands: Commands,
    mut run: ResMut<Run>,
    fx: Option<Res<Fx>>,
    mut rng: ResMut<Rng>,
    enemies: Query<(Entity, &Transform, &Enemy)>,
) {
    let Some(fx) = fx else { return };
    for (e, t, en) in &enemies {
        if en.hp > 0.0 {
            continue;
        }
        commands.spawn_one_shot(
            ParticleEffect::shared(fx.death.clone(), rng.seed()),
            Transform::from_translation(t.translation).with_scale(Vec3::splat(if en.big { 1.4 } else { 1.0 })),
        );
        commands.entity(e).despawn();
        run.kills += 1;
        run.score += (10 + 2 * run.wave) * if en.big { 3 } else { 1 };
    }
}

/// Lowers the quality level while the frame rate stays under ~52 fps (and
/// raises it again after 20 s of headroom); applies render scale, effects
/// quality and bloom.
#[allow(clippy::too_many_arguments)]
fn adapt_quality(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mode: Res<RunMode>,
    run: Res<Run>,
    mut q: ResMut<Quality>,
    mut windows: Query<&mut Window>,
    mut fxq: ResMut<FxQuality>,
    cams: Query<Entity, With<MainCamera>>,
) {
    if mode.headless() {
        return;
    }
    let Ok(mut window) = windows.single_mut() else { return };
    let base = *q.base_scale.get_or_insert(window.resolution.base_scale_factor());
    if run.phase == Phase::Playing {
        q.frames += 1;
        q.acc += time.delta_secs();
        if q.acc >= 2.0 {
            q.fps = q.frames as f32 / q.acc;
            if q.fps < 52.0 && q.level + 1 < LEVELS.len() {
                q.level += 1;
                q.calm = 0.0;
            } else if q.fps > 59.0 {
                q.calm += q.acc;
                if q.calm > 20.0 && q.level > 0 {
                    q.level -= 1;
                    q.calm = -60.0;
                }
            }
            q.frames = 0;
            q.acc = 0.0;
        }
    }
    if q.applied == Some(q.level) {
        return;
    }
    q.applied = Some(q.level);
    let (cap, level, bloom) = LEVELS[q.level];
    let scale = base.min(cap);
    // Render scale only in the browser (the canvas keeps its CSS size; a
    // native window would resize instead).
    if cfg!(target_arch = "wasm32") {
        window.resolution.set_scale_factor_override((scale != base).then_some(scale));
    }
    fxq.0 = level;
    for c in &cams {
        if bloom {
            commands.entity(c).insert(Bloom::NATURAL);
        } else {
            commands.entity(c).remove::<Bloom>();
        }
    }
}
