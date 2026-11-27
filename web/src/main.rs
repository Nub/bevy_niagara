//! Browser demo of the particle plugin: the campfire and sparks library
//! effects, loaded over HTTP as `.fx.ron` assets (cooked in the page) and
//! simulated and drawn on the GPU through WebGPU. Drag (mouse or one finger)
//! orbits, wheel or pinch zooms.
//!
//! Built by `web/build.sh`. `?limits=default` in the page URL caps the device
//! at WebGPU's default limits (what the strictest browsers give), to test the
//! fallbacks on a desktop GPU. Natively, `cargo run -p fx_web` opens a window
//! with the same scene once `web/build.sh` has copied the assets to web/dist.

use bevy::camera::Hdr;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::input::touch::Touches;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use fx_render::{FxEffectAsset, FxPlugin, ParticleEffect};

/// The page's query string (`?limits=default`), empty natively.
fn query() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.location().search().ok())
            .unwrap_or_default()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("FX_WEB_QUERY").unwrap_or_default()
    }
}

fn main() {
    let mut wgpu = WgpuSettings::default();
    if query().contains("limits=default") {
        wgpu.constrained_limits = Some(bevy::render::settings::WgpuLimits::default());
    }
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "animatey: particles in the browser".into(),
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
                .set(AssetPlugin {
                    // Next to index.html in the browser; web/dist/assets natively.
                    file_path: if cfg!(target_arch = "wasm32") { "assets" } else { "dist/assets" }.into(),
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    ..default()
                }),
            FxPlugin::default(),
        ))
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)))
        .add_systems(Startup, setup)
        .add_systems(Update, (spawn_loaded, restart_sparks, orbit))
        .add_systems(Update, forced_panic.run_if(|| query().contains("fail=panic")))
        .run();
}

/// `?fail=panic`: panics after 2 s, to check the page's diagnostics overlay
/// shows a wasm panic.
fn forced_panic(time: Res<Time>) {
    if time.elapsed_secs() > 2.0 {
        panic!("forced panic (?fail=panic)");
    }
}

#[derive(Component)]
struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
}

/// An effect to spawn once its asset is loaded, at a position.
#[derive(Component)]
struct Pending(Handle<FxEffectAsset>);

/// The sparks burst stops after 10 s; it is spawned again every `PERIOD`.
#[derive(Component)]
struct Sparks;
const SPARKS_PERIOD: f32 = 10.0;

#[derive(Resource)]
struct SparksAsset(Handle<FxEffectAsset>, f32);

fn setup(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let orbit = Orbit {
        yaw: 0.0,
        pitch: 0.35,
        distance: 7.0,
        target: Vec3::new(0.0, 1.0, 0.0),
    };
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        bevy::core_pipeline::tonemapping::Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        orbit_transform(&orbit),
        orbit,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(12.0, 12.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.08, 0.09),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 800.0,
            ..default()
        },
        Transform::from_xyz(3.0, 6.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Pending(assets.load("effects/campfire.fx.ron")),
        Transform::from_xyz(-1.3, 0.0, 0.0),
    ));
    let sparks = assets.load("effects/sparks_burst.fx.ron");
    commands.spawn((Pending(sparks.clone()), Sparks, Transform::from_xyz(1.6, 0.0, 0.0)));
    commands.insert_resource(SparksAsset(sparks, 0.0));
}

fn spawn_loaded(
    mut commands: Commands,
    pending: Query<(Entity, &Pending)>,
    effects: Res<Assets<FxEffectAsset>>,
) {
    for (entity, p) in &pending {
        if let Some(fx) = effects.get(&p.0) {
            commands
                .entity(entity)
                .remove::<Pending>()
                .insert(ParticleEffect::shared(fx.system.clone(), 1));
        }
    }
}

fn restart_sparks(
    mut commands: Commands,
    time: Res<Time>,
    mut res: ResMut<SparksAsset>,
    sparks: Query<(Entity, &Transform), (With<Sparks>, With<ParticleEffect>)>,
) {
    res.1 += time.delta_secs();
    if res.1 < SPARKS_PERIOD {
        return;
    }
    res.1 = 0.0;
    for (e, t) in &sparks {
        commands.entity(e).despawn();
        commands.spawn((Pending(res.0.clone()), Sparks, *t));
    }
}

fn orbit_transform(o: &Orbit) -> Transform {
    let dir = Vec3::new(
        o.yaw.sin() * o.pitch.cos(),
        o.pitch.sin(),
        o.yaw.cos() * o.pitch.cos(),
    );
    Transform::from_translation(o.target + dir * o.distance).looking_at(o.target, Vec3::Y)
}

fn orbit(
    mut cams: Query<(&mut Orbit, &mut Transform)>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    touches: Res<Touches>,
    mut last_pinch: Local<Option<f32>>,
) {
    let mut rotate = Vec2::ZERO;
    let mut zoom = 1.0;
    if buttons.pressed(MouseButton::Left) {
        rotate += motion.delta;
    }
    zoom *= (1.0 - scroll.delta.y * 0.001).clamp(0.5, 2.0);
    let active: Vec<_> = touches.iter().collect();
    match active.as_slice() {
        [t] => {
            rotate += t.delta();
            *last_pinch = None;
        }
        [a, b, ..] => {
            let d = a.position().distance(b.position());
            if let Some(prev) = *last_pinch
                && d > 1.0
            {
                zoom *= prev / d;
            }
            *last_pinch = Some(d);
        }
        [] => *last_pinch = None,
    }
    for (mut o, mut t) in &mut cams {
        o.yaw -= rotate.x * 0.005;
        o.pitch = (o.pitch + rotate.y * 0.005).clamp(-0.2, 1.4);
        o.distance = (o.distance * zoom).clamp(2.0, 30.0);
        *t = orbit_transform(&o);
    }
}
