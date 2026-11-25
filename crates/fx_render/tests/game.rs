//! The game API in a headless Bevy app on the GPU: one-shots end and
//! despawn, attached effects follow their parent, typed `User.` parameters,
//! pool reuse without new GPU instances, and scalability culling.

use std::sync::{Arc, Mutex};

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::window::ExitCondition;
use fx_core::CookedSystem;
use fx_core::types::{Value, ValueType};
use fx_render::{
    CullReason, EffectControl, EffectMode, FxClock, FxCommandsExt, FxPlugin, FxPoolStats, FxState,
    ParamError, ParticleEffect,
};

/// Bevy apps with a GPU, one at a time.
static GPU: Mutex<()> = Mutex::new(());

fn cook(src: &str) -> Arc<CookedSystem> {
    let c = fx_core::check_source("test.fx.ron", src);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    Arc::new(c.cooked.unwrap_or_else(|| panic!("{errs:?}")))
}

/// A burst of short-lived sparks at t = 0, plus whatever `extra` adds at the system level.
fn burst(extra: &str) -> Arc<CookedSystem> {
    cook(&format!(
        r#"(
    name: "Burst",
    {extra}
    user_parameters: {{ "User.Speed": Float(2.0), "User.Tint": Color((1.0, 0.5, 0.2, 1.0)) }},
    emitters: [(
        name: "Sparks",
        max_particles: 256,
        emitter_update: [
            (module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int(64), "Spawn Time": Float(0.0) }}),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": Float(0.3), "Color": Link(Color, "User.Tint") }}),
            (module: "Add Velocity in Cone", inputs: {{ "Velocity Strength": Link(Float, "User.Speed") }}),
        ],
        particle_update: [(module: "Particle State"), (module: "Solve Forces and Velocity")],
        renderers: [Sprite(())],
    )],
)"#
    ))
}

/// A steady fountain (Spawn Rate: unbounded duration).
fn fountain(extra: &str) -> Arc<CookedSystem> {
    cook(&format!(
        r#"(
    name: "Fountain",
    {extra}
    emitters: [(
        name: "Drops",
        max_particles: 512,
        emitter_update: [(module: "Spawn Rate", inputs: {{ "Spawn Rate": Float(100.0) }})],
        particle_spawn: [(module: "Initialize Particle", inputs: {{ "Lifetime": Float(0.5) }})],
        particle_update: [(module: "Particle State")],
        renderers: [Sprite(())],
    )],
)"#
    ))
}

fn app() -> App {
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
        FxPlugin::default(),
    ))
    .insert_resource(FxClock::manual(1.0 / 60.0));
    app.finish();
    app.cleanup();
    let image = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(Image::new_target_texture(
            64,
            64,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            None,
        ))
    };
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::from(image),
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    app
}

/// Runs `frames` frames of one simulation step each.
fn frames(app: &mut App, frames: u32) {
    for _ in 0..frames {
        app.world_mut().resource_mut::<FxClock>().request_steps(1);
        app.update();
    }
}

fn render_effect<R>(app: &App, e: Entity, f: impl FnOnce(&fx_gpu::GpuEffect, Mat4) -> R) -> Option<R> {
    let state = app.sub_app(RenderApp).world().get_resource::<FxState>()?;
    state.effect(e).map(|(g, m)| f(g, m))
}

/// `(living, next UniqueID)` of each emitter of effect `e`, read back.
fn counts(app: &App, e: Entity) -> Vec<(u32, u32)> {
    let device = app.world().resource::<bevy::render::renderer::RenderDevice>();
    let queue = app.world().resource::<bevy::render::renderer::RenderQueue>();
    render_effect(app, e, |g, _| g.read_counts(device.wgpu_device(), queue)).unwrap()
}

fn spawn(app: &mut App, bundle: impl Bundle) -> Entity {
    app.world_mut().spawn(bundle).id()
}

#[test]
fn one_shot_despawns_when_done_and_looping_stays() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let sys = burst("");
    assert_eq!(sys.duration, Some(0.0), "bursts only: duration is the last burst");
    let one = spawn(
        &mut app,
        (ParticleEffect::shared(sys.clone(), 1), EffectControl::one_shot(), Transform::default()),
    );
    let looping = spawn(&mut app, (ParticleEffect::shared(sys.clone(), 2), Transform::default()));
    frames(&mut app, 3);
    assert!(app.world().get_entity(one).is_ok());
    let alive: u32 = counts(&app, one).iter().map(|c| c.0).sum();
    assert!(alive > 0, "one-shot spawned nothing");
    // 0.3 s lifetimes: gone after ~18 steps, plus a few frames of read-back lag.
    let mut despawned_at = None;
    for f in 0..90 {
        frames(&mut app, 1);
        if app.world().get_entity(one).is_err() {
            despawned_at = Some(f + 3);
            break;
        }
    }
    let at = despawned_at.expect("one-shot never despawned");
    eprintln!("one-shot despawned after {at} frames");
    assert!(at >= 18, "despawned before its particles died ({at})");
    let c = app.world().get::<EffectControl>(looping).unwrap();
    assert!(!c.is_finished() && c.is_active());
    frames(&mut app, 2);
    let stats = app.world().resource::<FxPoolStats>().get();
    assert_eq!((stats.live, stats.idle), (1, 1), "{stats:?}");
}

#[test]
fn attached_effect_follows_its_parent() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let sys = fountain("");
    let parent = spawn(&mut app, Transform::from_xyz(0.0, 0.0, 0.0));
    let effect = {
        let mut commands = app.world_mut().commands();
        let id = commands
            .attach_effect(
                parent,
                ParticleEffect::shared(sys, 3),
                EffectMode::Looping,
                Transform::from_xyz(0.0, 1.0, 0.0),
            )
            .id();
        app.world_mut().flush();
        id
    };
    frames(&mut app, 2);
    for (i, x) in [2.0f32, -3.5, 7.25].into_iter().enumerate() {
        app.world_mut().get_mut::<Transform>(parent).unwrap().translation =
            Vec3::new(x, 0.5 * i as f32, -x);
        frames(&mut app, 1);
        let want = Vec3::new(x, 0.5 * i as f32 + 1.0, -x);
        let gt = app.world().get::<GlobalTransform>(effect).unwrap().translation();
        assert!(gt.distance(want) < 1e-5, "main world: {gt} vs {want}");
        let rt = render_effect(&app, effect, |_, m| m.w_axis.truncate()).unwrap();
        assert!(rt.distance(want) < 1e-5, "render world: {rt} vs {want}");
    }
    // Despawning the parent takes the attached effect with it.
    app.world_mut().entity_mut(parent).despawn();
    frames(&mut app, 2);
    assert!(app.world().get_entity(effect).is_err());
}

#[test]
fn user_parameters_are_typed() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut effect = ParticleEffect::shared(burst(""), 1);
    assert_eq!(
        effect.set_param("User.Speed", 3i32),
        Err(ParamError::WrongType {
            name: "User.Speed".into(),
            expected: ValueType::Float,
            found: ValueType::Int,
        })
    );
    assert_eq!(
        effect.set_param("User.Tint", Vec3::ONE),
        Err(ParamError::WrongType {
            name: "User.Tint".into(),
            expected: ValueType::Color,
            found: ValueType::Vec3,
        })
    );
    let unknown = effect.set_param("User.Nope", 1.0).unwrap_err();
    assert!(matches!(unknown, ParamError::Unknown { .. }));
    assert_eq!(unknown.to_string(), "effect \"Burst\" has no user parameter \"User.Nope\"");
    // Unchanged by the failed calls.
    assert_eq!(effect.param("Speed"), Some(Value::Float(2.0)));
    effect.set_param("Speed", 5.5).unwrap();
    effect.set_param("User.Tint", Color::linear_rgb(0.0, 1.0, 0.0)).unwrap();

    // Reaches the GPU instance, and changes per frame.
    let mut app = app();
    let e = spawn(&mut app, (effect, Transform::default()));
    frames(&mut app, 2);
    let user = render_effect(&app, e, |g, _| g.user.clone()).unwrap();
    assert_eq!(user[0], Value::Float(5.5));
    assert_eq!(user[1], Value::Color([0.0, 1.0, 0.0, 1.0]));
    app.world_mut()
        .get_mut::<ParticleEffect>(e)
        .unwrap()
        .set_param("User.Speed", 9.0)
        .unwrap();
    frames(&mut app, 1);
    let user = render_effect(&app, e, |g, _| g.user.clone()).unwrap();
    assert_eq!(user[0], Value::Float(9.0));
}

#[test]
fn pool_reuses_instances_without_new_gpu_allocations() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let sys = burst("");
    let wave = |app: &mut App, seed: u32| -> Vec<Entity> {
        let mut commands = app.world_mut().commands();
        let ids: Vec<Entity> = (0..24)
            .map(|i| {
                commands
                    .spawn_one_shot(
                        ParticleEffect::shared(sys.clone(), seed + i),
                        Transform::from_xyz(i as f32 * 0.2 - 2.4, 0.0, 0.0),
                    )
                    .id()
            })
            .collect();
        app.world_mut().flush();
        ids
    };
    let wait_gone = |app: &mut App, ids: &[Entity]| {
        for _ in 0..120 {
            frames(app, 1);
            if ids.iter().all(|&e| app.world().get_entity(e).is_err()) {
                frames(app, 1);
                return;
            }
        }
        panic!("one-shots never ended");
    };
    let ids = wave(&mut app, 0);
    wait_gone(&mut app, &ids);
    let warm = app.world().resource::<FxPoolStats>().get();
    eprintln!("after the first wave: {warm:?}");
    assert_eq!((warm.programs, warm.instances_created, warm.idle), (1, 24, 24));
    for w in 1..=4 {
        let ids = wave(&mut app, 100 * w);
        frames(&mut app, 2);
        let mid = app.world().resource::<FxPoolStats>().get();
        assert_eq!((mid.live, mid.idle), (24, 0), "{mid:?}");
        wait_gone(&mut app, &ids);
        let s = app.world().resource::<FxPoolStats>().get();
        eprintln!("after wave {w}: {s:?}");
        assert_eq!(s.programs, 1, "recompiled");
        assert_eq!(s.instances_created, 24, "new GPU instances after warm-up");
        assert_eq!(s.instances_reused, 24 * w as u64);
        assert_eq!(s.gpu_bytes, warm.gpu_bytes, "GPU memory grew");
    }
}

#[test]
fn scalability_culls_by_distance_and_budget() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let sys = fountain(
        "scalability: (max_distance: 20.0, max_instances: 3, bounds_radius: 1.0, \
         spawn_count_scale_by_distance: [(5.0, 1.0), (15.0, 0.5)], \
         quality: (Low: (spawn_count_scale: 0.5, max_instances: 1))),",
    );
    // The camera is at (0, 2, 8): five effects along -z, nearest first.
    let near: Vec<Entity> = (0..5)
        .map(|i| {
            spawn(
                &mut app,
                (ParticleEffect::shared(sys.clone(), i), Transform::from_xyz(0.0, 0.0, -(i as f32))),
            )
        })
        .collect();
    let far = spawn(&mut app, (ParticleEffect::shared(sys.clone(), 9), Transform::from_xyz(0.0, 0.0, -40.0)));
    let far_shot = {
        let mut commands = app.world_mut().commands();
        let id = commands
            .spawn_one_shot(ParticleEffect::shared(sys.clone(), 10), Transform::from_xyz(30.0, 0.0, 0.0))
            .id();
        app.world_mut().flush();
        id
    };
    frames(&mut app, 3);
    let culled = |app: &App, e: Entity| app.world().get::<EffectControl>(e).unwrap().culled();
    assert_eq!(culled(&app, far), Some(CullReason::Distance));
    assert!(app.world().get_entity(far_shot).is_err(), "culled one-shot not ended");
    let reasons: Vec<_> = near.iter().map(|&e| culled(&app, e)).collect();
    assert_eq!(
        reasons,
        [None, None, None, Some(CullReason::Budget), Some(CullReason::Budget)]
    );
    // Culled instances do not simulate.
    let frame_of = |app: &App, e: Entity| render_effect(app, e, |g, _| g.frame).unwrap();
    let (a, b) = (frame_of(&app, near[0]), frame_of(&app, far));
    frames(&mut app, 5);
    assert_eq!(frame_of(&app, near[0]), a + 5);
    assert_eq!(frame_of(&app, far), b, "a culled effect simulated");
    // Spawn scale by distance (camera at distance ~8.2..12.2 from these).
    let scale = |app: &App, e: Entity| app.world().get::<EffectControl>(e).unwrap().spawn_scale();
    let s0 = scale(&app, near[0]);
    assert!(s0 < 1.0 && s0 > 0.5, "{s0}");
    assert_eq!(render_effect(&app, near[0], |g, _| g.spawn_scale).unwrap(), s0);
    // Low quality: half the spawns, one instance.
    app.insert_resource(fx_render::FxQuality(fx_render::QualityLevel::Low));
    frames(&mut app, 1);
    assert!((scale(&app, near[0]) - s0 * 0.5).abs() < 1e-5);
    let kept = near.iter().filter(|&&e| culled(&app, e).is_none()).count();
    assert_eq!(kept, 1);
    // Scalability off: nothing culled.
    app.insert_resource(fx_render::FxScalability { enabled: false });
    frames(&mut app, 1);
    assert!(near.iter().chain([&far]).all(|&e| culled(&app, e).is_none()));
}

#[test]
fn deactivate_reset_and_fire() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = app();
    let e = spawn(&mut app, (ParticleEffect::shared(fountain(""), 4), Transform::default()));
    frames(&mut app, 30);
    let next_id = |app: &App| counts(app, e)[0].1;
    let before = next_id(&app);
    app.world_mut().get_mut::<EffectControl>(e).unwrap().deactivate();
    frames(&mut app, 30);
    assert_eq!(next_id(&app), before, "spawned while deactivated");
    app.world_mut().get_mut::<EffectControl>(e).unwrap().reset();
    frames(&mut app, 1);
    assert_eq!(render_effect(&app, e, |g, _| g.frame).unwrap(), 1, "reset restarts at frame 0");
    assert!(next_id(&app) < before);
}
