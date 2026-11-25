//! The game API in a headless Bevy app on the GPU: one-shots end and
//! despawn, attached effects follow their parent, typed `User.` parameters,
//! pool reuse without new GPU instances, and scalability culling.

use std::sync::Arc;

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

/// One app (one GPU device) runs every scenario in turn: several GPU apps in
/// one process crash the driver at exit.
#[test]
fn game_api() {
    let mut app = app();
    let scenarios: [(&str, fn(&mut App)); 7] = [
        ("one-shot despawns, looping stays", one_shot_despawns_when_done_and_looping_stays),
        ("attached effect follows its parent", attached_effect_follows_its_parent),
        ("user parameters are typed", user_parameters_are_typed),
        ("pool reuse without new GPU instances", pool_reuses_instances_without_new_gpu_allocations),
        ("scalability culls by distance and budget", scalability_culls_by_distance_and_budget),
        ("deactivate, reset and fire", deactivate_reset_and_fire),
        ("owner velocity drives Spawn Per Unit", owner_velocity_drives_spawn_per_unit),
    ];
    for (name, run) in scenarios {
        eprintln!("scenario: {name}");
        run(&mut app);
        clean(&mut app);
    }
}

/// Despawns every effect (and its parents' trees) and restores the settings.
fn clean(app: &mut App) {
    let effects: Vec<Entity> = app
        .world_mut()
        .query_filtered::<Entity, With<ParticleEffect>>()
        .iter(app.world())
        .collect();
    let roots: Vec<Entity> = effects
        .into_iter()
        .map(|mut e| {
            while let Some(p) = app.world().get::<ChildOf>(e) {
                e = p.parent();
            }
            e
        })
        .collect();
    for e in roots {
        if let Ok(em) = app.world_mut().get_entity_mut(e) {
            em.despawn();
        }
    }
    app.insert_resource(fx_render::FxQuality::default());
    app.insert_resource(fx_render::FxScalability::default());
    frames(app, 3);
    assert_eq!(app.world().resource::<FxPoolStats>().get().live, 0, "effects left live");
}

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

/// Never dropped: tearing down several GPU apps in one process crashes at
/// exit (SIGSEGV in the driver), so each test leaks its app.
fn app() -> std::mem::ManuallyDrop<App> {
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
    std::mem::ManuallyDrop::new(app)
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

fn one_shot_despawns_when_done_and_looping_stays(app: &mut App) {
    let sys = burst("");
    assert_eq!(sys.duration, Some(0.0), "bursts only: duration is the last burst");
    let one = spawn(
        app,
        (ParticleEffect::shared(sys.clone(), 1), EffectControl::one_shot(), Transform::default()),
    );
    let looping = spawn(app, (ParticleEffect::shared(sys.clone(), 2), Transform::default()));
    frames(app, 3);
    assert!(app.world().get_entity(one).is_ok());
    let alive: u32 = counts(app, one).iter().map(|c| c.0).sum();
    assert!(alive > 0, "one-shot spawned nothing");
    // 0.3 s lifetimes: gone after ~18 steps, plus a few frames of read-back lag.
    let mut despawned_at = None;
    for f in 0..90 {
        frames(app, 1);
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
    frames(app, 2);
    let stats = app.world().resource::<FxPoolStats>().get();
    assert_eq!((stats.live, stats.idle), (1, 1), "{stats:?}");
}

fn attached_effect_follows_its_parent(app: &mut App) {
    let sys = fountain("");
    let parent = spawn(app, Transform::from_xyz(0.0, 0.0, 0.0));
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
    frames(app, 2);
    for (i, x) in [2.0f32, -3.5, 7.25].into_iter().enumerate() {
        app.world_mut().get_mut::<Transform>(parent).unwrap().translation =
            Vec3::new(x, 0.5 * i as f32, -x);
        frames(app, 1);
        let want = Vec3::new(x, 0.5 * i as f32 + 1.0, -x);
        let gt = app.world().get::<GlobalTransform>(effect).unwrap().translation();
        assert!(gt.distance(want) < 1e-5, "main world: {gt} vs {want}");
        let rt = render_effect(app, effect, |_, m| m.w_axis.truncate()).unwrap();
        assert!(rt.distance(want) < 1e-5, "render world: {rt} vs {want}");
    }
    // Despawning the parent takes the attached effect with it.
    app.world_mut().entity_mut(parent).despawn();
    frames(app, 2);
    assert!(app.world().get_entity(effect).is_err());
}

fn user_parameters_are_typed(app: &mut App) {
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
    let e = spawn(app, (effect, Transform::default()));
    frames(app, 2);
    let user = render_effect(app, e, |g, _| g.user.clone()).unwrap();
    assert_eq!(user[0], Value::Float(5.5));
    assert_eq!(user[1], Value::Color([0.0, 1.0, 0.0, 1.0]));
    app.world_mut()
        .get_mut::<ParticleEffect>(e)
        .unwrap()
        .set_param("User.Speed", 9.0)
        .unwrap();
    frames(app, 1);
    let user = render_effect(app, e, |g, _| g.user.clone()).unwrap();
    assert_eq!(user[0], Value::Float(9.0));
}

fn pool_reuses_instances_without_new_gpu_allocations(app: &mut App) {
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
    let base = app.world().resource::<FxPoolStats>().get();
    let ids = wave(app, 0);
    wait_gone(app, &ids);
    let warm = app.world().resource::<FxPoolStats>().get();
    eprintln!("after the first wave: {warm:?}");
    assert_eq!(
        (warm.programs - base.programs, warm.instances_created - base.instances_created),
        (1, 24)
    );
    for w in 1..=4 {
        let ids = wave(app, 100 * w);
        frames(app, 2);
        let mid = app.world().resource::<FxPoolStats>().get();
        assert_eq!(mid.live, 24, "{mid:?}");
        assert_eq!(mid.instances_created, warm.instances_created, "{mid:?}");
        wait_gone(app, &ids);
        let s = app.world().resource::<FxPoolStats>().get();
        eprintln!("after wave {w}: {s:?}");
        assert_eq!(s.programs, warm.programs, "recompiled");
        assert_eq!(s.instances_created, warm.instances_created, "new GPU instances after warm-up");
        assert_eq!(s.instances_reused - warm.instances_reused, 24 * w as u64);
        assert_eq!(s.gpu_bytes, warm.gpu_bytes, "GPU memory grew");
    }
}

fn scalability_culls_by_distance_and_budget(app: &mut App) {
    let sys = fountain(
        "scalability: (max_distance: 20.0, max_instances: 3, bounds_radius: 1.0, \
         spawn_count_scale_by_distance: [(5.0, 1.0), (15.0, 0.5)], \
         quality: (Low: (spawn_count_scale: 0.5, max_instances: 1))),",
    );
    // The camera is at (0, 2, 8): five effects along -z, nearest first.
    let near: Vec<Entity> = (0..5)
        .map(|i| {
            spawn(
                app,
                (ParticleEffect::shared(sys.clone(), i), Transform::from_xyz(0.0, 0.0, -(i as f32))),
            )
        })
        .collect();
    let far = spawn(app, (ParticleEffect::shared(sys.clone(), 9), Transform::from_xyz(0.0, 0.0, -40.0)));
    let far_shot = {
        let mut commands = app.world_mut().commands();
        let id = commands
            .spawn_one_shot(ParticleEffect::shared(sys.clone(), 10), Transform::from_xyz(30.0, 0.0, 0.0))
            .id();
        app.world_mut().flush();
        id
    };
    frames(app, 3);
    let culled = |app: &App, e: Entity| app.world().get::<EffectControl>(e).unwrap().culled();
    assert_eq!(culled(app, far), Some(CullReason::Distance));
    assert!(app.world().get_entity(far_shot).is_err(), "culled one-shot not ended");
    let reasons: Vec<_> = near.iter().map(|&e| culled(app, e)).collect();
    assert_eq!(
        reasons,
        [None, None, None, Some(CullReason::Budget), Some(CullReason::Budget)]
    );
    // Culled instances do not simulate.
    let frame_of = |app: &App, e: Entity| render_effect(app, e, |g, _| g.frame).unwrap();
    let (a, b) = (frame_of(app, near[0]), frame_of(app, far));
    frames(app, 5);
    assert_eq!(frame_of(app, near[0]), a + 5);
    assert_eq!(frame_of(app, far), b, "a culled effect simulated");
    // Spawn scale by distance (camera at distance ~8.2..12.2 from these).
    let scale = |app: &App, e: Entity| app.world().get::<EffectControl>(e).unwrap().spawn_scale();
    let s0 = scale(app, near[0]);
    assert!(s0 < 1.0 && s0 > 0.5, "{s0}");
    assert_eq!(render_effect(app, near[0], |g, _| g.spawn_scale).unwrap(), s0);
    // Low quality: half the spawns, one instance.
    app.insert_resource(fx_render::FxQuality(fx_render::QualityLevel::Low));
    frames(app, 1);
    assert!((scale(app, near[0]) - s0 * 0.5).abs() < 1e-5);
    let kept = near.iter().filter(|&&e| culled(app, e).is_none()).count();
    assert_eq!(kept, 1);
    // Scalability off: nothing culled.
    app.insert_resource(fx_render::FxScalability { enabled: false });
    frames(app, 1);
    assert!(near.iter().chain([&far]).all(|&e| culled(app, e).is_none()));
}

fn deactivate_reset_and_fire(app: &mut App) {
    let e = spawn(app, (ParticleEffect::shared(fountain(""), 4), Transform::default()));
    frames(app, 30);
    let next_id = |app: &App| counts(app, e)[0].1;
    let before = next_id(app);
    app.world_mut().get_mut::<EffectControl>(e).unwrap().deactivate();
    frames(app, 30);
    assert_eq!(next_id(app), before, "spawned while deactivated");
    app.world_mut().get_mut::<EffectControl>(e).unwrap().reset();
    frames(app, 1);
    assert_eq!(render_effect(app, e, |g, _| g.frame).unwrap(), 1, "reset restarts at frame 0");
    assert!(next_id(app) < before);
}

/// Engine.Owner.Velocity follows the entity: zero on the first frame, the
/// move per simulated second after, zero across a teleport, in effect space,
/// and for an effect attached to a moving parent. Spawn Per Unit spawns by it.
fn owner_velocity_drives_spawn_per_unit(app: &mut App) {
    let sys = cook(
        r#"(
    name: "Trail",
    emitters: [(
        name: "T",
        max_particles: 1000,
        emitter_update: [(module: "Spawn Per Unit", inputs: { "Spawn Per Unit": Float(20.0) })],
        particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(5.0) })],
        particle_update: [(module: "Particle State")],
        renderers: [Sprite(())],
    )],
)"#,
    );
    // Rotated 90° about Y: world +X is effect-space +Z.
    let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let e = spawn(
        app,
        (ParticleEffect::shared(sys.clone(), 5), Transform::from_rotation(rot)),
    );
    frames(app, 1);
    let c = app.world().get::<EffectControl>(e).unwrap();
    assert_eq!(c.owner_velocity(), Vec3::ZERO, "first frame");
    // 0.05 m per 1/60 s step = 3 m/s; 20 per metre = 1 particle per step.
    let step = 0.05f32;
    for k in 1..=30 {
        app.world_mut().get_mut::<Transform>(e).unwrap().translation.x = step * k as f32;
        frames(app, 1);
    }
    let v = app.world().get::<EffectControl>(e).unwrap().owner_velocity();
    assert!((v - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-3, "world velocity {v}");
    let local = render_effect(app, e, |g, _| Vec3::from_array(g.owner_velocity)).unwrap();
    assert!((local - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-3, "effect-space velocity {local}");
    frames(app, 3); // read-back lag
    let (alive, next) = counts(app, e)[0];
    eprintln!("spawn per unit: {alive} alive, next id {next}");
    assert!((28..=34).contains(&next), "one particle per step of movement, got {next}");
    // A teleport (100 m in a frame) measures nothing and spawns nothing.
    app.world_mut().get_mut::<Transform>(e).unwrap().translation.x += 100.0;
    frames(app, 1);
    assert_eq!(app.world().get::<EffectControl>(e).unwrap().owner_velocity(), Vec3::ZERO, "teleport");
    frames(app, 3);
    assert_eq!(counts(app, e)[0].1, next, "teleport spawned particles");

    // Attached to a moving parent: the child's world motion counts.
    let parent = spawn(app, Transform::default());
    let child = {
        let mut commands = app.world_mut().commands();
        let id = commands
            .attach_effect(parent, ParticleEffect::shared(sys, 6), EffectMode::Looping, Transform::from_xyz(0.0, 1.0, 0.0))
            .id();
        app.world_mut().flush();
        id
    };
    frames(app, 1);
    for k in 1..=5 {
        app.world_mut().get_mut::<Transform>(parent).unwrap().translation.z = -0.1 * k as f32;
        frames(app, 1);
    }
    let v = app.world().get::<EffectControl>(child).unwrap().owner_velocity();
    assert!((v - Vec3::new(0.0, 0.0, -6.0)).length() < 1e-3, "attached velocity {v}");
}
