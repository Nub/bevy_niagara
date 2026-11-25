//! The Component renderer in a headless Bevy app on the GPU: never more
//! entities than `max_components`, the oldest particles drive them, each
//! particle keeps its entity while it lives and updates it every frame, the
//! game's registered component follows, and entities whose particle died are
//! hidden and kept for reuse.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::window::ExitCondition;
use fx_core::CookedSystem;
use fx_render::{
    FromParticle, FxClock, FxComponentAppExt, FxPlugin, ParticleAttributes, ParticleComponent,
    ParticleEffect,
};

#[derive(Component, Debug, Clone, Copy, PartialEq)]
struct Probe {
    id: u32,
    y: f32,
}

impl FromParticle for Probe {
    fn from_particle(p: &ParticleAttributes) -> Self {
        Probe {
            id: p.unique_id,
            y: p.position.y,
        }
    }
}

/// One app (one GPU device) for every scenario: several GPU apps in one
/// process crash the driver at exit.
#[test]
fn component_renderer() {
    let mut app = app();
    capped_stable_and_updated(&mut app);
    dead_particles_free_their_entities(&mut app);
}

fn cook(src: &str) -> Arc<CookedSystem> {
    let c = fx_core::check_source("test.fx.ron", src);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    Arc::new(c.cooked.unwrap_or_else(|| panic!("{errs:?}")))
}

fn system(spawn: &str, lifetime: f32, max_components: u32) -> Arc<CookedSystem> {
    cook(&format!(
        r#"(
    name: "Probes",
    emitters: [(
        name: "Probes",
        max_particles: 128,
        emitter_update: [{spawn}],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": Float({lifetime}) }}),
            (module: "Add Velocity", inputs: {{ "Velocity": Vec3((0.0, 1.0, 0.0)) }}),
        ],
        particle_update: [(module: "Particle State"), (module: "Solve Forces and Velocity")],
        renderers: [Component((component: "Probe", max_components: {max_components}))],
    )],
)"#
    ))
}

/// `(entity, component, attributes, transform, visibility)` of every driven entity.
fn driven(app: &mut App) -> Vec<(Entity, ParticleComponent, ParticleAttributes, Transform, Visibility, Option<Probe>)> {
    app.world_mut()
        .query::<(
            Entity,
            &ParticleComponent,
            &ParticleAttributes,
            &Transform,
            &Visibility,
            Option<&Probe>,
        )>()
        .iter(app.world())
        .map(|(e, c, a, t, v, p)| (e, c.clone(), *a, *t, *v, p.copied()))
        .collect()
}

fn capped_stable_and_updated(app: &mut App) {
    // One particle per step, living 0.5 s (30 steps): ~30 alive, 8 driven.
    let sys = system(
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Float(60.0) })"#,
        0.5,
        8,
    );
    let effect = app.world_mut().spawn((ParticleEffect::shared(sys, 1), Transform::default())).id();
    let mut max_seen = 0;
    for _ in 0..45 {
        frames(app, 1);
        let d = driven(app);
        max_seen = max_seen.max(d.len());
        assert!(d.len() <= 8, "{} entities for max_components 8", d.len());
    }
    assert_eq!(max_seen, 8, "the cap was never reached");

    let before = driven(app);
    assert_eq!(before.len(), 8);
    let ids: Vec<u32> = before.iter().map(|d| d.2.unique_id).collect();
    let unique: HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(unique.len(), 8, "one particle per entity: {ids:?}");
    // The oldest living particles: 8 consecutive UniqueIDs.
    let (lo, hi) = (ids.iter().min().unwrap(), ids.iter().max().unwrap());
    assert_eq!(hi - lo, 7, "not the 8 oldest: {ids:?}");
    for (e, c, a, t, v, probe) in &before {
        assert!(c.active, "{e} inactive");
        assert_eq!(&*c.name, "Probe");
        assert_eq!(*v, Visibility::Inherited);
        assert_eq!(t.translation, a.position, "Transform follows the particle");
        assert!(app.world().get::<ChildOf>(*e).is_some_and(|p| p.parent() == effect));
        let probe = probe.expect("registered component missing");
        assert_eq!(probe, Probe::from_particle(a));
    }

    frames(app, 3);
    let after = driven(app);
    assert_eq!(after.len(), 8, "the pool grew or shrank");
    let by_id: HashMap<u32, (Entity, f32)> =
        after.iter().map(|d| (d.2.unique_id, (d.0, d.2.position.y))).collect();
    let mut kept = 0;
    for (e, _, a, ..) in &before {
        if let Some((e2, y2)) = by_id.get(&a.unique_id) {
            kept += 1;
            assert_eq!(e, e2, "particle {} moved to another entity", a.unique_id);
            assert!(*y2 > a.position.y + 0.02, "particle {} not updated: {} -> {y2}", a.unique_id, a.position.y);
        }
    }
    // Three steps: the three oldest died, the rest kept their entities.
    assert!(kept >= 4, "only {kept} particles kept their entity");
    for d in &after {
        assert_eq!(d.5, Some(Probe::from_particle(&d.2)), "registered component stale");
    }

    app.world_mut().entity_mut(effect).despawn();
    frames(app, 2);
    assert!(driven(app).is_empty(), "entities outlived their effect");
}

fn dead_particles_free_their_entities(app: &mut App) {
    // A burst of 5 particles living 0.2 s, cap 16: 5 entities, then none active.
    let sys = system(
        r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(5), "Spawn Time": Float(0.0) })"#,
        0.2,
        16,
    );
    let effect = app.world_mut().spawn((ParticleEffect::shared(sys, 2), Transform::default())).id();
    frames(app, 4);
    let d = driven(app);
    assert_eq!(d.len(), 5, "one entity per particle under the cap");
    assert!(d.iter().all(|d| d.1.active));
    frames(app, 30);
    let d = driven(app);
    assert_eq!(d.len(), 5, "pooled entities are kept");
    assert!(d.iter().all(|d| !d.1.active && d.4 == Visibility::Hidden), "dead particles still drive entities");
    app.world_mut().entity_mut(effect).despawn();
    frames(app, 2);
}

/// Never dropped: tearing down a GPU app crashes the driver at exit.
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
        // Deterministic: each frame waits for its read-back (one frame of lag).
        FxPlugin { deterministic: true },
    ))
    .insert_resource(FxClock::manual(1.0 / 60.0))
    .add_particle_component::<Probe>("Probe");
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
