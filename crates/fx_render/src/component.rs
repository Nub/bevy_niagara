//! Component renderer (Niagara's Component Renderer): particles drive
//! pooled Bevy entities.
//!
//! Per effect instance and renderer, the GPU sorts the emitter's particles by
//! `Particles.UniqueID` and gathers the attributes of the first
//! `max_components` (the oldest); only those rows are read back
//! (asynchronously, a frame or two late, like particle lights). In the main
//! world each row drives a child entity of the effect carrying
//! [`ParticleComponent`] and [`ParticleAttributes`] (and, with
//! `transform: true`, its `Transform`). A particle keeps its entity while it
//! lives; entities without a particle are hidden and marked inactive, kept
//! for reuse. Never more than `max_components` entities exist per renderer.
//!
//! Games turn the attributes into their own component with
//! [`FxComponentAppExt::add_particle_component`]:
//!
//! ```ignore
//! #[derive(Component)]
//! struct Debris { heat: f32 }
//! impl FromParticle for Debris {
//!     fn from_particle(p: &ParticleAttributes) -> Self { Debris { heat: p.color.w } }
//! }
//! app.add_particle_component::<Debris>("Debris"); // renderer `component: "Debris"`
//! ```

use std::sync::{Arc, Mutex};

use bevy::math::{Quat, Vec2, Vec3, Vec4};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use fx_core::cook::CookedComponent;

/// Particle words gathered per row, in [`ParticleAttributes::from_row`] order.
pub(crate) const WORDS: [u32; 29] = [
    0, 1, 2, // position
    3, 4, 5, // velocity
    6, 7, 8, 9, // color
    10, 11, // sprite size
    12, 13, 14, 15, 16, // rotation, mass, lifetime, age, normalized age
    27, // unique id (bits)
    28, 29, 30, // scale
    31, 32, 33, 34, // mesh orientation
    40, 41, 42, 43, // dynamic material parameter
];

/// A particle's attributes, in the effect's local space (the entity is a
/// child of the effect).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct ParticleAttributes {
    pub position: Vec3,
    pub velocity: Vec3,
    pub color: Vec4,
    pub sprite_size: Vec2,
    /// Degrees.
    pub sprite_rotation: f32,
    pub mass: f32,
    pub lifetime: f32,
    pub age: f32,
    pub normalized_age: f32,
    pub unique_id: u32,
    pub scale: Vec3,
    pub mesh_orientation: Quat,
    pub dynamic_param: Vec4,
}

impl ParticleAttributes {
    fn from_row(r: &[f32]) -> Self {
        let v3 = |o: usize| Vec3::new(r[o], r[o + 1], r[o + 2]);
        let v4 = |o: usize| Vec4::new(r[o], r[o + 1], r[o + 2], r[o + 3]);
        let q = v4(22);
        ParticleAttributes {
            position: v3(0),
            velocity: v3(3),
            color: v4(6),
            sprite_size: Vec2::new(r[10], r[11]),
            sprite_rotation: r[12],
            mass: r[13],
            lifetime: r[14],
            age: r[15],
            normalized_age: r[16],
            unique_id: r[17].to_bits(),
            scale: v3(18),
            mesh_orientation: if q.length_squared() > 1e-12 {
                Quat::from_vec4(q).normalize()
            } else {
                Quat::IDENTITY
            },
            dynamic_param: v4(25),
        }
    }
}

/// An entity driven by a component renderer (a child of its effect).
#[derive(Component, Clone, Debug, PartialEq)]
pub struct ParticleComponent {
    /// Index of the renderer among the effect's renderers.
    pub renderer: usize,
    /// The renderer's `component` name.
    pub name: Arc<str>,
    /// Whether a particle drives it now (hidden when not).
    pub active: bool,
}

/// Builds a game component from a particle's attributes
/// ([`FxComponentAppExt::add_particle_component`]).
pub trait FromParticle: Component {
    fn from_particle(p: &ParticleAttributes) -> Self;
}

pub trait FxComponentAppExt {
    /// Entities of component renderers named `name` get a `C` built from
    /// their particle, updated whenever the particle's attributes change.
    fn add_particle_component<C: FromParticle>(&mut self, name: &str) -> &mut Self;
}

impl FxComponentAppExt for App {
    fn add_particle_component<C: FromParticle>(&mut self, name: &str) -> &mut Self {
        let name: Arc<str> = name.into();
        self.add_systems(
            PostUpdate,
            (move |mut commands: Commands,
                   q: Query<
                (Entity, &ParticleComponent, &ParticleAttributes),
                Changed<ParticleAttributes>,
            >| {
                for (e, pc, attrs) in &q {
                    if pc.active && *pc.name == *name {
                        commands.entity(e).try_insert(C::from_particle(attrs));
                    }
                }
            })
            .after(apply_particle_components),
        )
    }
}

/// Rows read back in the render world, waiting for the main world (internal).
#[derive(Resource, Clone, Default)]
pub struct ComponentInbox(pub(crate) Arc<Mutex<Vec<ComponentUpdate>>>);

pub(crate) struct ComponentUpdate {
    pub effect: Entity,
    pub renderer: usize,
    pub settings: CookedComponent,
    /// `WORDS.len()` words per particle.
    pub rows: Vec<f32>,
    /// The particles are in world space (a `local_space: false` emitter), not the effect's.
    pub world_space: bool,
}

/// One renderer's entities and which particle drives each.
#[derive(Default)]
struct Pool {
    entities: Vec<Entity>,
    /// UniqueID driving each entity (`None`: free).
    ids: Vec<Option<u32>>,
}

/// Main world: entity pools per (effect, component renderer).
#[derive(Resource, Default)]
pub struct ComponentPools(HashMap<(Entity, usize), Pool>);

/// Applies the newest rows: each particle keeps its entity while it lives;
/// new particles take free entities (spawned up to the cap).
pub fn apply_particle_components(
    mut commands: Commands,
    inbox: Res<ComponentInbox>,
    mut pools: ResMut<ComponentPools>,
    effects: Query<&GlobalTransform, With<crate::ParticleEffect>>,
    mut driven: Query<(
        &mut ParticleComponent,
        &mut ParticleAttributes,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    let updates: Vec<ComponentUpdate> = std::mem::take(&mut *inbox.0.lock().unwrap());
    for u in updates {
        let Ok(gt) = effects.get(u.effect) else {
            continue;
        };
        // World-space particles: into the effect's space (the entities are its children).
        let local_from_world = u.world_space.then(|| gt.affine().inverse());
        let pool = pools.0.entry((u.effect, u.renderer)).or_default();
        let cap = u.settings.max_components as usize;
        let attrs: Vec<ParticleAttributes> = u
            .rows
            .chunks_exact(WORDS.len())
            .take(cap)
            .map(ParticleAttributes::from_row)
            .collect();
        // Entity of each row: the one its particle had, else a free one.
        let mut slot_of: Vec<Option<usize>> = attrs
            .iter()
            .map(|a| pool.ids.iter().position(|id| *id == Some(a.unique_id)))
            .collect();
        let mut taken = vec![false; pool.entities.len()];
        for s in slot_of.iter().flatten() {
            taken[*s] = true;
        }
        for s in slot_of.iter_mut() {
            if s.is_none() {
                *s = match taken.iter().position(|t| !t) {
                    Some(free) => Some(free),
                    None if pool.entities.len() < cap => {
                        pool.entities.push(Entity::PLACEHOLDER);
                        pool.ids.push(None);
                        taken.push(false);
                        Some(pool.entities.len() - 1)
                    }
                    None => None,
                };
                if let Some(f) = *s {
                    taken[f] = true;
                }
            }
        }
        let name: Arc<str> = u.settings.component.as_str().into();
        for (a, slot) in attrs.iter().zip(&slot_of) {
            let Some(slot) = *slot else { continue };
            pool.ids[slot] = Some(a.unique_id);
            let pc = ParticleComponent {
                renderer: u.renderer,
                name: name.clone(),
                active: true,
            };
            let transform = |old: Transform| {
                if u.settings.transform {
                    let t = Transform {
                        translation: a.position,
                        rotation: a.mesh_orientation,
                        scale: a.scale,
                    };
                    match local_from_world {
                        Some(m) => Transform::from_matrix(Mat4::from(m * t.compute_affine())),
                        None => t,
                    }
                } else {
                    old
                }
            };
            let e = pool.entities[slot];
            match driven.get_mut(e) {
                Ok((mut c, mut at, mut t, mut v)) => {
                    if *c != pc {
                        *c = pc;
                    }
                    if *at != *a {
                        *at = *a;
                    }
                    let nt = transform(*t);
                    if *t != nt {
                        *t = nt;
                    }
                    if *v != Visibility::Inherited {
                        *v = Visibility::Inherited;
                    }
                }
                Err(_) if e != Entity::PLACEHOLDER => {
                    // Spawned by last frame's commands, not applied yet (or despawned).
                    commands
                        .entity(e)
                        .try_insert((pc, *a, transform(Transform::default()), Visibility::Inherited));
                }
                Err(_) => {
                    pool.entities[slot] = commands
                        .spawn((
                            pc,
                            *a,
                            transform(Transform::default()),
                            Visibility::Inherited,
                            ChildOf(u.effect),
                        ))
                        .id();
                }
            }
        }
        // Entities without a particle this time.
        for (slot, t) in taken.iter().enumerate() {
            if *t {
                continue;
            }
            pool.ids[slot] = None;
            if let Ok((mut c, _, _, mut v)) = driven.get_mut(pool.entities[slot]) {
                if c.active {
                    c.active = false;
                }
                if *v != Visibility::Hidden {
                    *v = Visibility::Hidden;
                }
            }
        }
    }
    pools.0.retain(|(effect, _), _| effects.get(*effect).is_ok());
}
