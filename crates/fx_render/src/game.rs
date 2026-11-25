//! The game-facing API: spawning, attaching, controlling and pooling effects,
//! and Niagara-style scalability.
//!
//! - **Spawn** a [`ParticleEffect`] with a `Transform` (at a point) or as a
//!   child of any entity (`ChildOf`, attached: it follows the entity, a
//!   skeleton joint included; see [`find_joint`]). [`FxCommandsExt`] has
//!   one-liners for both.
//! - **Lifecycle** ([`EffectControl`]): `Looping` effects run until despawned;
//!   `OneShot` effects spawn for the system's `duration` (source file, or the
//!   last burst), then despawn themselves once their particles are gone
//!   (counts read back asynchronously, a few frames late). `activate`,
//!   `deactivate` (stop spawning, let particles die), `reset`/`fire`
//!   (restart from time zero).
//! - **Parameters**: [`ParticleEffect::set_param`] sets a `User.` value by
//!   name, typed, with a [`ParamError`] for an unknown name or wrong type.
//! - **Pooling**: each cooked system is compiled once ([`fx_gpu::GpuProgram`]);
//!   an effect instance whose entity is gone goes back to a pool with its GPU
//!   buffers, and the next spawn of the same system (same
//!   `Arc<CookedSystem>`) reuses it after a reset: no shader compilation, no
//!   GPU allocation, no per-spawn heap allocation in the render world.
//!   [`FxPoolStats`] counts it.
//! - **Scalability**, from the effect file's `scalability` block and the
//!   global [`FxQuality`]: distance culling, spawn-count scaling by quality
//!   and distance, visibility culling (bounds outside every camera frustum
//!   for `visibility_grace` seconds) and per-system instance budgets (the
//!   nearest kept). A culled looping effect pauses (no simulation, no draw);
//!   a culled one-shot ends. [`FxScalability::enabled`] turns it all off.

use std::sync::{Arc, Mutex};

use bevy::camera::primitives::{Frustum, Sphere};
use bevy::ecs::system::EntityCommands;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use fx_core::cook::CookedSystem;
use fx_core::scalability::QualityLevel;
use fx_core::types::{Value, ValueType};

use crate::{FxClock, ParticleEffect};

/// Seconds a one-shot spawns when its system has no `duration` and an
/// emitter spawns forever (Spawn Rate).
pub const DEFAULT_ONE_SHOT_SECONDS: f32 = 1.0;

/// How an effect instance lives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffectMode {
    /// Runs until despawned (or deactivated).
    #[default]
    Looping,
    /// Spawns for the system's duration, then ends once its particles are gone.
    OneShot,
}

/// Why scalability culled an instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullReason {
    /// Farther than `max_distance` from every camera.
    Distance,
    /// Over the system's `max_instances`; the nearest are kept.
    Budget,
    /// Out of every camera's view for longer than `visibility_grace`.
    NotVisible,
}

/// Lifecycle and control of one effect instance (required by
/// [`ParticleEffect`]; Looping by default).
#[derive(Component, Clone, Debug)]
pub struct EffectControl {
    pub mode: EffectMode,
    /// Despawn the entity (and its children) when a one-shot ends. Default true.
    pub despawn_when_finished: bool,
    active: bool,
    generation: u32,
    age: f32,
    finished: bool,
    /// Spawning this frame (sent to the render world).
    pub(crate) spawning: bool,
    distance: f32,
    spawn_scale: f32,
    culled: Option<CullReason>,
    offscreen: f32,
}

impl Default for EffectControl {
    fn default() -> Self {
        EffectControl {
            mode: EffectMode::Looping,
            despawn_when_finished: true,
            active: true,
            generation: 0,
            age: 0.0,
            finished: false,
            spawning: true,
            distance: 0.0,
            spawn_scale: 1.0,
            culled: None,
            offscreen: 0.0,
        }
    }
}

impl EffectControl {
    pub fn looping() -> Self {
        Self::default()
    }
    pub fn one_shot() -> Self {
        EffectControl {
            mode: EffectMode::OneShot,
            ..default()
        }
    }
    /// Starts spawning again. A finished one-shot restarts (as Niagara's
    /// Activate on a completed system).
    pub fn activate(&mut self) {
        if self.finished || (self.mode == EffectMode::OneShot && !self.spawning) {
            self.reset();
        }
        self.active = true;
    }
    /// Stops spawning; living particles finish their lives.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
    /// Restarts from time zero: particles cleared, emitters restarted, active.
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.age = 0.0;
        self.finished = false;
        self.active = true;
        self.spawning = true;
    }
    /// Plays the effect again from the start (a one-shot's replay): [`Self::reset`].
    pub fn fire(&mut self) {
        self.reset();
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    /// A one-shot that stopped spawning and has no particles left (or was culled).
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    /// Simulated seconds since the (re)start.
    pub fn age(&self) -> f32 {
        self.age
    }
    /// Distance to the nearest camera, as scalability last measured it.
    pub fn distance(&self) -> f32 {
        self.distance
    }
    /// The spawn-count scale scalability applies (quality × distance).
    pub fn spawn_scale(&self) -> f32 {
        self.spawn_scale
    }
    pub fn culled(&self) -> Option<CullReason> {
        self.culled
    }
    /// Not simulated or drawn this frame (a culled looping effect).
    pub fn paused(&self) -> bool {
        self.culled.is_some()
    }
    pub(crate) fn generation(&self) -> u32 {
        self.generation
    }
    /// A one-shot waiting for its particles to die.
    pub(crate) fn draining(&self) -> bool {
        self.mode == EffectMode::OneShot && !self.spawning && !self.finished && self.culled.is_none()
    }
}

/// A `User.` parameter could not be set.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamError {
    /// The system declares no such `User.` parameter.
    Unknown { name: String, system: String },
    /// The parameter has another type.
    WrongType {
        name: String,
        expected: ValueType,
        found: ValueType,
    },
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::Unknown { name, system } => {
                write!(f, "effect \"{system}\" has no user parameter \"{name}\"")
            }
            ParamError::WrongType {
                name,
                expected,
                found,
            } => write!(f, "user parameter \"{name}\" is {expected:?}, not {found:?}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// A value [`ParticleEffect::set_param`] accepts.
pub trait IntoParam {
    fn into_param(self) -> Value;
}

impl IntoParam for Value {
    fn into_param(self) -> Value {
        self
    }
}
impl IntoParam for f32 {
    fn into_param(self) -> Value {
        Value::Float(self)
    }
}
impl IntoParam for i32 {
    fn into_param(self) -> Value {
        Value::Int(self)
    }
}
impl IntoParam for bool {
    fn into_param(self) -> Value {
        Value::Bool(self)
    }
}
impl IntoParam for Vec2 {
    fn into_param(self) -> Value {
        Value::Vec2(self.to_array())
    }
}
impl IntoParam for Vec3 {
    fn into_param(self) -> Value {
        Value::Vec3(self.to_array())
    }
}
impl IntoParam for Vec4 {
    fn into_param(self) -> Value {
        Value::Vec4(self.to_array())
    }
}
impl IntoParam for Quat {
    fn into_param(self) -> Value {
        Value::Quat(self.to_array())
    }
}
impl IntoParam for Color {
    fn into_param(self) -> Value {
        Value::Color(self.to_linear().to_f32_array())
    }
}
impl IntoParam for LinearRgba {
    fn into_param(self) -> Value {
        Value::Color(self.to_f32_array())
    }
}

/// Index of user parameter `name` (`"User.Rate"` or `"Rate"`).
pub(crate) fn user_index(system: &CookedSystem, name: &str) -> Option<usize> {
    let short = name.strip_prefix("User.").unwrap_or(name);
    system
        .user_params
        .iter()
        .position(|p| p.name.strip_prefix("User.").unwrap_or(&p.name) == short)
}

/// The global quality level (Niagara's effects quality), High by default. Changing it applies to every effect from the next frame.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FxQuality(pub QualityLevel);

/// Global scalability switch, on by default (turn off to compare costs).
#[derive(Resource, Clone, Copy, Debug)]
pub struct FxScalability {
    pub enabled: bool,
}

impl Default for FxScalability {
    fn default() -> Self {
        FxScalability { enabled: true }
    }
}

/// Pool counters, shared by the main and render worlds; read with
/// [`FxPoolStats::get`].
#[derive(Resource, Clone, Default)]
pub struct FxPoolStats(pub(crate) Arc<Mutex<PoolStats>>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Systems compiled for the GPU (once each).
    pub programs: u64,
    /// Instances built (GPU buffers allocated).
    pub instances_created: u64,
    /// Spawns served from the pool.
    pub instances_reused: u64,
    /// Instances simulating now.
    pub live: u64,
    /// Instances waiting in the pool.
    pub idle: u64,
    /// GPU bytes held by all instances, live and idle.
    pub gpu_bytes: u64,
}

impl FxPoolStats {
    pub fn get(&self) -> PoolStats {
        *self.0.lock().unwrap()
    }
}

/// One-shots found empty by the render world's count readback:
/// (entity, generation).
#[derive(Resource, Clone, Default)]
pub(crate) struct DoneInbox(pub Arc<Mutex<Vec<(Entity, u32)>>>);

/// Spawning helpers on `Commands`.
pub trait FxCommandsExt {
    /// A one-shot at `at` (world space): despawns itself when it ends.
    fn spawn_one_shot(&mut self, effect: ParticleEffect, at: Transform) -> EntityCommands<'_>;
    /// A looping effect at `at` (world space).
    fn spawn_looping(&mut self, effect: ParticleEffect, at: Transform) -> EntityCommands<'_>;
    /// An effect attached to `parent` (any entity, a skeleton joint
    /// included) at `offset` in its space: it follows the parent and is
    /// despawned with it. Particles simulate in the effect's local space.
    fn attach_effect(
        &mut self,
        parent: Entity,
        effect: ParticleEffect,
        mode: EffectMode,
        offset: Transform,
    ) -> EntityCommands<'_>;
}

impl FxCommandsExt for Commands<'_, '_> {
    fn spawn_one_shot(&mut self, effect: ParticleEffect, at: Transform) -> EntityCommands<'_> {
        self.spawn((effect, EffectControl::one_shot(), at))
    }
    fn spawn_looping(&mut self, effect: ParticleEffect, at: Transform) -> EntityCommands<'_> {
        self.spawn((effect, EffectControl::looping(), at))
    }
    fn attach_effect(
        &mut self,
        parent: Entity,
        effect: ParticleEffect,
        mode: EffectMode,
        offset: Transform,
    ) -> EntityCommands<'_> {
        let control = EffectControl {
            mode,
            ..default()
        };
        self.spawn((effect, control, offset, ChildOf(parent)))
    }
}

/// The descendant of `root` named `name` (a glTF skeleton joint is an entity
/// with the node's `Name`), to attach an effect to a bone.
pub fn find_joint(
    root: Entity,
    name: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    if names.get(root).is_ok_and(|n| n.as_str() == name) {
        return Some(root);
    }
    children
        .iter_descendants(root)
        .find(|&e| names.get(e).is_ok_and(|n| n.as_str() == name))
}

/// Main world, after transform propagation: scalability, then the lifecycle
/// (one-shot duration, age).
#[allow(clippy::type_complexity)]
pub(crate) fn update_effects(
    mut effects: Query<(Entity, &ParticleEffect, &mut EffectControl, &GlobalTransform)>,
    cameras: Query<(&Camera, &GlobalTransform, &Frustum)>,
    clock: Res<FxClock>,
    time: Res<Time>,
    quality: Res<FxQuality>,
    settings: Res<FxScalability>,
    mut budgets: Local<HashMap<usize, Vec<(f32, Entity)>>>,
) {
    let dt_sim = clock.steps_this_frame() as f32 * clock.step;
    let dt_real = time.delta_secs();
    for list in budgets.values_mut() {
        list.clear();
    }
    for (entity, effect, mut c, gt) in &mut effects {
        let s = &effect.system.scalability;
        let r = s.resolve(quality.0);
        let pos = gt.translation();
        let radius = s.bounds_radius * gt.scale().abs().max_element();
        let mut distance = f32::INFINITY;
        let mut visible = false;
        let mut any_camera = false;
        for (cam, cgt, frustum) in &cameras {
            if !cam.is_active {
                continue;
            }
            any_camera = true;
            distance = distance.min(cgt.translation().distance(pos));
            visible |= frustum.intersects_sphere(
                &Sphere {
                    center: pos.into(),
                    radius,
                },
                true,
            );
        }
        if !any_camera {
            distance = 0.0;
            visible = true;
        }
        c.distance = distance;
        if settings.enabled {
            c.spawn_scale = r.spawn_count_scale * s.spawn_scale_at(distance);
            c.offscreen = if visible { 0.0 } else { c.offscreen + dt_real };
            c.culled = if distance - radius > r.max_distance {
                Some(CullReason::Distance)
            } else if s.cull_when_not_visible && c.offscreen > s.visibility_grace {
                Some(CullReason::NotVisible)
            } else {
                None
            };
            if c.culled.is_none() && r.max_instances > 0 && !c.finished {
                let key = Arc::as_ptr(&effect.system) as usize;
                budgets.entry(key).or_default().push((distance, entity));
            }
        } else {
            c.spawn_scale = 1.0;
            c.offscreen = 0.0;
            c.culled = None;
        }
        // A culled one-shot ends.
        if c.culled.is_some() && c.mode == EffectMode::OneShot {
            c.finished = true;
        }
    }
    // Budgets: the nearest `max_instances` of each system stay.
    if settings.enabled {
        for list in budgets.values_mut() {
            let Some(&(_, first)) = list.first() else {
                continue;
            };
            let Ok((_, effect, _, _)) = effects.get(first) else {
                continue;
            };
            let max = effect.system.scalability.resolve(quality.0).max_instances as usize;
            if list.len() <= max {
                continue;
            }
            list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, e) in &list[max..] {
                if let Ok((_, _, mut c, _)) = effects.get_mut(e) {
                    c.culled = Some(CullReason::Budget);
                    if c.mode == EffectMode::OneShot {
                        c.finished = true;
                    }
                }
            }
        }
    }
    // Lifecycle: whether this frame's steps spawn, then the age.
    for (_, effect, mut c, _) in &mut effects {
        let duration = effect
            .system
            .duration
            .unwrap_or(DEFAULT_ONE_SHOT_SECONDS);
        c.spawning = c.active
            && !c.finished
            && (c.mode == EffectMode::Looping || c.age <= duration);
        if c.culled.is_none() && !c.finished {
            c.age += dt_sim;
        }
    }
}

/// Main world: one-shots the render world found empty end; finished
/// one-shots are despawned.
pub(crate) fn finish_effects(
    mut commands: Commands,
    inbox: Res<DoneInbox>,
    mut effects: Query<(Entity, &mut EffectControl)>,
) {
    {
        let mut done = inbox.0.lock().unwrap();
        for (e, generation) in done.drain(..) {
            if let Ok((_, mut c)) = effects.get_mut(e)
                && c.generation == generation
                && c.draining()
            {
                c.finished = true;
            }
        }
    }
    for (e, c) in &effects {
        if c.finished && c.mode == EffectMode::OneShot && c.despawn_when_finished {
            commands.entity(e).try_despawn();
        }
    }
}
