//! CPU reference simulation of the module set.
//!
//! This is the ground truth the GPU simulation must match: deterministic for
//! a given seed, with every random draw a hash of where it happens
//! ([`rng::RandKey`]), never a stateful generator.
//!
//! Each [`Simulation::step`], per emitter, in order:
//! 1. on the first step, the Emitter Spawn stack;
//! 2. the Emitter Update stack, which decides how many particles to spawn;
//! 3. the emitter's events of the previous step are cleared; the Particle
//!    Update stack runs over the living particles, each followed by the
//!    EveryParticle event handlers (once per event); then the events the
//!    update wrote are sorted by source UniqueID and dead particles are
//!    removed (order kept);
//! 4. new particles are created (capped at `max_particles`): first the
//!    emitter stack's request, then `spawn_number` per event of each
//!    SpawnedParticles handler, in handler and event order. They run the
//!    Particle Spawn stack (and the handler's modules) and first update on
//!    the next step.
//!
//! `Engine.Time` and the ages read during a step are their values at the
//! start of the step.

pub mod noise;
pub mod rng;

use bevy_math::{Quat, Vec2, Vec3, Vec4};

use crate::cook::{CookedInput, CookedModule, CookedSystem};
use crate::data::{self, CameraData, DiBinding, MeshData, MeshPose, NO_SCENE_DEPTH, TextureData};
use crate::events::{self as fx_events, Event, EventKind};
use crate::modules::{self, ModuleKind};
use crate::params::{Namespace, ParamRef, attr};
use crate::types::{Value, ValueType};
use rng::RandKey;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub color: Vec4,
    pub sprite_size: Vec2,
    pub sprite_rotation: f32,
    pub mass: f32,
    pub lifetime: f32,
    pub age: f32,
    pub normalized_age: f32,
    pub unique_id: u32,
    pub initial_color: Vec4,
    pub initial_sprite_size: Vec2,
    pub force: Vec3,
    pub drag: f32,
    /// `Particles.Scale`: mesh scale per axis.
    pub scale: Vec3,
    /// `Particles.MeshOrientation`.
    pub mesh_orientation: Quat,
    /// `Particles.RibbonID`.
    pub ribbon_id: i32,
    /// `Particles.RibbonWidth`, metres.
    pub ribbon_width: f32,
    /// `Particles.RibbonTwist`, degrees.
    pub ribbon_twist: f32,
    /// `Particles.RibbonLinkOrder`.
    pub ribbon_link_order: f32,
    /// `Particles.LightRadius`, metres.
    pub light_radius: f32,
    /// `Particles.DynamicMaterialParameter`.
    pub dynamic_material_parameter: Vec4,
    /// `Particles.SubImageIndex`: flipbook frame (SubUV Animation).
    pub sub_image_index: f32,
    /// `Particles.Initial.Scale`.
    pub initial_scale: Vec3,
    /// `Particles.Initial.RibbonWidth`.
    pub initial_ribbon_width: f32,
    /// `Particles.CameraOffset`, metres towards the camera (Camera Offset).
    pub camera_offset: f32,
    pub alive: bool,
}

impl Particle {
    /// A fresh particle; its defaults match Initialize Particle's.
    pub fn new(unique_id: u32) -> Self {
        Particle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            color: Vec4::ONE,
            sprite_size: Vec2::splat(0.1),
            sprite_rotation: 0.0,
            mass: 1.0,
            lifetime: 2.0,
            age: 0.0,
            normalized_age: 0.0,
            unique_id,
            initial_color: Vec4::ONE,
            initial_sprite_size: Vec2::splat(0.1),
            force: Vec3::ZERO,
            drag: 0.0,
            scale: Vec3::ONE,
            mesh_orientation: Quat::IDENTITY,
            ribbon_id: 0,
            ribbon_width: 0.1,
            ribbon_twist: 0.0,
            ribbon_link_order: 0.0,
            light_radius: 1.0,
            dynamic_material_parameter: Vec4::ZERO,
            sub_image_index: 0.0,
            initial_scale: Vec3::ONE,
            initial_ribbon_width: 0.1,
            camera_offset: 0.0,
            alive: true,
        }
    }

    /// Reads attribute `index` of [`crate::params::PARTICLES`].
    pub fn get(&self, index: u16) -> Value {
        match index {
            attr::POSITION => Value::Vec3(self.position.to_array()),
            attr::VELOCITY => Value::Vec3(self.velocity.to_array()),
            attr::COLOR => Value::Color(self.color.to_array()),
            attr::SPRITE_SIZE => Value::Vec2(self.sprite_size.to_array()),
            attr::SPRITE_ROTATION => Value::Float(self.sprite_rotation),
            attr::MASS => Value::Float(self.mass),
            attr::LIFETIME => Value::Float(self.lifetime),
            attr::AGE => Value::Float(self.age),
            attr::NORMALIZED_AGE => Value::Float(self.normalized_age),
            attr::UNIQUE_ID => Value::Int(self.unique_id as i32),
            attr::INITIAL_COLOR => Value::Color(self.initial_color.to_array()),
            attr::INITIAL_SPRITE_SIZE => Value::Vec2(self.initial_sprite_size.to_array()),
            attr::PHYSICS_FORCE => Value::Vec3(self.force.to_array()),
            attr::PHYSICS_DRAG => Value::Float(self.drag),
            attr::SCALE => Value::Vec3(self.scale.to_array()),
            attr::MESH_ORIENTATION => Value::Quat(self.mesh_orientation.to_array()),
            attr::RIBBON_ID => Value::Int(self.ribbon_id),
            attr::RIBBON_WIDTH => Value::Float(self.ribbon_width),
            attr::RIBBON_TWIST => Value::Float(self.ribbon_twist),
            attr::RIBBON_LINK_ORDER => Value::Float(self.ribbon_link_order),
            attr::LIGHT_RADIUS => Value::Float(self.light_radius),
            attr::DYNAMIC_MATERIAL_PARAMETER => {
                Value::Vec4(self.dynamic_material_parameter.to_array())
            }
            attr::SUB_IMAGE_INDEX => Value::Float(self.sub_image_index),
            attr::INITIAL_SCALE => Value::Vec3(self.initial_scale.to_array()),
            attr::INITIAL_RIBBON_WIDTH => Value::Float(self.initial_ribbon_width),
            attr::CAMERA_OFFSET => Value::Float(self.camera_offset),
            _ => Value::Float(0.0),
        }
    }

    /// Bit pattern of every field, for exact comparisons and hashing.
    pub fn bits(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(51);
        let mut f = |xs: &[f32]| v.extend(xs.iter().map(|x| x.to_bits()));
        f(&self.position.to_array());
        f(&self.velocity.to_array());
        f(&self.color.to_array());
        f(&self.sprite_size.to_array());
        f(&[
            self.sprite_rotation,
            self.mass,
            self.lifetime,
            self.age,
            self.normalized_age,
        ]);
        f(&self.initial_color.to_array());
        f(&self.initial_sprite_size.to_array());
        f(&self.force.to_array());
        f(&[self.drag]);
        f(&self.scale.to_array());
        f(&self.mesh_orientation.to_array());
        f(&[
            f32::from_bits(self.ribbon_id as u32),
            self.ribbon_width,
            self.ribbon_twist,
            self.ribbon_link_order,
            self.light_radius,
        ]);
        f(&self.dynamic_material_parameter.to_array());
        f(&[self.sub_image_index]);
        f(&self.initial_scale.to_array());
        f(&[self.initial_ribbon_width, self.camera_offset]);
        v.push(self.unique_id);
        v.push(self.alive as u32);
        v
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmitterState {
    pub age: f32,
    pub spawn_remainder: f32,
    pub next_id: u32,
    pub spawned_last_step: u32,
    pub particles: Vec<Particle>,
    /// Events written by the last particle update, by [`EventKind::index`],
    /// sorted by source UniqueID.
    pub events: [Vec<Event>; 3],
}

/// A host-provided collision shape for the CPU path of the Collision
/// module's SceneDepth and RayTraced types (the GPU uses the depth buffer or
/// the ray-tracing scene instead). In the effect's simulation space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// Half-space below the plane through `origin` facing `normal`.
    Plane { normal: Vec3, origin: Vec3 },
    Sphere { center: Vec3, radius: f32 },
    /// Axis-aligned box.
    Box { center: Vec3, half_extents: Vec3 },
}

impl Collider {
    /// Distance along the unit ray from `origin` in `dir` to the shape's
    /// surface, if the ray meets it in front of the origin.
    pub fn ray_distance(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        match *self {
            Collider::Plane { normal, origin: o } => {
                let n = unit_or_y(normal);
                let denom = dir.dot(n);
                if denom == 0.0 {
                    return None;
                }
                let t = (o - origin).dot(n) / denom;
                (t >= 0.0).then_some(t)
            }
            Collider::Sphere { center, radius } => {
                let oc = origin - center;
                let b = oc.dot(dir);
                let c = oc.length_squared() - radius * radius;
                let disc = b * b - c;
                if disc < 0.0 {
                    return None;
                }
                let s = disc.sqrt();
                [-b - s, -b + s].into_iter().find(|t| *t >= 0.0)
            }
            Collider::Box {
                center,
                half_extents: h,
            } => {
                let inv = dir.recip();
                let t0 = (center - h - origin) * inv;
                let t1 = (center + h - origin) * inv;
                let near = t0.min(t1).max_element();
                let far = t0.max(t1).min_element();
                if near > far || far < 0.0 {
                    return None;
                }
                Some(if near >= 0.0 { near } else { far })
            }
        }
    }

    /// The plane a particle at `pos` (radius `r`) is pushed out through, if
    /// it is inside the shape: `(normal, point on the surface)`.
    pub fn contact(&self, pos: Vec3, r: f32) -> Option<(Vec3, Vec3)> {
        match *self {
            Collider::Plane { normal, origin } => Some((unit_or_y(normal), origin)),
            Collider::Sphere { center, radius } => {
                let v = pos - center;
                let dist = v.length();
                (dist < radius + r).then(|| {
                    let n = if dist > 0.0 { v / dist } else { Vec3::Y };
                    (n, center + n * radius)
                })
            }
            Collider::Box {
                center,
                half_extents: h,
            } => {
                let q = pos - center;
                let depth = h + Vec3::splat(r) - q.abs();
                if depth.min_element() <= 0.0 {
                    return None;
                }
                // Out through the nearest face.
                let axis = if depth.x <= depth.y && depth.x <= depth.z {
                    0
                } else if depth.y <= depth.z {
                    1
                } else {
                    2
                };
                let mut n = Vec3::ZERO;
                n[axis] = if q[axis] < 0.0 { -1.0 } else { 1.0 };
                Some((n, center + n * h[axis]))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Simulation {
    pub system: CookedSystem,
    pub seed: u32,
    /// Steps taken so far.
    pub frame: u32,
    /// Simulated seconds so far (`Engine.Time`, `System.Age`).
    pub time: f32,
    /// Current `User.` parameter values, in declaration order.
    pub user: Vec<Value>,
    /// `Engine.Owner.Velocity`: the owning entity's velocity (m/s), set by
    /// the host; read by Inherit Velocity.
    pub owner_velocity: Vec3,
    /// What SceneDepth and RayTraced collisions hit on the CPU (set by the host).
    pub colliders: Vec<Collider>,
    /// The camera the effect is seen from (Camera Query, Scene Depth), set by the host.
    pub camera: Option<CameraData>,
    /// What each data interface of `system.data_interfaces` is bound to.
    pub data: Vec<DiBinding>,
    pub emitters: Vec<EmitterState>,
}

const STAGE_EMITTER_SPAWN: u32 = 2;
const STAGE_EMITTER_UPDATE: u32 = 3;
const STAGE_PARTICLE_SPAWN: u32 = 4;
const STAGE_PARTICLE_UPDATE: u32 = 5;

/// Read-only state inputs are evaluated against.
struct Env<'a> {
    user: &'a [Value],
    dt: f32,
    time: f32,
    emitter_age: f32,
    owner_velocity: Vec3,
    /// Flipbook frames of the emitter's sprite renderer (SubUV Animation).
    sub_image_frames: u32,
    colliders: &'a [Collider],
    camera: Option<CameraData>,
    data: &'a [DiBinding],
}

/// Per-particle state of one pass over a module stack that is not a
/// particle attribute: the event being handled, the collision found and the
/// events to write.
#[derive(Default)]
struct ParticleCx<'a> {
    /// The event a handler runs for (`Event.` parameters).
    event: Option<&'a Event>,
    /// A Collision module hit this step: the surface normal.
    collision: Option<Vec3>,
    /// Events to write at the end of the update, by kind.
    pending: [Option<Event>; 3],
    /// Generate Death Event ran: write a death event if the particle dies.
    death: bool,
}

fn event_of(p: &Particle, normal: Vec3) -> Event {
    Event {
        position: p.position.to_array(),
        velocity: p.velocity.to_array(),
        normal: normal.to_array(),
        age: p.age,
        particle_id: p.unique_id,
    }
}

/// The events handler `h` reads: `(events in order, how many it takes)`.
fn handled(events: &[Event], max: u32) -> usize {
    if max == 0 {
        events.len()
    } else {
        events.len().min(max as usize)
    }
}

impl Simulation {
    pub fn new(system: CookedSystem, seed: u32) -> Self {
        let user = system.user_params.iter().map(|p| p.value).collect();
        let emitters = vec![EmitterState::default(); system.emitters.len()];
        let data = vec![DiBinding::None; system.data_interfaces.len()];
        Simulation {
            system,
            seed,
            frame: 0,
            time: 0.0,
            user,
            owner_velocity: Vec3::ZERO,
            colliders: Vec::new(),
            camera: None,
            data,
            emitters,
        }
    }

    /// Sets a `User.` parameter. Returns false if it is not declared or the type differs.
    pub fn set_user(&mut self, name: &str, value: Value) -> bool {
        match self.system.user_params.iter().position(|p| p.name == name) {
            Some(i) if self.user[i].ty() == value.ty() => {
                self.user[i] = value;
                true
            }
            _ => false,
        }
    }

    fn data_index(&self, name: &str, ok: impl Fn(data::DiType) -> bool) -> Option<usize> {
        self.system
            .data_interfaces
            .iter()
            .position(|d| d.name == name && ok(d.ty))
    }

    /// Binds mesh data interface `name` (SkeletalMesh or StaticMesh) to
    /// `mesh`, posed by `pose`. Returns false if no such data interface is declared.
    pub fn set_mesh(&mut self, name: &str, mesh: std::sync::Arc<MeshData>, pose: MeshPose) -> bool {
        match self.data_index(name, data::DiType::is_mesh) {
            Some(i) => {
                self.data[i] = DiBinding::Mesh(mesh, pose);
                true
            }
            None => false,
        }
    }

    /// Moves a bound mesh to a new pose, `interval` seconds after its last one
    /// (see [`MeshPose`]). Returns false if `name` is not a bound mesh.
    pub fn set_mesh_pose(&mut self, name: &str, matrices: &[bevy_math::Mat4], interval: f32) -> bool {
        match self.data_index(name, data::DiType::is_mesh).map(|i| &mut self.data[i]) {
            Some(DiBinding::Mesh(_, pose)) => {
                pose.advance(matrices, interval);
                true
            }
            _ => false,
        }
    }

    /// Binds texture data interface `name`. Returns false if it is not declared.
    pub fn set_texture(&mut self, name: &str, texture: std::sync::Arc<TextureData>) -> bool {
        match self.data_index(name, |t| t == data::DiType::Texture) {
            Some(i) => {
                self.data[i] = DiBinding::Texture(texture);
                true
            }
            None => false,
        }
    }

    pub fn particles(&self, emitter: usize) -> &[Particle] {
        &self.emitters[emitter].particles
    }

    /// Events of `kind` emitter `emitter` wrote in its last update, sorted by source UniqueID.
    pub fn events(&self, emitter: usize, kind: EventKind) -> &[Event] {
        &self.emitters[emitter].events[kind.index()]
    }

    pub fn particle_count(&self) -> usize {
        self.emitters.iter().map(|e| e.particles.len()).sum()
    }

    /// FNV-1a over every particle's bits and the emitter state: equal for equal states.
    pub fn state_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |w: u32| {
            for b in w.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat(self.frame);
        eat(self.time.to_bits());
        for e in &self.emitters {
            eat(e.age.to_bits());
            eat(e.spawn_remainder.to_bits());
            eat(e.next_id);
            eat(e.particles.len() as u32);
            for p in &e.particles {
                p.bits().into_iter().for_each(&mut eat);
            }
        }
        h
    }

    pub fn step(&mut self, dt: f32) {
        for ei in 0..self.emitters.len() {
            self.step_emitter(ei, dt);
        }
        self.time += dt;
        self.frame += 1;
    }

    fn step_emitter(&mut self, ei: usize, dt: f32) {
        let emitter = &self.system.emitters[ei];
        // Other emitters' events the EveryParticle handlers read (a handler
        // never reads its own emitter's, which this step clears).
        let every: Vec<Vec<Event>> = emitter
            .event_handlers
            .iter()
            .map(|h| {
                if h.every_particle {
                    let ev = &self.emitters[h.source as usize].events[h.event as usize];
                    ev[..handled(ev, h.max_events)].to_vec()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let state = &mut self.emitters[ei];
        let env = Env {
            user: &self.user,
            dt,
            time: self.time,
            emitter_age: state.age,
            owner_velocity: self.owner_velocity,
            sub_image_frames: emitter.sub_image_frames(),
            colliders: &self.colliders,
            camera: self.camera,
            data: &self.data,
        };
        let (seed, frame) = (self.seed, self.frame);
        let key = |stage: u32, module: usize, particle: u32, frame: u32| RandKey {
            seed,
            emitter: ei as u32,
            particle,
            frame,
            module: stage * 256 + module as u32,
        };

        // 1-2. Emitter stages.
        let mut spawn_count: u32 = 0;
        if frame == 0 {
            run_emitter_stack(
                &emitter.emitter_spawn,
                STAGE_EMITTER_SPAWN,
                &env,
                state,
                &mut spawn_count,
                &|m| key(STAGE_EMITTER_SPAWN, m, u32::MAX, frame),
            );
        }
        run_emitter_stack(
            &emitter.emitter_update,
            STAGE_EMITTER_UPDATE,
            &env,
            state,
            &mut spawn_count,
            &|m| key(STAGE_EMITTER_UPDATE, m, u32::MAX, frame),
        );

        // 3. Particle Update and EveryParticle handlers, then remove the dead.
        let events = &mut state.events;
        events.iter_mut().for_each(Vec::clear);
        for p in &mut state.particles {
            let mut cx = ParticleCx::default();
            for (mi, m) in emitter.particle_update.iter().enumerate() {
                let k = key(STAGE_PARTICLE_UPDATE, mi, p.unique_id, frame + 1);
                run_particle_module(m, &env, p, k, &mut cx);
            }
            for (hi, h) in emitter.event_handlers.iter().enumerate() {
                for (rank, ev) in every[hi].iter().enumerate() {
                    let mut hcx = ParticleCx {
                        event: Some(ev),
                        ..Default::default()
                    };
                    let kf = fx_events::every_particle_frame(frame, rank as u32);
                    for (mi, m) in h.modules.iter().enumerate() {
                        let k = key(fx_events::handler_stage(hi), mi, p.unique_id, kf);
                        run_particle_module(m, &env, p, k, &mut hcx);
                    }
                }
            }
            if cx.death && !p.alive {
                cx.pending[EventKind::Death.index()] = Some(event_of(p, Vec3::ZERO));
            }
            for (kind, ev) in cx.pending.into_iter().enumerate() {
                if let Some(ev) = ev {
                    events[kind].push(ev);
                }
            }
        }
        for ev in events.iter_mut() {
            ev.sort_by_key(|e| e.particle_id);
        }
        state.particles.retain(|p| p.alive);

        // 4. Spawn: the emitter stack's request, then the handlers'.
        let mut room = (emitter.max_particles as usize).saturating_sub(state.particles.len()) as u32;
        let n = spawn_count.min(room);
        room -= n;
        let mut spawned = n;
        let spawn_one = |state: &mut EmitterState, h: Option<(usize, &Event)>| {
            let mut p = Particle::new(state.next_id);
            state.next_id = state.next_id.wrapping_add(1);
            let mut cx = ParticleCx::default();
            for (mi, m) in emitter.particle_spawn.iter().enumerate() {
                let k = key(STAGE_PARTICLE_SPAWN, mi, p.unique_id, 0);
                run_particle_module(m, &env, &mut p, k, &mut cx);
            }
            if let Some((hi, ev)) = h {
                let mut cx = ParticleCx {
                    event: Some(ev),
                    ..Default::default()
                };
                for (mi, m) in emitter.event_handlers[hi].modules.iter().enumerate() {
                    let k = key(fx_events::handler_stage(hi), mi, p.unique_id, 0);
                    run_particle_module(m, &env, &mut p, k, &mut cx);
                }
            }
            if p.alive {
                state.particles.push(p);
            }
        };
        for _ in 0..n {
            spawn_one(state, None);
        }
        for (hi, h) in emitter.event_handlers.iter().enumerate() {
            if h.every_particle || h.spawn_number == 0 {
                continue;
            }
            let src = self.emitters[h.source as usize].events[h.event as usize].clone();
            let state = &mut self.emitters[ei];
            let want = handled(&src, h.max_events) as u64 * h.spawn_number as u64;
            let count = want.min(room as u64) as u32;
            room -= count;
            spawned += count;
            for k in 0..count {
                spawn_one(state, Some((hi, &src[(k / h.spawn_number) as usize])));
            }
        }
        let state = &mut self.emitters[ei];
        state.spawned_last_step = spawned;
        state.age += dt;
    }
}

/// What inputs read: parameters, and the age curves are sampled at.
pub trait InputEnv {
    fn param(&self, r: ParamRef) -> Value;
    /// `Particles.NormalizedAge` where there is a particle, else 0.
    fn normalized_age(&self) -> f32;
    /// The camera (Camera Query), if known.
    fn camera(&self) -> Option<CameraData> {
        None
    }
    /// Distance from the camera to the scene seen through `pos` (Scene Depth).
    fn scene_depth(&self, _pos: Vec3) -> f32 {
        NO_SCENE_DEPTH
    }
}

struct ParticleEnv<'a, 'b> {
    env: &'a Env<'b>,
    p: Option<&'a Particle>,
    event: Option<&'a Event>,
}

impl InputEnv for ParticleEnv<'_, '_> {
    fn param(&self, r: ParamRef) -> Value {
        read_param(r, self.env, self.p, self.event)
    }
    fn normalized_age(&self) -> f32 {
        self.p.map_or(0.0, |p| p.normalized_age)
    }
    fn camera(&self) -> Option<CameraData> {
        self.env.camera
    }
    fn scene_depth(&self, pos: Vec3) -> f32 {
        let Some(cam) = self.env.camera else {
            return NO_SCENE_DEPTH;
        };
        let Some(dir) = (pos - cam.position).try_normalize() else {
            return NO_SCENE_DEPTH;
        };
        self.env
            .colliders
            .iter()
            .filter_map(|c| c.ray_distance(cam.position, dir))
            .fold(NO_SCENE_DEPTH, f32::min)
    }
}

/// Evaluates input `slot` of module `m`.
pub fn eval_input(m: &CookedModule, slot: usize, env: &dyn InputEnv, key: &RandKey) -> Value {
    eval_at(m, &m.inputs[slot], slot as u32 * 8, env, key)
}

/// Uniform per component of `ty` between `lo` and `hi` (Int: whole numbers,
/// inclusive), with draws `salt + component`.
fn random_range(lo: Value, hi: Value, salt: u32, key: &RandKey) -> Value {
    match (lo, hi) {
        (Value::Int(lo), Value::Int(hi)) => {
            let (lo, hi) = (lo.min(hi), lo.max(hi));
            let span = (hi as i64 - lo as i64 + 1) as f32;
            let k = (key.unit(salt) * span).floor() as i64;
            Value::Int((lo as i64 + k).min(hi as i64) as i32)
        }
        _ => {
            let (a, b) = (lo.to_array(), hi.to_array());
            let mut out = [0.0f32; 4];
            for c in 0..lo.ty().components() {
                out[c] = a[c] + (b[c] - a[c]) * key.unit(salt + c as u32);
            }
            Value::from_array(lo.ty(), out)
        }
    }
}

/// Evaluates `input` (of module `m`); its random draws use `salt` (+ component).
pub fn eval_at(m: &CookedModule, input: &CookedInput, salt: u32, env: &dyn InputEnv, key: &RandKey) -> Value {
    match input {
        CookedInput::Const(v) => *v,
        CookedInput::Link(r) => env.param(*r),
        CookedInput::Random(a, b) => random_range(*a, *b, salt, key),
        CookedInput::Curve(c) => Value::from_array(c.ty, c.sample(env.normalized_age())),
        CookedInput::Dynamic(i) => eval_dynamic(m, *i as usize, env, key),
    }
}

fn eval_dynamic(m: &CookedModule, index: usize, env: &dyn InputEnv, key: &RandKey) -> Value {
    use crate::dynamic::{self, Op};
    let node = &m.dynamics[index];
    let Some(def) = dynamic::from_id(node.def) else {
        return Value::Float(0.0);
    };
    let salt = dynamic::node_salt(index);
    let arg = |i: usize| eval_at(m, &node.args[i], dynamic::arg_salt(index, i), env, key);
    let ty = def.ty;
    let n = ty.components();
    let zip = |a: Value, b: Value, f: &dyn Fn(f32, f32) -> f32| {
        let (a, b) = (a.to_array(), b.to_array());
        Value::from_array(ty, std::array::from_fn(|c| if c < n { f(a[c], b[c]) } else { 0.0 }))
    };
    match def.op {
        Op::UniformRanged => random_range(arg(0), arg(1), salt, key),
        Op::RandomByUniform => {
            let u = key.unit(salt);
            zip(arg(0), arg(1), &|a, b| a + (b - a) * u)
        }
        Op::FromCurve => {
            let t = arg(1).as_f32();
            match &node.args[0] {
                CookedInput::Curve(c) => Value::from_array(c.ty, c.sample(t)),
                _ => Value::from_array(ty, [0.0; 4]),
            }
        }
        Op::Add => zip(arg(0), arg(1), &|a, b| a + b),
        Op::Subtract => zip(arg(0), arg(1), &|a, b| a - b),
        Op::Multiply => zip(arg(0), arg(1), &|a, b| a * b),
        Op::MultiplyByFloat => {
            let s = arg(1).as_f32();
            zip(arg(0), arg(0), &|a, _| a * s)
        }
        Op::Divide => {
            let (a, b) = (arg(0).as_f32(), arg(1).as_f32());
            Value::Float(if b == 0.0 { 0.0 } else { a / b })
        }
        Op::Lerp => {
            let t = arg(2).as_f32();
            zip(arg(0), arg(1), &|a, b| a + (b - a) * t)
        }
        Op::Clamp => Value::Float(arg(0).as_f32().max(arg(1).as_f32()).min(arg(2).as_f32())),
        Op::MakeFromFloats => {
            let mut a = [0.0f32; 4];
            for (c, x) in a.iter_mut().enumerate().take(n) {
                *x = arg(c).as_f32();
            }
            Value::from_array(ty, a)
        }
        Op::MakeColorFromVec3 => {
            let v = arg(0).as_vec3();
            Value::Color([v.x, v.y, v.z, arg(1).as_f32()])
        }
        Op::Break => {
            let c = arg(1).as_i32().clamp(0, 3) as usize;
            Value::Float(arg(0).to_array()[c])
        }
        Op::Normalize => {
            let v = arg(0).as_vec3();
            let rl = 1.0 / v.length();
            let out = if rl > 0.0 && rl <= f32::MAX { v * rl } else { Vec3::ZERO };
            Value::Vec3(out.to_array())
        }
        Op::Length => Value::Float(arg(0).as_vec3().length()),
        Op::Distance => Value::Float((arg(0).as_vec3() - arg(1).as_vec3()).length()),
        Op::Sine | Op::Cosine => {
            let (amp, freq, phase, offset, t) = (
                arg(0).as_f32(),
                arg(1).as_f32(),
                arg(2).as_f32(),
                arg(3).as_f32(),
                arg(4).as_f32(),
            );
            let x = std::f32::consts::TAU * (freq * t + phase);
            Value::Float(offset + amp * if def.op == Op::Sine { x.sin() } else { x.cos() })
        }
        Op::SelectByBool => {
            if arg(0).as_bool() {
                arg(1)
            } else {
                arg(2)
            }
        }
        Op::SelectByInt => {
            let count = node.args.len() - 1;
            let i = arg(0).as_i32().clamp(0, count as i32 - 1) as usize;
            arg(1 + i)
        }
        Op::RandomFromList => {
            let count = node.args.len();
            let i = ((key.unit(salt) * count as f32) as usize).min(count - 1);
            arg(i)
        }
        Op::RandomUnitVector => {
            let (u1, u2) = (key.unit(salt), key.unit(salt + 1));
            let z = 1.0 - 2.0 * u1;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = std::f32::consts::TAU * u2;
            Value::Vec3([r * phi.cos(), r * phi.sin(), z])
        }
        Op::NormalizedAge => {
            let (age, dur) = (arg(0).as_f32(), arg(1).as_f32());
            Value::Float(if dur > 0.0 { (age / dur).clamp(0.0, 1.0) } else { 1.0 })
        }
        Op::CameraPosition => Value::Vec3(env.camera().map_or(Vec3::ZERO, |c| c.position).to_array()),
        Op::CameraForward => {
            Value::Vec3(env.camera().map_or(Vec3::NEG_Z, |c| c.forward).to_array())
        }
        Op::SceneDepth => Value::Float(env.scene_depth(arg(1).as_vec3())),
        Op::Compare => {
            let (a, b) = (arg(0).as_f32(), arg(1).as_f32());
            Value::Bool(match arg(2).as_i32() {
                0 => a < b,
                1 => a <= b,
                2 => a > b,
                3 => a >= b,
                4 => a == b,
                _ => a != b,
            })
        }
    }
}

fn read_param(r: ParamRef, env: &Env, p: Option<&Particle>, event: Option<&Event>) -> Value {
    match r.ns {
        Namespace::User => env
            .user
            .get(r.index as usize)
            .copied()
            .unwrap_or(Value::Float(0.0)),
        Namespace::Engine => match r.index {
            0 => Value::Float(env.dt),
            1 => Value::Float(env.time),
            _ => Value::Vec3(env.owner_velocity.to_array()),
        },
        Namespace::System => Value::Float(env.time),
        Namespace::Emitter => Value::Float(env.emitter_age),
        Namespace::Particles => p.map_or(Value::Float(0.0), |p| p.get(r.index)),
        Namespace::Event => {
            let e = event.copied().unwrap_or_default();
            match r.index {
                0 => Value::Vec3(e.position),
                1 => Value::Vec3(e.velocity),
                2 => Value::Vec3(e.normal),
                3 => Value::Float(e.age),
                _ => Value::Int(e.particle_id as i32),
            }
        }
    }
}

fn run_emitter_stack(
    stack: &[CookedModule],
    _stage: u32,
    env: &Env,
    state: &mut EmitterState,
    spawn_count: &mut u32,
    key: &dyn Fn(usize) -> RandKey,
) {
    for (mi, m) in stack.iter().enumerate() {
        let k = key(mi);
        let pe = ParticleEnv {
            env,
            p: None,
            event: None,
        };
        let input = |slot: usize| eval_input(m, slot, &pe, &k);
        match modules::from_id(m.module) {
            Some(ModuleKind::SpawnRate) => {
                let rate = input(0).as_f32().max(0.0);
                state.spawn_remainder += rate * env.dt;
                let whole = state.spawn_remainder.floor();
                state.spawn_remainder -= whole;
                *spawn_count = spawn_count.saturating_add(whole as u32);
            }
            Some(ModuleKind::SpawnBurstInstantaneous) => {
                let count = input(0).as_i32().max(0) as u32;
                let t = input(1).as_f32();
                if env.emitter_age <= t && t < env.emitter_age + env.dt {
                    *spawn_count = spawn_count.saturating_add(count);
                }
            }
            _ => {}
        }
    }
}

/// Orthonormal basis around unit vector `n` (Duff et al. 2017), as `(t1, t2)`.
pub fn basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = 1.0f32.copysign(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// Half of a degree in radians: Euler angles in degrees become half-angles.
const HALF_RADIANS_PER_DEGREE: f32 = std::f32::consts::PI / 360.0;

/// Quaternion `(x, y, z, w)` rotating by Euler angles `deg` (degrees about
/// X, Y and Z, X applied first): `qz * qy * qx`, written out. The GPU's
/// `euler_quat` in `sim_common.wgsl` is the same formula.
pub fn euler_quat(deg: Vec3) -> Quat {
    let h = deg * HALF_RADIANS_PER_DEGREE;
    let (sx, cx) = (h.x.sin(), h.x.cos());
    let (sy, cy) = (h.y.sin(), h.y.cos());
    let (sz, cz) = (h.z.sin(), h.z.cos());
    Quat::from_xyzw(
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    )
}

/// Hamilton product `a * b` (rotation `b` first, then `a`), written out so
/// the GPU's `quat_mul` matches it operation for operation.
pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    Quat::from_xyzw(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )
}

/// `q` scaled to unit length, or the identity if its length is 0 or not finite.
pub fn quat_normalize(q: Quat) -> Quat {
    let rl = 1.0 / Vec4::from_array(q.to_array()).length();
    if rl > 0.0 && rl <= f32::MAX {
        Quat::from_xyzw(q.x * rl, q.y * rl, q.z * rl, q.w * rl)
    } else {
        Quat::IDENTITY
    }
}

/// Largest normalized age the flipbook reads, so the last frame is shown at
/// the end of life instead of wrapping to the first (as the renderer does).
pub const SUB_UV_LAST: f32 = 0.99999;

/// `(1 - dist / radius) ^ exponent`, for `0 <= dist < radius`.
fn falloff(dist: f32, radius: f32, exponent: f32) -> f32 {
    (1.0 - dist / radius).powf(exponent)
}

/// `v` normalised, or +Y if it has no direction.
fn unit_or_y(v: Vec3) -> Vec3 {
    let rl = 1.0 / v.length();
    if rl > 0.0 && rl <= f32::MAX {
        v * rl
    } else {
        Vec3::Y
    }
}

/// For offset `r` from a point on unit `axis`: the unit direction around the
/// axis (counter-clockwise seen from its tip) and the unit direction towards
/// it; both zero on the axis.
fn vortex_dirs(axis: Vec3, r: Vec3) -> (Vec3, Vec3) {
    let radial = r - axis * axis.dot(r);
    let len = radial.length();
    if len > 0.0 {
        let out = radial / len;
        (axis.cross(out), -out)
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    }
}

/// `clamp(|velocity| / threshold, 0, 1)`; 1 when the threshold is not positive.
fn speed_fraction(velocity: Vec3, threshold: f32) -> f32 {
    if threshold > 0.0 {
        (velocity.length() / threshold).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Shape Location's size inputs.
pub struct ShapeInputs {
    pub surface: bool,
    pub sphere_radius: f32,
    pub box_size: Vec3,
    pub cylinder_height: f32,
    pub cylinder_radius: f32,
    pub large_radius: f32,
    pub handle_radius: f32,
    pub cone_length: f32,
    pub cone_angle: f32,
    pub ring_radius: f32,
    pub disc_coverage: f32,
}

/// A point of shape `primitive` (Shape Location's "Shape Primitive" index)
/// from five uniform draws `u`, uniform over the volume (or surface). Radial
/// draws use `1 - u`, which is never 0, so roots and powers stay defined on
/// the GPU. `shape_point` in `fx_gpu`'s `sim_common.wgsl` is the same code.
pub fn shape_point(primitive: i32, s: &ShapeInputs, u: [f32; 5]) -> Vec3 {
    const TAU: f32 = std::f32::consts::TAU;
    let around = |r: f32, phi: f32, y: f32| Vec3::new(r * phi.cos(), y, r * phi.sin());
    match primitive {
        // Box: a face chosen by area, then a point on it.
        1 => {
            let size = s.box_size;
            let mut p = Vec3::new(u[0] - 0.5, u[1] - 0.5, u[2] - 0.5) * size;
            if s.surface {
                let (ax, ay) = (size.y * size.z, size.x * size.z);
                let t = u[3] * (ax + ay + size.x * size.y);
                let side = if u[4] < 0.5 { -0.5 } else { 0.5 };
                if t < ax {
                    p.x = side * size.x;
                } else if t < ax + ay {
                    p.y = side * size.y;
                } else {
                    p.z = side * size.z;
                }
            }
            p
        }
        // Cylinder around Y, centred; Surface Only = the side wall.
        2 => {
            let r = if s.surface { 1.0 } else { (1.0 - u[1]).sqrt() };
            around(
                s.cylinder_radius * r,
                TAU * u[0],
                (u[2] - 0.5) * s.cylinder_height,
            )
        }
        // Torus around Y: the tube's cross-section sampled as a disc.
        3 => {
            let theta = TAU * u[0];
            let psi = TAU * u[1];
            let rho = s.handle_radius * if s.surface { 1.0 } else { (1.0 - u[2]).sqrt() };
            around(s.large_radius + rho * psi.cos(), theta, rho * psi.sin())
        }
        // Cone: apex at the origin, opening along +Y.
        4 => {
            let h = s.cone_length
                * if s.surface {
                    (1.0 - u[0]).sqrt()
                } else {
                    (1.0 - u[0]).powf(1.0 / 3.0)
                };
            let r = h
                * (s.cone_angle.clamp(0.0, 89.9) * DEGREES).tan()
                * if s.surface { 1.0 } else { (1.0 - u[1]).sqrt() };
            around(r, TAU * u[2], h)
        }
        // Ring in the XZ plane; Disc Coverage fills it inwards, uniform by area.
        5 => around(
            s.ring_radius * (1.0 - s.disc_coverage.clamp(0.0, 1.0) * u[1]).sqrt(),
            TAU * u[0],
            0.0,
        ),
        // Sphere.
        _ => {
            let y = 1.0 - 2.0 * u[0];
            let phi = TAU * u[1];
            let ring = (1.0 - y * y).max(0.0).sqrt();
            let r = s.sphere_radius * if s.surface { 1.0 } else { (1.0 - u[2]).powf(1.0 / 3.0) };
            Vec3::new(ring * phi.cos(), y, ring * phi.sin()) * r
        }
    }
}

/// Collision Type values of the Collision module.
pub const COLLISION_PLANE: i32 = 0;
pub const COLLISION_SCENE_DEPTH: i32 = 1;
pub const COLLISION_RAY_TRACED: i32 = 2;
/// Collision Response values.
pub const RESPONSE_BOUNCE: i32 = 0;
pub const RESPONSE_STOP: i32 = 1;
pub const RESPONSE_KILL: i32 = 2;

/// The Collision module's response inputs.
pub struct CollisionResponse {
    pub response: i32,
    pub restitution: f32,
    pub friction: f32,
    pub rest_speed: f32,
}

/// Collides particle `p` (radius `radius`) with the plane through `o` facing
/// unit `n`: if it is behind the plane it moves back onto it and bounces,
/// stops or dies. Returns whether this counts as a collision (an impact at
/// `rest_speed` or faster into the surface). `collide_plane` in fx_gpu's
/// `sim_common.wgsl` is the same code.
pub fn collide_plane(p: &mut Particle, n: Vec3, o: Vec3, radius: f32, c: &CollisionResponse) -> bool {
    let d = (p.position - o).dot(n) - radius;
    if !(d < 0.0) {
        return false;
    }
    p.position -= n * d;
    let vn = p.velocity.dot(n);
    if vn >= 0.0 {
        // Already leaving the surface.
        return false;
    }
    let speed_in = -vn;
    let impact = speed_in >= c.rest_speed;
    match c.response {
        RESPONSE_KILL => {
            p.alive = false;
            return true;
        }
        RESPONSE_STOP => {
            p.velocity = Vec3::ZERO;
            return impact;
        }
        _ => {}
    }
    let mut vt = p.velocity - n * vn;
    let t = vt.length();
    if t > 0.0 {
        vt *= (1.0 - c.friction * speed_in / t).max(0.0);
    }
    let out = if impact { speed_in * c.restitution } else { 0.0 };
    p.velocity = vt + n * out;
    impact
}

/// Radians per degree.
const DEGREES: f32 = std::f32::consts::PI / 180.0;

fn run_particle_module(m: &CookedModule, env: &Env, p: &mut Particle, key: RandKey, cx: &mut ParticleCx) {
    let Some(kind) = modules::from_id(m.module) else {
        return;
    };
    let event = cx.event;
    let input = |slot: usize, p: &Particle| {
        let pe = ParticleEnv {
            env,
            p: Some(p),
            event,
        };
        eval_input(m, slot, &pe, &key)
    };
    match kind {
        ModuleKind::InitializeParticle => {
            p.lifetime = input(0, p).as_f32();
            p.position = input(1, p).as_vec3();
            p.color = input(2, p).as_vec4();
            p.sprite_size = input(3, p).as_vec2();
            p.sprite_rotation = input(4, p).as_f32();
            p.mass = input(5, p).as_f32();
            p.initial_color = p.color;
            p.initial_sprite_size = p.sprite_size;
        }
        ModuleKind::AddVelocity => {
            p.velocity += input(0, p).as_vec3();
        }
        ModuleKind::AddVelocityInCone => {
            let axis = input(0, p).as_vec3().try_normalize().unwrap_or(Vec3::Y);
            let angle = input(1, p).as_f32().clamp(0.0, 180.0).to_radians();
            let strength = input(2, p).as_f32();
            let (u1, u2) = (key.unit(128), key.unit(129));
            // Uniform over the spherical cap: cos θ uniform in [cos angle, 1].
            let cos_t = 1.0 - u1 * (1.0 - angle.cos());
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let phi = std::f32::consts::TAU * u2;
            let (t1, t2) = basis(axis);
            let dir = t1 * (sin_t * phi.cos()) + t2 * (sin_t * phi.sin()) + axis * cos_t;
            p.velocity += dir * strength;
        }
        ModuleKind::GravityForce => {
            p.force += input(0, p).as_vec3() * p.mass;
        }
        ModuleKind::Drag => {
            p.drag += input(0, p).as_f32();
        }
        ModuleKind::CurlNoiseForce => {
            let strength = input(0, p).as_f32();
            let freq = input(1, p).as_f32();
            let offset = input(2, p).as_vec3();
            let pan = input(3, p).as_vec3();
            p.force += noise::curl_noise(p.position * freq + offset + pan * env.time) * strength;
        }
        ModuleKind::ScaleColor => {
            let rgb = input(0, p).as_vec3();
            let a = input(1, p).as_f32();
            p.color = p.initial_color * rgb.extend(a);
        }
        ModuleKind::ScaleSpriteSize => {
            p.sprite_size = p.initial_sprite_size * input(0, p).as_vec2();
        }
        ModuleKind::SolveForcesAndVelocity => {
            if p.mass > 0.0 {
                p.velocity += p.force / p.mass * env.dt;
            }
            p.velocity *= (-p.drag * env.dt).exp();
            p.position += p.velocity * env.dt;
            p.force = Vec3::ZERO;
            p.drag = 0.0;
        }
        ModuleKind::ParticleState => {
            p.age += env.dt;
            p.normalized_age = if p.lifetime > 0.0 {
                p.age / p.lifetime
            } else {
                1.0
            };
            if input(0, p).as_bool() && p.age > p.lifetime {
                p.alive = false;
            }
        }
        ModuleKind::KillParticles => {
            if input(0, p).as_bool() {
                p.alive = false;
            }
        }
        ModuleKind::InitializeRibbon => {
            p.ribbon_width = input(0, p).as_f32();
            p.initial_ribbon_width = p.ribbon_width;
            p.ribbon_twist = input(1, p).as_f32();
            p.ribbon_id = input(2, p).as_i32();
            p.ribbon_link_order = input(3, p).as_f32();
        }
        ModuleKind::InitializeMesh => {
            p.scale = input(0, p).as_vec3();
            p.initial_scale = p.scale;
            p.mesh_orientation = euler_quat(input(1, p).as_vec3());
        }
        ModuleKind::UpdateMeshOrientation => {
            let rate = input(0, p).as_vec3();
            let step = euler_quat(rate * env.dt);
            p.mesh_orientation = quat_normalize(quat_mul(p.mesh_orientation, step));
        }
        ModuleKind::InitializeLight => {
            p.light_radius = input(0, p).as_f32();
        }
        ModuleKind::DynamicMaterialParameters => {
            p.dynamic_material_parameter = input(0, p).as_vec4();
        }
        ModuleKind::PointAttractionForce => {
            let center = input(0, p).as_vec3();
            let strength = input(1, p).as_f32();
            let radius = input(2, p).as_f32();
            let exponent = input(3, p).as_f32();
            let kill = input(4, p).as_f32();
            let d = center - p.position;
            let dist = d.length();
            if dist < kill {
                p.alive = false;
            }
            if dist > 0.0 && dist < radius {
                p.force += d * (strength * falloff(dist, radius, exponent) / dist);
            }
        }
        ModuleKind::VortexVelocity => {
            let axis = unit_or_y(input(0, p).as_vec3());
            let origin = input(1, p).as_vec3();
            let amount = input(2, p).as_f32();
            let (tangent, _) = vortex_dirs(axis, p.position - origin);
            p.velocity += tangent * amount;
        }
        ModuleKind::VortexForce => {
            let axis = unit_or_y(input(0, p).as_vec3());
            let origin = input(1, p).as_vec3();
            let amount = input(2, p).as_f32();
            let pull = input(3, p).as_f32();
            let (tangent, inward) = vortex_dirs(axis, p.position - origin);
            p.force += tangent * amount + inward * pull;
        }
        ModuleKind::WindForce => {
            let wind = input(0, p).as_vec3();
            let speed = input(1, p).as_f32();
            let resistance = input(2, p).as_f32();
            p.force += (wind * speed - p.velocity) * resistance;
        }
        ModuleKind::LinearForce => {
            p.force += input(0, p).as_vec3();
        }
        ModuleKind::LimitForce => {
            let limit = input(0, p).as_f32().max(0.0);
            let len = p.force.length();
            if len > limit {
                p.force *= limit / len;
            }
        }
        ModuleKind::AccelerationForce => {
            p.force += input(0, p).as_vec3() * p.mass;
        }
        ModuleKind::PointForce => {
            let center = input(0, p).as_vec3();
            let strength = input(1, p).as_f32();
            let radius = input(2, p).as_f32();
            let exponent = input(3, p).as_f32();
            let d = p.position - center;
            let dist = d.length();
            if dist > 0.0 && dist < radius {
                p.force += d * (strength * falloff(dist, radius, exponent) / dist);
            }
        }
        ModuleKind::InheritVelocity => {
            p.velocity += env.owner_velocity * input(0, p).as_f32();
        }
        ModuleKind::ShapeLocation => {
            let s = ShapeInputs {
                surface: input(1, p).as_bool(),
                sphere_radius: input(3, p).as_f32(),
                box_size: input(4, p).as_vec3(),
                cylinder_height: input(5, p).as_f32(),
                cylinder_radius: input(6, p).as_f32(),
                large_radius: input(7, p).as_f32(),
                handle_radius: input(8, p).as_f32(),
                cone_length: input(9, p).as_f32(),
                cone_angle: input(10, p).as_f32(),
                ring_radius: input(11, p).as_f32(),
                disc_coverage: input(12, p).as_f32(),
            };
            let u: [f32; 5] = std::array::from_fn(|k| key.unit(128 + k as u32));
            p.position = input(2, p).as_vec3() + shape_point(input(0, p).as_i32(), &s, u);
        }
        ModuleKind::InitialMeshOrientation => {
            p.mesh_orientation = euler_quat(input(0, p).as_vec3());
        }
        ModuleKind::SpriteRotationRate => {
            p.sprite_rotation += input(0, p).as_f32() * env.dt;
        }
        ModuleKind::SubUvAnimation => {
            let n = input(1, p).as_i32();
            let frames = if n > 0 { n as f32 } else { env.sub_image_frames as f32 };
            let start = input(2, p).as_i32() as f32;
            let f = match input(0, p).as_i32() {
                // One draw per particle: the key ignores the step.
                1 => (RandKey { frame: 0, ..key }.unit(128) * frames).floor(),
                2 => {
                    let x = p.age * input(3, p).as_f32();
                    x - frames * (x / frames).floor()
                }
                _ => p.normalized_age.clamp(0.0, SUB_UV_LAST) * frames,
            };
            p.sub_image_index = start + f;
        }
        ModuleKind::ScaleMeshSize => {
            p.scale = p.initial_scale * input(0, p).as_vec3();
        }
        ModuleKind::ScaleRibbonWidth => {
            p.ribbon_width = p.initial_ribbon_width * input(0, p).as_f32();
        }
        ModuleKind::Color => {
            p.color = input(0, p).as_vec4();
        }
        ModuleKind::ScaleColorBySpeed => {
            let t = speed_fraction(p.velocity, input(0, p).as_f32());
            let (lo, hi) = (input(1, p).as_vec4(), input(2, p).as_vec4());
            p.color = p.initial_color * (lo + (hi - lo) * t);
        }
        ModuleKind::ScaleSpriteSizeBySpeed => {
            let t = speed_fraction(p.velocity, input(0, p).as_f32());
            let (lo, hi) = (input(1, p).as_vec2(), input(2, p).as_vec2());
            p.sprite_size = p.initial_sprite_size * (lo + (hi - lo) * t);
        }
        ModuleKind::KillParticlesInVolume => {
            let d = p.position - input(1, p).as_vec3();
            let inside = if input(0, p).as_i32() == 1 {
                let h = input(3, p).as_vec3() * 0.5;
                d.x.abs() < h.x && d.y.abs() < h.y && d.z.abs() < h.z
            } else {
                let r = input(2, p).as_f32();
                d.dot(d) < r * r
            };
            if inside != input(4, p).as_bool() {
                p.alive = false;
            }
        }
        ModuleKind::Collision => {
            let ty = input(0, p).as_i32();
            let response = input(1, p).as_i32();
            let radius = input(4, p).as_f32();
            let c = CollisionResponse {
                response,
                restitution: input(5, p).as_f32(),
                friction: input(6, p).as_f32(),
                rest_speed: input(7, p).as_f32(),
            };
            let contact = if ty == COLLISION_PLANE {
                Some((unit_or_y(input(2, p).as_vec3()), input(3, p).as_vec3()))
            } else {
                env.colliders.iter().find_map(|c| c.contact(p.position, radius))
            };
            if let Some((n, o)) = contact
                && collide_plane(p, n, o, radius, &c)
            {
                cx.collision = Some(n);
            }
        }
        ModuleKind::GenerateDeathEvent => cx.death = true,
        ModuleKind::GenerateCollisionEvent => {
            if let Some(n) = cx.collision {
                cx.pending[EventKind::Collision.index()] = Some(event_of(p, n));
            }
        }
        ModuleKind::GenerateLocationEvent => {
            if key.unit(128) < input(0, p).as_f32() {
                cx.pending[EventKind::Location.index()] = Some(event_of(p, Vec3::ZERO));
            }
        }
        ModuleKind::ReceiveDeathEvent
        | ModuleKind::ReceiveCollisionEvent
        | ModuleKind::ReceiveLocationEvent => {
            let e = cx.event.copied().unwrap_or_default();
            p.position = Vec3::from_array(e.position) + input(0, p).as_vec3();
            p.velocity += Vec3::from_array(e.velocity) * input(1, p).as_f32();
        }
        ModuleKind::SkeletalMeshLocation | ModuleKind::StaticMeshLocation => {
            let skeletal = kind == ModuleKind::SkeletalMeshLocation;
            let di = input(0, p).as_i32();
            let (mode, offset, inherit) = if skeletal {
                (input(1, p).as_i32(), input(2, p).as_f32(), input(3, p).as_f32())
            } else {
                (data::SAMPLE_SURFACE, input(1, p).as_f32(), input(2, p).as_f32())
            };
            let Some(DiBinding::Mesh(mesh, pose)) = env.data.get(di.max(0) as usize) else {
                return;
            };
            let u: [f32; 3] = std::array::from_fn(|k| key.unit(128 + k as u32));
            let s = if mode == data::SAMPLE_BONES {
                data::sample_bone(mesh, pose, u[0])
            } else {
                data::sample_surface(mesh, pose, u)
            };
            if let Some(s) = s {
                p.position = s.position + s.normal * offset;
                p.velocity += s.velocity * inherit;
            }
        }
        ModuleKind::SampleTexture => {
            let uv = input(1, p).as_vec2().to_array();
            let c = match env.data.get(input(0, p).as_i32().max(0) as usize) {
                Some(DiBinding::Texture(t)) => Vec4::from(data::sample_texture(t, uv)),
                _ => Vec4::ONE,
            } * input(2, p).as_vec4();
            if input(3, p).as_i32() == 1 {
                p.color *= c;
            } else {
                p.color = c;
            }
        }
        ModuleKind::CameraOffset => {
            let amount = input(0, p).as_f32();
            if input(1, p).as_i32() == 1 {
                p.camera_offset += amount;
            } else {
                p.camera_offset = amount;
            }
        }
        ModuleKind::SpawnRate | ModuleKind::SpawnBurstInstantaneous => {}
    }
}

/// The type a link resolves to (for tools; validation already checked it).
pub fn param_type(r: ParamRef, sys: &CookedSystem) -> ValueType {
    match r.ns {
        Namespace::User => sys.user_params[r.index as usize].value.ty(),
        ns => crate::params::builtins(ns)[r.index as usize].ty,
    }
}
