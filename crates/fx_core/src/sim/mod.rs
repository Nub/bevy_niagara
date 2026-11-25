//! CPU reference simulation of the module set.
//!
//! This is the ground truth the GPU simulation must match: deterministic for
//! a given seed, with every random draw a hash of where it happens
//! ([`rng::RandKey`]), never a stateful generator.
//!
//! Each [`Simulation::step`], per emitter, in order:
//! 1. on the first step, the Emitter Spawn stack;
//! 2. the Emitter Update stack, which decides how many particles to spawn;
//! 3. the Particle Update stack over the living particles, then dead ones are
//!    removed (order kept);
//! 4. new particles are created (capped at `max_particles`) and run the
//!    Particle Spawn stack. They first update on the next step.
//!
//! `Engine.Time` and the ages read during a step are their values at the
//! start of the step.

pub mod noise;
pub mod rng;

use bevy_math::{Vec2, Vec3, Vec4};

use crate::cook::{CookedInput, CookedModule, CookedSystem};
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
            _ => Value::Float(0.0),
        }
    }

    /// Bit pattern of every field, for exact comparisons and hashing.
    pub fn bits(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(28);
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
}

impl Simulation {
    pub fn new(system: CookedSystem, seed: u32) -> Self {
        let user = system.user_params.iter().map(|p| p.value).collect();
        let emitters = vec![EmitterState::default(); system.emitters.len()];
        Simulation {
            system,
            seed,
            frame: 0,
            time: 0.0,
            user,
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

    pub fn particles(&self, emitter: usize) -> &[Particle] {
        &self.emitters[emitter].particles
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
        let state = &mut self.emitters[ei];
        let env = Env {
            user: &self.user,
            dt,
            time: self.time,
            emitter_age: state.age,
        };
        let key = |stage: u32, module: usize, particle: u32, frame: u32| RandKey {
            seed: self.seed,
            emitter: ei as u32,
            particle,
            frame,
            module: stage * 256 + module as u32,
        };

        // 1-2. Emitter stages.
        let mut spawn_count: u32 = 0;
        if self.frame == 0 {
            run_emitter_stack(
                &emitter.emitter_spawn,
                STAGE_EMITTER_SPAWN,
                &env,
                state,
                &mut spawn_count,
                &|m| key(STAGE_EMITTER_SPAWN, m, u32::MAX, self.frame),
            );
        }
        run_emitter_stack(
            &emitter.emitter_update,
            STAGE_EMITTER_UPDATE,
            &env,
            state,
            &mut spawn_count,
            &|m| key(STAGE_EMITTER_UPDATE, m, u32::MAX, self.frame),
        );

        // 3. Particle Update, then remove the dead.
        for p in &mut state.particles {
            for (mi, m) in emitter.particle_update.iter().enumerate() {
                let k = key(STAGE_PARTICLE_UPDATE, mi, p.unique_id, self.frame + 1);
                run_particle_module(m, &env, p, k);
            }
        }
        state.particles.retain(|p| p.alive);

        // 4. Spawn.
        let room = (emitter.max_particles as usize).saturating_sub(state.particles.len()) as u32;
        let n = spawn_count.min(room);
        for _ in 0..n {
            let mut p = Particle::new(state.next_id);
            state.next_id = state.next_id.wrapping_add(1);
            for (mi, m) in emitter.particle_spawn.iter().enumerate() {
                let k = key(STAGE_PARTICLE_SPAWN, mi, p.unique_id, 0);
                run_particle_module(m, &env, &mut p, k);
            }
            if p.alive {
                state.particles.push(p);
            }
        }
        state.spawned_last_step = n;
        state.age += dt;
    }
}

fn eval(input: &CookedInput, slot: usize, env: &Env, p: Option<&Particle>, key: &RandKey) -> Value {
    match input {
        CookedInput::Const(v) => *v,
        CookedInput::Link(r) => read_param(*r, env, p),
        CookedInput::Random(a, b) => {
            let base = slot as u32 * 8;
            match (a, b) {
                (Value::Int(lo), Value::Int(hi)) => {
                    let (lo, hi) = (*lo.min(hi), *lo.max(hi));
                    let span = (hi as i64 - lo as i64 + 1) as f32;
                    let k = (key.unit(base) * span).floor() as i64;
                    Value::Int((lo as i64 + k).min(hi as i64) as i32)
                }
                _ => {
                    let (lo, hi) = (a.to_array(), b.to_array());
                    let n = a.ty().components();
                    let mut out = [0.0f32; 4];
                    for c in 0..n {
                        out[c] = lo[c] + (hi[c] - lo[c]) * key.unit(base + c as u32);
                    }
                    Value::from_array(a.ty(), out)
                }
            }
        }
        CookedInput::Curve(c) => {
            let t = p.map_or(0.0, |p| p.normalized_age);
            Value::from_array(c.ty, c.sample(t))
        }
    }
}

fn read_param(r: ParamRef, env: &Env, p: Option<&Particle>) -> Value {
    match r.ns {
        Namespace::User => env
            .user
            .get(r.index as usize)
            .copied()
            .unwrap_or(Value::Float(0.0)),
        Namespace::Engine => Value::Float(if r.index == 0 { env.dt } else { env.time }),
        Namespace::System => Value::Float(env.time),
        Namespace::Emitter => Value::Float(env.emitter_age),
        Namespace::Particles => p.map_or(Value::Float(0.0), |p| p.get(r.index)),
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
        let input = |slot: usize| eval(&m.inputs[slot], slot, env, None, &k);
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

fn run_particle_module(m: &CookedModule, env: &Env, p: &mut Particle, key: RandKey) {
    let Some(kind) = modules::from_id(m.module) else {
        return;
    };
    let input = |slot: usize, p: &Particle| eval(&m.inputs[slot], slot, env, Some(p), &key);
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
            p.force += noise::curl_noise(p.position * freq + offset) * strength;
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
