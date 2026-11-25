//! The emitter stacks run on the CPU: they only read user, engine and emitter
//! values (never particles), so the CPU decides each step's spawn request
//! with `fx_core::sim::run_emitter_stack`, the CPU simulation's own code, and
//! the GPU clamps it to the free room.

use fx_core::cook::CookedModule;
use fx_core::lifecycle::Lifecycle;
use fx_core::params::{Namespace, ParamRef};
use fx_core::sim::rng::RandKey;
use fx_core::sim::{EmitterStep, EmitterValues};
use fx_core::types::Value;

pub(crate) const STAGE_EMITTER_SPAWN: u32 = 2;
pub(crate) const STAGE_EMITTER_UPDATE: u32 = 3;

pub(crate) struct Env<'a> {
    pub user: &'a [Value],
    pub dt: f32,
    pub time: f32,
    pub emitter: EmitterValues,
    pub owner_velocity: [f32; 3],
    pub lifecycle: &'a Lifecycle,
}

fn read_param(r: ParamRef, env: &Env) -> Value {
    match r.ns {
        Namespace::User => env
            .user
            .get(r.index as usize)
            .copied()
            .unwrap_or(Value::Float(0.0)),
        Namespace::Engine => match r.index {
            0 => Value::Float(env.dt),
            1 => Value::Float(env.time),
            _ => Value::Vec3(env.owner_velocity),
        },
        Namespace::System => Value::Float(env.time),
        Namespace::Emitter => env.emitter.param(r.index),
        Namespace::Particles | Namespace::Event => Value::Float(0.0),
    }
}

impl fx_core::sim::InputEnv for Env<'_> {
    fn param(&self, r: ParamRef) -> Value {
        read_param(r, self)
    }
    fn normalized_age(&self) -> f32 {
        0.0
    }
}

/// Runs an emitter stack; returns the number of particles it asks to spawn.
pub(crate) fn run_stack(
    stack: &[CookedModule],
    stage: u32,
    env: &Env,
    spawn_remainder: &mut f32,
    seed: u32,
    emitter: u32,
    frame: u32,
) -> u32 {
    let step = EmitterStep {
        age: env.emitter.age,
        dt: env.dt,
        owner_velocity: bevy_math::Vec3::from_array(env.owner_velocity),
        lifecycle: env.lifecycle,
    };
    fx_core::sim::run_emitter_stack(stack, env, &step, spawn_remainder, &|mi| RandKey {
        seed,
        emitter,
        particle: u32::MAX,
        frame,
        module: stage * 256 + mi as u32,
    })
}
