//! The emitter stacks run on the CPU: they only read user, engine and emitter
//! values (never particles), so the CPU decides each step's spawn request
//! exactly as `fx_core::sim` does and the GPU clamps it to the free room.

use fx_core::cook::{CookedInput, CookedModule};
use fx_core::modules::{self, ModuleKind};
use fx_core::params::{Namespace, ParamRef};
use fx_core::sim::rng::RandKey;
use fx_core::types::Value;

pub(crate) const STAGE_EMITTER_SPAWN: u32 = 2;
pub(crate) const STAGE_EMITTER_UPDATE: u32 = 3;

pub(crate) struct Env<'a> {
    pub user: &'a [Value],
    pub dt: f32,
    pub time: f32,
    pub emitter_age: f32,
    pub owner_velocity: [f32; 3],
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
        Namespace::Emitter => Value::Float(env.emitter_age),
        Namespace::Particles => Value::Float(0.0),
    }
}

fn eval(input: &CookedInput, slot: usize, env: &Env, key: &RandKey) -> Value {
    match input {
        CookedInput::Const(v) => *v,
        CookedInput::Link(r) => read_param(*r, env),
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
                    let mut out = [0.0f32; 4];
                    for c in 0..a.ty().components() {
                        out[c] = lo[c] + (hi[c] - lo[c]) * key.unit(base + c as u32);
                    }
                    Value::from_array(a.ty(), out)
                }
            }
        }
        CookedInput::Curve(c) => Value::from_array(c.ty, c.sample(0.0)),
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
    let mut spawn_count = 0u32;
    for (mi, m) in stack.iter().enumerate() {
        let k = RandKey {
            seed,
            emitter,
            particle: u32::MAX,
            frame,
            module: stage * 256 + mi as u32,
        };
        let input = |slot: usize| eval(&m.inputs[slot], slot, env, &k);
        match modules::from_id(m.module) {
            Some(ModuleKind::SpawnRate) => {
                let rate = input(0).as_f32().max(0.0);
                *spawn_remainder += rate * env.dt;
                let whole = spawn_remainder.floor();
                *spawn_remainder -= whole;
                spawn_count = spawn_count.saturating_add(whole as u32);
            }
            Some(ModuleKind::SpawnBurstInstantaneous) => {
                let count = input(0).as_i32().max(0) as u32;
                let t = input(1).as_f32();
                if env.emitter_age <= t && t < env.emitter_age + env.dt {
                    spawn_count = spawn_count.saturating_add(count);
                }
            }
            _ => {}
        }
    }
    spawn_count
}
