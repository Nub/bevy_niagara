//! Compiles an emitter's particle module stacks to WGSL.
//!
//! The generated code mirrors `fx_core::sim::run_particle_module` statement
//! for statement. Constants are baked in as exact bit patterns; `User.`
//! parameters and engine values come from the per-step [`crate::FrameUniform`],
//! so changing a user parameter never recompiles.

use std::fmt::Write;

use fx_core::cook::{CookedCurve, CookedEmitter, CookedInput, CookedModule, CookedSystem};
use fx_core::modules::{self, ModuleKind};
use fx_core::params::{Namespace, ParamRef, attr};
use fx_core::types::{Value, ValueType};

const COMMON: &str = include_str!("wgsl/sim_common.wgsl");

/// Stage numbers used in random keys (as in `fx_core::sim`).
const STAGE_PARTICLE_SPAWN: u32 = 4;
const STAGE_PARTICLE_UPDATE: u32 = 5;

/// An exact f32 literal.
fn f(x: f32) -> String {
    if x == x.trunc() && x.abs() < 1.0e7 {
        // Small integers are exact as decimal literals and keep shaders readable.
        format!("{:.1}", x)
    } else {
        format!("bitcast<f32>({}u)", x.to_bits())
    }
}

fn wgsl_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Float => "f32",
        ValueType::Int => "i32",
        ValueType::Bool => "bool",
        ValueType::Vec2 => "vec2<f32>",
        ValueType::Vec3 => "vec3<f32>",
        ValueType::Vec4 | ValueType::Color | ValueType::Quat => "vec4<f32>",
    }
}

fn literal(v: &Value) -> String {
    match *v {
        Value::Float(x) => f(x),
        Value::Int(i) => format!("i32({i})"),
        Value::Bool(b) => b.to_string(),
        Value::Vec2([x, y]) => format!("vec2<f32>({}, {})", f(x), f(y)),
        Value::Vec3([x, y, z]) => format!("vec3<f32>({}, {}, {})", f(x), f(y), f(z)),
        Value::Vec4(a) | Value::Color(a) | Value::Quat(a) => {
            format!("vec4<f32>({}, {}, {}, {})", f(a[0]), f(a[1]), f(a[2]), f(a[3]))
        }
    }
}

/// Swizzle taking `ty`'s components out of a vec4.
fn swizzle(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Float => ".x",
        ValueType::Vec2 => ".xy",
        ValueType::Vec3 => ".xyz",
        _ => "",
    }
}

/// Generated shader text plus what the host needs to know about it.
pub struct EmitterShader {
    pub source: String,
}

struct Gen<'a> {
    sys: &'a CookedSystem,
    out: String,
    /// Helper functions (curves), emitted before the stacks.
    helpers: String,
    curves: usize,
}

impl Gen<'_> {
    fn param(&self, r: ParamRef) -> String {
        match r.ns {
            Namespace::User => {
                let ty = self.sys.user_params[r.index as usize].value.ty();
                let v = format!("fr.user[{}]", r.index);
                match ty {
                    ValueType::Int => format!("bitcast<i32>({v}.x)"),
                    ValueType::Bool => format!("({v}.x != 0.0)"),
                    t => format!("{v}{}", swizzle(t)),
                }
            }
            Namespace::Engine => if r.index == 0 { "fr.dt" } else { "fr.time" }.into(),
            Namespace::System => "fr.time".into(),
            Namespace::Emitter => "fr.emitter_age".into(),
            Namespace::Particles => match r.index {
                attr::POSITION => "(*p).position",
                attr::VELOCITY => "(*p).velocity",
                attr::COLOR => "(*p).color",
                attr::SPRITE_SIZE => "(*p).sprite_size",
                attr::SPRITE_ROTATION => "(*p).sprite_rotation",
                attr::MASS => "(*p).mass",
                attr::LIFETIME => "(*p).lifetime",
                attr::AGE => "(*p).age",
                attr::NORMALIZED_AGE => "(*p).normalized_age",
                attr::UNIQUE_ID => "bitcast<i32>((*p).id)",
                attr::INITIAL_COLOR => "(*p).initial_color",
                attr::INITIAL_SPRITE_SIZE => "(*p).initial_sprite_size",
                attr::PHYSICS_FORCE => "(*p).force",
                attr::PHYSICS_DRAG => "(*p).drag",
                _ => "0.0",
            }
            .into(),
        }
    }

    fn curve(&mut self, c: &CookedCurve) -> String {
        let name = format!("curve_{}", self.curves);
        self.curves += 1;
        let ty = wgsl_type(c.ty);
        let sw = swizzle(c.ty);
        let n = c.times.len();
        let v = |i: usize| literal(&Value::from_array(c.ty, c.values[i]));
        let h = &mut self.helpers;
        writeln!(h, "fn {name}(t: f32) -> {ty} {{").unwrap();
        if n == 0 {
            writeln!(h, "    return {}(vec4<f32>(0.0){sw});", ty).unwrap();
        } else {
            writeln!(h, "    if (t <= {}) {{ return {}; }}", f(c.times[0]), v(0)).unwrap();
            writeln!(h, "    if (t >= {}) {{ return {}; }}", f(c.times[n - 1]), v(n - 1)).unwrap();
            for i in 1..n {
                let (t0, t1) = (c.times[i - 1], c.times[i]);
                let (a, b) = (c.values[i - 1], c.values[i]);
                let d: [f32; 4] = std::array::from_fn(|k| b[k] - a[k]);
                let fr = if t1 > t0 {
                    format!("(t - {}) / ({})", f(t0), f(t1 - t0))
                } else {
                    "0.0".into()
                };
                writeln!(
                    h,
                    "    if (t < {}) {{ return {} + {} * ({fr}); }}",
                    f(t1),
                    v(i - 1),
                    literal(&Value::from_array(c.ty, d))
                )
                .unwrap();
            }
            writeln!(h, "    return {};", v(n - 1)).unwrap();
        }
        writeln!(h, "}}").unwrap();
        name
    }

    /// WGSL expression for input `slot` of a module, using random key `k`.
    fn input(&mut self, input: &CookedInput, slot: usize) -> String {
        match input {
            CookedInput::Const(v) => literal(v),
            CookedInput::Link(r) => self.param(*r),
            CookedInput::Random(a, b) => {
                let base = slot as u32 * 8;
                match (a, b) {
                    (Value::Int(lo), Value::Int(hi)) => {
                        let (lo, hi) = (*lo.min(hi), *lo.max(hi));
                        let span = (hi as i64 - lo as i64 + 1) as f32;
                        format!(
                            "min(i32({lo}) + i32(floor(rand_unit(k, {base}u) * {})), i32({hi}))",
                            f(span)
                        )
                    }
                    _ => {
                        let (lo, hi) = (a.to_array(), b.to_array());
                        let n = a.ty().components();
                        let comps: Vec<String> = (0..n)
                            .map(|c| {
                                format!(
                                    "({} + {} * rand_unit(k, {}u))",
                                    f(lo[c]),
                                    f(hi[c] - lo[c]),
                                    base + c as u32
                                )
                            })
                            .collect();
                        if n == 1 {
                            comps[0].clone()
                        } else {
                            format!("{}({})", wgsl_type(a.ty()), comps.join(", "))
                        }
                    }
                }
            }
            CookedInput::Curve(c) => {
                let name = self.curve(c);
                format!("{name}((*p).normalized_age)")
            }
        }
    }

    fn module(&mut self, m: &CookedModule, stage: u32, index: usize, frame_expr: &str) {
        let Some(kind) = modules::from_id(m.module) else {
            return;
        };
        let name = modules::by_kind(kind).name;
        let o = &mut String::new();
        writeln!(o, "    // {name}").unwrap();
        writeln!(o, "    {{").unwrap();
        writeln!(
            o,
            "    let k = Key((*p).id, {frame_expr}, {}u);",
            stage * 256 + index as u32
        )
        .unwrap();
        let inp = |slot: usize, g: &mut Self| g.input(&m.inputs[slot], slot);
        match kind {
            ModuleKind::InitializeParticle => {
                let fields = [
                    "lifetime",
                    "position",
                    "color",
                    "sprite_size",
                    "sprite_rotation",
                    "mass",
                ];
                for (slot, field) in fields.iter().enumerate() {
                    let e = inp(slot, self);
                    writeln!(o, "    (*p).{field} = {e};").unwrap();
                }
                writeln!(o, "    (*p).initial_color = (*p).color;").unwrap();
                writeln!(o, "    (*p).initial_sprite_size = (*p).sprite_size;").unwrap();
            }
            ModuleKind::AddVelocity => {
                let e = inp(0, self);
                writeln!(o, "    (*p).velocity += {e};").unwrap();
            }
            ModuleKind::AddVelocityInCone => {
                let (a, ang, s) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    var axis = {a};").unwrap();
                writeln!(o, "    let rl = 1.0 / length(axis);").unwrap();
                writeln!(
                    o,
                    "    if (rl > 0.0 && rl <= 3.4028235e38) {{ axis = axis * rl; }} else {{ axis = vec3<f32>(0.0, 1.0, 0.0); }}"
                )
                .unwrap();
                writeln!(
                    o,
                    "    let angle = clamp({ang}, 0.0, 180.0) * {};",
                    f(1.0f32.to_radians())
                )
                .unwrap();
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let u1 = rand_unit(k, 128u);").unwrap();
                writeln!(o, "    let u2 = rand_unit(k, 129u);").unwrap();
                writeln!(o, "    let cos_t = 1.0 - u1 * (1.0 - cos(angle));").unwrap();
                writeln!(o, "    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));").unwrap();
                writeln!(o, "    let phi = {} * u2;", f(std::f32::consts::TAU)).unwrap();
                writeln!(
                    o,
                    "    let dir = basis_t1(axis) * (sin_t * cos(phi)) + basis_t2(axis) * (sin_t * sin(phi)) + axis * cos_t;"
                )
                .unwrap();
                writeln!(o, "    (*p).velocity += dir * strength;").unwrap();
            }
            ModuleKind::GravityForce => {
                let e = inp(0, self);
                writeln!(o, "    (*p).force += {e} * (*p).mass;").unwrap();
            }
            ModuleKind::Drag => {
                let e = inp(0, self);
                writeln!(o, "    (*p).drag += {e};").unwrap();
            }
            ModuleKind::CurlNoiseForce => {
                let (s, fq, off) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let freq = {fq};").unwrap();
                writeln!(o, "    let offset = {off};").unwrap();
                writeln!(
                    o,
                    "    (*p).force += curl_noise((*p).position * freq + offset) * strength;"
                )
                .unwrap();
            }
            ModuleKind::ScaleColor => {
                let (rgb, a) = (inp(0, self), inp(1, self));
                writeln!(o, "    let rgb = {rgb};").unwrap();
                writeln!(o, "    let a = {a};").unwrap();
                writeln!(o, "    (*p).color = (*p).initial_color * vec4<f32>(rgb, a);").unwrap();
            }
            ModuleKind::ScaleSpriteSize => {
                let e = inp(0, self);
                writeln!(o, "    (*p).sprite_size = (*p).initial_sprite_size * {e};").unwrap();
            }
            ModuleKind::SolveForcesAndVelocity => {
                o.push_str(
                    "    if ((*p).mass > 0.0) { (*p).velocity += (*p).force / (*p).mass * fr.dt; }
    (*p).velocity *= exp(-(*p).drag * fr.dt);
    (*p).position += (*p).velocity * fr.dt;
    (*p).force = vec3<f32>(0.0);
    (*p).drag = 0.0;
",
                );
            }
            ModuleKind::ParticleState => {
                let e = inp(0, self);
                o.push_str(
                    "    (*p).age += fr.dt;
    (*p).normalized_age = select(1.0, (*p).age / (*p).lifetime, (*p).lifetime > 0.0);
",
                );
                writeln!(o, "    if ({e} && (*p).age > (*p).lifetime) {{ (*p).alive = false; }}")
                    .unwrap();
            }
            ModuleKind::KillParticles => {
                let e = inp(0, self);
                writeln!(o, "    if ({e}) {{ (*p).alive = false; }}").unwrap();
            }
            ModuleKind::SpawnRate | ModuleKind::SpawnBurstInstantaneous => {}
        }
        writeln!(o, "    }}").unwrap();
        self.out.push_str(o);
    }
}

/// WGSL for one emitter: the shared library plus its two stacks.
pub fn emitter_shader(sys: &CookedSystem, emitter: &CookedEmitter) -> EmitterShader {
    let mut g = Gen {
        sys,
        out: String::new(),
        helpers: String::new(),
        curves: 0,
    };
    g.out
        .push_str("fn spawn_particle(p: ptr<function, Particle>) {\n");
    for (i, m) in emitter.particle_spawn.iter().enumerate() {
        g.module(m, STAGE_PARTICLE_SPAWN, i, "0u");
    }
    g.out.push_str("}\n\nfn update_particle(p: ptr<function, Particle>) {\n");
    for (i, m) in emitter.particle_update.iter().enumerate() {
        g.module(m, STAGE_PARTICLE_UPDATE, i, "fr.frame + 1u");
    }
    g.out.push_str("}\n");
    EmitterShader {
        source: format!(
            "// Emitter \"{}\" of \"{}\" (generated by fx_gpu).\n{COMMON}\n// ---- curves ----\n{}\n// ---- module stacks ----\n{}",
            emitter.name, sys.name, g.helpers, g.out
        ),
    }
}
