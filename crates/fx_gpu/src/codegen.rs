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
            format!(
                "vec4<f32>({}, {}, {}, {})",
                f(a[0]),
                f(a[1]),
                f(a[2]),
                f(a[3])
            )
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
    /// Whether the emitter stores the extended attributes (words 28..48,
    /// see [`CookedEmitter::uses_extended_attributes`]).
    pub extended: bool,
}

struct Gen<'a> {
    sys: &'a CookedSystem,
    out: String,
    /// Helper functions (curves), emitted before the stacks.
    helpers: String,
    curves: usize,
    /// Dynamic input helpers emitted so far.
    dyns: usize,
    /// Flipbook frames of the emitter's sprite renderer (SubUV Animation).
    sub_frames: u32,
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
            Namespace::Engine => match r.index {
                0 => "fr.dt",
                1 => "fr.time",
                _ => "fr.owner_velocity",
            }
            .into(),
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
                attr::SCALE => "(*p).scale",
                attr::MESH_ORIENTATION => "(*p).mesh_orientation",
                attr::RIBBON_ID => "(*p).ribbon_id",
                attr::RIBBON_WIDTH => "(*p).ribbon_width",
                attr::RIBBON_TWIST => "(*p).ribbon_twist",
                attr::RIBBON_LINK_ORDER => "(*p).ribbon_link_order",
                attr::LIGHT_RADIUS => "(*p).light_radius",
                attr::DYNAMIC_MATERIAL_PARAMETER => "(*p).dynamic_material_parameter",
                attr::SUB_IMAGE_INDEX => "(*p).sub_image_index",
                attr::INITIAL_SCALE => "(*p).initial_scale",
                attr::INITIAL_RIBBON_WIDTH => "(*p).initial_ribbon_width",
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
            writeln!(
                h,
                "    if (t >= {}) {{ return {}; }}",
                f(c.times[n - 1]),
                v(n - 1)
            )
            .unwrap();
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

    /// WGSL expression for `input` of module `m`, random draws from `salt`.
    fn input(&mut self, m: &CookedModule, input: &CookedInput, salt: u32) -> String {
        match input {
            CookedInput::Const(v) => literal(v),
            CookedInput::Link(r) => self.param(*r),
            CookedInput::Random(a, b) => match (a, b) {
                (Value::Int(lo), Value::Int(hi)) => {
                    let (lo, hi) = (*lo.min(hi), *lo.max(hi));
                    let span = (hi as i64 - lo as i64 + 1) as f32;
                    format!(
                        "min(i32({lo}) + i32(floor(rand_unit(k, {salt}u) * {})), i32({hi}))",
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
                                salt + c as u32
                            )
                        })
                        .collect();
                    if n == 1 {
                        comps[0].clone()
                    } else {
                        format!("{}({})", wgsl_type(a.ty()), comps.join(", "))
                    }
                }
            },
            CookedInput::Curve(c) => {
                let name = self.curve(c);
                format!("{name}((*p).normalized_age)")
            }
            CookedInput::Dynamic(i) => self.dynamic(m, *i as usize),
        }
    }

    /// Emits a helper function for dynamic input `index` of `m` and returns its call.
    fn dynamic(&mut self, m: &CookedModule, index: usize) -> String {
        use fx_core::dynamic::{self, Op};
        let node = &m.dynamics[index];
        let def = dynamic::from_id(node.def).expect("cooked dynamic input id");
        let ty = def.ty;
        let wt = wgsl_type(ty);
        let salt = dynamic::node_salt(index);
        let args: Vec<String> = node
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| match (def.op, a) {
                // The curve itself is not a value: sampled below.
                (Op::FromCurve, CookedInput::Curve(_)) if i == 0 => String::new(),
                _ => self.input(m, a, dynamic::arg_salt(index, i)),
            })
            .collect();
        let name = format!("dyn_{}", self.dyns);
        self.dyns += 1;
        let mut b = String::new();
        let n = ty.components();
        let sw = ["x", "y", "z", "w"];
        match def.op {
            Op::UniformRanged => {
                writeln!(b, "    let lo = {};\n    let hi = {};", args[0], args[1]).unwrap();
                if ty == ValueType::Int {
                    writeln!(
                        b,
                        "    let a = min(lo, hi);\n    let z = max(lo, hi);\n    let span = f32(z - a) + 1.0;"
                    )
                    .unwrap();
                                        writeln!(
                        b,
                        "    return min(a + i32(floor(rand_unit(k, {salt}u) * span)), z);"
                    )
                    .unwrap();
                } else {
                    let comps: Vec<String> = (0..n)
                        .map(|c| {
                            let s = if n == 1 { String::new() } else { format!(".{}", sw[c]) };
                            format!("lo{s} + (hi{s} - lo{s}) * rand_unit(k, {}u)", salt + c as u32)
                        })
                        .collect();
                    if n == 1 {
                        writeln!(b, "    return {};", comps[0]).unwrap();
                    } else {
                        writeln!(b, "    return {wt}({});", comps.join(", ")).unwrap();
                    }
                }
            }
            Op::RandomByUniform => {
                writeln!(
                    b,
                    "    let a = {};\n    let u = rand_unit(k, {salt}u);\n    return a + ({} - a) * u;",
                    args[0], args[1]
                )
                .unwrap();
            }
            Op::FromCurve => {
                let CookedInput::Curve(c) = &node.args[0] else {
                    unreachable!("cooked \"from Curve\" without a curve")
                };
                let cn = self.curve(c);
                writeln!(b, "    return {cn}({});", args[1]).unwrap();
            }
            Op::Add => writeln!(b, "    return {} + {};", args[0], args[1]).unwrap(),
            Op::Subtract => writeln!(b, "    return {} - {};", args[0], args[1]).unwrap(),
            Op::Multiply | Op::MultiplyByFloat => {
                writeln!(b, "    return {} * {};", args[0], args[1]).unwrap()
            }
            Op::Divide => writeln!(
                b,
                "    let a = {};\n    let d = {};\n    if (d == 0.0) {{ return 0.0; }}\n    return a / d;",
                args[0], args[1]
            )
            .unwrap(),
            Op::Lerp => writeln!(
                b,
                "    let a = {};\n    return a + ({} - a) * {};",
                args[0], args[1], args[2]
            )
            .unwrap(),
            Op::Clamp => writeln!(
                b,
                "    return min(max({}, {}), {});",
                args[0], args[1], args[2]
            )
            .unwrap(),
            Op::MakeFromFloats => {
                writeln!(b, "    return {wt}({});", args[..n].join(", ")).unwrap()
            }
            Op::MakeColorFromVec3 => {
                writeln!(b, "    return vec4<f32>({}, {});", args[0], args[1]).unwrap()
            }
            Op::Break => {
                let c = match &node.args[1] {
                    CookedInput::Const(Value::Int(i)) => (*i).clamp(0, 3) as usize,
                    _ => 0,
                };
                writeln!(b, "    return ({}).{};", args[0], sw[c]).unwrap();
            }
            Op::Normalize => writeln!(
                b,
                "    let v = {};\n    let rl = 1.0 / length(v);\n    if (rl > 0.0 && rl <= 3.4028235e38) {{ return v * rl; }}\n    return vec3<f32>(0.0);",
                args[0]
            )
            .unwrap(),
            Op::Length => writeln!(b, "    return length({});", args[0]).unwrap(),
            Op::Distance => writeln!(b, "    return length({} - {});", args[0], args[1]).unwrap(),
            Op::Sine | Op::Cosine => {
                let func = if def.op == Op::Sine { "sin" } else { "cos" };
                writeln!(
                    b,
                    "    let x = {} * ({} * {} + {});\n    return {} + {} * {func}(x);",
                    f(std::f32::consts::TAU),
                    args[1],
                    args[4],
                    args[2],
                    args[3],
                    args[0]
                )
                .unwrap();
            }
            Op::SelectByBool => writeln!(
                b,
                "    if ({}) {{ return {}; }}\n    return {};",
                args[0], args[1], args[2]
            )
            .unwrap(),
            Op::SelectByInt => {
                let count = args.len() - 1;
                writeln!(b, "    let i = clamp({}, 0, {});", args[0], count as i32 - 1).unwrap();
                for (j, a) in args[1..].iter().enumerate().take(count - 1) {
                    writeln!(b, "    if (i == {j}) {{ return {a}; }}").unwrap();
                }
                writeln!(b, "    return {};", args[count]).unwrap();
            }
            Op::RandomFromList => {
                let count = args.len();
                writeln!(
                    b,
                    "    let i = min(u32(rand_unit(k, {salt}u) * {}), {}u);",
                    f(count as f32),
                    count - 1
                )
                .unwrap();
                for (j, a) in args.iter().enumerate().take(count - 1) {
                    writeln!(b, "    if (i == {j}u) {{ return {a}; }}").unwrap();
                }
                writeln!(b, "    return {};", args[count - 1]).unwrap();
            }
            Op::RandomUnitVector => writeln!(
                b,
                "    let z = 1.0 - 2.0 * rand_unit(k, {salt}u);\n    let r = sqrt(max(1.0 - z * z, 0.0));\n    let phi = {} * rand_unit(k, {}u);\n    return vec3<f32>(r * cos(phi), r * sin(phi), z);",
                f(std::f32::consts::TAU),
                salt + 1
            )
            .unwrap(),
            Op::NormalizedAge => writeln!(
                b,
                "    let age = {};\n    let d = {};\n    if (d > 0.0) {{ return clamp(age / d, 0.0, 1.0); }}\n    return 1.0;",
                args[0], args[1]
            )
            .unwrap(),
            Op::Compare => {
                let op = match &node.args[2] {
                    CookedInput::Const(Value::Int(i)) => *i,
                    _ => 0,
                };
                let o = ["<", "<=", ">", ">=", "==", "!="][op.clamp(0, 5) as usize];
                writeln!(b, "    return {} {o} {};", args[0], args[1]).unwrap();
            }
        }
        writeln!(
            self.helpers,
            "// {}\nfn {name}(p: ptr<function, Particle>, k: Key) -> {wt} {{\n{b}}}",
            def.name
        )
        .unwrap();
        format!("{name}(p, k)")
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
        let inp = |slot: usize, g: &mut Self| g.input(m, &m.inputs[slot], slot as u32 * 8);
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
                let (s, fq, off, pan) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self));
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let freq = {fq};").unwrap();
                writeln!(o, "    let offset = {off};").unwrap();
                writeln!(o, "    let pan = {pan};").unwrap();
                writeln!(
                    o,
                    "    (*p).force += curl_noise((*p).position * freq + offset + pan * fr.time) * strength;"
                )
                .unwrap();
            }
            ModuleKind::ScaleColor => {
                let (rgb, a) = (inp(0, self), inp(1, self));
                writeln!(o, "    let rgb = {rgb};").unwrap();
                writeln!(o, "    let a = {a};").unwrap();
                writeln!(
                    o,
                    "    (*p).color = (*p).initial_color * vec4<f32>(rgb, a);"
                )
                .unwrap();
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
                writeln!(
                    o,
                    "    if ({e} && (*p).age > (*p).lifetime) {{ (*p).alive = false; }}"
                )
                .unwrap();
            }
            ModuleKind::KillParticles => {
                let e = inp(0, self);
                writeln!(o, "    if ({e}) {{ (*p).alive = false; }}").unwrap();
            }
            ModuleKind::InitializeRibbon => {
                let fields = [
                    "ribbon_width",
                    "ribbon_twist",
                    "ribbon_id",
                    "ribbon_link_order",
                ];
                for (slot, field) in fields.iter().enumerate() {
                    let e = inp(slot, self);
                    writeln!(o, "    (*p).{field} = {e};").unwrap();
                }
                writeln!(o, "    (*p).initial_ribbon_width = (*p).ribbon_width;").unwrap();
            }
            ModuleKind::InitializeMesh => {
                let (s, r) = (inp(0, self), inp(1, self));
                writeln!(o, "    (*p).scale = {s};").unwrap();
                writeln!(o, "    (*p).initial_scale = (*p).scale;").unwrap();
                writeln!(o, "    (*p).mesh_orientation = euler_quat({r});").unwrap();
            }
            ModuleKind::UpdateMeshOrientation => {
                let e = inp(0, self);
                writeln!(o, "    let rate = {e};").unwrap();
                writeln!(o, "    let dq = euler_quat(rate * fr.dt);").unwrap();
                writeln!(
                    o,
                    "    (*p).mesh_orientation = quat_normalize(quat_mul((*p).mesh_orientation, dq));"
                )
                .unwrap();
            }
            ModuleKind::InitializeLight => {
                let e = inp(0, self);
                writeln!(o, "    (*p).light_radius = {e};").unwrap();
            }
            ModuleKind::DynamicMaterialParameters => {
                let e = inp(0, self);
                writeln!(o, "    (*p).dynamic_material_parameter = {e};").unwrap();
            }
            ModuleKind::PointAttractionForce => {
                let (c, s, r, e, kill) = (
                    inp(0, self),
                    inp(1, self),
                    inp(2, self),
                    inp(3, self),
                    inp(4, self),
                );
                writeln!(o, "    let center = {c};").unwrap();
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let radius = {r};").unwrap();
                writeln!(o, "    let exponent = {e};").unwrap();
                writeln!(o, "    let kill = {kill};").unwrap();
                o.push_str(
                    "    let d = center - (*p).position;
    let dist = length(d);
    if (dist < kill) { (*p).alive = false; }
    if (dist > 0.0 && dist < radius) { (*p).force += d * (strength * falloff(dist, radius, exponent) / dist); }
",
                );
            }
            ModuleKind::VortexVelocity => {
                let (a, c, s) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let axis = unit_or_y({a});").unwrap();
                writeln!(o, "    let origin = {c};").unwrap();
                writeln!(o, "    let amount = {s};").unwrap();
                writeln!(o, "    let dirs = vortex_dirs(axis, (*p).position - origin);").unwrap();
                writeln!(o, "    (*p).velocity += dirs.tangent * amount;").unwrap();
            }
            ModuleKind::VortexForce => {
                let (a, c, s, pull) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self));
                writeln!(o, "    let axis = unit_or_y({a});").unwrap();
                writeln!(o, "    let origin = {c};").unwrap();
                writeln!(o, "    let amount = {s};").unwrap();
                writeln!(o, "    let pull = {pull};").unwrap();
                writeln!(o, "    let dirs = vortex_dirs(axis, (*p).position - origin);").unwrap();
                writeln!(o, "    (*p).force += dirs.tangent * amount + dirs.inward * pull;").unwrap();
            }
            ModuleKind::WindForce => {
                let (w, s, r) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let wind = {w};").unwrap();
                writeln!(o, "    let speed = {s};").unwrap();
                writeln!(o, "    let resistance = {r};").unwrap();
                writeln!(o, "    (*p).force += (wind * speed - (*p).velocity) * resistance;").unwrap();
            }
            ModuleKind::LinearForce => {
                let e = inp(0, self);
                writeln!(o, "    (*p).force += {e};").unwrap();
            }
            ModuleKind::LimitForce => {
                let e = inp(0, self);
                writeln!(o, "    let limit = max({e}, 0.0);").unwrap();
                writeln!(o, "    let len = length((*p).force);").unwrap();
                writeln!(o, "    if (len > limit) {{ (*p).force *= limit / len; }}").unwrap();
            }
            ModuleKind::AccelerationForce => {
                let e = inp(0, self);
                writeln!(o, "    (*p).force += {e} * (*p).mass;").unwrap();
            }
            ModuleKind::PointForce => {
                let (c, s, r, e) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self));
                writeln!(o, "    let center = {c};").unwrap();
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let radius = {r};").unwrap();
                writeln!(o, "    let exponent = {e};").unwrap();
                o.push_str(
                    "    let d = (*p).position - center;
    let dist = length(d);
    if (dist > 0.0 && dist < radius) { (*p).force += d * (strength * falloff(dist, radius, exponent) / dist); }
",
                );
            }
            ModuleKind::InheritVelocity => {
                let e = inp(0, self);
                writeln!(o, "    (*p).velocity += fr.owner_velocity * {e};").unwrap();
            }
            ModuleKind::ShapeLocation => {
                // "Shape Primitive" is a static switch: only its shape is generated.
                let primitive = match m.inputs[0] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let surface = inp(1, self);
                let offset = inp(2, self);
                writeln!(o, "    let surface = {surface};").unwrap();
                writeln!(o, "    let offset = {offset};").unwrap();
                writeln!(
                    o,
                    "    let u = array<f32, 5>(rand_unit(k, 128u), rand_unit(k, 129u), rand_unit(k, 130u), rand_unit(k, 131u), rand_unit(k, 132u));"
                )
                .unwrap();
                let call = match primitive {
                    1 => format!("shape_box({}, surface, u)", inp(4, self)),
                    2 => format!("shape_cylinder({}, {}, surface, u)", inp(5, self), inp(6, self)),
                    3 => format!("shape_torus({}, {}, surface, u)", inp(7, self), inp(8, self)),
                    4 => format!("shape_cone({}, {}, surface, u)", inp(9, self), inp(10, self)),
                    5 => format!("shape_ring({}, {}, u)", inp(11, self), inp(12, self)),
                    _ => format!("shape_sphere({}, surface, u)", inp(3, self)),
                };
                writeln!(o, "    (*p).position = offset + {call};").unwrap();
            }
            ModuleKind::InitialMeshOrientation => {
                let e = inp(0, self);
                writeln!(o, "    (*p).mesh_orientation = euler_quat({e});").unwrap();
            }
            ModuleKind::SpriteRotationRate => {
                let e = inp(0, self);
                writeln!(o, "    (*p).sprite_rotation += {e} * fr.dt;").unwrap();
            }
            ModuleKind::SubUvAnimation => {
                let mode = match m.inputs[0] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let (n, start, fps) = (inp(1, self), inp(2, self), inp(3, self));
                writeln!(o, "    let n = {n};").unwrap();
                writeln!(
                    o,
                    "    let frames = select({}, f32(n), n > 0);",
                    f(self.sub_frames as f32)
                )
                .unwrap();
                writeln!(o, "    let start = f32({start});").unwrap();
                match mode {
                    1 => {
                        // One draw per particle: the key ignores the step.
                        writeln!(
                            o,
                            "    let fi = floor(rand_unit(Key(k.particle, 0u, k.module_id), 128u) * frames);"
                        )
                        .unwrap();
                    }
                    2 => {
                        writeln!(o, "    let x = (*p).age * {fps};").unwrap();
                        writeln!(o, "    let fi = x - frames * floor(x / frames);").unwrap();
                    }
                    _ => {
                        writeln!(
                            o,
                            "    let fi = clamp((*p).normalized_age, 0.0, SUB_UV_LAST) * frames;"
                        )
                        .unwrap();
                    }
                }
                writeln!(o, "    (*p).sub_image_index = start + fi;").unwrap();
            }
            ModuleKind::ScaleMeshSize => {
                let e = inp(0, self);
                writeln!(o, "    (*p).scale = (*p).initial_scale * {e};").unwrap();
            }
            ModuleKind::ScaleRibbonWidth => {
                let e = inp(0, self);
                writeln!(o, "    (*p).ribbon_width = (*p).initial_ribbon_width * {e};").unwrap();
            }
            ModuleKind::Color => {
                let e = inp(0, self);
                writeln!(o, "    (*p).color = {e};").unwrap();
            }
            ModuleKind::ScaleColorBySpeed => {
                let (t, lo, hi) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let t = speed_fraction((*p).velocity, {t});").unwrap();
                writeln!(o, "    let lo = {lo};").unwrap();
                writeln!(o, "    let hi = {hi};").unwrap();
                writeln!(o, "    (*p).color = (*p).initial_color * (lo + (hi - lo) * t);").unwrap();
            }
            ModuleKind::ScaleSpriteSizeBySpeed => {
                let (t, lo, hi) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let t = speed_fraction((*p).velocity, {t});").unwrap();
                writeln!(o, "    let lo = {lo};").unwrap();
                writeln!(o, "    let hi = {hi};").unwrap();
                writeln!(
                    o,
                    "    (*p).sprite_size = (*p).initial_sprite_size * (lo + (hi - lo) * t);"
                )
                .unwrap();
            }
            ModuleKind::KillParticlesInVolume => {
                let shape = match m.inputs[0] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let (c, r, size, invert) = (inp(1, self), inp(2, self), inp(3, self), inp(4, self));
                writeln!(o, "    let d = (*p).position - {c};").unwrap();
                if shape == 1 {
                    writeln!(o, "    let h = {size} * 0.5;").unwrap();
                    writeln!(o, "    let inside = all(abs(d) < h);").unwrap();
                } else {
                    writeln!(o, "    let r = {r};").unwrap();
                    writeln!(o, "    let inside = dot(d, d) < r * r;").unwrap();
                }
                writeln!(o, "    if (inside != {invert}) {{ (*p).alive = false; }}").unwrap();
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
        dyns: 0,
        sub_frames: emitter.sub_image_frames(),
    };
    g.out
        .push_str("fn spawn_particle(p: ptr<function, Particle>) {\n");
    for (i, m) in emitter.particle_spawn.iter().enumerate() {
        g.module(m, STAGE_PARTICLE_SPAWN, i, "0u");
    }
    g.out
        .push_str("}\n\nfn update_particle(p: ptr<function, Particle>) {\n");
    for (i, m) in emitter.particle_update.iter().enumerate() {
        g.module(m, STAGE_PARTICLE_UPDATE, i, "fr.frame + 1u");
    }
    g.out.push_str("}\n");
    let extended = emitter.uses_extended_attributes();
    EmitterShader {
        extended,
        source: format!(
            "// Emitter \"{}\" of \"{}\" (generated by fx_gpu).\n// Words 28..48 of the particle layout are stored when true.\nconst EXT: bool = {extended};\n{COMMON}\n// ---- curves and dynamic inputs ----\n{}\n// ---- module stacks ----\n{}",
            emitter.name, sys.name, g.helpers, g.out
        ),
    }
}
