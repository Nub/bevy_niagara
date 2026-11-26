//! Compiles an emitter's particle module stacks to WGSL.
//!
//! The generated code mirrors `fx_core::sim::run_particle_module` statement
//! for statement. Constants are baked in as exact bit patterns; `User.`
//! parameters and engine values come from the per-step [`crate::FrameUniform`],
//! so changing a user parameter never recompiles.

use std::fmt::Write;

use fx_core::cook::{CookedCurve, CookedInput, CookedModule, CookedSystem};
use fx_core::events::EventKind;
use fx_core::sim::{COLLISION_PLANE, COLLISION_RAY_TRACED};

use crate::events::EventLayout;
use fx_core::modules::{self, ModuleKind};
use fx_core::params::{Namespace, ParamRef, attr};
use fx_core::types::{Value, ValueType};

const COMMON: &str = include_str!("wgsl/sim_common.wgsl");
const SCENE: &str = include_str!("wgsl/scene.wgsl");
const SCENE_RT: &str = include_str!("wgsl/scene_rt.wgsl");
const DATA: &str = include_str!("wgsl/data.wgsl");

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
    /// RayTraced collisions may use the ray-tracing scene (group 3).
    ray_tracing: bool,
    /// A module reads the scene bindings (group 2).
    uses_scene: bool,
    /// A module traces rays (group 3).
    uses_tlas: bool,
    /// A module reads mesh or texture data interfaces (group 1).
    uses_data: bool,
    /// The emitter's grids.
    grids: &'a [fx_core::custom::CookedGrid],
    /// Custom module wrappers emitted: (custom module, grid context).
    wrappers: Vec<(usize, Option<usize>)>,
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
            Namespace::Emitter => match r.index {
                0 => "fr.emitter_age",
                1 => "fr.looped_age",
                2 => "bitcast<i32>(fr.loop_count)",
                _ => "fr.normalized_loop_age",
            }
            .into(),
            Namespace::Event => match r.index {
                0 => "cur_event.position",
                1 => "cur_event.velocity",
                2 => "cur_event.normal",
                3 => "cur_event.age",
                _ => "bitcast<i32>(cur_event.id)",
            }
            .into(),
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
                attr::CAMERA_OFFSET => "(*p).camera_offset",
                attr::SPRITE_FACING => "(*p).sprite_facing",
                attr::SPRITE_ALIGNMENT => "(*p).sprite_alignment",
                attr::DYNAMIC_MATERIAL_PARAMETER_1 => "(*p).dynamic_material_parameter_1",
                attr::ROTATIONAL_VELOCITY => "(*p).rotational_velocity",
                attr::PHYSICS_ROTATIONAL_FORCE => "(*p).rotational_force",
                attr::PHYSICS_ROTATIONAL_DRAG => "(*p).rotational_drag",
                attr::DYNAMIC_MATERIAL_PARAMETER_2 => "(*p).dynamic_material_parameter_2",
                attr::DYNAMIC_MATERIAL_PARAMETER_3 => "(*p).dynamic_material_parameter_3",
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
            Op::CameraPosition => {
                self.uses_scene = true;
                writeln!(b, "    return scene_camera_position();").unwrap();
            }
            Op::CameraForward => {
                self.uses_scene = true;
                writeln!(b, "    return scene_camera_forward();").unwrap();
            }
            Op::SceneDepth => {
                self.uses_scene = true;
                writeln!(b, "    return scene_depth_at({});", args[1]).unwrap();
            }
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
            if let Some(c) = fx_core::custom::custom_index(m.module) {
                self.custom_call(m, c, None, stage, index, frame_expr);
            }
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
        let inp = |slot: usize, g: &mut Self| g.input(m, &m.inputs[slot], fx_core::sim::input_salt(slot));
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
                let (h, s, v, a) = (inp(6, self), inp(7, self), inp(8, self), inp(9, self));
                writeln!(o, "    let hue = {h};\n    let sat = {s};\n    let val = {v};").unwrap();
                writeln!(o, "    if (hue != 0.0 || sat != 1.0 || val != 1.0) {{ (*p).color = hsv_shift((*p).color, hue, sat, val); }}").unwrap();
                writeln!(o, "    (*p).color.w *= {a};").unwrap();
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
                let (inner, spread, fall) = (inp(3, self), inp(4, self), inp(5, self));
                writeln!(o, "    let inner = clamp({inner}, 0.0, 180.0) * {};", f(1.0f32.to_radians())).unwrap();
                writeln!(o, "    let spread = clamp({spread}, 0.0, 1.0);").unwrap();
                writeln!(o, "    let falloff = clamp({fall}, 0.0, 1.0);").unwrap();
                writeln!(o, "    let u1 = rand_unit(k, 128u);").unwrap();
                writeln!(o, "    let u2 = rand_unit(k, 129u);").unwrap();
                writeln!(o, "    var outer = angle;").unwrap();
                writeln!(
                    o,
                    "    if (spread > 0.0) {{ let lo = 1.0 - 1.5 * spread; outer = angle * clamp(lo + (1.0 - lo) * rand_unit(k, 130u), 0.0, 1.0); }}"
                )
                .unwrap();
                writeln!(o, "    var cos_t = 1.0 - u1 * (1.0 - cos(outer));").unwrap();
                writeln!(o, "    if (inner > 0.0) {{ cos_t = cos(inner) - u1 * (cos(inner) - cos(outer)); }}").unwrap();
                writeln!(o, "    var speed = strength;").unwrap();
                writeln!(o, "    if (falloff > 0.0) {{ speed = strength * cone_speed_falloff(cos_t, falloff); }}").unwrap();
                writeln!(o, "    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));").unwrap();
                writeln!(o, "    let phi = {} * u2;", f(std::f32::consts::TAU)).unwrap();
                writeln!(
                    o,
                    "    let dir = basis_t1(axis) * (sin_t * cos(phi)) + basis_t2(axis) * (sin_t * sin(phi)) + axis * cos_t;"
                )
                .unwrap();
                writeln!(o, "    (*p).velocity += dir * speed;").unwrap();
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
                let (rgb, a, rgba) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let rgb = {rgb};").unwrap();
                writeln!(o, "    let a = {a};").unwrap();
                writeln!(o, "    let rgba = {rgba};").unwrap();
                writeln!(
                    o,
                    "    (*p).color = (*p).initial_color * vec4<f32>(rgb, a) * rgba;"
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
    if (any((*p).rotational_velocity != vec3<f32>(0.0)) || any((*p).rotational_force != vec3<f32>(0.0)) || (*p).rotational_drag != 0.0) {
        if ((*p).mass > 0.0) { (*p).rotational_velocity += (*p).rotational_force / (*p).mass * fr.dt; }
        (*p).rotational_velocity *= exp(-(*p).rotational_drag * fr.dt);
        let spin = rotation_vector_quat((*p).rotational_velocity * fr.dt);
        (*p).mesh_orientation = quat_normalize(quat_mul(spin, (*p).mesh_orientation));
        (*p).rotational_force = vec3<f32>(0.0);
        (*p).rotational_drag = 0.0;
    }
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
                let off = inp(5, self);
                writeln!(o, "    let attracted = (*p).position + {off};").unwrap();
                o.push_str(
                    "    let d = center - attracted;
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
                writeln!(o, "    var wind_v = wind * speed;").unwrap();
                let (ts, tf, toff, tpos) = (inp(3, self), inp(4, self), inp(5, self), inp(6, self));
                writeln!(o, "    let turbulence = {ts};").unwrap();
                writeln!(o, "    if (turbulence != 0.0) {{").unwrap();
                writeln!(o, "        let freq = {tf};").unwrap();
                writeln!(o, "        let at = {tpos} + {toff} * fr.time;").unwrap();
                writeln!(o, "        wind_v += curl_noise(at * freq) * (length(wind_v) * turbulence);").unwrap();
                writeln!(o, "    }}").unwrap();
                writeln!(o, "    (*p).force += (wind_v - (*p).velocity) * resistance;").unwrap();
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
                    1 => format!("shape_box_mid({}, {}, surface, u)", inp(4, self), inp(13, self)),
                    2 => format!("shape_cylinder({}, {}, surface, u)", inp(5, self), inp(6, self)),
                    3 => format!("shape_torus({}, {}, surface, u)", inp(7, self), inp(8, self)),
                    4 => format!(
                        "shape_cone_ex({}, {}, {}, {}, {}, surface, u)",
                        inp(9, self),
                        inp(10, self),
                        inp(14, self),
                        inp(15, self),
                        inp(16, self)
                    ),
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
            ModuleKind::Collision => {
                let int = |slot: usize| match m.inputs[slot] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let (ty, response) = (int(0), int(1));
                let (normal, origin, radius, rest, fric, rs, thick) = (
                    inp(2, self),
                    inp(3, self),
                    inp(4, self),
                    inp(5, self),
                    inp(6, self),
                    inp(7, self),
                    inp(8, self),
                );
                writeln!(o, "    let radius = {radius};").unwrap();
                if ty == COLLISION_PLANE {
                    writeln!(o, "    let n = unit_or_y({normal});").unwrap();
                    writeln!(o, "    let o = {origin};").unwrap();
                    writeln!(o, "    let hit = true;").unwrap();
                } else {
                    self.uses_scene = true;
                    if ty == COLLISION_RAY_TRACED && self.ray_tracing {
                        self.uses_tlas = true;
                        writeln!(
                            o,
                            "    let c = scene_ray_contact((*p).position - (*p).velocity * fr.dt, (*p).position, radius, {thick});"
                        )
                        .unwrap();
                    } else {
                        writeln!(o, "    let c = scene_depth_contact((*p).position, radius, {thick});").unwrap();
                    }
                    writeln!(o, "    let hit = c.hit;\n    let n = c.n;\n    let o = c.o;").unwrap();
                }
                writeln!(
                    o,
                    "    if (hit && collide_plane(p, n, o, radius, i32({response}), {rest}, {fric}, {rs})) {{ ev_collided = true; ev_normal = n; }}"
                )
                .unwrap();
            }
            ModuleKind::GenerateDeathEvent => {
                writeln!(o, "    ev_death = true;").unwrap();
            }
            ModuleKind::GenerateCollisionEvent => {
                writeln!(
                    o,
                    "    if (ev_collided) {{ ev_has[1] = true; ev_pending[1] = make_event(*p, ev_normal); }}"
                )
                .unwrap();
            }
            ModuleKind::GenerateLocationEvent => {
                let prob = inp(0, self);
                writeln!(
                    o,
                    "    if (rand_unit(k, 128u) < {prob}) {{ ev_has[2] = true; ev_pending[2] = make_event(*p, vec3<f32>(0.0)); }}"
                )
                .unwrap();
            }
            ModuleKind::ReceiveDeathEvent
            | ModuleKind::ReceiveCollisionEvent
            | ModuleKind::ReceiveLocationEvent => {
                let (off, s) = (inp(0, self), inp(1, self));
                writeln!(o, "    (*p).position = cur_event.position + {off};").unwrap();
                writeln!(o, "    (*p).velocity += cur_event.velocity * {s};").unwrap();
            }
            ModuleKind::SkeletalMeshLocation | ModuleKind::StaticMeshLocation => {
                self.uses_data = true;
                let int = |slot: usize| match m.inputs[slot] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let skeletal = kind == ModuleKind::SkeletalMeshLocation;
                let di = int(0).max(0);
                let (mode, offset, inherit) = if skeletal {
                    (int(1), inp(2, self), inp(3, self))
                } else {
                    (fx_core::data::SAMPLE_SURFACE, inp(1, self), inp(2, self))
                };
                writeln!(o, "    let offset = {offset};\n    let inherit = {inherit};").unwrap();
                if mode == fx_core::data::SAMPLE_BONES {
                    writeln!(o, "    let s = mesh_sample_bone({di}u, rand_unit(k, 128u));").unwrap();
                } else {
                    writeln!(
                        o,
                        "    let s = mesh_sample_surface({di}u, vec3<f32>(rand_unit(k, 128u), rand_unit(k, 129u), rand_unit(k, 130u)));"
                    )
                    .unwrap();
                }
                writeln!(
                    o,
                    "    if (s.ok) {{\n        (*p).position = s.position + s.normal * offset;\n        (*p).velocity += s.velocity * inherit;\n    }}"
                )
                .unwrap();
            }
            ModuleKind::SampleTexture => {
                self.uses_data = true;
                let di = match m.inputs[0] {
                    CookedInput::Const(Value::Int(i)) => i.max(0),
                    _ => 0,
                };
                let blend = match m.inputs[3] {
                    CookedInput::Const(Value::Int(i)) => i,
                    _ => 0,
                };
                let (uv, scale) = (inp(1, self), inp(2, self));
                writeln!(o, "    let c = texture_sample({di}u, {uv}) * {scale};").unwrap();
                if blend == 1 {
                    writeln!(o, "    (*p).color *= c;").unwrap();
                } else {
                    writeln!(o, "    (*p).color = c;").unwrap();
                }
            }
            ModuleKind::CameraOffset => {
                let amount = inp(0, self);
                let additive = matches!(m.inputs[1], CookedInput::Const(Value::Int(1)));
                let op = if additive { "+=" } else { "=" };
                writeln!(o, "    (*p).camera_offset {op} {amount};").unwrap();
            }
            ModuleKind::SetLifetime => {
                let e = inp(0, self);
                writeln!(o, "    (*p).lifetime = {e};").unwrap();
            }
            ModuleKind::SetMass => {
                let e = inp(0, self);
                writeln!(o, "    (*p).mass = {e};").unwrap();
            }
            ModuleKind::SetSpriteSize => {
                let (e, init) = (inp(0, self), inp(1, self));
                writeln!(o, "    (*p).sprite_size = {e};").unwrap();
                writeln!(o, "    if ({init}) {{ (*p).initial_sprite_size = (*p).sprite_size; }}").unwrap();
            }
            ModuleKind::SetSpriteRotation => {
                let e = inp(0, self);
                writeln!(o, "    (*p).sprite_rotation = {e};").unwrap();
            }
            ModuleKind::SetVelocity => {
                let e = inp(0, self);
                writeln!(o, "    (*p).velocity = {e};").unwrap();
            }
            ModuleKind::SetPosition => {
                let e = inp(0, self);
                writeln!(o, "    (*p).position = {e};").unwrap();
            }
            ModuleKind::SetScale => {
                let (e, init) = (inp(0, self), inp(1, self));
                writeln!(o, "    (*p).scale = {e};").unwrap();
                writeln!(o, "    if ({init}) {{ (*p).initial_scale = (*p).scale; }}").unwrap();
            }
            ModuleKind::AddVelocityFromPoint => {
                let (s, c, sc) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let origin = {c};").unwrap();
                writeln!(o, "    let scale = {sc};").unwrap();
                writeln!(o, "    (*p).velocity += unit_or_y((*p).position - origin) * strength * scale;").unwrap();
            }
            ModuleKind::GridLocation => {
                let (size, x, y, z, off) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self), inp(4, self));
                writeln!(o, "    let size = {size};").unwrap();
                writeln!(o, "    let n = vec3<u32>(u32(clamp({x}, 1, 1024)), u32(clamp({y}, 1, 1024)), u32(clamp({z}, 1, 1024)));").unwrap();
                writeln!(o, "    let offset = {off};").unwrap();
                writeln!(o, "    (*p).position = offset + size * grid_point((*p).id, n);").unwrap();
            }
            ModuleKind::AddRotationalVelocity => {
                let e = inp(0, self);
                writeln!(o, "    (*p).rotational_velocity += {e};").unwrap();
            }
            ModuleKind::MeshRotationForce => {
                let e = inp(0, self);
                writeln!(o, "    (*p).rotational_force += {e};").unwrap();
            }
            ModuleKind::RotationalDrag => {
                let e = inp(0, self);
                writeln!(o, "    (*p).rotational_drag += {e};").unwrap();
            }
            ModuleKind::OrientMeshToVector => {
                let (v, a) = (inp(0, self), inp(1, self));
                writeln!(o, "    let v = {v};").unwrap();
                writeln!(o, "    let axis = {a};").unwrap();
                writeln!(o, "    if (length(v) > 0.0) {{ (*p).mesh_orientation = shortest_arc(axis, v); }}").unwrap();
            }
            ModuleKind::AlignSpriteToMeshOrientation => {
                let (f, a) = (inp(0, self), inp(1, self));
                writeln!(o, "    let q = (*p).mesh_orientation;").unwrap();
                writeln!(o, "    (*p).sprite_facing = quat_rotate(q, {f});").unwrap();
                writeln!(o, "    (*p).sprite_alignment = quat_rotate(q, {a});").unwrap();
            }
            ModuleKind::SetSpriteFacing => {
                let e = inp(0, self);
                writeln!(o, "    (*p).sprite_facing = {e};").unwrap();
            }
            ModuleKind::SetSpriteAlignment => {
                let e = inp(0, self);
                writeln!(o, "    (*p).sprite_alignment = {e};").unwrap();
            }
            ModuleKind::ScaleMeshSizeBySpeed => {
                let (t, lo, hi) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let t = speed_fraction((*p).velocity, {t});").unwrap();
                writeln!(o, "    let lo = {lo};").unwrap();
                writeln!(o, "    let hi = {hi};").unwrap();
                writeln!(o, "    (*p).scale = (*p).initial_scale * (lo + (hi - lo) * t);").unwrap();
            }
            ModuleKind::ScaleColorByDistance | ModuleKind::ScaleSpriteSizeByDistance => {
                let (r, n, fa, lo, hi) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self), inp(4, self));
                writeln!(o, "    let d = length((*p).position - {r});").unwrap();
                writeln!(o, "    let near = {n};").unwrap();
                writeln!(o, "    let far = {fa};").unwrap();
                writeln!(o, "    let t = distance_fraction(d, near, far);").unwrap();
                writeln!(o, "    let lo = {lo};").unwrap();
                writeln!(o, "    let hi = {hi};").unwrap();
                if kind == ModuleKind::ScaleColorByDistance {
                    writeln!(o, "    (*p).color = (*p).initial_color * (lo + (hi - lo) * t);").unwrap();
                } else {
                    writeln!(o, "    (*p).sprite_size = (*p).initial_sprite_size * (lo + (hi - lo) * t);").unwrap();
                }
            }
            ModuleKind::DynamicMaterialParameters2 => {
                let e = inp(0, self);
                writeln!(o, "    (*p).dynamic_material_parameter_1 = {e};").unwrap();
            }
            ModuleKind::DynamicMaterialParameters3 => {
                let e = inp(0, self);
                writeln!(o, "    (*p).dynamic_material_parameter_2 = {e};").unwrap();
            }
            ModuleKind::DynamicMaterialParameters4 => {
                let e = inp(0, self);
                writeln!(o, "    (*p).dynamic_material_parameter_3 = {e};").unwrap();
            }
            ModuleKind::JitterPosition => {
                let (a, dl) = (inp(0, self), inp(1, self));
                writeln!(o, "    let amount = {a};").unwrap();
                writeln!(o, "    let delay = {dl};").unwrap();
                writeln!(o, "    let fire = !(delay > 0.0) || floor((*p).age / delay) != floor(((*p).age - fr.dt) / delay);").unwrap();
                writeln!(o, "    if (fire) {{").unwrap();
                writeln!(o, "        let u = vec3<f32>(rand_unit(k, 128u), rand_unit(k, 129u), rand_unit(k, 130u));").unwrap();
                writeln!(o, "        (*p).position += (u * 2.0 - vec3<f32>(1.0)) * amount;").unwrap();
                writeln!(o, "    }}").unwrap();
            }
            ModuleKind::KillParticlesBySpeed => {
                let (lo, hi) = (inp(0, self), inp(1, self));
                writeln!(o, "    let speed = length((*p).velocity);").unwrap();
                writeln!(o, "    if (speed < {lo} || speed > {hi}) {{ (*p).alive = false; }}").unwrap();
            }
            ModuleKind::KillParticlesByDistance => {
                let (c, r, closer) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let d = length((*p).position - {c});").unwrap();
                writeln!(o, "    let r = {r};").unwrap();
                writeln!(o, "    if (select(d > r, d < r, {closer})) {{ (*p).alive = false; }}").unwrap();
            }
            ModuleKind::RotateAroundPoint => {
                let (c, a, r, rate, ph) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self), inp(4, self));
                writeln!(o, "    let center = {c};").unwrap();
                writeln!(o, "    let axis = unit_or_y({a});").unwrap();
                writeln!(o, "    let radius = {r};").unwrap();
                writeln!(o, "    let rate = {rate};").unwrap();
                writeln!(o, "    let phase = {ph};").unwrap();
                writeln!(o, "    let theta = (phase + rate * (*p).age) * DEGREES;").unwrap();
                writeln!(o, "    (*p).position = center + (basis_t1(axis) * cos(theta) + basis_t2(axis) * sin(theta)) * radius;").unwrap();
            }
            ModuleKind::NoiseForce => {
                let (s, fq, oc, pan) = (inp(0, self), inp(1, self), inp(2, self), inp(3, self));
                writeln!(o, "    let strength = {s};").unwrap();
                writeln!(o, "    let freq = {fq};").unwrap();
                writeln!(o, "    let octaves = u32(clamp({oc}, 1, 8));").unwrap();
                writeln!(o, "    let pan = {pan};").unwrap();
                writeln!(o, "    (*p).force += noise_force((*p).position * freq + pan * fr.time, octaves) * strength;").unwrap();
            }
            ModuleKind::LimitVelocity => {
                let e = inp(0, self);
                writeln!(o, "    let limit = max({e}, 0.0);").unwrap();
                writeln!(o, "    let len = length((*p).velocity);").unwrap();
                writeln!(o, "    if (len > limit) {{ (*p).velocity *= limit / len; }}").unwrap();
            }
            ModuleKind::ScaleVelocity => {
                let e = inp(0, self);
                writeln!(o, "    (*p).velocity *= {e};").unwrap();
            }
            ModuleKind::RibbonTwistRate => {
                let e = inp(0, self);
                writeln!(o, "    (*p).ribbon_twist += {e} * fr.dt;").unwrap();
            }
            ModuleKind::ScaleRibbonWidthBySpeed => {
                let (t, lo, hi) = (inp(0, self), inp(1, self), inp(2, self));
                writeln!(o, "    let t = speed_fraction((*p).velocity, {t});").unwrap();
                writeln!(o, "    let lo = {lo};").unwrap();
                writeln!(o, "    let hi = {hi};").unwrap();
                writeln!(o, "    (*p).ribbon_width = (*p).initial_ribbon_width * (lo + (hi - lo) * t);").unwrap();
            }
            ModuleKind::SpawnRate
            | ModuleKind::SpawnBurstInstantaneous
            | ModuleKind::EmitterState
            | ModuleKind::SpawnPerUnit
            | ModuleKind::SpawnPerFrame
            | ModuleKind::SpawnBurstInterval => {}
        }
        writeln!(o, "    }}").unwrap();
        self.out.push_str(o);
    }
}

/// What the host offers the generated code.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShaderOptions {
    /// RayTraced collisions trace rays in the ray-tracing scene (bind group 3,
    /// Solari's scene layout) instead of falling back to the depth buffer.
    pub ray_tracing: bool,
}

/// Generated shader text plus what the host needs to know about it.
pub struct EmitterShader {
    pub source: String,
    /// Whether the emitter stores the extended attributes (words 28..48,
    /// see [`fx_core::cook::CookedEmitter::uses_extended_attributes`]).
    pub extended: bool,
    /// Reads the scene bindings (bind group 2: depth buffer and view).
    pub uses_scene: bool,
    /// Traces rays (bind group 3: the ray-tracing scene).
    pub uses_tlas: bool,
    /// Reads mesh or texture data interfaces (bind group 1, see `wgsl/data.wgsl`).
    pub uses_data: bool,
    /// Writes events: has an `event_rank` pass.
    pub writes_events: bool,
    /// Simulation stage kernels, in order.
    pub stages: Vec<StageKernel>,
}

/// WGSL for emitter `ei`: the shared library, its stacks and handlers, and
/// the event glue for `events`.
pub fn emitter_shader(
    sys: &CookedSystem,
    ei: usize,
    events: &EventLayout,
    options: ShaderOptions,
) -> EmitterShader {
    let emitter = &sys.emitters[ei];
    let mut g = Gen {
        sys,
        out: String::new(),
        helpers: String::new(),
        curves: 0,
        dyns: 0,
        sub_frames: emitter.sub_image_frames(),
        ray_tracing: options.ray_tracing,
        uses_scene: false,
        uses_tlas: false,
        uses_data: false,
        grids: &emitter.grids,
        wrappers: Vec::new(),
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
    // Interpolated spawning (fx_core::sim::spawn_step_dt): the k-th of the n
    // regular spawns runs Particle Update over (n - k) / n of the step; the
    // events it writes are dropped, as on the CPU.
    g.out.push_str("\nfn spawn_update(k: u32, p: ptr<function, Particle>) {\n");
    if emitter.interpolated_spawning {
        g.out.push_str(
            "    let n = atomicLoad(&state[S_REGULAR_N]);
    if (k >= n || !(*p).alive) {
        return;
    }
    let full = fr.dt;
    fr.dt = full * (f32(n - k) / f32(n));
    ev_reset();
    update_particle(p);
    ev_reset();
    fr.dt = full;
",
        );
    }
    g.out.push_str("}\n");
    // World space (fx_core::sim::spawn_fraction): after Particle Spawn, the
    // owner pose at the particle's birth moves it to world space.
    g.out.push_str("\nfn spawn_to_world(k: u32, p: ptr<function, Particle>) {\n");
    if !emitter.local_space {
        writeln!(
            g.out,
            "    let n = atomicLoad(&state[S_REGULAR_N]);
    var f = 1.0;
    if ({} && k < n) {{
        f = f32(k) / f32(n);
    }}
    owner_to_world(f, p);",
            emitter.interpolated_spawning
        )
        .unwrap();
    }
    g.out.push_str("}\n");
    for (h, handler) in emitter.event_handlers.iter().enumerate() {
        writeln!(g.out, "\nfn handler_{h}(p: ptr<function, Particle>) {{").unwrap();
        for (i, m) in handler.modules.iter().enumerate() {
            g.module(m, fx_core::events::handler_stage(h), i, "ev_frame");
        }
        g.out.push_str("}\n");
    }
    let (stages_src, stages) = stage_kernels(&mut g, ei);
    let glue = event_glue(sys, ei, events);
    let extended = emitter.uses_extended_attributes();
    let extended_2 = emitter.uses_material_parameters_2_3();
    let mut source = String::new();
    if g.uses_tlas {
        source.push_str("enable wgpu_ray_query;\n\n");
    }
    write!(
        source,
        "// Emitter \"{}\" of \"{}\" (generated by fx_gpu).\n// Words 28..66 of the particle layout are stored when EXT, 67..74 when EXT2.\nconst EXT: bool = {extended};\nconst EXT2: bool = {extended_2};\n{}\n{COMMON}\n",
        emitter.name, sys.name, fx_core::custom::LIB
    )
    .unwrap();
    source.push_str(&fx_core::custom::grid_helpers(&emitter.grids, &fx_core::custom::grid_bases(&emitter.grids)));
    if g.uses_scene {
        source.push_str(SCENE);
    }
    if g.uses_tlas {
        source.push_str(SCENE_RT);
    }
    if g.uses_data {
        source.push_str(DATA);
    }
    write!(
        source,
        "\n// ---- curves and dynamic inputs ----\n{}\n// ---- module stacks ----\n{}\n// ---- events ----\n{glue}\n// ---- simulation stages ----\n{stages_src}",
        g.helpers, g.out
    )
    .unwrap();
    // The instance's part of the batch's state and event buffers (see `Frame` in sim_common.wgsl).
    let source = source
        .replace("&state[", "&state[fr.state_base + ")
        .replace("&event_counts[", "&event_counts[fr.ev_count_base + ")
        .replace("event_data[", "event_data[fr.ev_data_base + ");
    EmitterShader {
        source,
        extended,
        uses_scene: g.uses_scene,
        uses_tlas: g.uses_tlas,
        uses_data: g.uses_data,
        writes_events: EventKind::ALL
            .iter()
            .any(|k| events.region(ei, *k).is_some()),
        stages,
    }
}

/// The event functions the kernels call (see `sim_common.wgsl`), for emitter `ei`.
fn event_glue(sys: &CookedSystem, ei: usize, events: &EventLayout) -> String {
    let emitter = &sys.emitters[ei];
    let own: Vec<_> = EventKind::ALL
        .iter()
        .filter_map(|k| events.region(ei, *k))
        .collect();
    let mut s = String::new();
    s.push_str("fn reset_events() {\n");
    for r in &own {
        writeln!(s, "    atomicStore(&event_counts[{}u], 0u);", r.index).unwrap();
    }
    s.push_str("}\n\nfn max_event_count() -> u32 {\n    var m = 0u;\n");
    for r in &own {
        writeln!(s, "    m = max(m, event_count({}u, {}u));", r.index, r.capacity).unwrap();
    }
    s.push_str("    return m;\n}\n\nfn rank_events(i: u32, lid: u32) {\n");
    for r in &own {
        writeln!(
            s,
            "    rank_region(i, lid, {}u, {}u, {}u, {}u);",
            r.index, r.data, r.sorted, r.capacity
        )
        .unwrap();
    }
    s.push_str("}\n\nfn write_events(p: Particle) {\n");
    for r in &own {
        let args = format!("{}u, {}u, {}u", r.index, r.data, r.capacity);
        match r.kind {
            EventKind::Death => writeln!(
                s,
                "    if (ev_death && !p.alive) {{ write_event({args}, make_event(p, vec3<f32>(0.0))); }}"
            ),
            k => writeln!(
                s,
                "    if (ev_has[{0}]) {{ write_event({args}, ev_pending[{0}]); }}",
                k.index()
            ),
        }
        .unwrap();
    }
    // Where handler h reads: (region, data, sorted, capacity, max events).
    let src = |h: &fx_core::cook::CookedEventHandler| {
        let r = events
            .region(h.source as usize, h.kind())
            .expect("cook checks that handler sources write the event");
        (r.index, r.data, r.sorted, r.capacity, h.max_events)
    };
    s.push_str("}\n\nfn every_handlers(p: ptr<function, Particle>) {\n");
    for (h, handler) in emitter.event_handlers.iter().enumerate() {
        if !handler.every_particle {
            continue;
        }
        let (r, data, sorted, cap, max) = src(handler);
        writeln!(
            s,
            "    {{\n        let n = events_taken({r}u, {cap}u, {max}u);\n        for (var e = 0u; e < n; e++) {{\n            cur_event = load_event({data}u, {sorted}u, e);\n            ev_frame = fr.frame + 1u + e * {}u;\n            handler_{h}(p);\n        }}\n    }}",
            0x0100_0193u32
        )
        .unwrap();
    }
    s.push_str("}\n\nfn spawn_counts(room0: u32) -> u32 {\n    var room = room0;\n    let n0 = min(fr.spawn_request, room);\n    room -= n0;\n    var total = n0;\n    atomicStore(&state[S_REGULAR_N], n0);\n");
    let mut spawning = Vec::new();
    for (h, handler) in emitter.event_handlers.iter().enumerate() {
        if handler.every_particle {
            continue;
        }
        let (r, _, _, cap, max) = src(handler);
        writeln!(
            s,
            "    {{\n        let c = handler_spawns(events_taken({r}u, {cap}u, {max}u), {}u, room);\n        room -= c;\n        total += c;\n        atomicStore(&state[S_HANDLER_END + {h}u], total);\n    }}",
            handler.spawn_number
        )
        .unwrap();
        spawning.push(h);
    }
    s.push_str("    return total;\n}\n\nfn spawn_event(k: u32, p: ptr<function, Particle>) {\n    ev_frame = 0u;\n");
    let mut start = "atomicLoad(&state[S_REGULAR_N])".to_string();
    for h in spawning {
        let handler = &emitter.event_handlers[h];
        let (_, data, sorted, _, _) = src(handler);
        writeln!(
            s,
            "    {{\n        let start = {start};\n        let end = atomicLoad(&state[S_HANDLER_END + {h}u]);\n        if (k >= start && k < end) {{\n            cur_event = load_event({data}u, {sorted}u, (k - start) / {}u);\n            handler_{h}(p);\n            return;\n        }}\n    }}",
            handler.spawn_number.max(1)
        )
        .unwrap();
        start = format!("atomicLoad(&state[S_HANDLER_END + {h}u])");
    }
    s.push_str("}\n");
    s
}

/// Random-key stage number of simulation stage `s` (after the event handlers').
pub fn sim_stage_key(s: usize) -> u32 {
    32 + s as u32
}

/// Sets every grid's current copy (`gh_<g>`) for code running after `done[g]`
/// writes of grid g this step; `extra` adds the iteration uniform to one grid.
fn grid_halves(grids: &[fx_core::custom::CookedGrid], per_step: &[u32], done: &[u32], extra: Option<usize>) -> String {
    let mut s = String::new();
    for g in 0..grids.len() {
        let it = if extra == Some(g) { " + it.j" } else { "" };
        writeln!(
            s,
            "    gh_{g} = (fr.frame * {}u + {}u{it}) & 1u;",
            per_step[g], done[g]
        )
        .unwrap();
    }
    s
}

impl Gen<'_> {
    /// A call of custom module `c`; `grid` is the iterated grid of a grid stage
    /// (whose attributes are the kernel's `ga_<i>`), else it runs on `*p`.
    fn custom_call(&mut self, m: &CookedModule, c: usize, grid: Option<usize>, stage: u32, index: usize, frame_expr: &str) {
        use fx_core::custom::{self, Arg, Context};
        let def = &self.sys.custom_modules[c];
        let ctx = match grid {
            Some(g) => Context::Grid(&self.grids[g]),
            None => Context::Particles,
        };
        let fname = match grid {
            Some(g) => format!("cm_{c}_g{g}"),
            None => format!("cm_{c}_p"),
        };
        if !self.wrappers.contains(&(c, grid)) {
            self.wrappers.push((c, grid));
            let (w, _) = custom::wrapper(def, &fname, ctx);
            self.helpers.push_str(&w);
        }
        let mut o = String::new();
        writeln!(o, "    // {} (custom)\n    {{", def.name).unwrap();
        writeln!(o, "    let k = Key({}, {frame_expr}, {}u);", if grid.is_some() { "ci" } else { "(*p).id" }, stage * 256 + index as u32).unwrap();
        writeln!(o, "    cm_key = k;").unwrap();
        let expr = |g: &mut Self, a: Arg| -> String {
            match a {
                Arg::Attribute(i) => g.param(ParamRef { ns: Namespace::Particles, index: i }),
                Arg::DeltaTime => "fr.dt".into(),
                Arg::Time => "fr.time".into(),
                Arg::EmitterAge => "fr.emitter_age".into(),
                Arg::Cell => "cell".into(),
                Arg::CellPosition => "cellpos".into(),
                Arg::GridAttribute(i) => format!("ga_{i}"),
                Arg::Input(i) => g.input(m, &m.inputs[i], fx_core::sim::input_salt(i)),
            }
        };
        let args: Vec<String> = custom::args(def, ctx).into_iter().map(|(_, _, a)| expr(self, a)).collect();
        let outs = custom::outputs(def, ctx);
        if outs.is_empty() {
            writeln!(o, "    {fname}({});", args.join(", ")).unwrap();
        } else {
            writeln!(o, "    let o = {fname}({});", args.join(", ")).unwrap();
            for (n, _, a) in outs {
                let target = expr(self, a);
                writeln!(o, "    {target} = o.{n};").unwrap();
            }
        }
        writeln!(o, "    }}").unwrap();
        self.out.push_str(&o);
    }
}

/// A simulation stage kernel of an emitter shader.
#[derive(Clone, Debug)]
pub struct StageKernel {
    /// Entry point.
    pub entry: String,
    pub iterations: u32,
    /// Workgroups of a grid stage; `None` runs over the particles (indirect).
    pub groups: Option<[u32; 3]>,
}

/// The simulation stage kernels of emitter `ei` (appended to its shader).
fn stage_kernels(g: &mut Gen, ei: usize) -> (String, Vec<StageKernel>) {
    let emitter = &g.sys.emitters[ei];
    let grids = &emitter.grids;
    let mut per_step = vec![0u32; grids.len()];
    for s in &emitter.stages {
        if let Some(gi) = s.grid {
            per_step[gi as usize] += s.iterations;
        }
    }
    let mut done = vec![0u32; grids.len()];
    let mut src = String::new();
    let mut kernels = Vec::new();
    // fn grids_at_update(), called by the update and spawn kernels.
    src.push_str("fn grids_at_update() {\n");
    src.push_str(&grid_halves(grids, &per_step, &done, None));
    src.push_str("}\n");
    for (si, stage) in emitter.stages.iter().enumerate() {
        let entry = format!("stage_{si}");
        let key = sim_stage_key(si);
        let saved = std::mem::take(&mut g.out);
        match stage.grid {
            None => {
                writeln!(src, "\n// simulation stage \"{}\" over the particles\n@compute @workgroup_size(WG)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    load_frame(wid.y);\n    let i = gid.x;\n    if (i >= atomicLoad(&state[S_COUNT])) {{\n        return;\n    }}", stage.name).unwrap();
                src.push_str(&grid_halves(grids, &per_step, &done, None));
                src.push_str("    read_dst = true;\n    var particle = load_particle(i);\n    let p = &particle;\n");
                for (mi, m) in stage.modules.iter().enumerate() {
                    g.module(m, key, mi, "fr.frame + 1u");
                }
                src.push_str(&std::mem::replace(&mut g.out, saved));
                src.push_str("    store_particle(i, particle);\n}\n");
                kernels.push(StageKernel { entry, iterations: stage.iterations, groups: None });
            }
            Some(gi) => {
                let gi = gi as usize;
                let gr = &grids[gi];
                let n = &gr.name;
                let wg = if gr.dims == 2 { [8u32, 8, 1] } else { [4, 4, 4] };
                let groups: [u32; 3] = std::array::from_fn(|a| gr.cells[a].div_ceil(wg[a]));
                let (iv, sw) = if gr.dims == 2 { ("vec2<i32>", "xy") } else { ("vec3<i32>", "xyz") };
                writeln!(src, "\n// simulation stage \"{}\" over grid \"{n}\"\n@compute @workgroup_size({}, {}, {})\nfn {entry}(@builtin(global_invocation_id) gid0: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n    // Instances are stacked along z, {} workgroups each.\n    let inst = wid.z / {}u;\n    load_frame(inst);\n    let gid = vec3<u32>(gid0.x, gid0.y, gid0.z - inst * {}u);\n    if (gid.x >= {}u || gid.y >= {}u || gid.z >= {}u) {{\n        return;\n    }}", stage.name, wg[0], wg[1], wg[2], groups[2], groups[2], groups[2] * wg[2], gr.cells[0], gr.cells[1], gr.cells[2]).unwrap();
                src.push_str(&grid_halves(grids, &per_step, &done, Some(gi)));
                writeln!(src, "    let cell = {iv}(gid.{sw});\n    let cellpos = {n}_cell_position(cell);\n    let ci = {n}_index(cell);\n    var particle = new_particle(ci);\n    let p = &particle;").unwrap();
                for (ai, a) in gr.attributes.iter().enumerate() {
                    writeln!(src, "    var ga_{ai} = {n}_{}(cell);", a.name).unwrap();
                }
                for (mi, m) in stage.modules.iter().enumerate() {
                    if let Some(c) = fx_core::custom::custom_index(m.module) {
                        g.custom_call(m, c, Some(gi), key, mi, "fr.frame + 1u");
                    }
                }
                src.push_str(&std::mem::replace(&mut g.out, saved));
                let bases = fx_core::custom::grid_bases(grids);
                let cells = gr.cell_count();
                writeln!(src, "    let w = {}u + (1u - gh_{gi}) * {}u + ci;", bases[gi], gr.half_words()).unwrap();
                for (ai, a) in gr.attributes.iter().enumerate() {
                    let comps = a.ty.components();
                    for c in 0..comps {
                        let v = if comps == 1 { format!("ga_{ai}") } else { format!("ga_{ai}.{}", ["x", "y", "z", "w"][c]) };
                        writeln!(src, "    grid_st(w + {}u, {v});", (a.word + c as u32) * cells).unwrap();
                    }
                }
                src.push_str("}\n");
                done[gi] += stage.iterations;
                kernels.push(StageKernel { entry, iterations: stage.iterations, groups: Some(groups) });
            }
        }
    }
    (src, kernels)
}
