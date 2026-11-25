//! A small model of effect source (`.fx.ron`) values and its writer, used
//! by the converter.

use std::fmt::Write;

/// A typed literal.
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Quat([f32; 4]),
}

/// Our value types, by name (`fx_core::types::ValueType::name`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Quat,
}

impl Ty {
    pub fn name(self) -> &'static str {
        match self {
            Ty::Float => "Float",
            Ty::Int => "Int",
            Ty::Bool => "Bool",
            Ty::Vec2 => "Vec2",
            Ty::Vec3 => "Vec3",
            Ty::Vec4 => "Vec4",
            Ty::Color => "Color",
            Ty::Quat => "Quat",
        }
    }
    pub fn from_name(n: &str) -> Option<Ty> {
        [Ty::Float, Ty::Int, Ty::Bool, Ty::Vec2, Ty::Vec3, Ty::Vec4, Ty::Color, Ty::Quat]
            .into_iter()
            .find(|t| t.name() == n)
    }
    pub fn components(self) -> usize {
        match self {
            Ty::Float | Ty::Int | Ty::Bool => 1,
            Ty::Vec2 => 2,
            Ty::Vec3 => 3,
            Ty::Vec4 | Ty::Color | Ty::Quat => 4,
        }
    }
    /// The type's curve declaration word for user parameters.
    pub fn curve_decl(self) -> &'static str {
        match self {
            Ty::Vec2 => "Vec2Curve",
            Ty::Vec3 => "VectorCurve",
            Ty::Vec4 => "Vec4Curve",
            Ty::Color => "ColorCurve",
            _ => "FloatCurve",
        }
    }
}

impl Lit {
    pub fn ty(&self) -> Ty {
        match self {
            Lit::Float(_) => Ty::Float,
            Lit::Int(_) => Ty::Int,
            Lit::Bool(_) => Ty::Bool,
            Lit::Vec2(_) => Ty::Vec2,
            Lit::Vec3(_) => Ty::Vec3,
            Lit::Vec4(_) => Ty::Vec4,
            Lit::Color(_) => Ty::Color,
            Lit::Quat(_) => Ty::Quat,
        }
    }
    pub fn floats(&self) -> Vec<f32> {
        match self {
            Lit::Float(v) => vec![*v],
            Lit::Int(v) => vec![*v as f32],
            Lit::Bool(v) => vec![if *v { 1.0 } else { 0.0 }],
            Lit::Vec2(v) => v.to_vec(),
            Lit::Vec3(v) => v.to_vec(),
            Lit::Vec4(v) | Lit::Color(v) | Lit::Quat(v) => v.to_vec(),
        }
    }
    /// A literal of type `ty` from components (padded with the last one).
    pub fn from_floats(ty: Ty, f: &[f32]) -> Lit {
        let g = |k: usize| f.get(k).or(f.last()).copied().unwrap_or(0.0);
        match ty {
            Ty::Float => Lit::Float(g(0)),
            Ty::Int => Lit::Int(g(0).round() as i32),
            Ty::Bool => Lit::Bool(g(0) != 0.0),
            Ty::Vec2 => Lit::Vec2([g(0), g(1)]),
            Ty::Vec3 => Lit::Vec3([g(0), g(1), g(2)]),
            Ty::Vec4 => Lit::Vec4([g(0), g(1), g(2), g(3)]),
            Ty::Color => Lit::Color([g(0), g(1), g(2), if f.len() >= 4 { f[3] } else { 1.0 }]),
            Ty::Quat => Lit::Quat([g(0), g(1), g(2), g(3)]),
        }
    }
}

/// A curve: key times and one value per component.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub ty: Ty,
    pub keys: Vec<(f32, Vec<f32>)>,
}

/// An input value in effect source.
#[derive(Clone, Debug, PartialEq)]
pub enum Fx {
    Lit(Lit),
    /// A static switch value, written bare.
    Choice(String),
    /// `Link(Type, "Name")`; the type is a value type or a data interface /
    /// curve link type (`FloatCurve`).
    Link(String, String),
    Range(Lit, Lit),
    CurveOverAge(Curve),
    /// A bare curve, as a dynamic input's curve argument.
    Curve(Curve),
    Dyn(String, Vec<(String, Fx)>),
    /// A list (dynamic input list arguments).
    List(Vec<Fx>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FxModule {
    pub name: String,
    pub enabled: bool,
    pub inputs: Vec<(String, Fx)>,
    /// A comment written above it (the Niagara module it came from).
    pub comment: Option<String>,
}

impl FxModule {
    pub fn new(name: &str) -> Self {
        FxModule {
            name: name.to_string(),
            enabled: true,
            inputs: Vec::new(),
            comment: None,
        }
    }
    pub fn set(&mut self, name: &str, v: Fx) {
        if let Some(e) = self.inputs.iter_mut().find(|(n, _)| n == name) {
            e.1 = v;
        } else {
            self.inputs.push((name.to_string(), v));
        }
    }
    pub fn get(&self, name: &str) -> Option<&Fx> {
        self.inputs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// Formats a float so the effect parser reads it back (always with a
/// decimal point, no exponent).
pub fn f(x: f32) -> String {
    let x = if x.is_finite() { x } else { 0.0 };
    let r = (x as f64 * 1e6).round() / 1e6;
    let mut s = format!("{r}");
    if s == "-0" {
        s = "0".into();
    }
    if !s.contains('.') {
        s.push_str(".0");
    }
    s
}

fn tuple(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| f(*x)).collect();
    format!("({})", parts.join(", "))
}

pub fn write_lit(l: &Lit) -> String {
    match l {
        Lit::Float(v) => format!("Float({})", f(*v)),
        Lit::Int(v) => format!("Int({v})"),
        Lit::Bool(v) => format!("Bool({v})"),
        Lit::Vec2(v) => format!("Vec2({})", tuple(v)),
        Lit::Vec3(v) => format!("Vec3({})", tuple(v)),
        Lit::Vec4(v) => format!("Vec4({})", tuple(v)),
        Lit::Color(v) => format!("Color({})", tuple(v)),
        Lit::Quat(v) => format!("Quat({})", tuple(v)),
    }
}

pub fn write_curve(c: &Curve) -> String {
    let keys: Vec<String> = c
        .keys
        .iter()
        .map(|(t, v)| {
            if v.len() == 1 {
                format!("({}, {})", f(*t), f(v[0]))
            } else {
                format!("({}, {})", f(*t), tuple(v))
            }
        })
        .collect();
    format!("{}([{}])", c.ty.name(), keys.join(", "))
}

pub fn write_fx(v: &Fx) -> String {
    match v {
        Fx::Lit(l) => write_lit(l),
        Fx::Choice(c) => c.clone(),
        Fx::Link(t, n) => format!("Link({t}, {})", quote(n)),
        Fx::Range(a, b) => format!("RandomRange({}, {})", write_lit(a), write_lit(b)),
        Fx::CurveOverAge(c) => format!("CurveOverAge({})", write_curve(c)),
        Fx::Curve(c) => write_curve(c),
        Fx::Dyn(n, args) if args.is_empty() => format!("Dynamic({})", quote(n)),
        Fx::Dyn(n, args) => format!("Dynamic({}, {{ {} }})", quote(n), write_inputs(args)),
        Fx::List(items) => format!("[{}]", items.iter().map(write_fx).collect::<Vec<_>>().join(", ")),
    }
}

pub fn write_inputs(args: &[(String, Fx)]) -> String {
    args.iter()
        .map(|(k, v)| format!("{}: {}", quote(k), write_fx(v)))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn quote(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Writes a stack: `name: [ (module: ..., inputs: { ... }), ... ],`.
pub fn write_stack(o: &mut String, ind: &str, name: &str, mods: &[FxModule]) {
    if mods.is_empty() {
        return;
    }
    let _ = writeln!(o, "{ind}{name}: [");
    write_modules(o, &format!("{ind}    "), mods);
    let _ = writeln!(o, "{ind}],");
}

pub fn write_modules(o: &mut String, ind: &str, mods: &[FxModule]) {
    for m in mods {
        if let Some(c) = &m.comment {
            for line in c.lines() {
                let _ = writeln!(o, "{ind}// {line}");
            }
        }
        let mut s = format!("(module: {}", quote(&m.name));
        if !m.enabled {
            s.push_str(", enabled: false");
        }
        if !m.inputs.is_empty() {
            let _ = write!(s, ", inputs: {{ {} }}", write_inputs(&m.inputs));
        }
        s.push_str("),");
        let _ = writeln!(o, "{ind}{s}");
    }
}
