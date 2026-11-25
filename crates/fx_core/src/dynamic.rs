//! Dynamic inputs: Niagara's small functions that compute a module input
//! (Uniform Ranged Float, Multiply Vector, Float from Curve, ...). Each takes
//! named inputs of its own, which may be literals, links, random ranges,
//! curves or further dynamic inputs, nested as deep as needed.
//!
//! Source form: `Dynamic("Multiply Vec3 by Float", { "A": ..., "B": ... })`.
//! The name says the type it gives. Cooked, a dynamic input is its index in
//! [`DYNAMIC_INPUTS`] plus its cooked inputs; `fx_core::sim` evaluates it on
//! the CPU and `fx_gpu` compiles it to WGSL.

use std::sync::LazyLock;

use crate::types::{Value, ValueType};

/// What a dynamic input computes. Every op works over the components of its
/// result type (Float, Vec2, Vec3, Vec4, Color; Int where noted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Minimum + (Maximum - Minimum) * random, one draw per component (Int: whole numbers).
    UniformRanged,
    /// Like [`Op::UniformRanged`] with one draw shared by every component.
    RandomByUniform,
    /// Samples a curve at "Curve Index".
    FromCurve,
    Add,
    Subtract,
    Multiply,
    /// Vector times a Float.
    MultiplyByFloat,
    /// A / B, 0 where B is 0.
    Divide,
    /// A + (B - A) * Alpha.
    Lerp,
    /// max(Value, Minimum) then min(.., Maximum).
    Clamp,
    /// Components from Floats.
    MakeFromFloats,
    /// Color from Vec3 RGB and Float alpha.
    MakeColorFromVec3,
    /// One component of a vector.
    Break,
    /// Unit vector, 0 for a zero vector.
    Normalize,
    Length,
    Distance,
    /// Offset + Amplitude * sin(TAU * (Frequency * Time + Phase)).
    Sine,
    /// As [`Op::Sine`] with cos.
    Cosine,
    /// Condition ? If True : If False.
    SelectByBool,
    /// Values[clamp(Index, 0, n - 1)].
    SelectByInt,
    /// One of Values, chosen uniformly at random.
    RandomFromList,
    /// Uniform on the unit sphere.
    RandomUnitVector,
    /// clamp(Age / Duration, 0, 1); 1 when Duration is not positive.
    NormalizedAge,
    /// A <op> B as a Bool.
    Compare,
    /// The camera position (Camera Query).
    CameraPosition,
    /// The camera view direction (Camera Query).
    CameraForward,
    /// Distance from the camera to the scene surface seen through a point (Scene Depth).
    SceneDepth,
}

/// The kind of value an input of a dynamic input takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgKind {
    /// Any input form giving this type (literal, link, random, curve over age, dynamic input).
    Value(ValueType),
    /// A static switch: a bare name, cooked to its index (an Int).
    Choice(&'static [&'static str]),
    /// A curve literal `Type([(time, value), ...])`.
    Curve(ValueType),
    /// A list `[input, input, ...]` of inputs of this type; always the last input.
    List(ValueType),
    /// A data interface of this type: `Link(CameraQuery, "User.Camera")`.
    DataInterface(crate::data::DiType),
}

/// What an input left out of the source becomes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgDefault {
    /// The input must be given.
    Required,
    Value(Value),
    /// Reads this parameter, e.g. `Particles.NormalizedAge`.
    Link(&'static str),
}

#[derive(Clone, Debug)]
pub struct ArgDef {
    pub name: &'static str,
    pub kind: ArgKind,
    pub default: ArgDefault,
    pub doc: &'static str,
}

#[derive(Clone, Debug)]
pub struct DynamicDef {
    pub name: String,
    pub op: Op,
    /// The type it gives.
    pub ty: ValueType,
    pub args: Vec<ArgDef>,
    pub doc: String,
    /// The Niagara dynamic input it stands for.
    pub niagara: String,
}

impl DynamicDef {
    pub fn arg(&self, name: &str) -> Option<(usize, &ArgDef)> {
        self.args.iter().enumerate().find(|(_, a)| a.name == name)
    }
    /// "\"A\" (Float), ..." for error messages.
    pub fn list_args(&self) -> String {
        if self.args.is_empty() {
            return "(it has no inputs)".into();
        }
        self.args
            .iter()
            .map(|a| format!("\"{}\" ({})", a.name, arg_type(a)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// "Float", "Float curve", "list of Vec3", "one of X | Y".
pub fn arg_type(a: &ArgDef) -> String {
    match a.kind {
        ArgKind::Value(t) => t.name().into(),
        ArgKind::Choice(c) => format!("one of {}", c.join(" | ")),
        ArgKind::Curve(t) => format!("{} curve", t.name()),
        ArgKind::List(t) => format!("list of {}", t.name()),
        ArgKind::DataInterface(t) => format!("{} data interface", t.name()),
    }
}

/// Niagara's name for a type, used in the docs.
fn niagara_type(t: ValueType) -> &'static str {
    match t {
        ValueType::Float => "Float",
        ValueType::Int => "Int",
        ValueType::Bool => "Bool",
        ValueType::Vec2 => "Vector 2D",
        ValueType::Vec3 => "Vector",
        ValueType::Vec4 => "Vector 4",
        ValueType::Color => "Linear Color",
        ValueType::Quat => "Quat",
    }
}

const fn arg(name: &'static str, kind: ArgKind, default: ArgDefault, doc: &'static str) -> ArgDef {
    ArgDef {
        name,
        kind,
        default,
        doc,
    }
}

fn zero(t: ValueType) -> ArgDefault {
    ArgDefault::Value(match t {
        ValueType::Int => Value::Int(0),
        ValueType::Bool => Value::Bool(false),
        t => Value::from_array(t, [0.0; 4]),
    })
}

fn one(t: ValueType) -> ArgDefault {
    ArgDefault::Value(match t {
        ValueType::Int => Value::Int(1),
        t => Value::from_array(t, [1.0; 4]),
    })
}

const COMPARE: &[&str] = &[
    "LessThan",
    "LessOrEqual",
    "GreaterThan",
    "GreaterOrEqual",
    "Equal",
    "NotEqual",
];

/// Component names of a vector type.
pub fn components(t: ValueType) -> &'static [&'static str] {
    match t {
        ValueType::Vec2 => &["X", "Y"],
        ValueType::Vec3 => &["X", "Y", "Z"],
        ValueType::Vec4 => &["X", "Y", "Z", "W"],
        ValueType::Color => &["R", "G", "B", "A"],
        _ => &[],
    }
}

/// The dynamic inputs; the position is the cooked id, so only append.
pub static DYNAMIC_INPUTS: LazyLock<Vec<DynamicDef>> = LazyLock::new(build);

fn build() -> Vec<DynamicDef> {
    use ArgDefault as D;
    use ArgKind as K;
    use ValueType as T;
    const ARITH: [T; 5] = [T::Float, T::Vec2, T::Vec3, T::Vec4, T::Color];
    const VECS: [T; 4] = [T::Vec2, T::Vec3, T::Vec4, T::Color];
    const SELECTABLE: [T; 6] = [T::Float, T::Int, T::Vec2, T::Vec3, T::Vec4, T::Color];
    let mut v = Vec::new();
    let mut add = |name: String, op: Op, ty: T, args: Vec<ArgDef>, doc: String, niagara: String| {
        v.push(DynamicDef {
            name,
            op,
            ty,
            args,
            doc,
            niagara,
        })
    };

    for t in [T::Float, T::Int, T::Vec2, T::Vec3, T::Vec4, T::Color] {
        let n = t.name();
        let per = if t.components() > 1 {
            " Each component is drawn on its own."
        } else {
            ""
        };
        add(
            format!("Uniform Ranged {n}"),
            Op::UniformRanged,
            t,
            vec![
                arg("Minimum", K::Value(t), zero(t), "smallest value"),
                arg("Maximum", K::Value(t), one(t), "largest value"),
            ],
            format!("A random {n} between Minimum and Maximum (inclusive for Int), drawn once per particle in spawn, every step in update.{per}"),
            format!("Uniform Ranged {}", niagara_type(t)),
        );
    }
    for t in VECS {
        let n = t.name();
        add(
            format!("Random Range {n} by Uniform"),
            Op::RandomByUniform,
            t,
            vec![
                arg("Minimum", K::Value(t), zero(t), "value at random 0"),
                arg("Maximum", K::Value(t), one(t), "value at random 1"),
            ],
            format!("A random {n} on the line from Minimum to Maximum: one random number scales every component, so a grey-to-brown range stays on that line."),
            format!("Random Range {} by Uniform", niagara_type(t)),
        );
    }
    for t in ARITH {
        let n = t.name();
        add(
            format!("{n} from Curve"),
            Op::FromCurve,
            t,
            vec![
                arg("Curve", K::Curve(t), D::Required, "the curve, e.g. `Float([(0.0, 1.0), (1.0, 0.0)])`"),
                arg(
                    "Curve Index",
                    K::Value(T::Float),
                    D::Link("Particles.NormalizedAge"),
                    "where to sample, usually 0..1 (normalised age, speed, distance...)",
                ),
            ],
            format!("Samples a {n} curve at Curve Index, any normalised Float: age (the default), speed, distance, emitter age..."),
            format!("{} from Curve", niagara_type(t)),
        );
    }
    for t in ARITH {
        let n = t.name();
        let ab = |what: &'static str| {
            vec![
                arg("A", K::Value(t), zero(t), what),
                arg("B", K::Value(t), zero(t), "second operand"),
            ]
        };
        add(format!("Add {n}"), Op::Add, t, ab("first operand"), format!("A + B, per component."), format!("Add {}s", niagara_type(t)));
        add(format!("Subtract {n}"), Op::Subtract, t, ab("first operand"), format!("A - B, per component."), format!("Subtract {}s", niagara_type(t)));
        let mut m = ab("first operand");
        m[0].default = one(t);
        m[1].default = one(t);
        add(format!("Multiply {n}"), Op::Multiply, t, m, format!("A * B, per component."), format!("Multiply {}", niagara_type(t)));
        add(
            format!("Lerp {n}"),
            Op::Lerp,
            t,
            vec![
                arg("A", K::Value(t), zero(t), "value at Alpha 0"),
                arg("B", K::Value(t), one(t), "value at Alpha 1"),
                arg("Alpha", K::Value(T::Float), D::Value(Value::Float(0.5)), "blend, not clamped"),
            ],
            format!("A + (B - A) * Alpha."),
            format!("Lerp {}s", niagara_type(t)),
        );
    }
    for t in VECS {
        let n = t.name();
        add(
            format!("Multiply {n} by Float"),
            Op::MultiplyByFloat,
            t,
            vec![
                arg("A", K::Value(t), one(t), "the vector"),
                arg("B", K::Value(T::Float), D::Value(Value::Float(1.0)), "the scale"),
            ],
            format!("A * B: every component of A times the Float B (e.g. a direction times a speed)."),
            format!("Multiply {} by Float", niagara_type(t)),
        );
    }
    add(
        "Divide Float".into(),
        Op::Divide,
        T::Float,
        vec![
            arg("A", K::Value(T::Float), D::Value(Value::Float(1.0)), "dividend"),
            arg("B", K::Value(T::Float), D::Value(Value::Float(1.0)), "divisor"),
        ],
        "A / B, or 0 where B is 0 (e.g. speed / max speed for a Curve Index).".into(),
        "Divide Float (safe)".into(),
    );
    add(
        "Clamp Float".into(),
        Op::Clamp,
        T::Float,
        vec![
            arg("Value", K::Value(T::Float), D::Value(Value::Float(0.0)), "value to clamp"),
            arg("Minimum", K::Value(T::Float), D::Value(Value::Float(0.0)), "lower bound"),
            arg("Maximum", K::Value(T::Float), D::Value(Value::Float(1.0)), "upper bound"),
        ],
        "Value limited to Minimum..Maximum.".into(),
        "Clamp Float".into(),
    );
    for t in VECS {
        let n = t.name();
        let names = components(t);
        let args = names
            .iter()
            .map(|c| arg(c, K::Value(T::Float), D::Value(Value::Float(0.0)), "component"))
            .collect();
        add(
            format!("Make {n} from Floats"),
            Op::MakeFromFloats,
            t,
            args,
            format!("A {n} from one Float per component ({}).", names.join(", ")),
            format!("Make {} from Floats", niagara_type(t)),
        );
    }
    add(
        "Make Color from Vec3 and Float".into(),
        Op::MakeColorFromVec3,
        T::Color,
        vec![
            arg("RGB", K::Value(T::Vec3), D::Value(Value::Vec3([1.0; 3])), "red, green, blue"),
            arg("Alpha", K::Value(T::Float), D::Value(Value::Float(1.0)), "alpha"),
        ],
        "A Color from a Vec3 (RGB) and a Float (alpha).".into(),
        "Make Linear Color from Vector and Float".into(),
    );
    for t in VECS {
        let n = t.name();
        add(
            format!("Break {n}"),
            Op::Break,
            T::Float,
            vec![
                arg("Value", K::Value(t), zero(t), "the vector"),
                arg("Component", K::Choice(components(t)), D::Value(Value::Int(0)), "which component"),
            ],
            format!("One component of a {n} ({}).", components(t).join(", ")),
            format!("Break {}", niagara_type(t)),
        );
    }
    add(
        "Normalize Vec3".into(),
        Op::Normalize,
        T::Vec3,
        vec![arg("Vector", K::Value(T::Vec3), D::Value(Value::Vec3([0.0, 1.0, 0.0])), "vector")],
        "The unit vector along Vector, or 0 for a zero vector.".into(),
        "Normalize Vector (safe)".into(),
    );
    add(
        "Vec3 Length".into(),
        Op::Length,
        T::Float,
        vec![arg("Vector", K::Value(T::Vec3), D::Value(Value::Vec3([0.0; 3])), "vector")],
        "Length of Vector (e.g. of `Particles.Velocity`: the speed).".into(),
        "Vector Length".into(),
    );
    add(
        "Vec3 Distance".into(),
        Op::Distance,
        T::Float,
        vec![
            arg("A", K::Value(T::Vec3), D::Link("Particles.Position"), "first point"),
            arg("B", K::Value(T::Vec3), D::Value(Value::Vec3([0.0; 3])), "second point"),
        ],
        "Distance between A (default: the particle's position) and B.".into(),
        "Distance".into(),
    );
    for (op, fname) in [(Op::Sine, "Sine"), (Op::Cosine, "Cosine")] {
        let f = if op == Op::Sine { "sin" } else { "cos" };
        add(
            format!("{fname} over Time"),
            op,
            T::Float,
            vec![
                arg("Amplitude", K::Value(T::Float), D::Value(Value::Float(1.0)), "peak distance from Offset"),
                arg("Frequency", K::Value(T::Float), D::Value(Value::Float(1.0)), "cycles per second"),
                arg("Phase", K::Value(T::Float), D::Value(Value::Float(0.0)), "shift in cycles (0..1)"),
                arg("Offset", K::Value(T::Float), D::Value(Value::Float(0.0)), "centre value"),
                arg("Time", K::Value(T::Float), D::Link("Engine.Time"), "seconds; link `Particles.Age` to follow each particle"),
            ],
            format!("Offset + Amplitude * {f}(2π * (Frequency * Time + Phase))."),
            format!("{fname} (with time)"),
        );
    }
    for t in SELECTABLE {
        let n = t.name();
        add(
            format!("Select {n} by Bool"),
            Op::SelectByBool,
            t,
            vec![
                arg("Condition", K::Value(T::Bool), D::Value(Value::Bool(true)), "which one"),
                arg("If True", K::Value(t), one(t), "given when Condition is true"),
                arg("If False", K::Value(t), zero(t), "given when Condition is false"),
            ],
            format!("If True when Condition holds, else If False."),
            format!("Select {} (by Bool)", niagara_type(t)),
        );
        add(
            format!("Select {n} by Int"),
            Op::SelectByInt,
            t,
            vec![
                arg("Index", K::Value(T::Int), D::Value(Value::Int(0)), "0 picks the first value; clamped to the list"),
                arg("Values", K::List(t), D::Required, "the values, e.g. `[Float(1.0), Float(2.0)]`"),
            ],
            format!("Values[Index], Index clamped to the list."),
            format!("Select {} (by Int)", niagara_type(t)),
        );
        add(
            format!("Random {n} from List"),
            Op::RandomFromList,
            t,
            vec![arg("Values", K::List(t), D::Required, "the values to choose from")],
            format!("One of Values, each equally likely."),
            format!("Random {} from List", niagara_type(t)),
        );
    }
    add(
        "Random Unit Vec3".into(),
        Op::RandomUnitVector,
        T::Vec3,
        vec![],
        "A random direction, uniform over the unit sphere.".into(),
        "Random Unit Vector".into(),
    );
    add(
        "Normalized Emitter Age".into(),
        Op::NormalizedAge,
        T::Float,
        vec![
            arg("Age", K::Value(T::Float), D::Link("Emitter.Age"), "seconds"),
            arg("Duration", K::Value(T::Float), D::Value(Value::Float(1.0)), "seconds that map to 1"),
        ],
        "Emitter.Age / Duration, clamped to 0..1 (1 if Duration is not positive). There is no emitter loop, so this stands in for Emitter.NormalizedLoopAge.".into(),
        "Emitter.NormalizedLoopAge".into(),
    );
    add(
        "Normalized System Age".into(),
        Op::NormalizedAge,
        T::Float,
        vec![
            arg("Age", K::Value(T::Float), D::Link("System.Age"), "seconds"),
            arg("Duration", K::Value(T::Float), D::Value(Value::Float(1.0)), "seconds that map to 1"),
        ],
        "System.Age / Duration, clamped to 0..1 (1 if Duration is not positive).".into(),
        "System.NormalizedAge".into(),
    );
    add(
        "Compare Floats".into(),
        Op::Compare,
        T::Bool,
        vec![
            arg("A", K::Value(T::Float), D::Value(Value::Float(0.0)), "left side"),
            arg("B", K::Value(T::Float), D::Value(Value::Float(0.0)), "right side"),
            arg("Operation", K::Choice(COMPARE), D::Value(Value::Int(0)), "comparison"),
        ],
        "A compared with B, as a Bool (for Select ... by Bool or Kill Particles).".into(),
        "Compare Float".into(),
    );
    use crate::data::DiType;
    let camera = || arg("Camera", K::DataInterface(DiType::CameraQuery), D::Required, "the camera: `Link(CameraQuery, \"User.Camera\")`");
    add(
        "Get Camera Position".into(),
        Op::CameraPosition,
        T::Vec3,
        vec![camera()],
        "Where the camera the effect is drawn for is, in the effect's space (the origin while no camera is known).".into(),
        "Camera Query: Get Camera Properties (Camera Position World)".into(),
    );
    add(
        "Get Camera Forward Vector".into(),
        Op::CameraForward,
        T::Vec3,
        vec![camera()],
        "The unit direction the camera looks in, in the effect's space (0, 0, -1 while no camera is known).".into(),
        "Camera Query: Get Camera Properties (Forward Vector World)".into(),
    );
    add(
        "Sample Scene Depth".into(),
        Op::SceneDepth,
        T::Float,
        vec![
            arg("Scene Depth", K::DataInterface(DiType::SceneDepth), D::Required, "the depth buffer: `Link(SceneDepth, \"User.Depth\")`"),
            arg("Position", K::Value(T::Vec3), D::Link("Particles.Position"), "the point whose pixel is read (metres, effect space)"),
        ],
        "Distance (metres) from the camera to the scene surface seen through Position, from the previous frame's depth buffer; 1e30 off screen, on the sky or with no depth. Compare it with the distance to Get Camera Position to tell whether a point is hidden. On the CPU, the host's colliders stand in for the depth buffer.".into(),
        "Collision Query: Query Scene Depth GPU".into(),
    );
    v
}

pub fn by_name(name: &str) -> Option<(u16, &'static DynamicDef)> {
    DYNAMIC_INPUTS
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == name)
        .map(|(i, d)| (i as u16, d))
}

pub fn from_id(id: u16) -> Option<&'static DynamicDef> {
    DYNAMIC_INPUTS.get(id as usize)
}

/// The random-draw salt of the dynamic input at `index` in its module's
/// list; its own draws use salt + component, input `i` gets salt + 8 + 8 * i.
/// Module inputs use 0..128 and 256.. ([`crate::sim::input_salt`]), modules' own draws 128..256.
pub fn node_salt(index: usize) -> u32 {
    4096 * (index as u32 + 1)
}

pub fn arg_salt(node: usize, arg: usize) -> u32 {
    node_salt(node) + 8 + 8 * arg as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_lists_last() {
        let mut names: Vec<_> = DYNAMIC_INPUTS.iter().map(|d| d.name.as_str()).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
        for d in DYNAMIC_INPUTS.iter() {
            for (i, a) in d.args.iter().enumerate() {
                if matches!(a.kind, ArgKind::List(_)) {
                    assert_eq!(i, d.args.len() - 1, "{}", d.name);
                }
            }
        }
    }
}
