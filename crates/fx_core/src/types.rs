//! Niagara's parameter types, values and curves.

use bevy_math::{Quat, Vec2, Vec3, Vec4};
use serde::Serialize;

/// A Niagara parameter type.
///
/// `Color` is Niagara's *Linear Color* (linear RGBA); `Vec3` is Niagara's
/// *Vector*. Curves are not a value type of their own here: a curve is a
/// dynamic input (`CurveOverAge`) producing one of these types.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum ValueType {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Quat,
}

impl ValueType {
    pub const ALL: [ValueType; 8] = [
        ValueType::Float,
        ValueType::Int,
        ValueType::Bool,
        ValueType::Vec2,
        ValueType::Vec3,
        ValueType::Vec4,
        ValueType::Color,
        ValueType::Quat,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ValueType::Float => "Float",
            ValueType::Int => "Int",
            ValueType::Bool => "Bool",
            ValueType::Vec2 => "Vec2",
            ValueType::Vec3 => "Vec3",
            ValueType::Vec4 => "Vec4",
            ValueType::Color => "Color",
            ValueType::Quat => "Quat",
        }
    }

    pub fn from_name(name: &str) -> Option<ValueType> {
        ValueType::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Number of float components (0 for Int and Bool).
    pub fn components(self) -> usize {
        match self {
            ValueType::Float => 1,
            ValueType::Vec2 => 2,
            ValueType::Vec3 => 3,
            ValueType::Vec4 | ValueType::Color | ValueType::Quat => 4,
            ValueType::Int | ValueType::Bool => 0,
        }
    }

    /// An example literal of this type, used in error messages.
    pub fn example(self) -> &'static str {
        match self {
            ValueType::Float => "Float(1.0)",
            ValueType::Int => "Int(10)",
            ValueType::Bool => "Bool(true)",
            ValueType::Vec2 => "Vec2((1.0, 1.0))",
            ValueType::Vec3 => "Vec3((0.0, 1.0, 0.0))",
            ValueType::Vec4 => "Vec4((0.0, 0.0, 0.0, 1.0))",
            ValueType::Color => "Color((1.0, 0.5, 0.2, 1.0))",
            ValueType::Quat => "Quat((0.0, 0.0, 0.0, 1.0))",
        }
    }

    /// Whether `RandomRange` and `CurveOverAge` can produce this type.
    pub fn interpolable(self) -> bool {
        self.components() > 0 && self != ValueType::Quat
    }
}

/// A typed constant value. Serialises to RON as `Float(1.0)`, `Vec3((0.0, 1.0, 0.0))`, ...
#[derive(
    Clone, Copy, Debug, PartialEq, Serialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum Value {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    /// (x, y, z, w)
    Quat([f32; 4]),
}

impl Value {
    pub fn ty(&self) -> ValueType {
        match self {
            Value::Float(_) => ValueType::Float,
            Value::Int(_) => ValueType::Int,
            Value::Bool(_) => ValueType::Bool,
            Value::Vec2(_) => ValueType::Vec2,
            Value::Vec3(_) => ValueType::Vec3,
            Value::Vec4(_) => ValueType::Vec4,
            Value::Color(_) => ValueType::Color,
            Value::Quat(_) => ValueType::Quat,
        }
    }

    /// Float components padded to 4 (Int/Bool give zeros).
    pub fn to_array(&self) -> [f32; 4] {
        match *self {
            Value::Float(x) => [x, 0.0, 0.0, 0.0],
            Value::Vec2([x, y]) => [x, y, 0.0, 0.0],
            Value::Vec3([x, y, z]) => [x, y, z, 0.0],
            Value::Vec4(v) | Value::Color(v) | Value::Quat(v) => v,
            Value::Int(_) | Value::Bool(_) => [0.0; 4],
        }
    }

    /// Builds a value of float-component type `ty` from components.
    pub fn from_array(ty: ValueType, a: [f32; 4]) -> Value {
        match ty {
            ValueType::Float => Value::Float(a[0]),
            ValueType::Vec2 => Value::Vec2([a[0], a[1]]),
            ValueType::Vec3 => Value::Vec3([a[0], a[1], a[2]]),
            ValueType::Vec4 => Value::Vec4(a),
            ValueType::Color => Value::Color(a),
            ValueType::Quat => Value::Quat(a),
            ValueType::Int => Value::Int(a[0] as i32),
            ValueType::Bool => Value::Bool(a[0] != 0.0),
        }
    }

    pub fn as_f32(&self) -> f32 {
        match *self {
            Value::Float(x) => x,
            Value::Int(i) => i as f32,
            Value::Bool(b) => b as i32 as f32,
            _ => self.to_array()[0],
        }
    }
    pub fn as_i32(&self) -> i32 {
        match *self {
            Value::Int(i) => i,
            Value::Bool(b) => b as i32,
            _ => self.as_f32() as i32,
        }
    }
    pub fn as_bool(&self) -> bool {
        match *self {
            Value::Bool(b) => b,
            Value::Int(i) => i != 0,
            _ => self.as_f32() != 0.0,
        }
    }
    pub fn as_vec2(&self) -> Vec2 {
        let a = self.to_array();
        Vec2::new(a[0], a[1])
    }
    pub fn as_vec3(&self) -> Vec3 {
        let a = self.to_array();
        Vec3::new(a[0], a[1], a[2])
    }
    pub fn as_vec4(&self) -> Vec4 {
        Vec4::from_array(self.to_array())
    }
    pub fn as_quat(&self) -> Quat {
        Quat::from_array(self.to_array())
    }
}

/// A key on a curve: `(time, value)`.
pub type Key<T> = (f32, T);

/// A piecewise-linear curve, typed by the value it produces. Times are
/// normalised (0..=1) and must be ascending; outside the keys the curve holds
/// its end values.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Curve {
    Float(Vec<Key<f32>>),
    Vec2(Vec<Key<[f32; 2]>>),
    Vec3(Vec<Key<[f32; 3]>>),
    Vec4(Vec<Key<[f32; 4]>>),
    Color(Vec<Key<[f32; 4]>>),
}

impl Curve {
    pub fn ty(&self) -> ValueType {
        match self {
            Curve::Float(_) => ValueType::Float,
            Curve::Vec2(_) => ValueType::Vec2,
            Curve::Vec3(_) => ValueType::Vec3,
            Curve::Vec4(_) => ValueType::Vec4,
            Curve::Color(_) => ValueType::Color,
        }
    }

    /// Keys as `(time, padded components)`.
    pub fn keys(&self) -> Vec<(f32, [f32; 4])> {
        match self {
            Curve::Float(k) => k.iter().map(|&(t, v)| (t, [v, 0.0, 0.0, 0.0])).collect(),
            Curve::Vec2(k) => k
                .iter()
                .map(|&(t, v)| (t, [v[0], v[1], 0.0, 0.0]))
                .collect(),
            Curve::Vec3(k) => k
                .iter()
                .map(|&(t, v)| (t, [v[0], v[1], v[2], 0.0]))
                .collect(),
            Curve::Vec4(k) | Curve::Color(k) => k.clone(),
        }
    }
}
