//! Parameter namespaces and the built-in parameters modules and links can read.

use crate::model::Stage;
use crate::types::ValueType;

/// Niagara's parameter namespaces.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum Namespace {
    User,
    System,
    Emitter,
    Particles,
    Engine,
}

impl Namespace {
    pub const ALL: [Namespace; 5] = [
        Namespace::User,
        Namespace::System,
        Namespace::Emitter,
        Namespace::Particles,
        Namespace::Engine,
    ];
    pub fn prefix(self) -> &'static str {
        match self {
            Namespace::User => "User",
            Namespace::System => "System",
            Namespace::Emitter => "Emitter",
            Namespace::Particles => "Particles",
            Namespace::Engine => "Engine",
        }
    }
    /// Whether a module in `stage` may read parameters of this namespace.
    pub fn readable_in(self, stage: Stage) -> bool {
        match self {
            Namespace::User | Namespace::System | Namespace::Engine => true,
            Namespace::Emitter => !matches!(stage, Stage::SystemSpawn | Stage::SystemUpdate),
            Namespace::Particles => stage.is_particle(),
        }
    }
}

/// A built-in parameter: `(name without namespace, type, meaning)`.
pub struct Builtin {
    pub name: &'static str,
    pub ty: ValueType,
    pub doc: &'static str,
}

const fn b(name: &'static str, ty: ValueType, doc: &'static str) -> Builtin {
    Builtin { name, ty, doc }
}

pub const ENGINE: &[Builtin] = &[
    b("DeltaTime", ValueType::Float, "seconds simulated this step"),
    b(
        "Time",
        ValueType::Float,
        "seconds since the system started, at the start of this step",
    ),
];
pub const SYSTEM: &[Builtin] = &[b("Age", ValueType::Float, "seconds the system has run")];
pub const EMITTER: &[Builtin] = &[b("Age", ValueType::Float, "seconds the emitter has run")];

/// Index of each particle attribute in [`PARTICLES`].
pub mod attr {
    pub const POSITION: u16 = 0;
    pub const VELOCITY: u16 = 1;
    pub const COLOR: u16 = 2;
    pub const SPRITE_SIZE: u16 = 3;
    pub const SPRITE_ROTATION: u16 = 4;
    pub const MASS: u16 = 5;
    pub const LIFETIME: u16 = 6;
    pub const AGE: u16 = 7;
    pub const NORMALIZED_AGE: u16 = 8;
    pub const UNIQUE_ID: u16 = 9;
    pub const INITIAL_COLOR: u16 = 10;
    pub const INITIAL_SPRITE_SIZE: u16 = 11;
    pub const PHYSICS_FORCE: u16 = 12;
    pub const PHYSICS_DRAG: u16 = 13;
}

/// Particle attributes (`Particles.` namespace), Niagara's names.
pub const PARTICLES: &[Builtin] = &[
    b("Position", ValueType::Vec3, "world position (metres, Y up)"),
    b("Velocity", ValueType::Vec3, "metres per second"),
    b("Color", ValueType::Color, "linear RGBA"),
    b(
        "SpriteSize",
        ValueType::Vec2,
        "sprite width and height in metres",
    ),
    b(
        "SpriteRotation",
        ValueType::Float,
        "sprite rotation in degrees",
    ),
    b("Mass", ValueType::Float, "kilograms"),
    b("Lifetime", ValueType::Float, "seconds the particle lives"),
    b("Age", ValueType::Float, "seconds since spawn"),
    b("NormalizedAge", ValueType::Float, "Age / Lifetime"),
    b("UniqueID", ValueType::Int, "id unique within the emitter"),
    b(
        "Initial.Color",
        ValueType::Color,
        "Color as set by Initialize Particle",
    ),
    b(
        "Initial.SpriteSize",
        ValueType::Vec2,
        "SpriteSize as set by Initialize Particle",
    ),
    b(
        "Physics.Force",
        ValueType::Vec3,
        "force accumulated this step (cleared by Solve Forces and Velocity)",
    ),
    b(
        "Physics.Drag",
        ValueType::Float,
        "drag accumulated this step (cleared by Solve Forces and Velocity)",
    ),
];

pub fn builtins(ns: Namespace) -> &'static [Builtin] {
    match ns {
        Namespace::Engine => ENGINE,
        Namespace::System => SYSTEM,
        Namespace::Emitter => EMITTER,
        Namespace::Particles => PARTICLES,
        Namespace::User => &[],
    }
}

/// A resolved parameter reference: namespace + index (into the builtin table,
/// or into the system's user parameters for `User`).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ParamRef {
    pub ns: Namespace,
    pub index: u16,
}

/// Splits `"Particles.Physics.Force"` into its namespace and the rest.
pub fn split_namespace(full: &str) -> Option<(Namespace, &str)> {
    let (head, rest) = full.split_once('.')?;
    let ns = Namespace::ALL.into_iter().find(|n| n.prefix() == head)?;
    (!rest.is_empty()).then_some((ns, rest))
}

/// All full names of a namespace's builtins, for error messages.
pub fn list_builtins(ns: Namespace) -> String {
    builtins(ns)
        .iter()
        .map(|b| format!("{}.{} ({})", ns.prefix(), b.name, b.ty.name()))
        .collect::<Vec<_>>()
        .join(", ")
}
