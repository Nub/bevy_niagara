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
    /// The event being handled (event handler modules only).
    Event,
}

impl Namespace {
    pub const ALL: [Namespace; 6] = [
        Namespace::User,
        Namespace::System,
        Namespace::Emitter,
        Namespace::Particles,
        Namespace::Engine,
        Namespace::Event,
    ];
    pub fn prefix(self) -> &'static str {
        match self {
            Namespace::User => "User",
            Namespace::System => "System",
            Namespace::Emitter => "Emitter",
            Namespace::Particles => "Particles",
            Namespace::Engine => "Engine",
            Namespace::Event => "Event",
        }
    }
    /// Whether a module in `stage` may read parameters of this namespace.
    pub fn readable_in(self, stage: Stage) -> bool {
        match self {
            Namespace::User | Namespace::System | Namespace::Engine => true,
            Namespace::Emitter => !matches!(stage, Stage::SystemSpawn | Stage::SystemUpdate),
            Namespace::Particles => stage.is_particle(),
            Namespace::Event => stage == Stage::ParticleEvent,
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
    b(
        "Owner.Velocity",
        ValueType::Vec3,
        "velocity of the entity that owns the effect (m/s; set by the host, 0 by default)",
    ),
];
/// The event a handler is running for (`Event.` namespace), as in Niagara's event payloads.
pub const EVENT: &[Builtin] = &[
    b("Position", ValueType::Vec3, "where the source particle was (metres, effect space)"),
    b("Velocity", ValueType::Vec3, "the source particle's velocity (m/s)"),
    b(
        "Normal",
        ValueType::Vec3,
        "surface normal of a collision event; zero for death and location events",
    ),
    b("Age", ValueType::Float, "the source particle's age (seconds)"),
    b("ParticleID", ValueType::Int, "the source particle's UniqueID"),
];
pub const SYSTEM: &[Builtin] = &[b("Age", ValueType::Float, "seconds the system has run")];
pub const EMITTER: &[Builtin] = &[
    b("Age", ValueType::Float, "seconds the emitter has run"),
    b(
        "LoopedAge",
        ValueType::Float,
        "seconds since the current loop started (Emitter State); negative during its Loop Delay",
    ),
    b("LoopCount", ValueType::Int, "loops started before the current one (from 0)"),
    b(
        "NormalizedLoopAge",
        ValueType::Float,
        "LoopedAge / Loop Duration, 0..1 (0 for an endless loop)",
    ),
];

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
    pub const SCALE: u16 = 14;
    pub const MESH_ORIENTATION: u16 = 15;
    pub const RIBBON_ID: u16 = 16;
    pub const RIBBON_WIDTH: u16 = 17;
    pub const RIBBON_TWIST: u16 = 18;
    pub const RIBBON_LINK_ORDER: u16 = 19;
    pub const LIGHT_RADIUS: u16 = 20;
    pub const DYNAMIC_MATERIAL_PARAMETER: u16 = 21;
    pub const SUB_IMAGE_INDEX: u16 = 22;
    pub const INITIAL_SCALE: u16 = 23;
    pub const INITIAL_RIBBON_WIDTH: u16 = 24;
    pub const CAMERA_OFFSET: u16 = 25;
    pub const SPRITE_FACING: u16 = 26;
    pub const SPRITE_ALIGNMENT: u16 = 27;
    pub const DYNAMIC_MATERIAL_PARAMETER_1: u16 = 28;
    pub const ROTATIONAL_VELOCITY: u16 = 29;
    pub const PHYSICS_ROTATIONAL_FORCE: u16 = 30;
    pub const PHYSICS_ROTATIONAL_DRAG: u16 = 31;
    pub const DYNAMIC_MATERIAL_PARAMETER_2: u16 = 32;
    pub const DYNAMIC_MATERIAL_PARAMETER_3: u16 = 33;
    /// Attributes from here on are stored only by emitters that use them
    /// (see [`crate::cook::CookedEmitter::uses_material_parameters_2_3`]).
    pub const FIRST_EXTENDED_2: u16 = DYNAMIC_MATERIAL_PARAMETER_2;
    /// Attributes from here on are stored only by emitters that use them
    /// (see [`crate::cook::CookedEmitter::uses_extended_attributes`]).
    pub const FIRST_EXTENDED: u16 = SCALE;
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
    b("Scale", ValueType::Vec3, "mesh scale per axis"),
    b(
        "MeshOrientation",
        ValueType::Quat,
        "mesh rotation as a quaternion (x, y, z, w)",
    ),
    b(
        "RibbonID",
        ValueType::Int,
        "which ribbon the particle joins",
    ),
    b("RibbonWidth", ValueType::Float, "ribbon width in metres"),
    b(
        "RibbonTwist",
        ValueType::Float,
        "degrees around the ribbon direction",
    ),
    b(
        "RibbonLinkOrder",
        ValueType::Float,
        "order along the ribbon when the renderer's link_order is Custom",
    ),
    b("LightRadius", ValueType::Float, "light radius in metres"),
    b(
        "DynamicMaterialParameter",
        ValueType::Vec4,
        "free values passed to custom materials",
    ),
    b(
        "SubImageIndex",
        ValueType::Float,
        "flipbook frame the sprite renderer shows (set by SubUV Animation)",
    ),
    b(
        "Initial.Scale",
        ValueType::Vec3,
        "Scale as set by Initialize Mesh",
    ),
    b(
        "Initial.RibbonWidth",
        ValueType::Float,
        "RibbonWidth as set by Initialize Ribbon",
    ),
    b(
        "CameraOffset",
        ValueType::Float,
        "metres the sprite or mesh is drawn towards the camera (set by Camera Offset)",
    ),
    b(
        "SpriteFacing",
        ValueType::Vec3,
        "direction the sprite faces when the renderer's facing_mode is CustomFacingVector (0 = use the renderer's vector)",
    ),
    b(
        "SpriteAlignment",
        ValueType::Vec3,
        "the sprite's up when the renderer's alignment is CustomAlignment (0 = use the renderer's vector)",
    ),
    b(
        "DynamicMaterialParameter1",
        ValueType::Vec4,
        "second set of free values passed to custom materials",
    ),
    b(
        "RotationalVelocity",
        ValueType::Vec3,
        "mesh spin: degrees per second around each axis (a rotation vector)",
    ),
    b(
        "Physics.RotationalForce",
        ValueType::Vec3,
        "torque accumulated this step (cleared by Solve Forces and Velocity)",
    ),
    b(
        "Physics.RotationalDrag",
        ValueType::Float,
        "rotational drag accumulated this step (cleared by Solve Forces and Velocity)",
    ),
    b(
        "DynamicMaterialParameter2",
        ValueType::Vec4,
        "third set of free values passed to custom materials",
    ),
    b(
        "DynamicMaterialParameter3",
        ValueType::Vec4,
        "fourth set of free values passed to custom materials",
    ),
];

pub fn builtins(ns: Namespace) -> &'static [Builtin] {
    match ns {
        Namespace::Engine => ENGINE,
        Namespace::System => SYSTEM,
        Namespace::Emitter => EMITTER,
        Namespace::Particles => PARTICLES,
        Namespace::Event => EVENT,
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
