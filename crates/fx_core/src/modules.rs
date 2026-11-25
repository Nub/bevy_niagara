//! The module library: Niagara's core modules with their names, inputs,
//! defaults and the stages they may be placed in.
//!
//! Units follow Bevy (metres, seconds, Y up), not Unreal (centimetres, Z up):
//! Niagara's default gravity of (0, 0, -980) cm/s² is (0, -9.81, 0) m/s² here.

use crate::model::Stage;
use crate::types::{Value, ValueType};

/// The modules the runtime implements. The discriminant is the cooked module id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ModuleKind {
    SpawnRate = 0,
    SpawnBurstInstantaneous = 1,
    InitializeParticle = 2,
    AddVelocity = 3,
    AddVelocityInCone = 4,
    GravityForce = 5,
    Drag = 6,
    CurlNoiseForce = 7,
    ScaleColor = 8,
    ScaleSpriteSize = 9,
    SolveForcesAndVelocity = 10,
    ParticleState = 11,
    KillParticles = 12,
    InitializeRibbon = 13,
    InitializeMesh = 14,
    UpdateMeshOrientation = 15,
    InitializeLight = 16,
    DynamicMaterialParameters = 17,
    PointAttractionForce = 18,
    VortexVelocity = 19,
    VortexForce = 20,
    WindForce = 21,
    LinearForce = 22,
    LimitForce = 23,
    AccelerationForce = 24,
    PointForce = 25,
    InheritVelocity = 26,
    ShapeLocation = 27,
    InitialMeshOrientation = 28,
    SpriteRotationRate = 29,
    SubUvAnimation = 30,
    ScaleMeshSize = 31,
    ScaleRibbonWidth = 32,
    Color = 33,
    ScaleColorBySpeed = 34,
    ScaleSpriteSizeBySpeed = 35,
    KillParticlesInVolume = 36,
}

pub struct InputDef {
    pub name: &'static str,
    pub ty: ValueType,
    pub default: Value,
    pub doc: &'static str,
    /// For an Int input that is a Niagara static switch (an enum chosen when
    /// the effect is written): the names of its values, written bare in
    /// source (`"Shape Primitive": Box`) and cooked to their index.
    pub choices: &'static [&'static str],
}

pub struct ModuleDef {
    pub kind: ModuleKind,
    pub name: &'static str,
    pub stages: &'static [Stage],
    pub inputs: &'static [InputDef],
    pub doc: &'static str,
}

impl ModuleDef {
    pub fn input(&self, name: &str) -> Option<(usize, &'static InputDef)> {
        self.inputs.iter().enumerate().find(|(_, i)| i.name == name)
    }
    pub fn allowed_in(&self, stage: Stage) -> bool {
        self.stages.contains(&stage)
    }
    /// "Spawn Rate (Float), ..." for error messages.
    pub fn list_inputs(&self) -> String {
        if self.inputs.is_empty() {
            return "(this module has no inputs)".into();
        }
        self.inputs
            .iter()
            .map(|i| format!("\"{}\" ({})", i.name, i.ty.name()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

const fn i(name: &'static str, ty: ValueType, default: Value, doc: &'static str) -> InputDef {
    InputDef {
        name,
        ty,
        default,
        doc,
        choices: &[],
    }
}

/// A static-switch input: one of `choices`, cooked to its index (an Int).
const fn e(
    name: &'static str,
    choices: &'static [&'static str],
    default: i32,
    doc: &'static str,
) -> InputDef {
    InputDef {
        name,
        ty: ValueType::Int,
        default: Value::Int(default),
        doc,
        choices,
    }
}

use Stage::*;
use ValueType as T;

const PARTICLE: &[Stage] = &[ParticleSpawn, ParticleUpdate];

pub static MODULES: &[ModuleDef] = &[
    ModuleDef {
        kind: ModuleKind::SpawnRate,
        name: "Spawn Rate",
        stages: &[EmitterUpdate],
        inputs: &[i(
            "Spawn Rate",
            T::Float,
            Value::Float(10.0),
            "particles per second",
        )],
        doc: "Spawns particles continuously; fractional particles carry over to the next step.",
    },
    ModuleDef {
        kind: ModuleKind::SpawnBurstInstantaneous,
        name: "Spawn Burst Instantaneous",
        stages: &[EmitterUpdate],
        inputs: &[
            i(
                "Spawn Count",
                T::Int,
                Value::Int(10),
                "particles spawned at once",
            ),
            i(
                "Spawn Time",
                T::Float,
                Value::Float(0.0),
                "emitter age (seconds) at which the burst fires",
            ),
        ],
        doc: "Spawns Spawn Count particles in the step where the emitter age reaches Spawn Time.",
    },
    ModuleDef {
        kind: ModuleKind::InitializeParticle,
        name: "Initialize Particle",
        stages: &[ParticleSpawn],
        inputs: &[
            i("Lifetime", T::Float, Value::Float(2.0), "seconds"),
            i(
                "Position",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "spawn position (metres)",
            ),
            i("Color", T::Color, Value::Color([1.0; 4]), "linear RGBA"),
            i("Sprite Size", T::Vec2, Value::Vec2([0.1, 0.1]), "metres"),
            i("Sprite Rotation", T::Float, Value::Float(0.0), "degrees"),
            i("Mass", T::Float, Value::Float(1.0), "kilograms"),
        ],
        doc: "Sets the particle's starting attributes.",
    },
    ModuleDef {
        kind: ModuleKind::AddVelocity,
        name: "Add Velocity",
        stages: PARTICLE,
        inputs: &[i(
            "Velocity",
            T::Vec3,
            Value::Vec3([0.0, 1.0, 0.0]),
            "metres per second, added to Particles.Velocity",
        )],
        doc: "Adds a linear velocity.",
    },
    ModuleDef {
        kind: ModuleKind::AddVelocityInCone,
        name: "Add Velocity in Cone",
        stages: PARTICLE,
        inputs: &[
            i(
                "Cone Axis",
                T::Vec3,
                Value::Vec3([0.0, 1.0, 0.0]),
                "cone direction (normalised for you)",
            ),
            i(
                "Cone Angle",
                T::Float,
                Value::Float(45.0),
                "half-angle in degrees (0 = along the axis, 180 = any direction)",
            ),
            i(
                "Velocity Strength",
                T::Float,
                Value::Float(1.0),
                "speed in metres per second",
            ),
        ],
        doc: "Adds a velocity in a random direction, uniform over the cone's solid angle.",
    },
    ModuleDef {
        kind: ModuleKind::GravityForce,
        name: "Gravity Force",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Gravity",
            T::Vec3,
            Value::Vec3([0.0, -9.81, 0.0]),
            "acceleration in m/s²",
        )],
        doc: "Adds Gravity * Mass to Particles.Physics.Force.",
    },
    ModuleDef {
        kind: ModuleKind::Drag,
        name: "Drag",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Drag",
            T::Float,
            Value::Float(1.0),
            "linear drag per second",
        )],
        doc: "Adds to Particles.Physics.Drag; Solve Forces and Velocity scales velocity by exp(-Drag * DeltaTime).",
    },
    ModuleDef {
        kind: ModuleKind::CurlNoiseForce,
        name: "Curl Noise Force",
        stages: &[ParticleUpdate],
        inputs: &[
            i(
                "Noise Strength",
                T::Float,
                Value::Float(1.0),
                "force scale (newtons)",
            ),
            i(
                "Noise Frequency",
                T::Float,
                Value::Float(1.0),
                "noise cells per metre",
            ),
            i(
                "Noise Offset",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "pans the noise field (link Engine.Time for motion)",
            ),
            i(
                "Pan Noise Field",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "moves the noise field over time (noise-space units per second)",
            ),
        ],
        doc: "Adds a divergence-free curl-noise force sampled at Position * Noise Frequency + Noise Offset + Pan Noise Field * Engine.Time.",
    },
    ModuleDef {
        kind: ModuleKind::ScaleColor,
        name: "Scale Color",
        stages: PARTICLE,
        inputs: &[
            i(
                "Scale RGB",
                T::Vec3,
                Value::Vec3([1.0; 3]),
                "multiplies the initial RGB",
            ),
            i(
                "Scale Alpha",
                T::Float,
                Value::Float(1.0),
                "multiplies the initial alpha",
            ),
        ],
        doc: "Particles.Color = Particles.Initial.Color * (Scale RGB, Scale Alpha).",
    },
    ModuleDef {
        kind: ModuleKind::ScaleSpriteSize,
        name: "Scale Sprite Size",
        stages: PARTICLE,
        inputs: &[i(
            "Scale Factor",
            T::Vec2,
            Value::Vec2([1.0, 1.0]),
            "multiplies the initial sprite size",
        )],
        doc: "Particles.SpriteSize = Particles.Initial.SpriteSize * Scale Factor.",
    },
    ModuleDef {
        kind: ModuleKind::SolveForcesAndVelocity,
        name: "Solve Forces and Velocity",
        stages: &[ParticleUpdate],
        inputs: &[],
        doc: "Velocity += Force / Mass * dt; Velocity *= exp(-Drag * dt); Position += Velocity * dt; then clears Force and Drag.",
    },
    ModuleDef {
        kind: ModuleKind::ParticleState,
        name: "Particle State",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Kill Particles When Lifetime Has Elapsed",
            T::Bool,
            Value::Bool(true),
            "kill once Age > Lifetime",
        )],
        doc: "Advances Age and NormalizedAge and kills particles past their Lifetime.",
    },
    ModuleDef {
        kind: ModuleKind::KillParticles,
        name: "Kill Particles",
        stages: PARTICLE,
        inputs: &[i(
            "Kill Particles",
            T::Bool,
            Value::Bool(true),
            "kill the particle when true",
        )],
        doc: "Kills the particle when Kill Particles is true.",
    },
    ModuleDef {
        kind: ModuleKind::InitializeRibbon,
        name: "Initialize Ribbon",
        stages: &[ParticleSpawn],
        inputs: &[
            i("Ribbon Width", T::Float, Value::Float(0.1), "metres"),
            i(
                "Ribbon Twist",
                T::Float,
                Value::Float(0.0),
                "degrees around the ribbon direction",
            ),
            i(
                "Ribbon ID",
                T::Int,
                Value::Int(0),
                "which ribbon the particle joins (0..255)",
            ),
            i(
                "Ribbon Link Order",
                T::Float,
                Value::Float(0.0),
                "order along the ribbon when the renderer's link_order is Custom",
            ),
        ],
        doc: "Sets Particles.RibbonWidth, RibbonTwist, RibbonID and RibbonLinkOrder for a Ribbon renderer.",
    },
    ModuleDef {
        kind: ModuleKind::InitializeMesh,
        name: "Initialize Mesh",
        stages: &[ParticleSpawn],
        inputs: &[
            i(
                "Mesh Scale",
                T::Vec3,
                Value::Vec3([1.0; 3]),
                "scale per axis",
            ),
            i(
                "Mesh Rotation",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "Euler angles in degrees, applied X, then Y, then Z",
            ),
        ],
        doc: "Sets Particles.Scale and Particles.MeshOrientation for a Mesh renderer.",
    },
    ModuleDef {
        kind: ModuleKind::UpdateMeshOrientation,
        name: "Update Mesh Orientation",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Rotation Rate",
            T::Vec3,
            Value::Vec3([0.0; 3]),
            "degrees per second around X, Y and Z",
        )],
        doc: "Particles.MeshOrientation = normalize(MeshOrientation * rotation by Rotation Rate * DeltaTime).",
    },
    ModuleDef {
        kind: ModuleKind::InitializeLight,
        name: "Initialize Light",
        stages: PARTICLE,
        inputs: &[i("Light Radius", T::Float, Value::Float(1.0), "metres")],
        doc: "Sets Particles.LightRadius for a Light renderer.",
    },
    ModuleDef {
        kind: ModuleKind::DynamicMaterialParameters,
        name: "Dynamic Material Parameters",
        stages: PARTICLE,
        inputs: &[i(
            "Dynamic Material Parameter",
            T::Vec4,
            Value::Vec4([0.0; 4]),
            "four free values exposed to custom materials",
        )],
        doc: "Sets Particles.DynamicMaterialParameter, which custom materials can read.",
    },
    ModuleDef {
        kind: ModuleKind::PointAttractionForce,
        name: "Point Attraction Force",
        stages: &[ParticleUpdate],
        inputs: &[
            i(
                "Attractor Position",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "point the particles are pulled towards (metres, emitter space)",
            ),
            i(
                "Attraction Strength",
                T::Float,
                Value::Float(1.0),
                "force at the attractor (newtons)",
            ),
            i(
                "Attraction Radius",
                T::Float,
                Value::Float(1.0),
                "no pull at or beyond this distance (metres)",
            ),
            i(
                "Falloff Exponent",
                T::Float,
                Value::Float(1.0),
                "shape of the falloff: (1 - distance / radius) ^ exponent",
            ),
            i(
                "Kill Radius",
                T::Float,
                Value::Float(0.0),
                "particles closer than this are killed; 0 = never",
            ),
        ],
        doc: "Pulls particles towards Attractor Position: Physics.Force += direction * Attraction Strength * (1 - distance / Attraction Radius) ^ Falloff Exponent, within the radius. Kills particles inside Kill Radius.",
    },
    ModuleDef {
        kind: ModuleKind::VortexVelocity,
        name: "Vortex Velocity",
        stages: PARTICLE,
        inputs: &[
            i(
                "Vortex Axis",
                T::Vec3,
                Value::Vec3([0.0, 1.0, 0.0]),
                "axis to swirl around (normalised for you; Y is up)",
            ),
            i(
                "Vortex Origin",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "a point on the axis (metres)",
            ),
            i(
                "Velocity Amount",
                T::Float,
                Value::Float(1.0),
                "metres per second, counter-clockwise seen from the axis tip",
            ),
        ],
        doc: "Adds Velocity Amount of velocity around the vortex axis, tangent to the circle through the particle (none on the axis itself).",
    },
    ModuleDef {
        kind: ModuleKind::VortexForce,
        name: "Vortex Force",
        stages: &[ParticleUpdate],
        inputs: &[
            i(
                "Vortex Axis",
                T::Vec3,
                Value::Vec3([0.0, 1.0, 0.0]),
                "axis to swirl around (normalised for you; Y is up)",
            ),
            i(
                "Vortex Origin",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "a point on the axis (metres)",
            ),
            i(
                "Vortex Force Amount",
                T::Float,
                Value::Float(1.0),
                "tangential force (newtons), counter-clockwise seen from the axis tip",
            ),
            i(
                "Origin Pull Amount",
                T::Float,
                Value::Float(0.0),
                "force towards the axis (newtons); negative pushes out",
            ),
        ],
        doc: "Adds a force around the vortex axis (Vortex Force Amount, tangent) and towards it (Origin Pull Amount) to Physics.Force.",
    },
    ModuleDef {
        kind: ModuleKind::WindForce,
        name: "Wind Force",
        stages: &[ParticleUpdate],
        inputs: &[
            i(
                "Wind Velocity",
                T::Vec3,
                Value::Vec3([1.0, 0.0, 0.0]),
                "wind direction and speed (metres per second)",
            ),
            i(
                "Wind Speed",
                T::Float,
                Value::Float(1.0),
                "multiplies Wind Velocity",
            ),
            i(
                "Air Resistance",
                T::Float,
                Value::Float(1.0),
                "how strongly particles are pulled to the wind's velocity (per second, per kg)",
            ),
        ],
        doc: "Physics.Force += (Wind Velocity * Wind Speed - Velocity) * Air Resistance: particles approach the wind's velocity.",
    },
    ModuleDef {
        kind: ModuleKind::LinearForce,
        name: "Linear Force",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Linear Force",
            T::Vec3,
            Value::Vec3([0.0, 1.0, 0.0]),
            "force in newtons",
        )],
        doc: "Adds a constant force to Physics.Force (heavier particles accelerate less).",
    },
    ModuleDef {
        kind: ModuleKind::LimitForce,
        name: "Limit Force",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Force Limit",
            T::Float,
            Value::Float(1.0),
            "largest force length (newtons)",
        )],
        doc: "Scales Physics.Force down to Force Limit if it is longer. Place after the forces it limits.",
    },
    ModuleDef {
        kind: ModuleKind::AccelerationForce,
        name: "Acceleration Force",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Acceleration",
            T::Vec3,
            Value::Vec3([0.0, 1.0, 0.0]),
            "metres per second squared",
        )],
        doc: "Physics.Force += Acceleration * Mass: the same acceleration whatever the mass.",
    },
    ModuleDef {
        kind: ModuleKind::PointForce,
        name: "Point Force",
        stages: &[ParticleUpdate],
        inputs: &[
            i(
                "Point Position",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "centre of the force (metres, emitter space)",
            ),
            i(
                "Point Force Strength",
                T::Float,
                Value::Float(1.0),
                "force at the centre (newtons); positive pushes away, negative pulls in",
            ),
            i(
                "Falloff Radius",
                T::Float,
                Value::Float(1.0),
                "no force at or beyond this distance (metres)",
            ),
            i(
                "Falloff Exponent",
                T::Float,
                Value::Float(1.0),
                "shape of the falloff: (1 - distance / radius) ^ exponent",
            ),
        ],
        doc: "Pushes particles away from Point Position: Physics.Force += direction * Point Force Strength * (1 - distance / Falloff Radius) ^ Falloff Exponent, within the radius.",
    },
    ModuleDef {
        kind: ModuleKind::InheritVelocity,
        name: "Inherit Velocity",
        stages: &[ParticleSpawn],
        inputs: &[i(
            "Velocity Scale",
            T::Float,
            Value::Float(1.0),
            "fraction of the owner's velocity to inherit",
        )],
        doc: "Velocity += Engine.Owner.Velocity * Velocity Scale: new particles carry the owning entity's motion (the host sets the owner velocity; 0 until it does).",
    },
    ModuleDef {
        kind: ModuleKind::ShapeLocation,
        name: "Shape Location",
        stages: &[ParticleSpawn],
        inputs: &[
            e(
                "Shape Primitive",
                &["Sphere", "Box", "Cylinder", "Torus", "Cone", "Ring"],
                0,
                "the shape: Sphere, Box, Cylinder, Torus, Cone or Ring (Y is the shape's axis)",
            ),
            i(
                "Surface Only",
                T::Bool,
                Value::Bool(false),
                "on the surface instead of inside (Sphere, Box, Cylinder side, Torus, Cone side)",
            ),
            i(
                "Offset",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "moves the shape (metres)",
            ),
            i("Sphere Radius", T::Float, Value::Float(0.5), "metres"),
            i(
                "Box Size",
                T::Vec3,
                Value::Vec3([1.0; 3]),
                "full size per axis (metres), centred",
            ),
            i(
                "Cylinder Height",
                T::Float,
                Value::Float(1.0),
                "metres along Y, centred",
            ),
            i("Cylinder Radius", T::Float, Value::Float(0.5), "metres"),
            i(
                "Large Radius",
                T::Float,
                Value::Float(0.5),
                "torus: centre of the ring to the centre of the tube (metres)",
            ),
            i(
                "Handle Radius",
                T::Float,
                Value::Float(0.1),
                "torus: tube radius (metres)",
            ),
            i(
                "Cone Length",
                T::Float,
                Value::Float(1.0),
                "metres from the apex (at Offset) along +Y",
            ),
            i(
                "Cone Angle",
                T::Float,
                Value::Float(30.0),
                "half-angle in degrees",
            ),
            i(
                "Ring Radius",
                T::Float,
                Value::Float(0.5),
                "metres, in the XZ plane",
            ),
            i(
                "Disc Coverage",
                T::Float,
                Value::Float(0.0),
                "ring: 0 = on the circle, 1 = anywhere on the disc",
            ),
        ],
        doc: "Sets Particles.Position to a uniformly random point in (or on) a shape centred at Offset. Place it after Initialize Particle, which also sets Position.",
    },
    ModuleDef {
        kind: ModuleKind::InitialMeshOrientation,
        name: "Initial Mesh Orientation",
        stages: &[ParticleSpawn],
        inputs: &[i(
            "Rotation",
            T::Vec3,
            Value::Vec3([0.0; 3]),
            "Euler angles in degrees, applied X, then Y, then Z (RandomRange for random orientations)",
        )],
        doc: "Sets Particles.MeshOrientation from Euler angles.",
    },
    ModuleDef {
        kind: ModuleKind::SpriteRotationRate,
        name: "Sprite Rotation Rate",
        stages: &[ParticleUpdate],
        inputs: &[i(
            "Rotation Rate",
            T::Float,
            Value::Float(90.0),
            "degrees per second",
        )],
        doc: "Particles.SpriteRotation += Rotation Rate * DeltaTime.",
    },
    ModuleDef {
        kind: ModuleKind::SubUvAnimation,
        name: "SubUV Animation",
        stages: PARTICLE,
        inputs: &[
            e(
                "SubUV Animation Mode",
                &["Linear", "Random", "InfiniteLoop"],
                0,
                "Linear: first to last frame over the particle's life; Random: one random frame per particle; InfiniteLoop: Frames Per Second, wrapping",
            ),
            i(
                "Number of Frames",
                T::Int,
                Value::Int(0),
                "frames to play; 0 = every cell of the sprite renderer's sub_image_size",
            ),
            i(
                "Start Frame",
                T::Int,
                Value::Int(0),
                "first frame (cells count left to right, top to bottom from 0)",
            ),
            i(
                "Frames Per Second",
                T::Float,
                Value::Float(30.0),
                "InfiniteLoop playback rate",
            ),
        ],
        doc: "Sets Particles.SubImageIndex, the flipbook frame the sprite renderer shows (fractional when the renderer blends frames). Without it the renderer plays the flipbook over the particle's life.",
    },
    ModuleDef {
        kind: ModuleKind::ScaleMeshSize,
        name: "Scale Mesh Size",
        stages: PARTICLE,
        inputs: &[i(
            "Scale Factor",
            T::Vec3,
            Value::Vec3([1.0; 3]),
            "multiplies the initial mesh scale",
        )],
        doc: "Particles.Scale = Particles.Initial.Scale * Scale Factor (Initial.Scale is set by Initialize Mesh).",
    },
    ModuleDef {
        kind: ModuleKind::ScaleRibbonWidth,
        name: "Scale Ribbon Width",
        stages: PARTICLE,
        inputs: &[i(
            "Ribbon Width Scale",
            T::Float,
            Value::Float(1.0),
            "multiplies the initial ribbon width",
        )],
        doc: "Particles.RibbonWidth = Particles.Initial.RibbonWidth * Ribbon Width Scale (Initial.RibbonWidth is set by Initialize Ribbon).",
    },
    ModuleDef {
        kind: ModuleKind::Color,
        name: "Color",
        stages: PARTICLE,
        inputs: &[i(
            "Color",
            T::Color,
            Value::Color([1.0; 4]),
            "linear RGBA; usually CurveOverAge",
        )],
        doc: "Sets Particles.Color (Initial.Color is left as Initialize Particle set it).",
    },
    ModuleDef {
        kind: ModuleKind::ScaleColorBySpeed,
        name: "Scale Color by Speed",
        stages: PARTICLE,
        inputs: &[
            i(
                "Velocity Threshold",
                T::Float,
                Value::Float(1.0),
                "speed (m/s) at which Max Scale Factor is reached",
            ),
            i(
                "Min Scale Factor",
                T::Color,
                Value::Color([0.5, 0.5, 0.5, 1.0]),
                "scale at rest",
            ),
            i(
                "Max Scale Factor",
                T::Color,
                Value::Color([1.0; 4]),
                "scale at Velocity Threshold and above",
            ),
        ],
        doc: "Particles.Color = Particles.Initial.Color * lerp(Min Scale Factor, Max Scale Factor, clamp(speed / Velocity Threshold, 0, 1)).",
    },
    ModuleDef {
        kind: ModuleKind::ScaleSpriteSizeBySpeed,
        name: "Scale Sprite Size by Speed",
        stages: PARTICLE,
        inputs: &[
            i(
                "Velocity Threshold",
                T::Float,
                Value::Float(1.0),
                "speed (m/s) at which Max Scale Factor is reached",
            ),
            i(
                "Min Scale Factor",
                T::Vec2,
                Value::Vec2([1.0, 1.0]),
                "scale at rest",
            ),
            i(
                "Max Scale Factor",
                T::Vec2,
                Value::Vec2([1.0, 2.0]),
                "scale at Velocity Threshold and above",
            ),
        ],
        doc: "Particles.SpriteSize = Particles.Initial.SpriteSize * lerp(Min Scale Factor, Max Scale Factor, clamp(speed / Velocity Threshold, 0, 1)).",
    },
    ModuleDef {
        kind: ModuleKind::KillParticlesInVolume,
        name: "Kill Particles in Volume",
        stages: PARTICLE,
        inputs: &[
            e(
                "Volume Shape",
                &["Sphere", "Box"],
                0,
                "Sphere or Box",
            ),
            i(
                "Volume Center",
                T::Vec3,
                Value::Vec3([0.0; 3]),
                "metres, emitter space",
            ),
            i("Sphere Radius", T::Float, Value::Float(1.0), "metres"),
            i(
                "Box Size",
                T::Vec3,
                Value::Vec3([1.0; 3]),
                "full size per axis (metres), centred",
            ),
            i(
                "Invert Volume",
                T::Bool,
                Value::Bool(false),
                "kill particles outside the volume instead",
            ),
        ],
        doc: "Kills particles strictly inside the volume (or outside it, with Invert Volume).",
    },
];

pub fn by_name(name: &str) -> Option<&'static ModuleDef> {
    MODULES.iter().find(|m| m.name == name)
}

/// The table's own copy of static-switch value `name`, if any module has it.
pub fn static_choice(name: &str) -> Option<&'static str> {
    MODULES
        .iter()
        .flat_map(|m| m.inputs)
        .flat_map(|i| i.choices)
        .find(|c| **c == name)
        .copied()
}

pub fn by_kind(kind: ModuleKind) -> &'static ModuleDef {
    &MODULES[kind as usize]
}

pub fn from_id(id: u16) -> Option<ModuleKind> {
    MODULES.get(id as usize).map(|m| m.kind)
}

/// Names of the modules allowed in `stage`, for error messages.
pub fn list_for_stage(stage: Stage) -> String {
    let names: Vec<_> = MODULES
        .iter()
        .filter(|m| m.allowed_in(stage))
        .map(|m| format!("\"{}\"", m.name))
        .collect();
    if names.is_empty() {
        "none yet".into()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_match_table_order() {
        for (i, m) in MODULES.iter().enumerate() {
            assert_eq!(m.kind as usize, i, "{}", m.name);
            for inp in m.inputs {
                assert_eq!(inp.default.ty(), inp.ty, "{}.{}", m.name, inp.name);
            }
        }
    }
}
