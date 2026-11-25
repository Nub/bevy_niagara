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
}

pub struct InputDef {
    pub name: &'static str,
    pub ty: ValueType,
    pub default: Value,
    pub doc: &'static str,
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
        ],
        doc: "Adds a divergence-free curl-noise force sampled at the particle position.",
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
];

pub fn by_name(name: &str) -> Option<&'static ModuleDef> {
    MODULES.iter().find(|m| m.name == name)
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
