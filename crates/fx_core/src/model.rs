//! The effect source model: what a `.fx.ron` file holds.
//!
//! A **System** holds **Emitters**; each stage is an ordered stack of module
//! calls; each module call sets typed inputs. This mirrors Niagara's System /
//! Emitter / Particle Spawn and Update stacks.

use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

use crate::types::{Curve, Value, ValueType};

/// A 1-based line/column position in a source file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

/// A Niagara stage (script). Events are reserved for later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    SystemSpawn,
    SystemUpdate,
    EmitterSpawn,
    EmitterUpdate,
    ParticleSpawn,
    ParticleUpdate,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::SystemSpawn,
        Stage::SystemUpdate,
        Stage::EmitterSpawn,
        Stage::EmitterUpdate,
        Stage::ParticleSpawn,
        Stage::ParticleUpdate,
    ];

    /// The field name in `.fx.ron`.
    pub fn field(self) -> &'static str {
        match self {
            Stage::SystemSpawn => "system_spawn",
            Stage::SystemUpdate => "system_update",
            Stage::EmitterSpawn => "emitter_spawn",
            Stage::EmitterUpdate => "emitter_update",
            Stage::ParticleSpawn => "particle_spawn",
            Stage::ParticleUpdate => "particle_update",
        }
    }

    /// Niagara's display name.
    pub fn niagara_name(self) -> &'static str {
        match self {
            Stage::SystemSpawn => "System Spawn",
            Stage::SystemUpdate => "System Update",
            Stage::EmitterSpawn => "Emitter Spawn",
            Stage::EmitterUpdate => "Emitter Update",
            Stage::ParticleSpawn => "Particle Spawn",
            Stage::ParticleUpdate => "Particle Update",
        }
    }

    pub fn is_particle(self) -> bool {
        matches!(self, Stage::ParticleSpawn | Stage::ParticleUpdate)
    }
    pub fn is_emitter(self) -> bool {
        matches!(self, Stage::EmitterSpawn | Stage::EmitterUpdate)
    }
}

/// One effect file: a Niagara System.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemSource {
    pub name: String,
    /// Exposed `User.` parameters with their default values.
    pub user_parameters: ParamMap,
    pub system_spawn: Vec<ModuleCall>,
    pub system_update: Vec<ModuleCall>,
    pub emitters: Vec<EmitterSource>,
    #[serde(skip)]
    pub span: Span,
}

/// Where an emitter simulates (Niagara's Sim Target).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum SimTarget {
    Cpu,
    #[default]
    Gpu,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmitterSource {
    pub name: String,
    pub sim_target: SimTarget,
    /// Capacity: spawns beyond it are dropped.
    pub max_particles: u32,
    pub emitter_spawn: Vec<ModuleCall>,
    pub emitter_update: Vec<ModuleCall>,
    pub particle_spawn: Vec<ModuleCall>,
    pub particle_update: Vec<ModuleCall>,
    #[serde(skip)]
    pub span: Span,
}

impl EmitterSource {
    pub fn stage(&self, stage: Stage) -> &[ModuleCall] {
        match stage {
            Stage::EmitterSpawn => &self.emitter_spawn,
            Stage::EmitterUpdate => &self.emitter_update,
            Stage::ParticleSpawn => &self.particle_spawn,
            Stage::ParticleUpdate => &self.particle_update,
            Stage::SystemSpawn | Stage::SystemUpdate => &[],
        }
    }
}

/// A module placed in a stage, e.g. `(module: "Spawn Rate", inputs: { "Spawn Rate": Float(20.0) })`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModuleCall {
    pub module: String,
    pub enabled: bool,
    pub inputs: InputMap,
    #[serde(skip)]
    pub span: Span,
}

/// Ordered `name: input` entries; serialises as a RON map.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputMap(pub Vec<InputEntry>);

#[derive(Clone, Debug, PartialEq)]
pub struct InputEntry {
    pub name: String,
    pub input: Input,
    pub span: Span,
}

impl Serialize for InputMap {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for e in &self.0 {
            m.serialize_entry(&e.name, &e.input)?;
        }
        m.end()
    }
}

/// `User.` parameter declarations; serialises as a RON map of name → literal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamMap(pub Vec<ParamEntry>);

#[derive(Clone, Debug, PartialEq)]
pub struct ParamEntry {
    pub name: String,
    pub value: Value,
    pub span: Span,
}

impl Serialize for ParamMap {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for e in &self.0 {
            m.serialize_entry(&e.name, &e.value)?;
        }
        m.end()
    }
}

/// A module input: a literal or a dynamic input. Every form names its type.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// `Float(1.0)`, `Vec3((0.0, 1.0, 0.0))`, ...
    Literal(Value),
    /// `Link(Vec3, "User.Wind")`: reads a parameter.
    Link(ValueType, String),
    /// `RandomRange(Float(1.0), Float(2.0))`: uniform per component.
    RandomRange(Value, Value),
    /// `CurveOverAge(Color([(0.0, (1,1,1,1)), (1.0, (1,1,1,0))]))`:
    /// samples the curve at `Particles.NormalizedAge`.
    CurveOverAge(Curve),
}

impl Input {
    pub fn ty(&self) -> ValueType {
        match self {
            Input::Literal(v) => v.ty(),
            Input::Link(t, _) => *t,
            Input::RandomRange(a, _) => a.ty(),
            Input::CurveOverAge(c) => c.ty(),
        }
    }
}

impl Serialize for Input {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTupleVariant;
        match self {
            Input::Literal(v) => v.serialize(s),
            Input::Link(t, p) => {
                let mut tv = s.serialize_tuple_variant("Input", 1, "Link", 2)?;
                tv.serialize_field(t)?;
                tv.serialize_field(p)?;
                tv.end()
            }
            Input::RandomRange(a, b) => {
                let mut tv = s.serialize_tuple_variant("Input", 2, "RandomRange", 2)?;
                tv.serialize_field(a)?;
                tv.serialize_field(b)?;
                tv.end()
            }
            Input::CurveOverAge(c) => s.serialize_newtype_variant("Input", 3, "CurveOverAge", c),
        }
    }
}
