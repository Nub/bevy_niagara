//! The effect source model: what a `.fx.ron` file holds.
//!
//! A **System** holds **Emitters**; each stage is an ordered stack of module
//! calls; each module call sets typed inputs. This mirrors Niagara's System /
//! Emitter / Particle Spawn and Update stacks.

use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

pub use crate::render::{
    BlendMode, FacingMode, MeshBlendMode, MeshFacingMode, RibbonFacingMode, RibbonLinkOrderMode,
    RibbonUvMode, SortMode, SpriteAlignment,
};
use crate::types::{Curve, Value, ValueType};

/// A 1-based line/column position in a source file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

/// A Niagara stage (script). `ParticleEvent` is the module stack of an event
/// handler (Niagara's Event Handler script).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    SystemSpawn,
    SystemUpdate,
    EmitterSpawn,
    EmitterUpdate,
    ParticleSpawn,
    ParticleUpdate,
    ParticleEvent,
}

impl Stage {
    pub const ALL: [Stage; 7] = [
        Stage::SystemSpawn,
        Stage::SystemUpdate,
        Stage::EmitterSpawn,
        Stage::EmitterUpdate,
        Stage::ParticleSpawn,
        Stage::ParticleUpdate,
        Stage::ParticleEvent,
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
            Stage::ParticleEvent => "event_handlers[..].modules",
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
            Stage::ParticleEvent => "Event Handler",
        }
    }

    pub fn is_particle(self) -> bool {
        matches!(
            self,
            Stage::ParticleSpawn | Stage::ParticleUpdate | Stage::ParticleEvent
        )
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
    /// Modules written in WGSL ([`crate::custom`]), usable in any emitter.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub custom_modules: Vec<crate::custom::CustomModuleSource>,
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
    /// Niagara event handlers: what this emitter does with other emitters' (or its own) events.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub event_handlers: Vec<crate::events::EventHandlerSource>,
    /// How particles are drawn; empty means the emitter draws nothing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub renderers: Vec<RendererSource>,
    /// Grid collections (Grid2D / Grid3D) the simulation stages and custom modules use.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub grids: Vec<crate::custom::GridSource>,
    /// Extra passes per step over the particles or a grid, after the update.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub simulation_stages: Vec<crate::custom::SimStageSource>,
    #[serde(skip)]
    pub span: Span,
}

/// One entry of an emitter's `renderers: [ ... ]`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum RendererSource {
    /// `Sprite(( ... ))`: camera-facing quads (Niagara's Sprite Renderer).
    Sprite(SpriteSource),
    /// `Ribbon(( ... ))`: strips joining particles (Niagara's Ribbon Renderer).
    Ribbon(RibbonSource),
    /// `Mesh(( ... ))`: one mesh per particle (Niagara's Mesh Renderer).
    Mesh(MeshSource),
    /// `Light(( ... ))`: one point light per particle (Niagara's Light Renderer).
    Light(LightSource),
}

impl RendererSource {
    /// Where the renderer is written.
    pub fn span(&self) -> Span {
        match self {
            RendererSource::Sprite(s) => s.span,
            RendererSource::Ribbon(r) => r.span,
            RendererSource::Mesh(m) => m.span,
            RendererSource::Light(l) => l.span,
        }
    }
}

/// Sprite renderer settings, with Niagara's defaults filled in for fields
/// the source leaves out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpriteSource {
    pub blend_mode: BlendMode,
    pub lit: bool,
    pub facing_mode: FacingMode,
    pub custom_facing_vector: [f32; 3],
    pub alignment: SpriteAlignment,
    pub custom_alignment_vector: [f32; 3],
    pub pivot_in_uv_space: [f32; 2],
    /// Flipbook (columns, rows), each at least 1.
    pub sub_image_size: [u32; 2],
    pub sub_uv_blending: bool,
    pub sort_mode: SortMode,
    /// Metres; 0 disables soft-particle fading.
    pub soft_particle_fade_distance: f32,
    /// Visible to Solari's rays (reflections, bounce light, shadows); emissive
    /// sprites light the scene. Off by default, as in Niagara.
    pub ray_tracing: bool,
    /// Asset path; `None` uses the built-in soft round sprite.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub material: Option<String>,
    #[serde(skip)]
    pub span: Span,
}

impl Default for SpriteSource {
    fn default() -> Self {
        SpriteSource {
            blend_mode: BlendMode::default(),
            lit: false,
            facing_mode: FacingMode::default(),
            custom_facing_vector: [0.0, 0.0, 1.0],
            alignment: SpriteAlignment::default(),
            custom_alignment_vector: [0.0, 1.0, 0.0],
            pivot_in_uv_space: [0.5, 0.5],
            sub_image_size: [1, 1],
            sub_uv_blending: false,
            sort_mode: SortMode::default(),
            soft_particle_fade_distance: 0.0,
            ray_tracing: false,
            texture: None,
            material: None,
            span: Span::default(),
        }
    }
}

/// Ribbon renderer settings, with defaults filled in for fields the source
/// leaves out. A ribbon joins the particles of each `Particles.RibbonID`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RibbonSource {
    pub blend_mode: BlendMode,
    pub facing_mode: RibbonFacingMode,
    pub custom_facing_vector: [f32; 3],
    pub uv_mode: RibbonUvMode,
    /// Metres of ribbon per texture repeat (`TiledOverRibbonLength`); more than 0.
    pub tiling_length: f32,
    pub link_order: RibbonLinkOrderMode,
    /// Subdivisions per segment, 1..=16.
    pub tessellation_factor: u32,
    /// Visible to Solari's rays (reflections, bounce light, shadows); the
    /// (emissive) ribbon lights the scene. Off by default, as in Niagara.
    pub ray_tracing: bool,
    /// Asset path; `None` draws plain colour.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub material: Option<String>,
    #[serde(skip)]
    pub span: Span,
}

impl Default for RibbonSource {
    fn default() -> Self {
        RibbonSource {
            blend_mode: BlendMode::default(),
            facing_mode: RibbonFacingMode::default(),
            custom_facing_vector: [0.0, 0.0, 1.0],
            uv_mode: RibbonUvMode::default(),
            tiling_length: 1.0,
            link_order: RibbonLinkOrderMode::default(),
            tessellation_factor: 1,
            ray_tracing: false,
            texture: None,
            material: None,
            span: Span::default(),
        }
    }
}

/// Mesh renderer settings, with defaults filled in for fields the source
/// leaves out. Each particle draws the mesh at `Particles.Position`, scaled by
/// `Particles.Scale` and rotated by `Particles.MeshOrientation`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeshSource {
    /// A name from [`MESH_BUILTINS`](crate::render::MESH_BUILTINS) or an
    /// asset path (contains a '.' or '/').
    pub mesh: String,
    pub blend_mode: MeshBlendMode,
    pub facing_mode: MeshFacingMode,
    pub lit: bool,
    pub sort_mode: SortMode,
    /// Asset path; `None` draws plain colour.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_path"
    )]
    pub material: Option<String>,
    #[serde(skip)]
    pub span: Span,
}

impl Default for MeshSource {
    fn default() -> Self {
        MeshSource {
            mesh: "cube".into(),
            blend_mode: MeshBlendMode::default(),
            facing_mode: MeshFacingMode::default(),
            lit: true,
            sort_mode: SortMode::default(),
            texture: None,
            material: None,
            span: Span::default(),
        }
    }
}

/// Light renderer settings, with defaults filled in for fields the source
/// leaves out. Each particle is a point light of radius
/// `Particles.LightRadius * radius_scale`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LightSource {
    /// Multiplies `Particles.LightRadius`; at least 0.
    pub radius_scale: f32,
    /// Added to the particle colour before it becomes the light colour.
    pub color_add: [f32; 3],
    /// Lumens per unit of colour; at least 0.
    pub intensity: f32,
    /// Whether the particle alpha multiplies the brightness.
    pub alpha_scales_brightness: bool,
    /// Most lights drawn per emitter, 1..=256.
    pub max_lights: u32,
    #[serde(skip)]
    pub span: Span,
}

impl Default for LightSource {
    fn default() -> Self {
        LightSource {
            radius_scale: 1.0,
            color_add: [0.0; 3],
            intensity: 10000.0,
            alpha_scales_brightness: true,
            max_lights: 16,
            span: Span::default(),
        }
    }
}

/// Written as a plain string (the source form), not `Some("...")`.
fn serialize_path<S: Serializer>(t: &Option<String>, s: S) -> Result<S::Ok, S::Error> {
    match t {
        Some(p) => s.serialize_str(p),
        None => s.serialize_none(),
    }
}

impl EmitterSource {
    pub fn stage(&self, stage: Stage) -> &[ModuleCall] {
        match stage {
            Stage::EmitterSpawn => &self.emitter_spawn,
            Stage::EmitterUpdate => &self.emitter_update,
            Stage::ParticleSpawn => &self.particle_spawn,
            Stage::ParticleUpdate => &self.particle_update,
            Stage::SystemSpawn | Stage::SystemUpdate | Stage::ParticleEvent => &[],
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

/// `User.` parameter declarations: value parameters (`.0`) and data
/// interfaces (`.1`, see [`crate::data`]); serialises as one RON map of
/// name → literal or data interface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamMap(pub Vec<ParamEntry>, pub Vec<crate::data::DataInterfaceSource>);

#[derive(Clone, Debug, PartialEq)]
pub struct ParamEntry {
    pub name: String,
    pub value: Value,
    pub span: Span,
}

impl Serialize for ParamMap {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len() + self.1.len()))?;
        for e in &self.0 {
            m.serialize_entry(&e.name, &e.value)?;
        }
        for di in &self.1 {
            m.serialize_entry(&di.name, di)?;
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
    /// `Box`: a bare name, the value of a static-switch input (an enum such
    /// as Shape Location's "Shape Primitive"); cooks to its index.
    Choice(String),
    /// `Dynamic("Multiply Vec3 by Float", { "A": ..., "B": ... })`: a Niagara
    /// dynamic input from [`crate::dynamic::DYNAMIC_INPUTS`]; its inputs nest.
    Dynamic(DynamicCall),
    /// `Link(SkeletalMesh, "User.Character")`: a data interface declared in
    /// `user_parameters`, given to a module input of that type.
    DataInterface(crate::data::DiType, String),
    /// `[Float(1.0), Float(2.0)]`: the list input of a dynamic input
    /// (Select ... by Int, Random ... from List).
    List(Vec<InputEntry>),
    /// `Float([(0.0, 1.0), (1.0, 0.0)])`: a bare curve, the "Curve" input of
    /// a "... from Curve" dynamic input.
    Curve(Curve),
}

/// A dynamic input with its own inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicCall {
    pub name: String,
    pub inputs: InputMap,
    pub span: Span,
}

impl Input {
    pub fn ty(&self) -> ValueType {
        match self {
            Input::Literal(v) => v.ty(),
            Input::Link(t, _) => *t,
            Input::RandomRange(a, _) => a.ty(),
            Input::CurveOverAge(c) => c.ty(),
            Input::Choice(_) => ValueType::Int,
            Input::DataInterface(..) => ValueType::Int,
            Input::Dynamic(c) => crate::dynamic::by_name(&c.name).map_or(ValueType::Float, |(_, d)| d.ty),
            Input::List(items) => items.first().map_or(ValueType::Float, |e| e.input.ty()),
            Input::Curve(c) => c.ty(),
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
            // A bare identifier in RON; serde needs it 'static, so take it from the module table.
            Input::Choice(name) => match crate::modules::static_choice(name) {
                Some(c) => s.serialize_unit_variant("Input", 4, c),
                None => s.serialize_str(name),
            },
            Input::Dynamic(c) => {
                let mut tv = s.serialize_tuple_variant("Input", 5, "Dynamic", 2)?;
                tv.serialize_field(&c.name)?;
                tv.serialize_field(&c.inputs)?;
                tv.end()
            }
            Input::DataInterface(t, p) => {
                let mut tv = s.serialize_tuple_variant("Input", 1, "Link", 2)?;
                tv.serialize_field(t)?;
                tv.serialize_field(p)?;
                tv.end()
            }
            Input::List(items) => s.collect_seq(items.iter().map(|e| &e.input)),
            Input::Curve(c) => c.serialize(s),
        }
    }
}
