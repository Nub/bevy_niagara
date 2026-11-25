//! Validation of the effect model and cooking to the runtime asset.
//!
//! Cooking resolves module names to ids, input names to slots (with defaults
//! filled in), and parameter links to [`ParamRef`]s, then serialises with
//! rkyv so the runtime can read the asset in place (zero-copy).

use crate::diag::{Diags, suggest};
use crate::dynamic::{self, ArgDefault, ArgKind};
use crate::events::{EventExecutionMode, EventKind};
use crate::model::*;
use crate::modules::{self, ModuleDef, ModuleKind};
use crate::params::{self, Namespace, ParamRef, attr};
use crate::types::{Value, ValueType};

/// Bumped whenever the cooked layout changes.
pub const COOKED_FORMAT_VERSION: u32 = 8;
/// File extension of cooked effects.
pub const COOKED_EXTENSION: &str = "fxb";

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedSystem {
    pub format_version: u32,
    pub name: String,
    pub user_params: Vec<CookedParam>,
    /// Data interfaces (`crate::data`), in declaration order; modules refer to
    /// them by index.
    pub data_interfaces: Vec<crate::data::CookedDataInterface>,
    pub emitters: Vec<CookedEmitter>,
    /// Modules written in WGSL; a [`CookedModule`] whose id is
    /// [`crate::custom::custom_id`]`(i)` calls entry `i`.
    pub custom_modules: Vec<crate::custom::CookedCustomModule>,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedParam {
    pub name: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedEmitter {
    pub name: String,
    pub gpu: bool,
    pub max_particles: u32,
    pub emitter_spawn: Vec<CookedModule>,
    pub emitter_update: Vec<CookedModule>,
    pub particle_spawn: Vec<CookedModule>,
    pub particle_update: Vec<CookedModule>,
    /// How the particles are drawn, in source order; empty draws nothing.
    pub renderers: Vec<CookedRenderer>,
    /// Event handlers, in source order.
    pub event_handlers: Vec<CookedEventHandler>,
    /// Grid collections, in source order.
    pub grids: Vec<crate::custom::CookedGrid>,
    /// Simulation stages, run after the spawn in source order.
    pub stages: Vec<crate::custom::CookedSimStage>,
    /// A custom module writes an extended attribute (see [`Self::uses_extended_attributes`]).
    pub custom_extended: bool,
}

/// An event handler with its source resolved.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedEventHandler {
    /// Index of the source emitter.
    pub source: u32,
    /// [`EventKind::index`].
    pub event: u8,
    /// EveryParticle (else SpawnedParticles).
    pub every_particle: bool,
    pub spawn_number: u32,
    /// 0 = every event.
    pub max_events: u32,
    pub modules: Vec<CookedModule>,
}

impl CookedEventHandler {
    pub fn kind(&self) -> EventKind {
        EventKind::from_index(self.event)
    }
}

impl CookedEmitter {
    /// Whether the emitter reads or writes any particle attribute from
    /// [`attr::FIRST_EXTENDED`] on (Scale, MeshOrientation, the Ribbon
    /// attributes, LightRadius, DynamicMaterialParameter), or has a renderer
    /// other than Sprite (those read them). Emitters that do not store only
    /// the base attributes.
    pub fn uses_extended_attributes(&self) -> bool {
        let renders = self
            .renderers
            .iter()
            .any(|r| !matches!(r, CookedRenderer::Sprite(_)));
        let module = |m: &CookedModule| {
            matches!(
                modules::from_id(m.module),
                Some(
                    ModuleKind::InitializeRibbon
                        | ModuleKind::InitializeMesh
                        | ModuleKind::UpdateMeshOrientation
                        | ModuleKind::InitializeLight
                        | ModuleKind::DynamicMaterialParameters
                        | ModuleKind::InitialMeshOrientation
                        | ModuleKind::SubUvAnimation
                        | ModuleKind::ScaleMeshSize
                        | ModuleKind::ScaleRibbonWidth
                        | ModuleKind::CameraOffset
                )
            ) || m.all_inputs().any(|i| {
                matches!(i, CookedInput::Link(r)
                    if r.ns == Namespace::Particles && r.index >= attr::FIRST_EXTENDED)
            })
        };
        renders || self.custom_extended || self.particle_modules().any(module)
    }

    /// Every particle-stage module: spawn, update and the event handlers'.
    pub fn particle_modules(&self) -> impl Iterator<Item = &CookedModule> {
        self.particle_spawn
            .iter()
            .chain(&self.particle_update)
            .chain(self.event_handlers.iter().flat_map(|h| h.modules.iter()))
            .chain(crate::custom::particle_stage_modules(&self.stages))
    }

    /// Whether a Collision module reads the scene (SceneDepth or RayTraced).
    pub fn uses_scene_collision(&self) -> bool {
        self.particle_update.iter().any(|m| {
            modules::from_id(m.module) == Some(ModuleKind::Collision)
                && !matches!(m.inputs.first(), Some(CookedInput::Const(Value::Int(0))))
        })
    }

    /// Whether the particle update writes events of `kind`.
    pub fn generates(&self, kind: EventKind) -> bool {
        let generator = match kind {
            EventKind::Death => ModuleKind::GenerateDeathEvent,
            EventKind::Collision => ModuleKind::GenerateCollisionEvent,
            EventKind::Location => ModuleKind::GenerateLocationEvent,
        };
        self.particle_update
            .iter()
            .any(|m| modules::from_id(m.module) == Some(generator))
    }

    /// Whether a module sets Particles.SubImageIndex (SubUV Animation): the
    /// sprite renderer then shows that frame instead of playing the flipbook
    /// over the particle's life.
    pub fn sets_sub_image_index(&self) -> bool {
        self.particle_modules()
            .any(|m| modules::from_id(m.module) == Some(ModuleKind::SubUvAnimation))
    }

    /// Flipbook frames of the first Sprite renderer (columns * rows), 1 without one:
    /// what SubUV Animation plays when its Number of Frames is 0.
    pub fn sub_image_frames(&self) -> u32 {
        self.renderers
            .iter()
            .find_map(|r| match r {
                CookedRenderer::Sprite(s) => {
                    Some(s.sub_image_size[0].max(1) * s.sub_image_size[1].max(1))
                }
                _ => None,
            })
            .unwrap_or(1)
    }
}

/// A renderer with every setting resolved.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum CookedRenderer {
    Sprite(CookedSprite),
    Ribbon(CookedRibbon),
    Mesh(CookedMesh),
    Light(CookedLight),
}

/// Sprite renderer settings (Niagara's Sprite Renderer properties).
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedSprite {
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
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    pub material: Option<String>,
}

impl Default for CookedSprite {
    fn default() -> Self {
        cook_sprite(&SpriteSource::default())
    }
}

fn cook_sprite(s: &SpriteSource) -> CookedSprite {
    CookedSprite {
        blend_mode: s.blend_mode,
        lit: s.lit,
        facing_mode: s.facing_mode,
        custom_facing_vector: s.custom_facing_vector,
        alignment: s.alignment,
        custom_alignment_vector: s.custom_alignment_vector,
        pivot_in_uv_space: s.pivot_in_uv_space,
        sub_image_size: s.sub_image_size,
        sub_uv_blending: s.sub_uv_blending,
        sort_mode: s.sort_mode,
        soft_particle_fade_distance: s.soft_particle_fade_distance,
        ray_tracing: s.ray_tracing,
        texture: s.texture.clone(),
        material: s.material.clone(),
    }
}

/// Ribbon renderer settings (Niagara's Ribbon Renderer properties). Each
/// `Particles.RibbonID` is one ribbon.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedRibbon {
    pub blend_mode: BlendMode,
    pub facing_mode: RibbonFacingMode,
    pub custom_facing_vector: [f32; 3],
    pub uv_mode: RibbonUvMode,
    /// Metres of ribbon per texture repeat (`TiledOverRibbonLength`).
    pub tiling_length: f32,
    pub link_order: RibbonLinkOrderMode,
    /// Subdivisions per segment, 1..=16.
    pub tessellation_factor: u32,
    /// In Solari's ray-traced scene (needs fx_solari).
    pub ray_tracing: bool,
    /// Asset path; `None` draws plain colour.
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    pub material: Option<String>,
}

impl Default for CookedRibbon {
    fn default() -> Self {
        cook_ribbon(&RibbonSource::default())
    }
}

fn cook_ribbon(r: &RibbonSource) -> CookedRibbon {
    CookedRibbon {
        blend_mode: r.blend_mode,
        facing_mode: r.facing_mode,
        custom_facing_vector: r.custom_facing_vector,
        uv_mode: r.uv_mode,
        tiling_length: r.tiling_length,
        link_order: r.link_order,
        tessellation_factor: r.tessellation_factor,
        ray_tracing: r.ray_tracing,
        texture: r.texture.clone(),
        material: r.material.clone(),
    }
}

/// Mesh renderer settings (Niagara's Mesh Renderer properties).
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedMesh {
    /// A name from [`MESH_BUILTINS`](crate::render::MESH_BUILTINS) or an asset path.
    pub mesh: String,
    pub blend_mode: MeshBlendMode,
    pub facing_mode: MeshFacingMode,
    pub lit: bool,
    pub sort_mode: SortMode,
    /// Asset path; `None` draws plain colour.
    pub texture: Option<String>,
    /// Asset path of a custom WESL fragment shader; `None` uses the built-in one.
    pub material: Option<String>,
}

impl Default for CookedMesh {
    fn default() -> Self {
        cook_mesh(&MeshSource::default())
    }
}

fn cook_mesh(m: &MeshSource) -> CookedMesh {
    CookedMesh {
        mesh: m.mesh.clone(),
        blend_mode: m.blend_mode,
        facing_mode: m.facing_mode,
        lit: m.lit,
        sort_mode: m.sort_mode,
        texture: m.texture.clone(),
        material: m.material.clone(),
    }
}

/// Light renderer settings (Niagara's Light Renderer properties).
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedLight {
    /// Multiplies `Particles.LightRadius`.
    pub radius_scale: f32,
    /// Added to the particle colour.
    pub color_add: [f32; 3],
    /// Lumens per unit of colour.
    pub intensity: f32,
    pub alpha_scales_brightness: bool,
    /// Most lights drawn per emitter, 1..=256.
    pub max_lights: u32,
}

impl Default for CookedLight {
    fn default() -> Self {
        cook_light(&LightSource::default())
    }
}

fn cook_light(l: &LightSource) -> CookedLight {
    CookedLight {
        radius_scale: l.radius_scale,
        color_add: l.color_add,
        intensity: l.intensity,
        alpha_scales_brightness: l.alpha_scales_brightness,
        max_lights: l.max_lights,
    }
}

fn cook_renderer(r: &RendererSource, d: &mut Diags) -> CookedRenderer {
    // The decoder rejects bad values; models built in code may still hold them.
    match r {
        RendererSource::Sprite(s) => {
            if s.sub_image_size.contains(&0) {
                d.error(
                    s.span,
                    "`sub_image_size` columns and rows must be at least 1",
                    Some("write e.g. `sub_image_size: (1, 1)` (no flipbook)".into()),
                );
            }
            if s.soft_particle_fade_distance.is_nan() || s.soft_particle_fade_distance < 0.0 {
                d.error(
                    s.span,
                    "`soft_particle_fade_distance` must be at least 0 metres",
                    Some("write `soft_particle_fade_distance: 0.0` to turn soft fading off".into()),
                );
            }
            CookedRenderer::Sprite(cook_sprite(s))
        }
        RendererSource::Ribbon(r) => {
            if !(r.tiling_length > 0.0 && r.tiling_length.is_finite()) {
                d.error(
                    r.span,
                    "`tiling_length` must be more than 0 metres",
                    Some("write e.g. `tiling_length: 1.0`".into()),
                );
            }
            if !(1..=16).contains(&r.tessellation_factor) {
                d.error(
                    r.span,
                    "`tessellation_factor` must be from 1 to 16",
                    Some("write `tessellation_factor: 1` (no subdivision)".into()),
                );
            }
            CookedRenderer::Ribbon(cook_ribbon(r))
        }
        RendererSource::Mesh(m) => {
            if let Some((msg, help)) = crate::decode::mesh_name_error(&m.mesh) {
                d.error(m.span, msg, Some(help));
            }
            CookedRenderer::Mesh(cook_mesh(m))
        }
        RendererSource::Light(l) => {
            if !(l.radius_scale >= 0.0 && l.radius_scale.is_finite()) {
                d.error(
                    l.span,
                    "`radius_scale` must be at least 0",
                    Some("write `radius_scale: 1.0`".into()),
                );
            }
            if !(l.intensity >= 0.0 && l.intensity.is_finite()) {
                d.error(
                    l.span,
                    "`intensity` must be at least 0 lumens per unit colour",
                    Some("write e.g. `intensity: 10000.0`".into()),
                );
            }
            if !(1..=256).contains(&l.max_lights) {
                d.error(
                    l.span,
                    "`max_lights` must be from 1 to 256",
                    Some("write e.g. `max_lights: 16`".into()),
                );
            }
            CookedRenderer::Light(cook_light(l))
        }
    }
}

/// An enabled module with one input per [`ModuleDef::inputs`] slot, in order.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedModule {
    pub module: u16,
    pub inputs: Vec<CookedInput>,
    /// The dynamic inputs used by `inputs` (and by each other), referred to by
    /// [`CookedInput::Dynamic`] index. A parent comes before its children.
    pub dynamics: Vec<CookedDynamic>,
}

impl CookedModule {
    /// Every input, the module's own and its dynamic inputs'.
    pub fn all_inputs(&self) -> impl Iterator<Item = &CookedInput> {
        self.inputs.iter().chain(self.dynamics.iter().flat_map(|n| n.args.iter()))
    }
}

/// A dynamic input: [`crate::dynamic::DYNAMIC_INPUTS`] index and one input per
/// [`crate::dynamic::DynamicDef::args`] entry, in order; a list input
/// (always last) gives one input per item.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedDynamic {
    pub def: u16,
    pub args: Vec<CookedInput>,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum CookedInput {
    Const(Value),
    Link(ParamRef),
    Random(Value, Value),
    Curve(CookedCurve),
    /// Index into the module's [`CookedModule::dynamics`].
    Dynamic(u16),
}

/// Piecewise-linear curve sampled at `Particles.NormalizedAge`.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedCurve {
    pub ty: ValueType,
    pub times: Vec<f32>,
    pub values: Vec<[f32; 4]>,
}

impl CookedCurve {
    pub fn sample(&self, t: f32) -> [f32; 4] {
        let n = self.times.len();
        if n == 0 {
            return [0.0; 4];
        }
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[n - 1] {
            return self.values[n - 1];
        }
        let i = self.times.partition_point(|&k| k <= t); // first key after t; 1..n
        let (t0, t1) = (self.times[i - 1], self.times[i]);
        let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        let (a, b) = (self.values[i - 1], self.values[i]);
        std::array::from_fn(|c| a[c] + (b[c] - a[c]) * f)
    }
}

/// Serialises a cooked system to bytes.
pub fn to_bytes(sys: &CookedSystem) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(sys)
        .expect("serialising a cooked system cannot fail")
        .into_vec()
}

/// Validates cooked bytes and gives zero-copy access. `bytes` must be
/// 16-byte aligned (see [`load_aligned`]).
pub fn access(bytes: &[u8]) -> Result<&ArchivedCookedSystem, String> {
    let sys = rkyv::access::<ArchivedCookedSystem, rkyv::rancor::Error>(bytes)
        .map_err(|e| format!("not a valid cooked effect: {e}"))?;
    if sys.format_version != COOKED_FORMAT_VERSION {
        return Err(format!(
            "cooked effect has format version {}, this build reads version {COOKED_FORMAT_VERSION}; cook it again",
            sys.format_version
        ));
    }
    Ok(sys)
}

/// Copies bytes (e.g. read from a file) into an aligned buffer for [`access`].
pub fn load_aligned(bytes: &[u8]) -> rkyv::util::AlignedVec {
    let mut v = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
    v.extend_from_slice(bytes);
    v
}

/// Deserialises cooked bytes into an owned [`CookedSystem`].
pub fn from_bytes(bytes: &[u8]) -> Result<CookedSystem, String> {
    let aligned = load_aligned(bytes);
    let archived = access(&aligned)?;
    rkyv::deserialize::<CookedSystem, rkyv::rancor::Error>(archived).map_err(|e| e.to_string())
}

/// Validates the model and compiles it. Returns `None` if there are errors
/// (they are in `d`); warnings are added to `d` either way.
pub fn compile(src: &SystemSource, d: &mut Diags) -> Option<CookedSystem> {
    let mut user_params = Vec::new();
    for p in &src.user_parameters.0 {
        match params::split_namespace(&p.name) {
            Some((Namespace::User, _)) => {}
            _ => {
                let rest = p.name.split_once('.').map_or(p.name.as_str(), |(_, r)| r);
                d.error(
                    p.span,
                    format!(
                        "user parameter \"{}\" must be in the User namespace",
                        p.name
                    ),
                    Some(format!(
                        "rename it to \"User.{rest}\" (and link it as `Link({}, \"User.{rest}\")`)",
                        p.value.ty().name()
                    )),
                );
            }
        }
        if user_params.iter().any(|q: &CookedParam| q.name == p.name) {
            d.error(
                p.span,
                format!("user parameter \"{}\" is declared twice", p.name),
                Some("keep one declaration".into()),
            );
        }
        user_params.push(CookedParam {
            name: p.name.clone(),
            value: p.value,
        });
    }

    let data_interfaces = cook_data_interfaces(src, &user_params, d);
    let custom_modules = crate::custom::cook_declarations(src, d);
    let cx = Ctx {
        user: &user_params,
        dis: &data_interfaces,
        custom: &custom_modules,
    };
    if src.name.trim().is_empty() {
        d.error(
            src.span,
            "the effect's `name` is empty",
            Some("give it a name, e.g. `name: \"Fountain\"`".into()),
        );
    }
    for (stage, calls) in [
        (Stage::SystemSpawn, &src.system_spawn),
        (Stage::SystemUpdate, &src.system_update),
    ] {
        cx.stack(stage, calls, d);
    }
    if src.emitters.is_empty() {
        d.warning(
            src.span,
            "the effect has no emitters, so it shows nothing",
            Some("add an emitter to `emitters: [ ... ]`".into()),
        );
    }

    let mut emitters = Vec::new();
    for (i, e) in src.emitters.iter().enumerate() {
        if e.name.trim().is_empty() {
            d.error(
                e.span,
                "emitter `name` is empty",
                Some("give every emitter a unique name".into()),
            );
        } else if src.emitters[..i].iter().any(|o| o.name == e.name) {
            d.error(
                e.span,
                format!("two emitters are named \"{}\"", e.name),
                Some("emitter names must be unique within an effect; rename one".into()),
            );
        }
        let [es, eu, ps, pu] = [
            Stage::EmitterSpawn,
            Stage::EmitterUpdate,
            Stage::ParticleSpawn,
            Stage::ParticleUpdate,
        ]
        .map(|s| cx.stack(s, e.stage(s), d));
        if !d.has_errors() {
            lint_emitter(e, d);
        }
        emitters.push(CookedEmitter {
            name: e.name.clone(),
            gpu: e.sim_target == SimTarget::Gpu,
            max_particles: e.max_particles,
            emitter_spawn: es,
            emitter_update: eu,
            particle_spawn: ps,
            particle_update: pu,
            renderers: e.renderers.iter().map(|r| cook_renderer(r, d)).collect(),
            event_handlers: Vec::new(),
            grids: Vec::new(),
            stages: Vec::new(),
            custom_extended: false,
        });
    }
    for (e, ce) in src.emitters.iter().zip(emitters.iter_mut()) {
        ce.event_handlers = cx.handlers(src, e, d);
    }
    for (e, ce) in src.emitters.iter().zip(emitters.iter_mut()) {
        ce.grids = crate::custom::cook_grids(e, d);
        ce.stages = cx.sim_stages(e, &ce.grids, d);
        crate::custom::check_emitter(src, e, ce, &custom_modules, d);
    }
    if !d.has_errors() {
        for (e, ce) in src.emitters.iter().zip(&emitters) {
            lint_events(src, e, ce, d);
        }
    }
    if d.has_errors() {
        return None;
    }
    Some(CookedSystem {
        format_version: COOKED_FORMAT_VERSION,
        name: src.name.clone(),
        user_params,
        data_interfaces,
        emitters,
        custom_modules,
    })
}

fn enabled_kinds(calls: &[ModuleCall]) -> impl Iterator<Item = ModuleKind> + '_ {
    calls
        .iter()
        .filter(|c| c.enabled)
        .filter_map(|c| modules::by_name(&c.module))
        .map(|m| m.kind)
}

/// Warnings about events and collisions that cook but probably do not do
/// what was meant.
fn lint_events(src: &SystemSource, e: &EmitterSource, ce: &CookedEmitter, d: &mut Diags) {
    let update: Vec<(ModuleKind, Span)> = e
        .particle_update
        .iter()
        .filter(|c| c.enabled)
        .filter_map(|c| modules::by_name(&c.module).map(|m| (m.kind, c.span)))
        .collect();
    let pos = |k: ModuleKind| update.iter().position(|(m, _)| *m == k);
    for (i, (k, span)) in update.iter().enumerate() {
        match k {
            ModuleKind::Collision => {
                if pos(ModuleKind::SolveForcesAndVelocity).is_none_or(|s| s > i) {
                    d.warning(
                        *span,
                        "\"Collision\" runs before \"Solve Forces and Velocity\"; it tests the move the solver makes, so place it after",
                        Some("move `(module: \"Collision\")` below \"Solve Forces and Velocity\"".into()),
                    );
                }
            }
            ModuleKind::GenerateCollisionEvent => {
                if pos(ModuleKind::Collision).is_none_or(|c| c > i) {
                    d.warning(
                        *span,
                        "\"Generate Collision Event\" has no \"Collision\" module before it, so it never fires",
                        Some("add `(module: \"Collision\")` above it".into()),
                    );
                }
            }
            _ => {}
        }
    }
    for (eg, kind) in EventKind::ALL.iter().map(|k| (ce.generates(*k), *k)) {
        let received = src
            .emitters
            .iter()
            .flat_map(|o| &o.event_handlers)
            .any(|h| h.source == e.name && h.event == kind);
        if eg && !received {
            d.warning(
                e.span,
                format!(
                    "emitter \"{}\" writes {} events but no emitter handles them",
                    e.name,
                    kind.name()
                ),
                Some(format!(
                    "add `event_handlers: [(source: \"{}\", event: {}, ...)]` to the receiving emitter",
                    e.name,
                    kind.name()
                )),
            );
        }
    }
    for h in &e.event_handlers {
        if h.execution_mode == EventExecutionMode::SpawnedParticles && h.spawn_number == 0 {
            d.warning(
                h.span,
                "this SpawnedParticles handler has `spawn_number: 0`, so it spawns nothing",
                Some("set e.g. `spawn_number: 20`".into()),
            );
        }
        for c in h.modules.iter().filter(|c| c.enabled) {
            let want = match modules::by_name(&c.module).map(|m| m.kind) {
                Some(ModuleKind::ReceiveDeathEvent) => EventKind::Death,
                Some(ModuleKind::ReceiveCollisionEvent) => EventKind::Collision,
                Some(ModuleKind::ReceiveLocationEvent) => EventKind::Location,
                _ => continue,
            };
            if want != h.event {
                d.warning(
                    c.span,
                    format!(
                        "\"{}\" in a handler of {} events",
                        c.module,
                        h.event.name()
                    ),
                    Some(format!("use \"Receive {} Event\"", h.event.name())),
                );
            }
        }
    }
}

fn lint_emitter(e: &EmitterSource, d: &mut Diags) {
    let event_spawns = e
        .event_handlers
        .iter()
        .any(|h| h.execution_mode == EventExecutionMode::SpawnedParticles);
    let spawns = event_spawns || enabled_kinds(&e.emitter_update).any(|k| {
        matches!(
            k,
            ModuleKind::SpawnRate | ModuleKind::SpawnBurstInstantaneous
        )
    });
    if !spawns {
        d.warning(
            e.span,
            format!("emitter \"{}\" never spawns particles", e.name),
            Some("add `(module: \"Spawn Rate\", inputs: { \"Spawn Rate\": Float(20.0) })` or \"Spawn Burst Instantaneous\" to `emitter_update`".into()),
        );
        return;
    }
    let update: Vec<_> = enabled_kinds(&e.particle_update).collect();
    if !update.contains(&ModuleKind::ParticleState) && !update.contains(&ModuleKind::KillParticles)
    {
        d.warning(
            e.span,
            format!("emitter \"{}\" has no \"Particle State\" in `particle_update`, so particles never age or die", e.name),
            Some("add `(module: \"Particle State\")` at the top of `particle_update`".into()),
        );
    }
    let forces = update.iter().any(|k| {
        matches!(
            k,
            ModuleKind::GravityForce
                | ModuleKind::Drag
                | ModuleKind::CurlNoiseForce
                | ModuleKind::PointAttractionForce
                | ModuleKind::VortexForce
                | ModuleKind::WindForce
                | ModuleKind::LinearForce
                | ModuleKind::AccelerationForce
                | ModuleKind::PointForce
        )
    });
    let moves = enabled_kinds(&e.particle_spawn)
        .chain(update.iter().copied())
        .any(|k| {
            matches!(
                k,
                ModuleKind::AddVelocity
                    | ModuleKind::AddVelocityInCone
                    | ModuleKind::VortexVelocity
                    | ModuleKind::InheritVelocity
            )
        });
    if (forces || moves) && !update.contains(&ModuleKind::SolveForcesAndVelocity) {
        d.warning(
            e.span,
            format!("emitter \"{}\" sets velocity or forces but has no \"Solve Forces and Velocity\", so particles never move", e.name),
            Some("add `(module: \"Solve Forces and Velocity\")` at the end of `particle_update`".into()),
        );
    }
}

/// A static-switch input (`"Shape Primitive": Box`), or a bare name given to an
/// input that is not one. Static switches are fixed when the effect is
/// written, so they take a name (or its index as `Int`), never a link,
/// random range or curve. `owner` is the module or dynamic input's name.
fn choice_input(
    owner: &str,
    iname: &str,
    ty: ValueType,
    choices: &[&str],
    e: &InputEntry,
    d: &mut Diags,
) -> Option<CookedInput> {
    let names = || {
        choices
            .iter()
            .map(|c| format!("`{c}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match &e.input {
        Input::Choice(name) if choices.is_empty() => {
            d.error(
                e.span,
                format!(
                    "input \"{iname}\" of \"{owner}\" is {}, but the value given is the bare name `{name}`",
                    ty.name()
                ),
                Some(format!(
                    "give a {} value, e.g. `\"{iname}\": {}`",
                    ty.name(),
                    ty.example()
                )),
            );
            None
        }
        Input::Choice(name) => match choices.iter().position(|c| c == name) {
            Some(i) => Some(CookedInput::Const(Value::Int(i as i32))),
            None => {
                let help = match suggest(name, choices.iter().copied()) {
                    Some(s) => format!("did you mean `{s}`? Choices: {}", names()),
                    None => format!("choices: {}", names()),
                };
                d.error(
                    e.span,
                    format!("\"{iname}\" of \"{owner}\" has no choice `{name}`"),
                    Some(help),
                );
                None
            }
        },
        Input::Literal(Value::Int(i)) if (0..choices.len() as i32).contains(i) => {
            Some(CookedInput::Const(Value::Int(*i)))
        }
        _ => {
            d.error(
                e.span,
                format!("\"{iname}\" of \"{owner}\" is a static switch: give one of its names"),
                Some(format!(
                    "e.g. `\"{iname}\": {}`; choices: {}",
                    choices[0],
                    names()
                )),
            );
            None
        }
    }
}

/// Checks a curve's keys and cooks it.
fn cook_curve(c: &crate::types::Curve, span: Span, d: &mut Diags) -> Option<CookedCurve> {
    let keys = c.keys();
    if keys.is_empty() {
        d.error(
            span,
            "curve has no keys",
            Some("add at least one `(time, value)` key".into()),
        );
        return None;
    }
    if keys.windows(2).any(|w| w[1].0 < w[0].0) {
        d.error(
            span,
            "curve key times must be in ascending order",
            Some("sort the keys by their first number (time)".into()),
        );
        return None;
    }
    Some(CookedCurve {
        ty: c.ty(),
        times: keys.iter().map(|k| k.0).collect(),
        values: keys.iter().map(|k| k.1).collect(),
    })
}

/// Checks the data interfaces declared in `user_parameters` and cooks them.
fn cook_data_interfaces(
    src: &SystemSource,
    user: &[CookedParam],
    d: &mut Diags,
) -> Vec<crate::data::CookedDataInterface> {
    let mut out: Vec<crate::data::CookedDataInterface> = Vec::new();
    for di in &src.user_parameters.1 {
        match params::split_namespace(&di.name) {
            Some((Namespace::User, _)) => {}
            _ => {
                let rest = di.name.split_once('.').map_or(di.name.as_str(), |(_, r)| r);
                d.error(
                    di.span,
                    format!("data interface \"{}\" must be in the User namespace", di.name),
                    Some(format!(
                        "rename it to \"User.{rest}\" (and link it as `Link({}, \"User.{rest}\")`)",
                        di.ty.name()
                    )),
                );
            }
        }
        if out.iter().any(|o| o.name == di.name) || user.iter().any(|p| p.name == di.name) {
            d.error(
                di.span,
                format!("user parameter \"{}\" is declared twice", di.name),
                Some("keep one declaration".into()),
            );
        }
        let curve = di.curve.as_ref().and_then(|c| cook_curve(c, di.span, d));
        out.push(crate::data::CookedDataInterface {
            name: di.name.clone(),
            ty: di.ty,
            asset: di.asset.clone(),
            curve,
        });
    }
    out
}

/// Most dynamic inputs one module may hold (their random salts must stay apart).
const MAX_DYNAMICS: usize = 4096;

struct Ctx<'a> {
    user: &'a [CookedParam],
    dis: &'a [crate::data::CookedDataInterface],
    custom: &'a [crate::custom::CookedCustomModule],
}

impl Ctx<'_> {
    /// Cooks emitter `e`'s event handlers, resolving their sources.
    fn handlers(
        &self,
        src: &SystemSource,
        e: &EmitterSource,
        d: &mut Diags,
    ) -> Vec<CookedEventHandler> {
        let mut out = Vec::new();
        for h in &e.event_handlers {
            let modules = self.stack(Stage::ParticleEvent, &h.modules, d);
            let Some(si) = src.emitters.iter().position(|o| o.name == h.source) else {
                let names = src.emitters.iter().map(|o| o.name.as_str());
                let help = match suggest(&h.source, names) {
                    Some(s) => format!("did you mean \"{s}\"?"),
                    None => format!(
                        "emitters: {}",
                        src.emitters
                            .iter()
                            .map(|o| format!("\"{}\"", o.name))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                };
                d.error(
                    h.span,
                    format!("event handler source \"{}\" is not an emitter of this effect", h.source),
                    Some(help),
                );
                continue;
            };
            let source = &src.emitters[si];
            let generator = h.event.generator();
            let generates = source
                .particle_update
                .iter()
                .any(|c| c.enabled && c.module == generator);
            if !generates {
                d.error(
                    h.span,
                    format!(
                        "emitter \"{}\" writes no {} events, so this handler would never run",
                        h.source,
                        h.event.name()
                    ),
                    Some(format!(
                        "add `(module: \"{generator}\")` to the `particle_update` of \"{}\"",
                        h.source
                    )),
                );
            }
            let every = h.execution_mode == EventExecutionMode::EveryParticle;
            if every && source.name == e.name {
                d.error(
                    h.span,
                    "an EveryParticle handler cannot receive its own emitter's events",
                    Some("handle them in another emitter, or use `execution_mode: SpawnedParticles`".into()),
                );
            }
            out.push(CookedEventHandler {
                source: si as u32,
                event: h.event.index() as u8,
                every_particle: every,
                spawn_number: if every { 0 } else { h.spawn_number },
                max_events: h.max_events_per_frame,
                modules,
            });
        }
        out
    }

    fn stack(&self, stage: Stage, calls: &[ModuleCall], d: &mut Diags) -> Vec<CookedModule> {
        let mut out = Vec::new();
        for call in calls {
            if self.custom.iter().any(|c| c.name == call.module) {
                if let Some(m) = self.custom_call(stage, call, d) {
                    if call.enabled {
                        out.push(m);
                    }
                }
                continue;
            }
            let Some(def) = self.module_def(stage, call, d) else {
                continue;
            };
            let mut inputs: Vec<CookedInput> = def
                .inputs
                .iter()
                .map(|i| CookedInput::Const(i.default))
                .collect();
            let mut dynamics = Vec::new();
            for entry in &call.inputs.0 {
                let Some((slot, idef)) = def.input(&entry.name) else {
                    let names = def.inputs.iter().map(|i| i.name);
                    let help = match suggest(&entry.name, names) {
                        Some(s) => format!(
                            "did you mean \"{s}\"? Inputs of \"{}\": {}",
                            def.name,
                            def.list_inputs()
                        ),
                        None => format!("inputs of \"{}\": {}", def.name, def.list_inputs()),
                    };
                    d.error(
                        entry.span,
                        format!("module \"{}\" has no input \"{}\"", def.name, entry.name),
                        Some(help),
                    );
                    continue;
                };
                if let Some(want) = idef.di {
                    if let Some(i) = self.data_interface(def.name, idef.name, want, entry, d) {
                        inputs[slot] = CookedInput::Const(Value::Int(i as i32));
                    }
                    continue;
                }
                if !idef.choices.is_empty() || matches!(entry.input, Input::Choice(_)) {
                    if let Some(c) =
                        choice_input(def.name, idef.name, idef.ty, idef.choices, entry, d)
                    {
                        inputs[slot] = c;
                    }
                    continue;
                }
                if let Some(c) =
                    self.input(stage, def.name, idef.name, idef.ty, entry, &mut dynamics, d)
                {
                    inputs[slot] = c;
                }
            }
            for (idef, c) in def.inputs.iter().zip(&inputs) {
                if let (Some(t), CookedInput::Const(Value::Int(-1))) = (idef.di, c) {
                    let declared: Vec<_> = self
                        .dis
                        .iter()
                        .filter(|x| x.ty == t)
                        .map(|x| x.name.as_str())
                        .collect();
                    let example = declared.first().copied().unwrap_or("User.Name");
                    let help = if declared.is_empty() {
                        format!(
                            "declare one in `user_parameters` (e.g. `\"User.Name\": {}`) and add `\"{}\": Link({}, \"User.Name\")`",
                            t.example(),
                            idef.name,
                            t.name()
                        )
                    } else {
                        format!("add `\"{}\": Link({}, \"{example}\")`", idef.name, t.name())
                    };
                    d.error(
                        call.span,
                        format!(
                            "module \"{}\" needs its \"{}\" input: a {} data interface",
                            def.name,
                            idef.name,
                            t.name()
                        ),
                        Some(help),
                    );
                }
            }
            if dynamics.len() > MAX_DYNAMICS {
                d.error(
                    call.span,
                    format!(
                        "module \"{}\" uses {} dynamic inputs; at most {MAX_DYNAMICS} are allowed",
                        def.name,
                        dynamics.len()
                    ),
                    Some("split the work over several modules".into()),
                );
            }
            if call.enabled {
                out.push(CookedModule {
                    module: def.kind as u16,
                    inputs,
                    dynamics,
                });
            }
        }
        out
    }

    fn module_def(
        &self,
        stage: Stage,
        call: &ModuleCall,
        d: &mut Diags,
    ) -> Option<&'static ModuleDef> {
        let Some(def) = modules::by_name(&call.module) else {
            let help = match suggest(&call.module, modules::MODULES.iter().map(|m| m.name)) {
                Some(s) => format!(
                    "did you mean \"{s}\"? Modules for `{}`: {}",
                    stage.field(),
                    modules::list_for_stage(stage)
                ),
                None => format!(
                    "modules for `{}`: {}",
                    stage.field(),
                    modules::list_for_stage(stage)
                ),
            };
            d.error(
                call.span,
                format!("unknown module \"{}\"", call.module),
                Some(help),
            );
            return None;
        };
        if !def.allowed_in(stage) {
            let homes: Vec<_> = def
                .stages
                .iter()
                .map(|s| format!("`{}`", s.field()))
                .collect();
            d.error(
                call.span,
                format!(
                    "module \"{}\" cannot be used in `{}` ({})",
                    def.name,
                    stage.field(),
                    stage.niagara_name()
                ),
                Some(format!("move it to {}", homes.join(" or "))),
            );
            return None;
        }
        Some(def)
    }

    /// Cooks input `e` (named `iname`, of module or dynamic input `owner`),
    /// which must give a `want`. Dynamic inputs go into `dyns`.
    #[allow(clippy::too_many_arguments)]
    fn input(
        &self,
        stage: Stage,
        owner: &str,
        iname: &str,
        want: ValueType,
        e: &InputEntry,
        dyns: &mut Vec<CookedDynamic>,
        d: &mut Diags,
    ) -> Option<CookedInput> {
        match &e.input {
            Input::Dynamic(call) => return self.dynamic(stage, owner, iname, want, call, e.span, dyns, d),
            Input::DataInterface(t, _) => {
                let help = match t.curve_type() {
                    Some(ct) => format!(
                        "a {} curve is read with `Dynamic(\"{} from Curve\", {{ \"Curve\": Link({}, ...) }})`",
                        ct.name(),
                        ct.name(),
                        t.name()
                    ),
                    None => format!(
                        "give a {} value, e.g. `\"{iname}\": {}`; data interfaces go in the inputs of the modules that read them (Skeletal Mesh Location, Static Mesh Location, Sample Texture) or of the camera dynamic inputs",
                        want.name(),
                        want.example()
                    ),
                };
                d.error(
                    e.span,
                    format!(
                        "input \"{iname}\" of \"{owner}\" is {}, but a {} data interface is given",
                        want.name(),
                        t.name()
                    ),
                    Some(help),
                );
                return None;
            }
            Input::List(_) => {
                d.error(
                    e.span,
                    format!("input \"{iname}\" of \"{owner}\" takes one {}, not a list", want.name()),
                    Some("lists go only in the \"Values\" input of `Select ... by Int` and `Random ... from List` dynamic inputs".into()),
                );
                return None;
            }
            Input::Curve(c) => {
                d.error(
                    e.span,
                    format!("input \"{iname}\" of \"{owner}\" is {}, but the value given is a bare curve", want.name()),
                    Some(format!(
                        "wrap it: `CurveOverAge({}(...))` samples it over the particle's age, `Dynamic(\"{} from Curve\", {{ \"Curve\": ..., \"Curve Index\": ... }})` at any Float",
                        c.ty().name(),
                        c.ty().name()
                    )),
                );
                return None;
            }
            _ => {}
        }
        let got = e.input.ty();
        if got != want {
            d.error(
                e.span,
                format!(
                    "input \"{iname}\" of \"{owner}\" is {}, but the value given is {}",
                    want.name(),
                    got.name()
                ),
                Some(format!(
                    "give a {} value, e.g. `\"{iname}\": {}`",
                    want.name(),
                    want.example()
                )),
            );
            return None;
        }
        match &e.input {
            // Handled by `choice_input` or above.
            Input::Choice(_)
            | Input::Dynamic(_)
            | Input::List(_)
            | Input::Curve(_)
            | Input::DataInterface(..) => None,
            Input::Literal(v) => Some(CookedInput::Const(*v)),
            Input::Link(ty, target) => self
                .link(stage, *ty, target, e.span, d)
                .map(CookedInput::Link),
            Input::RandomRange(a, b) => {
                if a.ty() != b.ty() {
                    d.error(
                        e.span,
                        format!(
                            "`RandomRange` min is {} but max is {}",
                            a.ty().name(),
                            b.ty().name()
                        ),
                        Some(format!("give both as {}", want.name())),
                    );
                    return None;
                }
                if !(want.interpolable() || want == ValueType::Int) {
                    d.error(
                        e.span,
                        format!("`RandomRange` cannot produce a {}", want.name()),
                        Some(
                            "use RandomRange for Float, Int, Vec2, Vec3, Vec4 or Color inputs"
                                .into(),
                        ),
                    );
                    return None;
                }
                Some(CookedInput::Random(*a, *b))
            }
            Input::CurveOverAge(c) => {
                if !stage.is_particle() {
                    d.error(
                        e.span,
                        format!("`CurveOverAge` samples Particles.NormalizedAge, which `{}` does not have", stage.field()),
                        Some("use CurveOverAge only in `particle_spawn` or `particle_update`; here use a literal or a Link".into()),
                    );
                    return None;
                }
                cook_curve(c, e.span, d).map(CookedInput::Curve)
            }
        }
    }

    /// A `Dynamic("Name", { ... })` given to input `iname` of `owner`.
    #[allow(clippy::too_many_arguments)]
    fn dynamic(
        &self,
        stage: Stage,
        owner: &str,
        iname: &str,
        want: ValueType,
        call: &DynamicCall,
        span: Span,
        dyns: &mut Vec<CookedDynamic>,
        d: &mut Diags,
    ) -> Option<CookedInput> {
        let giving = |t: ValueType| {
            dynamic::DYNAMIC_INPUTS
                .iter()
                .filter(move |x| x.ty == t)
                .map(|x| x.name.as_str())
        };
        let Some((id, def)) = dynamic::by_name(&call.name) else {
            let all = dynamic::DYNAMIC_INPUTS.iter().map(|x| x.name.as_str());
            let list = giving(want).map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ");
            let help = match suggest(&call.name, all) {
                Some(s) => format!("did you mean \"{s}\"? Dynamic inputs giving a {}: {list}", want.name()),
                None => format!("dynamic inputs giving a {}: {list}", want.name()),
            };
            d.error(span, format!("unknown dynamic input \"{}\"", call.name), Some(help));
            return None;
        };
        if def.ty != want {
            // "Multiply Vec3" given to a Float: suggest the same name with the type swapped.
            let swapped = call.name.replace(def.ty.name(), want.name());
            let help = if dynamic::by_name(&swapped).is_some_and(|(_, x)| x.ty == want) {
                format!("use \"{swapped}\"")
            } else {
                format!(
                    "dynamic inputs giving a {}: {}",
                    want.name(),
                    giving(want).map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ")
                )
            };
            d.error(
                span,
                format!(
                    "input \"{iname}\" of \"{owner}\" is {}, but dynamic input \"{}\" gives a {}",
                    want.name(),
                    def.name,
                    def.ty.name()
                ),
                Some(help),
            );
            return None;
        }
        for entry in &call.inputs.0 {
            if def.arg(&entry.name).is_none() {
                let help = match suggest(&entry.name, def.args.iter().map(|a| a.name)) {
                    Some(s) => format!("did you mean \"{s}\"? Inputs of \"{}\": {}", def.name, def.list_args()),
                    None => format!("inputs of \"{}\": {}", def.name, def.list_args()),
                };
                d.error(
                    entry.span,
                    format!("dynamic input \"{}\" has no input \"{}\"", def.name, entry.name),
                    Some(help),
                );
            }
        }
        // Reserve this node's index before its children take theirs.
        let index = dyns.len();
        dyns.push(CookedDynamic {
            def: id,
            args: Vec::new(),
        });
        let mut args = Vec::new();
        let mut ok = true;
        for a in &def.args {
            let given = call.inputs.0.iter().find(|x| x.name == a.name);
            let cooked: Option<Vec<CookedInput>> = match (a.kind, given) {
                (_, None) => match a.default {
                    ArgDefault::Required => {
                        d.error(
                            span,
                            format!("dynamic input \"{}\" needs its input \"{}\" ({})", def.name, a.name, dynamic::arg_type(a)),
                            Some(format!("add `\"{}\": ...` to its inputs: {}", a.name, a.doc)),
                        );
                        None
                    }
                    ArgDefault::Value(v) => Some(vec![CookedInput::Const(v)]),
                    ArgDefault::Link(path) => {
                        let t = match a.kind {
                            ArgKind::Value(t) => t,
                            _ => ValueType::Float,
                        };
                        if path.starts_with("Particles.") && !stage.is_particle() {
                            d.error(
                                span,
                                format!(
                                    "input \"{}\" of \"{}\" reads {path} by default, which `{}` does not have",
                                    a.name,
                                    def.name,
                                    stage.field()
                                ),
                                Some(format!("give \"{}\" a value, e.g. `\"{}\": Link(Float, \"Emitter.Age\")`", a.name, a.name)),
                            );
                            None
                        } else {
                            self.link(stage, t, path, span, d)
                                .map(|r| vec![CookedInput::Link(r)])
                        }
                    }
                },
                (ArgKind::Value(t), Some(g)) => self
                    .input(stage, &def.name, a.name, t, g, dyns, d)
                    .map(|c| vec![c]),
                (ArgKind::Choice(choices), Some(g)) => {
                    choice_input(&def.name, a.name, ValueType::Int, choices, g, d).map(|c| vec![c])
                }
                (ArgKind::Curve(t), Some(g)) => match &g.input {
                    Input::Curve(c) if c.ty() == t => cook_curve(c, g.span, d).map(|c| vec![CookedInput::Curve(c)]),
                    // A curve data interface: its keys, baked in like a curve literal.
                    Input::DataInterface(dt, _) if dt.curve_type().is_some() => {
                        let want = crate::data::DiType::for_curve(t).expect("every curve type has a data interface");
                        self.data_interface(&def.name, a.name, want, g, d)
                            .and_then(|i| self.dis[i].curve.clone())
                            .map(|c| vec![CookedInput::Curve(c)])
                    }
                    Input::Curve(c) => {
                        d.error(
                            g.span,
                            format!("\"Curve\" of \"{}\" is a {} curve, but a {} curve is given", def.name, t.name(), c.ty().name()),
                            Some(format!("write the keys as `{}([(time, value), ...])`, or use \"{} from Curve\"", t.name(), c.ty().name())),
                        );
                        None
                    }
                    _ => {
                        d.error(
                            g.span,
                            format!("\"{}\" of \"{}\" takes a {} curve", a.name, def.name, t.name()),
                            Some(format!("write it as `\"{}\": {}([(0.0, ...), (1.0, ...)])`", a.name, t.name())),
                        );
                        None
                    }
                },
                (ArgKind::DataInterface(t), Some(g)) => {
                    if !stage.is_particle() {
                        d.error(
                            g.span,
                            format!("\"{}\" reads the camera, which `{}` does not have", def.name, stage.field()),
                            Some("use camera and scene depth dynamic inputs only in `particle_spawn`, `particle_update` or event handlers".into()),
                        );
                        None
                    } else {
                        self.data_interface(&def.name, a.name, t, g, d)
                            .map(|i| vec![CookedInput::Const(Value::Int(i as i32))])
                    }
                }
                (ArgKind::List(t), Some(g)) => match &g.input {
                    Input::List(items) if items.is_empty() => {
                        d.error(
                            g.span,
                            format!("\"{}\" of \"{}\" is an empty list", a.name, def.name),
                            Some(format!("give at least one {}", t.name())),
                        );
                        None
                    }
                    Input::List(items) => {
                        let cooked: Vec<_> = items
                            .iter()
                            .map(|it| self.input(stage, &def.name, &it.name, t, it, dyns, d))
                            .collect();
                        cooked.into_iter().collect()
                    }
                    _ => {
                        d.error(
                            g.span,
                            format!("\"{}\" of \"{}\" takes a list of {}", a.name, def.name, t.name()),
                            Some(format!("write it as `\"{}\": [{}, ...]`", a.name, t.example())),
                        );
                        None
                    }
                },
            };
            match cooked {
                Some(c) => args.extend(c),
                None => ok = false,
            }
        }
        if !ok {
            return None;
        }
        dyns[index].args = args;
        Some(CookedInput::Dynamic(index as u16))
    }

    /// The data interface input `e` (of `owner`'s input `iname`) names, which
    /// must be a declared data interface of type `want`: its index.
    fn data_interface(
        &self,
        owner: &str,
        iname: &str,
        want: crate::data::DiType,
        e: &InputEntry,
        d: &mut Diags,
    ) -> Option<usize> {
        let declared = || {
            let names: Vec<_> = self
                .dis
                .iter()
                .filter(|x| x.ty == want)
                .map(|x| format!("\"{}\"", x.name))
                .collect();
            if names.is_empty() {
                format!(
                    "declare one in `user_parameters`, e.g. `\"User.Name\": {}`",
                    want.example()
                )
            } else {
                format!("declared {} data interfaces: {}", want.name(), names.join(", "))
            }
        };
        let Input::DataInterface(t, target) = &e.input else {
            d.error(
                e.span,
                format!(
                    "input \"{iname}\" of \"{owner}\" takes a {} data interface",
                    want.name()
                ),
                Some(format!(
                    "write `\"{iname}\": Link({}, \"User.Name\")`; {}",
                    want.name(),
                    declared()
                )),
            );
            return None;
        };
        let Some(i) = self.dis.iter().position(|x| &x.name == target) else {
            let names = self.dis.iter().map(|x| x.name.as_str());
            let help = match suggest(target, names) {
                Some(s) => format!("did you mean \"{s}\"? {}", declared()),
                None => declared(),
            };
            d.error(
                e.span,
                format!("data interface \"{target}\" is not declared"),
                Some(help),
            );
            return None;
        };
        let found = self.dis[i].ty;
        if found != *t {
            d.error(
                e.span,
                format!("\"{target}\" is a {}, but the link says {}", found.name(), t.name()),
                Some(format!("write `Link({}, \"{target}\")`", found.name())),
            );
            return None;
        }
        if found != want {
            d.error(
                e.span,
                format!(
                    "input \"{iname}\" of \"{owner}\" takes a {} data interface, but \"{target}\" is a {}",
                    want.name(),
                    found.name()
                ),
                Some(declared()),
            );
            return None;
        }
        Some(i)
    }

    fn link(
        &self,
        stage: Stage,
        ty: ValueType,
        target: &str,
        span: Span,
        d: &mut Diags,
    ) -> Option<ParamRef> {
        let Some((ns, rest)) = params::split_namespace(target) else {
            let nss: Vec<_> = Namespace::ALL
                .iter()
                .map(|n| format!("{}.", n.prefix()))
                .collect();
            d.error(
                span,
                format!("\"{target}\" does not start with a parameter namespace"),
                Some(format!("parameter names start with one of {} (e.g. \"User.Wind\", \"Particles.Velocity\", \"Engine.DeltaTime\")", nss.join(", "))),
            );
            return None;
        };
        if !ns.readable_in(stage) {
            d.error(
                span,
                format!("`{}` cannot read {}. parameters", stage.field(), ns.prefix()),
                Some(format!("{}. parameters are readable only in particle stages; link a User., System., Emitter. or Engine. parameter here", ns.prefix())),
            );
            return None;
        }
        let (index, found_ty) = if ns == Namespace::User {
            let Some(i) = self.user.iter().position(|p| p.name == target) else {
                let help = if self.user.is_empty() {
                    format!(
                        "declare it in the effect: `user_parameters: {{ \"{target}\": {} }}`",
                        ty.example()
                    )
                } else {
                    let names: Vec<_> = self.user.iter().map(|p| p.name.as_str()).collect();
                    match suggest(target, names.iter().copied()) {
                        Some(s) => format!("did you mean \"{s}\"? Declared: {}", names.join(", ")),
                        None => format!(
                            "declare it in `user_parameters`, or use one of: {}",
                            names.join(", ")
                        ),
                    }
                };
                d.error(
                    span,
                    format!("user parameter \"{target}\" is not declared"),
                    Some(help),
                );
                return None;
            };
            (i, self.user[i].value.ty())
        } else {
            let table = params::builtins(ns);
            let Some(i) = table.iter().position(|b| b.name == rest) else {
                let help = match suggest(rest, table.iter().map(|b| b.name)) {
                    Some(s) => format!(
                        "did you mean \"{}.{s}\"? Available: {}",
                        ns.prefix(),
                        params::list_builtins(ns)
                    ),
                    None => format!("available: {}", params::list_builtins(ns)),
                };
                d.error(span, format!("unknown parameter \"{target}\""), Some(help));
                return None;
            };
            (i, table[i].ty)
        };
        if found_ty != ty {
            d.error(
                span,
                format!(
                    "\"{target}\" is {}, but the link says {}",
                    found_ty.name(),
                    ty.name()
                ),
                Some(format!(
                    "write `Link({}, \"{target}\")` and use it for a {} input",
                    found_ty.name(),
                    found_ty.name()
                )),
            );
            return None;
        }
        Some(ParamRef {
            ns,
            index: index as u16,
        })
    }
}

impl Ctx<'_> {
    /// A call of custom module `call.module` in `stage` (`EmitterUpdate` for
    /// grid stages: no particle reads there).
    fn custom_call(&self, stage: Stage, call: &ModuleCall, d: &mut Diags) -> Option<CookedModule> {
        let ci = self.custom.iter().position(|c| c.name == call.module)?;
        let def = &self.custom[ci];
        if !stage.is_particle() && stage != Stage::EmitterUpdate {
            d.error(
                call.span,
                format!("custom module \"{}\" cannot be used in `{}`", def.name, stage.field()),
                Some("custom modules run per particle (`particle_spawn`, `particle_update`, event handlers, `iterate: Particles` stages) or per grid cell (`iterate: Grid(...)` stages)".into()),
            );
            return None;
        }
        let mut inputs: Vec<CookedInput> =
            def.inputs.iter().map(|i| CookedInput::Const(i.default)).collect();
        let mut dynamics = Vec::new();
        for entry in &call.inputs.0 {
            let Some(slot) = def.inputs.iter().position(|i| i.name == entry.name) else {
                let names: Vec<&str> = def.inputs.iter().map(|i| i.name.as_str()).collect();
                let help = match suggest(&entry.name, names.iter().copied()) {
                    Some(s) => format!("did you mean \"{s}\"? Inputs: {}", names.join(", ")),
                    None => format!("inputs of \"{}\": {}", def.name, names.join(", ")),
                };
                d.error(
                    entry.span,
                    format!("custom module \"{}\" has no input \"{}\"", def.name, entry.name),
                    Some(help),
                );
                continue;
            };
            let want = def.inputs[slot].default.ty();
            if let Some(c) =
                self.input(stage, &def.name, &def.inputs[slot].name, want, entry, &mut dynamics, d)
            {
                inputs[slot] = c;
            }
        }
        Some(CookedModule {
            module: crate::custom::custom_id(ci),
            inputs,
            dynamics,
        })
    }

    /// An emitter's simulation stages.
    fn sim_stages(
        &self,
        e: &EmitterSource,
        grids: &[crate::custom::CookedGrid],
        d: &mut Diags,
    ) -> Vec<crate::custom::CookedSimStage> {
        use crate::custom::Iterate;
        let mut out = Vec::new();
        for (si, s) in e.simulation_stages.iter().enumerate() {
            if s.name.trim().is_empty() {
                d.error(s.span, "simulation stage `name` is empty", Some("name every stage, e.g. `name: \"Advect\"`".into()));
            } else if e.simulation_stages[..si].iter().any(|o| o.name == s.name) {
                d.error(s.span, format!("two simulation stages are named \"{}\"", s.name), Some("rename one".into()));
            }
            let (grid, modules) = match &s.iterate {
                Iterate::Particles => {
                    for call in &s.modules {
                        if let Some(m) = modules::by_name(&call.module) {
                            use ModuleKind::*;
                            if matches!(
                                m.kind,
                                Collision
                                    | KillParticles
                                    | KillParticlesInVolume
                                    | GenerateDeathEvent
                                    | GenerateCollisionEvent
                                    | GenerateLocationEvent
                                    | ReceiveDeathEvent
                                    | ReceiveCollisionEvent
                                    | ReceiveLocationEvent
                            ) {
                                d.error(
                                    call.span,
                                    format!("module \"{}\" cannot run in a simulation stage", m.name),
                                    Some("particle simulation stages update particles in place; collisions, kills and events go in `particle_update`".into()),
                                );
                            }
                        }
                    }
                    (None, self.stack(Stage::ParticleUpdate, &s.modules, d))
                }
                Iterate::Grid(name) => {
                    let Some(gi) = grids.iter().position(|g| &g.name == name) else {
                        let names: Vec<&str> = grids.iter().map(|g| g.name.as_str()).collect();
                        let help = if names.is_empty() {
                            "declare it in the emitter's `grids: [ Grid2D(...) ]`".to_string()
                        } else {
                            match suggest(name, names.iter().copied()) {
                                Some(x) => format!("did you mean \"{x}\"? Grids: {}", names.join(", ")),
                                None => format!("grids of this emitter: {}", names.join(", ")),
                            }
                        };
                        d.error(s.span, format!("stage \"{}\" iterates grid \"{name}\", which the emitter does not declare", s.name), Some(help));
                        continue;
                    };
                    let mut ms = Vec::new();
                    for call in &s.modules {
                        if !self.custom.iter().any(|c| c.name == call.module) {
                            d.error(
                                call.span,
                                format!("\"{}\" is not a custom module; grid stages run only custom WGSL modules", call.module),
                                Some("declare it in the effect's `custom_modules: [ ... ]`".into()),
                            );
                            continue;
                        }
                        if let Some(m) = self.custom_call(Stage::EmitterUpdate, call, d) {
                            if call.enabled {
                                ms.push(m);
                            }
                        }
                    }
                    (Some(gi as u32), ms)
                }
            };
            out.push(crate::custom::CookedSimStage {
                name: s.name.clone(),
                grid,
                iterations: s.iterations,
                modules,
            });
        }
        out
    }
}
