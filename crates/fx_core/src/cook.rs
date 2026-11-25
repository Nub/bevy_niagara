//! Validation of the effect model and cooking to the runtime asset.
//!
//! Cooking resolves module names to ids, input names to slots (with defaults
//! filled in), and parameter links to [`ParamRef`]s, then serialises with
//! rkyv so the runtime can read the asset in place (zero-copy).

use crate::diag::{Diags, suggest};
use crate::model::*;
use crate::modules::{self, ModuleDef, ModuleKind};
use crate::params::{self, Namespace, ParamRef, attr};
use crate::types::{Value, ValueType};

/// Bumped whenever the cooked layout changes.
pub const COOKED_FORMAT_VERSION: u32 = 4;
/// File extension of cooked effects.
pub const COOKED_EXTENSION: &str = "fxb";

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedSystem {
    pub format_version: u32,
    pub name: String,
    pub user_params: Vec<CookedParam>,
    pub emitters: Vec<CookedEmitter>,
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
                )
            ) || m.inputs.iter().any(|i| {
                matches!(i, CookedInput::Link(r)
                    if r.ns == Namespace::Particles && r.index >= attr::FIRST_EXTENDED)
            })
        };
        renders
            || self
                .particle_spawn
                .iter()
                .chain(&self.particle_update)
                .any(module)
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
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum CookedInput {
    Const(Value),
    Link(ParamRef),
    Random(Value, Value),
    Curve(CookedCurve),
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

    let cx = Ctx { user: &user_params };
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
        });
    }
    if d.has_errors() {
        return None;
    }
    Some(CookedSystem {
        format_version: COOKED_FORMAT_VERSION,
        name: src.name.clone(),
        user_params,
        emitters,
    })
}

fn enabled_kinds(calls: &[ModuleCall]) -> impl Iterator<Item = ModuleKind> + '_ {
    calls
        .iter()
        .filter(|c| c.enabled)
        .filter_map(|c| modules::by_name(&c.module))
        .map(|m| m.kind)
}

fn lint_emitter(e: &EmitterSource, d: &mut Diags) {
    let spawns = enabled_kinds(&e.emitter_update).any(|k| {
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
            ModuleKind::GravityForce | ModuleKind::Drag | ModuleKind::CurlNoiseForce
        )
    });
    let moves = enabled_kinds(&e.particle_spawn)
        .chain(update.iter().copied())
        .any(|k| matches!(k, ModuleKind::AddVelocity | ModuleKind::AddVelocityInCone));
    if (forces || moves) && !update.contains(&ModuleKind::SolveForcesAndVelocity) {
        d.warning(
            e.span,
            format!("emitter \"{}\" sets velocity or forces but has no \"Solve Forces and Velocity\", so particles never move", e.name),
            Some("add `(module: \"Solve Forces and Velocity\")` at the end of `particle_update`".into()),
        );
    }
}

struct Ctx<'a> {
    user: &'a [CookedParam],
}

impl Ctx<'_> {
    fn stack(&self, stage: Stage, calls: &[ModuleCall], d: &mut Diags) -> Vec<CookedModule> {
        let mut out = Vec::new();
        for call in calls {
            let Some(def) = self.module_def(stage, call, d) else {
                continue;
            };
            let mut inputs: Vec<CookedInput> = def
                .inputs
                .iter()
                .map(|i| CookedInput::Const(i.default))
                .collect();
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
                if let Some(c) = self.input(stage, def, idef.name, idef.ty, entry, d) {
                    inputs[slot] = c;
                }
            }
            if call.enabled {
                out.push(CookedModule {
                    module: def.kind as u16,
                    inputs,
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

    fn input(
        &self,
        stage: Stage,
        def: &ModuleDef,
        iname: &str,
        want: ValueType,
        e: &InputEntry,
        d: &mut Diags,
    ) -> Option<CookedInput> {
        let got = e.input.ty();
        if got != want {
            d.error(
                e.span,
                format!(
                    "input \"{iname}\" of \"{}\" is {}, but the value given is {}",
                    def.name,
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
                let keys = c.keys();
                if keys.is_empty() {
                    d.error(
                        e.span,
                        "curve has no keys",
                        Some("add at least one `(time, value)` key".into()),
                    );
                    return None;
                }
                if keys.windows(2).any(|w| w[1].0 < w[0].0) {
                    d.error(
                        e.span,
                        "curve key times must be in ascending order",
                        Some("sort the keys by their first number (time)".into()),
                    );
                    return None;
                }
                Some(CookedInput::Curve(CookedCurve {
                    ty: c.ty(),
                    times: keys.iter().map(|k| k.0).collect(),
                    values: keys.iter().map(|k| k.1).collect(),
                }))
            }
        }
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
