//! Converts an interpreted Niagara system ([`crate::niagara::NSystem`]) to
//! an effect source file (`.fx.ron`) and a coverage report.
//!
//! - Units: Unreal centimetres become metres and Unreal's Z-up axes become
//!   Bevy's Y-up ones by swapping Y and Z (Unreal (X, Y, Z) -> (X, Z, Y)).
//!   The swap mirrors the effect, which keeps random ranges and splatted
//!   vectors as they are (no component changes sign).
//! - Known Niagara modules and dynamic inputs map to ours ([`map_module`],
//!   [`Cx::dynamic`]); a module's mode (static switches) is read from which
//!   of its inputs the compiled stack still uses.
//! - Materials, textures and meshes live in other packages: sprites use
//!   the built-in soft sprite (blend mode guessed from the material name),
//!   meshes a built-in shape.
//! - Scratch pad modules with custom HLSL are translated to WGSL custom
//!   modules where they only read and write parameters we have
//!   ([`crate::hlsl`]); the rest, and unknown modules, are listed in the
//!   report.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use crate::fx::{self, f, quote, Curve, Fx, FxModule, Lit, Ty};
use crate::hlsl;
use crate::niagara::{CurveKey, ModuleKind, NCurve, NEmitter, NInput, NModule, NStack, NSystem, NValue, Usage};

/// How a value's unit changes from Unreal to ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    None,
    /// A distance (centimetres): scaled by 0.01; vectors also swap Y and Z.
    Len,
    /// A direction or per-axis factor: vectors swap Y and Z.
    Dir,
}

/// Unreal to our axes: (X, Y, Z) -> (X, Z, Y).
pub fn swap(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], v[1]]
}

const CM: f32 = 0.01;
/// Size in metres the placeholder mesh stands in for (Unreal meshes are
/// not in the package; 20 cm is a typical prop or debris size).
pub const PLACEHOLDER_MESH_SIZE: f32 = 0.2;

/// How one Niagara module fared.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Mapped with every input that matters.
    Mapped,
    /// Mapped, but some inputs were dropped or approximated (listed).
    Partial(Vec<String>),
    /// Not mapped (why).
    Unmapped(String),
    /// Disabled in Niagara: written disabled or left out.
    Disabled,
    /// Has no effect here (e.g. System State, folded into emitter loops).
    Folded(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModuleReport {
    pub emitter: String,
    pub stack: String,
    pub module: String,
    pub script: String,
    pub status: Status,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub source: String,
    pub system: String,
    pub modules: Vec<ModuleReport>,
    pub dynamic_total: usize,
    pub dynamic_mapped: usize,
    /// Dynamic inputs not mapped: (where, script).
    pub unmapped_dynamic: Vec<(String, String)>,
    pub renderers: Vec<String>,
    pub placeholders: Vec<String>,
    /// Materials, textures and meshes read from their packages.
    pub materials: Vec<String>,
    pub notes: Vec<String>,
    /// Custom HLSL translations: (module, outcome).
    pub hlsl: Vec<(String, String)>,
    /// Errors from validating the written effect (empty when it checks).
    pub validation: Vec<String>,
}

impl Report {
    /// Modules that count: enabled ones.
    pub fn counted(&self) -> impl Iterator<Item = &ModuleReport> {
        self.modules.iter().filter(|m| m.status != Status::Disabled)
    }
    pub fn mapped(&self) -> usize {
        self.counted()
            .filter(|m| matches!(m.status, Status::Mapped | Status::Partial(_) | Status::Folded(_)))
            .count()
    }
    pub fn total(&self) -> usize {
        self.counted().count()
    }
    pub fn percent(&self) -> f32 {
        if self.total() == 0 {
            100.0
        } else {
            100.0 * self.mapped() as f32 / self.total() as f32
        }
    }

    pub fn markdown(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "# Niagara import report: {}\n", self.system);
        let _ = writeln!(o, "Source: `{}`\n", self.source);
        let _ = writeln!(
            o,
            "**Modules mapped: {}/{} ({:.0}%)**, of which {} fully, {} with inputs dropped or approximated, {} folded. Dynamic inputs mapped: {}/{}. Disabled modules (not counted): {}.\n",
            self.mapped(),
            self.total(),
            self.percent(),
            self.counted().filter(|m| m.status == Status::Mapped).count(),
            self.counted().filter(|m| matches!(m.status, Status::Partial(_))).count(),
            self.counted().filter(|m| matches!(m.status, Status::Folded(_))).count(),
            self.dynamic_mapped,
            self.dynamic_total,
            self.modules.iter().filter(|m| m.status == Status::Disabled).count(),
        );
        let _ = writeln!(
            o,
            "Validation: {}\n",
            if self.validation.is_empty() {
                "the effect checks (fxc check).".to_string()
            } else {
                format!("{} errors:\n\n{}", self.validation.len(), self.validation.iter().map(|e| format!("- {e}")).collect::<Vec<_>>().join("\n"))
            }
        );
        let unmapped: Vec<_> = self.counted().filter(|m| matches!(m.status, Status::Unmapped(_))).collect();
        let _ = writeln!(o, "## Not mapped ({})\n", unmapped.len());
        if unmapped.is_empty() {
            let _ = writeln!(o, "None.\n");
        }
        for m in &unmapped {
            if let Status::Unmapped(why) = &m.status {
                let _ = writeln!(o, "- {} / {}: **{}** (`{}`): {why}", m.emitter, m.stack, m.module, m.script);
            }
        }
        if !unmapped.is_empty() {
            o.push('\n');
        }
        if !self.unmapped_dynamic.is_empty() {
            let _ = writeln!(o, "## Dynamic inputs not mapped ({}; the input keeps our default)\n", self.unmapped_dynamic.len());
            for (w, s) in &self.unmapped_dynamic {
                let _ = writeln!(o, "- {w}: `{s}`");
            }
            o.push('\n');
        }
        let partial: Vec<_> = self.counted().filter(|m| matches!(m.status, Status::Partial(_))).collect();
        if !partial.is_empty() {
            let _ = writeln!(o, "## Mapped with changes ({})\n", partial.len());
            for m in partial {
                if let Status::Partial(items) = &m.status {
                    let _ = writeln!(o, "- {} / {}: **{}**: {}", m.emitter, m.stack, m.module, items.join("; "));
                }
            }
            o.push('\n');
        }
        if !self.hlsl.is_empty() {
            let _ = writeln!(o, "## Custom HLSL\n");
            for (m, r) in &self.hlsl {
                let _ = writeln!(o, "- {m}: {r}");
            }
            o.push('\n');
        }
        if !self.materials.is_empty() {
            let _ = writeln!(o, "## Materials and meshes\n");
            for m in &self.materials {
                let _ = writeln!(o, "- {m}");
            }
            o.push('\n');
        }
        let _ = writeln!(o, "## Renderers and placeholders\n");
        for r in self.renderers.iter().chain(&self.placeholders) {
            let _ = writeln!(o, "- {r}");
        }
        if !self.notes.is_empty() {
            let _ = writeln!(o, "\n## Notes\n");
            for n in &self.notes {
                let _ = writeln!(o, "- {n}");
            }
        }
        let _ = writeln!(o, "\n## All modules\n");
        let _ = writeln!(o, "| Emitter | Stack | Niagara module | Result |");
        let _ = writeln!(o, "|---|---|---|---|");
        for m in &self.modules {
            let r = match &m.status {
                Status::Mapped => "mapped".to_string(),
                Status::Partial(_) => "mapped, with changes".to_string(),
                Status::Unmapped(_) => "**not mapped**".to_string(),
                Status::Disabled => "disabled".to_string(),
                Status::Folded(w) => format!("folded: {w}"),
            };
            let _ = writeln!(o, "| {} | {} | {} | {} |", m.emitter, m.stack, m.module, r);
        }
        o
    }
}

/// Where `convert_with_assets` finds referenced packages and what it writes
/// beside the effect (paths relative to the effect's folder).
pub struct AssetOut {
    pub store: crate::material::Store,
    /// Folder for the written files, relative to the effect file.
    pub dir: String,
    pub files: BTreeMap<String, Vec<u8>>,
    mats: HashMap<(String, bool, [bool; 4]), Result<MatOut, String>>,
    meshes: HashMap<String, Result<MeshOut, String>>,
}

impl AssetOut {
    pub fn new(store: crate::material::Store, dir: impl Into<String>) -> Self {
        AssetOut {
            store,
            dir: dir.into(),
            files: BTreeMap::new(),
            mats: HashMap::new(),
            meshes: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct MeshOut {
    name: String,
    glb: String,
    material: Option<String>,
    summary: String,
}

#[derive(Clone)]
struct MatOut {
    name: String,
    wesl: String,
    texture: Option<String>,
    blend: crate::material::Blend,
    unlit: bool,
    refraction: bool,
    textures: Vec<String>,
    missing_textures: Vec<String>,
    notes: Vec<String>,
}

impl MatOut {
    fn describe(&self) -> String {
        let mut s = format!(" ({:?}, {}", self.blend, if self.unlit { "unlit" } else { "lit" });
        if !self.textures.is_empty() {
            let _ = write!(s, "; textures {}", self.textures.join(", "));
        }
        s.push(')');
        for m in &self.missing_textures {
            let _ = write!(s, "; **missing texture** {m}: black stands in (the terms it feeds read 0)");
        }
        for n in &self.notes {
            let _ = write!(s, "; {n}");
        }
        s
    }
}

/// The converted effect.
pub struct Conversion {
    pub ron: String,
    pub report: Report,
    /// Files to write beside the effect (relative path, bytes): textures,
    /// materials, meshes.
    pub files: BTreeMap<String, Vec<u8>>,
}

/// Converts a system. `source` names the file it came from (for the
/// header and report).
pub fn convert(sys: &NSystem, source: &str) -> Conversion {
    convert_with_assets(sys, source, None)
}

/// [`convert`], with the referenced materials, textures and meshes read
/// from `assets`' package folders and written beside the effect.
pub fn convert_with_assets(sys: &NSystem, source: &str, assets: Option<AssetOut>) -> Conversion {
    let mut cx = Cx {
        assets,
        sys,
        user_units: HashMap::new(),
        report: Report {
            source: source.to_string(),
            system: sys.name.clone(),
            ..Report::default()
        },
        custom: Vec::new(),
        sys_values: HashMap::new(),
        user_overrides: HashMap::new(),
    };
    for n in &sys.notes {
        cx.report.notes.push(n.clone());
    }
    // Set Variables in the system stacks: what emitters can read as constants.
    let mut sys_status: HashMap<(usize, usize), Status> = HashMap::new();
    for (si, st) in sys.stacks.iter().enumerate() {
        for (mi, m) in st.modules.iter().enumerate() {
            let ModuleKind::Assignment { targets } = &m.kind else { continue };
            if !m.enabled {
                continue;
            }
            let (mut done, mut failed) = (Vec::new(), Vec::new());
            for (t, _) in targets {
                let why = match m.input(t) {
                    None => Some("no value".to_string()),
                    Some(NInput::Value(NValue::DataInterface(c))) => Some(format!("a {c} data interface")),
                    Some(NInput::Value(NValue::Curve(_) | NValue::Enum(_) | NValue::Matrix(_))) => Some("a value of an unsupported type".into()),
                    Some(NInput::Value(v)) if t.starts_with("User.") => {
                        cx.user_overrides.insert(t.clone(), v);
                        None
                    }
                    Some(i @ NInput::Value(_)) if t.starts_with("System.") => {
                        cx.sys_values.insert(t.clone(), i);
                        None
                    }
                    Some(i @ NInput::Link(l)) if t.starts_with("System.") && (l.starts_with("User.") || l.starts_with("Engine.")) => {
                        cx.sys_values.insert(t.clone(), i);
                        None
                    }
                    Some(NInput::Link(l)) => Some(format!("links {l}")),
                    Some(_) if !(t.starts_with("User.") || t.starts_with("System.")) => Some("not a User or System parameter".into()),
                    Some(_) => Some("computed per frame (not a constant)".into()),
                };
                match why {
                    None => done.push(t.clone()),
                    Some(w) => failed.push(format!("{t}: {w}")),
                }
            }
            let status = if done.is_empty() {
                Status::Unmapped(format!("Set Variables of {}", failed.join(", ")))
            } else if failed.is_empty() {
                Status::Folded(format!("{} folded into the emitters' inputs and user parameter defaults", done.join(", ")))
            } else {
                Status::Partial(failed.into_iter().map(|f| format!("not set: {f}")).collect())
            };
            sys_status.insert((si, mi), status);
        }
    }
    let loops = system_loops(sys);
    let mut emitters = Vec::new();
    for e in &sys.emitters {
        if !e.enabled {
            cx.report.notes.push(format!("emitter {} is disabled in Niagara: left out", e.name));
            continue;
        }
        emitters.push(cx.emitter(e, &loops));
    }
    for (si, st) in sys.stacks.iter().enumerate() {
        for (mi, m) in st.modules.iter().enumerate() {
            let status = if !m.enabled {
                Status::Disabled
            } else if let Some(s) = sys_status.remove(&(si, mi)) {
                s
            } else if m.script_name() == "SystemState" {
                Status::Folded("system loop settings go to emitters whose Life Cycle Mode is System".into())
            } else {
                Status::Unmapped("system stacks are not converted (no system modules here)".into())
            };
            cx.report.modules.push(ModuleReport {
                emitter: "(system)".into(),
                stack: stack_name(&st.usage).into(),
                module: m.name.clone(),
                script: m.script.clone(),
                status,
            });
        }
    }
    let ron = cx.write(&emitters);
    let files = cx.assets.take().map(|a| a.files).unwrap_or_default();
    let mut report = cx.report;
    // Validate.
    let checked = fx_core::check_source(&format!("{}.fx.ron", sys.name), &ron);
    report.validation = checked.errors().map(|d| d.to_string()).collect();
    Conversion { ron, report, files }
}

/// The system's loop settings (System State), used by emitters whose
/// Emitter State says Life Cycle Mode: System.
#[derive(Clone, Debug, Default)]
struct Loops {
    behavior: Option<String>,
    duration: Option<f32>,
    delay: Option<f32>,
    count: Option<i32>,
}

fn system_loops(sys: &NSystem) -> Loops {
    let mut l = Loops::default();
    for st in &sys.stacks {
        for m in &st.modules {
            if m.script_name() != "SystemState" {
                continue;
            }
            l.behavior = m.switch("Loop Behavior").map(str::to_string);
            l.duration = lit_f(m.input("Loop Duration"));
            l.delay = lit_f(m.input("Loop Delay"));
            l.count = lit_f(m.input("Loop Count")).map(|v| v as i32);
            if m.input("Loop Duration").is_none() && m.switch("Loop Behavior").is_some() {
                // Loop Duration Mode Infinite (no duration in the compiled stack).
                l.duration = None;
            }
        }
    }
    l
}

fn lit_f(i: Option<&NInput>) -> Option<f32> {
    match i? {
        NInput::Value(v) => v.floats().and_then(|f| f.first().copied()),
        _ => None,
    }
}

fn stack_name(u: &Usage) -> &'static str {
    match u {
        Usage::SystemSpawn => "system_spawn",
        Usage::SystemUpdate => "system_update",
        Usage::EmitterSpawn => "emitter_spawn",
        Usage::EmitterUpdate => "emitter_update",
        Usage::ParticleSpawn => "particle_spawn",
        Usage::ParticleUpdate => "particle_update",
        Usage::Event(_) => "event_handlers",
        Usage::SimStage(_) => "simulation_stages",
        Usage::Other(_) => "other",
    }
}

/// A converted emitter.
struct FxEmitter {
    name: String,
    gpu: bool,
    max_particles: u32,
    deterministic: bool,
    interpolated_spawning: bool,
    /// Niagara bLocalSpace: false means particles live in world space.
    local_space: bool,
    fixed_bounds: Option<([f32; 3], [f32; 3])>,
    stacks: Vec<(String, Vec<FxModule>)>,
    events: Vec<(String, Vec<FxModule>)>,
    stages: Vec<String>,
    renderers: Vec<String>,
    comment: Vec<String>,
}

struct Cx<'a> {
    sys: &'a NSystem,
    /// Units the user parameters are used with (by links).
    user_units: HashMap<String, Unit>,
    report: Report,
    /// Custom WGSL modules (source text) for the system's `custom_modules`.
    custom: Vec<(String, String)>,
    assets: Option<AssetOut>,
    /// `System.` parameters the system stacks set to a constant or a user
    /// or engine parameter: links to them read that instead.
    sys_values: HashMap<String, &'a NInput>,
    /// `User.` parameters the system stacks set to a constant: their defaults.
    user_overrides: HashMap<String, &'a NValue>,
}

/// One module's conversion in progress: the outputs and what was dropped.
struct Out {
    mods: Vec<FxModule>,
    changes: Vec<String>,
}

impl Out {
    fn new() -> Self {
        Out {
            mods: Vec::new(),
            changes: Vec::new(),
        }
    }
    fn drop_input(&mut self, name: &str, why: &str) {
        self.changes.push(format!("{name}: {why}"));
    }
}

/// Inputs that never matter for the result here (seeds, spawn groups, the
/// rotation helpers Niagara keeps for its own transforms).
const IGNORED: &[&str] = &[
    "Random Seed",
    "Spawn Group",
    "SpawnGroup",
    "Spawn Group Mask",
    "Rotation Matrix",
    "Rotation Quaternion",
    "Rotation Axis",
    "Rotation Angle",
    "Yaw / Pitch / Roll",
    "Additional Yaw / Pitch / Roll",
    "Apply Owner Scale",
    "Non Uniform Scale",
    "MinDistance",
    "MaxDistance",
    "Index0Param0Write",
    "Index0Param1Write",
    "Index0Param2Write",
    "Index0Param3Write",
    "Index1Param0Write",
    "Index1Param1Write",
    "Index1Param2Write",
    "Index1Param3Write",
    "Index2Param0Write",
    "Index2Param1Write",
    "Index2Param2Write",
    "Index2Param3Write",
    "Index3Param0Write",
    "Index3Param1Write",
    "Index3Param2Write",
    "Index3Param3Write",
    "Debug Attractor Radius Color",
    "Debug Every N Particles",
    "Debug Force Line Color",
    "Debug Force Line Length",
    "Default Position",
    "Position Offset",
    "Age",
    "Loop Count Limit",
];

impl<'a> Cx<'a> {
    fn emitter(&mut self, e: &NEmitter, loops: &Loops) -> FxEmitter {
        let mut out = FxEmitter {
            name: e.name.clone(),
            gpu: e.gpu,
            max_particles: 0,
            deterministic: e.deterministic,
            interpolated_spawning: e.interpolated_spawning,
            local_space: e.local_space,
            fixed_bounds: e.fixed_bounds.map(|(a, b)| {
                let a = swap(a).map(|v| v * CM);
                let b = swap(b).map(|v| v * CM);
                (
                    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])],
                    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])],
                )
            }),
            stacks: Vec::new(),
            events: Vec::new(),
            stages: Vec::new(),
            renderers: Vec::new(),
            comment: Vec::new(),
        };
        for n in &e.notes {
            self.report.notes.push(format!("{}: {n}", e.name));
        }
        let mut has_sprite_size = false;
        for st in &e.stacks {
            let sname = stack_name(&st.usage);
            let mut mods = Vec::new();
            for m in &st.modules {
                let status = if !m.enabled {
                    Status::Disabled
                } else {
                    let mut o = Out::new();
                    let res = self.module(e, st, m, loops, &mut o);
                    for x in &o.mods {
                        if x.name == "Initialize Particle" && x.get("Sprite Size").is_some() {
                            has_sprite_size = true;
                        }
                    }
                    if let Some(first) = o.mods.first_mut() {
                        first.comment = Some(format!("Niagara: {} ({})", m.name, m.script_name()));
                    }
                    mods.extend(o.mods);
                    match res {
                        Err(why) => Status::Unmapped(why),
                        Ok(Some(folded)) => Status::Folded(folded),
                        Ok(None) if o.changes.is_empty() => Status::Mapped,
                        Ok(None) => Status::Partial(o.changes),
                    }
                };
                self.report.modules.push(ModuleReport {
                    emitter: e.name.clone(),
                    stack: sname.into(),
                    module: m.name.clone(),
                    script: m.script.clone(),
                    status,
                });
            }
            match &st.usage {
                Usage::Event(ev) => {
                    let event = match ev.event_name.as_str() {
                        n if n.contains("Death") => "Death",
                        n if n.contains("Collision") => "Collision",
                        _ => "Location",
                    };
                    let mode = match ev.execution_mode.as_str() {
                        "SpawnedParticles" => "SpawnedParticles",
                        _ => "EveryParticle",
                    };
                    let src = ev.source_emitter.clone().unwrap_or_else(|| e.name.clone());
                    out.events.push((
                        format!(
                            "(source: {}, event: {event}, execution_mode: {mode}, spawn_number: {}, max_events_per_frame: {}, modules: [",
                            quote(&src),
                            ev.spawn_number,
                            ev.max_events_per_frame
                        ),
                        mods,
                    ));
                }
                Usage::SimStage(s) => {
                    if !s.enabled {
                        self.report.notes.push(format!("{}: simulation stage {} is disabled: left out", e.name, s.name));
                    } else if s.source != "Particles" {
                        self.report.notes.push(format!(
                            "{}: simulation stage {} iterates a data interface ({}): not converted",
                            e.name, s.name, s.source
                        ));
                    } else if !mods.is_empty() {
                        let mut t = String::new();
                        fx::write_modules(&mut t, "                ", &mods);
                        out.stages.push(format!(
                            "(name: {}, iterate: Particles, iterations: {}, modules: [\n{t}            ]),",
                            quote(&s.name),
                            s.iterations.clamp(1, 64)
                        ));
                    }
                }
                Usage::Other(o) => self.report.notes.push(format!("{}: stack {o} not converted", e.name)),
                _ => out.stacks.push((sname.to_string(), mods)),
            }
        }
        // Niagara's unset sprite size is 50 x 50 cm.
        if !has_sprite_size && e.renderers.iter().any(|r| r.class.contains("Sprite")) {
            for (name, mods) in out.stacks.iter_mut() {
                if name == "particle_spawn" {
                    if let Some(init) = mods.iter_mut().find(|m| m.name == "Initialize Particle") {
                        init.set("Sprite Size", Fx::Lit(Lit::Vec2([0.5, 0.5])));
                    }
                }
            }
        }
        // Dynamic Material Parameter indices the emitter writes.
        let mut written = [false; 4];
        for (_, mods) in out.stacks.iter().chain(out.events.iter()) {
            for m in mods {
                match m.name.as_str() {
                    "Dynamic Material Parameters" => written[0] = true,
                    "Dynamic Material Parameters 2" => written[1] = true,
                    "Dynamic Material Parameters 3" => written[2] = true,
                    "Dynamic Material Parameters 4" => written[3] = true,
                    _ => {}
                }
            }
        }
        for r in &e.renderers {
            if let Some(s) = self.renderer(e, r, written) {
                out.renderers.push(s);
            }
        }
        out.max_particles = self.estimate_capacity(e, &out);
        out
    }

    /// Converts one module into `o`. Ok(Some(why)) when it is folded
    /// into something else, Err(why) when it is not mapped.
    fn module(&mut self, e: &NEmitter, st: &NStack, m: &NModule, loops: &Loops, o: &mut Out) -> Result<Option<String>, String> {
        match &m.kind {
            ModuleKind::Assignment { targets } => return self.assignment(e, m, targets, o),
            ModuleKind::ScratchPad { hlsl } => return self.scratch_pad(e, st, m, hlsl, o),
            ModuleKind::Function => {}
        }
        if !m.script.starts_with("/Niagara/") {
            // A module script in another package of the project: read its
            // custom HLSL like a scratch pad's when the package is present.
            let loaded = self.assets.as_mut().and_then(|a| a.store.load(&m.script));
            let Some(l) = loaded else {
                return Err(format!("a module from another package ({}), not in the package folders", m.script));
            };
            let hlsl = crate::niagara::module_script_hlsl(&l.pkg).map_err(|w| format!("module package {}: {w}", m.script))?;
            if hlsl.is_empty() {
                return Err(format!("module from another package ({}) built from graph nodes (no custom HLSL to translate)", m.script));
            }
            return self.scratch_pad(e, st, m, &hlsl, o);
        }
        map_module(self, e, st, m, loops, o)
    }

    // ---- inputs ----

    /// The user parameter's value, by full name.
    fn user(&self, name: &str) -> Option<&'a NValue> {
        if let Some(v) = self.user_overrides.get(name) {
            return Some(*v);
        }
        self.sys.user_parameters.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Converts a module input to a value of type `want`.
    fn conv(&mut self, e: &NEmitter, i: &NInput, unit: Unit, want: Ty) -> Result<Fx, String> {
        let v = match i {
            NInput::Value(v) => value_fx(v, unit)?,
            NInput::Link(name) => self.link(name, unit)?,
            NInput::Dynamic(d) => self.dynamic(e, d, unit, want)?,
            NInput::Unsupported(s) => return Err(format!("not readable ({s})")),
        };
        coerce(v, want)
    }

    /// A module input converted, or None when absent; failures are noted.
    fn input(&mut self, e: &NEmitter, m: &NModule, name: &str, unit: Unit, want: Ty, o: &mut Out) -> Option<Fx> {
        let i = m.input(name)?;
        match self.conv(e, i, unit, want) {
            Ok(v) => Some(v),
            Err(why) => {
                o.drop_input(name, &why);
                None
            }
        }
    }

    fn link(&mut self, name: &str, unit: Unit) -> Result<Fx, String> {
        if let Some(i) = self.sys_values.get(name).copied() {
            // Set by a system stack: its constant, or the parameter it reads.
            return match i {
                NInput::Value(v) => value_fx(v, unit),
                NInput::Link(l) if l != name => self.link(l, unit),
                _ => Err(format!("links {name}, set by the system stack")),
            };
        }
        if name.starts_with("User.") {
            let Some(v) = self.user(name) else {
                return Err(format!("links {name}, which is not a user parameter"));
            };
            if unit != Unit::None {
                if let Some(prev) = self.user_units.insert(name.to_string(), unit) {
                    if prev != unit {
                        self.report.notes.push(format!("{name} is used with different units; converted as {unit:?}"));
                    }
                }
            }
            return match v {
                NValue::Curve(c) => Ok(Fx::Link(curve_ty(c).curve_decl().into(), name.into())),
                NValue::DataInterface(c) => Err(format!("links {name}, a {c} data interface")),
                NValue::Enum(_) | NValue::Matrix(_) => Err(format!("links {name} of an unsupported type")),
                other => Ok(Fx::Link(nvalue_ty(other).name().into(), name.into())),
            };
        }
        let mapped = match name {
            "Engine.DeltaTime" | "Engine.Time" | "Engine.Owner.Velocity" => name.to_string(),
            n if n.starts_with("Particles.") => n.to_string(),
            n if n.starts_with("Emitter.") || n.starts_with("System.") || n.starts_with("Engine.") => n.to_string(),
            n => return Err(format!("links {n}, which has no equivalent here")),
        };
        let Some((ns, rest)) = fx_core::params::split_namespace(&mapped) else {
            return Err(format!("links {name}"));
        };
        match fx_core::params::builtins(ns).iter().find(|b| b.name == rest) {
            Some(b) => Ok(Fx::Link(b.ty.name().into(), mapped)),
            None => Err(format!("links {name}, which has no equivalent here")),
        }
    }

    /// Maps a Niagara dynamic input.
    fn dynamic(&mut self, e: &NEmitter, d: &NModule, unit: Unit, want: Ty) -> Result<Fx, String> {
        self.report.dynamic_total += 1;
        let r = self.dynamic_inner(e, d, unit, want);
        match &r {
            Ok(_) => self.report.dynamic_mapped += 1,
            Err(_) => self.report.unmapped_dynamic.push((format!("{} / {}", e.name, d.name), d.script.clone())),
        }
        r
    }

    fn dynamic_inner(&mut self, e: &NEmitter, d: &NModule, unit: Unit, want: Ty) -> Result<Fx, String> {
        if !d.enabled {
            return Err("disabled dynamic input".into());
        }
        let name = d.script_name().to_string();
        let get = |cx: &mut Self, n: &str, unit: Unit, ty: Ty| -> Result<Option<Fx>, String> {
            match d.input(n) {
                None => Ok(None),
                Some(i) => cx.conv(e, i, unit, ty).map(Some),
            }
        };
        match name.as_str() {
            "RandomRangeFloat" | "RandomRangeInt" | "RandomRangeInteger" | "RandomRangeVector" | "RandomRangeVector2D"
            | "RandomRangeVector4" | "RandomRangeLinearColor" | "RandomRangeColor" => {
                let ty = match name.as_str() {
                    "RandomRangeFloat" => Ty::Float,
                    "RandomRangeInt" | "RandomRangeInteger" => Ty::Int,
                    "RandomRangeVector" => Ty::Vec3,
                    "RandomRangeVector2D" => Ty::Vec2,
                    "RandomRangeVector4" => Ty::Vec4,
                    _ => Ty::Color,
                };
                let lo = get(self, "Minimum", unit, ty)?.unwrap_or(Fx::Lit(Lit::from_floats(ty, &[0.0])));
                let hi = get(self, "Maximum", unit, ty)?.unwrap_or(Fx::Lit(Lit::from_floats(ty, &[1.0])));
                Ok(match (lo, hi) {
                    (Fx::Lit(a), Fx::Lit(b)) => Fx::Range(a, b),
                    (a, b) => Fx::Dyn(format!("Uniform Ranged {}", ty.name()), vec![("Minimum".into(), a), ("Maximum".into(), b)]),
                })
            }
            "FloatFromCurve" | "ColorFromCurve" | "VectorFromCurve" | "Vector2DFromCurve" | "Vector4FromCurve" => {
                let (ty, key) = match name.as_str() {
                    "FloatFromCurve" => (Ty::Float, "FloatCurve"),
                    "ColorFromCurve" => (Ty::Color, "ColorCurve"),
                    "VectorFromCurve" => (Ty::Vec3, "VectorCurve"),
                    "Vector2DFromCurve" => (Ty::Vec2, "Vector2DCurve"),
                    _ => (Ty::Vec4, "Vector4Curve"),
                };
                let scale = lit_f(d.input("Scale Curve")).unwrap_or(1.0);
                let curve = match d.input(key) {
                    Some(NInput::Value(NValue::Curve(c))) => {
                        let mut c = curve(c, ty, unit);
                        for (_, v) in c.keys.iter_mut() {
                            for x in v.iter_mut() {
                                *x *= scale;
                            }
                        }
                        Fx::Curve(c)
                    }
                    Some(NInput::Link(l)) => {
                        let link = self.link(l, unit)?;
                        if scale != 1.0 {
                            return Err("a linked curve with a Scale Curve".into());
                        }
                        link
                    }
                    Some(other) => return Err(format!("curve input {other:?}")),
                    None => return Err("no curve".into()),
                };
                // A literal curve index is Niagara's placeholder for its
                // default (normalised age).
                let index = match d.input("Curve Index") {
                    Some(i @ (NInput::Link(_) | NInput::Dynamic(_))) => Some(self.conv(e, i, Unit::None, Ty::Float)?),
                    _ => None,
                };
                let out = match (curve, index) {
                    (Fx::Curve(c), None) => Fx::CurveOverAge(c),
                    (c, None) => Fx::Dyn(format!("{} from Curve", ty.name()), vec![("Curve".into(), c)]),
                    (c, Some(i)) => Fx::Dyn(
                        format!("{} from Curve", ty.name()),
                        vec![("Curve".into(), c), ("Curve Index".into(), i)],
                    ),
                };
                // A Scale Curve that is not a constant (a random range, a
                // link) multiplies the curve's value.
                match d.input("Scale Curve") {
                    Some(i @ (NInput::Link(_) | NInput::Dynamic(_))) if ty == Ty::Float => {
                        let s = match self.conv(e, i, Unit::None, Ty::Float)? {
                            // A random range here would be drawn again every
                            // update and flicker: its mean stands in.
                            Fx::Range(Lit::Float(a), Lit::Float(b)) => Fx::Lit(Lit::Float((a + b) * 0.5)),
                            s => s,
                        };
                        Ok(Fx::Dyn("Multiply Float".into(), vec![("A".into(), out), ("B".into(), s)]))
                    }
                    Some(NInput::Link(_) | NInput::Dynamic(_)) => Err("a non-constant Scale Curve on a vector curve".into()),
                    _ => Ok(out),
                }
            }
            "VectorFromFloat" => {
                let v = get(self, "Value", unit, Ty::Float)?.unwrap_or(Fx::Lit(Lit::Float(0.0)));
                coerce(v, Ty::Vec3)
            }
            "Multiply_Float" | "Multiply_VectorByFloat" | "Multiply_Vector" | "Multiply_LinearColor" => {
                let (ta, ka, kb, tb) = match name.as_str() {
                    "Multiply_Float" => (Ty::Float, "A", "B", Ty::Float),
                    "Multiply_VectorByFloat" => (Ty::Vec3, "Vector", "Float", Ty::Float),
                    "Multiply_Vector" => (Ty::Vec3, "A", "B", Ty::Vec3),
                    _ => (Ty::Color, "A", "B", Ty::Color),
                };
                // The unit goes to the first operand only.
                let a = get(self, ka, unit, ta)?.unwrap_or(Fx::Lit(Lit::from_floats(ta, &[1.0])));
                let bu = if tb == Ty::Float { Unit::None } else { Unit::Dir };
                let b = get(self, kb, bu, tb)?.unwrap_or(Fx::Lit(Lit::from_floats(tb, &[1.0])));
                if let (Fx::Lit(x), Fx::Lit(y)) = (&a, &b) {
                    let xs = x.floats();
                    let ys = y.floats();
                    let r: Vec<f32> = xs.iter().enumerate().map(|(k, v)| v * ys.get(k).or(ys.last()).unwrap_or(&1.0)).collect();
                    return Ok(Fx::Lit(Lit::from_floats(ta, &r)));
                }
                let n = match (ta, tb) {
                    (Ty::Float, _) => "Multiply Float",
                    (Ty::Vec3, Ty::Float) => "Multiply Vec3 by Float",
                    (Ty::Vec3, _) => "Multiply Vec3",
                    _ => "Multiply Color",
                };
                Ok(Fx::Dyn(n.into(), vec![("A".into(), a), ("B".into(), b)]))
            }
            "Add_Float" | "Add_Vector" | "AddVectorToPosition" => {
                let (ty, ka, kb) = match name.as_str() {
                    "Add_Float" => (Ty::Float, "A", "B"),
                    "Add_Vector" => (Ty::Vec3, "A", "B"),
                    _ => (Ty::Vec3, "Position", "Vector"),
                };
                let a = get(self, ka, unit, ty)?;
                let b = get(self, kb, unit, ty)?;
                let a = a.unwrap_or_else(|| {
                    if ka == "Position" {
                        Fx::Link("Vec3".into(), "Particles.Position".into())
                    } else {
                        Fx::Lit(Lit::from_floats(ty, &[0.0]))
                    }
                });
                let b = b.unwrap_or(Fx::Lit(Lit::from_floats(ty, &[0.0])));
                Ok(Fx::Dyn(format!("Add {}", ty.name()), vec![("A".into(), a), ("B".into(), b)]))
            }
            "Sine" => {
                // Niagara's Sine (Angles): Scale * sin(2π * Time / Period) + Bias.
                let scale = get(self, "Scale", unit, Ty::Float)?.unwrap_or(Fx::Lit(Lit::Float(1.0)));
                let bias = get(self, "Bias", unit, Ty::Float)?.unwrap_or(Fx::Lit(Lit::Float(0.0)));
                let period = lit_f(d.input("Period")).unwrap_or(1.0);
                if d.has("Normalized Angle") || d.has("Angle") {
                    return Err("Sine of an angle input (only time-based Sine is mapped)".into());
                }
                Ok(Fx::Dyn(
                    "Sine over Time".into(),
                    vec![
                        ("Amplitude".into(), scale),
                        ("Frequency".into(), Fx::Lit(Lit::Float(if period != 0.0 { 1.0 / period } else { 0.0 }))),
                        ("Offset".into(), bias),
                    ],
                ))
            }
            "MakeVectorFromLinearColorRGB" => {
                let c = get(self, "Color", Unit::None, Ty::Color)?.unwrap_or(Fx::Lit(Lit::Color([1.0; 4])));
                if let Fx::Lit(l) = &c {
                    let v = l.floats();
                    return Ok(Fx::Lit(Lit::Vec3([v[0], v[1], v[2]])));
                }
                let comp = |k: &str| Fx::Dyn("Break Color".into(), vec![("Value".into(), c.clone()), ("Component".into(), Fx::Choice(k.into()))]);
                Ok(Fx::Dyn(
                    "Make Vec3 from Floats".into(),
                    vec![("X".into(), comp("R")), ("Y".into(), comp("G")), ("Z".into(), comp("B"))],
                ))
            }
            "MakeCustomVectorFromBool" => {
                let c = get(self, "Bool", Unit::None, Ty::Bool)?.unwrap_or(Fx::Lit(Lit::Bool(false)));
                let t = get(self, "True Vector", unit, Ty::Vec3)?.unwrap_or(Fx::Lit(Lit::Vec3([0.0; 3])));
                let fv = get(self, "False Vector", unit, Ty::Vec3)?.unwrap_or(Fx::Lit(Lit::Vec3([0.0; 3])));
                Ok(Fx::Dyn(
                    "Select Vec3 by Bool".into(),
                    vec![("Condition".into(), c), ("If True".into(), t), ("If False".into(), fv)],
                ))
            }
            "SimulationPosition" => {
                // The system's position in world space, the origin in local
                // space: the effect's origin here.
                let _ = want;
                Ok(Fx::Lit(Lit::Vec3([0.0; 3])))
            }
            _ => Err(format!("dynamic input {name} has no equivalent here")),
        }
    }

    // ---- set variables and scratch pads ----

    fn assignment(&mut self, e: &NEmitter, m: &NModule, targets: &[(String, String)], o: &mut Out) -> Result<Option<String>, String> {
        let mut any = false;
        let mut failed = Vec::new();
        for (t, _) in targets {
            let (module, input, unit) = match t.as_str() {
                "Particles.Velocity" => ("Set Particle Velocity", "Velocity", Unit::Len),
                "Particles.Position" => ("Set Particle Position", "Position", Unit::Len),
                "Particles.Lifetime" => ("Set Particle Lifetime", "Lifetime", Unit::None),
                "Particles.Mass" => ("Set Particle Mass", "Mass", Unit::None),
                "Particles.SpriteSize" => ("Set Particle Sprite Size", "Sprite Size", Unit::Len),
                "Particles.SpriteRotation" => ("Set Particle Sprite Rotation", "Sprite Rotation", Unit::None),
                "Particles.Scale" => ("Set Particle Scale", "Scale", Unit::Dir),
                "Particles.Color" => ("Color", "Color", Unit::None),
                other => {
                    failed.push(format!("{other} (no equivalent attribute or module)"));
                    continue;
                }
            };
            let Some(def) = module_def(module) else { continue };
            let want = def_input_ty(def, input).unwrap_or(Ty::Float);
            let Some(i) = m.input(t) else {
                failed.push(format!("{t} (no value)"));
                continue;
            };
            match self.conv(e, i, unit, want) {
                Ok(v) => {
                    let mut x = FxModule::new(module);
                    x.set(input, v);
                    o.mods.push(x);
                    any = true;
                }
                Err(why) => failed.push(format!("{t}: {why}")),
            }
        }
        if !any {
            return Err(format!("Set Variables of {}", failed.join(", ")));
        }
        for f in failed {
            o.changes.push(format!("not set: {f}"));
        }
        Ok(None)
    }

    fn scratch_pad(&mut self, e: &NEmitter, st: &NStack, m: &NModule, nodes: &[crate::niagara::HlslNode], o: &mut Out) -> Result<Option<String>, String> {
        if nodes.is_empty() {
            return Err("scratch pad built from graph nodes (no custom HLSL to translate)".into());
        }
        if !matches!(st.usage, Usage::ParticleSpawn | Usage::ParticleUpdate | Usage::SimStage(_)) {
            return Err("custom HLSL outside particle stacks".into());
        }
        let label = format!("{} / {}", e.name, m.name);
        let mut inputs = Vec::new();
        for (k, v) in &m.inputs {
            if let Ok(fx) = self.conv(e, v, Unit::None, Ty::Float).or_else(|_| self.conv(e, v, Unit::None, Ty::Vec3)) {
                inputs.push((k.clone(), fx));
            }
        }
        match hlsl::translate(&m.name, nodes, &m.inputs, self.sys) {
            Ok(t) => {
                // Validate it on its own before keeping it.
                let probe = probe_effect(&t);
                let checked = fx_core::check_source("probe.fx.ron", &probe);
                if let Some(err) = checked.errors().next() {
                    let why = format!("translated, but the WGSL does not validate: {err}");
                    self.report.hlsl.push((label, why.clone()));
                    return Err(why);
                }
                if !e.gpu {
                    o.changes.push("custom WGSL modules run on the GPU only; the emitter is CPU here".into());
                }
                self.report.hlsl.push((label, "translated to a WGSL custom module".into()));
                let mut x = FxModule::new(&t.name);
                for (k, v) in &t.call_inputs {
                    x.set(k, v.clone());
                }
                o.mods.push(x);
                self.custom.push((t.name.clone(), t.source.clone()));
                Ok(None)
            }
            Err(why) => {
                self.report.hlsl.push((label, format!("not translated: {why}")));
                Err(format!("custom HLSL not translated: {why}"))
            }
        }
    }

    // ---- renderers ----

    fn renderer(&mut self, e: &NEmitter, r: &crate::niagara::NRenderer, written: [bool; 4]) -> Option<String> {
        use crate::property::find;
        let enum_tail = |n: &str| find(&r.properties, n).and_then(|v| v.as_str()).map(|s| s.rsplit("::").next().unwrap_or(s).to_string());
        let is_asset = |v: &String| v.starts_with('/') && !v.starts_with("/Script/");
        let material = r.refs.get("Material").filter(|v| is_asset(v)).cloned().or_else(|| {
            let mut m: Vec<_> = r
                .refs
                .iter()
                .filter(|(k, v)| (k.starts_with("OverrideMaterials") || k.ends_with("Material")) && is_asset(v))
                .collect();
            m.sort();
            m.first().map(|(_, v)| (*v).clone())
        });
        if !r.enabled {
            self.report.renderers.push(format!("{}: {} disabled: left out", e.name, r.class));
            return None;
        }
        let is_mesh = r.class == "NiagaraMeshRendererProperties";
        // Meshes: the first of the renderer's meshes, and its own material
        // when the renderer does not override it.
        let mut mesh_paths: Vec<(String, String)> = r
            .refs
            .iter()
            .filter(|(k, v)| k.starts_with("Meshes[") && k.ends_with("].Mesh") && is_asset(v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        mesh_paths.sort();
        let mesh_path = mesh_paths.first().map(|(_, v)| v.clone());
        let mesh_out = if is_mesh { mesh_path.as_deref().map(|p| self.mesh_asset(p)) } else { None };
        let material = material.or_else(|| match &mesh_out {
            Some(Ok(m)) => m.material.clone(),
            _ => None,
        });
        // Niagara's default materials are what our built-ins draw (particle
        // colour; sprites a soft round falloff).
        let engine_default = material.as_deref().is_some_and(|m| m.starts_with("/Niagara/DefaultAssets/"));
        let mat = if engine_default { None } else { material.as_deref().map(|m| self.material_asset(m, is_mesh, written)) };
        if let Some(Ok(m)) = &mat {
            if m.refraction {
                self.report.renderers.push(format!(
                    "{}: {} with a distortion (refraction) material `{}`: no refraction renderer here, left out (the heat haze is missing)",
                    e.name, r.class, m.name
                ));
                return None;
            }
        } else if material.as_deref().is_some_and(|m| {
            let l = m.to_lowercase();
            l.contains("distort") || l.contains("refract")
        }) {
            self.report.renderers.push(format!(
                "{}: {} with a distortion material (`{}`): no equivalent, left out",
                e.name,
                r.class,
                material.as_deref().unwrap_or("")
            ));
            return None;
        }
        let blend = match &mat {
            Some(Ok(m)) => m.blend.ours(is_mesh).to_string(),
            _ if engine_default => "Alpha".to_string(),
            _ => guess_blend(material.as_deref().unwrap_or(""), &e.name).to_string(),
        };
        let mut fields = Vec::new();
        // The material's files, or the reason there are none.
        let material_fields = |fields: &mut Vec<String>, report: &mut Report, what: &str| match &mat {
            Some(Ok(m)) => {
                if let Some(t) = &m.texture {
                    fields.push(format!("texture: {}", quote(t)));
                }
                fields.push(format!("material: {}", quote(&m.wesl)));
                report.materials.push(format!("{}: {what} material `{}` -> `{}`{}", e.name, m.name, m.wesl, m.describe()));
            }
            Some(Err(why)) => report.placeholders.push(format!(
                "{}: {what} material `{}` -> built-in, blend {blend} guessed from the name ({why})",
                e.name,
                material.as_deref().unwrap_or("")
            )),
            None if engine_default => report.materials.push(format!(
                "{}: {what} material `{}` (Niagara's default) -> our built-in {what} material, blend {blend}",
                e.name,
                crate::material::split_object_path(material.as_deref().unwrap_or("")).1
            )),
            None => report.placeholders.push(format!("{}: {what} has no material -> built-in, blend {blend}", e.name)),
        };
        let what = match r.class.as_str() {
            "NiagaraSpriteRendererProperties" => {
                fields.push(format!("blend_mode: {blend}"));
                match enum_tail("Alignment").as_deref() {
                    Some("VelocityAligned") => fields.push("alignment: VelocityAligned".into()),
                    Some("CustomAlignment") => fields.push("alignment: CustomAlignment".into()),
                    _ => {}
                }
                match enum_tail("FacingMode").as_deref() {
                    Some("FaceCameraPlane") => fields.push("facing_mode: FaceCameraPlane".into()),
                    Some("FaceCameraPosition") | Some("FaceCameraDistanceBlend") => fields.push("facing_mode: FaceCameraPosition".into()),
                    Some("CustomFacingVector") => fields.push("facing_mode: CustomFacingVector".into()),
                    _ => {}
                }
                // Same UV-space convention (V down): (0.5, 0.85) stands a
                // card on its particle (floor fires).
                if let Some(v) = find(&r.properties, "PivotInUVSpace") {
                    let x = v.field("X").and_then(|x| x.as_f64()).unwrap_or(0.5);
                    let y = v.field("Y").and_then(|x| x.as_f64()).unwrap_or(0.5);
                    if (x - 0.5).abs() > 1e-6 || (y - 0.5).abs() > 1e-6 {
                        fields.push(format!("pivot_in_uv_space: ({}, {})", f(x as f32), f(y as f32)));
                    }
                }
                match enum_tail("SortMode").as_deref() {
                    Some("ViewDepth") => fields.push("sort_mode: ViewDepth".into()),
                    Some("ViewDistance") => fields.push("sort_mode: ViewDistance".into()),
                    _ => {}
                }
                if let Some(v) = find(&r.properties, "SubImageSize") {
                    let x = v.field("X").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    let y = v.field("Y").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    if x > 1.0 || y > 1.0 {
                        if matches!(mat, Some(Ok(_))) {
                            fields.push(format!("sub_image_size: ({}, {})", x.max(1.0) as u32, y.max(1.0) as u32));
                            if matches!(find(&r.properties, "bSubImageBlend"), Some(crate::property::Value::Bool(true))) {
                                fields.push("sub_uv_blending: true".into());
                            }
                        } else {
                            self.report.placeholders.push(format!(
                                "{}: sprite flipbook {x}x{y} of the material's texture: not kept (the placeholder sprite has one frame)",
                                e.name
                            ));
                        }
                    }
                }
                material_fields(&mut fields, &mut self.report, "sprite");
                format!("Sprite(({}))", fields.join(", "))
            }
            "NiagaraMeshRendererProperties" => {
                let shape = match &mesh_out {
                    Some(Ok(m)) => {
                        let extra = if mesh_paths.len() > 1 {
                            format!(
                                "; Niagara picks one of {} meshes per particle, only the first is drawn ({} left out)",
                                mesh_paths.len(),
                                mesh_paths[1..].iter().map(|(_, p)| crate::material::split_object_path(p).1).collect::<Vec<_>>().join(", ")
                            )
                        } else {
                            String::new()
                        };
                        self.report.materials.push(format!("{}: mesh `{}` -> `{}` ({}){extra}", e.name, m.name, m.glb, m.summary));
                        m.glb.clone()
                    }
                    other => {
                        let shape = mesh_path.as_deref().map(guess_mesh).unwrap_or("cube");
                        let why = match other {
                            Some(Err(w)) => w.clone(),
                            _ => "no mesh".into(),
                        };
                        self.report.placeholders.push(format!(
                            "{}: mesh `{}` -> built-in \"{shape}\" (guessed from the name: {why}), {} m per unit of mesh scale",
                            e.name,
                            mesh_path.as_deref().unwrap_or("none"),
                            PLACEHOLDER_MESH_SIZE
                        ));
                        shape.to_string()
                    }
                };
                fields.push(format!("mesh: {}", quote(&shape)));
                if blend != "Opaque" && (matches!(mat, Some(Ok(_))) || blend != "Alpha" || material.as_deref().is_some_and(|m| m.to_lowercase().contains("translucent"))) {
                    fields.push(format!("blend_mode: {blend}"));
                }
                match enum_tail("FacingMode").as_deref() {
                    Some("Velocity") => fields.push("facing_mode: Velocity".into()),
                    Some("CameraPosition") => fields.push("facing_mode: CameraPosition".into()),
                    Some("CameraPlane") => fields.push("facing_mode: CameraPlane".into()),
                    _ => {}
                }
                material_fields(&mut fields, &mut self.report, "mesh");
                format!("Mesh(({}))", fields.join(", "))
            }
            "NiagaraRibbonRendererProperties" => {
                fields.push(format!("blend_mode: {blend}"));
                material_fields(&mut fields, &mut self.report, "ribbon");
                format!("Ribbon(({}))", fields.join(", "))
            }
            "NiagaraLightRendererProperties" => {
                let radius = find(&r.properties, "RadiusScale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let mut fields = vec![format!("radius_scale: {}", f(radius as f32))];
                // Niagara light colour (particle colour + ColorAdd) is a
                // brightness in candela-like units: 4π lumens per unit here.
                // Fire packs put all of it in ColorAdd (e.g. 10000, 4000, 0).
                if let Some(v) = find(&r.properties, "ColorAdd") {
                    let c = ["X", "Y", "Z"].map(|k| v.field(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32);
                    if c.iter().any(|x| *x != 0.0) {
                        fields.push(format!("color_add: ({}, {}, {})", f(c[0]), f(c[1]), f(c[2])));
                    }
                }
                fields.push(format!("intensity: {}", f(4.0 * std::f32::consts::PI)));
                // Off unless the renderer turns it on (Niagara's default).
                let alpha = matches!(find(&r.properties, "bAlphaScalesBrightness"), Some(crate::property::Value::Bool(true)));
                fields.push(format!("alpha_scales_brightness: {alpha}"));
                format!("Light(({}))", fields.join(", "))
            }
            other => {
                self.report.renderers.push(format!("{}: {other}: no equivalent renderer, left out", e.name));
                return None;
            }
        };
        self.report.renderers.push(format!("{}: {} -> `{}`", e.name, r.class, what));
        Some(what)
    }

    /// Size in metres of one unit of Niagara mesh scale for the emitter's
    /// mesh renderer: 1 for real meshes (already in metres), the
    /// placeholder size otherwise.
    fn mesh_unit(&mut self, e: &NEmitter) -> f32 {
        let Some(r) = e.renderers.iter().find(|r| r.class == "NiagaraMeshRendererProperties" && r.enabled) else {
            return PLACEHOLDER_MESH_SIZE;
        };
        let mut meshes: Vec<_> = r.refs.iter().filter(|(k, v)| k.starts_with("Meshes[") && k.ends_with("].Mesh") && v.starts_with('/')).collect();
        meshes.sort();
        match meshes.first().map(|(_, v)| (*v).clone()) {
            Some(p) if self.mesh_asset(&p).is_ok() => 1.0,
            _ => PLACEHOLDER_MESH_SIZE,
        }
    }

    /// A static mesh written as glTF beside the effect (once per mesh).
    fn mesh_asset(&mut self, path: &str) -> Result<MeshOut, String> {
        let Some(a) = self.assets.as_mut() else {
            return Err("asset packages not searched".into());
        };
        if let Some(m) = a.meshes.get(path) {
            return m.clone();
        }
        let res = (|| {
            let l = a.store.load(path).ok_or_else(|| "its package is not in the package folders".to_string())?;
            let data = crate::mesh::read_static_mesh(&l.pkg, &l.bytes)?;
            // Our mesh renderer draws one primitive: material slots merge.
            let mut one = data.clone();
            one.triangle_slots.iter_mut().for_each(|s| *s = 0);
            one.slot_names.truncate(1);
            let name = crate::material::split_object_path(path).1;
            let glb = format!("{}/{name}.glb", a.dir);
            a.files.insert(glb.clone(), crate::mesh::write_glb(&one));
            // The mesh's own first material.
            let i = l.export(&name);
            let material = i.and_then(|i| match crate::property::find(l.props(i), "StaticMaterials") {
                Some(crate::property::Value::Array(s)) => s.iter().find_map(|m| match m.field("MaterialInterface") {
                    Some(crate::property::Value::Object(o)) if !o.is_null() => Some(l.path(*o)),
                    _ => None,
                }),
                _ => None,
            });
            let (lo, hi) = data.bounds();
            Ok(MeshOut {
                name,
                glb,
                material,
                summary: format!(
                    "{} triangles, {:.2} x {:.2} x {:.2} m",
                    data.triangle_count(),
                    hi[0] - lo[0],
                    hi[1] - lo[1],
                    hi[2] - lo[2]
                ),
            })
        })();
        a.meshes.insert(path.to_string(), res.clone());
        res
    }

    /// A material translated to WESL plus its texture atlas, written beside
    /// the effect (once per material and use).
    fn material_asset(&mut self, path: &str, mesh: bool, written: [bool; 4]) -> Result<MatOut, String> {
        let Some(a) = self.assets.as_mut() else {
            return Err("asset packages not searched".into());
        };
        let key = (path.to_string(), mesh, written);
        if let Some(m) = a.mats.get(&key) {
            return m.clone();
        }
        let res = (|| {
            use crate::material as mt;
            let m = mt::Material::load(&mut a.store, path)?;
            if let Some(miss) = &m.missing {
                return Err(format!("parent material `{miss}` is not in the package folders"));
            }
            let available = [0, 1, 2, 3].map(|k| match k {
                0 => true,
                k => fx_core::modules::MODULES.iter().any(|d| d.name == format!("Dynamic Material Parameters {}", k + 1)),
            });
            let needs = mt::Needs {
                dynamic_written: written,
                dynamic_available: available,
                mesh,
            };
            let c = mt::compile(&m, &needs)?;
            // A pure distortion material: Refraction wired and no colour
            // output. Translucent materials that also wire Emissive or Base
            // Color (a visible trail with a little haze) are drawn; only
            // their refraction is lost.
            let wired = |input: &str| {
                m.base.as_ref().is_some_and(|(l, ed)| {
                    crate::property::find(l.props(*ed), input)
                        .and_then(|i| i.field("Expression"))
                        .is_some_and(|x| matches!(x, crate::property::Value::Object(o) if !o.is_null()))
                })
            };
            // Any blend mode: an additive heat haze with no colour draws nothing useful.
            let refraction = wired("Refraction") && !wired("EmissiveColor") && !wired("BaseColor");
            let refraction_lost = wired("Refraction") && !refraction;
            let mut imgs = Vec::new();
            let mut missing = Vec::new();
            for t in &c.textures {
                match mt::read_texture(&mut a.store, t) {
                    Ok(i) => imgs.push(i),
                    Err(why) => {
                        missing.push(format!("{} ({why})", mt::split_object_path(t).1));
                        imgs.push(mt::stand_in_missing());
                    }
                }
            }
            let mut base = m.name();
            let mut n = 2;
            while a.files.contains_key(&format!("{}/{base}.wesl", a.dir)) {
                base = format!("{}_{n}", m.name());
                n += 1;
            }
            let wesl_path = format!("{}/{base}.wesl", a.dir);
            let (texture, wesl) = if imgs.is_empty() {
                (None, mt::with_rects(&c.wesl, &[]))
            } else {
                let at = mt::atlas(&imgs);
                let png = format!("{}/{base}.png", a.dir);
                a.files.insert(png.clone(), mt::encode_png(&at.image));
                (Some(png), mt::with_rects(&c.wesl, &at.rects))
            };
            a.files.insert(wesl_path.clone(), wesl.into_bytes());
            Ok(MatOut {
                name: m.name(),
                wesl: wesl_path,
                texture,
                blend: m.blend,
                unlit: m.unlit,
                refraction,
                textures: c.textures.iter().map(|t| mt::split_object_path(t).1).collect(),
                missing_textures: missing,
                notes: {
                    let mut notes = c.notes.clone();
                    if refraction_lost {
                        notes.push("Refraction: the distortion offset is not drawn (colour kept)".into());
                    }
                    notes
                },
            })
        })();
        a.mats.insert(key, res.clone());
        res
    }

    /// A capacity from the spawn rates, bursts and lifetimes written.
    fn estimate_capacity(&self, _e: &NEmitter, out: &FxEmitter) -> u32 {
        let num = |v: Option<&Fx>, default: f32| -> f32 {
            match v {
                Some(Fx::Lit(l)) => l.floats()[0],
                Some(Fx::Range(_, b)) => b.floats()[0],
                Some(Fx::Link(_, n)) => self.user(n).and_then(|v| v.floats()).map(|f| f[0]).unwrap_or(default),
                Some(Fx::Dyn(_, args)) => args
                    .iter()
                    .filter_map(|(_, a)| match a {
                        Fx::Lit(l) => Some(l.floats()[0]),
                        Fx::Range(_, b) => Some(b.floats()[0]),
                        Fx::Link(_, n) => self.user(n).and_then(|v| v.floats()).map(|f| f[0]),
                        _ => None,
                    })
                    .fold(default, |acc, x| if acc == default { x } else { acc * x }),
                _ => default,
            }
        };
        let mut rate = 0.0f32;
        let mut burst = 0.0f32;
        let mut life = 0.0f32;
        for (_, mods) in &out.stacks {
            for m in mods {
                match m.name.as_str() {
                    "Spawn Rate" => rate += num(m.get("Spawn Rate"), 10.0).max(0.0),
                    "Spawn Burst Instantaneous" => burst += num(m.get("Spawn Count"), 10.0).max(0.0),
                    "Spawn Burst Interval" => burst += num(m.get("Spawn Count"), 10.0).max(0.0),
                    "Spawn Per Frame" => rate += 60.0 * num(m.get("Spawn Count"), 1.0).max(0.0),
                    "Initialize Particle" => life = life.max(num(m.get("Lifetime"), 2.0)),
                    "Set Particle Lifetime" => life = life.max(num(m.get("Lifetime"), 2.0)),
                    _ => {}
                }
            }
        }
        if life <= 0.0 {
            life = 2.0;
        }
        let n = rate * life * 1.5 + burst * (1.0 + (life / 1.0).ceil().min(4.0));
        (n.ceil() as u32).clamp(64, 262_144).next_power_of_two().min(262_144)
    }

    // ---- writing ----

    fn write(&self, emitters: &[FxEmitter]) -> String {
        let mut o = String::new();
        let r = &self.report;
        let _ = writeln!(o, "// Converted from the Unreal Niagara system {} ({}) by `fximport convert`.", self.sys.name, r.source);
        let _ = writeln!(
            o,
            "// Modules mapped: {}/{} ({:.0}%); the report beside this file lists the rest.",
            r.mapped(),
            r.total(),
            r.percent()
        );
        let _ = writeln!(o, "// Units: centimetres -> metres; Unreal (X, Y, Z) -> (X, Z, Y), Z up -> Y up (mirrored).");
        if r.placeholders.is_empty() {
            let _ = writeln!(o, "// Materials, textures and meshes: read from their Unreal packages (see the report).");
        } else {
            let _ = writeln!(o, "// Placeholders where a material or mesh package was missing: built-in sprite (blend guessed) or shape.");
        }
        let _ = writeln!(o, "(");
        let _ = writeln!(o, "    name: {},", quote(&self.sys.name));
        if self.sys.warmup_time > 0.0 {
            let _ = writeln!(o, "    warmup_time: {},", f(self.sys.warmup_time));
        }
        let users: BTreeMap<&str, String> = self
            .sys
            .user_parameters
            .iter()
            .map(|(n, _)| n.as_str())
            .chain(self.user_overrides.keys().map(String::as_str))
            .filter_map(|n| self.user(n).and_then(|v| self.user_decl(n, v)).map(|d| (n, d)))
            .collect();
        if !users.is_empty() {
            let _ = writeln!(o, "    user_parameters: {{");
            for (n, d) in &users {
                let _ = writeln!(o, "        {}: {d},", quote(n));
            }
            let _ = writeln!(o, "    }},");
        }
        if !self.custom.is_empty() {
            let _ = writeln!(o, "    custom_modules: [");
            for (_, src) in &self.custom {
                for line in src.lines() {
                    let _ = writeln!(o, "        {line}");
                }
            }
            let _ = writeln!(o, "    ],");
        }
        let _ = writeln!(o, "    emitters: [");
        for e in emitters {
            let _ = writeln!(o, "        (");
            for c in &e.comment {
                let _ = writeln!(o, "            // {c}");
            }
            let _ = writeln!(o, "            name: {},", quote(&e.name));
            let _ = writeln!(o, "            sim_target: {},", if e.gpu { "Gpu" } else { "Cpu" });
            let _ = writeln!(o, "            max_particles: {},", e.max_particles);
            if !e.deterministic {
                let _ = writeln!(o, "            deterministic: false,");
            }
            if e.interpolated_spawning {
                let _ = writeln!(o, "            interpolated_spawning: true,");
            }
            if !e.local_space {
                let _ = writeln!(o, "            local_space: false,");
            }
            if let Some((a, b)) = e.fixed_bounds {
                let _ = writeln!(
                    o,
                    "            fixed_bounds: (({}, {}, {}), ({}, {}, {})),",
                    f(a[0]),
                    f(a[1]),
                    f(a[2]),
                    f(b[0]),
                    f(b[1]),
                    f(b[2])
                );
            }
            for (name, mods) in &e.stacks {
                fx::write_stack(&mut o, "            ", name, mods);
            }
            if !e.events.is_empty() {
                let _ = writeln!(o, "            event_handlers: [");
                for (head, mods) in &e.events {
                    let _ = writeln!(o, "                {head}");
                    fx::write_modules(&mut o, "                    ", mods);
                    let _ = writeln!(o, "                ]),");
                }
                let _ = writeln!(o, "            ],");
            }
            if !e.stages.is_empty() {
                let _ = writeln!(o, "            simulation_stages: [");
                for s in &e.stages {
                    let _ = writeln!(o, "            {s}");
                }
                let _ = writeln!(o, "            ],");
            }
            if !e.renderers.is_empty() {
                let _ = writeln!(o, "            renderers: [{}],", e.renderers.join(", "));
            }
            let _ = writeln!(o, "        ),");
        }
        let _ = writeln!(o, "    ],");
        let _ = writeln!(o, ")");
        o
    }

    /// A user parameter declaration, converted with the unit it is used with.
    fn user_decl(&self, name: &str, v: &NValue) -> Option<String> {
        let unit = self.user_units.get(name).copied().unwrap_or(Unit::None);
        Some(match v {
            NValue::Curve(c) => {
                let ty = curve_ty(c);
                let cv = curve(c, ty, unit);
                let body = fx::write_curve(&cv);
                // `Float([...])` -> `FloatCurve([...])`.
                let keys = &body[body.find('(').unwrap_or(0)..];
                format!("{}{keys}", ty.curve_decl())
            }
            NValue::DataInterface(c) if c.contains("StaticMesh") => "StaticMesh(\"cube\")".into(),
            NValue::DataInterface(_) | NValue::Enum(_) | NValue::Matrix(_) => return None,
            other => fx::write_lit(&match value_fx(other, unit).ok()? {
                Fx::Lit(l) => l,
                _ => return None,
            }),
        })
    }
}

/// A tiny effect holding one custom module, to validate its WGSL.
fn probe_effect(t: &hlsl::Translated) -> String {
    let mut src = String::new();
    for line in t.source.lines() {
        let _ = writeln!(src, "        {line}");
    }
    format!(
        "(\n    name: \"probe\",\n    custom_modules: [\n{src}    ],\n    emitters: [(\n        name: \"E\",\n        sim_target: Gpu,\n        max_particles: 16,\n        particle_update: [(module: {})],\n    )],\n)\n",
        quote(&t.name)
    )
}

// ---- module table ----

pub(crate) fn module_def(name: &str) -> Option<&'static fx_core::modules::ModuleDef> {
    fx_core::modules::MODULES.iter().find(|m| m.name == name)
}

fn def_input_ty(def: &fx_core::modules::ModuleDef, input: &str) -> Option<Ty> {
    def.input(input).and_then(|(_, i)| Ty::from_name(i.ty.name()))
}

/// Maps one Niagara module (by its script) to ours.
fn map_module(cx: &mut Cx, e: &NEmitter, st: &NStack, m: &NModule, loops: &Loops, o: &mut Out) -> Result<Option<String>, String> {
    let script = m.script_name().to_string();
    // Inputs consumed by the mapping; the others are reported as dropped.
    let mut used: Vec<&str> = Vec::new();
    // Puts a Niagara input into an output module input.
    macro_rules! put {
        ($x:expr, $from:expr, $to:expr, $unit:expr) => {{
            used.push($from);
            let def = module_def(&$x.name).expect("known module");
            let want = def_input_ty(def, $to).expect("known input");
            if let Some(v) = cx.input(e, m, $from, $unit, want, o) {
                $x.set($to, v);
            }
        }};
    }
    match script.as_str() {
        "EmitterState" => {
            let mut x = FxModule::new("Emitter State");
            let own = m.has("Loop Duration") || m.switch("Life Cycle Mode") == Some("NewEnumerator1");
            let (behavior, duration, delay, count) = if own {
                (
                    m.switch("Loop Behavior").map(str::to_string),
                    lit_f(m.input("Loop Duration")),
                    lit_f(m.input("Loop Delay")),
                    lit_f(m.input("Loop Count")).map(|v| v as i32),
                )
            } else {
                (loops.behavior.clone(), loops.duration, loops.delay, loops.count)
            };
            used.extend(["Loop Duration", "Loop Delay", "Loop Count"]);
            // ENiagaraLoopBehavior as saved: 0 Infinite, 1 Once, 2 Multiple
            // (Multiple also shows as a Loop Count in the compiled stack).
            let b = if count.is_some() && m.has("Loop Count") {
                "Multiple"
            } else {
                match behavior.as_deref() {
                    Some("NewEnumerator1") => "Once",
                    Some("NewEnumerator2") => "Multiple",
                    _ => "Infinite",
                }
            };
            x.set("Loop Behavior", Fx::Choice(b.into()));
            if b == "Multiple" {
                x.set("Loop Count", Fx::Lit(Lit::Int(count.unwrap_or(1).max(1))));
            }
            match duration {
                Some(d) if d > 0.0 => x.set("Loop Duration", Fx::Lit(Lit::Float(d))),
                _ => x.set("Loop Duration Mode", Fx::Choice("Infinite".into())),
            }
            if let Some(d) = delay.filter(|d| *d > 0.0) {
                x.set("Loop Delay", Fx::Lit(Lit::Float(d)));
            }
            if !own {
                o.changes.push("Life Cycle Mode System: loop settings taken from the system's System State".into());
            }
            o.mods.push(x);
        }
        "SpawnRate" => {
            let mut x = FxModule::new("Spawn Rate");
            put!(x, "SpawnRate", "Spawn Rate", Unit::None);
            used.push("Spawn Probability");
            if let Some(p) = lit_f(m.input("Spawn Probability")).filter(|p| *p < 1.0 && m.switch("Use Spawn Probability") != Some("false")) {
                let rate = x.get("Spawn Rate").cloned().unwrap_or(Fx::Lit(Lit::Float(10.0)));
                let v = match rate {
                    Fx::Lit(Lit::Float(r)) => Fx::Lit(Lit::Float(r * p)),
                    other => Fx::Dyn("Multiply Float".into(), vec![("A".into(), other), ("B".into(), Fx::Lit(Lit::Float(p)))]),
                };
                x.set("Spawn Rate", v);
                o.changes.push(format!("Spawn Probability {p} folded into the rate"));
            }
            o.mods.push(x);
        }
        "SpawnBurst_Instantaneous" => {
            let mut x = FxModule::new("Spawn Burst Instantaneous");
            put!(x, "Spawn Count", "Spawn Count", Unit::None);
            put!(x, "Spawn Time", "Spawn Time", Unit::None);
            put!(x, "Spawn Probability", "Spawn Probability", Unit::None);
            o.mods.push(x);
        }
        "SpawnPerUnit" => {
            let mut x = FxModule::new("Spawn Per Unit");
            used.push("Spawn Spacing");
            if let Some(v) = cx.input(e, m, "Spawn Spacing", Unit::Len, Ty::Float, o) {
                x.set("Spawn Spacing", v);
            }
            o.mods.push(x);
        }
        "InitializeParticle" => init_particle(cx, e, m, o, &mut used),
        "ShapeLocation" => {
            let mut x = FxModule::new("Shape Location");
            // ENiagara_LocationShapes as saved: 0 Sphere, 2 Box / Plane, 3 Cylinder,
            // 4 Torus, 5 Cone, 6 Ring / Disc (read off the test files; the
            // size inputs present confirm it).
            let by_switch = match m.switch("Shape Primitive") {
                Some("NewEnumerator0") => Some("Sphere"),
                Some("NewEnumerator1") | Some("NewEnumerator2") => Some("Box"),
                Some("NewEnumerator3") => Some("Cylinder"),
                Some("NewEnumerator4") => Some("Torus"),
                Some("NewEnumerator5") => Some("Cone"),
                Some("NewEnumerator6") => Some("Ring"),
                _ => None,
            };
            let shape = if let Some(s) = by_switch {
                s
            } else if m.has("Box Size") {
                "Box"
            } else if m.has("Cylinder Height") || m.has("Cylinder Radius") {
                "Cylinder"
            } else if m.has("Torus Radius") || m.has("Large Radius") {
                "Torus"
            } else if m.has("Cone Length") || m.has("Cone Angle") {
                "Cone"
            } else if m.has("Ring Radius") {
                "Ring"
            } else if m.has("Sphere Radius") {
                "Sphere"
            } else {
                return Err("shape not recognised (none of its size inputs are set)".into());
            };
            x.set("Shape Primitive", Fx::Choice(shape.into()));
            if m.switch("Surface Only") == Some("true") {
                x.set("Surface Only", Fx::Lit(Lit::Bool(true)));
            }
            used.extend(["Sphere Surface Distribution", "Hemisphere Distribution", "Surface Only Band Thickness"]);
            // Inputs of the other shapes (stale constants) do not matter.
            for (k, _) in &m.inputs {
                let other = ["Sphere", "Box", "Cylinder", "Torus", "Cone", "Ring", "Disc", "Plane", "Large Radius", "Handle Radius", "Wedge", "Hemisphere"];
                if other.iter().any(|p| k.starts_with(p)) && !k.starts_with(shape) {
                    used.push(k.as_str());
                }
            }
            // U / V / Surface Distribution place points directly in the
            // Direct distribution modes; the default Random one (0) ignores them.
            let dist_switch = match shape {
                "Sphere" => Some("Sphere Distribution"),
                "Cylinder" => Some("Cylinder Distribution"),
                "Torus" => Some("Torus Distribution Mode"),
                "Ring" => Some("Ring / Disc Distribution Mode"),
                _ => None,
            };
            if dist_switch.is_none_or(|s| mode(m, s).unwrap_or(0) == 0) {
                used.extend(["U Distribution", "V Distribution", "Surface Distribution"]);
            }
            // The torus knot inputs matter only for a torus with Enable Torus Knot Radius.
            if shape != "Torus" || m.switch("Enable Torus Knot Radius") != Some("true") {
                used.extend(["Knot Helix Loops", "Knot Ring Loops", "Torus Knot Tube Radius"]);
            }
            match shape {
                "Sphere" => put!(x, "Sphere Radius", "Sphere Radius", Unit::Len),
                "Box" => {
                    // Niagara: (random - Box Midpoint) * Box Size, as ours.
                    put!(x, "Box Midpoint", "Box Midpoint", Unit::Dir);
                    put!(x, "Box Size", "Box Size", Unit::Len)
                }
                "Cylinder" => {
                    // Midpoint 0.5 is centred, as ours.
                    if lit_f(m.input("Cylinder Height Midpoint")) == Some(0.5) {
                        used.push("Cylinder Height Midpoint");
                    }
                    put!(x, "Cylinder Height", "Cylinder Height", Unit::Len);
                    put!(x, "Cylinder Radius", "Cylinder Radius", Unit::Len);
                }
                "Torus" => {
                    put!(x, "Torus Radius", "Large Radius", Unit::Len);
                    put!(x, "Large Radius", "Large Radius", Unit::Len);
                    put!(x, "Handle Radius", "Handle Radius", Unit::Len);
                }
                "Cone" => {
                    // Niagara's cone angles are full openings (its cone helper
                    // halves them); ours are half-angles. Cone Surface
                    // Distribution: the distance from the apex is cbrt(random
                    // in Distribution..1) of Cone Length, as ours.
                    put!(x, "Cone Length", "Cone Length", Unit::Len);
                    put!(x, "Cone Surface Distribution", "Cone Surface Distribution", Unit::None);
                    if mode(m, "Cone Mode") == Some(1) {
                        // Wedge: random yaw and pitch within ± half the
                        // horizontal and vertical angles; drawn as a cone of
                        // their mean.
                        used.extend(["Cone Angle", "Cone Inner Angle", "Cone Radial Angle", "Wedge Horizontal Angle", "Wedge Vertical Angle"]);
                        let h = lit_f(m.input("Wedge Horizontal Angle")).unwrap_or(90.0);
                        let v = lit_f(m.input("Wedge Vertical Angle")).unwrap_or(90.0);
                        x.set("Cone Angle", Fx::Lit(Lit::Float((h + v) * 0.25)));
                        o.changes.push(format!("Cone Mode Wedge ({h}° x {v}°): drawn as a cone of half-angle {}°", (h + v) * 0.25));
                    } else {
                        used.extend(["Cone Angle", "Cone Inner Angle"]);
                        for (from, to) in [("Cone Angle", "Cone Angle"), ("Cone Inner Angle", "Cone Inner Angle")] {
                            if let Some(v) = cx.input(e, m, from, Unit::None, Ty::Float, o) {
                                x.set(to, scale_fx(v, 0.5));
                            }
                        }
                        put!(x, "Cone Radial Angle", "Cone Radial Angle", Unit::None);
                    }
                }
                _ => {
                    put!(x, "Ring Radius", "Ring Radius", Unit::Len);
                    put!(x, "Disc Coverage", "Disc Coverage", Unit::None);
                }
            }
            put!(x, "Offset", "Offset", Unit::Len);
            o.mods.push(x);
        }
        "AddVelocity" => {
            used.extend([
                "Velocity Speed Scale",
                "UseConeFalloff",
                "UseVelDistribution",
                "Constrain To Radius",
                "Radius Falloff Exponent",
                "Radius Falloff Near / Far",
                "Velocity Origin",
            ]);
            let speed_scale = lit_f(m.input("Velocity Speed Scale")).unwrap_or(1.0);
            // UE4's per-axis multiplier: the identity changes nothing.
            if matches!(m.input("Scale Added Velocity"), Some(NInput::Value(v)) if v.floats().is_some_and(|f| f.iter().all(|&c| c == 1.0))) {
                used.push("Scale Added Velocity");
            }
            // Velocity Mode as saved: 0 Linear, 1 From Point, 2 In Cone.
            let vmode = mode(m, "Velocity Mode").unwrap_or(if m.has("Cone Angle") || m.has("Cone Axis") {
                2
            } else if m.has("Velocity Speed") && !m.has("Velocity") {
                1
            } else {
                0
            });
            for (k, _) in &m.inputs {
                let cone = k.starts_with("Cone") || k.contains("Cone Axis") || k == "Inner Cone Angle" || k == "Speed Falloff From Cone Axis" || k == "Distribution Along Cone Axis";
                let stale = match vmode {
                    0 => cone || k == "Velocity Speed" || k == "Velocity Origin",
                    1 => cone || k == "Velocity",
                    _ => k == "Velocity Origin",
                };
                if stale {
                    used.push(k.as_str());
                }
            }
            if vmode == 2 {
                let mut x = FxModule::new("Add Velocity in Cone");
                put!(x, "Cone Axis", "Cone Axis", Unit::Dir);
                used.push("Cone Angle");
                if let Some(a) = lit_f(m.input("Cone Angle")) {
                    // Niagara's cone angle is the full opening; ours the half-angle.
                    x.set("Cone Angle", Fx::Lit(Lit::Float((a * 0.5).clamp(0.0, 180.0))));
                }
                used.push("Velocity Speed");
                if let Some(v) = cx.input(e, m, "Velocity Speed", Unit::Len, Ty::Float, o) {
                    x.set("Velocity Strength", scale_fx(v, speed_scale));
                } else if let Some(NInput::Value(v)) = m.input("Velocity") {
                    // Speed given as the Velocity vector's length.
                    if let Some(f) = v.floats() {
                        let len = (f.iter().map(|x| x * x).sum::<f32>()).sqrt() * CM * speed_scale;
                        x.set("Velocity Strength", Fx::Lit(Lit::Float(len)));
                    }
                }
                used.push("Velocity");
                // The compiled AddVelocity (In Cone): cos θ uniform between the
                // halves of Cone Angle and Inner Cone Angle (full openings), the
                // cone angle scaled by a random saturate(lerp(1, -0.5,
                // Distribution)..1) when UseVelDistribution, the speed by
                // lerp(1, saturate(cos θ) ^ (10 * Falloff), Falloff) when UseConeFalloff.
                let off = |k: &str| matches!(m.input(k), Some(NInput::Value(NValue::Bool(false))));
                used.push("Inner Cone Angle");
                if let Some(v) = cx.input(e, m, "Inner Cone Angle", Unit::None, Ty::Float, o) {
                    x.set("Inner Cone Angle", scale_fx(v, 0.5));
                }
                if off("UseVelDistribution") {
                    used.push("Distribution Along Cone Axis");
                } else {
                    put!(x, "Distribution Along Cone Axis", "Distribution Along Cone Axis", Unit::None);
                }
                if off("UseConeFalloff") {
                    used.push("Speed Falloff From Cone Axis");
                } else {
                    put!(x, "Speed Falloff From Cone Axis", "Speed Falloff", Unit::None);
                }
                // Origin Offset moves the point of the From Point mode only.
                used.push("Origin Offset");
                o.mods.push(x);
            } else if vmode == 1 {
                let mut x = FxModule::new("Add Velocity from Point");
                used.push("Velocity Speed");
                if let Some(v) = cx.input(e, m, "Velocity Speed", Unit::Len, Ty::Float, o) {
                    x.set("Velocity Strength", scale_fx(v, speed_scale));
                }
                put!(x, "Velocity Origin", "Velocity Origin", Unit::Len);
                // The compiled From Point mode measures from Velocity Origin + Origin Offset.
                used.push("Origin Offset");
                if let Some(v) = cx.input(e, m, "Origin Offset", Unit::Len, Ty::Vec3, o) {
                    if v != Fx::Lit(Lit::Vec3([0.0; 3])) {
                        let base = x.get("Velocity Origin").cloned().unwrap_or(Fx::Lit(Lit::Vec3([0.0; 3])));
                        x.set("Velocity Origin", add_vec3(base, v));
                    }
                }
                o.mods.push(x);
            } else {
                let mut x = FxModule::new("Add Velocity");
                used.extend(["Velocity", "Origin Offset"]);
                if let Some(v) = cx.input(e, m, "Velocity", Unit::Len, Ty::Vec3, o) {
                    x.set("Velocity", scale_fx(v, speed_scale));
                }
                o.mods.push(x);
            }
        }
        "ParticleState" => {
            let mut x = FxModule::new("Particle State");
            if m.switch("Kill Particles When Lifetime Has Elapsed") == Some("false") {
                x.set("Kill Particles When Lifetime Has Elapsed", Fx::Lit(Lit::Bool(false)));
            }
            used.push("Loop Particles Lifetime");
            if matches!(m.input("Loop Particles Lifetime"), Some(NInput::Value(NValue::Bool(true)))) {
                x.set("Kill Particles When Lifetime Has Elapsed", Fx::Lit(Lit::Bool(false)));
                o.changes.push("Loop Particles Lifetime: particles are kept alive instead of restarting their age".into());
            }
            o.mods.push(x);
        }
        "SolveForcesAndVelocity" => {
            o.mods.push(FxModule::new("Solve Forces and Velocity"));
            used.extend([
                "Acceleration Limit",
                "Manually Enable Rotational Solver",
                "Mesh Orientation",
                "Rotational Force",
                "Rotational Velocity",
            ]);
            if m.switch("Clamp Velocity") == Some("true") {
                let mut l = FxModule::new("Limit Velocity");
                put!(l, "Speed Limit", "Velocity Limit", Unit::Len);
                o.mods.push(l);
            } else {
                used.push("Speed Limit");
            }
        }
        "GravityForce" => {
            let mut x = FxModule::new("Gravity Force");
            put!(x, "Gravity", "Gravity", Unit::Len);
            o.mods.push(x);
        }
        "Drag" => {
            let mut x = FxModule::new("Drag");
            put!(x, "Drag", "Drag", Unit::None);
            used.push("Rotational Drag");
            o.mods.push(x);
        }
        "WindForce" => {
            let mut x = FxModule::new("Wind Force");
            put!(x, "Wind Speed", "Wind Velocity", Unit::Len);
            put!(x, "Wind Speed Scale", "Wind Speed", Unit::None);
            put!(x, "Air Resistance", "Air Resistance", Unit::None);
            // Friction and surface alignment act near the landscape / scene
            // depth (CollisionQuery, Landscape Sample); with no surface in
            // reach, as here, they leave the wind as it is.
            used.extend([
                "Friction",
                "Friction Distance Limit",
                "Normal Friction Mask",
                "Alignment Falloff Start / End",
                "Turbulence Flow Loop Duration",
            ]);
            // Turbulence (Turbulence Mode 0 = none): Niagara samples a 32³
            // wind vector field at Position * 0.125 * Frequency per cm, about
            // one tile every 2.56 m / Frequency, with a strength of the wind's
            // speed * Scale; here curl noise of 0.4 * Frequency cells per metre.
            if mode(m, "Turbulence Mode") == Some(0) {
                used.extend(["Scale", "Frequency", "Offset By Time", "Position"]);
            } else if m.has("Scale") {
                put!(x, "Scale", "Turbulence Scale", Unit::None);
                used.push("Frequency");
                if let Some(v) = cx.input(e, m, "Frequency", Unit::None, Ty::Float, o) {
                    x.set("Turbulence Frequency", scale_fx(v, 0.4));
                }
                put!(x, "Offset By Time", "Turbulence Offset By Time", Unit::Len);
                put!(x, "Position", "Turbulence Position", Unit::Len);
                o.changes.push("turbulence: Niagara's wind vector field (advected over Turbulence Flow Loop Duration) is curl noise here".into());
            }
            o.mods.push(x);
        }
        "PointForce" => {
            let mut x = FxModule::new("Point Force");
            put!(x, "Force Strength", "Point Force Strength", Unit::Len);
            put!(x, "Force Falloff Distance", "Falloff Radius", Unit::Len);
            put!(x, "Point", "Point Position", Unit::Len);
            used.push("Use Falloff");
            o.mods.push(x);
        }
        "PointAttractionForce" => {
            let mut x = FxModule::new("Point Attraction Force");
            put!(x, "Attractor Position", "Attractor Position", Unit::Len);
            put!(x, "Attraction Strength", "Attraction Strength", Unit::Len);
            put!(x, "Attraction Radius", "Attraction Radius", Unit::Len);
            put!(x, "Falloff Exponent", "Falloff Exponent", Unit::None);
            put!(x, "Kill Radius", "Kill Radius", Unit::Len);
            used.push("Position");
            if let Some(off) = m.input("Attractor Position Offset") {
                used.push("Attractor Position Offset");
                if let Ok(off) = cx.conv(e, off, Unit::Len, Ty::Vec3) {
                    let base = x.get("Attractor Position").cloned().unwrap_or(Fx::Lit(Lit::Vec3([0.0; 3])));
                    x.set("Attractor Position", add_vec3(base, off));
                } else {
                    o.drop_input("Attractor Position Offset", "not readable");
                }
            }
            // Position To Attract (default Particles.Position): ours measures
            // from Particles.Position + Attracted Position Offset.
            used.push("Position To Attract");
            if let Some(i) = m.input("Position To Attract") {
                match cx.conv(e, i, Unit::Len, Ty::Vec3) {
                    Ok(v) => {
                        let is_pos = |f: &Fx| matches!(f, Fx::Link(_, n) if n == "Particles.Position");
                        let offset = match v {
                            ref p if is_pos(p) => None,
                            Fx::Dyn(ref n, ref args) if n == "Add Vec3" && args.len() == 2 && is_pos(&args[0].1) => Some(args[1].1.clone()),
                            other => Some(Fx::Dyn(
                                "Add Vec3".into(),
                                vec![
                                    ("A".into(), other),
                                    (
                                        "B".into(),
                                        Fx::Dyn(
                                            "Multiply Vec3 by Float".into(),
                                            vec![
                                                ("A".into(), Fx::Link("Vec3".into(), "Particles.Position".into())),
                                                ("B".into(), Fx::Lit(Lit::Float(-1.0))),
                                            ],
                                        ),
                                    ),
                                ],
                            )),
                        };
                        if let Some(off) = offset {
                            x.set("Attracted Position Offset", off);
                        }
                    }
                    Err(why) => o.drop_input("Position To Attract", &why),
                }
            }
            o.mods.push(x);
        }
        "CurlNoiseForce" => {
            let mut x = FxModule::new("Curl Noise Force");
            put!(x, "Noise Strength", "Noise Strength", Unit::Len);
            used.push("Noise Frequency");
            if let Some(v) = lit_f(m.input("Noise Frequency")) {
                // Niagara samples the field at Position (cm) * Frequency / 1000
                // (CurlNoiseForce V2, NoiseFrequencyScaled): per metre, F / 10.
                x.set("Noise Frequency", Fx::Lit(Lit::Float(v / 10.0)));
            }
            o.mods.push(x);
        }
        "VortexVelocity" | "VortexForce" => {
            let mut x = FxModule::new(if script == "VortexForce" { "Vortex Force" } else { "Vortex Velocity" });
            put!(x, "Vortex Axis", "Vortex Axis", Unit::Dir);
            o.mods.push(x);
        }
        "Collision" => {
            let mut x = FxModule::new("Collision");
            x.set("Collision Type", Fx::Choice("Plane".into()));
            put!(x, "Restitution", "Restitution", Unit::None);
            put!(x, "Friction", "Friction", Unit::None);
            o.changes.push("Niagara collides with the scene; here with the ground plane (Plane, y = 0)".into());
            for (k, _) in &m.inputs {
                used.push(k.as_str());
            }
            o.mods.push(x);
        }
        "Color" => {
            let mut x = FxModule::new("Color");
            put!(x, "Color", "Color", Unit::None);
            used.extend(["Scale Color", "Scale Alpha"]);
            let sc = m.input("Scale Color").and_then(|i| match i {
                NInput::Value(v) => v.floats(),
                _ => None,
            });
            let sa = lit_f(m.input("Scale Alpha")).unwrap_or(1.0);
            let s = sc.map(|v| [v[0], v[1], v[2], sa]).unwrap_or([1.0, 1.0, 1.0, sa]);
            if s != [1.0; 4] {
                let c = x.get("Color").cloned().unwrap_or(Fx::Lit(Lit::Color([1.0; 4])));
                x.set("Color", mul_color(c, s));
            }
            o.mods.push(x);
        }
        // UE4-era location and velocity modules (UE5 folded them into
        // ShapeLocation and AddVelocity's modes); Fab packs still ship them.
        "SphereLocation" => {
            let mut x = FxModule::new("Shape Location");
            used.extend(["Sphere Orientation Axis", "Random Seed", "Spawn Group Mask", "Surface Only Band Thickness", "U Distribution", "V Distribution"]);
            let r = lit_f(m.input("Sphere Radius")).unwrap_or(100.0) * CM;
            used.push("Sphere Radius");
            let s = match m.input("Non Uniform Scale") {
                Some(NInput::Value(v)) if m.switch("UseNonUniformScale") != Some("false") => {
                    v.floats().filter(|f| f.len() == 3).map(|f| [f[0], f[1], f[2]]).unwrap_or([1.0; 3])
                }
                _ => [1.0; 3],
            };
            used.push("Non Uniform Scale");
            if m.switch("Surface Only") == Some("true") {
                x.set("Surface Only", Fx::Lit(Lit::Bool(true)));
            }
            if s[0] == s[1] && s[1] == s[2] {
                x.set("Shape Primitive", Fx::Choice("Sphere".into()));
                x.set("Sphere Radius", Fx::Lit(Lit::Float(r * s[0])));
            } else if s[0] == s[1] {
                // Squashed or stretched along Unreal Z (our Y, the cylinder
                // axis): a cylinder of the same radius and height.
                x.set("Shape Primitive", Fx::Choice("Cylinder".into()));
                x.set("Cylinder Radius", Fx::Lit(Lit::Float(r * s[0])));
                x.set("Cylinder Height", Fx::Lit(Lit::Float(2.0 * r * s[2])));
                o.changes.push(format!(
                    "Non Uniform Scale {s:?}: the ellipsoid is drawn as a cylinder of radius {} m and height {} m",
                    r * s[0],
                    2.0 * r * s[2]
                ));
            } else {
                let mean = (s[0] + s[1] + s[2]) / 3.0;
                x.set("Shape Primitive", Fx::Choice("Sphere".into()));
                x.set("Sphere Radius", Fx::Lit(Lit::Float(r * mean)));
                o.changes.push(format!("Non Uniform Scale {s:?}: drawn as a sphere of the mean radius"));
            }
            put!(x, "Offset", "Offset", Unit::Len);
            o.mods.push(x);
        }
        "AddVelocityFromPoint" => {
            let mut x = FxModule::new("Add Velocity from Point");
            put!(x, "Velocity Strength", "Velocity Strength", Unit::Len);
            put!(x, "Origin Offset", "Velocity Origin", Unit::Len);
            o.mods.push(x);
        }
        "ScaleVelocity" => {
            let mut x = FxModule::new("Scale Velocity");
            put!(x, "Velocity Scale", "Velocity Scale", Unit::Dir);
            o.mods.push(x);
        }
        "ScaleColor" => {
            let mut x = FxModule::new("Scale Color");
            // Scale Mode as saved: 0 RGB and alpha separately (Scale RGBA is
            // then a stale constant, as in NS_Fire), 1 RGBA.
            let rgba = mode(m, "Scale Mode") == Some(1) || (m.has("Scale RGBA") && !m.has("Scale RGB") && !m.has("Scale Alpha"));
            if rgba {
                used.extend(["Scale RGB", "Scale Alpha"]);
                put!(x, "Scale RGBA", "Scale RGBA", Unit::None);
            } else {
                used.push("Scale RGBA");
                put!(x, "Scale RGB", "Scale RGB", Unit::None);
                put!(x, "Scale Alpha", "Scale Alpha", Unit::None);
            }
            o.mods.push(x);
        }
        "ScaleSpriteSize" => {
            let mut x = FxModule::new("Scale Sprite Size");
            used.extend(["Initial Sprite Size", "Uniform Curve Index", "Curve Index", "Uniform Curve Scale", "Curve Scale"]);
            let uscale = lit_f(m.input("Uniform Curve Scale")).unwrap_or(1.0);
            if let Some(i) = m.input("Uniform Curve Sprite Scale") {
                used.push("Uniform Curve Sprite Scale");
                match i {
                    NInput::Value(NValue::Curve(c)) => {
                        let mut cv = curve(c, Ty::Float, Unit::None);
                        cv.ty = Ty::Vec2;
                        for (_, v) in cv.keys.iter_mut() {
                            let s = v[0] * uscale;
                            *v = vec![s, s];
                        }
                        x.set("Scale Factor", Fx::CurveOverAge(cv));
                    }
                    other => match cx.conv(e, other, Unit::None, Ty::Float) {
                        Ok(v) => x.set("Scale Factor", coerce(scale_fx(v, uscale), Ty::Vec2)?),
                        Err(w) => o.drop_input("Uniform Curve Sprite Scale", &w),
                    },
                }
            } else if let Some(i) = m.input("Sprite Size Scale Curve").or(m.input("Scale Curve")) {
                used.extend(["Sprite Size Scale Curve", "Scale Curve"]);
                match i {
                    NInput::Value(NValue::Curve(c)) => x.set("Scale Factor", Fx::CurveOverAge(curve(c, Ty::Vec2, Unit::None))),
                    other => match cx.conv(e, other, Unit::None, Ty::Vec2) {
                        Ok(v) => x.set("Scale Factor", v),
                        Err(w) => o.drop_input("Scale Curve", &w),
                    },
                }
            } else if m.has("Uniform Scale Factor") || m.has("Uniform Sprite Scale") {
                let k = if m.has("Uniform Scale Factor") { "Uniform Scale Factor" } else { "Uniform Sprite Scale" };
                used.push(k);
                if let Some(v) = cx.input(e, m, k, Unit::None, Ty::Float, o) {
                    x.set("Scale Factor", coerce(v, Ty::Vec2)?);
                }
            } else {
                put!(x, "Scale Factor", "Scale Factor", Unit::None);
                put!(x, "Sprite Scale", "Scale Factor", Unit::None);
            }
            o.mods.push(x);
        }
        "ScaleMeshSize" => {
            let mut x = FxModule::new("Scale Mesh Size");
            used.extend(["Initial Mesh Size", "Curve Index", "Scale Curve"]);
            if let Some(i) = m.input("Uniform Curve Mesh Scale").or(m.input("Uniform Scale Factor")) {
                used.extend(["Uniform Curve Mesh Scale", "Uniform Scale Factor"]);
                match i {
                    NInput::Value(NValue::Curve(c)) => {
                        let mut cv = curve(c, Ty::Float, Unit::None);
                        cv.ty = Ty::Vec3;
                        for (_, v) in cv.keys.iter_mut() {
                            *v = vec![v[0]; 3];
                        }
                        x.set("Scale Factor", Fx::CurveOverAge(cv));
                    }
                    other => match cx.conv(e, other, Unit::None, Ty::Float) {
                        Ok(v) => x.set("Scale Factor", coerce(v, Ty::Vec3)?),
                        Err(w) => o.drop_input("Uniform Scale Factor", &w),
                    },
                }
            } else {
                put!(x, "Scale Factor", "Scale Factor", Unit::Dir);
            }
            o.mods.push(x);
        }
        "DynamicMaterialParameters" => {
            // Parameter index N goes to our module of index N (Particles.
            // DynamicMaterialParameter, ...1, ...2, ...3); indices 1..3 only
            // when their Write Parameter Index switch is on (else the inputs
            // are stale). Index 0 is written unless its switch is off.
            let targets = [
                ("Dynamic Material Parameters", "Dynamic Material Parameter"),
                ("Dynamic Material Parameters 2", "Dynamic Material Parameter 1"),
                ("Dynamic Material Parameters 3", "Dynamic Material Parameter 2"),
                ("Dynamic Material Parameters 4", "Dynamic Material Parameter 3"),
            ];
            for (k, _) in &m.inputs {
                if k.starts_with("Index ") {
                    used.push(k.as_str());
                }
            }
            for (idx, (module, input)) in targets.iter().enumerate() {
                let on = match m.switch(&format!("Write Parameter Index {idx}")) {
                    Some(s) => s == "true",
                    None => idx == 0,
                };
                if !on {
                    continue;
                }
                let mut comps = Vec::new();
                for c in 1..=4 {
                    comps.push(cx.input(e, m, &format!("Index {idx} Param {c}"), Unit::None, Ty::Float, o).unwrap_or(Fx::Lit(Lit::Float(0.0))));
                }
                let v = if comps.iter().all(|c| matches!(c, Fx::Lit(_))) {
                    let f: Vec<f32> = comps.iter().map(|c| if let Fx::Lit(l) = c { l.floats()[0] } else { 0.0 }).collect();
                    Fx::Lit(Lit::Vec4([f[0], f[1], f[2], f[3]]))
                } else {
                    Fx::Dyn(
                        "Make Vec4 from Floats".into(),
                        ["X", "Y", "Z", "W"].iter().zip(comps).map(|(k, v)| (k.to_string(), v)).collect(),
                    )
                };
                let mut x = FxModule::new(module);
                x.set(input, v);
                o.mods.push(x);
            }
        }
        "CameraOffset" => {
            let mut x = FxModule::new("Camera Offset");
            put!(x, "Camera Offset Amount", "Camera Offset Amount", Unit::Len);
            o.mods.push(x);
        }
        "UpdateMeshOrientation" => {
            let mut x = FxModule::new("Update Mesh Orientation");
            put!(x, "Rotation Rate", "Rotation Rate", Unit::Dir);
            used.push("Rotation Vector");
            o.mods.push(x);
        }
        "AlignSpriteToMeshOrientation" => {
            let mut x = FxModule::new("Align Sprite to Mesh Orientation");
            put!(x, "Mesh Orientation Relative Sprite Facing Vector", "Facing Axis", Unit::Dir);
            put!(x, "Mesh Orientation Relative Sprite Alignment Vector", "Alignment Axis", Unit::Dir);
            // Orientation Quaternion defaults to Particles.MeshOrientation,
            // which the module reads; a fixed identity is the same as unset.
            used.push("Orientation Quaternion");
            match m.input("Orientation Quaternion") {
                Some(NInput::Value(NValue::Quat(q))) if q[..3].iter().any(|c| c.abs() > 1e-6) => {
                    o.drop_input("Orientation Quaternion", "a fixed rotation is not carried over (the particle's mesh orientation is used)");
                }
                _ => {}
            }
            o.mods.push(x);
        }
        "InitialMeshOrientation" => {
            let mut x = FxModule::new("Initial Mesh Orientation");
            put!(x, "Rotation", "Rotation", Unit::Dir);
            // Niagara's Rotation is in normalised turns (0..1).
            if let Some(v) = x.get("Rotation").cloned() {
                x.set("Rotation", scale_fx(v, 360.0));
            }
            o.mods.push(x);
        }
        "SubUVAnimation" => {
            let mut x = FxModule::new("SubUV Animation");
            for (k, _) in &m.inputs {
                used.push(k.as_str());
            }
            o.changes.push("flipbook settings not kept (placeholder sprite)".into());
            let _ = &mut x;
            o.mods.push(x);
        }
        "SystemState" => return Ok(Some("system loop settings".into())),
        // Spawn-time forces: Add Velocity here writes Velocity directly and
        // the forces modules act in Particle Update, so there is nothing left
        // to apply at spawn.
        "ApplyInitialForces" => return Ok(Some("spawn velocity is written directly by the velocity modules".into())),
        "ScaleRibbonWidth" => {
            let mut x = FxModule::new("Scale Ribbon Width");
            put!(x, "Ribbon Width Scale", "Ribbon Width Scale", Unit::None);
            o.mods.push(x);
        }
        _ => return Err(format!("{script} has no equivalent module here")),
    }
    let _ = st;
    for (k, v) in &m.inputs {
        if used.contains(&k.as_str()) || IGNORED.contains(&k.as_str()) || matches!(v, NInput::Value(NValue::Matrix(_))) {
            continue;
        }
        o.changes.push(format!("{k}: not carried over"));
    }
    Ok(None)
}

/// A static switch's enumerator index (`NewEnumerator3` -> 3).
fn mode(m: &NModule, name: &str) -> Option<u32> {
    m.switch(name)?.strip_prefix("NewEnumerator")?.parse().ok()
}

/// Random range of two converted values.
fn range_of(a: Fx, b: Fx, ty: Ty) -> Fx {
    match (a, b) {
        (Fx::Lit(a), Fx::Lit(b)) => Fx::Range(a, b),
        (a, b) => Fx::Dyn(format!("Uniform Ranged {}", ty.name()), vec![("Minimum".into(), a), ("Maximum".into(), b)]),
    }
}

/// One random draw scaling a uniform size: `Random Range VecN by Uniform`.
fn uniform_random(a: Fx, b: Fx, ty: Ty) -> Fx {
    match (a, b) {
        (Fx::Lit(Lit::Float(a)), Fx::Lit(Lit::Float(b))) => Fx::Dyn(
            format!("Random Range {} by Uniform", ty.name()),
            vec![("Minimum".into(), Fx::Lit(Lit::from_floats(ty, &[a]))), ("Maximum".into(), Fx::Lit(Lit::from_floats(ty, &[b])))],
        ),
        (a, b) => {
            let r = Fx::Dyn("Uniform Ranged Float".into(), vec![("Minimum".into(), a), ("Maximum".into(), b)]);
            Fx::Dyn(format!("Multiply {} by Float", ty.name()), vec![("A".into(), Fx::Lit(Lit::from_floats(ty, &[1.0]))), ("B".into(), r)])
        }
    }
}

/// Initialize Particle (V2). Its modes are static switches; the enumerator
/// indices below were read off the test files (the inputs each mode keeps
/// in the compiled stack): Lifetime Mode 0 Direct, 1 Random; Color Mode 0
/// Unset, 1 Direct, 2 Random Range, 3 Direct with HSV randomisation;
/// Sprite Size / Mesh Scale Mode 0 Unset, 1 Random Uniform, 2 Random
/// Non-Uniform, 3 Uniform, 4 Non-Uniform; Sprite Rotation Mode 2 Random
/// Range, 3 Unset; Position Mode 2 Simulation Position.
fn init_particle<'m>(cx: &mut Cx, e: &NEmitter, m: &'m NModule, o: &mut Out, used: &mut Vec<&'m str>) {
    let mut x = FxModule::new("Initialize Particle");
    // Every input is decided by a mode: none is reported as left over.
    for (k, _) in &m.inputs {
        used.push(k.as_str());
    }
    let pair = |cx: &mut Cx, o: &mut Out, lo: &str, hi: &str, unit: Unit, ty: Ty| -> Option<(Fx, Fx)> {
        Some((cx.input(e, m, lo, unit, ty, o)?, cx.input(e, m, hi, unit, ty, o)?))
    };
    // Lifetime.
    let lifetime = match mode(m, "Lifetime Mode") {
        Some(1) => pair(cx, o, "Lifetime Min", "Lifetime Max", Unit::None, Ty::Float).map(|(a, b)| range_of(a, b, Ty::Float)),
        Some(_) => cx.input(e, m, "Lifetime", Unit::None, Ty::Float, o),
        None if m.has("Lifetime Min") => pair(cx, o, "Lifetime Min", "Lifetime Max", Unit::None, Ty::Float).map(|(a, b)| range_of(a, b, Ty::Float)),
        None => cx.input(e, m, "Lifetime", Unit::None, Ty::Float, o),
    };
    if let Some(v) = lifetime {
        x.set("Lifetime", v);
    }
    // Colour.
    let color = match mode(m, "Color Mode") {
        Some(0) => None,
        Some(2) => pair(cx, o, "Color Minimum", "Color Maximum", Unit::None, Ty::Color).map(|(a, b)| range_of(a, b, Ty::Color)),
        Some(3) => {
            // The compiled module turns the hue by random(Hue Shift Range) *
            // 360° and scales saturation, value and alpha by random draws of
            // their ranges, each when its Adjust switch is on (Hue's default on).
            for (switch, range, input, scale, default_on) in [
                ("AdjustHue", "Hue Shift Range", "Hue Shift", 360.0, true),
                ("AdjustSaturation", "Saturation Range", "Saturation Scale", 1.0, false),
                ("AdjustValue", "Value Range", "Value Scale", 1.0, false),
                ("AdjustAlpha", "Alpha Scale Range", "Alpha Scale", 1.0, false),
            ] {
                let on = m.switch(switch).map_or(default_on, |s| s == "true");
                if !on || !m.has(range) {
                    continue;
                }
                match m.input(range) {
                    Some(NInput::Value(NValue::Vec2([lo, hi]))) => {
                        x.set(input, Fx::Range(Lit::Float(lo * scale), Lit::Float(hi * scale)));
                    }
                    _ => match cx.input(e, m, range, Unit::None, Ty::Vec2, o) {
                        Some(v) => {
                            let part = |c: &str| Fx::Dyn("Break Vec2".into(), vec![("Value".into(), v.clone()), ("Component".into(), Fx::Choice(c.into()))]);
                            x.set(input, scale_fx(range_of(part("X"), part("Y"), Ty::Float), scale));
                        }
                        None => o.changes.push(format!("Color: {range} not kept")),
                    },
                }
            }
            cx.input(e, m, "Color", Unit::None, Ty::Color, o)
        }
        _ => cx.input(e, m, "Color", Unit::None, Ty::Color, o),
    };
    if let Some(v) = color {
        x.set("Color", v);
    }
    // Mass.
    let mass = if m.has("Mass Min") && m.has("Mass Max") {
        pair(cx, o, "Mass Min", "Mass Max", Unit::None, Ty::Float).map(|(a, b)| range_of(a, b, Ty::Float))
    } else {
        cx.input(e, m, "Mass", Unit::None, Ty::Float, o)
    };
    if let Some(v) = mass {
        x.set("Mass", v);
    }
    // Position: only Direct Set writes it (Simulation Position, mode 2, is
    // the effect's origin here).
    if mode(m, "Position Mode").is_some_and(|p| p != 2) {
        if let Some(v) = cx.input(e, m, "Position", Unit::Len, Ty::Vec3, o) {
            x.set("Position", v);
        }
    } else if m.switch("UsePositionOffset") == Some("true") {
        // Simulation Position plus Position Offset (e.g. a fire's light
        // 40 cm above the floor rather than on it).
        if let Some(v) = cx.input(e, m, "Position Offset", Unit::Len, Ty::Vec3, o) {
            x.set("Position", v);
        }
    }
    // Sprite size.
    let size = |cx: &mut Cx, o: &mut Out, md: Option<u32>, prefix: &str, ty: Ty, unit: Unit| -> Option<Fx> {
        let uni = format!("{prefix}Uniform {}", if ty == Ty::Vec2 { "Sprite Size" } else { "Scale" });
        let non = if ty == Ty::Vec2 { "Sprite Size".to_string() } else { format!("{prefix}Scale") };
        let (uni, non) = if ty == Ty::Vec2 { (uni, non) } else { (format!("Mesh Uniform Scale"), "Mesh Scale".to_string()) };
        match md {
            Some(0) => None,
            Some(1) => pair(cx, o, &format!("{uni} Min"), &format!("{uni} Max"), unit, Ty::Float).map(|(a, b)| uniform_random(a, b, ty)),
            Some(2) => pair(cx, o, &format!("{non} Min"), &format!("{non} Max"), if ty == Ty::Vec3 { Unit::Dir } else { unit }, ty).map(|(a, b)| range_of(a, b, ty)),
            Some(3) => cx.input(e, m, &uni, unit, Ty::Float, o).and_then(|v| coerce(v, ty).ok()),
            Some(_) => cx.input(e, m, &non, if ty == Ty::Vec3 { Unit::Dir } else { unit }, ty, o),
            None => None,
        }
    };
    if let Some(v) = size(cx, o, mode(m, "Sprite Size Mode"), "", Ty::Vec2, Unit::Len) {
        x.set("Sprite Size", v);
    }
    // Sprite rotation (degrees).
    match mode(m, "Sprite Rotation Mode") {
        Some(2) => {
            if let Some((a, b)) = pair(cx, o, "Sprite Rotation Angle Min", "Sprite Rotation Angle Max", Unit::None, Ty::Float) {
                x.set("Sprite Rotation", range_of(a, b, Ty::Float));
            }
        }
        Some(0) => x.set("Sprite Rotation", Fx::Range(Lit::Float(0.0), Lit::Float(360.0))),
        Some(1) => {
            if let Some(v) = cx.input(e, m, "Sprite Rotation Angle", Unit::None, Ty::Float, o) {
                x.set("Sprite Rotation", v);
            }
        }
        _ => {}
    }
    o.mods.push(x);

    // Mesh scale -> Initialize Mesh, when there is a mesh renderer.
    if e.renderers.iter().any(|r| r.class.contains("Mesh")) {
        // Real meshes are in metres already; a placeholder shape (1 m) is
        // drawn at PLACEHOLDER_MESH_SIZE per unit of Niagara's mesh scale.
        let mut im = FxModule::new("Initialize Mesh");
        let v = size(cx, o, mode(m, "Mesh Scale Mode"), "Mesh ", Ty::Vec3, Unit::None).unwrap_or(Fx::Lit(Lit::Vec3([1.0; 3])));
        let unit = cx.mesh_unit(e);
        im.set("Mesh Scale", if unit == 1.0 { v } else { scale_fx(v, unit) });
        o.mods.push(im);
    }
    // Ribbon width -> Initialize Ribbon, when there is a ribbon renderer.
    if e.renderers.iter().any(|r| r.class.contains("Ribbon")) {
        let mut ir = FxModule::new("Initialize Ribbon");
        if let Some(v) = cx.input(e, m, "Ribbon Width", Unit::Len, Ty::Float, o) {
            ir.set("Ribbon Width", v);
        }
        o.mods.push(ir);
    }
}

// ---- values ----

fn nvalue_ty(v: &NValue) -> Ty {
    match v {
        NValue::Float(_) => Ty::Float,
        NValue::Int(_) | NValue::Enum(_) => Ty::Int,
        NValue::Bool(_) => Ty::Bool,
        NValue::Vec2(_) => Ty::Vec2,
        NValue::Vec3(_) | NValue::Position(_) => Ty::Vec3,
        NValue::Vec4(_) => Ty::Vec4,
        NValue::Color(_) => Ty::Color,
        NValue::Quat(_) => Ty::Quat,
        NValue::Matrix(_) | NValue::DataInterface(_) => Ty::Vec4,
        NValue::Curve(c) => curve_ty(c),
    }
}

fn curve_ty(c: &NCurve) -> Ty {
    match c.channels.len() {
        2 => Ty::Vec2,
        3 => Ty::Vec3,
        4 if c.class.contains("Color") => Ty::Color,
        4 => Ty::Vec4,
        _ => Ty::Float,
    }
}

/// Applies a unit to components of a value of type `ty`.
fn apply_unit(ty: Ty, v: &mut [f32], unit: Unit) {
    match (unit, ty) {
        (Unit::None, _) => {}
        (Unit::Len, Ty::Float | Ty::Vec2) => v.iter_mut().for_each(|x| *x *= CM),
        (Unit::Len, Ty::Vec3) => {
            v.swap(1, 2);
            v.iter_mut().for_each(|x| *x *= CM);
        }
        (Unit::Dir, Ty::Vec3) => v.swap(1, 2),
        _ => {}
    }
}

/// A Niagara value as a literal (or a curve over age), units converted.
fn value_fx(v: &NValue, unit: Unit) -> Result<Fx, String> {
    Ok(match v {
        NValue::Curve(c) => Fx::Curve(curve(c, curve_ty(c), unit)),
        NValue::Enum(e) => return Err(format!("an enum value {e}")),
        NValue::Matrix(_) => return Err("a matrix".into()),
        NValue::DataInterface(c) => return Err(format!("a {c} data interface")),
        NValue::Bool(b) => Fx::Lit(Lit::Bool(*b)),
        NValue::Int(i) => Fx::Lit(Lit::Int(*i)),
        other => {
            let ty = nvalue_ty(other);
            let mut f = other.floats().unwrap_or_default();
            apply_unit(ty, &mut f, unit);
            Fx::Lit(Lit::from_floats(ty, &f))
        }
    })
}

fn fx_ty(v: &Fx) -> Option<Ty> {
    match v {
        Fx::Lit(l) => Some(l.ty()),
        Fx::Range(a, _) => Some(a.ty()),
        Fx::CurveOverAge(c) => Some(c.ty),
        Fx::Curve(_) | Fx::List(_) => None,
        Fx::Choice(_) => Some(Ty::Int),
        Fx::Link(t, _) => Ty::from_name(t),
        Fx::Dyn(n, _) => fx_core::dynamic::DYNAMIC_INPUTS
            .iter()
            .find(|d| &d.name == n)
            .and_then(|d| Ty::from_name(d.ty.name())),
    }
}

/// Converts a value to type `want` where that is a plain widening.
fn coerce(v: Fx, want: Ty) -> Result<Fx, String> {
    // A bare curve as a value: sampled over the particle's age.
    let v = match v {
        Fx::Curve(c) => Fx::CurveOverAge(c),
        other => other,
    };
    let Some(have) = fx_ty(&v) else {
        return Err("a value of unknown type".into());
    };
    if have == want {
        return Ok(v);
    }
    match (v, have, want) {
        (Fx::Lit(l), _, _) if l.ty() != Ty::Quat && want != Ty::Quat => {
            let f = l.floats();
            if f.len() == 1 || f.len() >= want.components() {
                Ok(Fx::Lit(Lit::from_floats(want, &f)))
            } else {
                Err(format!("a {} where a {} is needed", have.name(), want.name()))
            }
        }
        (Fx::Range(a, b), _, _) if a.floats().len() == 1 => Ok(Fx::Dyn(
            format!("Random Range {} by Uniform", want.name()),
            vec![
                ("Minimum".into(), Fx::Lit(Lit::from_floats(want, &a.floats()))),
                ("Maximum".into(), Fx::Lit(Lit::from_floats(want, &b.floats()))),
            ],
        )),
        (Fx::CurveOverAge(mut c), Ty::Float, _) => {
            c.ty = want;
            for (_, k) in c.keys.iter_mut() {
                let x = k[0];
                *k = Lit::from_floats(want, &[x]).floats();
            }
            Ok(Fx::CurveOverAge(c))
        }
        (v, Ty::Float, Ty::Vec2 | Ty::Vec3 | Ty::Vec4 | Ty::Color) => {
            let names: &[&str] = match want {
                Ty::Vec2 => &["X", "Y"],
                Ty::Vec3 => &["X", "Y", "Z"],
                Ty::Vec4 => &["X", "Y", "Z", "W"],
                _ => &["R", "G", "B", "A"],
            };
            if want != Ty::Color {
                // One draw scales every component (random draws stay shared).
                return Ok(Fx::Dyn(
                    format!("Multiply {} by Float", want.name()),
                    vec![("A".into(), Fx::Lit(Lit::from_floats(want, &[1.0]))), ("B".into(), v)],
                ));
            }
            Ok(Fx::Dyn(
                format!("Make {} from Floats", want.name()),
                names.iter().map(|n| (n.to_string(), v.clone())).collect(),
            ))
        }
        (v, Ty::Vec3, Ty::Color) => Ok(Fx::Dyn(
            "Make Color from Vec3 and Float".into(),
            vec![("RGB".into(), v), ("Alpha".into(), Fx::Lit(Lit::Float(1.0)))],
        )),
        (_, h, w) => Err(format!("a {} where a {} is needed", h.name(), w.name())),
    }
}

/// v * s for a Float or vector value.
fn scale_fx(v: Fx, s: f32) -> Fx {
    if s == 1.0 {
        return v;
    }
    match v {
        Fx::Lit(l) => {
            let f: Vec<f32> = l.floats().iter().map(|x| x * s).collect();
            Fx::Lit(Lit::from_floats(l.ty(), &f))
        }
        Fx::Range(a, b) => {
            let fa: Vec<f32> = a.floats().iter().map(|x| x * s).collect();
            let fb: Vec<f32> = b.floats().iter().map(|x| x * s).collect();
            Fx::Range(Lit::from_floats(a.ty(), &fa), Lit::from_floats(b.ty(), &fb))
        }
        Fx::CurveOverAge(mut c) => {
            for (_, k) in c.keys.iter_mut() {
                k.iter_mut().for_each(|x| *x *= s);
            }
            Fx::CurveOverAge(c)
        }
        Fx::Dyn(n, args) if (n.starts_with("Random Range") || n.starts_with("Uniform Ranged")) && args.iter().all(|(_, a)| matches!(a, Fx::Lit(_))) => {
            Fx::Dyn(n, args.into_iter().map(|(k, a)| (k, scale_fx(a, s))).collect())
        }
        Fx::Dyn(n, mut args) if n.starts_with("Multiply") && n.ends_with("by Float") && matches!(args.first(), Some((_, Fx::Lit(_)))) => {
            let a = args[0].1.clone();
            args[0].1 = scale_fx(a, s);
            Fx::Dyn(n, args)
        }
        other => match fx_ty(&other) {
            Some(Ty::Float) => Fx::Dyn("Multiply Float".into(), vec![("A".into(), other), ("B".into(), Fx::Lit(Lit::Float(s)))]),
            Some(t @ (Ty::Vec2 | Ty::Vec3 | Ty::Vec4 | Ty::Color)) => Fx::Dyn(
                format!("Multiply {} by Float", t.name()),
                vec![("A".into(), other), ("B".into(), Fx::Lit(Lit::Float(s)))],
            ),
            _ => other,
        },
    }
}

fn add_vec3(a: Fx, b: Fx) -> Fx {
    match (&a, &b) {
        (Fx::Lit(x), Fx::Lit(y)) => {
            let (x, y) = (x.floats(), y.floats());
            Fx::Lit(Lit::Vec3([x[0] + y[0], x[1] + y[1], x[2] + y[2]]))
        }
        (_, Fx::Lit(Lit::Vec3([0.0, 0.0, 0.0]))) => a,
        _ => Fx::Dyn("Add Vec3".into(), vec![("A".into(), a), ("B".into(), b)]),
    }
}

fn mul_color(c: Fx, s: [f32; 4]) -> Fx {
    match c {
        Fx::Lit(l) => {
            let f = l.floats();
            Fx::Lit(Lit::Color([f[0] * s[0], f[1] * s[1], f[2] * s[2], f[3] * s[3]]))
        }
        Fx::CurveOverAge(mut cv) => {
            for (_, k) in cv.keys.iter_mut() {
                for (i, x) in k.iter_mut().enumerate() {
                    *x *= s[i.min(3)];
                }
            }
            Fx::CurveOverAge(cv)
        }
        other => Fx::Dyn("Multiply Color".into(), vec![("A".into(), other), ("B".into(), Fx::Lit(Lit::Color(s)))]),
    }
}

// ---- curves ----

/// Value of a rich curve at `t`.
pub fn eval_curve(keys: &[CurveKey], t: f32) -> f32 {
    let Some(first) = keys.first() else { return 0.0 };
    if t <= first.time {
        return first.value;
    }
    let last = keys.last().unwrap();
    if t >= last.time {
        return last.value;
    }
    for w in keys.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if t >= a.time && t <= b.time {
            let dt = b.time - a.time;
            if dt <= 0.0 {
                return b.value;
            }
            let u = (t - a.time) / dt;
            return match a.interp {
                1 => a.value,
                2 => {
                    // Hermite with the keys' tangents (per unit time).
                    let (u2, u3) = (u * u, u * u * u);
                    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
                    let h10 = u3 - 2.0 * u2 + u;
                    let h01 = -2.0 * u3 + 3.0 * u2;
                    let h11 = u3 - u2;
                    h00 * a.value + h10 * dt * a.leave + h01 * b.value + h11 * dt * b.arrive
                }
                _ => a.value + (b.value - a.value) * u,
            };
        }
    }
    last.value
}

/// A Niagara curve as linear keys of type `ty` (cubic segments sampled,
/// then points a straight line already gives dropped), units converted.
pub fn curve(c: &NCurve, ty: Ty, unit: Unit) -> Curve {
    let mut times: Vec<f32> = Vec::new();
    for ch in &c.channels {
        for w in ch.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            times.push(a.time);
            match a.interp {
                2 => {
                    for k in 1..12 {
                        times.push(a.time + (b.time - a.time) * k as f32 / 12.0);
                    }
                }
                1 => times.push(b.time - (b.time - a.time) * 1e-3),
                _ => {}
            }
        }
        if let Some(l) = ch.last() {
            times.push(l.time);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if times.is_empty() {
        times.push(0.0);
    }
    let n = ty.components();
    let mut keys: Vec<(f32, Vec<f32>)> = times
        .iter()
        .map(|&t| {
            let mut v: Vec<f32> = (0..n).map(|k| c.channels.get(k).map(|ch| eval_curve(ch, t)).unwrap_or(1.0)).collect();
            if n == 1 && c.channels.len() > 1 {
                v = vec![eval_curve(&c.channels[0], t)];
            }
            apply_unit(ty, &mut v, unit);
            (t, v)
        })
        .collect();
    // Drop points the line through their neighbours already gives.
    let range = keys
        .iter()
        .flat_map(|(_, v)| v.iter().copied())
        .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    let tol = ((range.1 - range.0).abs() * 0.01).max(1e-5);
    let mut i = 1;
    while i + 1 < keys.len() {
        let (t0, v0) = &keys[i - 1];
        let (t1, v1) = &keys[i];
        let (t2, v2) = &keys[i + 1];
        let u = if t2 > t0 { (t1 - t0) / (t2 - t0) } else { 0.0 };
        let fits = (0..n).all(|k| (v0[k] + (v2[k] - v0[k]) * u - v1[k]).abs() <= tol);
        if fits {
            keys.remove(i);
        } else {
            i += 1;
        }
    }
    Curve { ty, keys }
}

// ---- placeholders ----

/// Blend mode guessed from a material's name and the emitter's (smoke,
/// dust and the like are alpha blended whatever the material is called).
pub fn guess_blend(material: &str, emitter: &str) -> &'static str {
    let em = emitter.to_lowercase();
    if ["smoke", "dust", "cloud", "mist", "steam", "fog", "liquid", "water", "splash"].iter().any(|k| em.contains(k)) {
        return "Alpha";
    }
    let m = material.rsplit(['/', '.']).next().unwrap_or(material).to_lowercase();
    let additive = ["add", "fire", "flame", "glow", "spark", "ember", "glimmer", "flare", "shine", "light", "emissive", "magic"];
    if additive.iter().any(|k| m.contains(k)) && !m.contains("smoke") {
        "Additive"
    } else {
        "Alpha"
    }
}

/// A built-in mesh guessed from a mesh's name.
pub fn guess_mesh(mesh: &str) -> &'static str {
    let m = mesh.rsplit(['/', '.']).next().unwrap_or(mesh).to_lowercase();
    if ["sphere", "rock", "pebble", "ball", "stone", "drop"].iter().any(|k| m.contains(k)) {
        "sphere"
    } else if m.contains("cylinder") || m.contains("pipe") {
        "cylinder"
    } else if ["fish", "boid", "cone", "bird", "arrow"].iter().any(|k| m.contains(k)) {
        "cone"
    } else if m.contains("torus") || m.contains("ring") {
        "torus"
    } else if m.contains("capsule") {
        "capsule"
    } else {
        "cube"
    }
}
