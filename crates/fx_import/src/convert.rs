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

/// The converted effect.
pub struct Conversion {
    pub ron: String,
    pub report: Report,
}

/// Converts a system. `source` names the file it came from (for the
/// header and report).
pub fn convert(sys: &NSystem, source: &str) -> Conversion {
    let mut cx = Cx {
        sys,
        user_units: HashMap::new(),
        report: Report {
            source: source.to_string(),
            system: sys.name.clone(),
            ..Report::default()
        },
        custom: Vec::new(),
    };
    for n in &sys.notes {
        cx.report.notes.push(n.clone());
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
    for st in &sys.stacks {
        for m in &st.modules {
            let status = if !m.enabled {
                Status::Disabled
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
    let mut report = cx.report;
    // Validate.
    let checked = fx_core::check_source(&format!("{}.fx.ron", sys.name), &ron);
    report.validation = checked.errors().map(|d| d.to_string()).collect();
    Conversion { ron, report }
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
        if !e.local_space {
            out.comment.push("Niagara simulates this emitter in world space; here particles live in the effect's space.".into());
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
        for r in &e.renderers {
            if let Some(s) = self.renderer(e, r) {
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
            return Err(format!("a module from another package ({}), not available here", m.script));
        }
        map_module(self, e, st, m, loops, o)
    }

    // ---- inputs ----

    /// The user parameter's value, by full name.
    fn user(&self, name: &str) -> Option<&'a NValue> {
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
                Ok(match (curve, index) {
                    (Fx::Curve(c), None) => Fx::CurveOverAge(c),
                    (c, None) => Fx::Dyn(format!("{} from Curve", ty.name()), vec![("Curve".into(), c)]),
                    (c, Some(i)) => Fx::Dyn(
                        format!("{} from Curve", ty.name()),
                        vec![("Curve".into(), c), ("Curve Index".into(), i)],
                    ),
                })
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

    fn renderer(&mut self, e: &NEmitter, r: &crate::niagara::NRenderer) -> Option<String> {
        use crate::property::find;
        let enum_tail = |n: &str| find(&r.properties, n).and_then(|v| v.as_str()).map(|s| s.rsplit("::").next().unwrap_or(s).to_string());
        let material = r.refs.get("Material").cloned().or_else(|| {
            r.refs.iter().find(|(k, _)| k.starts_with("OverrideMaterials") || k.contains("Material")).map(|(_, v)| v.clone())
        });
        let blend = guess_blend(material.as_deref().unwrap_or(""), &e.name);
        let mut fields = Vec::new();
        if !r.enabled {
            self.report.renderers.push(format!("{}: {} disabled: left out", e.name, r.class));
            return None;
        }
        if material.as_deref().is_some_and(|m| { let l = m.to_lowercase(); l.contains("distort") || l.contains("refract") }) {
            self.report.renderers.push(format!(
                "{}: {} with a distortion material (`{}`): no equivalent, left out",
                e.name,
                r.class,
                material.as_deref().unwrap_or("")
            ));
            return None;
        }
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
                match enum_tail("SortMode").as_deref() {
                    Some("ViewDepth") => fields.push("sort_mode: ViewDepth".into()),
                    Some("ViewDistance") => fields.push("sort_mode: ViewDistance".into()),
                    _ => {}
                }
                if let Some(v) = find(&r.properties, "SubImageSize") {
                    let x = v.field("X").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    let y = v.field("Y").and_then(|x| x.as_f64()).unwrap_or(1.0);
                    if x > 1.0 || y > 1.0 {
                        self.report.placeholders.push(format!(
                            "{}: sprite flipbook {x}x{y} of the material's texture: not kept (the placeholder sprite has one frame)",
                            e.name
                        ));
                    }
                }
                self.report.placeholders.push(format!(
                    "{}: sprite material `{}` -> built-in soft sprite, blend {blend} (guessed from the name)",
                    e.name,
                    material.as_deref().unwrap_or("none")
                ));
                format!("Sprite(({}))", fields.join(", "))
            }
            "NiagaraMeshRendererProperties" => {
                let mesh_path = r
                    .refs
                    .iter()
                    .filter(|(k, _)| k.starts_with("Meshes[") && k.ends_with("].Mesh"))
                    .map(|(_, v)| v.clone())
                    .min();
                let shape = mesh_path.as_deref().map(guess_mesh).unwrap_or("cube");
                fields.push(format!("mesh: {}", quote(shape)));
                if blend != "Alpha" || material.as_deref().is_some_and(|m| m.to_lowercase().contains("translucent")) {
                    fields.push(format!("blend_mode: {blend}"));
                }
                match enum_tail("FacingMode").as_deref() {
                    Some("Velocity") => fields.push("facing_mode: Velocity".into()),
                    Some("CameraPosition") => fields.push("facing_mode: CameraPosition".into()),
                    Some("CameraPlane") => fields.push("facing_mode: CameraPlane".into()),
                    _ => {}
                }
                self.report.placeholders.push(format!(
                    "{}: mesh `{}` -> built-in \"{shape}\" (guessed from the name), {} m per unit of mesh scale",
                    e.name,
                    mesh_path.as_deref().unwrap_or("none"),
                    PLACEHOLDER_MESH_SIZE
                ));
                format!("Mesh(({}))", fields.join(", "))
            }
            "NiagaraRibbonRendererProperties" => {
                fields.push(format!("blend_mode: {blend}"));
                self.report.placeholders.push(format!(
                    "{}: ribbon material `{}` -> built-in, blend {blend}",
                    e.name,
                    material.as_deref().unwrap_or("none")
                ));
                format!("Ribbon(({}))", fields.join(", "))
            }
            "NiagaraLightRendererProperties" => {
                let radius = find(&r.properties, "RadiusScale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                format!("Light((radius_scale: {}))", f(radius as f32))
            }
            other => {
                self.report.renderers.push(format!("{}: {other}: no equivalent renderer, left out", e.name));
                return None;
            }
        };
        self.report.renderers.push(format!("{}: {} -> `{}`", e.name, r.class, what));
        Some(what)
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
        let _ = writeln!(o, "// Placeholders: materials -> built-in sprite (blend guessed), meshes -> built-in shapes.");
        let _ = writeln!(o, "(");
        let _ = writeln!(o, "    name: {},", quote(&self.sys.name));
        if self.sys.warmup_time > 0.0 {
            let _ = writeln!(o, "    warmup_time: {},", f(self.sys.warmup_time));
        }
        let users: BTreeMap<&str, String> = self
            .sys
            .user_parameters
            .iter()
            .filter_map(|(n, v)| self.user_decl(n, v).map(|d| (n.as_str(), d)))
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
            match shape {
                "Sphere" => put!(x, "Sphere Radius", "Sphere Radius", Unit::Len),
                "Box" => {
                    if matches!(m.input("Box Midpoint"), Some(NInput::Value(NValue::Vec3([a, b, c]))) if [*a, *b, *c] == [0.5; 3]) {
                        used.push("Box Midpoint");
                    }
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
                    put!(x, "Cone Length", "Cone Length", Unit::Len);
                    put!(x, "Cone Angle", "Cone Angle", Unit::None);
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
                for n in ["Inner Cone Angle", "Distribution Along Cone Axis", "Speed Falloff From Cone Axis"] {
                    used.push(n);
                }
                if m.has("Inner Cone Angle") || m.has("Speed Falloff From Cone Axis") {
                    o.changes.push("cone falloffs (Inner Cone Angle, Distribution Along Cone Axis, Speed Falloff) not kept: uniform over the cone".into());
                }
                o.mods.push(x);
            } else if vmode == 1 {
                let mut x = FxModule::new("Add Velocity from Point");
                used.push("Velocity Speed");
                if let Some(v) = cx.input(e, m, "Velocity Speed", Unit::Len, Ty::Float, o) {
                    x.set("Velocity Strength", scale_fx(v, speed_scale));
                }
                put!(x, "Velocity Origin", "Velocity Origin", Unit::Len);
                o.mods.push(x);
            } else {
                let mut x = FxModule::new("Add Velocity");
                used.push("Velocity");
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
            o.mods.push(x);
        }
        "CurlNoiseForce" => {
            let mut x = FxModule::new("Curl Noise Force");
            put!(x, "Noise Strength", "Noise Strength", Unit::Len);
            used.push("Noise Frequency");
            if let Some(v) = lit_f(m.input("Noise Frequency")) {
                // Niagara's frequency is per centimetre... per 100 units.
                x.set("Noise Frequency", Fx::Lit(Lit::Float(v * 100.0)));
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
        "ScaleColor" => {
            let mut x = FxModule::new("Scale Color");
            put!(x, "Scale RGB", "Scale RGB", Unit::None);
            put!(x, "Scale Alpha", "Scale Alpha", Unit::None);
            if m.has("Scale RGBA") && !m.has("Scale RGB") {
                used.push("Scale RGBA");
                if let Some(NInput::Value(v)) = m.input("Scale RGBA") {
                    if let Some(f) = v.floats() {
                        x.set("Scale RGB", Fx::Lit(Lit::Vec3([f[0], f[1], f[2]])));
                        x.set("Scale Alpha", Fx::Lit(Lit::Float(f[3])));
                    }
                } else {
                    o.drop_input("Scale RGBA", "only a constant RGBA scale is mapped");
                }
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
            let mut x = FxModule::new("Dynamic Material Parameters");
            let names = ["Index 0 Param 1", "Index 0 Param 2", "Index 0 Param 3", "Index 0 Param 4"];
            let mut comps = Vec::new();
            for n in names {
                used.push(n);
                comps.push(cx.input(e, m, n, Unit::None, Ty::Float, o).unwrap_or(Fx::Lit(Lit::Float(0.0))));
            }
            for (k, _) in &m.inputs {
                if k.starts_with("Index 1") || k.starts_with("Index 2") || k.starts_with("Index 3") {
                    used.push(k.as_str());
                }
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
            x.set("Dynamic Material Parameter", v);
            o.mods.push(x);
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
            if m.has("Hue Shift Range") || m.has("Saturation Range") || m.has("Value Range") {
                o.changes.push("Color: random hue / saturation / value shifts not kept (base colour used)".into());
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
        // The meshes are not in the package: the placeholder shape (1 m) is
        // drawn at PLACEHOLDER_MESH_SIZE per unit of Niagara's mesh scale.
        let mut im = FxModule::new("Initialize Mesh");
        let v = size(cx, o, mode(m, "Mesh Scale Mode"), "Mesh ", Ty::Vec3, Unit::None).unwrap_or(Fx::Lit(Lit::Vec3([1.0; 3])));
        im.set("Mesh Scale", scale_fx(v, PLACEHOLDER_MESH_SIZE));
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
