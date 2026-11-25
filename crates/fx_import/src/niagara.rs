//! Interpretation of a Niagara system package: the system, its emitters,
//! their stacks (emitter spawn/update, particle spawn/update, event
//! handlers, simulation stages) and each module's function call with the
//! values of its inputs, as the Niagara editor stores them:
//!
//! - every stack is a chain of graph nodes from a `NiagaraNodeOutput`
//!   (its `ScriptType` says which stack) back through `ParameterMap` pins:
//!   `NiagaraNodeFunctionCall` (a module), `NiagaraNodeAssignment` (Set
//!   Variables), and `NiagaraNodeParameterMapSet` nodes holding overridden
//!   inputs as pins named `<Function>.<Input>`;
//! - an override pin is linked to a dynamic input (another function call),
//!   a `NiagaraNodeInput` (a data interface such as a curve, or a local
//!   value), a `NiagaraNodeParameterMapGet` output (a parameter link such as
//!   `User.SpawnRate`), or holds a default value string;
//! - inputs left as constants live in the scripts' rapid-iteration
//!   parameter stores as `Constants.<Emitter>.<Function>.<Input>`; only the
//!   inputs the compiled module still uses (after static switches) are
//!   there, which tells which mode a module runs in;
//! - static switches are pins on the function call node itself.
//!
//! Unreal units are kept here (centimetres, Z up); [`crate::convert`] turns
//! the result into an effect source file.

use std::collections::HashMap;

use crate::package::{Package, PackageIndex};
use crate::pins::{Direction, Pin};
use crate::property::{find, Property, Value};

/// A Niagara value.
#[derive(Clone, Debug, PartialEq)]
pub enum NValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    /// NiagaraPosition: a Vec3 that is a location.
    Position([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Quat([f32; 4]),
    Matrix(Vec<f32>),
    /// A user-defined enum value (`NewEnumerator3`) or its index.
    Enum(String),
    Curve(NCurve),
    /// A data interface other than a curve, by class name.
    DataInterface(String),
}

impl NValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            NValue::Float(_) => "Float",
            NValue::Int(_) => "Int",
            NValue::Bool(_) => "Bool",
            NValue::Vec2(_) => "Vec2",
            NValue::Vec3(_) => "Vec3",
            NValue::Position(_) => "Position",
            NValue::Vec4(_) => "Vec4",
            NValue::Color(_) => "Color",
            NValue::Quat(_) => "Quat",
            NValue::Matrix(_) => "Matrix",
            NValue::Enum(_) => "Enum",
            NValue::Curve(_) => "Curve",
            NValue::DataInterface(_) => "DataInterface",
        }
    }
    /// Components as floats (scalars give one).
    pub fn floats(&self) -> Option<Vec<f32>> {
        Some(match self {
            NValue::Float(v) => vec![*v],
            NValue::Int(v) => vec![*v as f32],
            NValue::Bool(v) => vec![if *v { 1.0 } else { 0.0 }],
            NValue::Vec2(v) => v.to_vec(),
            NValue::Vec3(v) | NValue::Position(v) => v.to_vec(),
            NValue::Vec4(v) | NValue::Color(v) | NValue::Quat(v) => v.to_vec(),
            NValue::Matrix(v) => v.clone(),
            _ => return None,
        })
    }
}

/// A rich curve key (FRichCurveKey).
#[derive(Clone, Debug, PartialEq)]
pub struct CurveKey {
    pub time: f32,
    pub value: f32,
    /// ERichCurveInterpMode: 0 linear, 1 constant, 2 cubic.
    pub interp: u8,
    pub arrive: f32,
    pub leave: f32,
}

/// A curve data interface: one key list per channel (1 for Float, 2 for
/// Vec2, 3 for Vec3, 4 for Vec4 and Color).
#[derive(Clone, Debug, PartialEq)]
pub struct NCurve {
    pub class: String,
    pub channels: Vec<Vec<CurveKey>>,
}

/// Where a module input's value comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum NInput {
    Value(NValue),
    /// A parameter, e.g. `User.SpawnRate`, `Particles.Velocity`.
    Link(String),
    Dynamic(Box<NModule>),
    /// Something this importer does not interpret (with a description).
    Unsupported(String),
}

/// How the module is written.
#[derive(Clone, Debug, PartialEq)]
pub enum ModuleKind {
    /// A module or dynamic input script (`/Niagara/Modules/...`, or a
    /// script in another package).
    Function,
    /// Set Variables (NiagaraNodeAssignment): assigns each target.
    Assignment { targets: Vec<(String, String)> },
    /// A scratch pad module in this package: its custom HLSL nodes and the
    /// scratch pad's own inputs.
    ScratchPad { hlsl: Vec<HlslNode> },
}

/// A custom HLSL node of a scratch pad: its code and where each pin comes
/// from or goes to (a parameter such as `Particles.Position`,
/// `Module.Strength`, `Engine.DeltaTime`; None when wired to something else).
#[derive(Clone, Debug, PartialEq)]
pub struct HlslNode {
    pub code: String,
    /// (pin name, Niagara type name, source parameter).
    pub inputs: Vec<(String, String, Option<String>)>,
    /// (pin name, Niagara type name, target parameter).
    pub outputs: Vec<(String, String, Option<String>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NModule {
    /// The function's unique name in the stack (`ScaleColor`, `FloatFromCurve002`).
    pub name: String,
    /// The script's object path (`/Niagara/Modules/Emitter/SpawnRate.SpawnRate`)
    /// or the scratch pad's name in this package.
    pub script: String,
    pub kind: ModuleKind,
    pub enabled: bool,
    /// Inputs with a value: overrides, then rapid-iteration constants, in
    /// that order of precedence (one entry per input name).
    pub inputs: Vec<(String, NInput)>,
    /// Static switch pins on the call node (name, default value string).
    pub switches: Vec<(String, String)>,
}

impl NModule {
    /// The script's short name (`SpawnRate`).
    pub fn script_name(&self) -> &str {
        let s = self.script.rsplit('/').next().unwrap_or(&self.script);
        s.rsplit('.').next().unwrap_or(s)
    }
    pub fn input(&self, name: &str) -> Option<&NInput> {
        self.inputs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
    pub fn has(&self, name: &str) -> bool {
        self.input(name).is_some()
    }
    pub fn switch(&self, name: &str) -> Option<&str> {
        self.switches.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Usage {
    SystemSpawn,
    SystemUpdate,
    EmitterSpawn,
    EmitterUpdate,
    ParticleSpawn,
    ParticleUpdate,
    Event(EventHandler),
    SimStage(SimStage),
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct EventHandler {
    /// "SpawnedParticles", "EveryParticle", "SingleParticle".
    pub execution_mode: String,
    pub spawn_number: u32,
    pub max_events_per_frame: u32,
    pub source_emitter: Option<String>,
    pub event_name: String,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct SimStage {
    pub name: String,
    pub iterations: u32,
    /// "Particles" or "DataInterface".
    pub source: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NStack {
    pub usage: Usage,
    pub modules: Vec<NModule>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NRenderer {
    /// E.g. `NiagaraSpriteRendererProperties`.
    pub class: String,
    pub enabled: bool,
    pub properties: Vec<Property>,
    /// Object paths referenced by the renderer's properties (materials,
    /// meshes), resolved.
    pub refs: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NEmitter {
    pub name: String,
    pub enabled: bool,
    pub gpu: bool,
    pub local_space: bool,
    pub deterministic: bool,
    /// Min/max in centimetres, when the bounds are fixed.
    pub fixed_bounds: Option<([f32; 3], [f32; 3])>,
    pub stacks: Vec<NStack>,
    pub renderers: Vec<NRenderer>,
    /// Things noticed while reading the emitter that the report should list.
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NSystem {
    pub name: String,
    pub user_parameters: Vec<(String, NValue)>,
    pub emitters: Vec<NEmitter>,
    pub stacks: Vec<NStack>,
    pub warmup_time: f32,
    pub notes: Vec<String>,
}

/// Reads the Niagara system in a package.
pub fn read_system(pkg: &Package) -> Result<NSystem, String> {
    let si = (0..pkg.exports.len())
        .find(|&i| pkg.export_class(i) == "NiagaraSystem" && pkg.exports[i].is_asset)
        .or_else(|| pkg.find_export_by_class("NiagaraSystem"))
        .ok_or("no NiagaraSystem export in the package")?;
    let r = Interp { pkg };
    let props = &pkg.exports[si].object.properties;
    let mut sys = NSystem {
        name: pkg.exports[si].object_name.clone(),
        user_parameters: Vec::new(),
        emitters: Vec::new(),
        stacks: Vec::new(),
        warmup_time: find(props, "WarmupTime").and_then(Value::as_f64).unwrap_or(0.0) as f32,
        notes: Vec::new(),
    };
    if let Some(store) = find(props, "ExposedParameters") {
        for (name, v) in r.store(store) {
            sys.user_parameters.push((name, v));
        }
    }
    // System scripts: their constants are named Constants.System.<Function>.<Input>.
    let mut sys_scripts = Vec::new();
    for key in ["SystemSpawnScript", "SystemUpdateScript"] {
        if let Some(i) = obj(find(props, key)) {
            sys_scripts.push(i);
        }
    }
    let consts = r.constants(&sys_scripts);
    if let Some(src) = sys_scripts.first().and_then(|&s| r.script_source(s)) {
        sys.stacks = r.stacks(src, &consts, &HashMap::new(), &HashMap::new());
    }

    let handles = match find(props, "EmitterHandles") {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    // Emitter ids, for event handler sources.
    let mut ids = HashMap::new();
    for h in &handles {
        if let (Some(Value::Guid(g)), Some(n)) = (h.field("Id"), h.field("Name").and_then(Value::as_str)) {
            ids.insert(g.to_string(), n.to_string());
        }
    }
    for h in &handles {
        let name = h.field("Name").and_then(Value::as_str).unwrap_or("Emitter").to_string();
        let enabled = !matches!(h.field("bIsEnabled"), Some(Value::Bool(false)));
        let emitter = h
            .field("VersionedInstance")
            .and_then(|v| v.field("Emitter"))
            .or_else(|| h.field("Instance"));
        let Some(ei) = obj(emitter) else {
            sys.notes.push(format!("emitter {name}: no emitter object (stateless or missing)"));
            continue;
        };
        sys.emitters.push(r.emitter(ei, name, enabled, &ids));
    }
    Ok(sys)
}

fn obj(v: Option<&Value>) -> Option<usize> {
    match v {
        Some(Value::Object(i)) => i.export(),
        _ => None,
    }
}

struct Interp<'a> {
    pkg: &'a Package,
}

/// Overrides collected from the map-set nodes of one stack:
/// `<Function>.<Input>` -> the pin.
type Overrides<'a> = HashMap<String, (usize, &'a Pin)>;

impl<'a> Interp<'a> {
    fn props(&self, i: usize) -> &'a [Property] {
        &self.pkg.exports[i].object.properties
    }
    fn class(&self, i: usize) -> String {
        self.pkg.export_class(i)
    }
    /// The struct or enum name of an object reference (`NiagaraFloat`).
    fn short_path(&self, idx: PackageIndex) -> String {
        let p = self.pkg.path(idx);
        p.rsplit(['.', ':']).next().unwrap_or(&p).to_string()
    }

    /// The version data of a versioned asset (the last entry).
    fn version_data(&self, i: usize) -> Option<&'a [Property]> {
        match find(self.props(i), "VersionData") {
            Some(Value::Array(a)) => match a.last() {
                Some(Value::Struct { fields, .. }) => Some(fields),
                _ => None,
            },
            _ => None,
        }
    }

    fn script_source(&self, script: usize) -> Option<usize> {
        let vd = self.version_data(script)?;
        obj(find(vd, "Source")).or_else(|| obj(find(self.props(script), "Source")))
    }

    /// The graph of a script source (NiagaraScriptSource.NodeGraph).
    fn graph(&self, source: usize) -> Option<usize> {
        obj(find(self.props(source), "NodeGraph")).or_else(|| {
            (0..self.pkg.exports.len())
                .find(|&i| self.pkg.exports[i].outer.export() == Some(source) && self.class(i) == "NiagaraGraph")
        })
    }

    fn nodes(&self, graph: usize) -> Vec<usize> {
        (0..self.pkg.exports.len())
            .filter(|&i| self.pkg.exports[i].outer.export() == Some(graph))
            .collect()
    }

    /// Values of a parameter store (NiagaraParameterStore), by name.
    fn store(&self, store: &Value) -> Vec<(String, NValue)> {
        let mut out = Vec::new();
        let data: Vec<u8> = match store.field("ParameterData") {
            Some(Value::Bytes(b)) => b.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_f64().map(|f| f as u8)).collect(),
            _ => Vec::new(),
        };
        let dis: Vec<Option<usize>> = match store.field("DataInterfaces") {
            Some(Value::Array(a)) => a.iter().map(|v| obj(Some(v))).collect(),
            _ => Vec::new(),
        };
        let Some(Value::Array(vars)) = store.field("SortedParameterOffsets") else {
            return out;
        };
        for v in vars {
            let Some(name) = v.field("Name").and_then(Value::as_str) else { continue };
            let offset = v.field("Offset").and_then(Value::as_f64).unwrap_or(0.0) as usize;
            let td = v.field("TypeDef");
            let Some(ty) = td.and_then(|t| self.type_ref(t)) else { continue };
            let val = match self.type_kind(ty) {
                TypeKind::DataInterface(class) => match dis.get(offset).copied().flatten() {
                    Some(di) => self.data_interface(di),
                    None => NValue::DataInterface(class),
                },
                TypeKind::Object => continue,
                TypeKind::Enum => match le_i32(&data, offset) {
                    Some(n) => NValue::Enum(format!("NewEnumerator{n}")),
                    None => continue,
                },
                TypeKind::Struct(s) => match decode_struct(&s, &data, offset) {
                    Some(v) => v,
                    None => continue,
                },
            };
            out.push((name.to_string(), val));
        }
        out
    }

    fn type_ref(&self, typedef: &Value) -> Option<PackageIndex> {
        for f in ["ClassStructOrEnum", "Struct", "Enum"] {
            if let Some(Value::Object(i)) = typedef.field(f) {
                if !i.is_null() {
                    return Some(*i);
                }
            }
        }
        None
    }

    fn type_kind(&self, ty: PackageIndex) -> TypeKind {
        let class = match ty.import() {
            Some(i) => self.pkg.imports[i].class_name.clone(),
            None => ty.export().map(|e| self.class(e)).unwrap_or_default(),
        };
        let name = self.short_path(ty);
        match class.as_str() {
            "Class" | "BlueprintGeneratedClass" if name.contains("DataInterface") => TypeKind::DataInterface(name),
            "Class" | "BlueprintGeneratedClass" => TypeKind::Object,
            "Enum" | "UserDefinedEnum" => TypeKind::Enum,
            _ => TypeKind::Struct(name),
        }
    }

    /// A data interface export: curves are read, others named.
    fn data_interface(&self, i: usize) -> NValue {
        let class = self.class(i);
        let channels: &[&str] = match class.as_str() {
            "NiagaraDataInterfaceCurve" => &["Curve"],
            "NiagaraDataInterfaceVector2DCurve" => &["XCurve", "YCurve"],
            "NiagaraDataInterfaceVectorCurve" => &["XCurve", "YCurve", "ZCurve"],
            "NiagaraDataInterfaceVector4Curve" => &["XCurve", "YCurve", "ZCurve", "WCurve"],
            "NiagaraDataInterfaceColorCurve" => &["RedCurve", "GreenCurve", "BlueCurve", "AlphaCurve"],
            _ => return NValue::DataInterface(class),
        };
        let props = self.props(i);
        let channels = channels
            .iter()
            .map(|c| match find(props, c).and_then(|v| v.field("Keys")) {
                Some(Value::Array(keys)) => keys.iter().map(curve_key).collect(),
                _ => Vec::new(),
            })
            .collect();
        NValue::Curve(NCurve { class, channels })
    }

    /// Rapid-iteration constants of these scripts: `<Function>.<Input>` ->
    /// value (the `Constants.<Owner>.` prefix dropped).
    fn constants(&self, scripts: &[usize]) -> HashMap<String, NValue> {
        let mut out = HashMap::new();
        for &s in scripts {
            let Some(store) = find(self.props(s), "RapidIterationParameters") else { continue };
            for (name, v) in self.store(store) {
                let Some(rest) = name.strip_prefix("Constants.") else { continue };
                if let Some((_, key)) = rest.split_once('.') {
                    out.entry(key.to_string()).or_insert(v.clone());
                }
                // System scripts have no owner segment (`Constants.SystemState.X`).
                out.entry(rest.to_string()).or_insert(v);
            }
        }
        out
    }

    fn emitter(&self, ei: usize, handle_name: String, enabled: bool, ids: &HashMap<String, String>) -> NEmitter {
        let eprops = self.props(ei);
        let vd = self.version_data(ei).unwrap_or(eprops);
        let mut e = NEmitter {
            name: handle_name,
            enabled,
            gpu: find(vd, "SimTarget").and_then(Value::as_str).is_some_and(|s| s.contains("GPU")),
            local_space: matches!(find(vd, "bLocalSpace"), Some(Value::Bool(true))),
            deterministic: matches!(find(vd, "bDeterminism"), Some(Value::Bool(true))),
            fixed_bounds: None,
            stacks: Vec::new(),
            renderers: Vec::new(),
            notes: Vec::new(),
        };
        if find(vd, "CalculateBoundsMode").and_then(Value::as_str).is_some_and(|s| s.ends_with("Fixed")) {
            if let Some(b) = find(vd, "FixedBounds") {
                let v3 = |n: &str| -> Option<[f32; 3]> {
                    let v = b.field(n)?;
                    Some([
                        v.field("X")?.as_f64()? as f32,
                        v.field("Y")?.as_f64()? as f32,
                        v.field("Z")?.as_f64()? as f32,
                    ])
                };
                if let (Some(a), Some(b)) = (v3("Min"), v3("Max")) {
                    e.fixed_bounds = Some((a, b));
                }
            }
        }

        // Scripts by usage, for constants and event / stage properties.
        let script_of = |key: &str| find(vd, key).and_then(|v| obj(v.field("Script")));
        let mut scripts = Vec::new();
        // The scripts the emitter runs: a GPU emitter's particle stacks are
        // compiled into its GPU compute script, a CPU emitter's into the spawn
        // and update scripts; the other kind holds stale constants.
        for key in ["EmitterSpawnScriptProps", "EmitterUpdateScriptProps"] {
            scripts.extend(script_of(key));
        }
        if e.gpu {
            scripts.extend(obj(find(vd, "GPUComputeScript")));
        } else {
            for key in ["SpawnScriptProps", "UpdateScriptProps"] {
                scripts.extend(script_of(key));
            }
        }
        // Event handlers and simulation stages, by their script's usage id.
        let mut events: HashMap<String, EventHandler> = HashMap::new();
        if let Some(Value::Array(a)) = find(vd, "EventHandlerScriptProps") {
            for ev in a {
                let Some(s) = obj(ev.field("Script")) else { continue };
                scripts.push(s);
                let id = usage_id(self.props(s));
                let src = ev.field("SourceEmitterID").map(|g| match g {
                    Value::Guid(g) => g.to_string(),
                    _ => String::new(),
                });
                events.insert(
                    id,
                    EventHandler {
                        execution_mode: ev
                            .field("ExecutionMode")
                            .and_then(Value::as_str)
                            .map(|s| s.rsplit("::").next().unwrap_or(s).to_string())
                            .unwrap_or_else(|| "EveryParticle".into()),
                        spawn_number: ev.field("SpawnNumber").and_then(Value::as_f64).unwrap_or(0.0) as u32,
                        max_events_per_frame: ev.field("MaxEventsPerFrame").and_then(Value::as_f64).unwrap_or(0.0) as u32,
                        source_emitter: src.and_then(|g| ids.get(&g).cloned()),
                        event_name: ev.field("SourceEventName").and_then(Value::as_str).unwrap_or("").to_string(),
                    },
                );
            }
        }
        let mut stages: HashMap<String, SimStage> = HashMap::new();
        if let Some(Value::Array(a)) = find(vd, "SimulationStages") {
            for st in a {
                let Some(si) = obj(Some(st)) else { continue };
                let p = self.props(si);
                let Some(s) = obj(find(p, "Script")) else { continue };
                scripts.push(s);
                stages.insert(
                    usage_id(self.props(s)),
                    SimStage {
                        name: find(p, "SimulationStageName").and_then(Value::as_str).unwrap_or("Stage").to_string(),
                        iterations: find(p, "NumIterations")
                            .or_else(|| find(p, "Iterations"))
                            .and_then(|v| v.as_f64().or_else(|| v.field("DefaultValue").and_then(Value::as_f64)))
                            .unwrap_or(1.0) as u32,
                        source: find(p, "IterationSource")
                            .and_then(Value::as_str)
                            .map(|s| s.rsplit("::").next().unwrap_or(s).to_string())
                            .unwrap_or_else(|| "Particles".into()),
                        enabled: !matches!(find(p, "bEnabled"), Some(Value::Bool(false))),
                    },
                );
            }
        }
        let consts = self.constants(&scripts);
        if let Some(src) = obj(find(vd, "GraphSource")) {
            e.stacks = self.stacks(src, &consts, &events, &stages);
        } else {
            e.notes.push("no graph source (cooked or stateless emitter)".into());
        }

        if let Some(Value::Array(a)) = find(vd, "RendererProperties") {
            for r in a {
                let Some(ri) = obj(Some(r)) else { continue };
                let props = self.props(ri).to_vec();
                let mut refs = HashMap::new();
                collect_refs(self.pkg, "", &props, &mut refs);
                e.renderers.push(NRenderer {
                    class: self.class(ri),
                    enabled: !matches!(find(&props, "bIsEnabled"), Some(Value::Bool(false))),
                    properties: props,
                    refs,
                });
            }
        }
        e
    }

    /// Every stack of a graph, in Niagara's order.
    fn stacks(
        &self,
        source: usize,
        consts: &HashMap<String, NValue>,
        events: &HashMap<String, EventHandler>,
        stages: &HashMap<String, SimStage>,
    ) -> Vec<NStack> {
        let Some(graph) = self.graph(source) else { return Vec::new() };
        let nodes = self.nodes(graph);
        let mut out = Vec::new();
        for &n in &nodes {
            if self.class(n) != "NiagaraNodeOutput" {
                continue;
            }
            let p = self.props(n);
            let ty = find(p, "ScriptType").and_then(Value::as_str).unwrap_or("");
            let ty = ty.rsplit("::").next().unwrap_or(ty);
            let id = match find(p, "ScriptTypeId") {
                Some(Value::Guid(g)) => g.to_string(),
                _ => String::new(),
            };
            let usage = match ty {
                "SystemSpawnScript" => Usage::SystemSpawn,
                "SystemUpdateScript" => Usage::SystemUpdate,
                "EmitterSpawnScript" => Usage::EmitterSpawn,
                "EmitterUpdateScript" => Usage::EmitterUpdate,
                "ParticleSpawnScript" | "ParticleSpawnScriptInterpolated" => Usage::ParticleSpawn,
                "ParticleUpdateScript" => Usage::ParticleUpdate,
                "ParticleEventScript" => Usage::Event(events.get(&id).cloned().unwrap_or_default()),
                "ParticleSimulationStageScript" => match stages.get(&id) {
                    Some(s) => Usage::SimStage(s.clone()),
                    None => continue, // a deleted stage's leftover output node
                },
                other => Usage::Other(other.to_string()),
            };
            if let Usage::Event(ev) = &usage {
                if !events.is_empty() && *ev == EventHandler::default() {
                    continue;
                }
            }
            out.push(NStack {
                usage,
                modules: self.chain(n, consts),
            });
        }
        out.sort_by_key(|s| match &s.usage {
            Usage::SystemSpawn => 0,
            Usage::SystemUpdate => 1,
            Usage::EmitterSpawn => 2,
            Usage::EmitterUpdate => 3,
            Usage::ParticleSpawn => 4,
            Usage::ParticleUpdate => 5,
            Usage::Event(_) => 6,
            Usage::SimStage(_) => 7,
            Usage::Other(_) => 8,
        });
        out
    }

    fn pins(&self, node: usize) -> &'a [Pin] {
        &self.pkg.exports[node].object.pins
    }

    /// The node and pin a pin is linked to (its first link).
    fn linked(&self, pin: &Pin) -> Option<(usize, &'a Pin)> {
        let l = pin.linked_to.first()?;
        let node = l.node.export()?;
        let p = self.pins(node).iter().find(|q| q.id == l.pin)?;
        Some((node, p))
    }

    fn is_map(&self, pin: &Pin) -> bool {
        self.short_path(pin.type_object()) == "NiagaraParameterMap"
    }

    /// Walks a stack back from its output node.
    fn chain(&self, output: usize, consts: &HashMap<String, NValue>) -> Vec<NModule> {
        let mut calls = Vec::new();
        let mut overrides: Overrides = HashMap::new();
        let mut cur = self
            .pins(output)
            .iter()
            .find(|p| p.direction == Direction::Input && self.is_map(p))
            .and_then(|p| self.linked(p));
        let mut guard = 0;
        while let Some((node, _pin)) = cur {
            guard += 1;
            if guard > 10_000 {
                break;
            }
            let class = self.class(node);
            let map_in = self
                .pins(node)
                .iter()
                .find(|p| p.direction == Direction::Input && self.is_map(p));
            match class.as_str() {
                "NiagaraNodeParameterMapSet" => {
                    for p in self.pins(node) {
                        if p.direction == Direction::Input && !self.is_map(p) && p.name != "Add" {
                            overrides.entry(p.name.clone()).or_insert((node, p));
                        }
                    }
                }
                "NiagaraNodeFunctionCall" | "NiagaraNodeAssignment" => calls.push(node),
                "NiagaraNodeInput" => break,
                _ => {}
            }
            cur = map_in.and_then(|p| self.linked(p));
        }
        calls.reverse();
        calls.iter().map(|&c| self.module(c, &overrides, consts, 0)).collect()
    }

    fn module(&self, node: usize, ov: &Overrides, consts: &HashMap<String, NValue>, depth: usize) -> NModule {
        let p = self.props(node);
        let name = find(p, "FunctionDisplayName")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| self.pkg.exports[node].object_name.clone());
        let fs = match find(p, "FunctionScript") {
            Some(Value::Object(i)) => *i,
            _ => PackageIndex(0),
        };
        let mut kind = ModuleKind::Function;
        let mut script = self.pkg.path(fs);
        if self.class(node) == "NiagaraNodeAssignment" {
            let targets: Vec<String> = match find(p, "AssignmentTargets") {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|t| t.field("Name").and_then(Value::as_str).map(str::to_string))
                    .collect(),
                _ => Vec::new(),
            };
            let defaults: Vec<String> = match find(p, "AssignmentDefaultValues") {
                Some(Value::Array(a)) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(),
                _ => Vec::new(),
            };
            kind = ModuleKind::Assignment {
                targets: targets
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (t, defaults.get(i).cloned().unwrap_or_default()))
                    .collect(),
            };
            script = "SetVariables".into();
        } else if let Some(fe) = fs.export() {
            // A scratch pad script in this package: collect its HLSL.
            let mut hlsl = Vec::new();
            if let Some(g) = self.script_source(fe).and_then(|s| self.graph(s)) {
                for n in self.nodes(g) {
                    if self.class(n) == "NiagaraNodeCustomHlsl" {
                        if let Some(h) = find(self.props(n), "CustomHlsl").and_then(Value::as_str) {
                            hlsl.push(self.hlsl_node(n, h));
                        }
                    }
                }
            }
            kind = ModuleKind::ScratchPad { hlsl };
            script = self.pkg.exports[fe].object_name.clone();
        }
        let enabled = !find(p, "EnabledState")
            .and_then(Value::as_str)
            .is_some_and(|s| s.ends_with("Disabled"));

        let mut inputs: Vec<(String, NInput)> = Vec::new();
        let prefix = format!("{name}.");
        let mut ov_names: Vec<&String> = ov.keys().filter(|k| k.starts_with(&prefix)).collect();
        ov_names.sort();
        for k in ov_names {
            let (_, pin) = ov[k];
            let input = &k[prefix.len()..];
            let v = self.override_value(pin, ov, consts, depth);
            inputs.push((input.to_string(), v));
        }
        let mut c_names: Vec<&String> = consts.keys().filter(|k| k.starts_with(&prefix)).collect();
        c_names.sort();
        for k in c_names {
            let input = &k[prefix.len()..];
            if !inputs.iter().any(|(n, _)| n == input) {
                inputs.push((input.to_string(), NInput::Value(consts[k].clone())));
            }
        }
        let switches = self
            .pins(node)
            .iter()
            .filter(|q| q.direction == Direction::Input && !self.is_map(q) && q.linked_to.is_empty())
            .map(|q| (q.name.clone(), q.default_value.clone()))
            .collect();
        NModule {
            name,
            script,
            kind,
            enabled,
            inputs,
            switches,
        }
    }

    fn hlsl_node(&self, n: usize, code: &str) -> HlslNode {
        let mut node = HlslNode {
            code: code.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        for p in self.pins(n) {
            if self.is_map(p) || p.name == "Add" || p.name.is_empty() {
                continue;
            }
            let ty = self.short_path(p.type_object());
            let other = self.linked(p).and_then(|(o, q)| {
                let c = self.class(o);
                (c == "NiagaraNodeParameterMapGet" || c == "NiagaraNodeParameterMapSet").then(|| q.name.clone())
            });
            match p.direction {
                Direction::Input => node.inputs.push((p.name.clone(), ty, other)),
                Direction::Output => node.outputs.push((p.name.clone(), ty, other)),
            }
        }
        node
    }

    fn override_value(&self, pin: &Pin, ov: &Overrides, consts: &HashMap<String, NValue>, depth: usize) -> NInput {
        let Some((node, src)) = self.linked(pin) else {
            return match self.pin_value(pin) {
                Some(v) => NInput::Value(v),
                None => NInput::Unsupported(format!("default {:?} of type {}", pin.default_value, self.short_path(pin.type_object()))),
            };
        };
        match self.class(node).as_str() {
            "NiagaraNodeFunctionCall" if depth < 16 => NInput::Dynamic(Box::new(self.module(node, ov, consts, depth + 1))),
            "NiagaraNodeParameterMapGet" => NInput::Link(src.name.clone()),
            "NiagaraNodeInput" => {
                let p = self.props(node);
                if let Some(di) = obj(find(p, "DataInterface")) {
                    return NInput::Value(self.data_interface(di));
                }
                let var = find(p, "Input");
                let data = match var.and_then(|v| v.field("VarData")) {
                    Some(Value::Bytes(b)) => b.clone(),
                    _ => Vec::new(),
                };
                let ty = var.and_then(|v| v.field("TypeDef")).and_then(|t| self.type_ref(t));
                if let Some(ty) = ty {
                    if let TypeKind::Struct(s) = self.type_kind(ty) {
                        if let Some(v) = decode_struct(&s, &data, 0) {
                            return NInput::Value(v);
                        }
                    }
                }
                NInput::Unsupported(format!(
                    "input node {}",
                    var.and_then(|v| v.field("Name")).and_then(Value::as_str).unwrap_or("?")
                ))
            }
            other => NInput::Unsupported(format!("{other} {:?}", src.name)),
        }
    }

    /// A pin's default value string, typed by the pin.
    fn pin_value(&self, pin: &Pin) -> Option<NValue> {
        let ty = pin.type_object();
        let kind = if ty.is_null() { TypeKind::Object } else { self.type_kind(ty) };
        let s = pin.default_value.as_str();
        match kind {
            TypeKind::Enum => Some(NValue::Enum(s.to_string())),
            TypeKind::Struct(name) => parse_default(&name, s),
            _ => None,
        }
    }
}

enum TypeKind {
    Struct(String),
    Enum,
    DataInterface(String),
    Object,
}

fn usage_id(props: &[Property]) -> String {
    match find(props, "UsageId") {
        Some(Value::Guid(g)) => g.to_string(),
        _ => String::new(),
    }
}

fn curve_key(v: &Value) -> CurveKey {
    let f = |n: &str| v.field(n).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let interp = match v.field("InterpMode") {
        Some(Value::Enum(s)) => {
            if s.ends_with("Linear") {
                0
            } else if s.ends_with("Constant") {
                1
            } else if s.ends_with("Cubic") {
                2
            } else {
                0
            }
        }
        Some(x) => x.as_f64().unwrap_or(0.0) as u8,
        // Property left at its default: RCIM_Linear.
        None => 0,
    };
    CurveKey {
        time: f("Time"),
        value: f("Value"),
        interp,
        arrive: f("ArriveTangent"),
        leave: f("LeaveTangent"),
    }
}

fn le_f32(d: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}
fn le_i32(d: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}
fn floats<const N: usize>(d: &[u8], o: usize) -> Option<[f32; N]> {
    let mut a = [0.0; N];
    for (k, x) in a.iter_mut().enumerate() {
        *x = le_f32(d, o + 4 * k)?;
    }
    Some(a)
}

/// A value of a Niagara type from parameter-store bytes.
fn decode_struct(name: &str, d: &[u8], o: usize) -> Option<NValue> {
    Some(match name {
        "NiagaraFloat" | "NiagaraHalf" => NValue::Float(le_f32(d, o)?),
        "NiagaraInt32" => NValue::Int(le_i32(d, o)?),
        "NiagaraBool" => NValue::Bool(le_i32(d, o)? != 0),
        "Vector2f" | "Vector2D" | "NiagaraHalfVector2" => NValue::Vec2(floats(d, o)?),
        "Vector3f" | "Vector" => NValue::Vec3(floats(d, o)?),
        "NiagaraPosition" => NValue::Position(floats(d, o)?),
        "Vector4f" | "Vector4" => NValue::Vec4(floats(d, o)?),
        "LinearColor" => NValue::Color(floats(d, o)?),
        "Quat4f" | "Quat" => NValue::Quat(floats(d, o)?),
        "NiagaraMatrix" => NValue::Matrix(floats::<16>(d, o)?.to_vec()),
        _ => return None,
    })
}

/// Every number in a string (`(R=1.0,G=0.5,...)`, `X=0.0 Y=1.0`, `1,2,3`).
pub fn numbers(s: &str) -> Vec<f32> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let starts = c.is_ascii_digit()
            || ((c == b'-' || c == b'+' || c == b'.') && b.get(i + 1).is_some_and(|n| n.is_ascii_digit() || *n == b'.'));
        // Skip letters directly before digits (e.g. a name like "R2").
        if starts && (i == 0 || !b[i - 1].is_ascii_alphabetic()) {
            let mut j = i + 1;
            while j < b.len()
                && (b[j].is_ascii_digit()
                    || b[j] == b'.'
                    || b[j] == b'e'
                    || b[j] == b'E'
                    || ((b[j] == b'-' || b[j] == b'+') && matches!(b[j - 1], b'e' | b'E')))
            {
                j += 1;
            }
            if let Ok(v) = s[i..j].parse::<f32>() {
                out.push(v);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// A pin default value string as a value of the named Niagara type.
fn parse_default(name: &str, s: &str) -> Option<NValue> {
    let n = numbers(s);
    let get = |k: usize| n.get(k).copied();
    Some(match name {
        "NiagaraBool" => NValue::Bool(s.trim().eq_ignore_ascii_case("true") || s.trim() == "1"),
        "NiagaraFloat" | "NiagaraNumeric" => NValue::Float(get(0)?),
        "NiagaraInt32" => NValue::Int(get(0)? as i32),
        "Vector2f" | "Vector2D" => NValue::Vec2([get(0)?, get(1)?]),
        "Vector3f" | "Vector" => NValue::Vec3([get(0)?, get(1)?, get(2)?]),
        "NiagaraPosition" => NValue::Position([get(0)?, get(1)?, get(2)?]),
        "Vector4f" | "Vector4" => NValue::Vec4([get(0)?, get(1)?, get(2)?, get(3)?]),
        "LinearColor" => NValue::Color([get(0)?, get(1)?, get(2)?, get(3)?]),
        "Quat4f" | "Quat" => NValue::Quat([get(0)?, get(1)?, get(2)?, get(3)?]),
        _ => return None,
    })
}

/// Object references inside renderer properties, by property path.
fn collect_refs(pkg: &Package, prefix: &str, props: &[Property], out: &mut HashMap<String, String>) {
    for p in props {
        let path = if prefix.is_empty() {
            p.name.clone()
        } else {
            format!("{prefix}.{}", p.name)
        };
        collect_value_refs(pkg, &path, &p.value, out);
    }
}

fn collect_value_refs(pkg: &Package, path: &str, v: &Value, out: &mut HashMap<String, String>) {
    match v {
        Value::Object(i) if !i.is_null() => {
            out.insert(path.to_string(), pkg.path(*i));
        }
        Value::SoftObject(s) if !s.package.is_empty() => {
            out.insert(path.to_string(), s.to_string());
        }
        Value::Struct { fields, .. } => collect_refs(pkg, path, fields, out),
        Value::Array(a) => {
            for (k, e) in a.iter().enumerate() {
                collect_value_refs(pkg, &format!("{path}[{k}]"), e, out);
            }
        }
        _ => {}
    }
}

/// A readable listing of an interpreted system (for `fximport niagara`).
pub fn describe(sys: &NSystem) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let _ = writeln!(o, "System {} (warm-up {} s)", sys.name, sys.warmup_time);
    for (n, v) in &sys.user_parameters {
        let _ = writeln!(o, "  user {n} = {}", short_value(v));
    }
    for n in &sys.notes {
        let _ = writeln!(o, "  note: {n}");
    }
    for s in &sys.stacks {
        describe_stack(&mut o, s, "  ");
    }
    for e in &sys.emitters {
        let _ = writeln!(
            o,
            "Emitter {}{} ({}, {}{})",
            e.name,
            if e.enabled { "" } else { " [disabled]" },
            if e.gpu { "GPU" } else { "CPU" },
            if e.local_space { "local" } else { "world" },
            if e.fixed_bounds.is_some() { ", fixed bounds" } else { "" }
        );
        for n in &e.notes {
            let _ = writeln!(o, "  note: {n}");
        }
        for s in &e.stacks {
            describe_stack(&mut o, s, "  ");
        }
        for r in &e.renderers {
            let _ = writeln!(o, "  renderer {}{}", r.class, if r.enabled { "" } else { " [disabled]" });
            let mut refs: Vec<_> = r.refs.iter().collect();
            refs.sort();
            for (k, v) in refs {
                let _ = writeln!(o, "    {k} = {v}");
            }
        }
    }
    o
}

fn describe_stack(o: &mut String, s: &NStack, ind: &str) {
    use std::fmt::Write;
    let _ = writeln!(o, "{ind}{:?}", s.usage);
    for m in &s.modules {
        describe_module(o, m, &format!("{ind}  "));
    }
}

fn describe_module(o: &mut String, m: &NModule, ind: &str) {
    use std::fmt::Write;
    let _ = writeln!(
        o,
        "{ind}{} <{}>{}{}",
        m.name,
        m.script,
        if m.enabled { "" } else { " [disabled]" },
        match &m.kind {
            ModuleKind::Function => String::new(),
            ModuleKind::Assignment { targets } => format!(" sets {targets:?}"),
            ModuleKind::ScratchPad { hlsl } => format!(" scratch pad, {} HLSL nodes", hlsl.len()),
        }
    );
    for (k, v) in &m.switches {
        let _ = writeln!(o, "{ind}  switch {k} = {v}");
    }
    for (k, v) in &m.inputs {
        match v {
            NInput::Dynamic(d) => {
                let _ = writeln!(o, "{ind}  {k} =");
                describe_module(o, d, &format!("{ind}    "));
            }
            NInput::Value(v) => {
                let _ = writeln!(o, "{ind}  {k} = {}", short_value(v));
            }
            NInput::Link(l) => {
                let _ = writeln!(o, "{ind}  {k} -> {l}");
            }
            NInput::Unsupported(u) => {
                let _ = writeln!(o, "{ind}  {k} ?? {u}");
            }
        }
    }
}

fn short_value(v: &NValue) -> String {
    match v {
        NValue::Curve(c) => format!(
            "{} curve ({} keys)",
            c.class.trim_start_matches("NiagaraDataInterface"),
            c.channels.iter().map(Vec::len).max().unwrap_or(0)
        ),
        NValue::Matrix(_) => "Matrix".into(),
        other => format!("{other:?}"),
    }
}
