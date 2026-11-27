//! Unreal Blueprint actors to prefabs (`fximport blueprint`): the VFX parts
//! only.
//!
//! - **Classes**: the Blueprint's parents (Blueprints, up to the native
//!   class) are loaded from the package folders; a child inherits their
//!   components, variables, graphs and timelines and overrides them (class
//!   default object values, inherited component templates).
//! - **Components** (SimpleConstructionScript nodes and their templates):
//!   Niagara (system, user parameter overrides, auto activate), static mesh
//!   (mesh, material overrides), point lights, cameras, spring arms and
//!   plain scene components keep their transforms and attachment; others
//!   (collision, movement, audio) become empty nodes or are left out, listed.
//! - **Construction script and BeginPlay**: run as far as they are a fixed
//!   sequence of calls on the components with arguments known at import
//!   (constants, class defaults, game time): sequences, branches on such
//!   values, `Delay`s (timed), custom events of the actor and parent calls
//!   are followed; a node reached twice (a loop), an unknown branch or an
//!   unknown node stops that path, listed in the report. The construction
//!   script's effects are baked; BeginPlay's become timed prefab actions
//!   (material parameters, custom primitive data, visibility, activation,
//!   Niagara variables, timeline play/stop/rate).
//! - **Timelines**: float, vector and colour tracks (their curves) and the
//!   calls their Update output makes with them: transforms, material
//!   parameters, custom primitive data, Niagara variables.
//!
//! Material parameters that something sets after spawn read from dynamic
//! material parameters fed by `User.` Vec4 parameters of the mesh's effect
//! ([`fx_core::prefab::MaterialParam`]); the rest are baked.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use fx_core::prefab::{self, Action, Channel, Do, Interp as KInterp, Key, MaterialParam, Node, NodeKind, Num, ParamValue, Prefab, Property, Timeline, Track};

use crate::convert::{AssetOut, MatTweak};
use crate::material::{split_object_path, Loaded, Store};
use crate::package::{Package, RF_CLASS_DEFAULT_OBJECT};
use crate::pins::{Direction, Pin};
use crate::property::{find, Property as UProp, Value};

/// Seconds of BeginPlay followed (later `Delay`s are left out).
const HORIZON: f64 = 120.0;
/// Prefix of a material key naming a custom primitive data index.
const CPD: &str = "\u{1}cpd:";

/// What [`import`] produced.
pub struct Imported {
    pub prefab: Prefab,
    /// Files to write beside the prefab (relative path, bytes): converted
    /// effects, mesh effects, meshes, materials, textures.
    pub files: BTreeMap<String, Vec<u8>>,
    pub report: String,
    /// One line per component imported, for the summary.
    pub components: usize,
    /// Logic not imported (report entries).
    pub skipped: usize,
}

/// Niagara systems converted in this run (object path -> effect file).
#[derive(Default)]
pub struct Effects {
    done: HashMap<String, Result<String, String>>,
}

// ---- values ----

#[derive(Clone, Debug, PartialEq)]
enum V {
    Num(f64),
    Bool(bool),
    Str(String),
    /// An object path (asset), or empty for none.
    Obj(String),
    /// A component of this actor, by variable name.
    Comp(String),
    /// A dynamic material instance of a component's material slot.
    Mid(String, usize),
    Timeline(String),
    Actor,
    /// Game time when the action runs, plus a constant.
    Time(f64),
    /// A timeline track's value (component `comp`), `scale * v + offset`.
    Track { timeline: String, track: String, comp: usize, scale: f64, offset: f64 },
    /// A struct's components: Vector (X, Y, Z), Rotator (Pitch, Yaw, Roll),
    /// LinearColor (R, G, B, A), Vector2D (X, Y).
    St(Vec<V>),
    Unknown(String),
}

impl V {
    fn num(&self) -> Option<f64> {
        match self {
            V::Num(x) => Some(*x),
            V::Bool(b) => Some(*b as i32 as f64),
            _ => None,
        }
    }
    fn nums<const N: usize>(&self) -> Option<[f64; N]> {
        let V::St(c) = self else { return None };
        let mut o = [0.0; N];
        for (i, x) in o.iter_mut().enumerate() {
            *x = c.get(i)?.num()?;
        }
        Some(o)
    }
    fn truth(&self) -> Option<bool> {
        match self {
            V::Bool(b) => Some(*b),
            V::Num(x) => Some(*x != 0.0),
            _ => None,
        }
    }
    fn describe(&self) -> String {
        match self {
            V::Num(x) => format!("{}", *x as f32),
            V::Bool(b) => b.to_string(),
            V::Str(s) => format!("\"{s}\""),
            V::Obj(p) if p.is_empty() => "None".into(),
            V::Obj(p) => split_object_path(p).1,
            V::Comp(c) => c.clone(),
            V::Mid(c, i) => format!("{c} material {i} (dynamic instance)"),
            V::Timeline(t) => t.clone(),
            V::Actor => "self".into(),
            V::Time(p) if *p == 0.0 => "game time".into(),
            V::Time(p) => format!("game time + {}", *p as f32),
            V::Track { timeline, track, comp, .. } => format!("{timeline}.{track}[{comp}]"),
            V::St(c) => format!("({})", c.iter().map(|x| x.describe()).collect::<Vec<_>>().join(", ")),
            V::Unknown(w) => format!("<{w}>"),
        }
    }
    fn to_num(&self) -> Option<Num> {
        match self {
            V::Time(p) => Some(Num::GameTime { plus: *p as f32 }),
            v => v.num().map(|x| Num::Value(x as f32)),
        }
    }
}

fn struct_fields(name: &str) -> &'static [&'static str] {
    match name {
        "Vector" | "Vector3f" | "Vector3d" => &["X", "Y", "Z"],
        "Rotator" | "Rotator3f" | "Rotator3d" => &["Pitch", "Yaw", "Roll"],
        "LinearColor" => &["R", "G", "B", "A"],
        "Color" => &["R", "G", "B", "A"],
        "Vector2D" | "Vector2f" => &["X", "Y"],
        "Vector4" | "Vector4f" | "Vector4d" => &["X", "Y", "Z", "W"],
        _ => &[],
    }
}

/// A property value as a `V`.
fn from_value(l: &Loaded, v: &Value) -> V {
    match v {
        Value::Bool(b) => V::Bool(*b),
        Value::Object(i) if i.is_null() => V::Obj(String::new()),
        Value::Object(i) => V::Obj(l.path(*i)),
        Value::SoftObject(p) => V::Obj(p.to_string()),
        Value::Name(s) | Value::Str(s) | Value::Enum(s) => V::Str(s.clone()),
        Value::Struct { name, fields } => {
            let names = struct_fields(name);
            if names.is_empty() {
                return V::Unknown(format!("struct {name}"));
            }
            let get = |n: &str| fields.iter().find(|p| p.name == n).and_then(|p| p.value.as_f64()).unwrap_or(0.0);
            if name == "Color" {
                // FColor: 0..255 sRGB bytes.
                return V::St(names.iter().map(|n| V::Num(srgb_to_linear(get(n) / 255.0))).collect());
            }
            V::St(names.iter().map(|n| V::Num(get(n))).collect())
        }
        v => match v.as_f64() {
            Some(x) => V::Num(x),
            None => V::Unknown("value".into()),
        },
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// A pin default string (`"0.0"`, `"1,2,3"`, `"(R=1,G=0,B=0,A=1)"`,
/// `"true"`) as type `category`/`sub` (struct name).
fn parse_default(s: &str, category: &str, sub: &str) -> V {
    let s = s.trim();
    match category {
        "real" | "float" | "double" | "int" | "int64" | "byte" if s.parse::<f64>().is_ok() || s.is_empty() => V::Num(s.parse().unwrap_or(0.0)),
        "byte" | "name" | "string" | "text" => V::Str(s.to_string()),
        "bool" => V::Bool(s.eq_ignore_ascii_case("true")),
        "struct" => {
            let names = struct_fields(sub);
            if names.is_empty() {
                return V::Unknown(format!("struct {sub}"));
            }
            if s.starts_with('(') {
                let inner = s.trim_start_matches('(').trim_end_matches(')');
                let kv: HashMap<&str, f64> = inner
                    .split(',')
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| (k.trim(), v.trim().parse().unwrap_or(0.0)))
                    .collect();
                return V::St(names.iter().map(|n| V::Num(kv.get(n).copied().unwrap_or(if *n == "A" { 1.0 } else { 0.0 }))).collect());
            }
            let parts: Vec<f64> = s.split(',').map(|p| p.trim().parse().unwrap_or(0.0)).collect();
            let default = if matches!(sub, "Vector" | "Vector3f" | "Vector3d") && s.is_empty() { 0.0 } else { 0.0 };
            V::St((0..names.len()).map(|i| V::Num(parts.get(i).copied().unwrap_or(default))).collect())
        }
        "object" | "class" | "softobject" => V::Obj(if s == "None" { String::new() } else { s.to_string() }),
        _ => V::Unknown(format!("{category} default")),
    }
}

// ---- classes ----

/// One Blueprint of the chain (index 0: the imported one).
struct Class {
    l: Rc<Loaded>,
    name: String,
    bp: usize,
    cdo: Option<usize>,
}

impl Class {
    fn props(&self, i: usize) -> &[UProp] {
        &self.l.pkg.exports[i].object.properties
    }
    fn bp_props(&self) -> &[UProp] {
        self.props(self.bp)
    }
    fn class(&self, i: usize) -> String {
        self.l.pkg.export_class(i)
    }
    fn obj(&self, v: Option<&Value>) -> Option<usize> {
        match v {
            Some(Value::Object(i)) => i.export(),
            _ => None,
        }
    }
    fn objs(&self, v: Option<&Value>) -> Vec<usize> {
        match v {
            Some(Value::Array(a)) => a.iter().filter_map(|x| self.obj(Some(x))).collect(),
            _ => Vec::new(),
        }
    }
    fn pins(&self, i: usize) -> &[Pin] {
        &self.l.pkg.exports[i].object.pins
    }
    fn pin(&self, i: usize, name: &str, dir: Direction) -> Option<&Pin> {
        self.pins(i).iter().find(|p| p.name == name && p.direction == dir)
    }
    /// The node and pin an exec output or data input links to.
    fn follow(&self, _node: usize, pin: &Pin) -> Option<(usize, String)> {
        let r = pin.linked_to.first()?;
        let n = r.node.export()?;
        let p = self.pins(n).iter().find(|p| p.id == r.pin)?;
        Some((n, p.name.clone()))
    }
    fn follow_out(&self, node: usize, name: &str) -> Option<(usize, String)> {
        let p = self.pin(node, name, Direction::Output)?;
        self.follow(node, p)
    }
    fn member(&self, i: usize, field: &str) -> Option<String> {
        find(self.props(i), field).and_then(|m| m.field("MemberName")).and_then(|n| n.as_str()).map(String::from)
    }
    fn member_parent(&self, i: usize, field: &str) -> Option<String> {
        match find(self.props(i), field).and_then(|m| m.field("MemberParent")) {
            Some(Value::Object(o)) if !o.is_null() => Some(self.l.path(*o)),
            _ => None,
        }
    }
    fn disabled(&self, i: usize) -> bool {
        find(self.props(i), "EnabledState").and_then(|v| v.as_str()).is_some_and(|s| s.ends_with("Disabled"))
    }
    /// Graph node exports (of the event and function graphs).
    fn nodes(&self) -> Vec<usize> {
        let p = self.bp_props();
        let mut graphs = self.objs(find(p, "UbergraphPages"));
        graphs.extend(self.objs(find(p, "FunctionGraphs")));
        let e = &self.l.pkg.exports;
        (0..e.len()).filter(|&i| e[i].outer.export().is_some_and(|o| graphs.contains(&o))).collect()
    }
    fn graph_name(&self, node: usize) -> String {
        self.l.pkg.exports[node].outer.export().map(|g| self.l.pkg.exports[g].object_name.clone()).unwrap_or_default()
    }
    fn node_name(&self, i: usize) -> String {
        self.l.pkg.exports[i].object_name.clone()
    }
}

fn load_package(file: &Path) -> Result<Rc<Loaded>, String> {
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let pkg = Package::read(&bytes).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Rc::new(Loaded { file: file.to_path_buf(), bytes, pkg }))
}

fn make_class(l: Rc<Loaded>) -> Result<Class, String> {
    let e = &l.pkg.exports;
    let bp = (0..e.len())
        .find(|&i| matches!(l.pkg.export_class(i).as_str(), "Blueprint" | "WidgetBlueprint" | "AnimBlueprint") && e[i].outer.is_null())
        .ok_or("no Blueprint export (not a Blueprint package)")?;
    let name = e[bp].object_name.clone();
    let cdo = (0..e.len()).find(|&i| e[i].object_flags & RF_CLASS_DEFAULT_OBJECT != 0 && e[i].object_name.starts_with("Default__"));
    Ok(Class { l, name, bp, cdo })
}

// ---- components ----

#[derive(Clone, Debug)]
struct Comp {
    name: String,
    class: String,
    parent: Option<String>,
    props: Vec<UProp>,
    /// Which class package `props` object references resolve in.
    owner: usize,
    loc: [f64; 3],
    rot: [f64; 3],
    scale: [f64; 3],
    visible: bool,
    auto_activate: bool,
    asset: Option<String>,
    mesh: Option<String>,
    materials: Vec<Option<String>>,
    cpd: BTreeMap<u32, f64>,
    mat_scalars: BTreeMap<String, f64>,
    mat_vectors: BTreeMap<String, [f64; 4]>,
    params: Vec<(String, ParamValue)>,
    light: ([f64; 3], f64, f64),
    fov: f64,
    arm: (f64, [f64; 3]),
}

/// Classes drawn as something; the rest become empty nodes or are left out.
fn scene_class(class: &str) -> bool {
    !matches!(
        class,
        "ProjectileMovementComponent" | "RotatingMovementComponent" | "InterpToMovementComponent" | "FloatingPawnMovement" | "TimelineComponent" | "InputComponent"
    ) && !class.ends_with("MovementComponent")
}

// ---- the import ----

struct Bp {
    store: Store,
    classes: Vec<Class>,
    comps: Vec<Comp>,
    /// Variables set by graphs (VariableSet), over the class defaults.
    vars: HashMap<String, V>,
    /// Outputs of executed impure nodes (class, node, pin).
    memo: HashMap<(usize, usize, String), V>,
    timelines: Vec<TimelineInfo>,
    construct: bool,
    t: f64,
    visited: HashSet<(usize, usize)>,
    pending: Vec<(f64, usize, usize, String)>,
    actions: Vec<(f64, String, Act)>,
    tracks: Vec<(String, Track)>,
    done: Vec<String>,
    skipped: Vec<String>,
    ignored: BTreeMap<String, usize>,
}

#[derive(Clone, Debug)]
enum Act {
    Mat { key: String, value: Vec<Num> },
    Param(String, ParamValue),
    Visible(bool),
    Activate(bool),
    Deactivate,
    Play(bool),
    Stop,
    Rate(f32),
}

struct TimelineInfo {
    name: String,
    length: f64,
    looping: bool,
    autoplay: bool,
    rate: f64,
    /// Track name -> per-component keys.
    curves: HashMap<String, Vec<Vec<Key>>>,
    notes: Vec<String>,
}

/// Calls that change nothing drawn (collision, shadows, ...): not listed one by one.
const IGNORED: &[&str] = &[
    "SetCollisionEnabled",
    "SetCastShadow",
    "SetGenerateOverlapEvents",
    "SetCollisionProfileName",
    "SetCollisionResponseToAllChannels",
    "SetMobility",
    "SetCanEverAffectNavigation",
    "PrintString",
    "SetReceivesDecals",
];

impl Bp {
    fn var_default(&self, name: &str) -> Option<V> {
        for c in &self.classes {
            if let Some(cdo) = c.cdo {
                if let Some(v) = find(c.props(cdo), name) {
                    return Some(from_value(&c.l, v));
                }
            }
        }
        for c in &self.classes {
            let Some(Value::Array(vars)) = find(c.bp_props(), "NewVariables") else { continue };
            for d in vars {
                if d.field("VarName").and_then(|v| v.as_str()) != Some(name) {
                    continue;
                }
                let ty = d.field("VarType");
                let cat = ty.and_then(|t| t.field("PinCategory")).and_then(|v| v.as_str()).unwrap_or("");
                let sub = match ty.and_then(|t| t.field("PinSubCategoryObject")) {
                    Some(Value::Object(o)) if !o.is_null() => split_object_path(&c.l.path(*o)).1,
                    _ => String::new(),
                };
                let s = d.field("DefaultValue").and_then(|v| v.as_str()).unwrap_or("");
                return Some(parse_default(s, cat, &sub));
            }
        }
        None
    }

    fn var(&self, name: &str) -> V {
        if self.comps.iter().any(|c| c.name == name) {
            return V::Comp(name.to_string());
        }
        if self.timelines.iter().any(|t| t.name == name) {
            return V::Timeline(name.to_string());
        }
        if let Some(v) = self.vars.get(name) {
            return v.clone();
        }
        self.var_default(name).unwrap_or(V::Unknown(format!("variable {name}")))
    }

    /// Every parent of component `name` sits at identity: its world
    /// transform (relative to the actor) is its relative one.
    fn world_is_local(&self, name: &str) -> bool {
        let mut cur = self.comps.iter().find(|x| x.name == name).and_then(|x| x.parent.clone());
        let mut n = 0;
        while let Some(p) = cur {
            let Some(c) = self.comps.iter().find(|x| x.name == p) else { return false };
            if c.loc != [0.0; 3] || c.rot != [0.0; 3] || c.scale != [1.0; 3] || n > 32 {
                return false;
            }
            cur = c.parent.clone();
            n += 1;
        }
        true
    }

    fn comp_mut(&mut self, name: &str) -> Option<&mut Comp> {
        self.comps.iter_mut().find(|c| c.name == name)
    }

    // ---- data ----

    /// The value of input pin `pin` of node `n` in class `ci`.
    fn input(&mut self, ci: usize, n: usize, pin: &str) -> V {
        let Some(p) = self.classes[ci].pin(n, pin, Direction::Input).cloned() else {
            return V::Unknown(format!("no pin {pin}"));
        };
        self.pin_value(ci, n, &p, 0)
    }

    fn pin_value(&mut self, ci: usize, n: usize, p: &Pin, depth: usize) -> V {
        if depth > 32 {
            return V::Unknown("too deep".into());
        }
        let cat = p.type_field("PinCategory").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let sub = match p.type_object() {
            o if !o.is_null() => split_object_path(&self.classes[ci].l.path(o)).1,
            _ => String::new(),
        };
        // A split struct pin: its sub-pins hold the values.
        if !p.sub_pins.is_empty() {
            let names = struct_fields(&sub);
            let mut out = Vec::new();
            for f in names {
                let sp = self.classes[ci].pins(n).iter().find(|x| x.name == format!("{}_{f}", p.name)).cloned();
                out.push(match sp {
                    Some(sp) => self.pin_value(ci, n, &sp, depth + 1),
                    None => V::Num(0.0),
                });
            }
            return V::St(out);
        }
        if let Some((src, out)) = self.classes[ci].follow(n, p) {
            return self.output(ci, src, &out, depth + 1);
        }
        if !p.default_object.is_null() {
            return V::Obj(self.classes[ci].l.path(p.default_object));
        }
        parse_default(&p.default_value, &cat, &sub)
    }

    /// The value of output pin `pin` of node `n`.
    fn output(&mut self, ci: usize, n: usize, pin: &str, depth: usize) -> V {
        if let Some(v) = self.memo.get(&(ci, n, pin.to_string())) {
            return v.clone();
        }
        let c = &self.classes[ci];
        let class = c.class(n);
        match class.as_str() {
            "K2Node_VariableGet" => {
                if c.member_parent(n, "VariableReference").is_some_and(|p| !p.contains(&c.name) && !self.classes.iter().any(|k| p.contains(&k.name))) {
                    return V::Unknown(format!("variable {pin} of another class"));
                }
                let name = c.member(n, "VariableReference").unwrap_or_default();
                self.var(&name)
            }
            "K2Node_VariableSet" => {
                let name = c.member(n, "VariableReference").unwrap_or_default();
                self.var(&name)
            }
            "K2Node_Self" => V::Actor,
            "K2Node_Knot" => {
                let Some(p) = c.pin(n, "InputPin", Direction::Input).cloned() else { return V::Unknown("knot".into()) };
                self.pin_value(ci, n, &p, depth + 1)
            }
            "K2Node_Timeline" => {
                let tl = find(c.props(n), "TimelineName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let Some(info) = self.timelines.iter().find(|t| t.name == tl) else { return V::Unknown(format!("timeline {tl}")) };
                match info.curves.get(pin) {
                    Some(comps) if comps.len() == 1 => V::Track { timeline: tl, track: pin.to_string(), comp: 0, scale: 1.0, offset: 0.0 },
                    Some(comps) => V::St((0..comps.len()).map(|i| V::Track { timeline: tl.clone(), track: pin.to_string(), comp: i, scale: 1.0, offset: 0.0 }).collect()),
                    None => V::Unknown(format!("timeline output {pin}")),
                }
            }
            "K2Node_CallFunction" | "K2Node_CallArrayFunction" => {
                let f = c.member(n, "FunctionReference").unwrap_or_default();
                self.pure(ci, n, &f, depth)
            }
            "K2Node_CommutativeAssociativeBinaryOperator" => {
                let f = c.member(n, "FunctionReference").unwrap_or_default();
                self.pure(ci, n, &f, depth)
            }
            other => V::Unknown(format!("{other} output {pin}")),
        }
    }

    /// A pure function node's return value.
    fn pure(&mut self, ci: usize, n: usize, f: &str, _depth: usize) -> V {
        let base = f.split('_').next().unwrap_or(f);
        match f {
            "GetGameTimeInSeconds" | "GetTimeSeconds" | "GetRealTimeSeconds" | "GetAccumulatedTime" => return V::Time(0.0),
            "MakeVector" => return V::St(vec![self.input(ci, n, "X"), self.input(ci, n, "Y"), self.input(ci, n, "Z")]),
            "MakeRotator" => return V::St(vec![self.input(ci, n, "Pitch"), self.input(ci, n, "Yaw"), self.input(ci, n, "Roll")]),
            "MakeColor" | "MakeLinearColor" => return V::St(vec![self.input(ci, n, "R"), self.input(ci, n, "G"), self.input(ci, n, "B"), self.input(ci, n, "A")]),
            _ => {}
        }
        if f.starts_with("Conv_") || f.starts_with("Convert") {
            let first = self.classes[ci].pins(n).iter().find(|p| p.direction == Direction::Input && p.name != "self").cloned();
            return match first {
                Some(p) => self.pin_value(ci, n, &p, 1),
                None => V::Unknown(f.into()),
            };
        }
        let arith = matches!(base, "Add" | "Subtract" | "Multiply" | "Divide");
        if arith {
            let ins: Vec<Pin> = self.classes[ci].pins(n).iter().filter(|p| p.direction == Direction::Input && p.name != "self" && p.parent_pin.is_none()).cloned().collect();
            let vals: Vec<V> = ins.iter().map(|p| self.pin_value(ci, n, p, 1)).collect();
            let mut it = vals.into_iter();
            let Some(mut acc) = it.next() else { return V::Unknown(f.into()) };
            for v in it {
                acc = arith_op(base, acc, v);
            }
            return acc;
        }
        V::Unknown(format!("function {f}"))
    }

    // ---- exec ----

    fn skip(&mut self, s: String) {
        if !self.skipped.contains(&s) {
            self.skipped.push(s);
        }
    }

    fn here(&self, ci: usize, n: usize) -> String {
        let c = &self.classes[ci];
        let mut s = format!("{}:{}", c.graph_name(n), c.node_name(n));
        if ci > 0 {
            s = format!("{} {s}", c.name);
        }
        s
    }

    /// Runs from node `n` (entered through exec input `entered`), then the
    /// `Delay`ed continuations in time order.
    fn run(&mut self, ci: usize, n: usize, entered: &str) {
        self.chain(ci, n, entered.to_string());
        while !self.pending.is_empty() {
            let i = (0..self.pending.len()).min_by(|&a, &b| self.pending[a].0.total_cmp(&self.pending[b].0)).unwrap();
            let (t, ci, n, pin) = self.pending.remove(i);
            if t > HORIZON {
                let here = self.here(ci, n);
                self.skip(format!("{here}: runs at {t:.2} s, after the {HORIZON} s followed"));
                continue;
            }
            self.t = t;
            self.chain(ci, n, pin);
        }
    }

    fn chain(&mut self, ci: usize, start: usize, entered: String) {
        let mut cur = Some((start, entered));
        while let Some((n, pin)) = cur {
            if !self.visited.insert((ci, n)) {
                let here = self.here(ci, n);
                self.skip(format!("{here}: reached again at {:.2} s (a loop): not followed further", self.t));
                return;
            }
            cur = self.step(ci, n, &pin);
        }
    }

    fn next(&self, ci: usize, n: usize, out: &str) -> Option<(usize, String)> {
        self.classes[ci].follow_out(n, out)
    }

    fn step(&mut self, ci: usize, n: usize, entered: &str) -> Option<(usize, String)> {
        let class = self.classes[ci].class(n);
        if self.classes[ci].disabled(n) {
            return None;
        }
        match class.as_str() {
            "K2Node_Event" | "K2Node_FunctionEntry" | "K2Node_CustomEvent" => self.next(ci, n, "then"),
            "K2Node_Knot" => self.next(ci, n, "OutputPin"),
            "K2Node_IfThenElse" => match self.input(ci, n, "Condition").truth() {
                Some(true) => self.next(ci, n, "then"),
                Some(false) => self.next(ci, n, "else"),
                None => {
                    let v = self.input(ci, n, "Condition").describe();
                    let here = self.here(ci, n);
                    self.skip(format!("{here}: branch on {v}, not known at import: not followed"));
                    None
                }
            },
            "K2Node_ExecutionSequence" => {
                let outs: Vec<String> = self.classes[ci].pins(n).iter().filter(|p| p.direction == Direction::Output).map(|p| p.name.clone()).collect();
                for o in outs {
                    if let Some((m, p)) = self.next(ci, n, &o) {
                        self.chain(ci, m, p);
                    }
                }
                None
            }
            "K2Node_VariableSet" => {
                let name = self.classes[ci].member(n, "VariableReference").unwrap_or_default();
                let v = self.input(ci, n, &name);
                self.done.push(format!("{:.2} s: {name} = {}", self.t, v.describe()));
                self.vars.insert(name, v);
                self.next(ci, n, "then")
            }
            "K2Node_Timeline" => {
                let tl = find(self.classes[ci].props(n), "TimelineName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let act = match entered {
                    "Play" => Some(Act::Play(false)),
                    "PlayFromStart" => Some(Act::Play(true)),
                    "Stop" => Some(Act::Stop),
                    _ => None,
                };
                match act {
                    Some(a) => self.act(&tl, a),
                    None => {
                        let here = self.here(ci, n);
                        self.skip(format!("{here}: timeline {entered}: not imported"));
                    }
                }
                None
            }
            "K2Node_CallFunction" | "K2Node_CallParentFunction" | "K2Node_CallFunctionOnMember" | "K2Node_CallMaterialParameterCollectionFunction" => {
                let f = self.classes[ci].member(n, "FunctionReference").unwrap_or_default();
                if class == "K2Node_CallParentFunction" {
                    self.call_parent(ci, &f);
                    return self.next(ci, n, "then");
                }
                if f == "Delay" || f == "RetriggerableDelay" {
                    let d = self.input(ci, n, "Duration").num();
                    let Some(d) = d else {
                        let here = self.here(ci, n);
                        self.skip(format!("{here}: Delay of an unknown duration: not followed"));
                        return None;
                    };
                    if self.construct {
                        let here = self.here(ci, n);
                        self.skip(format!("{here}: Delay in the construction script: not followed"));
                        return None;
                    }
                    if let Some((m, p)) = self.next(ci, n, "then") {
                        self.pending.push((self.t + d, ci, m, p));
                    }
                    return None;
                }
                self.call(ci, n, &f);
                self.next(ci, n, "then")
            }
            other => {
                let here = self.here(ci, n);
                self.skip(format!("{here}: {other}: not imported (stops here)"));
                None
            }
        }
    }

    /// The parent class's version of function or event `f`.
    fn call_parent(&mut self, ci: usize, f: &str) {
        for pi in ci + 1..self.classes.len() {
            if let Some((n, pin)) = self.entry(pi, f) {
                self.chain(pi, n, pin);
                return;
            }
        }
    }

    /// Where event or function `f` starts in class `ci` (enabled only).
    fn entry(&self, ci: usize, f: &str) -> Option<(usize, String)> {
        let c = &self.classes[ci];
        for n in c.nodes() {
            if c.disabled(n) {
                continue;
            }
            let hit = match c.class(n).as_str() {
                "K2Node_Event" => c.member(n, "EventReference").as_deref() == Some(f),
                "K2Node_CustomEvent" => find(c.props(n), "CustomFunctionName").and_then(|v| v.as_str()) == Some(f),
                "K2Node_FunctionEntry" => c.member(n, "FunctionReference").as_deref() == Some(f) || c.graph_name(n) == f,
                _ => false,
            };
            if hit {
                return Some((n, String::new()));
            }
        }
        None
    }

    fn act(&mut self, node: &str, a: Act) {
        let what = match &a {
            Act::Mat { key, value } => format!(
                "{} = {}",
                key.strip_prefix(CPD).map(|i| format!("custom primitive data {i}")).unwrap_or_else(|| format!("material {key}")),
                value.iter().map(|v| match v {
                    Num::Value(x) => format!("{x}"),
                    Num::GameTime { plus } if *plus == 0.0 => "game time".to_string(),
                    Num::GameTime { plus } => format!("game time + {plus}"),
                }).collect::<Vec<_>>().join(", ")
            ),
            Act::Param(n, v) => format!("{n} = {v:?}"),
            Act::Visible(v) => format!("visible {v}"),
            Act::Activate(r) => if *r { "activate (reset)".into() } else { "activate".into() },
            Act::Deactivate => "deactivate".into(),
            Act::Play(s) => if *s { "play from start".into() } else { "play".into() },
            Act::Stop => "stop".into(),
            Act::Rate(r) => format!("play rate {r}"),
        };
        self.done.push(format!("{:.2} s: {node}: {what}", self.t));
        self.actions.push((self.t, node.to_string(), a));
    }

    /// An impure function call on a component (or the actor).
    fn call(&mut self, ci: usize, n: usize, f: &str) {
        let here = self.here(ci, n);
        if IGNORED.contains(&f) {
            *self.ignored.entry(f.to_string()).or_default() += 1;
            return;
        }
        // The actor's own events and functions.
        let target = match self.classes[ci].pin(n, "self", Direction::Input).cloned() {
            Some(p) if !p.linked_to.is_empty() => self.pin_value(ci, n, &p, 0),
            _ => V::Actor,
        };
        if target == V::Actor {
            for k in 0..self.classes.len() {
                if let Some((m, p)) = self.entry(k, f) {
                    self.chain(k, m, p);
                    return;
                }
            }
        }
        let t = self.t;
        let bake = self.construct || t == 0.0;
        let comp = match &target {
            V::Comp(c) => Some(c.clone()),
            V::Mid(c, _) => Some(c.clone()),
            _ => None,
        };
        let get = |s: &mut Self, p: &str| s.input(ci, n, p);
        let statik = |s: &mut Self, what: String| {
            if !bake {
                s.skip(format!("{here}: {what} at {t:.2} s: only spawn-time changes are imported"));
                return false;
            }
            s.done.push(format!("{t:.2} s: {what}"));
            true
        };
        match (f, &target) {
            ("SetStaticMesh", V::Comp(c)) => {
                if let V::Obj(m) = get(self, "NewMesh") {
                    if statik(self, format!("{c}: mesh {}", split_object_path(&m).1)) {
                        self.comp_mut(c).map(|x| x.mesh = Some(m).filter(|m| !m.is_empty()));
                    }
                }
            }
            ("SetMaterial", V::Comp(c)) => {
                let i = get(self, "ElementIndex").num().unwrap_or(0.0) as usize;
                if let V::Obj(m) = get(self, "Material") {
                    if statik(self, format!("{c}: material {i} = {}", split_object_path(&m).1)) {
                        if let Some(x) = self.comp_mut(c) {
                            if x.materials.len() <= i {
                                x.materials.resize(i + 1, None);
                            }
                            x.materials[i] = Some(m).filter(|m| !m.is_empty());
                        }
                    }
                }
            }
            ("K2_SetRelativeLocation" | "K2_SetWorldLocation" | "K2_SetRelativeRotation" | "K2_SetWorldRotation" | "K2_SetRelativeLocationAndRotation" | "K2_SetWorldLocationAndRotation" | "SetRelativeScale3D" | "SetWorldScale3D", V::Comp(c)) => {
                let loc = f.contains("Location").then(|| get(self, "NewLocation"));
                let rot = f.contains("Rotation").then(|| get(self, "NewRotation"));
                let scale = f.contains("Scale").then(|| get(self, "NewScale3D"));
                if f.contains("World") && !self.world_is_local(c) {
                    self.skip(format!("{here}: {f} on {c}: world value taken as relative to its parent"));
                }
                let (l, r, s) = (loc.as_ref().map(|v| v.nums::<3>()), rot.as_ref().map(|v| v.nums::<3>()), scale.as_ref().map(|v| v.nums::<3>()));
                if [&l, &r, &s].iter().any(|x| matches!(x, Some(None))) {
                    self.skip(format!("{here}: {f} on {c} with a value not known at import"));
                } else if statik(self, format!("{c}: {f}")) {
                    if let Some(x) = self.comp_mut(c) {
                        if let Some(Some(v)) = l {
                            x.loc = v;
                        }
                        if let Some(Some(v)) = r {
                            x.rot = v;
                        }
                        if let Some(Some(v)) = s {
                            x.scale = v;
                        }
                    }
                }
            }
            ("SetVisibility" | "SetHiddenInGame", V::Comp(c)) => {
                let v = if f == "SetVisibility" { get(self, "bNewVisibility").truth() } else { get(self, "NewHidden").truth().map(|h| !h) };
                match v {
                    Some(v) if self.construct => {
                        self.done.push(format!("{c}: visible {v}"));
                        self.comp_mut(c).map(|x| x.visible = v);
                    }
                    Some(v) => self.act(c, Act::Visible(v)),
                    None => self.skip(format!("{here}: {f} on {c} with an unknown value")),
                }
            }
            ("SetAsset", V::Comp(c)) => {
                if let V::Obj(m) = get(self, "InAsset") {
                    if statik(self, format!("{c}: system {}", split_object_path(&m).1)) {
                        self.comp_mut(c).map(|x| x.asset = Some(m).filter(|m| !m.is_empty()));
                    }
                }
            }
            ("Activate" | "ActivateSystem" | "ResetSystem" | "ReinitializeSystem", V::Comp(c)) => {
                let reset = f != "Activate" || get(self, "bReset").truth().unwrap_or(false);
                if self.construct {
                    self.comp_mut(c).map(|x| x.auto_activate = true);
                } else {
                    self.act(c, Act::Activate(reset));
                }
            }
            ("Deactivate" | "DeactivateImmediate" | "DeactivateSystem", V::Comp(c)) => {
                if self.construct {
                    self.comp_mut(c).map(|x| x.auto_activate = false);
                } else {
                    self.act(c, Act::Deactivate);
                }
            }
            ("SetLightColor", V::Comp(c)) => {
                if let Some(v) = get(self, "NewLightColor").nums::<4>() {
                    if statik(self, format!("{c}: light colour {v:?}")) {
                        self.comp_mut(c).map(|x| x.light.0 = [v[0], v[1], v[2]]);
                    }
                }
            }
            ("SetIntensity", V::Comp(c)) => {
                if let Some(v) = get(self, "NewIntensity").num() {
                    if statik(self, format!("{c}: light intensity {v}")) {
                        self.comp_mut(c).map(|x| x.light.1 = v);
                    }
                }
            }
            ("SetAttenuationRadius", V::Comp(c)) => {
                if let Some(v) = get(self, "NewRadius").num() {
                    if statik(self, format!("{c}: light radius {v}")) {
                        self.comp_mut(c).map(|x| x.light.2 = v);
                    }
                }
            }
            ("SetCustomPrimitiveDataFloat" | "SetCustomPrimitiveDataVector2" | "SetCustomPrimitiveDataVector3" | "SetCustomPrimitiveDataVector4", V::Comp(c)) => {
                let i = get(self, "DataIndex").num().unwrap_or(0.0) as u32;
                let v = get(self, "Value");
                let vals: Vec<V> = match v {
                    V::St(c) => c,
                    v => vec![v],
                };
                for (k, x) in vals.iter().enumerate() {
                    self.set_material(c, format!("{CPD}{}", i + k as u32), vec![x.clone()], &here);
                }
            }
            ("SetScalarParameterValueOnMaterials" | "SetScalarParameterValue", V::Comp(c) | V::Mid(c, _)) => {
                let name = match get(self, "ParameterName") {
                    V::Str(s) => s,
                    v => v.describe(),
                };
                let v = get(self, if f.ends_with("OnMaterials") { "ParameterValue" } else { "Value" });
                self.set_material(c, name, vec![v], &here);
            }
            ("SetVectorParameterValueOnMaterials" | "SetVectorParameterValue", V::Comp(c) | V::Mid(c, _)) => {
                let name = match get(self, "ParameterName") {
                    V::Str(s) => s,
                    v => v.describe(),
                };
                let v = get(self, if f.ends_with("OnMaterials") { "ParameterValue" } else { "Value" });
                let mut vals = match v {
                    V::St(c) => c,
                    v => vec![v],
                };
                vals.resize(4, V::Num(1.0));
                self.set_material(c, name, vals, &here);
            }
            ("CreateDynamicMaterialInstance" | "CreateAndSetMaterialInstanceDynamic", V::Comp(c)) => {
                let i = get(self, "ElementIndex").num().unwrap_or(0.0) as usize;
                self.memo.insert((ci, n, "ReturnValue".into()), V::Mid(c.clone(), i));
            }
            (f, V::Comp(c)) if f.starts_with("SetNiagaraVariable") || f.starts_with("SetVariable") || f.starts_with("SetFloatParameter") || f.starts_with("SetVectorParameter") || f.starts_with("SetColorParameter") => {
                let name = match get(self, "InVariableName") {
                    V::Str(s) => s,
                    _ => match get(self, "ParameterName") {
                        V::Str(s) => s,
                        v => v.describe(),
                    },
                };
                let name = if name.starts_with("User.") { name } else { format!("User.{name}") };
                let v = ["InValue", "Param", "Value", "InVector", "InColor"].iter().map(|p| get(self, p)).find(|v| !matches!(v, V::Unknown(_))).unwrap_or(V::Unknown("value".into()));
                let pv = match (&v, f) {
                    (V::Bool(b), _) => Some(ParamValue::Bool(*b)),
                    (V::Num(x), f) if f.contains("Int") => Some(ParamValue::Int(*x as i32)),
                    (V::Num(x), _) => Some(ParamValue::Float(*x as f32)),
                    (V::St(_), f) if f.contains("Color") => v.nums::<4>().map(|c| ParamValue::Color(c.map(|x| x as f32))),
                    (V::St(_), _) => v.nums::<3>().map(|c| ParamValue::Vec3(swap3(c, f.contains("Position") || f.contains("Vec3")))),
                    _ => None,
                };
                match pv {
                    Some(pv) if self.construct => {
                        self.done.push(format!("{c}: {name} = {pv:?}"));
                        self.comp_mut(c).map(|x| x.params.push((name, pv)));
                    }
                    Some(pv) => self.act(c, Act::Param(name, pv)),
                    None => self.skip(format!("{here}: {f} {name} = {}: not imported", v.describe())),
                }
            }
            ("SetPlayRate", V::Timeline(tl)) => match get(self, "NewRate").num() {
                Some(r) => self.act(tl, Act::Rate(r as f32)),
                None => self.skip(format!("{here}: SetPlayRate of {tl}: unknown rate")),
            },
            ("Play" | "PlayFromStart", V::Timeline(tl)) => self.act(tl, Act::Play(f == "PlayFromStart")),
            ("Stop", V::Timeline(tl)) => self.act(tl, Act::Stop),
            _ => {
                let _ = comp;
                self.skip(format!("{here}: {f} on {}: not imported", target.describe()));
            }
        }
    }

    /// A material parameter (or custom primitive data index) set to `vals`.
    fn set_material(&mut self, c: &str, key: String, vals: Vec<V>, here: &str) {
        let shown = key.strip_prefix(CPD).map(|i| format!("custom primitive data {i}")).unwrap_or_else(|| key.clone());
        if self.construct {
            let nums: Option<Vec<f64>> = vals.iter().map(|v| v.num()).collect();
            let Some(nums) = nums else {
                self.skip(format!("{here}: {c}: {shown} = {} in the construction script: not imported", vals.iter().map(|v| v.describe()).collect::<Vec<_>>().join(", ")));
                return;
            };
            self.done.push(format!("{c}: {shown} = {nums:?}"));
            let Some(x) = self.comp_mut(c) else { return };
            if let Some(i) = key.strip_prefix(CPD).and_then(|i| i.parse::<u32>().ok()) {
                x.cpd.insert(i, nums[0]);
            } else if nums.len() == 1 {
                x.mat_scalars.insert(key, nums[0]);
            } else {
                x.mat_vectors.insert(key, [nums[0], nums[1], nums[2], nums[3]]);
            }
            return;
        }
        let nums: Option<Vec<Num>> = vals.iter().map(|v| v.to_num()).collect();
        match nums {
            Some(value) => self.act(c, Act::Mat { key, value }),
            None => self.skip(format!("{here}: {c}: {shown} = {}: not imported", vals.iter().map(|v| v.describe()).collect::<Vec<_>>().join(", "))),
        }
    }

    // ---- timelines ----

    fn read_timelines(&mut self) {
        for ci in 0..self.classes.len() {
            let c = &self.classes[ci];
            for t in c.objs(find(c.bp_props(), "Timelines")) {
                let p = c.props(t);
                let name = find(p, "VariableName").and_then(|v| v.as_str()).map(String::from).unwrap_or_else(|| c.l.pkg.exports[t].object_name.trim_end_matches("_Template").to_string());
                if self.timelines.iter().any(|x| x.name == name) {
                    continue;
                }
                let mut info = TimelineInfo {
                    name,
                    length: find(p, "TimelineLength").and_then(|v| v.as_f64()).unwrap_or(5.0),
                    looping: matches!(find(p, "bLoop"), Some(Value::Bool(true))),
                    autoplay: matches!(find(p, "bAutoPlay"), Some(Value::Bool(true))),
                    rate: 1.0,
                    curves: HashMap::new(),
                    notes: Vec::new(),
                };
                for (list, field, n) in [("FloatTracks", "CurveFloat", 1), ("VectorTracks", "CurveVector", 3), ("LinearColorTracks", "CurveLinearColor", 4)] {
                    let Some(Value::Array(tracks)) = find(p, list) else { continue };
                    for tr in tracks {
                        let track = tr.field("TrackName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let curve = match tr.field(field) {
                            Some(Value::Object(o)) if !o.is_null() => Some(*o),
                            _ => None,
                        };
                        match curve.and_then(|o| read_curve(&mut self.store, &c.l, o, n)) {
                            Some(keys) => {
                                info.curves.insert(track, keys);
                            }
                            None => info.notes.push(format!("track {track}: curve not found")),
                        }
                    }
                }
                if let Some(Value::Array(ev)) = find(p, "EventTracks") {
                    if !ev.is_empty() {
                        info.notes.push(format!("{} event track(s): not imported", ev.len()));
                    }
                }
                self.timelines.push(info);
            }
        }
    }

    /// Follows each Timeline node's Update output and turns the calls fed by
    /// its tracks into prefab tracks.
    fn bind_timelines(&mut self) {
        for ci in 0..self.classes.len() {
            let nodes: Vec<usize> = self.classes[ci].nodes().into_iter().filter(|&n| self.classes[ci].class(n) == "K2Node_Timeline").collect();
            for n in nodes {
                let tl = find(self.classes[ci].props(n), "TimelineName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let mut cur = self.next(ci, n, "Update");
                let mut seen = HashSet::new();
                while let Some((m, _)) = cur {
                    if !seen.insert(m) {
                        break;
                    }
                    let class = self.classes[ci].class(m);
                    if class == "K2Node_Knot" {
                        cur = self.next(ci, m, "OutputPin");
                        continue;
                    }
                    if class != "K2Node_CallFunction" {
                        let here = self.here(ci, m);
                        self.skip(format!("{here}: {class} in {tl}'s Update: not imported (stops here)"));
                        break;
                    }
                    let f = self.classes[ci].member(m, "FunctionReference").unwrap_or_default();
                    self.track_call(ci, m, &f, &tl);
                    cur = self.next(ci, m, "then");
                }
                if let Some((m, _)) = self.next(ci, n, "Finished") {
                    let here = self.here(ci, m);
                    self.skip(format!("{here}: {tl}'s Finished output: not imported"));
                }
            }
        }
    }

    fn channel(&self, v: &V) -> Option<Channel> {
        match v {
            V::Num(x) => Some(Channel::Const(*x as f32)),
            V::Bool(b) => Some(Channel::Const(*b as i32 as f32)),
            V::Track { timeline, track, comp, scale, offset } => {
                let keys = self.timelines.iter().find(|t| &t.name == timeline)?.curves.get(track)?.get(*comp)?.clone();
                Some(Channel::Curve { keys, scale: *scale as f32, offset: *offset as f32 })
            }
            _ => None,
        }
    }

    fn track_call(&mut self, ci: usize, n: usize, f: &str, tl: &str) {
        let here = self.here(ci, n);
        let target = match self.classes[ci].pin(n, "self", Direction::Input).cloned() {
            Some(p) if !p.linked_to.is_empty() => self.pin_value(ci, n, &p, 0),
            _ => V::Actor,
        };
        let comp = match &target {
            V::Comp(c) | V::Mid(c, _) => c.clone(),
            _ => {
                self.skip(format!("{here}: {f} on {} in {tl}'s Update: not imported", target.describe()));
                return;
            }
        };
        let mut props: Vec<(Property, V)> = Vec::new();
        let get = |s: &mut Self, p: &str| s.input(ci, n, p);
        match f {
            _ if f.contains("Location") && f.starts_with("K2_Set") => props.push((Property::Location, get(self, "NewLocation"))),
            _ => {}
        }
        if f.starts_with("K2_Set") && f.contains("Rotation") {
            props.push((Property::Rotation, get(self, "NewRotation")));
        }
        match f {
            "SetRelativeScale3D" | "SetWorldScale3D" => props.push((Property::Scale, get(self, "NewScale3D"))),
            "SetScalarParameterValueOnMaterials" | "SetScalarParameterValue" | "SetVectorParameterValueOnMaterials" | "SetVectorParameterValue" => {
                let name = match get(self, "ParameterName") {
                    V::Str(s) => s,
                    v => v.describe(),
                };
                let v = get(self, if f.ends_with("OnMaterials") { "ParameterValue" } else { "Value" });
                props.push((Property::Material(name), v));
            }
            f if f.starts_with("SetCustomPrimitiveData") => {
                let i = get(self, "DataIndex").num().unwrap_or(0.0) as u32;
                props.push((Property::Material(format!("{CPD}{i}")), get(self, "Value")));
            }
            f if f.starts_with("SetNiagaraVariable") || f.starts_with("SetVariable") => {
                let name = match get(self, "InVariableName") {
                    V::Str(s) => s,
                    v => v.describe(),
                };
                let name = if name.starts_with("User.") { name } else { format!("User.{name}") };
                props.push((Property::Param(name), get(self, "InValue")));
            }
            _ => {}
        }
        if props.is_empty() {
            self.skip(format!("{here}: {f} in {tl}'s Update: not imported"));
            return;
        }
        if f.contains("World") && !self.world_is_local(&comp) {
            self.skip(format!("{here}: {f} on {comp}: world value taken as relative to its parent"));
        }
        for (prop, v) in props {
            let parts = match &v {
                V::St(c) => c.clone(),
                v => vec![v.clone()],
            };
            let channels: Option<Vec<Channel>> = parts.iter().map(|p| self.channel(p)).collect();
            let Some(mut channels) = channels else {
                self.skip(format!("{here}: {f} on {comp} = {}: not imported", v.describe()));
                continue;
            };
            if !parts.iter().any(|p| matches!(p, V::Track { .. })) {
                // Constant in the Update: a value set every frame; keep it as a track.
            }
            if matches!(prop, Property::Material(_)) && channels.len() == 3 {
                channels.push(Channel::Const(1.0));
            }
            self.done.push(format!("timeline {tl}: {comp} {prop:?} <- {}", v.describe()));
            self.tracks.push((tl.to_string(), Track { node: comp.clone(), property: prop, channels }));
        }
    }
}

fn swap3(c: [f64; 3], position: bool) -> [f32; 3] {
    let s = if position { 0.01 } else { 1.0 };
    [(c[0] * s) as f32, (c[2] * s) as f32, (c[1] * s) as f32]
}

fn arith_op(op: &str, a: V, b: V) -> V {
    let f = |x: f64, y: f64| match op {
        "Add" => x + y,
        "Subtract" => x - y,
        "Multiply" => x * y,
        _ => {
            if y == 0.0 {
                0.0
            } else {
                x / y
            }
        }
    };
    match (a, b) {
        (V::Num(x), V::Num(y)) => V::Num(f(x, y)),
        (V::Time(p), V::Num(y)) if op == "Add" => V::Time(p + y),
        (V::Time(p), V::Num(y)) if op == "Subtract" => V::Time(p - y),
        (V::Num(y), V::Time(p)) if op == "Add" => V::Time(p + y),
        (V::Track { timeline, track, comp, scale, offset }, V::Num(y)) | (V::Num(y), V::Track { timeline, track, comp, scale, offset }) if op == "Multiply" => {
            V::Track { timeline, track, comp, scale: scale * y, offset: offset * y }
        }
        (V::Track { timeline, track, comp, scale, offset }, V::Num(y)) if op == "Divide" && y != 0.0 => V::Track { timeline, track, comp, scale: scale / y, offset: offset / y },
        (V::Track { timeline, track, comp, scale, offset }, V::Num(y)) | (V::Num(y), V::Track { timeline, track, comp, scale, offset }) if op == "Add" => {
            V::Track { timeline, track, comp, scale, offset: offset + y }
        }
        (V::Track { timeline, track, comp, scale, offset }, V::Num(y)) if op == "Subtract" => V::Track { timeline, track, comp, scale, offset: offset - y },
        (V::St(a), V::St(b)) if a.len() == b.len() => V::St(a.into_iter().zip(b).map(|(x, y)| arith_op(op, x, y)).collect()),
        (V::St(a), y @ V::Num(_)) => V::St(a.into_iter().map(|x| arith_op(op, x, y.clone())).collect()),
        (a, b) => V::Unknown(format!("{op}({}, {})", a.describe(), b.describe())),
    }
}

/// A curve asset's keys, one list per component (`n` of them).
fn read_curve(store: &mut Store, l: &Rc<Loaded>, o: crate::package::PackageIndex, n: usize) -> Option<Vec<Vec<Key>>> {
    let (ll, i) = match o.export() {
        Some(i) => (l.clone(), i),
        None => {
            let (pkg, name) = split_object_path(&l.path(o));
            let other = store.load(&format!("{pkg}.{name}"))?;
            let i = other.export(&name)?;
            (other, i)
        }
    };
    let props = ll.props(i);
    let keys = |c: &Value| -> Vec<Key> {
        let Some(Value::Array(ks)) = c.field("Keys") else { return Vec::new() };
        ks.iter()
            .map(|k| {
                let g = |f: &str| k.field(f).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let interp = match k.field("InterpMode").and_then(|v| v.as_f64().or_else(|| v.as_str().map(|s| if s.ends_with("Constant") { 1.0 } else if s.ends_with("Cubic") { 2.0 } else { 0.0 }))) {
                    Some(x) if x == 1.0 => KInterp::Constant,
                    Some(x) if x == 2.0 => KInterp::Cubic,
                    _ => KInterp::Linear,
                };
                Key { time: g("Time"), value: g("Value"), interp, arrive: g("ArriveTangent"), leave: g("LeaveTangent") }
            })
            .collect()
    };
    if n == 1 {
        let c = find(props, "FloatCurve")?;
        return Some(vec![keys(c)]);
    }
    let mut out = vec![Vec::new(); n];
    for p in props.iter().filter(|p| p.name == "FloatCurves") {
        if let Some(slot) = out.get_mut(p.array_index as usize) {
            *slot = keys(&p.value);
        }
    }
    Some(out)
}

// ---- entry ----

/// Reads Blueprint `file` (and its parents, from the package folders) into a
/// prefab, converting its Niagara systems (once per run, via `effects`)
/// and static meshes into effect files.
pub fn import(file: &Path, effects: &mut Effects) -> Result<Imported, String> {
    let mut store = Store::for_system(file);
    let first = make_class(load_package(file)?)?;
    let mut classes = vec![first];
    let mut chain_note = Vec::new();
    loop {
        let last = classes.last().unwrap();
        let parent = match find(last.bp_props(), "ParentClass") {
            Some(Value::Object(o)) if !o.is_null() => last.l.path(*o),
            _ => break,
        };
        if parent.starts_with("/Script/") {
            chain_note.push(format!("native parent `{}`", split_object_path(&parent).1));
            break;
        }
        let (pkg, cls) = split_object_path(&parent);
        let name = cls.strip_suffix("_C").unwrap_or(&cls).to_string();
        match store.load(&format!("{pkg}.{name}")).ok_or_else(|| format!("parent Blueprint `{pkg}` not in the package folders")).and_then(make_class) {
            Ok(c) => {
                chain_note.push(format!("parent `{}`", c.name));
                classes.push(c);
            }
            Err(e) => {
                chain_note.push(e);
                break;
            }
        }
        if classes.len() > 16 {
            break;
        }
    }
    let mut bp = Bp {
        store,
        classes,
        comps: Vec::new(),
        vars: HashMap::new(),
        memo: HashMap::new(),
        timelines: Vec::new(),
        construct: true,
        t: 0.0,
        visited: HashSet::new(),
        pending: Vec::new(),
        actions: Vec::new(),
        tracks: Vec::new(),
        done: Vec::new(),
        skipped: Vec::new(),
        ignored: BTreeMap::new(),
    };
    read_components(&mut bp);
    bp.read_timelines();

    // Construction script: the most derived one (it calls its parents').
    let mut ucs_ran = false;
    for ci in 0..bp.classes.len() {
        if let Some((n, p)) = bp.entry(ci, "UserConstructionScript") {
            bp.run(ci, n, &p);
            ucs_ran = true;
            break;
        }
    }
    let ucs_log: Vec<String> = std::mem::take(&mut bp.done);
    // BeginPlay.
    bp.construct = false;
    bp.visited.clear();
    bp.t = 0.0;
    let mut begin = false;
    for ci in 0..bp.classes.len() {
        if let Some((n, p)) = bp.entry(ci, "ReceiveBeginPlay") {
            bp.run(ci, n, &p);
            begin = true;
            break;
        }
    }
    let play_log: Vec<String> = std::mem::take(&mut bp.done);
    bp.bind_timelines();
    let tl_log: Vec<String> = std::mem::take(&mut bp.done);
    // Events never run here.
    let mut entries = Vec::new();
    for ci in 0..bp.classes.len() {
        let c = &bp.classes[ci];
        for n in c.nodes() {
            if c.disabled(n) {
                continue;
            }
            let what = match c.class(n).as_str() {
                "K2Node_Event" => c.member(n, "EventReference").filter(|e| e != "ReceiveBeginPlay"),
                "K2Node_CustomEvent" => find(c.props(n), "CustomFunctionName").and_then(|v| v.as_str()).map(|s| format!("custom event {s}")),
                "K2Node_InputKeyEvent" | "K2Node_InputAction" | "K2Node_InputAxisEvent" | "K2Node_InputTouch" => Some(format!("input event {}", c.node_name(n))),
                "K2Node_ComponentBoundEvent" | "K2Node_ActorBoundEvent" => Some(format!("bound event {}", c.node_name(n))),
                _ => None,
            };
            let Some(what) = what else { continue };
            // Only entries with something wired to them.
            let wired = c.pins(n).iter().any(|p| p.direction == Direction::Output && !p.linked_to.is_empty() && p.type_field("PinCategory").and_then(|v| v.as_str()) == Some("exec"));
            if wired && !bp.visited.contains(&(ci, n)) {
                entries.push(format!("{}{what} ({}): not imported", if ci > 0 { format!("{} ", c.name) } else { String::new() }, c.graph_name(n)));
            }
        }
    }

    // ---- the prefab ----
    let stem = file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| bp.classes[0].name.clone());
    let mut files = BTreeMap::new();
    let mut nodes = Vec::new();
    let mut comp_lines = Vec::new();
    let mut mesh_assets = AssetOut::new(Store::for_system(file), format!("{stem}_assets"));
    let mut mat_maps: HashMap<String, Vec<MaterialParam>> = HashMap::new();
    let comps = bp.comps.clone();
    for c in &comps {
        let transform = prefab::Transform {
            translation: prefab::ue_location(c.loc.map(|x| x as f32)),
            rotation: prefab::ue_rotator(c.rot[0] as f32, c.rot[1] as f32, c.rot[2] as f32),
            scale: prefab::ue_scale(c.scale.map(|x| x as f32)),
        };
        let mut parent = c.parent.clone();
        // Children of a spring arm hang at its end.
        if let Some(p) = &c.parent {
            if comps.iter().any(|x| &x.name == p && x.class == "SpringArmComponent") {
                parent = Some(format!("{p}.Socket"));
            }
        }
        let kind = match c.class.as_str() {
            "NiagaraComponent" => match &c.asset {
                Some(sys) => match convert_system(&mut bp.store, sys, effects, &mut files) {
                    Ok(effect) => {
                        comp_lines.push(format!("{}: Niagara `{}` -> `{effect}`{}", c.name, split_object_path(sys).1, if c.auto_activate { "" } else { " (not auto-activated)" }));
                        let mut params = niagara_overrides(&bp, c);
                        params.extend(c.params.iter().cloned());
                        NodeKind::Effect { effect, auto_activate: c.auto_activate, params, material_params: Vec::new() }
                    }
                    Err(e) => {
                        comp_lines.push(format!("{}: Niagara `{}`: **not converted** ({e})", c.name, split_object_path(sys).1));
                        NodeKind::Empty
                    }
                },
                None => {
                    comp_lines.push(format!("{}: Niagara component without a system (set elsewhere, e.g. by the level or logic not imported): empty node", c.name));
                    NodeKind::Empty
                }
            },
            "StaticMeshComponent" | "InstancedStaticMeshComponent" | "HierarchicalInstancedStaticMeshComponent" => match &c.mesh {
                Some(mesh) => {
                    let name = format!("{stem}_{}", c.name);
                    let material = c.materials.first().cloned().flatten();
                    let (tweak, mps, user) = material_setup(&mut mesh_assets, c, material.as_deref().or(None), mesh, &bp.actions, &bp.tracks);
                    let me = mesh_assets.mesh_effect(&name, mesh, material.as_deref(), &tweak, &user);
                    let effect = format!("{name}.fx.ron");
                    files.insert(effect.clone(), me.ron.into_bytes());
                    let mut line = format!("{}: static mesh -> `{effect}`: {}", c.name, me.notes.join("; "));
                    if c.materials.iter().filter(|m| m.is_some()).count() > 1 {
                        line.push_str("; only material slot 0 is drawn (mesh slots merge)");
                    }
                    if !mps.is_empty() {
                        let _ = write!(line, "; parameters set after spawn: {}", mps.iter().map(|m| m.name.clone()).collect::<Vec<_>>().join(", "));
                    }
                    comp_lines.push(line);
                    mat_maps.insert(c.name.clone(), mps.clone());
                    NodeKind::Effect { effect, auto_activate: true, params: Vec::new(), material_params: mps }
                }
                None => {
                    comp_lines.push(format!("{}: static mesh component without a mesh: empty node", c.name));
                    NodeKind::Empty
                }
            },
            "PointLightComponent" | "SpotLightComponent" | "RectLightComponent" => {
                let (col, cd, radius) = c.light;
                // Candelas (Unreal 5's default unit) to lumens.
                let lm = cd * 4.0 * std::f64::consts::PI;
                comp_lines.push(format!("{}: {} -> point light, colour {:?}, {} cd = {lm:.0} lm, range {} m", c.name, c.class, col.map(|x| x as f32), cd as f32, (radius * 0.01) as f32));
                NodeKind::PointLight { color: col.map(|x| x as f32), intensity: lm as f32, range: (radius * 0.01) as f32 }
            }
            "CameraComponent" | "CineCameraComponent" => {
                // Unreal's field of view is horizontal; ours vertical (16:9).
                let v = 2.0 * ((c.fov.to_radians() / 2.0).tan() * 9.0 / 16.0).atan();
                comp_lines.push(format!("{}: camera, {}° horizontal = {:.1}° vertical at 16:9", c.name, c.fov, v.to_degrees()));
                NodeKind::Camera { fov: v.to_degrees() as f32 }
            }
            "SpringArmComponent" => {
                comp_lines.push(format!("{}: spring arm, {} cm long (no collision or lag)", c.name, c.arm.0));
                NodeKind::Empty
            }
            "SceneComponent" => NodeKind::Empty,
            other => {
                comp_lines.push(format!("{}: {other}: not drawn (empty node)", c.name));
                NodeKind::Empty
            }
        };
        nodes.push(Node { name: c.name.clone(), parent, transform, visible: c.visible, kind });
        if c.class == "SpringArmComponent" {
            let end = [-c.arm.0 + c.arm.1[0], c.arm.1[1], c.arm.1[2]];
            nodes.push(Node {
                name: format!("{}.Socket", c.name),
                parent: Some(c.name.clone()),
                transform: prefab::Transform { translation: prefab::ue_location(end.map(|x| x as f32)), ..Default::default() },
                visible: true,
                kind: NodeKind::Empty,
            });
        }
    }
    for (k, v) in mesh_assets.files.iter() {
        files.insert(k.clone(), v.clone());
    }

    // Actions and tracks, with material keys resolved.
    let mut actions = Vec::new();
    for (t, node, a) in &bp.actions {
        let what = match a {
            Act::Mat { key, value } => match resolve_material(mat_maps.get(node), key) {
                Some((name, comp)) => {
                    let mps = &mat_maps[node];
                    let mp = mps.iter().find(|m| m.name == name).unwrap();
                    if comp == 0 && value.len() == mp.count as usize {
                        Do::SetMaterial { name, value: value.clone() }
                    } else if value.len() == 1 {
                        // One component of a vector parameter.
                        let mut full: Vec<Num> = mp.default.iter().take(mp.count as usize).map(|x| Num::Value(*x)).collect();
                        if let Some(s) = full.get_mut(comp) {
                            *s = value[0];
                        }
                        Do::SetMaterial { name, value: full }
                    } else {
                        Do::SetMaterial { name, value: value.clone() }
                    }
                }
                None => {
                    bp.skipped.push(format!("{t:.2} s: {node}: {} (no such material parameter on its effect)", key.replace(CPD, "custom primitive data ")));
                    continue;
                }
            },
            Act::Param(n, v) => Do::SetParam { name: n.clone(), value: *v },
            Act::Visible(v) => Do::SetVisible(*v),
            Act::Activate(r) => Do::Activate { reset: *r },
            Act::Deactivate => Do::Deactivate,
            Act::Play(s) => Do::PlayTimeline { from_start: *s },
            Act::Stop => Do::StopTimeline,
            Act::Rate(r) => Do::SetTimelineRate(*r),
        };
        actions.push(Action { time: *t as f32, node: node.clone(), what });
    }
    let mut timelines = Vec::new();
    for info in &bp.timelines {
        let mut tracks = Vec::new();
        for (tl, tr) in &bp.tracks {
            if tl != &info.name {
                continue;
            }
            let mut tr = tr.clone();
            if let Property::Material(key) = &tr.property {
                match resolve_material(mat_maps.get(&tr.node), key) {
                    Some((name, 0)) => tr.property = Property::Material(name),
                    _ => {
                        bp.skipped.push(format!("timeline {}: {} {}: no such material parameter on its effect", info.name, tr.node, key.replace(CPD, "custom primitive data ")));
                        continue;
                    }
                }
            }
            tracks.push(tr);
        }
        timelines.push(Timeline { name: info.name.clone(), length: info.length as f32, looping: info.looping, autoplay: info.autoplay, rate: info.rate as f32, tracks });
    }

    let prefab = Prefab { name: stem.clone(), source: file.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), nodes, timelines, actions };

    // ---- report ----
    let mut r = String::new();
    let _ = writeln!(r, "# {stem}: Blueprint import\n");
    let _ = writeln!(r, "Source: `{}`, saved by {}.", prefab.source, bp.classes[0].l.pkg.summary.saved_by);
    let _ = writeln!(r, "Class chain: `{}`{}.\n", bp.classes[0].name, chain_note.iter().map(|s| format!(", {s}")).collect::<String>());
    let _ = writeln!(r, "## Components\n");
    for l in &comp_lines {
        let _ = writeln!(r, "- {l}");
    }
    let left: Vec<String> = bp_left_out(&bp);
    for l in &left {
        let _ = writeln!(r, "- {l}");
    }
    let _ = writeln!(r, "\n## Construction script\n");
    if !ucs_ran || ucs_log.is_empty() {
        let _ = writeln!(r, "- none");
    }
    for l in &ucs_log {
        let _ = writeln!(r, "- {l}");
    }
    let _ = writeln!(r, "\n## BeginPlay (timed actions)\n");
    if !begin {
        let _ = writeln!(r, "- no BeginPlay");
    } else if play_log.is_empty() {
        let _ = writeln!(r, "- nothing imported");
    }
    for l in &play_log {
        let _ = writeln!(r, "- {l}");
    }
    let _ = writeln!(r, "\n## Timelines\n");
    if bp.timelines.is_empty() {
        let _ = writeln!(r, "- none");
    }
    for t in &bp.timelines {
        let _ = writeln!(r, "- {}: {} s{}{}, tracks {}{}", t.name, t.length, if t.looping { ", looping" } else { "" }, if t.autoplay { ", auto play" } else { "" }, {
            let mut k: Vec<_> = t.curves.keys().cloned().collect();
            k.sort();
            k.join(", ")
        }, t.notes.iter().map(|n| format!("; {n}")).collect::<String>());
    }
    for l in &tl_log {
        let _ = writeln!(r, "- {l}");
    }
    let _ = writeln!(r, "\n## Logic not imported\n");
    let skipped = bp.skipped.len() + entries.len();
    if skipped == 0 {
        let _ = writeln!(r, "- none");
    }
    for l in entries.iter().chain(bp.skipped.iter()) {
        let _ = writeln!(r, "- {l}");
    }
    if !bp.ignored.is_empty() {
        let _ = writeln!(r, "- calls with no visible effect, ignored: {}", bp.ignored.iter().map(|(k, v)| format!("{k} x{v}")).collect::<Vec<_>>().join(", "));
    }
    Ok(Imported { components: comp_lines.len(), prefab, files, report: r, skipped })
}

fn bp_left_out(bp: &Bp) -> Vec<String> {
    let mut out = Vec::new();
    for c in &bp.classes {
        let Some(scs) = c.obj(find(c.bp_props(), "SimpleConstructionScript")) else { continue };
        for n in c.objs(find(c.props(scs), "AllNodes")) {
            let p = c.props(n);
            let class = match find(p, "ComponentClass") {
                Some(Value::Object(o)) if !o.is_null() => split_object_path(&c.l.path(*o)).1,
                _ => continue,
            };
            if !scene_class(&class) {
                let name = find(p, "InternalVariableName").and_then(|v| v.as_str()).unwrap_or("?");
                out.push(format!("{name}: {class}: left out (gameplay, not VFX)"));
            }
        }
    }
    out
}

/// Material parameter name and component for key `key` (a name or a
/// custom primitive data index).
fn resolve_material(mps: Option<&Vec<MaterialParam>>, key: &str) -> Option<(String, usize)> {
    let mps = mps?;
    if let Some(i) = key.strip_prefix(CPD).and_then(|i| i.parse::<u32>().ok()) {
        for m in mps {
            if let Some(p) = m.primitive_data {
                if i >= p && i < p + m.count {
                    return Some((m.name.clone(), (i - p) as usize));
                }
            }
        }
        return None;
    }
    mps.iter().find(|m| m.name == key).map(|m| (m.name.clone(), 0))
}

/// The material tweak (baked values, bound parameters), the bound
/// parameters' layout and the `User.` Vec4 defaults of mesh component `c`.
fn material_setup(a: &mut AssetOut, c: &Comp, material: Option<&str>, mesh: &str, actions: &[(f64, String, Act)], tracks: &[(String, Track)]) -> (MatTweak, Vec<MaterialParam>, Vec<(String, [f32; 4])>) {
    let mut tweak = MatTweak::default();
    // The material: the override, else the mesh's own (read by mesh_effect).
    let material = material.map(String::from).or_else(|| {
        let l = a.store.load(mesh)?;
        let i = l.export(&split_object_path(mesh).1)?;
        match find(l.props(i), "StaticMaterials") {
            Some(Value::Array(s)) => s.iter().find_map(|m| match m.field("MaterialInterface") {
                Some(Value::Object(o)) if !o.is_null() => Some(l.path(*o)),
                _ => None,
            }),
            _ => None,
        }
    });
    let Some(material) = material else { return (tweak, Vec::new(), Vec::new()) };
    let Ok(params) = a.material_parameters(&material) else { return (tweak, Vec::new(), Vec::new()) };
    // Baked: custom primitive data and parameter values from the construction script.
    for p in &params {
        if let Some(i) = p.primitive_data {
            if p.vector {
                if (0..4).any(|k| c.cpd.contains_key(&(i + k))) {
                    let mut v = p.value;
                    for k in 0..4 {
                        if let Some(x) = c.cpd.get(&(i + k)) {
                            v[k as usize] = *x as f32;
                        }
                    }
                    tweak.vectors.push((p.name.clone(), v));
                }
            } else if let Some(x) = c.cpd.get(&i) {
                tweak.scalars.push((p.name.clone(), *x as f32));
            }
        }
        if let Some(x) = c.mat_scalars.get(&p.name) {
            tweak.scalars.push((p.name.clone(), *x as f32));
        }
        if let Some(x) = c.mat_vectors.get(&p.name) {
            tweak.vectors.push((p.name.clone(), x.map(|v| v as f32)));
        }
    }
    // Animated: set by BeginPlay actions or timeline tracks.
    let mut wanted: Vec<&crate::material::ParamInfo> = Vec::new();
    let mut want = |key: &str| {
        let hit = if let Some(i) = key.strip_prefix(CPD).and_then(|i| i.parse::<u32>().ok()) {
            params.iter().find(|p| p.primitive_data.is_some_and(|d| i >= d && i < d + if p.vector { 4 } else { 1 }))
        } else {
            params.iter().find(|p| p.name == key)
        };
        if let Some(p) = hit {
            if !wanted.iter().any(|w| w.name == p.name) {
                wanted.push(p);
            }
        }
    };
    for (_, node, act) in actions {
        if let (true, Act::Mat { key, .. }) = (node == &c.name, act) {
            want(key);
        }
    }
    for (_, tr) in tracks {
        if let (true, Property::Material(key)) = (tr.node == c.name, &tr.property) {
            want(key);
        }
    }
    // Layout: vectors take a whole Vec4, scalars pack.
    let mut slots: Vec<[f32; 4]> = Vec::new();
    let mut used: Vec<usize> = Vec::new();
    let mut mps = Vec::new();
    let baked = |p: &crate::material::ParamInfo| -> [f32; 4] {
        if let Some(v) = tweak.vectors.iter().find(|(n, _)| n == &p.name) {
            return v.1;
        }
        if let Some(v) = tweak.scalars.iter().find(|(n, _)| n == &p.name) {
            return [v.1, 0.0, 0.0, 0.0];
        }
        p.value
    };
    for p in wanted.iter().filter(|p| p.vector).chain(wanted.iter().filter(|p| !p.vector)) {
        let (k, comp) = if p.vector {
            if slots.len() >= 4 {
                continue;
            }
            slots.push(baked(p));
            used.push(4);
            (slots.len() - 1, 0)
        } else {
            match used.iter().position(|u| *u < 4) {
                Some(k) => {
                    used[k] += 1;
                    (k, used[k] - 1)
                }
                None if slots.len() < 4 => {
                    slots.push([0.0; 4]);
                    used.push(1);
                    (slots.len() - 1, 0)
                }
                None => continue,
            }
        };
        let def = baked(p);
        if !p.vector {
            slots[k][comp] = def[0];
        }
        tweak.bound.push((p.name.clone(), k, comp));
        mps.push(MaterialParam {
            name: p.name.clone(),
            primitive_data: p.primitive_data,
            user: format!("User.MaterialParams{k}"),
            component: comp as u32,
            count: if p.vector { 4 } else { 1 },
            default: def,
        });
    }
    let user = slots.iter().enumerate().map(|(k, v)| (format!("User.MaterialParams{k}"), *v)).collect();
    (tweak, mps, user)
}

/// User parameter overrides of a Niagara component template.
fn niagara_overrides(bp: &Bp, c: &Comp) -> Vec<(String, ParamValue)> {
    let Some(store) = find(&c.props, "OverrideParameters") else { return Vec::new() };
    let l = &bp.classes[c.owner].l;
    crate::niagara::parameter_store(&l.pkg, store)
        .into_iter()
        .filter_map(|(n, v)| {
            use crate::niagara::NValue as N;
            let pv = match v {
                N::Float(x) => ParamValue::Float(x),
                N::Int(x) => ParamValue::Int(x),
                N::Bool(x) => ParamValue::Bool(x),
                N::Vec2(x) => ParamValue::Vec2(x),
                N::Vec3(x) => ParamValue::Vec3([x[0], x[2], x[1]]),
                N::Position(x) => ParamValue::Vec3([x[0] * 0.01, x[2] * 0.01, x[1] * 0.01]),
                N::Vec4(x) => ParamValue::Vec4(x),
                N::Color(x) => ParamValue::Color(x),
                _ => return None,
            };
            Some((n, pv))
        })
        .collect()
}

/// Converts Niagara system `path` once per run; its effect file name.
fn convert_system(store: &mut Store, path: &str, effects: &mut Effects, files: &mut BTreeMap<String, Vec<u8>>) -> Result<String, String> {
    if let Some(r) = effects.done.get(path) {
        return r.clone();
    }
    let res = (|| {
        let file = store.resolve(path).ok_or("its package is not in the package folders")?;
        let l = load_package(&file)?;
        let sys = crate::niagara::read_system(&l.pkg)?;
        let stem = file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| sys.name.clone());
        let assets = AssetOut::new(Store::for_system(&file), format!("{stem}_assets"));
        let conv = crate::convert::convert_with_assets(&sys, &file.file_name().unwrap_or_default().to_string_lossy(), Some(assets));
        let effect = format!("{stem}.fx.ron");
        files.insert(effect.clone(), conv.ron.into_bytes());
        files.insert(format!("{stem}.report.md"), conv.report.markdown().into_bytes());
        for (k, v) in conv.files {
            files.insert(k, v);
        }
        Ok(effect)
    })();
    effects.done.insert(path.to_string(), res.clone());
    res
}

/// Components of every class in the chain, parents' first, with the child
/// classes' inherited template overrides applied.
fn read_components(bp: &mut Bp) {
    let n = bp.classes.len();
    for ci in (0..n).rev() {
        let c = &bp.classes[ci];
        let Some(scs) = c.obj(find(c.bp_props(), "SimpleConstructionScript")) else { continue };
        let sp = c.props(scs).to_vec();
        let all = c.objs(find(&sp, "AllNodes"));
        let roots = c.objs(find(&sp, "RootNodes"));
        let default_root = c.obj(find(&sp, "DefaultSceneRootNode"));
        let mut parent_of: HashMap<usize, usize> = HashMap::new();
        for &node in &all {
            for ch in c.objs(find(c.props(node), "ChildNodes")) {
                parent_of.insert(ch, node);
            }
        }
        let name_of = |node: usize| find(c.props(node), "InternalVariableName").and_then(|v| v.as_str()).unwrap_or("").to_string();
        // A root scene component other than the default root replaces it.
        let has_root = roots.iter().any(|&r| {
            let class = match find(c.props(r), "ComponentClass") {
                Some(Value::Object(o)) => split_object_path(&c.l.path(*o)).1,
                _ => String::new(),
            };
            scene_class(&class) && Some(r) != default_root
        });
        let mut order: Vec<usize> = all.clone();
        if let Some(d) = default_root {
            if !order.contains(&d) && !has_root && ci == n - 1 {
                order.insert(0, d);
            }
        }
        let mut new = Vec::new();
        for node in order {
            let p = c.props(node);
            let class = match find(p, "ComponentClass") {
                Some(Value::Object(o)) if !o.is_null() => split_object_path(&c.l.path(*o)).1,
                _ => continue,
            };
            if !scene_class(&class) {
                continue;
            }
            let name = name_of(node);
            let parent = parent_of.get(&node).map(|&pn| name_of(pn)).or_else(|| {
                find(p, "ParentComponentOrVariableName").and_then(|v| v.as_str()).filter(|s| *s != "None" && !s.is_empty()).map(String::from)
            });
            let props = c.obj(find(p, "ComponentTemplate")).map(|t| c.props(t).to_vec()).unwrap_or_default();
            new.push(comp_from(name, class, parent, props, ci, &c.l));
        }
        bp.comps.extend(new);
    }
    // Inherited component template overrides (child classes win).
    for ci in (0..n).rev() {
        let c = &bp.classes[ci];
        let e = &c.l.pkg.exports;
        for h in (0..e.len()).filter(|&i| c.class(i) == "InheritableComponentHandler") {
            let Some(Value::Array(recs)) = find(c.props(h), "Records") else { continue };
            for rec in recs {
                let name = rec.field("ComponentKey").and_then(|k| k.field("SCSVariableName")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let Some(t) = c.obj(rec.field("ComponentTemplate")) else { continue };
                let over = c.props(t).to_vec();
                let l = c.l.clone();
                if let Some(x) = bp.comps.iter_mut().find(|x| x.name == name) {
                    let mut props = x.props.clone();
                    for p in over {
                        props.retain(|q| !(q.name == p.name && q.array_index == p.array_index));
                        props.push(p);
                    }
                    *x = comp_from(x.name.clone(), x.class.clone(), x.parent.clone(), props, ci, &l);
                }
            }
        }
    }
}

fn comp_from(name: String, class: String, parent: Option<String>, props: Vec<UProp>, owner: usize, l: &Rc<Loaded>) -> Comp {
    let v3 = |n: &str, d: [f64; 3], fields: [&str; 3]| -> [f64; 3] {
        match find(&props, n) {
            Some(v) => fields.map(|f| v.field(f).and_then(|x| x.as_f64()).unwrap_or(0.0)),
            None => d,
        }
    };
    let f = |n: &str, d: f64| find(&props, n).and_then(|v| v.as_f64()).unwrap_or(d);
    let b = |n: &str, d: bool| match find(&props, n) {
        Some(Value::Bool(x)) => *x,
        _ => d,
    };
    let obj = |n: &str| match find(&props, n) {
        Some(Value::Object(o)) if !o.is_null() => Some(l.path(*o)),
        _ => None,
    };
    let materials = match find(&props, "OverrideMaterials") {
        Some(Value::Array(a)) => a
            .iter()
            .map(|m| match m {
                Value::Object(o) if !o.is_null() => Some(l.path(*o)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let light = match find(&props, "LightColor") {
        Some(v) => match from_value(l, v) {
            V::St(c) => [c[0].num().unwrap_or(1.0), c[1].num().unwrap_or(1.0), c[2].num().unwrap_or(1.0)],
            _ => [1.0; 3],
        },
        None => [1.0; 3],
    };
    let mut cpd = BTreeMap::new();
    if let Some(Value::Array(d)) = find(&props, "CustomPrimitiveData").and_then(|v| v.field("Data")) {
        for (i, x) in d.iter().enumerate() {
            if let Some(x) = x.as_f64() {
                cpd.insert(i as u32, x);
            }
        }
    }
    Comp {
        loc: v3("RelativeLocation", [0.0; 3], ["X", "Y", "Z"]),
        rot: v3("RelativeRotation", [0.0; 3], ["Pitch", "Yaw", "Roll"]),
        scale: v3("RelativeScale3D", [1.0; 3], ["X", "Y", "Z"]),
        visible: b("bVisible", true) && !b("bHiddenInGame", false),
        auto_activate: b("bAutoActivate", true),
        asset: obj("Asset"),
        mesh: obj("StaticMesh"),
        materials,
        cpd,
        mat_scalars: BTreeMap::new(),
        mat_vectors: BTreeMap::new(),
        params: Vec::new(),
        light: (light, f("Intensity", 8.0), f("AttenuationRadius", 1000.0)),
        fov: f("FieldOfView", 90.0),
        arm: (f("TargetArmLength", 300.0), v3("SocketOffset", [0.0; 3], ["X", "Y", "Z"])),
        name,
        class,
        parent,
        props,
        owner,
    }
}

/// `fximport blueprint FILE --out DIR`: writes `DIR/<name>.prefab.ron`,
/// `DIR/<name>.report.md` and the effects and assets it uses. Returns the
/// prefab path and a one-line summary.
pub fn import_file(file: &Path, out: &Path, effects: &mut Effects) -> Result<(PathBuf, String), String> {
    let im = import(file, effects)?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let name = &im.prefab.name;
    let path = out.join(format!("{name}.{}", prefab::EXTENSION));
    let mut text = format!("// Imported from the Unreal Blueprint {} by `fximport blueprint`; the report beside it lists what was left out.\n", im.prefab.source);
    text.push_str(&prefab::to_string(&im.prefab));
    text.push('\n');
    std::fs::write(&path, text).map_err(|e| e.to_string())?;
    std::fs::write(out.join(format!("{name}.report.md")), &im.report).map_err(|e| e.to_string())?;
    for (rel, bytes) in &im.files {
        let p = out.join(rel);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    let summary = format!(
        "{}: {} nodes, {} timelines, {} actions; {} logic entries not imported; report {}",
        path.display(),
        im.prefab.nodes.len(),
        im.prefab.timelines.len(),
        im.prefab.actions.len(),
        im.skipped,
        out.join(format!("{name}.report.md")).display()
    );
    Ok((path, summary))
}
