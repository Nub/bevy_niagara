//! Custom modules written in WGSL and simulation stages over particles and
//! grids: Niagara's scratch pad / custom HLSL modules and simulation stages.
//!
//! A custom module is declared once in the effect (`custom_modules: [...]`)
//! with typed inputs (and their defaults), the particle attributes it writes
//! (`outputs`) and a WGSL function body. Placed in a stack, it is called like
//! a built-in module. The body sees:
//!
//! - its inputs, by name (`let`);
//! - in particle stacks: every particle attribute by its name without the
//!   namespace, dots as `_` (`Position`, `Initial_Color`); the `outputs` are
//!   `var`s whose final values are written back;
//! - in grid stages: `Cell` (cell coordinates), `CellPosition` (cell centre in
//!   world units) and every attribute of the iterated grid as a `var`;
//! - `DeltaTime`, `Time`, `EmitterAge`; `rand(slot)` (a uniform draw in [0, 1),
//!   different per slot, particle or cell, module and step);
//! - for every grid `G` of the emitter: `G_A(cell)` (attribute `A` at a cell,
//!   clamped to the grid), `G_sample_A(pos)` (bilinear / trilinear at a world
//!   position), `G_cell(pos)`, `G_cell_position(cell)` and the constants
//!   `G_NumCells`, `G_Min`, `G_CellSize`;
//! - the helper library [`LIB`] (`curl_noise`, `noise_grad`, `quat_mul`, ...).
//!
//! Bodies are validated with naga when the effect is cooked; errors point at
//! the line in the `.fx.ron` file. Custom modules and simulation stages run on
//! the GPU only: the CPU simulation skips them.

use std::fmt::Write;

use serde::Serialize;
use serde::ser::{SerializeStructVariant, Serializer};

use crate::cook::CookedModule;
use crate::model::{ModuleCall, ParamMap, Span};
use crate::params;
use crate::types::{Value, ValueType};

/// Pure WGSL helpers (random hashing, noise, quaternions, shapes) shared by
/// the GPU simulation and custom module validation.
pub const LIB: &str = include_str!("wgsl/lib.wgsl");

/// Cooked module ids from here on are custom modules
/// ([`crate::cook::CookedSystem::custom_modules`] index + this).
pub const CUSTOM_BASE: u16 = 0x8000;
/// Most iterations of one simulation stage per step.
pub const MAX_ITERATIONS: u32 = 64;
/// Most cells of one grid.
pub const MAX_GRID_CELLS: u64 = 1 << 22;

pub fn custom_id(index: usize) -> u16 {
    CUSTOM_BASE + index as u16
}

pub fn custom_index(id: u16) -> Option<usize> {
    (id >= CUSTOM_BASE).then(|| (id - CUSTOM_BASE) as usize)
}

// ---- source ----

/// A name with where it is written; serialises as the plain string.
#[derive(Clone, Debug, PartialEq)]
pub struct Named {
    pub name: String,
    pub span: Span,
}

impl Serialize for Named {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name)
    }
}

/// One entry of `custom_modules: [ ... ]`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CustomModuleSource {
    pub name: String,
    /// Typed inputs with their default values.
    pub inputs: ParamMap,
    /// Particle attributes the body writes (`"Particles.Velocity"`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<Named>,
    /// The function body.
    pub wgsl: String,
    /// Where the body's first character is in the file.
    #[serde(skip)]
    pub wgsl_span: Span,
    /// The body was written as a raw string (`r#"..."#`), so its lines are the
    /// file's lines; otherwise escapes may have joined lines and errors point
    /// at the string.
    #[serde(skip)]
    pub wgsl_exact: bool,
    #[serde(skip)]
    pub span: Span,
}

/// A grid attribute declaration, `"Velocity": Vec2`.
#[derive(Clone, Debug, PartialEq)]
pub struct GridAttributeSource {
    pub name: String,
    pub ty: ValueType,
    pub span: Span,
}

/// One entry of an emitter's `grids: [ ... ]`: `Grid2D(...)` or `Grid3D(...)`.
/// A 2D grid lies in the world XY plane.
#[derive(Clone, Debug, PartialEq)]
pub struct GridSource {
    pub name: String,
    /// 2 or 3.
    pub dims: u32,
    pub cells: [u32; 3],
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub attributes: Vec<GridAttributeSource>,
    pub span: Span,
}

impl Serialize for GridSource {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        struct Attrs<'a>(&'a [GridAttributeSource]);
        impl Serialize for Attrs<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_map(self.0.iter().map(|a| (&a.name, a.ty)))
            }
        }
        let n = self.dims as usize;
        let (idx, variant) = if n == 2 { (0, "Grid2D") } else { (1, "Grid3D") };
        let mut v = s.serialize_struct_variant("Grid", idx, variant, 5)?;
        v.serialize_field("name", &self.name)?;
        v.serialize_field("cells", &self.cells[..n])?;
        v.serialize_field("min", &self.min[..n])?;
        v.serialize_field("max", &self.max[..n])?;
        v.serialize_field("attributes", &Attrs(&self.attributes))?;
        v.end()
    }
}

/// What a simulation stage iterates over.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Iterate {
    /// Every living particle, after the spawn and update.
    Particles,
    /// Every cell of the named grid.
    Grid(String),
}

/// One entry of an emitter's `simulation_stages: [ ... ]`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SimStageSource {
    pub name: String,
    pub iterate: Iterate,
    /// Times the stage runs per step (Jacobi solvers run many).
    pub iterations: u32,
    pub modules: Vec<ModuleCall>,
    #[serde(skip)]
    pub span: Span,
}

// ---- cooked ----

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedCustomInput {
    pub name: String,
    pub default: Value,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedCustomModule {
    pub name: String,
    pub inputs: Vec<CookedCustomInput>,
    /// Particle attributes written ([`params::PARTICLES`] indices).
    pub outputs: Vec<u16>,
    pub wgsl: String,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedGridAttribute {
    pub name: String,
    pub ty: ValueType,
    /// First word of the attribute within a cell's words.
    pub word: u32,
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedGrid {
    pub name: String,
    pub dims: u32,
    /// Cells per axis; the z count of a 2D grid is 1.
    pub cells: [u32; 3],
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub attributes: Vec<CookedGridAttribute>,
    /// f32 words per cell.
    pub words: u32,
}

impl CookedGrid {
    pub fn cell_count(&self) -> u32 {
        self.cells[0] * self.cells[1] * self.cells[2]
    }
    /// Words of one copy of the grid (the GPU keeps two, ping-pong).
    pub fn half_words(&self) -> u32 {
        self.cell_count() * self.words
    }
    pub fn cell_size(&self) -> [f32; 3] {
        std::array::from_fn(|i| (self.max[i] - self.min[i]) / self.cells[i] as f32)
    }
}

#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedSimStage {
    pub name: String,
    /// The iterated grid ([`crate::cook::CookedEmitter::grids`] index), or
    /// `None` for particles.
    pub grid: Option<u32>,
    pub iterations: u32,
    pub modules: Vec<CookedModule>,
}

// ---- WGSL generation, shared by validation and fx_gpu ----

/// What a custom module runs over.
#[derive(Clone, Copy)]
pub enum Context<'a> {
    Particles,
    Grid(&'a CookedGrid),
}

/// One argument of a custom module's generated function, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    /// A particle attribute ([`params::PARTICLES`] index).
    Attribute(u16),
    DeltaTime,
    Time,
    EmitterAge,
    /// Grid stages: the cell (`vec2<i32>` / `vec3<i32>`).
    Cell,
    /// Grid stages: the cell centre in world units.
    CellPosition,
    /// Grid stages: attribute `i` of the iterated grid.
    GridAttribute(usize),
    /// Input `i` of the module.
    Input(usize),
}

/// The WGSL type of values of `ty`.
pub fn wgsl_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Float => "f32",
        ValueType::Int => "i32",
        ValueType::Bool => "bool",
        ValueType::Vec2 => "vec2<f32>",
        ValueType::Vec3 => "vec3<f32>",
        ValueType::Vec4 | ValueType::Color | ValueType::Quat => "vec4<f32>",
    }
}

/// The local name of a particle attribute in a body (`Initial.Color` → `Initial_Color`).
pub fn attribute_local(index: u16) -> String {
    params::PARTICLES[index as usize].name.replace('.', "_")
}

fn grid_vec(dims: u32, scalar: &str) -> String {
    if dims == 2 {
        format!("vec2<{scalar}>")
    } else {
        format!("vec3<{scalar}>")
    }
}

/// The names a body sees besides its inputs, in a context (inputs must not reuse them).
pub fn reserved_locals(ctx: Context) -> Vec<String> {
    let mut v = vec!["DeltaTime".to_string(), "Time".into(), "EmitterAge".into()];
    match ctx {
        Context::Particles => {
            v.extend((0..params::PARTICLES.len() as u16).map(attribute_local));
        }
        Context::Grid(g) => {
            v.push("Cell".into());
            v.push("CellPosition".into());
            v.extend(g.attributes.iter().map(|a| a.name.clone()));
        }
    }
    v
}

/// `(local name, WGSL type, what is passed)` for each argument of the
/// generated function of `m` in `ctx`.
pub fn args(m: &CookedCustomModule, ctx: Context) -> Vec<(String, String, Arg)> {
    let mut out = Vec::new();
    match ctx {
        Context::Particles => {
            for (i, a) in params::PARTICLES.iter().enumerate() {
                out.push((attribute_local(i as u16), wgsl_type(a.ty).into(), Arg::Attribute(i as u16)));
            }
        }
        Context::Grid(g) => {
            out.push(("Cell".into(), grid_vec(g.dims, "i32"), Arg::Cell));
            out.push(("CellPosition".into(), grid_vec(g.dims, "f32"), Arg::CellPosition));
            for (i, a) in g.attributes.iter().enumerate() {
                out.push((a.name.clone(), wgsl_type(a.ty).into(), Arg::GridAttribute(i)));
            }
        }
    }
    out.push(("DeltaTime".into(), "f32".into(), Arg::DeltaTime));
    out.push(("Time".into(), "f32".into(), Arg::Time));
    out.push(("EmitterAge".into(), "f32".into(), Arg::EmitterAge));
    for (i, input) in m.inputs.iter().enumerate() {
        out.push((input.name.clone(), wgsl_type(input.default.ty()).into(), Arg::Input(i)));
    }
    out
}

/// What the generated function returns, in its result struct's field order:
/// [`Arg::Attribute`] (particle outputs) or [`Arg::GridAttribute`] (every
/// attribute of the iterated grid).
pub fn outputs(m: &CookedCustomModule, ctx: Context) -> Vec<(String, String, Arg)> {
    match ctx {
        Context::Particles => m
            .outputs
            .iter()
            .map(|&a| {
                (
                    attribute_local(a),
                    wgsl_type(params::PARTICLES[a as usize].ty).into(),
                    Arg::Attribute(a),
                )
            })
            .collect(),
        Context::Grid(g) => g
            .attributes
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name.clone(), wgsl_type(a.ty).into(), Arg::GridAttribute(i)))
            .collect(),
    }
}

/// The generated function `name` wrapping `m`'s body in `ctx`, and the
/// 0-based line of the text where the body starts.
pub fn wrapper(m: &CookedCustomModule, name: &str, ctx: Context) -> (String, u32) {
    let outs = outputs(m, ctx);
    let mut s = String::new();
    writeln!(s, "// custom module \"{}\"", m.name).unwrap();
    if !outs.is_empty() {
        writeln!(s, "struct {name}_out {{").unwrap();
        for (n, t, _) in &outs {
            writeln!(s, "    {n}: {t},").unwrap();
        }
        writeln!(s, "}}").unwrap();
    }
    let params: Vec<String> = args(m, ctx)
        .iter()
        .map(|(n, t, a)| {
            if outs.iter().any(|o| o.2 == *a) {
                format!("{n}_in: {t}")
            } else {
                format!("{n}: {t}")
            }
        })
        .collect();
    let ret = if outs.is_empty() {
        String::new()
    } else {
        format!(" -> {name}_out")
    };
    writeln!(s, "fn {name}({}){ret} {{", params.join(", ")).unwrap();
    for (n, _, _) in &outs {
        writeln!(s, "    var {n} = {n}_in;").unwrap();
    }
    s.push_str("    {\n");
    let body_line = s.matches('\n').count() as u32;
    s.push_str(&m.wgsl);
    s.push_str("\n    }\n");
    if !outs.is_empty() {
        let names: Vec<&str> = outs.iter().map(|o| o.0.as_str()).collect();
        writeln!(s, "    return {name}_out({});", names.join(", ")).unwrap();
    }
    s.push_str("}\n");
    (s, body_line)
}

/// Grid constants and accessors for an emitter's grids; `base[g]` is grid g's
/// first word in the grid buffer. Reads go through `grid_ld(word) -> f32` and
/// the private `gh_<g>` (which copy holds the current values).
pub fn grid_helpers(grids: &[CookedGrid], base: &[u32]) -> String {
    let mut s = String::new();
    for (gi, g) in grids.iter().enumerate() {
        let n = &g.name;
        let d = g.dims as usize;
        let iv = grid_vec(g.dims, "i32");
        let fv = grid_vec(g.dims, "f32");
        let cs = g.cell_size();
        let fl = |a: &[f32]| a[..d].iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(", ");
        let cells: Vec<String> = g.cells[..d].iter().map(|c| c.to_string()).collect();
        writeln!(s, "// ---- grid \"{n}\" ----").unwrap();
        writeln!(s, "const {n}_NumCells = {iv}({});", cells.join(", ")).unwrap();
        writeln!(s, "const {n}_Min = {fv}({});", fl(&g.min)).unwrap();
        writeln!(s, "const {n}_CellSize = {fv}({});", fl(&cs)).unwrap();
        writeln!(s, "var<private> gh_{gi}: u32;").unwrap();
        let index = if d == 2 {
            format!("u32(q.y) * {}u + u32(q.x)", g.cells[0])
        } else {
            format!(
                "(u32(q.z) * {}u + u32(q.y)) * {}u + u32(q.x)",
                g.cells[1], g.cells[0]
            )
        };
        writeln!(
            s,
            "fn {n}_index(c: {iv}) -> u32 {{\n    let q = clamp(c, {iv}(0), {n}_NumCells - 1);\n    return {index};\n}}"
        )
        .unwrap();
        writeln!(
            s,
            "fn {n}_cell(pos: {fv}) -> {fv} {{\n    return (pos - {n}_Min) / {n}_CellSize - 0.5;\n}}"
        )
        .unwrap();
        writeln!(
            s,
            "fn {n}_cell_position(c: {iv}) -> {fv} {{\n    return {n}_Min + ({fv}(c) + 0.5) * {n}_CellSize;\n}}"
        )
        .unwrap();
        let cellc = g.cell_count();
        for a in &g.attributes {
            let t = wgsl_type(a.ty);
            let comps = a.ty.components();
            let loads: Vec<String> = (0..comps)
                .map(|c| format!("grid_ld(i + {}u)", (a.word + c as u32) * cellc))
                .collect();
            let val = if comps == 1 {
                loads[0].clone()
            } else {
                format!("{t}({})", loads.join(", "))
            };
            writeln!(
                s,
                "fn {n}_{an}(c: {iv}) -> {t} {{\n    let i = {b}u + gh_{gi} * {h}u + {n}_index(c);\n    return {val};\n}}",
                an = a.name,
                b = base[gi],
                h = g.half_words(),
            )
            .unwrap();
            // Linear interpolation between cell centres.
            let at = |o: &str| format!("{n}_{}(c + {iv}({o}))", a.name);
            let body = if d == 2 {
                format!(
                    "    return mix(mix({}, {}, f.x), mix({}, {}, f.x), f.y);",
                    at("0, 0"),
                    at("1, 0"),
                    at("0, 1"),
                    at("1, 1")
                )
            } else {
                format!(
                    "    let z0 = mix(mix({}, {}, f.x), mix({}, {}, f.x), f.y);\n    let z1 = mix(mix({}, {}, f.x), mix({}, {}, f.x), f.y);\n    return mix(z0, z1, f.z);",
                    at("0, 0, 0"),
                    at("1, 0, 0"),
                    at("0, 1, 0"),
                    at("1, 1, 0"),
                    at("0, 0, 1"),
                    at("1, 0, 1"),
                    at("0, 1, 1"),
                    at("1, 1, 1")
                )
            };
            writeln!(
                s,
                "fn {n}_sample_{an}(pos: {fv}) -> {t} {{\n    let g = {n}_cell(pos);\n    let b = floor(g);\n    let f = g - b;\n    let c = {iv}(b);\n{body}\n}}",
                an = a.name
            )
            .unwrap();
        }
    }
    s
}

/// Word offsets of each grid in an emitter's grid buffer (two copies each).
pub fn grid_bases(grids: &[CookedGrid]) -> Vec<u32> {
    let mut at = 0u32;
    grids
        .iter()
        .map(|g| {
            let b = at;
            at += 2 * g.half_words();
            b
        })
        .collect()
}

/// A WGSL error: 1-based line and column in the checked text, and the message.
#[derive(Clone, Debug, PartialEq)]
pub struct WgslError {
    pub line: u32,
    pub col: u32,
    pub message: String,
    /// Every (line, column) the error names, innermost first (`line, col` is the first).
    pub places: Vec<(u32, u32)>,
}

/// Parses and validates a whole WGSL module with naga.
pub fn validate_wgsl(source: &str) -> Result<(), WgslError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| {
        let (line, col) = e
            .labels()
            .next()
            .map(|(sp, _)| {
                let l = sp.location(source);
                (l.line_number, l.line_position)
            })
            .unwrap_or((0, 0));
        let label = e.labels().next().map(|(_, m)| m.to_string()).unwrap_or_default();
        let message = if label.is_empty() || label == e.message() {
            e.message().to_string()
        } else {
            format!("{} ({label})", e.message())
        };
        WgslError { line, col, message, places: vec![(line, col)] }
    })?;
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module).map_err(|e| {
        // The last span is the innermost (the expression or statement).
        let mut places: Vec<(u32, u32)> = e
            .spans()
            .map(|(sp, _)| {
                let l = sp.location(source);
                (l.line_number, l.line_position)
            })
            .collect();
        places.reverse();
        // Expressions the message names (`[29]`) but gives no span for: look
        // their spans up in the failing function.
        if let naga::valid::ValidationError::Function { handle, .. } = e.as_inner() {
            let f = &module.functions[*handle];
            let text = e.as_inner().to_string() + &std::error::Error::source(e.as_inner()).map(|s| s.to_string()).unwrap_or_default();
            let named: Vec<usize> = text
                .split('[')
                .skip(2) // the function's own handle comes first
                .filter_map(|p| p.split(']').next().and_then(|n| n.parse::<usize>().ok()))
                .collect();
            // The innermost statement using every named expression.
            let markers: Vec<String> = named.iter().map(|n| format!("[{n}]")).collect();
            fn walk(b: &naga::Block, markers: &[String], found: &mut Option<naga::Span>) {
                for (st, sp) in b.span_iter() {
                    match st {
                        naga::Statement::Block(inner) => walk(inner, markers, found),
                        naga::Statement::If { accept, reject, .. } => {
                            walk(accept, markers, found);
                            walk(reject, markers, found);
                        }
                        naga::Statement::Loop { body, continuing, .. } => {
                            walk(body, markers, found);
                            walk(continuing, markers, found);
                        }
                        _ => {}
                    }
                    let d = format!("{st:?}");
                    if found.is_none() && sp.is_defined() && !markers.is_empty() && markers.iter().all(|m| d.contains(m.as_str())) {
                        *found = Some(*sp);
                    }
                }
            }
            let mut found = None;
            walk(&f.body, &markers, &mut found);
            if let Some(sp) = found {
                let l = sp.location(source);
                places.insert(0, (l.line_number, l.line_position));
            }
            for num in named {
                if let Some((h, _)) = f.expressions.iter().find(|(h, _)| h.index() == num) {
                    let sp = f.expressions.get_span(h);
                    if sp.is_defined() {
                        let l = sp.location(source);
                        places.push((l.line_number, l.line_position));
                    }
                }
            }
        }
        let (line, col) = places.first().copied().unwrap_or((0, 0));
        let mut message = e.as_inner().to_string();
        let mut cause: Option<&dyn std::error::Error> = std::error::Error::source(e.as_inner());
        while let Some(c) = cause {
            message.push_str(": ");
            message.push_str(&c.to_string());
            cause = c.source();
        }
        if let Some((_, desc)) = e.spans().last() {
            if !desc.is_empty() {
                message.push_str(&format!(" ({desc})"));
            }
        }
        WgslError { line, col, message, places }
    })?;
    Ok(())
}

/// Where a body error is, relative to the body: 0-based line and 1-based column.
#[derive(Clone, Debug, PartialEq)]
pub enum BodyError {
    /// In the body: line within it (0-based), column (1-based).
    Body { line: u32, col: u32, message: String },
    /// Outside the body (the generated wrapper: an input name clash, a
    /// reserved word, an unbalanced brace in the body).
    Wrapper(String),
}

/// Validates `m`'s body in `ctx` for an emitter with `grids`, against the
/// helper library and stand-ins for the GPU bindings.
pub fn check_body(m: &CookedCustomModule, ctx: Context, grids: &[CookedGrid]) -> Result<(), BodyError> {
    let mut src = String::new();
    src.push_str(LIB);
    src.push_str("\n// stand-ins for the GPU simulation's bindings\nfn rand(slot: u32) -> f32 { return f32(slot) * 0.0; }\nfn grid_ld(w: u32) -> f32 { return f32(w) * 0.0; }\n");
    src.push_str(&grid_helpers(grids, &grid_bases(grids)));
    let first = src.matches('\n').count() as u32;
    let (w, body_line) = wrapper(m, "custom_module", ctx);
    src.push_str(&w);
    let body_lines = m.wgsl.matches('\n').count() as u32 + 1;
    match validate_wgsl(&src) {
        Ok(()) => Ok(()),
        Err(e) => {
            let start = first + body_line + 1; // 1-based line of the body's first line
            let inside = e
                .places
                .iter()
                .find(|(l, _)| *l >= start && *l < start + body_lines);
            match inside {
                Some(&(line, col)) => Err(BodyError::Body {
                    line: line - start,
                    col,
                    message: e.message,
                }),
                None => Err(BodyError::Wrapper(e.message)),
            }
        }
    }
}

/// Every module of an emitter's stages that runs on the particles.
pub fn particle_stage_modules(stages: &[CookedSimStage]) -> impl Iterator<Item = &CookedModule> {
    stages
        .iter()
        .filter(|s| s.grid.is_none())
        .flat_map(|s| s.modules.iter())
}

// ---- cooking (called from crate::cook) ----

use crate::diag::Diags;
use crate::model::{EmitterSource, SimTarget, SystemSource};

/// Whether `s` can name a WGSL value (`Strength`, `wind_2`).
pub fn is_identifier(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        && !s.starts_with("__")
        && s != "_"
}

pub(crate) fn cook_declarations(src: &SystemSource, d: &mut Diags) -> Vec<CookedCustomModule> {
    let mut out: Vec<CookedCustomModule> = Vec::new();
    let particle_locals = reserved_locals(Context::Particles);
    for m in &src.custom_modules {
        if m.name.trim().is_empty() {
            d.error(m.span, "custom module `name` is empty", Some("name it, e.g. `name: \"Swirl\"`".into()));
        } else if crate::modules::by_name(&m.name).is_some() {
            d.error(
                m.span,
                format!("custom module \"{}\" has the name of a built-in module", m.name),
                Some("give it another name, e.g. \"My {}\"".replace("{}", &m.name)),
            );
        } else if out.iter().any(|o| o.name == m.name) {
            d.error(m.span, format!("two custom modules are named \"{}\"", m.name), Some("rename one".into()));
        }
        let mut inputs: Vec<CookedCustomInput> = Vec::new();
        for p in &m.inputs.0 {
            if !is_identifier(&p.name) {
                d.error(
                    p.span,
                    format!("custom module input \"{}\" is not a WGSL name", p.name),
                    Some("input names are used in the body as written: letters, digits and `_`, not starting with a digit (e.g. \"Strength\")".into()),
                );
            } else if particle_locals.contains(&p.name) || ["Cell", "CellPosition"].contains(&p.name.as_str()) {
                d.error(
                    p.span,
                    format!("input \"{}\" has the name of a value the body already sees", p.name),
                    Some(format!("rename it, e.g. \"{}Input\"", p.name)),
                );
            } else if inputs.iter().any(|i| i.name == p.name) {
                d.error(p.span, format!("input \"{}\" is declared twice", p.name), Some("keep one".into()));
            }
            inputs.push(CookedCustomInput { name: p.name.clone(), default: p.value });
        }
        let mut outputs = Vec::new();
        for o in &m.outputs {
            let attr = o
                .name
                .strip_prefix("Particles.")
                .and_then(|n| params::PARTICLES.iter().position(|b| b.name == n));
            match attr {
                Some(i) if i as u16 == params::attr::UNIQUE_ID => d.error(
                    o.span,
                    "Particles.UniqueID cannot be written",
                    Some("remove it from `outputs`".into()),
                ),
                Some(i) => {
                    if outputs.contains(&(i as u16)) {
                        d.error(o.span, format!("\"{}\" is listed twice", o.name), Some("keep one".into()));
                    } else {
                        outputs.push(i as u16);
                    }
                }
                None => {
                    let names: Vec<String> =
                        params::PARTICLES.iter().map(|b| format!("Particles.{}", b.name)).collect();
                    let help = match crate::diag::suggest(&o.name, names.iter().map(|s| s.as_str())) {
                        Some(s) => format!("did you mean \"{s}\"?"),
                        None => format!("outputs are particle attributes: {}", names.join(", ")),
                    };
                    d.error(o.span, format!("\"{}\" is not a particle attribute", o.name), Some(help));
                }
            }
        }
        if m.wgsl.trim().is_empty() {
            d.warning(m.wgsl_span, format!("custom module \"{}\" has an empty body", m.name), None);
        }
        out.push(CookedCustomModule {
            name: m.name.clone(),
            inputs,
            outputs,
            wgsl: m.wgsl.clone(),
        });
    }
    out
}

pub(crate) fn cook_grids(e: &EmitterSource, d: &mut Diags) -> Vec<CookedGrid> {
    let mut out: Vec<CookedGrid> = Vec::new();
    for g in &e.grids {
        if !is_identifier(&g.name) {
            d.error(
                g.span,
                format!("grid name \"{}\" is not a WGSL name", g.name),
                Some("bodies call `<Grid>_<Attribute>(cell)`, so use letters, digits and `_` (e.g. \"Smoke\")".into()),
            );
        } else if out.iter().any(|o| o.name == g.name) {
            d.error(g.span, format!("two grids are named \"{}\"", g.name), Some("rename one".into()));
        }
        let n = g.dims as usize;
        let cells: u64 = g.cells[..n].iter().map(|&c| c as u64).product();
        if g.cells[..n].contains(&0) || cells > MAX_GRID_CELLS {
            d.error(
                g.span,
                format!("grid \"{}\" has {cells} cells; it needs at least 1 per axis and at most {MAX_GRID_CELLS}", g.name),
                Some("write e.g. `cells: (64, 64)`".into()),
            );
        }
        if (0..n).any(|i| !(g.max[i] > g.min[i])) {
            d.error(
                g.span,
                format!("grid \"{}\": every `max` component must be above its `min`", g.name),
                Some("`min` and `max` are opposite corners in world units".into()),
            );
        }
        if g.attributes.is_empty() {
            d.error(g.span, format!("grid \"{}\" has no attributes", g.name), Some("add e.g. `attributes: { \"Density\": Float }`".into()));
        }
        let mut attributes: Vec<CookedGridAttribute> = Vec::new();
        let mut word = 0;
        for a in &g.attributes {
            if !is_identifier(&a.name) || ["Cell", "CellPosition", "DeltaTime", "Time", "EmitterAge"].contains(&a.name.as_str()) {
                d.error(
                    a.span,
                    format!("grid attribute name \"{}\" cannot be used", a.name),
                    Some("use letters, digits and `_`, and not Cell, CellPosition, DeltaTime, Time or EmitterAge".into()),
                );
            } else if attributes.iter().any(|o| o.name == a.name) {
                d.error(a.span, format!("attribute \"{}\" is declared twice", a.name), Some("keep one".into()));
            }
            attributes.push(CookedGridAttribute { name: a.name.clone(), ty: a.ty, word });
            word += a.ty.components() as u32;
        }
        let mut cells3 = g.cells;
        if n == 2 {
            cells3[2] = 1;
        }
        out.push(CookedGrid {
            name: g.name.clone(),
            dims: g.dims,
            cells: cells3,
            min: g.min,
            max: g.max,
            attributes,
            words: word,
        });
    }
    out
}

/// What a body can use, for error help.
fn context_help(m: &CookedCustomModule, ctx: Context, grids: &[CookedGrid]) -> String {
    let mut s = String::from("the body sees its inputs");
    if !m.inputs.is_empty() {
        let names: Vec<&str> = m.inputs.iter().map(|i| i.name.as_str()).collect();
        s.push_str(&format!(" ({})", names.join(", ")));
    }
    match ctx {
        Context::Particles => s.push_str(", the particle attributes by name (Position, Velocity, Color, ...; outputs are `var`s)"),
        Context::Grid(g) => {
            let names: Vec<&str> = g.attributes.iter().map(|a| a.name.as_str()).collect();
            s.push_str(&format!(", Cell, CellPosition, grid \"{}\"'s attributes as `var`s ({})", g.name, names.join(", ")));
        }
    }
    s.push_str(", DeltaTime, Time, EmitterAge, rand(slot)");
    for g in grids {
        let a = g.attributes.first().map_or("A", |a| a.name.as_str());
        s.push_str(&format!(", {0}_{a}(cell), {0}_sample_{a}(pos), {0}_cell(pos)", g.name));
    }
    s.push_str(" and the helpers curl_noise, noise_grad, euler_quat, quat_mul, quat_normalize");
    s
}

/// GPU-only checks and naga validation of every custom module call in `ce`.
pub(crate) fn check_emitter(
    src: &SystemSource,
    e: &EmitterSource,
    ce: &mut crate::cook::CookedEmitter,
    custom: &[CookedCustomModule],
    d: &mut Diags,
) {
    let particle: Vec<usize> = ce
        .particle_modules()
        .filter_map(|m| custom_index(m.module))
        .collect();
    let grid: Vec<(usize, usize)> = ce
        .stages
        .iter()
        .filter_map(|s| s.grid.map(|g| (s, g as usize)))
        .flat_map(|(s, g)| s.modules.iter().filter_map(move |m| custom_index(m.module).map(|c| (c, g))))
        .collect();
    if e.sim_target == SimTarget::Cpu && (!particle.is_empty() || !e.grids.is_empty() || !e.simulation_stages.is_empty()) {
        d.error(
            e.span,
            format!("emitter \"{}\" simulates on the CPU, but custom WGSL modules, grids and simulation stages run only on the GPU", e.name),
            Some("set `sim_target: Gpu`".into()),
        );
        return;
    }
    ce.custom_extended = particle
        .iter()
        .any(|&c| custom[c].outputs.iter().any(|&a| a >= params::attr::FIRST_EXTENDED));
    ce.custom_extended_2 = particle
        .iter()
        .any(|&c| custom[c].outputs.iter().any(|&a| a >= params::attr::FIRST_EXTENDED_2));
    let mut seen: Vec<(usize, Option<usize>)> = Vec::new();
    let sites = particle.iter().map(|&c| (c, None)).chain(grid.iter().map(|&(c, g)| (c, Some(g))));
    for (c, g) in sites {
        if seen.contains(&(c, g)) {
            continue;
        }
        seen.push((c, g));
        let m = &custom[c];
        let ctx = match g {
            Some(g) => Context::Grid(&ce.grids[g]),
            None => Context::Particles,
        };
        if let Context::Grid(gr) = ctx {
            if let Some(i) = m.inputs.iter().find(|i| gr.attributes.iter().any(|a| a.name == i.name)) {
                d.error(
                    src.custom_modules[c].span,
                    format!("input \"{}\" of \"{}\" has the name of an attribute of grid \"{}\"", i.name, m.name, gr.name),
                    Some("rename the input".into()),
                );
                continue;
            }
        }
        let Err(err) = check_body(m, ctx, &ce.grids) else {
            continue;
        };
        let decl = &src.custom_modules[c];
        let place = match g {
            Some(g) => format!("in grid stages over \"{}\"", ce.grids[g].name),
            None => "on particles".to_string(),
        };
        let help = Some(context_help(m, ctx, &ce.grids));
        let (span, message) = match err {
            BodyError::Body { line, col, message } if decl.wgsl_exact => {
                let span = if line == 0 {
                    Span { line: decl.wgsl_span.line, col: decl.wgsl_span.col + col - 1 }
                } else {
                    Span { line: decl.wgsl_span.line + line, col }
                };
                (span, format!("WGSL error in custom module \"{}\" ({place}, emitter \"{}\"): {message}", m.name, e.name))
            }
            BodyError::Body { line, col, message } => (
                decl.wgsl_span,
                format!("WGSL error in custom module \"{}\" ({place}, emitter \"{}\") at line {}, column {col} of its body: {message}", m.name, e.name, line + 1),
            ),
            BodyError::Wrapper(message) => (
                decl.span,
                format!("custom module \"{}\" ({place}, emitter \"{}\") does not compile: {message} (check that the body's braces balance and no input is named after a WGSL keyword)", m.name, e.name),
            ),
        };
        if !d.list.iter().any(|x| x.span == span && x.message == message) {
            d.error(span, message, help);
        }
    }
}
