//! Tagged property serialization (FPropertyTag) and property values.
//!
//! Two tag layouts exist in UE5 editor saves:
//! - before UE5 1012: name, type name, size, array index, per-type data
//!   (struct name + GUID, enum name, inner/value types, bool value), GUID flag,
//!   and from 1011 an extension byte;
//! - from 1012 (UE 5.4): name, the complete type as a tree of
//!   `(FName, inner count)` nodes, size, a flags byte, optional array index,
//!   GUID and extensions. The flags say whether a struct was saved natively.

use crate::package::{read_fname, PackageIndex, Versions};
use crate::reader::{Error, Guid, Reader, Result};
use crate::summary::{custom, ue5};

/// A property type: `ArrayProperty(StructProperty(Vector))`. Parameters of
/// struct and enum types are their names (the package path dropped).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypeName {
    pub name: String,
    pub params: Vec<TypeName>,
}

impl TypeName {
    pub fn new(name: impl Into<String>) -> Self {
        TypeName {
            name: name.into(),
            params: Vec::new(),
        }
    }
    pub fn with(name: impl Into<String>, params: Vec<TypeName>) -> Self {
        TypeName {
            name: name.into(),
            params,
        }
    }
    pub fn param(&self, i: usize) -> Option<&TypeName> {
        self.params.get(i)
    }
    /// The struct name of a StructProperty, or the enum name of a
    /// Byte/EnumProperty, when known.
    pub fn inner_name(&self) -> Option<&str> {
        self.params.first().map(|p| p.name.as_str()).filter(|n| *n != "None")
    }
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)?;
        if !self.params.is_empty() {
            f.write_str("(")?;
            for (i, p) in self.params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{p}")?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub name: String,
    pub ty: TypeName,
    pub array_index: u32,
    pub value: Value,
}

/// FSoftObjectPath.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoftPath {
    pub package: String,
    pub asset: String,
    pub sub_path: String,
}

impl SoftPath {
    /// The in-line form (used by the package's soft object path list, and
    /// by properties when the package has no list).
    pub fn read_raw(r: &mut Reader, names: &[String], v: &Versions) -> Result<SoftPath> {
        let (package, asset) = if v.ue5 >= ue5::FSOFTOBJECTPATH_REMOVE_ASSET_PATH_FNAMES {
            (read_fname(r, names)?, read_fname(r, names)?)
        } else {
            let full = read_fname(r, names)?;
            match full.split_once('.') {
                Some((p, a)) => (p.to_string(), a.to_string()),
                None => (full, String::new()),
            }
        };
        let sub_path = if v.fortnite_main.unwrap_or(0) >= custom::FORTNITE_SOFT_OBJECT_PATH_UTF8 {
            r.utf8_string()?
        } else {
            r.fstring()?
        };
        Ok(SoftPath {
            package: none_empty(package),
            asset: none_empty(asset),
            sub_path,
        })
    }
}

fn none_empty(s: String) -> String {
    if s == "None" { String::new() } else { s }
}

impl std::fmt::Display for SoftPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.package.is_empty() && self.asset.is_empty() {
            return f.write_str("None");
        }
        write!(f, "{}", self.package)?;
        if !self.asset.is_empty() {
            write!(f, ".{}", self.asset)?;
        }
        if !self.sub_path.is_empty() {
            write!(f, ":{}", self.sub_path)?;
        }
        Ok(())
    }
}

/// FText, as far as editor assets usually use it.
#[derive(Clone, Debug, PartialEq)]
pub enum Text {
    /// No history: an empty or culture-invariant string.
    None { invariant: Option<String> },
    /// A localizable literal.
    Base {
        namespace: String,
        key: String,
        source: String,
    },
    /// Another history type (formatted text etc.), kept raw.
    Other { history: i8, bytes: Vec<u8> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int(i32),
    Int64(i64),
    Byte(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float(f32),
    Double(f64),
    Name(String),
    Str(String),
    Text(Text),
    /// An enum value (`EEnum::Value`) from a Byte or EnumProperty.
    Enum(String),
    Object(PackageIndex),
    SoftObject(SoftPath),
    Guid(Guid),
    /// A struct: tagged fields, or a native struct decoded into fields.
    Struct {
        name: String,
        fields: Vec<Property>,
    },
    Array(Vec<Value>),
    /// Array of bytes (common for blobs); kept compact.
    Bytes(Vec<u8>),
    Set(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Delegate {
        object: PackageIndex,
        function: String,
    },
    MulticastDelegate(Vec<(PackageIndex, String)>),
    FieldPath {
        path: Vec<String>,
        owner: PackageIndex,
    },
    /// Bytes this reader does not decode, with the reason.
    Raw { bytes: Vec<u8>, note: String },
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::Float(v) => v as f64,
            Value::Double(v) => v,
            Value::Int(v) => v as f64,
            Value::Int8(v) => v as f64,
            Value::Int16(v) => v as f64,
            Value::Int64(v) => v as f64,
            Value::Byte(v) => v as f64,
            Value::UInt16(v) => v as f64,
            Value::UInt32(v) => v as f64,
            Value::UInt64(v) => v as f64,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Name(s) | Value::Str(s) | Value::Enum(s) => Some(s),
            _ => None,
        }
    }
    /// Field of a struct value.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Struct { fields, .. } => fields.iter().find(|p| p.name == name).map(|p| &p.value),
            _ => None,
        }
    }
}

/// Look up a property by name in a list.
pub fn find<'a>(props: &'a [Property], name: &str) -> Option<&'a Value> {
    props.iter().find(|p| p.name == name).map(|p| &p.value)
}

/// Reading state: the name map, versions, and problems found so far
/// (with a path like `EmitterHandles[0].Instance`).
pub struct Ctx<'a> {
    pub names: &'a [String],
    pub soft_paths: &'a [SoftPath],
    pub v: &'a Versions,
    pub problems: Vec<(String, String)>,
    path: Vec<String>,
}

impl<'a> Ctx<'a> {
    pub fn new(names: &'a [String], soft_paths: &'a [SoftPath], v: &'a Versions) -> Self {
        Ctx {
            names,
            soft_paths,
            v,
            problems: Vec::new(),
            path: Vec::new(),
        }
    }
    fn fname(&self, r: &mut Reader) -> Result<String> {
        read_fname(r, self.names)
    }
    fn problem(&mut self, msg: String) {
        let path = self.path.join(".");
        self.problems.push((path, msg));
    }
    fn lwc(&self) -> bool {
        self.v.ue5 >= ue5::LARGE_WORLD_COORDINATES
    }
}

/// Tag header fields that matter for the value.
#[derive(Clone, Debug, Default)]
struct Tag {
    name: String,
    ty: TypeName,
    size: usize,
    array_index: u32,
    bool_value: bool,
    /// From the 1012 flags: the value was saved with native serialization.
    native: Option<bool>,
    /// From the 1012 flags: serialization was skipped (no value bytes).
    skipped: bool,
}

const FLAG_HAS_ARRAY_INDEX: u8 = 0x01;
const FLAG_HAS_PROPERTY_GUID: u8 = 0x02;
const FLAG_HAS_EXTENSIONS: u8 = 0x04;
const FLAG_BINARY_OR_NATIVE: u8 = 0x08;
const FLAG_BOOL_TRUE: u8 = 0x10;
const FLAG_SKIPPED: u8 = 0x20;
const EXT_OVERRIDABLE: u8 = 0x02;

/// Read a tag header; `None` at the "None" terminator.
fn read_tag(r: &mut Reader, ctx: &Ctx) -> Result<Option<Tag>> {
    let name = ctx.fname(r)?;
    if name == "None" {
        return Ok(None);
    }
    let mut t = Tag {
        name,
        ..Tag::default()
    };
    if ctx.v.ue5 >= ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME {
        t.ty = read_type_name(r, ctx)?;
        t.size = r.count(0)?;
        let flags = r.u8()?;
        t.bool_value = flags & FLAG_BOOL_TRUE != 0;
        t.native = Some(flags & FLAG_BINARY_OR_NATIVE != 0);
        t.skipped = flags & FLAG_SKIPPED != 0;
        if flags & FLAG_HAS_ARRAY_INDEX != 0 {
            t.array_index = r.u32()?;
        }
        if flags & FLAG_HAS_PROPERTY_GUID != 0 {
            r.guid()?;
        }
        if flags & FLAG_HAS_EXTENSIONS != 0 {
            read_tag_extensions(r)?;
        }
    } else {
        let ty = ctx.fname(r)?;
        t.size = r.count(0)?;
        t.array_index = r.u32()?;
        let params = match ty.as_str() {
            "StructProperty" => {
                let s = ctx.fname(r)?;
                r.guid()?; // StructGuid
                vec![TypeName::new(s)]
            }
            "BoolProperty" => {
                t.bool_value = r.u8()? != 0;
                vec![]
            }
            "ByteProperty" | "EnumProperty" => vec![TypeName::new(ctx.fname(r)?)],
            "ArrayProperty" | "SetProperty" | "OptionalProperty" => vec![TypeName::new(ctx.fname(r)?)],
            "MapProperty" => vec![TypeName::new(ctx.fname(r)?), TypeName::new(ctx.fname(r)?)],
            _ => vec![],
        };
        t.ty = TypeName::with(ty, params);
        if r.u8()? != 0 {
            r.guid()?; // PropertyGuid
        }
        if ctx.v.ue5 >= ue5::PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION {
            read_tag_extensions(r)?;
        }
    }
    Ok(Some(t))
}

fn read_tag_extensions(r: &mut Reader) -> Result<()> {
    let ext = r.u8()?;
    if ext & EXT_OVERRIDABLE != 0 {
        r.u8()?; // EOverriddenPropertyOperation
        r.bool32()?; // bExperimentalOverridableLogic
    }
    Ok(())
}

/// FPropertyTypeName: nodes in pre-order, each with its child count.
fn read_type_name(r: &mut Reader, ctx: &Ctx) -> Result<TypeName> {
    fn node(r: &mut Reader, ctx: &Ctx, depth: usize) -> Result<TypeName> {
        if depth > 16 {
            return Err(r.err("property type name nested too deep"));
        }
        let name = ctx.fname(r)?;
        let n = r.count(12)?;
        let mut params = Vec::with_capacity(n);
        for _ in 0..n {
            params.push(node(r, ctx, depth + 1)?);
        }
        Ok(TypeName { name, params })
    }
    let mut t = node(r, ctx, 0)?;
    strip_paths(&mut t);
    Ok(t)
}

/// Struct and enum names carry their package as a child node
/// (`Vector(/Script/CoreUObject)`); drop it.
fn strip_paths(t: &mut TypeName) {
    t.params.retain(|p| !p.name.starts_with('/'));
    for p in &mut t.params {
        strip_paths(p);
    }
}

/// Tagged properties up to the "None" terminator. `object` adds the
/// UObject serialization-control byte (UE5 1011+).
pub fn read_tagged_properties(r: &mut Reader, ctx: &mut Ctx, object: bool) -> Result<Vec<Property>> {
    if object && ctx.v.ue5 >= ue5::PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION {
        let ext = r.u8()?;
        if ext & EXT_OVERRIDABLE != 0 {
            r.u8()?;
        }
    }
    let mut props = Vec::new();
    loop {
        if props.len() > 100_000 {
            return Err(r.err("too many properties"));
        }
        let Some(tag) = read_tag(r, ctx)? else { break };
        let mut sub = r.sub(tag.size)?;
        ctx.path.push(if tag.array_index > 0 {
            format!("{}[{}]", tag.name, tag.array_index)
        } else {
            tag.name.clone()
        });
        let value = top_value(&mut sub, &tag, ctx);
        ctx.path.pop();
        props.push(Property {
            name: tag.name,
            ty: tag.ty,
            array_index: tag.array_index,
            value,
        });
    }
    Ok(props)
}

/// The value of a top-level tag, read from exactly its `size` bytes. A
/// value that fails to decode is kept raw and recorded as a problem.
fn top_value(r: &mut Reader, tag: &Tag, ctx: &mut Ctx) -> Value {
    let start = r.pos();
    let res = (|| -> Result<Value> {
        if tag.skipped {
            return Ok(Value::Raw {
                bytes: Vec::new(),
                note: "serialization skipped".into(),
            });
        }
        let v = match tag.ty.name.as_str() {
            "BoolProperty" => Value::Bool(tag.bool_value),
            "ByteProperty" if tag.size == 1 => Value::Byte(r.u8()?),
            "ByteProperty" => Value::Enum(ctx.fname(r)?),
            "StructProperty" => read_struct(r, tag.ty.inner_name(), tag.native, ctx)?,
            "ArrayProperty" => read_array(r, &tag.ty, tag.size, ctx)?,
            "MapProperty" if ctx.v.ue5 < ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME => read_map_guess(r, &tag.name, &tag.ty, ctx)?,
            _ => read_elem(r, &tag.ty, ctx, ElemHint::default())?,
        };
        if r.remaining() != 0 {
            return Err(r.err(format!("{} bytes left after the value", r.remaining())));
        }
        Ok(v)
    })();
    match res {
        Ok(v) => v,
        Err(e) => {
            ctx.problem(format!("{}: {}", tag.ty, e));
            let mut all = r.clone();
            let _ = all.seek(start);
            Value::Raw {
                bytes: all.bytes(all.remaining()).unwrap_or_default().to_vec(),
                note: e.msg,
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ElemHint {
    /// Byte elements are FNames (enum) rather than u8.
    byte_is_enum: bool,
}

/// One value of a given type inside a container (or a top-level scalar).
fn read_elem(r: &mut Reader, ty: &TypeName, ctx: &mut Ctx, hint: ElemHint) -> Result<Value> {
    Ok(match ty.name.as_str() {
        "BoolProperty" => Value::Bool(r.u8()? != 0),
        "Int8Property" => Value::Int8(r.i8()?),
        "Int16Property" => Value::Int16(r.i16()?),
        "IntProperty" => Value::Int(r.i32()?),
        "Int64Property" => Value::Int64(r.i64()?),
        "ByteProperty" if hint.byte_is_enum || ty.inner_name().is_some() => Value::Enum(ctx.fname(r)?),
        "ByteProperty" => Value::Byte(r.u8()?),
        "UInt16Property" => Value::UInt16(r.u16()?),
        "UInt32Property" => Value::UInt32(r.u32()?),
        "UInt64Property" => Value::UInt64(r.u64()?),
        "FloatProperty" => Value::Float(r.f32()?),
        "DoubleProperty" => Value::Double(r.f64()?),
        "NameProperty" => Value::Name(ctx.fname(r)?),
        "EnumProperty" => Value::Enum(ctx.fname(r)?),
        "StrProperty" => Value::Str(r.fstring()?),
        "Utf8StrProperty" | "AnsiStrProperty" => Value::Str(r.utf8_string()?),
        "TextProperty" => Value::Text(read_text(r, ctx)?),
        "ObjectProperty" | "ClassProperty" | "WeakObjectProperty" | "InterfaceProperty" => {
            Value::Object(PackageIndex(r.i32()?))
        }
        "LazyObjectProperty" => Value::Guid(r.guid()?),
        "SoftObjectProperty" | "SoftClassProperty" | "AssetObjectProperty" | "AssetClassProperty" => {
            Value::SoftObject(read_soft_path(r, ctx)?)
        }
        "DelegateProperty" => Value::Delegate {
            object: PackageIndex(r.i32()?),
            function: ctx.fname(r)?,
        },
        "MulticastDelegateProperty" | "MulticastInlineDelegateProperty" | "MulticastSparseDelegateProperty" => {
            let n = r.count(12)?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((PackageIndex(r.i32()?), ctx.fname(r)?));
            }
            Value::MulticastDelegate(v)
        }
        "FieldPathProperty" => {
            let n = r.count(8)?;
            let mut path = Vec::with_capacity(n);
            for _ in 0..n {
                path.push(ctx.fname(r)?);
            }
            Value::FieldPath {
                path,
                owner: PackageIndex(r.i32()?),
            }
        }
        "StructProperty" => read_struct(r, ty.inner_name(), None, ctx)?,
        "ArrayProperty" => {
            let size = r.remaining();
            read_array(r, ty, size, ctx)?
        }
        "SetProperty" => read_set(r, ty, ctx)?,
        "MapProperty" => read_map(r, ty, ctx)?,
        "OptionalProperty" => {
            let inner = ty.param(0).cloned().unwrap_or_default();
            if r.bool32()? {
                read_elem(r, &inner, ctx, hint)?
            } else {
                Value::Raw {
                    bytes: Vec::new(),
                    note: "unset optional".into(),
                }
            }
        }
        other => return Err(r.err(format!("unknown property type {other}"))),
    })
}

fn read_soft_path(r: &mut Reader, ctx: &Ctx) -> Result<SoftPath> {
    if !ctx.v.filter_editor_only && ctx.v.ue5 >= ue5::ADD_SOFTOBJECTPATH_LIST && !ctx.soft_paths.is_empty() {
        let at = r.pos();
        let i = r.i32()?;
        return usize::try_from(i)
            .ok()
            .and_then(|i| ctx.soft_paths.get(i))
            .cloned()
            .ok_or_else(|| Error::new(at, format!("soft object path index {i} outside the list")));
    }
    SoftPath::read_raw(r, ctx.names, ctx.v)
}

fn read_text(r: &mut Reader, ctx: &Ctx) -> Result<Text> {
    let _flags = r.u32()?;
    let history = r.i8()?;
    Ok(match history {
        -1 => {
            let invariant = if ctx.v.editor_object.unwrap_or(0) >= custom::EDITOR_CULTURE_INVARIANT_TEXT && r.bool32()? {
                Some(r.fstring()?)
            } else {
                None
            };
            Text::None { invariant }
        }
        0 => Text::Base {
            namespace: r.fstring()?,
            key: r.fstring()?,
            source: r.fstring()?,
        },
        h => {
            // Only decodable when the text owns the rest of the reader.
            Text::Other {
                history: h,
                bytes: r.bytes(r.remaining())?.to_vec(),
            }
        }
    })
}

/// ArrayProperty: i32 count, then elements. Before UE5 1012 an array of
/// structs carries one inner tag naming the struct.
fn read_array(r: &mut Reader, ty: &TypeName, size: usize, ctx: &mut Ctx) -> Result<Value> {
    let inner = ty.param(0).cloned().unwrap_or_default();
    let n = r.count(0)?;
    let mut inner = inner;
    let mut inner_native = None;
    if inner.name == "StructProperty" && ctx.v.ue5 < ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME {
        let Some(t) = read_tag(r, ctx)? else {
            return Err(r.err("array of structs without inner tag"));
        };
        inner = t.ty;
        inner_native = t.native;
    }
    if inner.name == "ByteProperty" && inner.inner_name().is_none() {
        let byte_is_enum = n > 0 && size.saturating_sub(4) / n == 8;
        if !byte_is_enum {
            return Ok(Value::Bytes(r.bytes(n)?.to_vec()));
        }
    }
    let byte_is_enum = n > 0 && size.saturating_sub(4) / n == 8;
    if n > r.remaining() && !(inner.name == "BoolProperty") {
        return Err(r.err(format!("array count {n} larger than its data")));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        ctx.path.push(format!("[{i}]"));
        let v = if inner.name == "StructProperty" {
            read_struct(r, inner.inner_name(), inner_native, ctx)
        } else {
            read_elem(r, &inner, ctx, ElemHint { byte_is_enum })
        };
        ctx.path.pop();
        out.push(v?);
    }
    Ok(Value::Array(out))
}

fn read_set(r: &mut Reader, ty: &TypeName, ctx: &mut Ctx) -> Result<Value> {
    let inner = ty.param(0).cloned().unwrap_or_default();
    let removed = r.count(0)?;
    for _ in 0..removed {
        read_elem(r, &inner, ctx, ElemHint::default())?;
    }
    let n = r.count(1)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_elem(r, &inner, ctx, ElemHint::default())?);
    }
    Ok(Value::Set(out))
}

fn read_map(r: &mut Reader, ty: &TypeName, ctx: &mut Ctx) -> Result<Value> {
    let k = ty.param(0).cloned().unwrap_or_default();
    let v = ty.param(1).cloned().unwrap_or_default();
    let removed = r.count(0)?;
    for _ in 0..removed {
        read_elem(r, &k, ctx, ElemHint::default())?;
    }
    let n = r.count(1)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        ctx.path.push(format!("{{{i}}}"));
        let res = (|| -> Result<(Value, Value)> {
            Ok((read_elem(r, &k, ctx, ElemHint::default())?, read_elem(r, &v, ctx, ElemHint::default())?))
        })();
        ctx.path.pop();
        out.push(res?);
    }
    Ok(Value::Map(out))
}

/// A struct value. `native` comes from the 1012+ tag flags when known;
/// otherwise the struct name decides (known native structs vs tagged).
fn read_struct(r: &mut Reader, name: Option<&str>, native: Option<bool>, ctx: &mut Ctx) -> Result<Value> {
    let name = name.unwrap_or("");
    let decoded = read_native_struct(r, name, ctx)?;
    if let Some(fields) = decoded {
        return Ok(Value::Struct {
            name: name.to_string(),
            fields,
        });
    }
    if native == Some(true) {
        // Many native serializers (FNiagaraTypeDefinition, ...) just wrap
        // tagged properties: accept that reading when it uses every byte.
        let mut rr = r.clone();
        let n = ctx.problems.len();
        if let Ok(fields) = read_tagged_properties(&mut rr, ctx, false) {
            if rr.remaining() == 0 {
                *r = rr;
                return Ok(Value::Struct {
                    name: name.to_string(),
                    fields,
                });
            }
        }
        ctx.problems.truncate(n);
        return Err(r.err(format!("natively serialized struct {name} is not known to this reader")));
    }
    let fields = read_tagged_properties(r, ctx, false)?;
    Ok(Value::Struct {
        name: name.to_string(),
        fields,
    })
}

fn field(name: &str, ty: &str, value: Value) -> Property {
    Property {
        name: name.to_string(),
        ty: TypeName::new(ty),
        array_index: 0,
        value,
    }
}

/// Native (binary) struct layouts. `Ok(None)` means the struct is tagged.
fn read_native_struct(r: &mut Reader, name: &str, ctx: &mut Ctx) -> Result<Option<Vec<Property>>> {
    let lwc = ctx.lwc();
    let real = |r: &mut Reader, double: bool| -> Result<Value> {
        Ok(if double { Value::Double(r.f64()?) } else { Value::Float(r.f32()?) })
    };
    let reals = |r: &mut Reader, names: &[&str], double: bool| -> Result<Vec<Property>> {
        let ty = if double { "DoubleProperty" } else { "FloatProperty" };
        names.iter().map(|n| Ok(field(n, ty, real(r, double)?))).collect()
    };
    let ints = |r: &mut Reader, names: &[&str]| -> Result<Vec<Property>> {
        names.iter().map(|n| Ok(field(n, "IntProperty", Value::Int(r.i32()?)))).collect()
    };
    const XYZ: &[&str] = &["X", "Y", "Z"];
    const XY: &[&str] = &["X", "Y"];
    const XYZW: &[&str] = &["X", "Y", "Z", "W"];
    let fields = match name {
        "Vector" | "Vector_NetQuantize" | "Vector_NetQuantize10" | "Vector_NetQuantize100" | "Vector_NetQuantizeNormal" => {
            reals(r, XYZ, lwc)?
        }
        "Vector3f" => reals(r, XYZ, false)?,
        "Vector3d" => reals(r, XYZ, true)?,
        "Vector2D" => reals(r, XY, lwc)?,
        "Vector2f" => reals(r, XY, false)?,
        "Vector4" => reals(r, XYZW, lwc)?,
        "Vector4f" => reals(r, XYZW, false)?,
        "Vector4d" => reals(r, XYZW, true)?,
        "Quat" => reals(r, XYZW, lwc)?,
        "Quat4f" => reals(r, XYZW, false)?,
        "Plane" => reals(r, &["X", "Y", "Z", "W"], lwc)?,
        "Rotator" => reals(r, &["Pitch", "Yaw", "Roll"], lwc)?,
        "Rotator3f" => reals(r, &["Pitch", "Yaw", "Roll"], false)?,
        "LinearColor" => reals(r, &["R", "G", "B", "A"], false)?,
        "Color" => {
            let [b, g, rr, a] = r.array::<4>()?;
            vec![
                field("B", "ByteProperty", Value::Byte(b)),
                field("G", "ByteProperty", Value::Byte(g)),
                field("R", "ByteProperty", Value::Byte(rr)),
                field("A", "ByteProperty", Value::Byte(a)),
            ]
        }
        "Guid" => vec![field("Guid", "Guid", Value::Guid(r.guid()?))],
        "IntPoint" | "Int32Point" | "IntVector2" => ints(r, XY)?,
        "IntVector" => ints(r, XYZ)?,
        "IntVector4" => ints(r, XYZW)?,
        "DateTime" | "Timespan" => vec![field("Ticks", "Int64Property", Value::Int64(r.i64()?))],
        "FrameNumber" => vec![field("Value", "IntProperty", Value::Int(r.i32()?))],
        "Box" | "Box3f" | "Box3d" | "Box2D" | "Box2f" => {
            let double = match name {
                "Box3f" | "Box2f" => false,
                "Box3d" => true,
                _ => lwc,
            };
            let comps: &[&str] = if name.starts_with("Box2") { XY } else { XYZ };
            let min = Value::Struct {
                name: "Vector".into(),
                fields: reals(r, comps, double)?,
            };
            let max = Value::Struct {
                name: "Vector".into(),
                fields: reals(r, comps, double)?,
            };
            vec![
                field("Min", "StructProperty", min),
                field("Max", "StructProperty", max),
                field("IsValid", "BoolProperty", Value::Bool(r.u8()? != 0)),
            ]
        }
        "Sphere" => {
            let mut f = reals(r, XYZ, lwc)?;
            f.push(field("W", "", real(r, lwc)?));
            f
        }
        "Matrix" | "Matrix44f" | "Matrix44d" => {
            let double = match name {
                "Matrix44f" => false,
                "Matrix44d" => true,
                _ => lwc,
            };
            let mut m = Vec::with_capacity(16);
            for _ in 0..16 {
                m.push(real(r, double)?);
            }
            vec![field("M", "ArrayProperty", Value::Array(m))]
        }
        "RichCurveKey" => {
            let interp = r.u8()?;
            let tangent = r.u8()?;
            let weight = r.u8()?;
            let mut f = vec![
                field("InterpMode", "ByteProperty", Value::Byte(interp)),
                field("TangentMode", "ByteProperty", Value::Byte(tangent)),
                field("TangentWeightMode", "ByteProperty", Value::Byte(weight)),
            ];
            f.extend(reals(
                r,
                &["Time", "Value", "ArriveTangent", "ArriveTangentWeight", "LeaveTangent", "LeaveTangentWeight"],
                false,
            )?);
            f
        }
        "SimpleCurveKey" => reals(r, &["Time", "Value"], false)?,
        "SoftObjectPath" | "SoftClassPath" | "StringAssetReference" | "StringClassReference" => {
            vec![field("Path", "SoftObjectProperty", Value::SoftObject(read_soft_path(r, ctx)?))]
        }
        "GameplayTagContainer" => {
            let n = r.count(8)?;
            let mut tags = Vec::with_capacity(n);
            for _ in 0..n {
                tags.push(Value::Name(ctx.fname(r)?));
            }
            vec![field("GameplayTags", "ArrayProperty", Value::Array(tags))]
        }
        "PerPlatformFloat" | "PerPlatformInt" | "PerPlatformBool" | "PerPlatformFrameRate" => {
            let cooked = r.bool32()?;
            let read_default = |r: &mut Reader| -> Result<Value> {
                Ok(match name {
                    "PerPlatformFloat" => Value::Float(r.f32()?),
                    "PerPlatformInt" => Value::Int(r.i32()?),
                    "PerPlatformBool" => Value::Bool(r.bool32()?),
                    _ => Value::Raw {
                        bytes: r.bytes(8)?.to_vec(),
                        note: "FFrameRate".into(),
                    },
                })
            };
            let default = read_default(r)?;
            let mut f = vec![field("Default", "", default)];
            if !cooked {
                let n = r.count(8)?;
                let mut m = Vec::with_capacity(n);
                for _ in 0..n {
                    let k = Value::Name(ctx.fname(r)?);
                    m.push((k, read_default(r)?));
                }
                f.push(field("PerPlatform", "MapProperty", Value::Map(m)));
            }
            f
        }
        "PerQualityLevelInt" | "PerQualityLevelFloat" => {
            let _cooked = r.bool32()?;
            let read_default = |r: &mut Reader| -> Result<Value> {
                Ok(if name == "PerQualityLevelInt" { Value::Int(r.i32()?) } else { Value::Float(r.f32()?) })
            };
            let default = read_default(r)?;
            let n = r.count(8)?;
            let mut m = Vec::with_capacity(n);
            for _ in 0..n {
                let k = Value::Int(r.i32()?);
                m.push((k, read_default(r)?));
            }
            vec![field("Default", "", default), field("PerQuality", "MapProperty", Value::Map(m))]
        }
        "NiagaraVariableBase" | "NiagaraVariable" | "NiagaraVariableWithOffset" | "NiagaraDataChannelVariable"
            if ctx.v.niagara.unwrap_or(0) >= custom::NIAGARA_VARIABLES_USE_TYPE_DEF_REGISTRY =>
        {
            let var_name = ctx.fname(r)?;
            ctx.path.push("TypeDef".into());
            let type_def = read_tagged_properties(r, ctx, false);
            ctx.path.pop();
            let mut f = vec![
                field("Name", "NameProperty", Value::Name(var_name)),
                field(
                    "TypeDef",
                    "StructProperty",
                    Value::Struct {
                        name: "NiagaraTypeDefinition".into(),
                        fields: type_def?,
                    },
                ),
            ];
            match name {
                "NiagaraVariable" => {
                    let n = r.count(1)?;
                    f.push(field("VarData", "ArrayProperty", Value::Bytes(r.bytes(n)?.to_vec())));
                }
                "NiagaraVariableWithOffset" => f.push(field("Offset", "IntProperty", Value::Int(r.i32()?))),
                _ => {}
            }
            f
        }
        "EdGraphPinType" => read_pin_type(r, ctx)?,
        "NiagaraDataInterfaceGPUParamInfo" => {
            let niagara = ctx.v.niagara.unwrap_or(0);
            let symbol = r.fstring()?;
            let class = r.fstring()?;
            let mut f = vec![
                field("DataInterfaceHLSLSymbol", "StrProperty", Value::Str(symbol)),
                field("DIClassName", "StrProperty", Value::Str(class)),
            ];
            if niagara >= custom::NIAGARA_GENERATED_FUNCTIONS_IN_GPU_PARAM_INFO {
                let n = r.count(16)?;
                let mut funcs = Vec::with_capacity(n);
                for _ in 0..n {
                    let def = ctx.fname(r)?;
                    let inst = r.fstring()?;
                    let ns = r.count(16)?;
                    let mut specs = Vec::with_capacity(ns);
                    for _ in 0..ns {
                        specs.push((Value::Name(ctx.fname(r)?), Value::Name(ctx.fname(r)?)));
                    }
                    let mut ff = vec![
                        field("DefinitionName", "NameProperty", Value::Name(def)),
                        field("InstanceName", "StrProperty", Value::Str(inst)),
                        field("Specifiers", "MapProperty", Value::Map(specs)),
                    ];
                    if niagara >= custom::NIAGARA_VARIADIC_PARAMETERS_IN_GPU_FUNCTION_INFO {
                        for which in ["VariadicInputs", "VariadicOutputs"] {
                            let nv = r.count(12)?;
                            let mut vars = Vec::with_capacity(nv);
                            for _ in 0..nv {
                                let name = ctx.fname(r)?;
                                let ty = PackageIndex(r.i32()?);
                                vars.push(Value::Struct {
                                    name: "NiagaraVariableCommonReference".into(),
                                    fields: vec![
                                        field("Name", "NameProperty", Value::Name(name)),
                                        field("UnderlyingType", "ObjectProperty", Value::Object(ty)),
                                    ],
                                });
                            }
                            ff.push(field(which, "ArrayProperty", Value::Array(vars)));
                        }
                    }
                    if niagara >= custom::NIAGARA_USAGE_BITMASK_IN_GPU_FUNCTION_INFO {
                        ff.push(field("MiscUsageBitMask", "UInt16Property", Value::UInt16(r.u16()?)));
                    }
                    funcs.push(Value::Struct {
                        name: "NiagaraDataInterfaceGeneratedFunction".into(),
                        fields: ff,
                    });
                }
                f.push(field("GeneratedFunctions", "ArrayProperty", Value::Array(funcs)));
            }
            f
        }
        _ => return Ok(None),
    };
    Ok(Some(fields))
}

/// FEdGraphPinType (UEdGraphPin's type), UE5 layout.
pub fn read_pin_type(r: &mut Reader, ctx: &mut Ctx) -> Result<Vec<Property>> {
    let fw = ctx.v.framework_object.unwrap_or(i32::MAX);
    if fw < custom::FRAMEWORK_PINS_STORE_FNAME || fw < custom::FRAMEWORK_PIN_CONTAINER_TYPE {
        return Err(r.err("EdGraphPinType from before UE 4.19 is not supported"));
    }
    let category = ctx.fname(r)?;
    let sub_category = ctx.fname(r)?;
    let sub_object = PackageIndex(r.i32()?);
    let container = r.u8()?;
    let mut f = vec![
        field("PinCategory", "NameProperty", Value::Name(category)),
        field("PinSubCategory", "NameProperty", Value::Name(sub_category)),
        field("PinSubCategoryObject", "ObjectProperty", Value::Object(sub_object)),
        field("ContainerType", "ByteProperty", Value::Byte(container)),
    ];
    if container == 3 {
        // Map: FEdGraphTerminalType for the value.
        let vc = ctx.fname(r)?;
        let vs = ctx.fname(r)?;
        let vo = PackageIndex(r.i32()?);
        let vconst = r.bool32()?;
        let vweak = r.bool32()?;
        f.push(field(
            "PinValueType",
            "StructProperty",
            Value::Struct {
                name: "EdGraphTerminalType".into(),
                fields: vec![
                    field("TerminalCategory", "NameProperty", Value::Name(vc)),
                    field("TerminalSubCategory", "NameProperty", Value::Name(vs)),
                    field("TerminalSubCategoryObject", "ObjectProperty", Value::Object(vo)),
                    field("bTerminalIsConst", "BoolProperty", Value::Bool(vconst)),
                    field("bTerminalIsWeakPointer", "BoolProperty", Value::Bool(vweak)),
                ],
            },
        ));
    }
    let is_reference = r.bool32()?;
    let is_weak = r.bool32()?;
    // PinSubCategoryMemberReference: FSimpleMemberReference (tagged struct
    // in UE5: MemberParent object, MemberName, MemberGuid).
    let member_parent = PackageIndex(r.i32()?);
    let member_name = ctx.fname(r)?;
    let member_guid = r.guid()?;
    let is_const = r.bool32()?;
    let is_uobject_wrapper = r.bool32()?;
    f.extend([
        field("bIsReference", "BoolProperty", Value::Bool(is_reference)),
        field("bIsWeakPointer", "BoolProperty", Value::Bool(is_weak)),
        field(
            "PinSubCategoryMemberReference",
            "StructProperty",
            Value::Struct {
                name: "SimpleMemberReference".into(),
                fields: vec![
                    field("MemberParent", "ObjectProperty", Value::Object(member_parent)),
                    field("MemberName", "NameProperty", Value::Name(member_name)),
                    field("MemberGuid", "Guid", Value::Guid(member_guid)),
                ],
            },
        ),
        field("bIsConst", "BoolProperty", Value::Bool(is_const)),
        field("bIsUObjectWrapper", "BoolProperty", Value::Bool(is_uobject_wrapper)),
    ]);
    if ctx.v.ue5 >= ue5::LARGE_WORLD_COORDINATES {
        f.push(field(
            "bSerializeAsSinglePrecisionFloat",
            "BoolProperty",
            Value::Bool(r.bool32()?),
        ));
    }
    Ok(f)
}
/// Struct names of map keys and values for maps whose tags (before UE5
/// 1012) do not name them, by property name. Taken from 5.4 saves of the
/// same properties.
const MAP_STRUCTS: &[(&str, &str, &str)] = &[
    ("BoundPinNames", "Guid", ""),
    ("DataSetToParameters", "", "NiagaraParameters"),
    ("InputDescriptions", "NiagaraVariableBase", ""),
    ("OutputDescriptions", "NiagaraVariableBase", ""),
    ("ParameterGuidMapping", "NiagaraVariable", "Guid"),
    ("ParameterOffsets", "NiagaraVariable", ""),
    ("PinOutputToPinDefaultPersistentId", "Guid", "Guid"),
    ("ResolvedDIBindings", "NiagaraVariableBase", "NiagaraVariableBase"),
    ("UserParameterRedirects", "NiagaraVariable", "NiagaraVariable"),
    ("VariableToScriptVariable", "NiagaraVariable", ""),
];

/// A map with struct keys or values of unknown type (tags before 1012):
/// try the known struct names for this property first, then common ones,
/// and keep the first reading that uses exactly all the bytes.
fn read_map_guess(r: &mut Reader, prop: &str, ty: &TypeName, ctx: &mut Ctx) -> Result<Value> {
    let hint = MAP_STRUCTS.iter().find(|(n, _, _)| *n == prop);
    let cands = |slot: usize, hinted: Option<&'static str>| -> Vec<Option<&'static str>> {
        if ty.param(slot).map(|p| p.name.as_str()) != Some("StructProperty") || ty.param(slot).and_then(|p| p.inner_name()).is_some() {
            return vec![None];
        }
        let mut v: Vec<Option<&'static str>> = Vec::new();
        if let Some(h) = hinted.filter(|h| !h.is_empty()) {
            v.push(Some(h));
        }
        for c in ["Guid", "NiagaraVariable", "NiagaraVariableBase", "Vector", ""] {
            if !v.contains(&Some(c)) {
                v.push(Some(c));
            }
        }
        v
    };
    let kc = cands(0, hint.map(|h| h.1));
    let vc = cands(1, hint.map(|h| h.2));
    let mut last = None;
    for k in &kc {
        for v in &vc {
            let mut t = ty.clone();
            if let Some(k) = k {
                t.params[0].params = vec![TypeName::new(*k)];
            }
            if let Some(v) = v {
                t.params[1].params = vec![TypeName::new(*v)];
            }
            let mut rr = r.clone();
            let n = ctx.problems.len();
            match read_map(&mut rr, &t, ctx) {
                Ok(val) if rr.remaining() == 0 => {
                    *r = rr;
                    return Ok(val);
                }
                Ok(_) => last = Some(rr.err(format!("{} bytes left", rr.remaining()))),
                Err(e) => last = Some(e),
            }
            ctx.problems.truncate(n);
        }
    }
    Err(last.unwrap_or_else(|| r.err("no reading of the map fits")))
}
