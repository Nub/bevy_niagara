//! Writer for small synthetic editor packages, used by the tests. It writes
//! the same layout the reader expects for a chosen UE5 object version, so
//! both property tag layouts (before and from UE5 1012) can be exercised
//! without third-party files.

use crate::reader::Guid;
use crate::summary::{custom, ue5, PACKAGE_FILE_TAG};

#[derive(Clone, Debug)]
pub struct Synth {
    pub ue5: i32,
    pub custom_versions: Vec<(Guid, i32)>,
    pub names: Vec<String>,
    imports: Vec<[i32; 4]>,
    exports: Vec<SynthExport>,
}

#[derive(Clone, Debug)]
struct SynthExport {
    class: i32,
    outer: i32,
    name: i32,
    flags: u32,
    props: Vec<u8>,
    native: Vec<u8>,
}

impl Synth {
    /// A package for UE5 object version `ue5` (1000..=1018), with the
    /// Niagara custom version new enough for native Niagara variables.
    pub fn new(ue5: i32) -> Self {
        let mut s = Synth {
            ue5,
            custom_versions: vec![(custom::NIAGARA, custom::NIAGARA_VARIABLES_USE_TYPE_DEF_REGISTRY + 1)],
            names: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
        };
        s.name("None");
        s
    }

    pub fn name(&mut self, n: &str) -> i32 {
        match self.names.iter().position(|x| x == n) {
            Some(i) => i as i32,
            None => {
                self.names.push(n.to_string());
                self.names.len() as i32 - 1
            }
        }
    }

    /// Add an import; returns its package index (negative).
    pub fn import(&mut self, class_package: &str, class: &str, outer: i32, name: &str) -> i32 {
        let e = [self.name(class_package), self.name(class), outer, self.name(name)];
        self.imports.push(e);
        -(self.imports.len() as i32)
    }

    /// Start a property list.
    pub fn props(&mut self) -> Props<'_> {
        Props {
            s: self,
            buf: Vec::new(),
        }
    }

    /// Add an export with tagged property bytes (from `Props::finish`) and
    /// trailing native bytes; returns its package index (positive).
    pub fn export(&mut self, class: i32, outer: i32, name: &str, flags: u32, props: Vec<u8>, native: Vec<u8>) -> i32 {
        let name = self.name(name);
        self.exports.push(SynthExport {
            class,
            outer,
            name,
            flags,
            props,
            native,
        });
        self.exports.len() as i32
    }

    fn object_bytes(&self, e: &SynthExport) -> (Vec<u8>, usize, usize) {
        let mut b = Vec::new();
        if self.ue5 >= ue5::PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION {
            b.push(0); // serialization control extension
        }
        b.extend_from_slice(&e.props);
        let script_end = b.len();
        if e.flags & crate::package::RF_CLASS_DEFAULT_OBJECT == 0 {
            b.extend_from_slice(&0u32.to_le_bytes()); // no object GUID
        }
        b.extend_from_slice(&e.native);
        (b, 0, script_end)
    }

    pub fn build(&self) -> Vec<u8> {
        let bodies: Vec<_> = self.exports.iter().map(|e| self.object_bytes(e)).collect();
        let layout = |summary_len: usize| {
            let names_off = summary_len;
            let names_len: usize = self.names.iter().map(|n| 4 + n.len() + 1 + 4).sum();
            let imports_off = names_off + names_len;
            let import_len = 8 + 8 + 4 + 8 + 8 + if self.ue5 >= ue5::OPTIONAL_RESOURCES { 4 } else { 0 };
            let exports_off = imports_off + import_len * self.imports.len();
            let export_len = 4 * 4 + 8 + 4 + 16 + 12
                + if self.ue5 < ue5::REMOVE_OBJECT_EXPORT_PACKAGE_GUID { 16 } else { 0 }
                + if self.ue5 >= ue5::TRACK_OBJECT_EXPORT_IS_INHERITED { 4 } else { 0 }
                + 4 + 8
                + if self.ue5 >= ue5::OPTIONAL_RESOURCES { 4 } else { 0 }
                + 20
                + if self.ue5 >= ue5::SCRIPT_SERIALIZATION_OFFSET { 16 } else { 0 };
            let data_off = exports_off + export_len * self.exports.len();
            (names_off, imports_off, exports_off, data_off)
        };
        let probe = self.summary(0, 0, 0, 0);
        let (names_off, imports_off, exports_off, data_off) = layout(probe.len());
        let mut out = self.summary(names_off, imports_off, exports_off, data_off);
        assert_eq!(out.len(), names_off);
        for n in &self.names {
            fstring(&mut out, n);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        assert_eq!(out.len(), imports_off);
        for [cp, c, outer, name] in &self.imports {
            fname(&mut out, *cp);
            fname(&mut out, *c);
            i32w(&mut out, *outer);
            fname(&mut out, *name);
            fname(&mut out, 0); // PackageName = None
            if self.ue5 >= ue5::OPTIONAL_RESOURCES {
                i32w(&mut out, 0);
            }
        }
        assert_eq!(out.len(), exports_off);
        let mut off = data_off;
        for (e, (body, s, t)) in self.exports.iter().zip(&bodies) {
            i32w(&mut out, e.class);
            i32w(&mut out, 0); // super
            i32w(&mut out, 0); // template
            i32w(&mut out, e.outer);
            fname(&mut out, e.name);
            out.extend_from_slice(&e.flags.to_le_bytes());
            out.extend_from_slice(&(body.len() as i64).to_le_bytes());
            out.extend_from_slice(&(off as i64).to_le_bytes());
            for _ in 0..3 {
                i32w(&mut out, 0); // forced export, not for client/server
            }
            if self.ue5 < ue5::REMOVE_OBJECT_EXPORT_PACKAGE_GUID {
                out.extend_from_slice(&[0; 16]);
            }
            if self.ue5 >= ue5::TRACK_OBJECT_EXPORT_IS_INHERITED {
                i32w(&mut out, 0);
            }
            i32w(&mut out, 0); // package flags
            i32w(&mut out, 0); // not always loaded for editor game
            i32w(&mut out, (e.outer == 0) as i32); // is asset
            if self.ue5 >= ue5::OPTIONAL_RESOURCES {
                i32w(&mut out, 0);
            }
            for _ in 0..5 {
                i32w(&mut out, -1);
            }
            if self.ue5 >= ue5::SCRIPT_SERIALIZATION_OFFSET {
                out.extend_from_slice(&(*s as i64).to_le_bytes());
                out.extend_from_slice(&(*t as i64).to_le_bytes());
            }
            off += body.len();
        }
        assert_eq!(out.len(), data_off);
        for (body, _, _) in &bodies {
            out.extend_from_slice(body);
        }
        out
    }

    fn summary(&self, names_off: usize, imports_off: usize, exports_off: usize, data_off: usize) -> Vec<u8> {
        let v5 = self.ue5;
        let mut b = Vec::new();
        b.extend_from_slice(&PACKAGE_FILE_TAG.to_le_bytes());
        i32w(&mut b, if v5 >= ue5::PACKAGE_SAVED_HASH { -9 } else { -8 });
        i32w(&mut b, 864);
        i32w(&mut b, 522);
        i32w(&mut b, v5);
        i32w(&mut b, 0);
        if v5 >= ue5::PACKAGE_SAVED_HASH {
            b.extend_from_slice(&[0; 20]);
            i32w(&mut b, data_off as i32);
        }
        i32w(&mut b, self.custom_versions.len() as i32);
        for (g, v) in &self.custom_versions {
            for w in g.0 {
                b.extend_from_slice(&w.to_le_bytes());
            }
            i32w(&mut b, *v);
        }
        if v5 < ue5::PACKAGE_SAVED_HASH {
            i32w(&mut b, data_off as i32);
        }
        fstring(&mut b, "/Game/Synth");
        i32w(&mut b, 0); // package flags
        i32w(&mut b, self.names.len() as i32);
        i32w(&mut b, names_off as i32);
        if v5 >= ue5::ADD_SOFTOBJECTPATH_LIST {
            i32w(&mut b, 0);
            i32w(&mut b, 0);
        }
        fstring(&mut b, ""); // localization id
        i32w(&mut b, 0); // gatherable text
        i32w(&mut b, 0);
        i32w(&mut b, self.exports.len() as i32);
        i32w(&mut b, exports_off as i32);
        i32w(&mut b, self.imports.len() as i32);
        i32w(&mut b, imports_off as i32);
        if v5 >= ue5::VERSE_CELLS {
            for _ in 0..4 {
                i32w(&mut b, 0);
            }
        }
        if v5 >= ue5::METADATA_SERIALIZATION_OFFSET {
            i32w(&mut b, 0);
        }
        for _ in 0..5 {
            i32w(&mut b, 0); // depends, soft package refs (2), searchable names, thumbnails
        }
        if v5 >= ue5::IMPORT_TYPE_HIERARCHIES {
            i32w(&mut b, 0);
            i32w(&mut b, 0);
        }
        if v5 < ue5::PACKAGE_SAVED_HASH {
            b.extend_from_slice(&[0; 16]);
        }
        b.extend_from_slice(&[0; 16]); // persistent guid
        i32w(&mut b, 1); // generations
        i32w(&mut b, self.exports.len() as i32);
        i32w(&mut b, self.names.len() as i32);
        for _ in 0..2 {
            b.extend_from_slice(&5u16.to_le_bytes());
            b.extend_from_slice(&4u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            fstring(&mut b, "++UE5+Synth");
        }
        i32w(&mut b, 0); // compression flags
        i32w(&mut b, 0); // chunks
        i32w(&mut b, 0); // package source
        i32w(&mut b, 0); // additional packages to cook
        i32w(&mut b, 0); // asset registry
        b.extend_from_slice(&(data_off as i64).to_le_bytes()); // bulk data start
        i32w(&mut b, 0); // world tile info
        i32w(&mut b, 0); // chunk ids
        i32w(&mut b, 0); // preload deps
        i32w(&mut b, 0);
        if v5 >= ue5::NAMES_REFERENCED_FROM_EXPORT_DATA {
            i32w(&mut b, self.names.len() as i32);
        }
        if v5 >= ue5::PAYLOAD_TOC {
            b.extend_from_slice(&0i64.to_le_bytes());
        }
        if v5 >= ue5::DATA_RESOURCES {
            i32w(&mut b, -1);
        }
        b
    }
}

/// A graph node pin for [`Synth::pins`]: links name (node export index, pin id).
#[derive(Clone, Debug, Default)]
pub struct SynthPin {
    pub id: u32,
    pub name: String,
    pub output: bool,
    /// PinSubCategoryObject (the Niagara type struct), a package index.
    pub type_object: i32,
    pub default_value: String,
    pub links: Vec<(i32, u32)>,
}

impl SynthPin {
    pub fn new(id: u32, name: &str, output: bool, type_object: i32) -> Self {
        SynthPin {
            id,
            name: name.to_string(),
            output,
            type_object,
            ..Default::default()
        }
    }
    pub fn default(mut self, v: &str) -> Self {
        self.default_value = v.to_string();
        self
    }
    pub fn link(mut self, node: i32, pin: u32) -> Self {
        self.links.push((node, pin));
        self
    }
}

fn pin_guid(b: &mut Vec<u8>, id: u32) {
    for w in [id, 0x5eed, 0, 0] {
        b.extend_from_slice(&w.to_le_bytes());
    }
}

impl Synth {
    /// Adds the custom versions graph node pins need (UE 4.19+ layout).
    pub fn with_graph_versions(mut self) -> Self {
        self.custom_versions.push((custom::BLUEPRINTS_OBJECT, custom::BLUEPRINTS_EDGRAPH_PIN_OPTIMIZED));
        self.custom_versions.push((custom::FRAMEWORK_OBJECT, custom::FRAMEWORK_PINS_STORE_FNAME));
        self
    }

    /// Native bytes of a graph node's pin array, owned by export `node`.
    pub fn pins(&mut self, node: i32, pins: &[SynthPin]) -> Vec<u8> {
        let mut b = Vec::new();
        i32w(&mut b, pins.len() as i32);
        for p in pins {
            i32w(&mut b, 0); // not null
            for _ in 0..2 {
                i32w(&mut b, node);
                pin_guid(&mut b, p.id);
            }
            let n = self.name(&p.name);
            fname(&mut b, n);
            // FriendlyName: empty text without history.
            i32w(&mut b, 0);
            b.push(0xff);
            fstring(&mut b, ""); // tooltip
            b.push(p.output as u8);
            let cat = self.name("Type");
            fname(&mut b, cat);
            fname(&mut b, 0); // sub-category None
            i32w(&mut b, p.type_object);
            b.push(0); // container
            i32w(&mut b, 0); // is reference
            i32w(&mut b, 0); // is weak
            i32w(&mut b, 0); // member parent
            fname(&mut b, 0);
            b.extend_from_slice(&[0; 16]);
            i32w(&mut b, 0); // const
            i32w(&mut b, 0); // uobject wrapper
            fstring(&mut b, &p.default_value);
            fstring(&mut b, "");
            i32w(&mut b, 0); // default object
            i32w(&mut b, 0);
            b.push(0xff); // default text
            i32w(&mut b, p.links.len() as i32);
            for (n, id) in &p.links {
                i32w(&mut b, 0);
                i32w(&mut b, *n);
                pin_guid(&mut b, *id);
            }
            i32w(&mut b, 0); // sub pins
            i32w(&mut b, 1); // parent pin: null
            i32w(&mut b, 1); // reference pass-through: null
            b.extend_from_slice(&[0; 16]); // persistent guid
            i32w(&mut b, 0); // bits
        }
        b
    }
}

/// A property type for the writer: `T::Struct("Vector")`, `T::Array(..)`.
#[derive(Clone, Debug)]
pub enum T {
    Int,
    Float,
    Bool,
    Name,
    Str,
    Object,
    /// ByteProperty without an enum (byte arrays).
    Byte,
    /// ByteProperty with an enum name (value is an FName).
    ByteEnum(&'static str),
    Struct(&'static str),
    Array(Box<T>),
    Map(Box<T>, Box<T>),
}

impl T {
    fn type_name(&self) -> &'static str {
        match self {
            T::Int => "IntProperty",
            T::Float => "FloatProperty",
            T::Bool => "BoolProperty",
            T::Name => "NameProperty",
            T::Str => "StrProperty",
            T::Object => "ObjectProperty",
            T::Byte => "ByteProperty",
            T::ByteEnum(_) => "ByteProperty",
            T::Struct(_) => "StructProperty",
            T::Array(_) => "ArrayProperty",
            T::Map(..) => "MapProperty",
        }
    }
}

/// Tagged property writer.
pub struct Props<'a> {
    s: &'a mut Synth,
    buf: Vec<u8>,
}

impl Props<'_> {
    pub fn name_index(&mut self, n: &str) -> i32 {
        self.s.name(n)
    }

    /// Bytes of an FName value.
    pub fn fname(&mut self, n: &str) -> Vec<u8> {
        let mut b = Vec::new();
        fname(&mut b, self.s.name(n));
        b
    }

    /// A property with an already-encoded value.
    pub fn raw(&mut self, name: &str, ty: &T, value: &[u8], bool_value: bool, native: bool) -> &mut Self {
        let n = self.s.name(name);
        let mut b = Vec::new();
        fname(&mut b, n);
        if self.s.ue5 >= ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME {
            self.type_nodes(&mut b, ty);
            i32w(&mut b, value.len() as i32);
            let mut flags = 0u8;
            if bool_value {
                flags |= 0x10;
            }
            if native {
                flags |= 0x08;
            }
            b.push(flags);
        } else {
            let t = self.s.name(ty.type_name());
            fname(&mut b, t);
            i32w(&mut b, value.len() as i32);
            i32w(&mut b, 0); // array index
            self.old_tag_data(&mut b, ty, bool_value);
            b.push(0); // no property guid
            if self.s.ue5 >= ue5::PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION {
                b.push(0);
            }
        }
        b.extend_from_slice(value);
        self.buf.extend_from_slice(&b);
        self
    }

    fn type_nodes(&mut self, b: &mut Vec<u8>, ty: &T) {
        let n = self.s.name(ty.type_name());
        fname(b, n);
        match ty {
            T::Struct(s) | T::ByteEnum(s) => {
                i32w(b, 1);
                let sn = self.s.name(s);
                fname(b, sn);
                i32w(b, 1);
                let p = self.s.name("/Script/CoreUObject");
                fname(b, p);
                i32w(b, 0);
            }
            T::Array(inner) => {
                i32w(b, 1);
                self.type_nodes(b, inner);
            }
            T::Map(k, v) => {
                i32w(b, 2);
                self.type_nodes(b, k);
                self.type_nodes(b, v);
            }
            _ => i32w(b, 0),
        }
    }

    fn old_tag_data(&mut self, b: &mut Vec<u8>, ty: &T, bool_value: bool) {
        match ty {
            T::Struct(s) => {
                let sn = self.s.name(s);
                fname(b, sn);
                b.extend_from_slice(&[0; 16]);
            }
            T::Bool => b.push(bool_value as u8),
            T::ByteEnum(e) => {
                let en = self.s.name(e);
                fname(b, en);
            }
            T::Array(inner) => {
                let n = self.s.name(inner.type_name());
                fname(b, n);
            }
            T::Map(k, v) => {
                let kn = self.s.name(k.type_name());
                fname(b, kn);
                let vn = self.s.name(v.type_name());
                fname(b, vn);
            }
            _ => {}
        }
    }

    pub fn int(&mut self, name: &str, v: i32) -> &mut Self {
        self.raw(name, &T::Int, &v.to_le_bytes(), false, false)
    }
    pub fn float(&mut self, name: &str, v: f32) -> &mut Self {
        self.raw(name, &T::Float, &v.to_le_bytes(), false, false)
    }
    pub fn bool(&mut self, name: &str, v: bool) -> &mut Self {
        if self.s.ue5 >= ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME {
            self.raw(name, &T::Bool, &[], v, false)
        } else {
            self.raw(name, &T::Bool, &[], v, false)
        }
    }
    pub fn name(&mut self, name: &str, v: &str) -> &mut Self {
        let b = self.fname(v);
        self.raw(name, &T::Name, &b, false, false)
    }
    pub fn str(&mut self, name: &str, v: &str) -> &mut Self {
        let mut b = Vec::new();
        fstring(&mut b, v);
        self.raw(name, &T::Str, &b, false, false)
    }
    pub fn object(&mut self, name: &str, v: i32) -> &mut Self {
        self.raw(name, &T::Object, &v.to_le_bytes(), false, false)
    }
    pub fn byte_enum(&mut self, name: &str, en: &'static str, v: &str) -> &mut Self {
        let b = self.fname(v);
        self.raw(name, &T::ByteEnum(en), &b, false, false)
    }
    /// FVector (doubles from UE5 1004).
    pub fn vector(&mut self, name: &str, v: [f64; 3]) -> &mut Self {
        let b = self.vector_bytes(v);
        self.raw(name, &T::Struct("Vector"), &b, false, true)
    }
    pub fn vector_bytes(&self, v: [f64; 3]) -> Vec<u8> {
        let mut b = Vec::new();
        for c in v {
            if self.s.ue5 >= ue5::LARGE_WORLD_COORDINATES {
                b.extend_from_slice(&c.to_le_bytes());
            } else {
                b.extend_from_slice(&(c as f32).to_le_bytes());
            }
        }
        b
    }
    pub fn linear_color(&mut self, name: &str, c: [f32; 4]) -> &mut Self {
        let b: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.raw(name, &T::Struct("LinearColor"), &b, false, true)
    }
    /// A tagged struct whose fields are written by `f` (terminator added).
    pub fn tagged_struct(&mut self, name: &str, struct_name: &'static str, f: impl FnOnce(&mut Props)) -> &mut Self {
        let body = self.nested(f);
        self.raw(name, &T::Struct(struct_name), &body, false, false)
    }
    /// Bytes of a nested tagged property list.
    pub fn nested(&mut self, f: impl FnOnce(&mut Props)) -> Vec<u8> {
        let mut inner = Props {
            s: &mut *self.s,
            buf: Vec::new(),
        };
        f(&mut inner);
        inner.finish()
    }
    pub fn int_array(&mut self, name: &str, v: &[i32]) -> &mut Self {
        let mut b = Vec::new();
        i32w(&mut b, v.len() as i32);
        for x in v {
            i32w(&mut b, *x);
        }
        self.raw(name, &T::Array(Box::new(T::Int)), &b, false, false)
    }
    /// Array of structs; each element is its already-encoded bytes.
    pub fn struct_array(&mut self, name: &str, struct_name: &'static str, elems: &[Vec<u8>]) -> &mut Self {
        let mut b = Vec::new();
        i32w(&mut b, elems.len() as i32);
        let data: Vec<u8> = elems.concat();
        if self.s.ue5 < ue5::PROPERTY_TAG_COMPLETE_TYPE_NAME {
            // Inner tag: name = array name, type StructProperty, size of
            // the element data, struct name.
            let n = self.s.name(name);
            fname(&mut b, n);
            let t = self.s.name("StructProperty");
            fname(&mut b, t);
            i32w(&mut b, data.len() as i32);
            i32w(&mut b, 0);
            let sn = self.s.name(struct_name);
            fname(&mut b, sn);
            b.extend_from_slice(&[0; 16]);
            b.push(0);
            if self.s.ue5 >= ue5::PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION {
                b.push(0);
            }
        }
        b.extend_from_slice(&data);
        self.raw(name, &T::Array(Box::new(T::Struct(struct_name))), &b, false, false)
    }
    /// Array of object references.
    pub fn object_array(&mut self, name: &str, v: &[i32]) -> &mut Self {
        let mut b = Vec::new();
        i32w(&mut b, v.len() as i32);
        for x in v {
            i32w(&mut b, *x);
        }
        self.raw(name, &T::Array(Box::new(T::Object)), &b, false, false)
    }
    /// Array of bytes (ArrayProperty(ByteProperty)).
    pub fn byte_array(&mut self, name: &str, v: &[u8]) -> &mut Self {
        let mut b = Vec::new();
        i32w(&mut b, v.len() as i32);
        b.extend_from_slice(v);
        self.raw(name, &T::Array(Box::new(T::Byte)), &b, false, false)
    }
    pub fn name_int_map(&mut self, name: &str, entries: &[(&str, i32)]) -> &mut Self {
        let mut b = Vec::new();
        i32w(&mut b, 0); // keys to remove
        i32w(&mut b, entries.len() as i32);
        for (k, v) in entries {
            let kn = self.s.name(k);
            fname(&mut b, kn);
            i32w(&mut b, *v);
        }
        self.raw(name, &T::Map(Box::new(T::Name), Box::new(T::Int)), &b, false, false)
    }

    /// The property list with its "None" terminator.
    pub fn finish(mut self) -> Vec<u8> {
        fname(&mut self.buf, 0);
        self.buf
    }
}

fn i32w(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn fname(b: &mut Vec<u8>, index: i32) {
    i32w(b, index);
    i32w(b, 0);
}

pub fn fstring(b: &mut Vec<u8>, s: &str) {
    if s.is_empty() {
        i32w(b, 0);
        return;
    }
    i32w(b, s.len() as i32 + 1);
    b.extend_from_slice(s.as_bytes());
    b.push(0);
}
