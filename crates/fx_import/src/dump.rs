//! Readable text dump of a package: header, imports, and the export tree
//! (exports nested under their outers) with each export's properties.

use std::fmt::Write;

use crate::package::{Package, PackageIndex};
use crate::property::{Property, Text, Value};

#[derive(Clone, Debug)]
pub struct DumpOptions {
    /// Array elements shown before "... N more".
    pub max_elems: usize,
    /// Bytes shown of byte blobs.
    pub max_bytes: usize,
    pub imports: bool,
    /// Only exports whose class or name matches one of these (with their
    /// subtrees). Empty: all.
    pub only: Vec<String>,
}

impl Default for DumpOptions {
    fn default() -> Self {
        DumpOptions {
            max_elems: 32,
            max_bytes: 32,
            imports: true,
            only: Vec::new(),
        }
    }
}

pub fn dump(pkg: &Package, opts: &DumpOptions) -> String {
    let mut out = String::new();
    let s = &pkg.summary;
    let _ = writeln!(out, "Package {}", s.package_name);
    let _ = writeln!(
        out,
        "  saved by {} (file version UE4 {} / UE5 {}, licensee {})",
        s.saved_by, s.ue4_version, s.ue5_version, s.licensee_version
    );
    let _ = writeln!(
        out,
        "  flags {:#010x}, {} names, {} imports, {} exports, {} soft object paths, {} custom versions",
        s.package_flags,
        pkg.names.len(),
        pkg.imports.len(),
        pkg.exports.len(),
        pkg.soft_object_paths.len(),
        s.custom_versions.len()
    );
    if opts.imports {
        let _ = writeln!(out, "\nImports:");
        for (i, im) in pkg.imports.iter().enumerate() {
            let _ = writeln!(
                out,
                "  [{}] {} ({}.{})",
                -(i as i64) - 1,
                pkg.path(PackageIndex(-(i as i32) - 1)),
                im.class_package,
                im.class_name
            );
        }
    }
    let _ = writeln!(out, "\nExports:");
    // Children by outer.
    let n = pkg.exports.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (i, e) in pkg.exports.iter().enumerate() {
        let parent = e.outer.export().filter(|&p| p < n).map(|p| p + 1).unwrap_or(0);
        children[parent].push(i);
    }
    let mut stack: Vec<(usize, usize, bool)> = children[0].iter().rev().map(|&i| (i, 0, false)).collect();
    let mut seen = vec![false; n];
    while let Some((i, depth, shown_parent)) = stack.pop() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        let e = &pkg.exports[i];
        let class = pkg.export_class(i);
        let selected = shown_parent || opts.only.is_empty() || opts.only.iter().any(|o| *o == class || *o == e.object_name);
        if selected {
            let pad = "  ".repeat(depth + 1);
            let _ = writeln!(
                out,
                "{pad}[{}] {} {}  ({} bytes at {:#x}{}{})",
                i + 1,
                class,
                e.object_name,
                e.serial_size,
                e.serial_offset,
                if e.is_asset { ", asset" } else { "" },
                if e.object_flags & crate::package::RF_CLASS_DEFAULT_OBJECT != 0 { ", CDO" } else { "" },
            );
            if let Some(err) = &e.object.error {
                let _ = writeln!(out, "{pad}  ERROR: {err}");
            }
            for p in &e.object.properties {
                write_prop(&mut out, pkg, p, depth + 2, opts);
            }
            if e.object.native_size > 0 {
                let _ = writeln!(out, "{pad}  <native data: {} bytes>", e.object.native_size);
            }
        }
        for &c in children[i + 1].iter().rev() {
            stack.push((c, if selected { depth + 1 } else { depth }, selected && !opts.only.is_empty()));
        }
    }
    if !pkg.problems.is_empty() {
        let _ = writeln!(out, "\nProblems ({}):", pkg.problems.len());
        for p in &pkg.problems {
            let _ = writeln!(
                out,
                "  [{}] {} {}: {}",
                p.export + 1,
                pkg.exports[p.export].object_name,
                p.path,
                p.msg
            );
        }
    }
    out
}

fn write_prop(out: &mut String, pkg: &Package, p: &Property, depth: usize, opts: &DumpOptions) {
    let pad = "  ".repeat(depth);
    let idx = if p.array_index > 0 { format!("[{}]", p.array_index) } else { String::new() };
    let _ = write!(out, "{pad}{}{idx}: {} = ", p.name, p.ty);
    write_value(out, pkg, &p.value, depth, opts);
    out.push('\n');
}

fn write_value(out: &mut String, pkg: &Package, v: &Value, depth: usize, opts: &DumpOptions) {
    let pad = "  ".repeat(depth);
    match v {
        Value::Bool(b) => {
            let _ = write!(out, "{b}");
        }
        Value::Int8(x) => {
            let _ = write!(out, "{x}");
        }
        Value::Int16(x) => {
            let _ = write!(out, "{x}");
        }
        Value::Int(x) => {
            let _ = write!(out, "{x}");
        }
        Value::Int64(x) => {
            let _ = write!(out, "{x}");
        }
        Value::Byte(x) => {
            let _ = write!(out, "{x}");
        }
        Value::UInt16(x) => {
            let _ = write!(out, "{x}");
        }
        Value::UInt32(x) => {
            let _ = write!(out, "{x}");
        }
        Value::UInt64(x) => {
            let _ = write!(out, "{x}");
        }
        Value::Float(x) => {
            let _ = write!(out, "{x:?}");
        }
        Value::Double(x) => {
            let _ = write!(out, "{x:?}");
        }
        Value::Name(s) | Value::Enum(s) => out.push_str(s),
        Value::Str(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Text(t) => match t {
            Text::None { invariant: Some(s) } => {
                let _ = write!(out, "{s:?}");
            }
            Text::None { invariant: None } => out.push_str("\"\""),
            Text::Base { namespace, key, source } => {
                let _ = write!(out, "{source:?} (loc {namespace:?}/{key:?})");
            }
            Text::Other { history, bytes } => {
                let _ = write!(out, "<text history {history}, {} bytes>", bytes.len());
            }
        },
        Value::Object(i) => write_ref(out, pkg, *i),
        Value::SoftObject(s) => {
            let _ = write!(out, "soft {s}");
        }
        Value::Guid(g) => {
            let _ = write!(out, "{g}");
        }
        Value::Struct { name, fields } => {
            // Short native structs on one line.
            if fields.len() <= 4 && fields.iter().all(|f| f.value.as_f64().is_some()) {
                let _ = write!(out, "{name}(");
                for (k, f) in fields.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}=", f.name);
                    write_value(out, pkg, &f.value, depth, opts);
                }
                out.push(')');
            } else if fields.is_empty() {
                let _ = write!(out, "{name} {{}}");
            } else {
                let _ = writeln!(out, "{name} {{");
                for f in fields {
                    write_prop(out, pkg, f, depth + 1, opts);
                }
                let _ = write!(out, "{pad}}}");
            }
        }
        Value::Array(items) | Value::Set(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            let scalar = items.iter().all(|x| !matches!(x, Value::Struct { .. } | Value::Array(_) | Value::Map(_) | Value::Set(_)));
            if scalar && items.len() <= 8 {
                out.push('[');
                for (k, x) in items.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    write_value(out, pkg, x, depth, opts);
                }
                out.push(']');
                return;
            }
            let _ = writeln!(out, "[ ({} items)", items.len());
            for (k, x) in items.iter().take(opts.max_elems).enumerate() {
                let _ = write!(out, "{pad}  [{k}] ");
                write_value(out, pkg, x, depth + 1, opts);
                out.push('\n');
            }
            if items.len() > opts.max_elems {
                let _ = writeln!(out, "{pad}  ... {} more", items.len() - opts.max_elems);
            }
            let _ = write!(out, "{pad}]");
        }
        Value::Bytes(b) => write_bytes(out, b, opts),
        Value::Map(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            let _ = writeln!(out, "{{ ({} entries)", entries.len());
            for (k, (key, val)) in entries.iter().take(opts.max_elems).enumerate() {
                let _ = write!(out, "{pad}  {{{k}}} ");
                write_value(out, pkg, key, depth + 1, opts);
                out.push_str(" => ");
                write_value(out, pkg, val, depth + 1, opts);
                out.push('\n');
            }
            if entries.len() > opts.max_elems {
                let _ = writeln!(out, "{pad}  ... {} more", entries.len() - opts.max_elems);
            }
            let _ = write!(out, "{pad}}}");
        }
        Value::Delegate { object, function } => {
            write_ref(out, pkg, *object);
            let _ = write!(out, "::{function}");
        }
        Value::MulticastDelegate(v) => {
            out.push('[');
            for (k, (o, f)) in v.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_ref(out, pkg, *o);
                let _ = write!(out, "::{f}");
            }
            out.push(']');
        }
        Value::FieldPath { path, owner } => {
            let _ = write!(out, "{} in ", path.join("."));
            write_ref(out, pkg, *owner);
        }
        Value::Raw { bytes, note } => {
            let _ = write!(out, "<raw: {note}> ");
            write_bytes(out, bytes, opts);
        }
    }
}

fn write_ref(out: &mut String, pkg: &Package, i: PackageIndex) {
    if i.is_null() {
        out.push_str("null");
    } else if let Some(e) = i.export() {
        let _ = write!(out, "[{}] {}", e + 1, pkg.path(i));
    } else {
        let class = i.import().and_then(|k| pkg.imports.get(k)).map(|im| im.class_name.as_str()).unwrap_or("?");
        let _ = write!(out, "{class}'{}'", pkg.path(i));
    }
}

fn write_bytes(out: &mut String, b: &[u8], opts: &DumpOptions) {
    let _ = write!(out, "<{} bytes", b.len());
    if !b.is_empty() {
        out.push(':');
        for x in b.iter().take(opts.max_bytes) {
            let _ = write!(out, " {x:02x}");
        }
        if b.len() > opts.max_bytes {
            out.push_str(" ...");
        }
    }
    out.push('>');
}
