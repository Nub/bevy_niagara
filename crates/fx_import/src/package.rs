//! A whole editor package: summary, name map, imports, exports and each
//! export's tagged properties.

use crate::property::{read_tagged_properties, Ctx, Property, SoftPath};
use crate::pins::{is_graph_node_class, pins_supported, read_pins, Pin};
use crate::reader::{Error, Guid, Reader, Result};
use crate::summary::{custom, ue5, Summary};

/// FPackageIndex: > 0 is export `i - 1`, < 0 is import `-i - 1`, 0 is null.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PackageIndex(pub i32);

impl PackageIndex {
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
    pub fn export(self) -> Option<usize> {
        (self.0 > 0).then(|| self.0 as usize - 1)
    }
    pub fn import(self) -> Option<usize> {
        (self.0 < 0).then(|| (-(self.0 as i64) - 1) as usize)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Import {
    pub class_package: String,
    pub class_name: String,
    pub outer: PackageIndex,
    pub object_name: String,
    pub package_name: String,
    pub optional: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Export {
    pub class: PackageIndex,
    pub super_: PackageIndex,
    pub template: PackageIndex,
    pub outer: PackageIndex,
    pub object_name: String,
    pub object_flags: u32,
    pub serial_size: usize,
    pub serial_offset: usize,
    pub is_asset: bool,
    /// Tagged-property range relative to `serial_offset` (UE5 1010+).
    pub script_range: Option<(usize, usize)>,
    /// The export's data as far as this reader understands it.
    pub object: Object,
}

/// EObjectFlags::RF_ClassDefaultObject.
pub const RF_CLASS_DEFAULT_OBJECT: u32 = 0x10;

/// What was read of one export's serialized data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Object {
    pub properties: Vec<Property>,
    pub guid: Option<Guid>,
    /// Graph node pins (UEdGraphNode subclasses), read from the native data.
    pub pins: Vec<Pin>,
    /// Bytes after the tagged properties (and object GUID): the class's
    /// native serialization, not decoded here.
    pub native_size: usize,
    /// Absolute file offset where the native data starts.
    pub native_offset: usize,
    /// Set when the tagged properties could not be read at all.
    pub error: Option<String>,
}

/// A value that could not be decoded (kept as raw bytes) or a size mismatch.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    pub export: usize,
    pub path: String,
    pub msg: String,
}

#[derive(Clone, Debug)]
pub struct Package {
    pub summary: Summary,
    pub names: Vec<String>,
    pub imports: Vec<Import>,
    pub exports: Vec<Export>,
    pub soft_object_paths: Vec<SoftPath>,
    pub problems: Vec<Problem>,
}

/// Classes whose native Serialize writes data before the tagged properties
/// (UStruct and friends); for these only the raw size is reported unless
/// the export table gives the tagged-property range.
const NATIVE_FIRST_CLASSES: &[&str] = &[
    "Class",
    "Function",
    "ScriptStruct",
    "Enum",
    "UserDefinedEnum",
    "UserDefinedStruct",
    "BlueprintGeneratedClass",
    "WidgetBlueprintGeneratedClass",
    "AnimBlueprintGeneratedClass",
];

impl Package {
    pub fn read(data: &[u8]) -> Result<Package> {
        let mut r = Reader::new(data);
        let summary = Summary::read(&mut r)?;
        let ctx_versions = Versions::from_summary(&summary);

        // Name map.
        let mut r = r.window(summary.name_offset, data.len())?;
        let mut names = Vec::with_capacity(summary.name_count);
        for _ in 0..summary.name_count {
            names.push(r.fstring()?);
            r.u32()?; // case-insensitive + case-preserving hashes
        }
        let names_ref = &names;
        let fname = |r: &mut Reader| read_fname(r, names_ref);

        // Soft object path list (read raw; properties index into it).
        let mut soft_object_paths = Vec::new();
        if summary.soft_object_paths_count > 0 {
            let mut r = r.window(summary.soft_object_paths_offset, data.len())?;
            for _ in 0..summary.soft_object_paths_count {
                soft_object_paths.push(SoftPath::read_raw(&mut r, names_ref, &ctx_versions)?);
            }
        }

        // Imports.
        let mut r = r.window(summary.import_offset, data.len())?;
        let mut imports = Vec::with_capacity(summary.import_count);
        for _ in 0..summary.import_count {
            let class_package = fname(&mut r)?;
            let class_name = fname(&mut r)?;
            let outer = PackageIndex(r.i32()?);
            let object_name = fname(&mut r)?;
            let package_name = if !summary.filter_editor_only() {
                fname(&mut r)?
            } else {
                String::new()
            };
            let optional = summary.ue5_version >= ue5::OPTIONAL_RESOURCES && r.bool32()?;
            imports.push(Import {
                class_package,
                class_name,
                outer,
                object_name,
                package_name,
                optional,
            });
        }

        // Exports.
        let mut r = r.window(summary.export_offset, data.len())?;
        let mut exports = Vec::with_capacity(summary.export_count);
        for _ in 0..summary.export_count {
            let class = PackageIndex(r.i32()?);
            let super_ = PackageIndex(r.i32()?);
            let template = PackageIndex(r.i32()?);
            let outer = PackageIndex(r.i32()?);
            let object_name = fname(&mut r)?;
            let object_flags = r.u32()?;
            let at = r.pos();
            let serial_size = r.i64()?;
            let serial_offset = r.i64()?;
            if serial_size < 0 || serial_offset < 0 || (serial_offset + serial_size) as usize > data.len() {
                return Err(Error::new(
                    at,
                    format!("export {object_name}: data {serial_offset}+{serial_size} outside the file"),
                ));
            }
            r.bool32()?; // bForcedExport
            r.bool32()?; // bNotForClient
            r.bool32()?; // bNotForServer
            if summary.ue5_version < ue5::REMOVE_OBJECT_EXPORT_PACKAGE_GUID {
                r.guid()?;
            }
            if summary.ue5_version >= ue5::TRACK_OBJECT_EXPORT_IS_INHERITED {
                r.bool32()?; // bIsInheritedInstance
            }
            r.u32()?; // PackageFlags
            r.bool32()?; // bNotAlwaysLoadedForEditorGame
            let is_asset = r.bool32()?;
            if summary.ue5_version >= ue5::OPTIONAL_RESOURCES {
                r.bool32()?; // bGeneratePublicHash
            }
            for _ in 0..5 {
                r.i32()?; // preload dependency ranges
            }
            let script_range = if summary.ue5_version >= ue5::SCRIPT_SERIALIZATION_OFFSET {
                let start = r.i64()?;
                let end = r.i64()?;
                (0 <= start && start <= end && end <= serial_size).then_some((start as usize, end as usize))
            } else {
                None
            };
            exports.push(Export {
                class,
                super_,
                template,
                outer,
                object_name,
                object_flags,
                serial_size: serial_size as usize,
                serial_offset: serial_offset as usize,
                is_asset,
                script_range,
                object: Object::default(),
            });
        }

        let mut pkg = Package {
            summary,
            names,
            imports,
            exports,
            soft_object_paths,
            problems: Vec::new(),
        };
        let root = Reader::new(data);
        for i in 0..pkg.exports.len() {
            let (object, problems) = pkg.read_object(&root, i, &ctx_versions);
            pkg.exports[i].object = object;
            pkg.problems.extend(problems);
        }
        Ok(pkg)
    }

    fn read_object(&self, root: &Reader, i: usize, versions: &Versions) -> (Object, Vec<Problem>) {
        let e = &self.exports[i];
        let start = e.serial_offset;
        let end = e.serial_offset + e.serial_size;
        let mut obj = Object {
            native_offset: start,
            native_size: e.serial_size,
            ..Object::default()
        };
        let class = self.class_name(e.class);
        let (tag_start, tag_end) = match e.script_range {
            Some((s, t)) => (start + s, Some(start + t)),
            None if NATIVE_FIRST_CLASSES.contains(&class.as_str()) => return (obj, Vec::new()),
            // UAssetImportData saves its source-file JSON (an FString) before
            // the tagged properties; without the export's script range (UE4
            // packages) skip it here. Its subclasses (InterchangeAssetImportData,
            // FbxAssetImportData, ...) do the same.
            None if class.ends_with("AssetImportData") && !versions.filter_editor_only => {
                match root.window(start, end).and_then(|mut r| r.fstring().map(|_| r.pos())) {
                    Ok(p) => (p, None),
                    Err(err) => {
                        obj.error = Some(err.to_string());
                        return (obj, Vec::new());
                    }
                }
            }
            None => (start, None),
        };
        let mut ctx = Ctx::new(&self.names, &self.soft_object_paths, versions);
        let mut r = match root.window(tag_start, end) {
            Ok(r) => r,
            Err(err) => {
                obj.error = Some(err.to_string());
                return (obj, Vec::new());
            }
        };
        let is_cdo = e.object_flags & RF_CLASS_DEFAULT_OBJECT != 0;
        let res = (|| -> Result<()> {
            obj.properties = read_tagged_properties(&mut r, &mut ctx, true)?;
            if let Some(t) = tag_end {
                if r.pos() != t {
                    return Err(r.err(format!(
                        "tagged properties end at {:#x}, export table says {t:#x}",
                        r.pos()
                    )));
                }
            }
            if !is_cdo && r.remaining() >= 4 {
                if r.bool32()? {
                    obj.guid = Some(r.guid()?);
                }
            }
            if !is_cdo && is_graph_node_class(&class) && pins_supported(versions) && r.remaining() > 0 {
                let mut rr = r.clone();
                match read_pins(&mut rr, &mut ctx) {
                    Ok(p) => {
                        obj.pins = p;
                        r = rr;
                    }
                    Err(err) => ctx.problems.push(("pins".into(), err.to_string())),
                }
            }
            Ok(())
        })();
        match res {
            Ok(()) => {
                obj.native_offset = r.pos();
                obj.native_size = end - r.pos();
                if let Some((s, _)) = e.script_range {
                    // Native data written before the tagged properties counts too.
                    obj.native_size += s;
                }
            }
            Err(err) => obj.error = Some(err.to_string()),
        }
        let problems = ctx
            .problems
            .into_iter()
            .map(|(path, msg)| Problem { export: i, path, msg })
            .chain(obj.error.iter().map(|m| Problem {
                export: i,
                path: String::new(),
                msg: m.clone(),
            }))
            .collect();
        (obj, problems)
    }

    /// Class name of an object reference (import or export), e.g. "NiagaraEmitter".
    pub fn class_name(&self, class: PackageIndex) -> String {
        if let Some(i) = class.import() {
            self.imports.get(i).map(|im| im.object_name.clone()).unwrap_or_default()
        } else if let Some(i) = class.export() {
            self.exports.get(i).map(|ex| ex.object_name.clone()).unwrap_or_default()
        } else {
            "Class".into()
        }
    }

    /// Class of an export.
    pub fn export_class(&self, i: usize) -> String {
        self.class_name(self.exports[i].class)
    }

    /// Full object path of an import or export, e.g.
    /// "/Script/Niagara.NiagaraEmitter" or "NS_Fire.Emitter:Script".
    pub fn path(&self, idx: PackageIndex) -> String {
        let mut parts = Vec::new();
        let mut cur = idx;
        let mut guard = 0;
        while !cur.is_null() && guard < 64 {
            guard += 1;
            if let Some(i) = cur.import() {
                let Some(im) = self.imports.get(i) else { break };
                parts.push(im.object_name.clone());
                cur = im.outer;
            } else if let Some(i) = cur.export() {
                let Some(ex) = self.exports.get(i) else { break };
                parts.push(ex.object_name.clone());
                cur = ex.outer;
            }
        }
        parts.reverse();
        let mut s = String::new();
        for (k, p) in parts.iter().enumerate() {
            if k > 0 {
                s.push(if k == 1 { '.' } else { ':' });
            }
            s.push_str(p);
        }
        s
    }

    /// Index of the first export with this class name.
    pub fn find_export_by_class(&self, class: &str) -> Option<usize> {
        (0..self.exports.len()).find(|&i| self.export_class(i) == class)
    }
}

/// Versions that change how values are encoded.
#[derive(Clone, Debug, Default)]
pub struct Versions {
    pub ue5: i32,
    pub niagara: Option<i32>,
    pub editor_object: Option<i32>,
    pub fortnite_main: Option<i32>,
    pub framework_object: Option<i32>,
    pub blueprints_object: Option<i32>,
    pub ue5_main_stream: Option<i32>,
    pub ue5_release_stream: Option<i32>,
    pub filter_editor_only: bool,
}

impl Versions {
    pub fn from_summary(s: &Summary) -> Self {
        Versions {
            ue5: s.ue5_version,
            niagara: s.custom_version(custom::NIAGARA),
            editor_object: s.custom_version(custom::EDITOR_OBJECT),
            fortnite_main: s.custom_version(custom::FORTNITE_MAIN),
            framework_object: s.custom_version(custom::FRAMEWORK_OBJECT),
            blueprints_object: s.custom_version(custom::BLUEPRINTS_OBJECT),
            ue5_main_stream: s.custom_version(custom::UE5_MAIN_STREAM),
            ue5_release_stream: s.custom_version(custom::UE5_RELEASE_STREAM),
            filter_editor_only: s.filter_editor_only(),
        }
    }
}

/// FName: index into the name map plus a number (`Name_3` is number 4).
pub fn read_fname(r: &mut Reader, names: &[String]) -> Result<String> {
    let at = r.pos();
    let index = r.i32()?;
    let number = r.i32()?;
    let base = usize::try_from(index)
        .ok()
        .and_then(|i| names.get(i))
        .ok_or_else(|| Error::new(at, format!("name index {index} outside the name map ({})", names.len())))?;
    Ok(if number > 0 {
        format!("{base}_{}", number - 1)
    } else {
        base.clone()
    })
}
