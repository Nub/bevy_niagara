//! Reader for Unreal Engine 5 editor packages (`.uasset` saved by the
//! editor, not cooked): package summary, name map, imports, exports and
//! each export's tagged properties (structs, arrays, sets, maps, native
//! math structs and the Niagara variable structs).
//!
//! Written from the public descriptions of the format and the open-source
//! readers CUE4Parse, UAssetAPI and unreal_asset (used as references, not
//! code). Cooked packages (unversioned properties, IoStore) are out of
//! scope.
//!
//! ```no_run
//! let bytes = std::fs::read("NS_Fire.uasset").unwrap();
//! let pkg = fx_import::Package::read(&bytes).unwrap();
//! print!("{}", fx_import::dump(&pkg, &Default::default()));
//! ```

pub mod convert;
pub mod dump;
pub mod fx;
pub mod hlsl;
pub mod niagara;
pub mod package;
pub mod pins;
pub mod property;
pub mod reader;
pub mod summary;
#[doc(hidden)]
pub mod synth;

pub use dump::{dump, DumpOptions};
pub use package::{Export, Import, Object, Package, PackageIndex, Problem};
pub use pins::{Direction, Pin, PinRef};
pub use property::{find, Property, SoftPath, Text, TypeName, Value};
pub use reader::{Error, Guid};
pub use summary::Summary;

/// What `fximport convert` produced.
pub struct Converted {
    pub ron_path: std::path::PathBuf,
    pub report_path: std::path::PathBuf,
    pub report: convert::Report,
}

impl Converted {
    /// One line: where it went and the coverage.
    pub fn summary(&self) -> String {
        format!(
            "{}: modules mapped {}/{} ({:.0}%), dynamic inputs {}/{}, {}; report {}",
            self.ron_path.display(),
            self.report.mapped(),
            self.report.total(),
            self.report.percent(),
            self.report.dynamic_mapped,
            self.report.dynamic_total,
            if self.report.validation.is_empty() { "validates".to_string() } else { format!("{} validation errors", self.report.validation.len()) },
            self.report_path.display()
        )
    }
}

/// Converts a Niagara system package to `<out>/<name>.fx.ron` and
/// `<out>/<name>.report.md` (name: the file's stem).
pub fn convert_file(file: &std::path::Path, out: &std::path::Path) -> Result<Converted, String> {
    let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
    let pkg = Package::read(&bytes).map_err(|e| e.to_string())?;
    let sys = niagara::read_system(&pkg)?;
    let source = file.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let conv = convert::convert(&sys, &source);
    let stem = file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| sys.name.clone());
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let ron_path = out.join(format!("{stem}.fx.ron"));
    let report_path = out.join(format!("{stem}.report.md"));
    std::fs::write(&ron_path, &conv.ron).map_err(|e| e.to_string())?;
    std::fs::write(&report_path, conv.report.markdown()).map_err(|e| e.to_string())?;
    Ok(Converted {
        ron_path,
        report_path,
        report: conv.report,
    })
}
