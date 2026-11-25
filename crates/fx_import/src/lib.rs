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

pub mod dump;
pub mod package;
pub mod property;
pub mod reader;
pub mod summary;
#[doc(hidden)]
pub mod synth;

pub use dump::{dump, DumpOptions};
pub use package::{Export, Import, Object, Package, PackageIndex, Problem};
pub use property::{find, Property, SoftPath, Text, TypeName, Value};
pub use reader::{Error, Guid};
pub use summary::Summary;
