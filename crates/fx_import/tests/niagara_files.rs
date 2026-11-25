//! The free Niagara test packages in testdata/niagara (fetched by
//! `sh testdata/niagara_assets.sh`, never committed). Skipped when absent.

use std::path::PathBuf;

use fx_import::{dump, find, DumpOptions, Package, Value};

fn files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/niagara");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: {} is absent (sh testdata/niagara_assets.sh)", dir.display());
        return Vec::new();
    };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "uasset"))
        .collect();
    v.sort();
    v
}

#[test]
fn every_test_file_reads_without_problems() {
    for f in files() {
        let bytes = std::fs::read(&f).unwrap();
        let pkg = Package::read(&bytes).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(pkg.problems.is_empty(), "{}: {:#?}", f.display(), &pkg.problems[..pkg.problems.len().min(10)]);
        for (i, e) in pkg.exports.iter().enumerate() {
            assert!(e.object.error.is_none(), "{} export {i}: {:?}", f.display(), e.object.error);
        }

        // One NiagaraSystem asset holding emitters and scripts.
        let sys = pkg.find_export_by_class("NiagaraSystem").unwrap_or_else(|| panic!("{}: no system", f.display()));
        assert!(pkg.exports[sys].is_asset);
        let handles = find(&pkg.exports[sys].object.properties, "EmitterHandles");
        let Some(Value::Array(handles)) = handles else { panic!("{}: no EmitterHandles", f.display()) };
        assert!(!handles.is_empty());
        assert!(pkg.find_export_by_class("NiagaraEmitter").is_some(), "{}", f.display());
        assert!(pkg.find_export_by_class("NiagaraScript").is_some(), "{}", f.display());

        let text = dump(&pkg, &DumpOptions::default());
        assert!(text.contains("NiagaraSystem"));
    }
}
