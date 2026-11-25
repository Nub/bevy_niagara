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

/// One module input value per test file, read from a graph node pin
/// (checked by hand against the dump).
const KNOWN_PIN_DEFAULTS: &[(&str, &str, &str)] = &[
    ("NS_Boids", "Mesh Scale Mode", "NewEnumerator3"),
    ("NS_DestructibleRig", "Automatically Calculate Particle Radius", "true"),
    ("NS_Fire", "DynamicMaterialParameters.Index0Param0Write", "true"),
    ("NS_FireSmokeOnly", "AddVelocity.UseConeFalloff", "true"),
    ("NS_Fish", "Color Mode", "NewEnumerator2"),
    ("NS_FluidSimulation", "Accumulate_External_Forces.EnableGravity", "true"),
    ("NS_GeyserEruption", "Cone Mode", "NewEnumerator1"),
    ("NS_ShinyGlimmer", "Color Mode", "NewEnumerator1"),
];

#[test]
fn graph_pins_hold_known_defaults_and_links_resolve() {
    for f in files() {
        let stem = f.file_stem().unwrap().to_string_lossy().to_string();
        let pkg = Package::read(&std::fs::read(&f).unwrap()).unwrap();
        let pins = || pkg.exports.iter().flat_map(|e| e.object.pins.iter());
        assert!(pins().count() > 50, "{stem}: few pins");

        if let Some((_, pin, want)) = KNOWN_PIN_DEFAULTS.iter().find(|(n, _, _)| *n == stem) {
            assert!(
                pins().any(|p| p.name == *pin && p.default_value == *want),
                "{stem}: no pin {pin:?} = {want:?}"
            );
        }

        // Every link names a pin that exists on the linked node, and links
        // are symmetric.
        for (i, e) in pkg.exports.iter().enumerate() {
            for p in &e.object.pins {
                for l in &p.linked_to {
                    let node = l.node.export().expect("link to an export");
                    let other = pkg.exports[node].object.pins.iter().find(|q| q.id == l.pin);
                    let other = other.unwrap_or_else(|| panic!("{stem}: dangling link from [{}] {:?}", i + 1, p.name));
                    assert!(
                        other.linked_to.iter().any(|b| b.pin == p.id && b.node.export() == Some(i)),
                        "{stem}: one-way link {:?} -> {:?}",
                        p.name,
                        other.name
                    );
                }
            }
        }
    }
}
