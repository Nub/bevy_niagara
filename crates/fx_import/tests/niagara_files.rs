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

/// Every test file converts to an effect that validates, with most of its
/// modules mapped, and known values come through converted.
#[test]
fn every_test_file_converts_and_validates() {
    // (file, least % of modules mapped, text the effect must contain)
    let expect: &[(&str, f32, &[&str])] = &[
        ("NS_FireSmokeOnly", 100.0, &["Link(Float, \"User.SpawnRate\")", "\"Sphere Radius\": Float(0.08)"]),
        ("NS_Fire", 90.0, &["\"Shape Primitive\": Cylinder", "name: \"Sparks\""]),
        ("NS_Fish", 90.0, &["\"Shape Primitive\": Box", "Mesh((mesh: "]),
        ("NS_ShinyGlimmer", 90.0, &["Add Velocity from Point"]),
        ("NS_GeyserEruption", 90.0, &["\"Shape Primitive\": Cone", "Wind Force"]),
        ("NS_Boids", 80.0, &["Set Particle Velocity"]),
        ("NS_FluidSimulation", 50.0, &["Link(Int, \"User.ParticleNum\")", "Spawn Burst Instantaneous"]),
        ("NS_DestructibleRig", 50.0, &["Collision"]),
    ];
    let out = std::env::temp_dir().join(format!("fx_import_convert_{}", std::process::id()));
    for f in files() {
        let stem = f.file_stem().unwrap().to_string_lossy().to_string();
        let r = fx_import::convert_file(&f, &out).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let ron = std::fs::read_to_string(&r.ron_path).unwrap();
        assert!(r.report.validation.is_empty(), "{stem}: {:?}\n{ron}", r.report.validation);
        if let Some((_, pct, wants)) = expect.iter().find(|(n, _, _)| *n == stem) {
            assert!(r.report.percent() >= *pct, "{stem}: {}", r.summary());
            for w in *wants {
                assert!(ron.contains(w), "{stem}: missing {w}\n{ron}");
            }
        }
        let md = std::fs::read_to_string(&r.report_path).unwrap();
        assert!(md.contains("Modules mapped"), "{stem}");
    }
    let _ = std::fs::remove_dir_all(&out);
}
