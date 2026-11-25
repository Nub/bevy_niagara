//! Sample effects validate and cook; broken samples give their expected
//! file:line errors; the cooked binary round-trips.

mod common;

use std::fs;

use common::*;
use fx_core::cook;
use fx_core::sim::Simulation;

fn ron_files(dir: &str) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = fs::read_dir(effects_dir().join(dir))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".fx.ron"))
        .collect();
    v.sort();
    v
}

#[test]
fn samples_validate_cleanly_and_cook() {
    let files = ron_files("samples");
    let names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    for want in ["fountain.fx.ron", "burst.fx.ron", "smoke_curl.fx.ron"] {
        assert!(names.iter().any(|n| n == want), "missing sample {want}");
    }
    for f in files {
        let c = fx_core::check_file(&f).unwrap();
        let diags: Vec<String> = c.diagnostics.iter().map(|d| d.to_string()).collect();
        assert!(
            diags.is_empty(),
            "{} should have no diagnostics:\n{}",
            f.display(),
            diags.join("\n")
        );
        let sys = c.cooked.unwrap();

        // Cooked bytes: zero-copy access sees the same data; deserialising gives it back.
        let bytes = cook::to_bytes(&sys);
        let aligned = cook::load_aligned(&bytes);
        let archived = cook::access(&aligned).unwrap();
        assert_eq!(archived.name.as_str(), sys.name);
        assert_eq!(archived.emitters.len(), sys.emitters.len());
        assert_eq!(cook::from_bytes(&bytes).unwrap(), sys);

        // And it simulates: particles appear within a second.
        let mut sim = Simulation::new(sys, 1);
        for _ in 0..60 {
            sim.step(1.0 / 60.0);
        }
        assert!(sim.particle_count() > 0, "{} spawned nothing", f.display());
        for p in sim.particles(0) {
            assert!(
                p.position.is_finite() && p.velocity.is_finite(),
                "{} produced non-finite state",
                f.display()
            );
        }
    }
}

#[test]
fn source_model_serialises_to_ron_that_parses_back() {
    // The model's serde form is valid effect source: what tools write, the validator reads.
    for f in ron_files("samples").into_iter().chain(ron_files("library")) {
        let c = fx_core::check_file(&f).unwrap();
        let src = c.source.unwrap();
        let text = ron::ser::to_string_pretty(&src, ron::ser::PrettyConfig::default()).unwrap();
        let again = fx_core::check_source("roundtrip.fx.ron", &text);
        let errs: Vec<String> = again.diagnostics.iter().map(|d| d.to_string()).collect();
        assert!(
            again.ok(),
            "{}: reserialised source fails:\n{}\n{text}",
            f.display(),
            errs.join("\n")
        );
        assert_eq!(again.cooked.unwrap(), c.cooked.unwrap(), "{}", f.display());
    }
}

#[test]
fn corrupt_cooked_bytes_are_rejected() {
    let sys = compile(&one_emitter("", r#"(module: "Spawn Rate")"#, "", ""));
    let mut bytes = cook::to_bytes(&sys);
    let n = bytes.len();
    bytes.truncate(n - 3);
    assert!(cook::from_bytes(&bytes).is_err());
    assert!(cook::from_bytes(b"not an effect").is_err());
}

/// Each broken sample starts with `// expect: <line>: <text>`; checking it
/// must give an error on that line whose message contains the text.
#[test]
fn invalid_samples_give_expected_errors() {
    let files = ron_files("invalid");
    assert!(
        files.len() >= 5,
        "need at least five broken samples, found {}",
        files.len()
    );
    for f in files {
        let text = fs::read_to_string(&f).unwrap();
        let first = text.lines().next().unwrap();
        let spec = first.strip_prefix("// expect: ").unwrap_or_else(|| {
            panic!(
                "{}: first line must be `// expect: <line>: <text>`",
                f.display()
            )
        });
        let (line, want) = spec.split_once(": ").unwrap();
        let line: u32 = line.parse().unwrap();

        let c = fx_core::check_file(&f).unwrap();
        assert!(!c.ok(), "{} should fail validation", f.display());
        let shown: Vec<String> = c.errors().map(|d| d.to_string()).collect();
        let hit = c
            .errors()
            .find(|d| d.span.line == line && d.message.contains(want));
        let d = hit.unwrap_or_else(|| {
            panic!(
                "{}: expected an error on line {line} containing {want:?}, got:\n{}",
                f.display(),
                shown.join("\n")
            )
        });
        // The rendered diagnostic leads with file:line:col and carries a fix.
        let rendered = d.to_string();
        assert!(
            rendered.starts_with(&format!("{}:{line}:", f.display())),
            "{rendered}"
        );
        assert!(
            d.help.is_some(),
            "{}: error without help: {rendered}",
            f.display()
        );
    }
}

/// The reference effect library: every effect validates without
/// diagnostics, cooks, simulates for two seconds with every emitter spawning
/// and finite state, and its header names the Niagara tutorial steps it
/// follows and has a preview strip.
#[test]
fn library_effects_validate_cook_and_simulate() {
    let files = ron_files("library");
    let names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    for want in [
        "campfire.fx.ron",
        "sparks_burst.fx.ron",
        "rising_smoke.fx.ron",
        "magic_swirl.fx.ron",
        "sword_trail.fx.ron",
        "debris.fx.ron",
    ] {
        assert!(names.iter().any(|n| n == want), "missing library effect {want}");
    }
    for f in files {
        let text = fs::read_to_string(&f).unwrap();
        assert!(
            text.contains("Niagara tutorial recipe"),
            "{}: header must name the Niagara tutorial steps it follows",
            f.display()
        );
        let stem = f.file_name().unwrap().to_string_lossy().replace(".fx.ron", "");
        let strip = effects_dir().join("library/previews").join(format!("{stem}_strip.png"));
        assert!(strip.exists(), "{}: no preview {}", f.display(), strip.display());

        let c = fx_core::check_file(&f).unwrap();
        let diags: Vec<String> = c.diagnostics.iter().map(|d| d.to_string()).collect();
        assert!(diags.is_empty(), "{}:\n{}", f.display(), diags.join("\n"));
        let sys = c.cooked.unwrap();
        let bytes = cook::to_bytes(&sys);
        assert_eq!(cook::from_bytes(&bytes).unwrap(), sys);

        let mut sim = Simulation::new(sys, 7);
        let mut seen = vec![false; sim.system.emitters.len()];
        for _ in 0..120 {
            sim.step(1.0 / 60.0);
            for (ei, s) in seen.iter_mut().enumerate() {
                *s |= !sim.particles(ei).is_empty();
            }
        }
        for (e, s) in sim.system.emitters.iter().zip(&seen) {
            assert!(*s, "{}: emitter {} spawned nothing in 2 s", f.display(), e.name);
        }
        for ei in 0..sim.system.emitters.len() {
            for p in sim.particles(ei) {
                assert!(
                    p.position.is_finite() && p.velocity.is_finite(),
                    "{} produced non-finite state",
                    f.display()
                );
            }
        }
    }
}
