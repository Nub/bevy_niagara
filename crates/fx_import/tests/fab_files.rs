//! Fab marketplace Niagara packs in testdata/fab (downloaded with
//! fab-downloader by `sh testdata/fab_assets.sh`; local only, never
//! committed). Skipped when absent. Perfect Fire VFX ships UE 4.27 packages
//! (legacy file version -7); Free Arrow Trail ships UE 5.8 ones.

use std::path::{Path, PathBuf};

use fx_import::Package;

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "uasset") {
            out.push(p);
        }
    }
}

fn packages() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fab");
    if !dir.is_dir() {
        eprintln!("skipped: {} is absent (sh testdata/fab_assets.sh)", dir.display());
    }
    let mut v = Vec::new();
    walk(&dir, &mut v);
    v.sort();
    v
}

/// The packs fab_assets.sh fetches and their Niagara system counts.
const PACKS: &[(&str, usize)] = &[
    ("perfect_fire", 15),
    ("arrow_trail", 10),
    ("lightning", 17),
    ("portals", 11),
    ("mixed_vfx", 67),
    ("mayu_basic", 4),
    ("niagara_examples", 67),
    ("m5_fire", 60),
    ("fire_explosion", 10),
    ("sparkles", 14),
    ("loot_drops", 7),
];

/// The pack folder a file is in (`testdata/fab/<pack>/...`).
fn pack_of(p: &Path) -> String {
    let s = p.to_string_lossy();
    s.split("/fab/").nth(1).and_then(|r| r.split('/').next()).unwrap_or("").to_string()
}

fn find_file(pack: &str, name: &str) -> Option<PathBuf> {
    packages().into_iter().find(|p| pack_of(p) == pack && p.file_name().is_some_and(|n| n == name))
}

fn is_system(p: &Path) -> bool {
    // M5 VFX names its systems NFire_* (and its blueprints NFire_*_BP, kept
    // in BluePrints/).
    let n = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    n.starts_with("NS_") || (n.starts_with("NFire_") && !p.to_string_lossy().contains("/BluePrints/"))
}

/// Converts one system into a temporary folder; its .fx.ron text.
fn convert_text(f: &Path, tag: &str) -> String {
    let out = std::env::temp_dir().join(format!("fx_import_fab_{tag}_{}", std::process::id()));
    let r = fx_import::convert_file(f, &out).unwrap();
    assert!(r.report.validation.is_empty(), "{}: {:?}", f.display(), r.report.validation);
    let text = std::fs::read_to_string(&r.ron_path).unwrap();
    let _ = std::fs::remove_dir_all(&out);
    text
}

#[test]
fn emitter_stack_curves_sample_the_emitter_age() {
    // NS_Explosion's spark emitter ramps its Spawn Rate down with a curve in
    // Emitter Update, indexed by the emitter's normalised age.
    let Some(f) = find_file("niagara_examples", "NS_Explosion.uasset") else { return };
    let text = convert_text(&f, "explosion");
    let line = text.lines().find(|l| l.contains("module: \"Spawn Rate\"") && l.contains("from Curve")).expect("Spawn Rate over a curve");
    assert!(line.contains("\"Normalized Emitter Age\""), "{line}");
}

#[test]
fn ue5_spawn_spacing_becomes_spawn_per_unit() {
    // Epic's trails save Spawn Spacing (cm between particles); ours takes
    // particles per metre.
    let Some(f) = find_file("niagara_examples", "NS_BulletTracer.uasset") else { return };
    let text = convert_text(&f, "tracer");
    let line = text.lines().find(|l| l.contains("module: \"Spawn Per Unit\"")).expect("Spawn Per Unit");
    assert!(line.contains("\"Spawn Per Unit\":") && !line.contains("Spawn Spacing"), "{line}");
}

#[test]
fn every_effect_package_reads_without_problems() {
    // Systems, emitters, materials and textures (maps, sounds and blueprints
    // are not the importer's business).
    let keep = ["NS_", "NE_", "M_", "MI_", "MF_", "T_"];
    for f in packages() {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if !keep.iter().any(|k| name.starts_with(k)) {
            continue;
        }
        let bytes = std::fs::read(&f).unwrap();
        let pkg = Package::read(&bytes).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(pkg.problems.is_empty(), "{}: {:#?}", f.display(), &pkg.problems[..pkg.problems.len().min(5)]);
        for (i, e) in pkg.exports.iter().enumerate() {
            assert!(e.object.error.is_none(), "{} export {i}: {:?}", f.display(), e.object.error);
        }
    }
}

#[test]
fn every_system_converts_validates_and_finds_its_textures() {
    let out = std::env::temp_dir().join(format!("fx_import_fab_{}", std::process::id()));
    let mut n: std::collections::BTreeMap<String, usize> = Default::default();
    for f in packages().into_iter().filter(|p| is_system(p)) {
        let r = fx_import::convert_file(&f, &out).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(r.report.validation.is_empty(), "{}: {:?}", f.display(), r.report.validation);
        let md = r.report.markdown();
        // Engine textures (/Engine/...) are borrowed, never shipped by a pack.
        let missing = md.lines().any(|l| l.contains("missing texture") && l.split("**missing texture**").skip(1).any(|t| !t.contains("`/Engine/")));
        assert!(!missing, "{}: dependencies from Content/ not found\n{md}", f.display());
        // Epic's Niagara Examples leans on lightweight (stateless) emitters,
        // attribute readers and data channels, not imported yet.
        if pack_of(&f) != "niagara_examples" {
            assert!(r.report.percent() >= 75.0, "{}: {}", f.display(), r.summary());
        }
        *n.entry(pack_of(&f)).or_default() += 1;
    }
    let _ = std::fs::remove_dir_all(&out);
    for (pack, count) in PACKS {
        if let Some(got) = n.get(*pack) {
            assert_eq!(got, count, "{pack}: systems converted");
        }
    }
}

#[test]
fn ue4_zlib_bulk_chunks_inflate() {
    // FArchive::SerializeCompressed: tag, summary, one size pair per chunk, zlib streams.
    let raw: Vec<u8> = (0..300u32).map(|i| (i * 7 % 251) as u8).collect();
    let chunk = 128usize;
    let streams: Vec<Vec<u8>> = raw.chunks(chunk).map(|c| miniz_oxide::deflate::compress_to_vec_zlib(c, 6)).collect();
    let mut b = Vec::new();
    let push = |b: &mut Vec<u8>, v: i64| b.extend_from_slice(&v.to_le_bytes());
    push(&mut b, 0x9E2A_83C1);
    push(&mut b, chunk as i64);
    push(&mut b, streams.iter().map(|s| s.len() as i64).sum());
    push(&mut b, raw.len() as i64);
    for (s, c) in streams.iter().zip(raw.chunks(chunk)) {
        push(&mut b, s.len() as i64);
        push(&mut b, c.len() as i64);
    }
    for s in &streams {
        b.extend_from_slice(s);
    }
    assert_eq!(fx_import::bulk::inflate_chunks(&b).unwrap(), raw);
    b[0] = 0;
    assert!(fx_import::bulk::inflate_chunks(&b).is_err());
}

fn system(f: &Path) -> fx_import::niagara::NSystem {
    let bytes = std::fs::read(f).unwrap();
    let pkg = Package::read(&bytes).unwrap();
    fx_import::niagara::read_system(&pkg).unwrap()
}

#[test]
fn ue4_era_modules_of_mixed_vfx_all_map() {
    // Mixed VFX ships UE 4.x saves with the UE4 module scripts
    // (AddVelocityInCone, BoxLocation, TorusLocation, InitializeRibbon,
    // Generate/ReceiveLocationEvent, SpriteRotationRate, ...).
    let out = std::env::temp_dir().join(format!("fx_import_fab_mix_{}", std::process::id()));
    let mut n = 0;
    for f in packages().into_iter().filter(|p| is_system(p) && pack_of(p) == "mixed_vfx") {
        let r = fx_import::convert_file(&f, &out).unwrap();
        for m in &r.report.modules {
            if let fx_import::convert::Status::Unmapped(why) = &m.status {
                // Set Variables of attributes we lack and one spawn-stage force are the known gaps.
                assert!(
                    m.script.ends_with("SetVariables") || why.contains("cannot run in"),
                    "{}: {} / {} not mapped: {why}",
                    f.display(),
                    m.emitter,
                    m.module
                );
            }
        }
        assert!(r.report.unmapped_dynamic.iter().all(|(_, s)| s.contains("Lerp")), "{}: {:?}", f.display(), r.report.unmapped_dynamic);
        n += 1;
    }
    let _ = std::fs::remove_dir_all(&out);
    if n > 0 {
        assert_eq!(n, 67);
    }
}

#[test]
fn ue4_event_handlers_with_a_zero_source_id_find_their_generator() {
    // NS_Wind's ribbon emitter keeps SourceEmitterID 0 in its own handler
    // copy; the Location events come from emitter "Empty".
    let Some(f) = find_file("mixed_vfx", "NS_Wind.uasset") else { return };
    let sys = system(&f);
    let ribbon = sys.emitters.iter().find(|e| e.name == "LocationBasedRibbon").unwrap();
    let ev = ribbon
        .stacks
        .iter()
        .find_map(|s| match &s.usage {
            fx_import::niagara::Usage::Event(ev) => Some(ev.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(ev.source_emitter.as_deref(), Some("Empty"));
}

#[test]
fn ue57_delta_coded_texture_source_decodes() {
    // Mayu's Basic VFX Pack (UE 5.7) saves source art as TSCF_UEDELTA.
    let Some(f) = find_file("mayu_basic", "T_SawBlade_01.uasset") else { return };
    let mut store = fx_import::material::Store::for_system(&f);
    let path = "/Game/MayuAssets/SawBlade/Textures/T_SawBlade_01.T_SawBlade_01";
    let rel = f.to_string_lossy().split("/Content/").nth(1).unwrap().trim_end_matches(".uasset").to_string();
    assert_eq!(format!("/Game/{rel}.T_SawBlade_01"), path, "texture moved in the pack");
    let img = fx_import::material::read_texture(&mut store, path).unwrap();
    let px = |x: u32, y: u32| img.rgba[((y * img.width + x) * 4) as usize];
    // A white saw blade on black: the centre is solid, the corners empty, and
    // rows straddling a delta block (32 rows here) do not streak.
    assert_eq!((img.width, img.height), (1024, 1024));
    assert!(px(512, 300) > 200 && px(512, 700) > 200, "blade body");
    assert!(px(5, 5) < 30 && px(1018, 1018) < 30, "corners");
    for y in [31u32, 32, 63, 64, 479, 480] {
        assert!(px(512, y + 200) > 200, "row {}", y + 200);
    }
}

#[test]
fn ue57_linear_color_material_inputs_read() {
    // UE 5.7 saves FColorMaterialInput's constant as an FLinearColor.
    let Some(f) = find_file("mayu_basic", "M_Fire.uasset") else { return };
    let bytes = std::fs::read(&f).unwrap();
    let pkg = Package::read(&bytes).unwrap();
    assert!(pkg.problems.is_empty(), "{:#?}", pkg.problems);
}

#[test]
fn engine_noise_textures_get_a_stand_in() {
    // Mixed VFX fires sample the engine's LowResBlurredNoise, which no pack ships.
    let Some(f) = find_file("mixed_vfx", "NS_Campfire.uasset") else { return };
    let out = std::env::temp_dir().join(format!("fx_import_fab_noise_{}", std::process::id()));
    let r = fx_import::convert_file(&f, &out).unwrap();
    let _ = std::fs::remove_dir_all(&out);
    let md = r.report.markdown();
    assert!(md.contains("LowResBlurredNoise not in the pack"), "{md}");
    assert!(!md.contains("missing texture"), "{md}");
}

#[test]
fn ue4_raw_mesh_static_meshes_read() {
    // UE 4.x static meshes keep LOD0 as a zlib-compressed FRawMesh in the
    // StaticMesh native data (no StaticMeshDescriptionBulkData export).
    let Some(f) = find_file("mixed_vfx", "SM_Slash_01.uasset") else {
        eprintln!("skipped: mixed_vfx is absent");
        return;
    };
    for name in ["SM_Slash_01.uasset", "SM_Arrow.uasset", "SM_Shuriken.uasset"] {
        let f = f.with_file_name(name);
        let bytes = std::fs::read(&f).unwrap();
        let pkg = Package::read(&bytes).unwrap();
        assert!(!pkg.exports.iter().any(|e| e.object_name.starts_with("StaticMeshDescriptionBulkData")));
        let m = fx_import::mesh::read_static_mesh(&pkg, &bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (lo, hi) = m.bounds();
        let size = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
        eprintln!("{name}: {} triangles, {} vertices, bounds {lo:?} .. {hi:?}", m.triangle_count(), m.positions.len());
        assert!(m.triangle_count() >= 2 && m.triangle_count() < 200_000, "{name}: {} triangles", m.triangle_count());
        assert_eq!(m.indices.len(), 3 * m.triangle_count());
        assert_eq!(m.triangle_slots.len(), m.triangle_count());
        assert!(m.indices.iter().all(|&i| (i as usize) < m.positions.len()));
        assert!((0.05..10.0).contains(&size), "{name}: largest extent {size} m");
        assert!(m.positions.iter().flatten().chain(m.uv0.iter().flatten()).all(|v| v.is_finite()));
        assert!(m.normals.iter().all(|n| ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-3));
        // Counter-clockwise fronts: face normals agree with the stored ones.
        let (mut agree, mut faces) = (0, 0);
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let s: f32 = t.iter().map(|&i| (0..3).map(|k| n[k] * m.normals[i as usize][k]).sum::<f32>()).sum();
            if s != 0.0 {
                faces += 1;
                agree += (s > 0.0) as usize;
            }
        }
        eprintln!("{name}: {agree} of {faces} faces agree with their normals");
        assert!(agree * 10 >= faces * 9, "{name}: {agree} of {faces} faces agree with their normals");
        assert!(!fx_import::mesh::write_glb(&m).is_empty());
    }
    // The slash system draws the real mesh, not a built-in stand-in.
    let Some(sys) = find_file("mixed_vfx", "NS_FireSlash.uasset") else { return };
    let out = std::env::temp_dir().join(format!("fx_import_ue4_mesh_{}", std::process::id()));
    let r = fx_import::convert_file(&sys, &out).unwrap();
    let _ = std::fs::remove_dir_all(&out);
    eprintln!("{:#?}", r.report.materials.iter().filter(|m| m.contains("mesh `")).collect::<Vec<_>>());
    assert!(r.report.materials.iter().any(|m| m.contains("mesh `SM_Slash_01` -> `") && m.contains("SM_Slash_01.glb")));
    assert!(!r.report.placeholders.iter().any(|p| p.contains("SM_Slash_01")), "{:?}", r.report.placeholders);
}
