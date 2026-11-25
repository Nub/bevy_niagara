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

fn is_system(p: &Path) -> bool {
    p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("NS_"))
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
    let mut n = 0;
    for f in packages().into_iter().filter(|p| is_system(p)) {
        let r = fx_import::convert_file(&f, &out).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(r.report.validation.is_empty(), "{}: {:?}", f.display(), r.report.validation);
        let md = r.report.markdown();
        assert!(!md.contains("missing texture"), "{}: dependencies from Content/ not found\n{md}", f.display());
        assert!(r.report.percent() >= 75.0, "{}: {}", f.display(), r.summary());
        n += 1;
    }
    let _ = std::fs::remove_dir_all(&out);
    if n > 0 {
        assert_eq!(n, 25, "Perfect Fire has 15 systems, Free Arrow Trail 10");
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
