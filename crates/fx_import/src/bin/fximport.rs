//! fximport: inspect Unreal editor packages.
//!
//!   fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports]
//!                 [--max-elems N] [--max-bytes N]
//!   fximport niagara <file.uasset>
//!   fximport convert <file.uasset> --out DIR
//!   fximport blueprint <file.uasset>... --out DIR
//!   fximport mesh <file.uasset> --out FILE.glb
//!   fximport texture <file.uasset> --out FILE.png
//!
//! `convert` writes DIR/<name>.fx.ron and DIR/<name>.report.md and exits
//! with 3 when the written effect does not validate. `mesh` reads a static
//! mesh's editor mesh description and writes it as glTF binary (metres,
//! Y up), printing the vertex/triangle counts and bounds.
//!
//! Prints the package header, imports and the export tree with each
//! export's properties. Exits with status 1 when the file cannot be read
//! and 2 when some values could not be decoded (listed under "Problems").

use std::process::ExitCode;

use fx_import::{dump, DumpOptions, Package};

fn usage() -> ExitCode {
    eprintln!(
        "usage: fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports] [--max-elems N] [--max-bytes N]\n       fximport niagara <file.uasset>            (the interpreted system: stacks, modules, inputs)\n       fximport convert <file.uasset> --out DIR  (writes DIR/<name>.fx.ron and DIR/<name>.report.md)\n       fximport blueprint <file.uasset>... --out DIR  (Blueprint actors: DIR/<name>.prefab.ron, report, effects)\n       fximport mesh <file.uasset> --out FILE.glb  (a static mesh as glTF binary, metres, Y up)\n       fximport texture <file.uasset> --out FILE.png  (a texture's source art, decoded; under a Content folder)"
    );
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { return usage() };
    if cmd == "convert" {
        let mut file = None;
        let mut out = None;
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--out" => out = it.next().cloned(),
                f if file.is_none() && !f.starts_with("--") => file = Some(f.to_string()),
                _ => return usage(),
            }
        }
        let (Some(file), Some(out)) = (file, out) else { return usage() };
        return match fx_import::convert_file(std::path::Path::new(&file), std::path::Path::new(&out)) {
            Ok(r) => {
                println!("{}", r.summary());
                if r.report.validation.is_empty() {
                    ExitCode::SUCCESS
                } else {
                    for e in &r.report.validation {
                        eprintln!("{e}");
                    }
                    ExitCode::from(3)
                }
            }
            Err(e) => {
                eprintln!("{file}: {e}");
                ExitCode::from(1)
            }
        };
    }
    if cmd == "blueprint" {
        // fximport blueprint FILE... --out DIR: prefabs (+ effects, assets, reports).
        let mut files = Vec::new();
        let mut out = None;
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--out" => out = it.next().cloned(),
                f if !f.starts_with("--") => files.push(f.to_string()),
                _ => return usage(),
            }
        }
        let Some(out) = out else { return usage() };
        if files.is_empty() {
            return usage();
        }
        let mut effects = fx_import::blueprint::Effects::default();
        let mut failed = false;
        for file in &files {
            match fx_import::blueprint::import_file(std::path::Path::new(file), std::path::Path::new(&out), &mut effects) {
                Ok((_, s)) => println!("{s}"),
                Err(e) => {
                    eprintln!("{file}: {e}");
                    failed = true;
                }
            }
        }
        return if failed { ExitCode::from(1) } else { ExitCode::SUCCESS };
    }
    if cmd == "niagara" {
        let Some(file) = args.get(1) else { return usage() };
        let pkg = match std::fs::read(file).map_err(|e| e.to_string()).and_then(|b| Package::read(&b).map_err(|e| e.to_string())) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{file}: {e}");
                return ExitCode::from(1);
            }
        };
        return match fx_import::niagara::read_system(&pkg) {
            Ok(s) => {
                print!("{}", fx_import::niagara::describe(&s));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{file}: {e}");
                ExitCode::from(1)
            }
        };
    }
    if cmd == "material" {
        // fximport material <package root> <object path> --out DIR: the material as WESL + atlas.
        let (Some(root), Some(path), Some("--out"), Some(out)) = (args.get(1), args.get(2), args.get(3).map(|s| s.as_str()), args.get(4)) else { return usage() };
        let mut store = fx_import::material::Store::new(vec![root.into()]);
        let m = match fx_import::material::Material::load(&mut store, path) { Ok(m) => m, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1) } };
        println!("chain {:?}\nblend {:?} unlit {} two_sided {} missing {:?}", m.chain, m.blend, m.unlit, m.two_sided, m.missing);
        let needs = fx_import::material::Needs { dynamic_written: [true; 4], dynamic_available: [true, true, false, false], mesh: false };
        let c = match fx_import::material::compile(&m, &needs) { Ok(c) => c, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1) } };
        let mut imgs = Vec::new();
        for t in &c.textures {
            match fx_import::material::read_texture(&mut store, t) {
                Ok(i) => { println!("texture {t}: {}x{} srgb {}", i.width, i.height, i.srgb); imgs.push(i) }
                Err(e) => println!("texture {t}: {e}"),
            }
        }
        let _ = std::fs::create_dir_all(out);
        let name = m.name();
        if !imgs.is_empty() {
            let a = fx_import::material::atlas(&imgs);
            let _ = std::fs::write(format!("{out}/{name}.png"), fx_import::material::encode_png(&a.image));
            let _ = std::fs::write(format!("{out}/{name}.wesl"), fx_import::material::with_rects(&c.wesl, &a.rects));
        } else {
            let _ = std::fs::write(format!("{out}/{name}.wesl"), fx_import::material::with_rects(&c.wesl, &[]));
        }
        for n in &c.notes { println!("note: {n}"); }
        return ExitCode::SUCCESS;
    }
    if cmd == "payloads" {
        // fximport payloads <file.uasset> --out DIR: each export's bulk data payloads, raw.
        let (Some(file), Some("--out"), Some(out)) = (args.get(1), args.get(2).map(|s| s.as_str()), args.get(3)) else { return usage() };
        let bytes = match std::fs::read(file) { Ok(b) => b, Err(e) => { eprintln!("{file}: {e}"); return ExitCode::from(1) } };
        let pkg = match Package::read(&bytes) { Ok(p) => p, Err(e) => { eprintln!("{file}: {e}"); return ExitCode::from(1) } };
        let _ = std::fs::create_dir_all(out);
        for (i, e) in pkg.exports.iter().enumerate() {
            let obj = &e.object;
            let _ = i;
            let Ok(ps) = fx_import::bulk::payloads_of(&pkg, &bytes, obj) else { continue };
            for (k, p) in ps.iter().enumerate() {
                match fx_import::bulk::read_payload(&bytes, p) {
                    Ok(data) => {
                        let path = format!("{out}/{}_{k}.bin", e.object_name);
                        let _ = std::fs::write(&path, &data);
                        println!("{path}: {} bytes ({} stored)", data.len(), p.compressed_size);
                    }
                    Err(err) => eprintln!("{}: payload {k}: {err}", e.object_name),
                }
            }
        }
        return ExitCode::SUCCESS;
    }
    if cmd == "texture" {
        // A Texture2D's decoded source art as PNG (to check colours and
        // channel order): FILE must sit under a project's Content folder.
        let (Some(file), Some("--out"), Some(out)) = (args.get(1), args.get(2).map(|s| s.as_str()), args.get(3)) else { return usage() };
        let path = std::path::Path::new(file);
        let Some(content) = path.ancestors().skip(1).find(|d| d.file_name().is_some_and(|n| n == "Content")) else {
            eprintln!("{file}: not under a Content folder");
            return ExitCode::from(1);
        };
        let rel = path.strip_prefix(content).unwrap_or(path).with_extension("");
        let rel = rel.to_string_lossy().replace('\\', "/");
        let name = rel.rsplit('/').next().unwrap_or(&rel).to_string();
        let mut store = fx_import::material::Store::new(vec![]);
        store.content_roots.push(content.to_path_buf());
        return match fx_import::material::read_texture(&mut store, &format!("/Game/{rel}.{name}")) {
            Ok(img) => match std::fs::write(out, fx_import::material::encode_png(&img)) {
                Ok(()) => {
                    println!("{out}: {}x{}, {}", img.width, img.height, if img.srgb { "sRGB" } else { "linear" });
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{out}: {e}");
                    ExitCode::from(1)
                }
            },
            Err(e) => {
                eprintln!("{file}: {e}");
                ExitCode::from(1)
            }
        };
    }
    if cmd == "mesh" {
        let (Some(file), Some("--out"), Some(out)) = (args.get(1), args.get(2).map(|s| s.as_str()), args.get(3)) else { return usage() };
        let bytes = match std::fs::read(file) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{file}: {e}");
                return ExitCode::from(1);
            }
        };
        let mesh = match Package::read(&bytes).map_err(|e| e.to_string()).and_then(|p| fx_import::mesh::read_static_mesh(&p, &bytes)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{file}: {e}");
                return ExitCode::from(1);
            }
        };
        if let Err(e) = std::fs::write(out, fx_import::mesh::write_glb(&mesh)) {
            eprintln!("{out}: {e}");
            return ExitCode::from(1);
        }
        let (lo, hi) = mesh.bounds();
        println!(
            "{out}: {} vertices, {} triangles, {} material slots ({}){}; bounds (m) min [{:.4}, {:.4}, {:.4}] max [{:.4}, {:.4}, {:.4}]",
            mesh.positions.len(),
            mesh.triangle_count(),
            mesh.slot_names.len(),
            mesh.slot_names.join(", "),
            if mesh.colors.is_some() { ", vertex colours" } else { "" },
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        );
        return ExitCode::SUCCESS;
    }
    if cmd != "dump" {
        return usage();
    }
    let mut opts = DumpOptions::default();
    let mut file = None;
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--only" => match it.next() {
                Some(v) => opts.only.push(v.clone()),
                None => return usage(),
            },
            "--no-imports" => opts.imports = false,
            "--max-elems" | "--max-bytes" => {
                let Some(n) = it.next().and_then(|v| v.parse().ok()) else { return usage() };
                if a == "--max-elems" {
                    opts.max_elems = n;
                } else {
                    opts.max_bytes = n;
                }
            }
            f if file.is_none() && !f.starts_with("--") => file = Some(f.to_string()),
            _ => return usage(),
        }
    }
    let Some(file) = file else { return usage() };
    let bytes = match std::fs::read(&file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{file}: {e}");
            return ExitCode::from(1);
        }
    };
    let pkg = match Package::read(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{file}: {e}");
            return ExitCode::from(1);
        }
    };
    print!("{}", dump(&pkg, &opts));
    if pkg.problems.is_empty() {
        ExitCode::SUCCESS
    } else {
        eprintln!("{file}: {} values could not be decoded", pkg.problems.len());
        ExitCode::from(2)
    }
}
