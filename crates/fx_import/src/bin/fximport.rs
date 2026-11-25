//! fximport: inspect Unreal editor packages.
//!
//!   fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports]
//!                 [--max-elems N] [--max-bytes N]
//!   fximport niagara <file.uasset>
//!   fximport convert <file.uasset> --out DIR
//!
//! `convert` writes DIR/<name>.fx.ron and DIR/<name>.report.md and exits
//! with 3 when the written effect does not validate.
//!
//! Prints the package header, imports and the export tree with each
//! export's properties. Exits with status 1 when the file cannot be read
//! and 2 when some values could not be decoded (listed under "Problems").

use std::process::ExitCode;

use fx_import::{dump, DumpOptions, Package};

fn usage() -> ExitCode {
    eprintln!(
        "usage: fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports] [--max-elems N] [--max-bytes N]\n       fximport niagara <file.uasset>            (the interpreted system: stacks, modules, inputs)\n       fximport convert <file.uasset> --out DIR  (writes DIR/<name>.fx.ron and DIR/<name>.report.md)"
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
