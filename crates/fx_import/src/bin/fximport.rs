//! fximport: inspect Unreal editor packages.
//!
//!   fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports]
//!                 [--max-elems N] [--max-bytes N]
//!
//! Prints the package header, imports and the export tree with each
//! export's properties. Exits with status 1 when the file cannot be read
//! and 2 when some values could not be decoded (listed under "Problems").

use std::process::ExitCode;

use fx_import::{dump, DumpOptions, Package};

fn usage() -> ExitCode {
    eprintln!(
        "usage: fximport dump <file.uasset> [--only CLASS_OR_NAME]... [--no-imports] [--max-elems N] [--max-bytes N]"
    );
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { return usage() };
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
