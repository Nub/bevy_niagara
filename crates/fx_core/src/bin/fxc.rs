//! `fxc`: check and cook `.fx.ron` effects.
//!
//! ```text
//! fxc check <file.fx.ron>...           validate; prints file:line:col diagnostics
//! fxc cook  <file.fx.ron> [-o out.fxb] validate and write the cooked binary
//! fxc modules                          list modules, their stages and inputs, and renderer settings
//! ```
//! Exit status: 0 ok (warnings allowed), 1 errors, 2 bad usage or I/O.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use fx_core::{check_file, cook, modules};

const USAGE: &str =
    "usage:\n  fxc check <file.fx.ron>...\n  fxc cook <file.fx.ron> [-o <out.fxb>]\n  fxc modules";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") if args.len() > 1 => check(&args[1..]),
        Some("cook") if args.len() > 1 => cook_cmd(&args[1..]),
        Some("modules") => {
            list_modules();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn check(files: &[String]) -> ExitCode {
    let mut failed = false;
    for f in files {
        match check_file(Path::new(f)) {
            Ok(c) => {
                for d in &c.diagnostics {
                    println!("{d}");
                }
                if c.ok() {
                    println!("{f}: ok");
                } else {
                    failed = true;
                }
            }
            Err(e) => {
                eprintln!("{f}: cannot read: {e}");
                return ExitCode::from(2);
            }
        }
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn cook_cmd(args: &[String]) -> ExitCode {
    let (input, out) = match args {
        [i] => (i, None),
        [i, o, out] if o == "-o" => (i, Some(PathBuf::from(out))),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let c = match check_file(Path::new(input)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{input}: cannot read: {e}");
            return ExitCode::from(2);
        }
    };
    for d in &c.diagnostics {
        println!("{d}");
    }
    let Some(sys) = c.cooked else {
        return ExitCode::from(1);
    };
    let out = out.unwrap_or_else(|| default_out(Path::new(input)));
    let bytes = cook::to_bytes(&sys);
    if let Err(e) = std::fs::write(&out, &bytes) {
        eprintln!("{}: cannot write: {e}", out.display());
        return ExitCode::from(2);
    }
    println!(
        "{input}: cooked to {} ({} bytes)",
        out.display(),
        bytes.len()
    );
    ExitCode::SUCCESS
}

/// `fountain.fx.ron` → `fountain.fxb` next to it.
fn default_out(input: &Path) -> PathBuf {
    let name = input
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("effect");
    let stem = name
        .strip_suffix(".fx.ron")
        .or_else(|| name.strip_suffix(".ron"))
        .unwrap_or(name);
    input.with_file_name(format!("{stem}.{}", cook::COOKED_EXTENSION))
}

fn list_modules() {
    for m in modules::MODULES {
        let stages: Vec<_> = m.stages.iter().map(|s| s.field()).collect();
        println!("\"{}\" [{}]\n  {}", m.name, stages.join(", "), m.doc);
        for i in m.inputs {
            println!(
                "  - \"{}\": {} = {}  ({})",
                i.name,
                fx_core::docs::input_type(i),
                fx_core::docs::input_default(i),
                i.doc
            );
        }
    }
    println!("\nRenderers (emitter field `renderers: [ ... ]`; every field optional):");
    for (kind, fields) in fx_core::render::RENDERER_FIELDS {
        println!("{kind}(( ... ))");
        for f in *fields {
            println!("  - {}: {} = {}  ({})", f.name, f.form, f.default, f.doc);
        }
    }
    println!("\nDynamic inputs (`Dynamic(\"Name\", {{ inputs }})`, nestable; any input of the type they give):");
    for d in fx_core::dynamic::DYNAMIC_INPUTS.iter() {
        println!("Dynamic \"{}\" -> {}\n  {}", d.name, d.ty.name(), d.doc);
        for a in &d.args {
            println!(
                "  - \"{}\": {}  ({})",
                a.name,
                fx_core::dynamic::arg_type(a),
                a.doc
            );
        }
    }
}
