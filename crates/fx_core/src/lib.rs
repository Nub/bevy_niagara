//! Effect model, `.fx.ron` source format, validator, cooker and CPU reference
//! simulation for the Niagara-style particle plugin.
//!
//! Pipeline: source text → [`ron_parse`] (positions kept) → [`decode`] (the
//! [`model`]) → [`cook::compile`] (validation, resolution) → rkyv bytes.
//! [`sim::Simulation`] runs a [`cook::CookedSystem`] on the CPU.

pub mod cook;
pub mod custom;
pub mod data;
pub mod decode;
pub mod diag;
pub mod docs;
pub mod dynamic;
pub mod events;
pub mod model;
pub mod modules;
pub mod params;
pub mod render;
pub mod ron_parse;
pub mod scalability;
pub mod sim;
pub mod types;

use std::path::Path;

pub use cook::CookedSystem;
pub use diag::{Diagnostic, Severity};
pub use model::SystemSource;
pub use scalability::{CookedScalability, QualityLevel, ResolvedScalability};

/// Result of checking one source file.
pub struct Checked {
    pub source: Option<SystemSource>,
    pub cooked: Option<CookedSystem>,
    /// Errors and warnings, in source order.
    pub diagnostics: Vec<Diagnostic>,
}

impl Checked {
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.is_error())
    }
    pub fn ok(&self) -> bool {
        self.cooked.is_some()
    }
}

/// Parses, validates and compiles effect source text. `file` is the name used in diagnostics.
pub fn check_source(file: &str, text: &str) -> Checked {
    let mut d = diag::Diags::new(file);
    let parsed = match ron_parse::parse(text) {
        Ok(v) => v,
        Err(e) => {
            d.error(e.span, format!("syntax error: {}", e.message), e.help);
            return Checked {
                source: None,
                cooked: None,
                diagnostics: d.list,
            };
        }
    };
    let source = decode::decode_system(&parsed, &mut d);
    let cooked = match &source {
        // Shape errors make the model incomplete; still run the semantic
        // checks so one pass reports as much as possible.
        Some(src) => cook::compile(src, &mut d).filter(|_| !d.has_errors()),
        None => None,
    };
    let mut diagnostics = d.list;
    diagnostics.sort_by_key(|x| x.span);
    Checked {
        source,
        cooked,
        diagnostics,
    }
}

/// Reads and checks a `.fx.ron` file.
pub fn check_file(path: &Path) -> std::io::Result<Checked> {
    let text = std::fs::read_to_string(path)?;
    Ok(check_source(&path.display().to_string(), &text))
}
