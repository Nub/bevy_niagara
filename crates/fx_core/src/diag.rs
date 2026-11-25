//! Diagnostics with file:line:col positions.

use std::fmt;

use crate::model::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub file: String,
    pub span: Span,
    pub message: String,
    /// How to fix it, written so it can be applied without other context.
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sev = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        write!(
            f,
            "{}:{}:{}: {}: {}",
            self.file, self.span.line, self.span.col, sev, self.message
        )?;
        if let Some(h) = &self.help {
            write!(f, "\n  help: {h}")?;
        }
        Ok(())
    }
}

/// Collects diagnostics for one file.
#[derive(Debug, Default)]
pub struct Diags {
    pub file: String,
    pub list: Vec<Diagnostic>,
}

impl Diags {
    pub fn new(file: impl Into<String>) -> Self {
        Diags {
            file: file.into(),
            list: Vec::new(),
        }
    }
    pub fn error(
        &mut self,
        span: Span,
        message: impl Into<String>,
        help: impl Into<Option<String>>,
    ) {
        self.push(Severity::Error, span, message.into(), help.into());
    }
    pub fn warning(
        &mut self,
        span: Span,
        message: impl Into<String>,
        help: impl Into<Option<String>>,
    ) {
        self.push(Severity::Warning, span, message.into(), help.into());
    }
    fn push(&mut self, severity: Severity, span: Span, message: String, help: Option<String>) {
        self.list.push(Diagnostic {
            severity,
            file: self.file.clone(),
            span,
            message,
            help,
        });
    }
    pub fn has_errors(&self) -> bool {
        self.list.iter().any(Diagnostic::is_error)
    }
}

/// Levenshtein distance, for "did you mean" suggestions.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.to_lowercase().chars().collect();
    let b: Vec<char> = b.to_lowercase().chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The closest candidate to `name`, if it is close enough to be a typo.
pub fn suggest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    candidates
        .into_iter()
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, c)| *d <= (c.len().max(name.len()) / 3).max(2))
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}
