//! A small RON parser that keeps the line and column of every value, so the
//! validator can point at the exact place in a `.fx.ron` file.
//!
//! It accepts the RON subset effect files use: structs and tuples (optionally
//! named, `Name(...)`), lists, maps, strings, numbers, booleans, unit
//! identifiers (also raw, `r#None`), `//` and `/* */` comments, trailing commas and a leading
//! `#![enable(...)]` attribute.

use crate::model::Span;

#[derive(Clone, Debug, PartialEq)]
pub struct SValue {
    pub kind: SKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub span: Span,
    pub value: SValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SKind {
    /// The number and whether it was written as an integer.
    Number(f64, bool),
    Str(String),
    Bool(bool),
    /// A bare identifier, e.g. a unit enum variant `Gpu`.
    Ident(String),
    /// `(a, b)` or `Name(a, b)`.
    Tuple(Option<String>, Vec<SValue>),
    /// `(x: a, y: b)` or `Name(x: a)`.
    Struct(Option<String>, Vec<Field>),
    List(Vec<SValue>),
    Map(Vec<(SValue, SValue)>),
}

impl SKind {
    /// What this is, for error messages.
    pub fn describe(&self) -> String {
        match self {
            SKind::Number(n, true) => format!("the integer {n}"),
            SKind::Number(n, false) => format!("the number {n}"),
            SKind::Str(s) => format!("the string \"{s}\""),
            SKind::Bool(b) => format!("`{b}`"),
            SKind::Ident(i) => format!("`{i}`"),
            SKind::Tuple(Some(n), _) | SKind::Struct(Some(n), _) => format!("`{n}(...)`"),
            SKind::Tuple(None, _) => "a tuple `(...)`".into(),
            SKind::Struct(None, _) => "a struct `(field: ...)`".into(),
            SKind::List(_) => "a list `[...]`".into(),
            SKind::Map(_) => "a map `{...}`".into(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
    pub help: Option<String>,
}

pub fn parse(src: &str) -> Result<SValue, ParseError> {
    let mut p = Parser {
        src: src.as_bytes(),
        pos: 0,
        line: 1,
        col: 1,
    };
    p.skip_ws()?;
    if p.peek() == Some(b'#') {
        p.skip_attribute()?;
        p.skip_ws()?;
    }
    let v = p.value()?;
    p.skip_ws()?;
    if p.pos < p.src.len() {
        return Err(p.err_here("unexpected text after the end of the effect", Some("an effect file holds exactly one `( ... )` value; remove what follows it or check for an extra closing bracket".into())));
    }
    Ok(v)
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl Parser<'_> {
    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }
    fn peek_at(&self, n: usize) -> Option<u8> {
        self.src.get(self.pos + n).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else if c & 0xC0 != 0x80 {
            // Count columns in characters, not UTF-8 continuation bytes.
            self.col += 1;
        }
        Some(c)
    }
    fn err_here(&self, msg: &str, help: Option<String>) -> ParseError {
        ParseError {
            span: self.span(),
            message: msg.into(),
            help,
        }
    }
    fn found(&self) -> String {
        match self.peek() {
            None => "the end of the file".into(),
            Some(c) => {
                let s = std::str::from_utf8(&self.src[self.pos..]).unwrap_or("?");
                let ch = s.chars().next().unwrap_or(c as char);
                format!("`{ch}`")
            }
        }
    }

    fn skip_ws(&mut self) -> Result<(), ParseError> {
        loop {
            match (self.peek(), self.peek_at(1)) {
                (Some(c), _) if c.is_ascii_whitespace() => {
                    self.bump();
                }
                (Some(b'/'), Some(b'/')) => {
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                (Some(b'/'), Some(b'*')) => {
                    let start = self.span();
                    self.bump();
                    self.bump();
                    let mut depth = 1;
                    while depth > 0 {
                        match (self.peek(), self.peek_at(1)) {
                            (None, _) => {
                                return Err(ParseError {
                                    span: start,
                                    message: "block comment `/*` is never closed".into(),
                                    help: Some("add `*/` where the comment should end".into()),
                                });
                            }
                            (Some(b'*'), Some(b'/')) => {
                                self.bump();
                                self.bump();
                                depth -= 1;
                            }
                            (Some(b'/'), Some(b'*')) => {
                                self.bump();
                                self.bump();
                                depth += 1;
                            }
                            _ => {
                                self.bump();
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn skip_attribute(&mut self) -> Result<(), ParseError> {
        // `#![enable(...)]`
        while let Some(c) = self.bump() {
            if c == b']' {
                return Ok(());
            }
        }
        Err(self.err_here("unterminated `#![...]` attribute", None))
    }

    fn value(&mut self) -> Result<SValue, ParseError> {
        self.skip_ws()?;
        let span = self.span();
        let kind = match self.peek() {
            Some(b'(') => self.compound(None)?,
            Some(b'[') => self.list()?,
            Some(b'{') => self.map()?,
            Some(b'"') => SKind::Str(self.string()?),
            Some(c) if c == b'-' || c == b'+' || c == b'.' || c.is_ascii_digit() => {
                self.number()?
            }
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                let mut id = self.ident();
                // A raw identifier, e.g. `r#None` (how `ron` writes a variant named `None`).
                let raw = id == "r"
                    && self.peek() == Some(b'#')
                    && self
                        .peek_at(1)
                        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_');
                if raw {
                    self.bump();
                    id = self.ident();
                }
                match id.as_str() {
                    "true" if !raw => SKind::Bool(true),
                    "false" if !raw => SKind::Bool(false),
                    "inf" if !raw => SKind::Number(f64::INFINITY, false),
                    "NaN" if !raw => SKind::Number(f64::NAN, false),
                    _ => {
                        if self.peek() == Some(b'(') {
                            self.compound(Some(id))?
                        } else {
                            SKind::Ident(id)
                        }
                    }
                }
            }
            _ => {
                return Err(self.err_here(
                    &format!("expected a value, found {}", self.found()),
                    Some("a value is a number, a \"string\", `true`/`false`, `(...)`, `[...]`, `{...}` or `Name(...)`".into()),
                ));
            }
        };
        Ok(SValue { kind, span })
    }

    fn ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.bump();
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.src[start..self.pos]).into_owned()
    }

    fn number(&mut self) -> Result<SKind, ParseError> {
        let span = self.span();
        let start = self.pos;
        let mut is_int = true;
        if matches!(self.peek(), Some(b'-' | b'+')) {
            self.bump();
        }
        if self.peek() == Some(b'i') {
            let id = self.ident();
            if id == "inf" {
                let neg = self.src[start] == b'-';
                return Ok(SKind::Number(
                    if neg {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                    false,
                ));
            }
        }
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' | b'_' => {}
                b'.' => is_int = false,
                b'e' | b'E' => {
                    is_int = false;
                    if matches!(self.peek_at(1), Some(b'-' | b'+')) {
                        self.bump();
                    }
                }
                _ => break,
            }
            self.bump();
        }
        let text: String = String::from_utf8_lossy(&self.src[start..self.pos]).replace('_', "");
        text.parse::<f64>()
            .map(|n| SKind::Number(n, is_int))
            .map_err(|_| ParseError {
                span,
                message: format!("`{text}` is not a valid number"),
                help: Some("write numbers like `1`, `-2.5` or `1e-3`".into()),
            })
    }

    fn string(&mut self) -> Result<String, ParseError> {
        let start = self.span();
        self.bump(); // opening quote
        let mut out = Vec::new();
        loop {
            match self.bump() {
                None | Some(b'\n') => {
                    return Err(ParseError {
                        span: start,
                        message: "string is never closed".into(),
                        help: Some("add the closing `\"` on the same line".into()),
                    });
                }
                Some(b'"') => break,
                Some(b'\\') => match self.bump() {
                    Some(b'n') => out.push(b'\n'),
                    Some(b't') => out.push(b'\t'),
                    Some(b'r') => out.push(b'\r'),
                    Some(b'0') => out.push(0),
                    Some(c @ (b'"' | b'\\' | b'/' | b'\'')) => out.push(c),
                    _ => {
                        return Err(self.err_here(
                            "unknown escape in string",
                            Some("use `\\\"`, `\\\\`, `\\n` or `\\t`".into()),
                        ));
                    }
                },
                Some(c) => out.push(c),
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    fn expect_close(&mut self, close: u8, open_span: Span, what: &str) -> Result<(), ParseError> {
        // Where the previous item ended: a missing `,` belongs there.
        let after_item = self.span();
        self.skip_ws()?;
        if self.peek() == Some(close) {
            self.bump();
            return Ok(());
        }
        let closer = close as char;
        let here = self.span();
        Err(ParseError {
            span: after_item,
            message: format!(
                "expected `,` or `{closer}` in {what}, found {} at {}:{}",
                self.found(),
                here.line,
                here.col
            ),
            help: Some(format!(
                "add a `,` here if another item follows; the {what} opened at {}:{} must end with `{closer}`",
                open_span.line, open_span.col
            )),
        })
    }

    /// After a value in a sequence: consumes a `,` if present. Returns false
    /// when the sequence must end (no comma).
    fn comma(&mut self) -> Result<bool, ParseError> {
        let saved = (self.pos, self.line, self.col);
        self.skip_ws()?;
        if self.peek() == Some(b',') {
            self.bump();
            Ok(true)
        } else {
            // Rewind so a missing-comma error points right after the item.
            (self.pos, self.line, self.col) = saved;
            Ok(false)
        }
    }

    /// Whether the parser is at `ident :` (a struct field), without consuming.
    fn at_field(&self) -> bool {
        let mut i = self.pos;
        let s = self.src;
        if !s
            .get(i)
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        {
            return false;
        }
        while s
            .get(i)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            i += 1;
        }
        while s.get(i).is_some_and(|c| *c == b' ' || *c == b'\t') {
            i += 1;
        }
        s.get(i) == Some(&b':')
    }

    fn compound(&mut self, name: Option<String>) -> Result<SKind, ParseError> {
        let open = self.span();
        self.bump(); // (
        self.skip_ws()?;
        if self.at_field() {
            let mut fields = Vec::new();
            loop {
                self.skip_ws()?;
                if self.peek() == Some(b')') {
                    break;
                }
                let span = self.span();
                if !self.at_field() {
                    return Err(self.err_here(
                        &format!("expected a field `name: value`, found {}", self.found()),
                        Some("inside `( ... )` every item is `field_name: value` once the first one is".into()),
                    ));
                }
                let fname = self.ident();
                self.skip_ws()?;
                self.bump(); // :
                let value = self.value()?;
                fields.push(Field {
                    name: fname,
                    span,
                    value,
                });
                if !self.comma()? {
                    break;
                }
            }
            self.expect_close(b')', open, "struct")?;
            Ok(SKind::Struct(name, fields))
        } else {
            let mut items = Vec::new();
            loop {
                self.skip_ws()?;
                if self.peek() == Some(b')') {
                    break;
                }
                items.push(self.value()?);
                if !self.comma()? {
                    break;
                }
            }
            self.expect_close(b')', open, "tuple")?;
            Ok(SKind::Tuple(name, items))
        }
    }

    fn list(&mut self) -> Result<SKind, ParseError> {
        let open = self.span();
        self.bump();
        let mut items = Vec::new();
        loop {
            self.skip_ws()?;
            if self.peek() == Some(b']') {
                break;
            }
            items.push(self.value()?);
            if !self.comma()? {
                break;
            }
        }
        self.expect_close(b']', open, "list")?;
        Ok(SKind::List(items))
    }

    fn map(&mut self) -> Result<SKind, ParseError> {
        let open = self.span();
        self.bump();
        let mut entries = Vec::new();
        loop {
            self.skip_ws()?;
            if self.peek() == Some(b'}') {
                break;
            }
            let k = self.value()?;
            self.skip_ws()?;
            if self.peek() != Some(b':') {
                return Err(self.err_here(
                    &format!("expected `:` after a map key, found {}", self.found()),
                    Some("map entries are written `\"Key\": value`".into()),
                ));
            }
            self.bump();
            let v = self.value()?;
            entries.push((k, v));
            if !self.comma()? {
                break;
            }
        }
        self.expect_close(b'}', open, "map")?;
        Ok(SKind::Map(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_with_positions() {
        let src = "// c\n(\n  name: \"A\",\n  xs: [1, 2.5, -3e1],\n  m: { \"k\": Float(1.0) },\n  v: Vec3((0, 1, 0)),\n)";
        let v = parse(src).unwrap();
        let SKind::Struct(None, fields) = v.kind else {
            panic!()
        };
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[0].span, Span { line: 3, col: 3 });
        assert_eq!(
            fields[1].value.kind,
            SKind::List(vec![
                SValue {
                    kind: SKind::Number(1.0, true),
                    span: Span { line: 4, col: 8 }
                },
                SValue {
                    kind: SKind::Number(2.5, false),
                    span: Span { line: 4, col: 11 }
                },
                SValue {
                    kind: SKind::Number(-30.0, false),
                    span: Span { line: 4, col: 16 }
                },
            ])
        );
        assert_eq!(fields[3].span.line, 6);
    }

    #[test]
    fn reports_missing_comma_position() {
        let err = parse("(\n  a: 1\n  b: 2,\n)").unwrap_err();
        // The missing comma belongs at the end of line 2.
        assert_eq!(err.span, Span { line: 2, col: 7 });
    }
}
