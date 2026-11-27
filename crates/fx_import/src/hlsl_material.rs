//! Translation of the HLSL body of an Unreal Custom material expression
//! (`MaterialExpressionCustom`) to a WGSL function.
//!
//! Token based: the body is tokenized, parsed into a small AST (statements
//! and expressions), typed (bool, int, uint and float scalars and vectors)
//! and emitted as WGSL with the conversions HLSL makes implicitly written
//! out: scalar to vector splats, int to float, vector truncation. Local
//! structs with member functions (Unreal's usual way to declare helper
//! functions in a Custom node) become free functions. Anything outside the
//! supported subset (matrices, arrays, `out` parameters, struct fields,
//! preprocessor lines, the material's `Parameters` / `View` globals, ...)
//! is reported as an error so the caller can fall back.

use std::collections::{BTreeMap, HashMap, HashSet};

// ---- types ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Base {
    Bool,
    Int,
    Uint,
    Float,
}

/// A scalar (`n == 1`) or vector type; `n == 0` is `void`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ty {
    base: Base,
    n: u8,
}

impl Ty {
    fn new(base: Base, n: u8) -> Ty {
        Ty { base, n }
    }
    fn f(n: u8) -> Ty {
        Ty::new(Base::Float, n)
    }
    fn void() -> Ty {
        Ty::new(Base::Float, 0)
    }
    fn wgsl(self) -> String {
        let s = match self.base {
            Base::Bool => "bool",
            Base::Int => "i32",
            Base::Uint => "u32",
            Base::Float => "f32",
        };
        if self.n == 1 {
            s.to_string()
        } else {
            format!("vec{}<{s}>", self.n)
        }
    }
    fn hlsl(self) -> String {
        let s = match self.base {
            Base::Bool => "bool",
            Base::Int => "int",
            Base::Uint => "uint",
            Base::Float => "float",
        };
        match self.n {
            0 => "void".into(),
            1 => s.into(),
            n => format!("{s}{n}"),
        }
    }
}

/// An HLSL type name: None when `name` is not one, Err for matrices.
fn hlsl_type(name: &str) -> Option<Result<Ty, String>> {
    let stem = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == 'x');
    let base = match stem {
        "float" | "half" | "double" | "min16float" | "min10float" => Base::Float,
        "int" | "min16int" | "min12int" => Base::Int,
        "uint" | "min16uint" | "dword" => Base::Uint,
        "bool" => Base::Bool,
        _ => return None,
    };
    let dims = &name[stem.len()..];
    if dims.contains('x') {
        return Some(Err(format!("matrix type `{name}` is not supported")));
    }
    // min16float etc. end in digits that are part of the name.
    let dims = if stem.starts_with("min") { &name[stem.len()..] } else { dims };
    Some(match dims {
        "" | "1" => Ok(Ty::new(base, 1)),
        "2" | "3" | "4" => Ok(Ty::new(base, dims.parse().unwrap())),
        _ => return None,
    })
}

// ---- tokens ----

#[derive(Clone, Debug, PartialEq)]
enum T {
    Id(String),
    /// Integer literal as WGSL text (`7`, `0x1F`, `3u`).
    Int(String),
    /// Float literal as WGSL text (`0.5`, `1.0e-4`).
    Float(String),
    P(&'static str),
    End,
}

#[derive(Clone, Debug)]
struct Tok {
    t: T,
    line: usize,
}

const PUNCT: &[&str] = &[
    "<<=", ">>=", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "::", "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "?", ":", ";", ",", ".", "(", ")", "{", "}", "[", "]",
];

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let s = src.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut line) = (0, 1);
    while i < s.len() {
        let c = s[i] as char;
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if src[i..].starts_with("//") {
            while i < s.len() && s[i] != b'\n' {
                i += 1;
            }
        } else if src[i..].starts_with("/*") {
            let end = src[i + 2..].find("*/").ok_or(format!("line {line}: unterminated comment"))?;
            line += src[i..i + 2 + end].matches('\n').count();
            i += end + 4;
        } else if c == '#' {
            return Err(format!("line {line}: preprocessor lines are not supported"));
        } else if c == '"' || c == '\'' {
            return Err(format!("line {line}: string literals are not supported"));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < s.len() && (s[i].is_ascii_alphanumeric() || s[i] == b'_') {
                i += 1;
            }
            out.push(Tok { t: T::Id(src[st..i].to_string()), line });
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < s.len() && s[i + 1].is_ascii_digit()) {
            let (t, len) = number(&src[i..]).map_err(|e| format!("line {line}: {e}"))?;
            out.push(Tok { t, line });
            i += len;
        } else {
            let p = PUNCT.iter().find(|p| src[i..].starts_with(**p)).ok_or(format!("line {line}: unexpected character `{c}`"))?;
            out.push(Tok { t: T::P(p), line });
            i += p.len();
        }
    }
    out.push(Tok { t: T::End, line });
    Ok(out)
}

/// A numeric literal at the start of `s`, and its length.
fn number(s: &str) -> Result<(T, usize), String> {
    let b = s.as_bytes();
    let mut i = 0;
    if s.starts_with("0x") || s.starts_with("0X") {
        i = 2;
        while i < b.len() && b[i].is_ascii_hexdigit() {
            i += 1;
        }
        let digits = &s[..i];
        let mut t = T::Int(digits.to_string());
        while i < b.len() && matches!(b[i], b'u' | b'U' | b'l' | b'L') {
            if matches!(b[i], b'u' | b'U') {
                t = T::Int(format!("{digits}u"));
            }
            i += 1;
        }
        return Ok((t, i));
    }
    let digits = |i: &mut usize| {
        let st = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        &s[st..*i]
    };
    let int = digits(&mut i);
    let mut frac = None;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        frac = Some(digits(&mut i));
    }
    let mut exp = String::new();
    if i < b.len() && matches!(b[i], b'e' | b'E') {
        let st = i;
        i += 1;
        if i < b.len() && matches!(b[i], b'+' | b'-') {
            i += 1;
        }
        if digits(&mut i).is_empty() {
            return Err(format!("bad number `{}`", &s[..i]));
        }
        exp = s[st..i].to_string();
    }
    let mut float = frac.is_some() || !exp.is_empty();
    let mut unsigned = false;
    while i < b.len() && matches!(b[i], b'f' | b'F' | b'h' | b'H' | b'l' | b'L' | b'u' | b'U') {
        match b[i] {
            b'f' | b'F' | b'h' | b'H' => float = true,
            b'u' | b'U' => unsigned = true,
            _ => {}
        }
        i += 1;
    }
    if i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
        return Err(format!("bad number `{}`", &s[..=i]));
    }
    let int = if int.is_empty() { "0" } else { int };
    let t = if float {
        let frac = match frac {
            Some(f) if !f.is_empty() => f,
            _ => "0",
        };
        T::Float(format!("{int}.{frac}{exp}"))
    } else if unsigned {
        T::Int(format!("{int}u"))
    } else {
        T::Int(int.to_string())
    };
    Ok((t, i))
}

// ---- AST ----

#[derive(Clone, Debug)]
enum Expr {
    Int(String),
    Float(String),
    Bool(bool),
    Ident(String),
    Unary(&'static str, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    /// `instance.method(args)`.
    Method(String, String, Vec<Expr>),
    Member(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Cast(Ty, Box<Expr>),
    Assign(&'static str, Box<Expr>, Box<Expr>),
    /// `++` / `--`, pre or post (statements only).
    Step(&'static str, Box<Expr>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Decl(Ty, Vec<(String, Option<Expr>)>),
    /// `Struct a, b;`
    Instance(String, Vec<String>),
    Expr(Expr),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    For(Option<Box<Stmt>>, Option<Expr>, Option<Expr>, Box<Stmt>),
    While(Expr, Box<Stmt>),
    Return(Option<Expr>),
    Break,
    Continue,
    Discard,
    Block(Vec<Stmt>),
}

#[derive(Clone, Debug)]
struct Method {
    name: String,
    ret: Ty,
    params: Vec<(Ty, String)>,
    body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
struct StructDef {
    name: String,
    methods: Vec<Method>,
}

// ---- parser ----

struct Parser<'a> {
    t: &'a [Tok],
    i: usize,
    structs: HashSet<String>,
}

/// Binary operators by precedence (higher binds tighter).
fn binary_prec(p: &str) -> Option<(u8, &'static str)> {
    Some(match p {
        "||" => (1, "||"),
        "&&" => (2, "&&"),
        "|" => (3, "|"),
        "^" => (4, "^"),
        "&" => (5, "&"),
        "==" => (6, "=="),
        "!=" => (6, "!="),
        "<" => (7, "<"),
        ">" => (7, ">"),
        "<=" => (7, "<="),
        ">=" => (7, ">="),
        "<<" => (8, "<<"),
        ">>" => (8, ">>"),
        "+" => (9, "+"),
        "-" => (9, "-"),
        "*" => (10, "*"),
        "/" => (10, "/"),
        "%" => (10, "%"),
        _ => return None,
    })
}

const ASSIGN_OPS: &[&str] = &["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="];

impl<'a> Parser<'a> {
    fn peek(&self) -> &T {
        &self.t[self.i].t
    }
    fn peek_at(&self, k: usize) -> &T {
        &self.t[(self.i + k).min(self.t.len() - 1)].t
    }
    fn err<X>(&self, msg: impl std::fmt::Display) -> Result<X, String> {
        Err(format!("line {}: {msg}", self.t[self.i].line))
    }
    fn is(&self, p: &str) -> bool {
        matches!(self.peek(), T::P(q) if *q == p)
    }
    fn is_id(&self, s: &str) -> bool {
        matches!(self.peek(), T::Id(q) if q == s)
    }
    fn eat(&mut self, p: &str) -> bool {
        if self.is(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, p: &str) -> Result<(), String> {
        if self.eat(p) {
            Ok(())
        } else {
            self.err(format!("expected `{p}`, found {}", self.show()))
        }
    }
    fn show(&self) -> String {
        match self.peek() {
            T::Id(s) | T::Int(s) | T::Float(s) => format!("`{s}`"),
            T::P(p) => format!("`{p}`"),
            T::End => "the end".into(),
        }
    }
    fn ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            T::Id(s) => {
                self.i += 1;
                Ok(s)
            }
            _ => self.err(format!("expected a name, found {}", self.show())),
        }
    }

    /// The body: local struct definitions and statements.
    fn program(&mut self) -> Result<(Vec<StructDef>, Vec<Stmt>), String> {
        let mut structs = Vec::new();
        let mut body = Vec::new();
        while self.peek() != &T::End {
            if self.is_id("struct") {
                structs.push(self.struct_def()?);
            } else {
                body.push(self.stmt()?);
            }
        }
        Ok((structs, body))
    }

    fn struct_def(&mut self) -> Result<StructDef, String> {
        self.i += 1;
        let name = self.ident()?;
        self.structs.insert(name.clone());
        self.expect("{")?;
        let mut methods = Vec::new();
        while !self.eat("}") {
            while self.is_id("static") || self.is_id("inline") {
                self.i += 1;
            }
            let ret = self.type_name()?;
            let mname = self.ident()?;
            if !self.eat("(") {
                return self.err(format!("struct fields are not supported (`{name}::{mname}`)"));
            }
            let mut params = Vec::new();
            while !self.eat(")") {
                if !params.is_empty() {
                    self.expect(",")?;
                }
                if self.is_id("out") || self.is_id("inout") {
                    return self.err("`out` / `inout` parameters are not supported");
                }
                while self.is_id("in") || self.is_id("const") {
                    self.i += 1;
                }
                let t = self.type_name()?;
                if t.n == 0 {
                    return self.err("void parameter");
                }
                params.push((t, self.ident()?));
            }
            self.expect("{")?;
            let body = self.block_rest()?;
            methods.push(Method { name: mname, ret, params, body });
        }
        self.eat(";");
        Ok(StructDef { name, methods })
    }

    /// A value type or `void`.
    fn type_name(&mut self) -> Result<Ty, String> {
        let name = self.ident()?;
        if name == "void" {
            return Ok(Ty::void());
        }
        match hlsl_type(&name) {
            Some(Ok(t)) => Ok(t),
            Some(Err(e)) => self.err(e),
            None => self.err(format!("unknown type `{name}`")),
        }
    }

    /// Statements up to the closing `}` (the `{` already read).
    fn block_rest(&mut self) -> Result<Vec<Stmt>, String> {
        let mut v = Vec::new();
        while !self.eat("}") {
            if self.peek() == &T::End {
                return self.err("missing `}`");
            }
            v.push(self.stmt()?);
        }
        Ok(v)
    }

    fn is_decl_start(&self) -> bool {
        match self.peek() {
            T::Id(s) if s == "const" || s == "static" => true,
            T::Id(s) => (hlsl_type(s).is_some() || self.structs.contains(s)) && matches!(self.peek_at(1), T::Id(_)),
            _ => false,
        }
    }

    fn decl(&mut self) -> Result<Stmt, String> {
        while self.is_id("const") || self.is_id("static") {
            self.i += 1;
        }
        if let T::Id(s) = self.peek().clone() {
            if self.structs.contains(&s) {
                self.i += 1;
                let mut names = vec![self.ident()?];
                while self.eat(",") {
                    names.push(self.ident()?);
                }
                self.expect(";")?;
                return Ok(Stmt::Instance(s, names));
            }
        }
        let ty = self.type_name()?;
        if ty.n == 0 {
            return self.err("void variable");
        }
        let mut vars = Vec::new();
        loop {
            let name = self.ident()?;
            if self.is("[") {
                return self.err("arrays are not supported");
            }
            let init = if self.eat("=") { Some(self.assign()?) } else { None };
            vars.push((name, init));
            if self.eat(";") {
                break;
            }
            self.expect(",")?;
        }
        Ok(Stmt::Decl(ty, vars))
    }

    fn sub(&mut self) -> Result<Box<Stmt>, String> {
        Ok(Box::new(self.stmt()?))
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        if self.eat("{") {
            return Ok(Stmt::Block(self.block_rest()?));
        }
        if self.eat(";") {
            return Ok(Stmt::Block(Vec::new()));
        }
        if self.is("[") {
            // An attribute: [unroll], [loop], [branch], [unroll(4)] ...
            while !self.eat("]") {
                if self.peek() == &T::End {
                    return self.err("missing `]`");
                }
                self.i += 1;
            }
            return self.stmt();
        }
        let kw = match self.peek() {
            T::Id(s) => s.clone(),
            _ => String::new(),
        };
        match kw.as_str() {
            "if" => {
                self.i += 1;
                self.expect("(")?;
                let c = self.assign()?;
                self.expect(")")?;
                let a = self.sub()?;
                let b = if self.is_id("else") {
                    self.i += 1;
                    Some(self.sub()?)
                } else {
                    None
                };
                Ok(Stmt::If(c, a, b))
            }
            "for" => {
                self.i += 1;
                self.expect("(")?;
                let init = if self.eat(";") {
                    None
                } else if self.is_decl_start() {
                    Some(Box::new(self.decl()?))
                } else {
                    let e = self.assign()?;
                    self.expect(";")?;
                    Some(Box::new(Stmt::Expr(e)))
                };
                let cond = if self.is(";") { None } else { Some(self.assign()?) };
                self.expect(";")?;
                let step = if self.is(")") { None } else { Some(self.assign()?) };
                self.expect(")")?;
                Ok(Stmt::For(init, cond, step, self.sub()?))
            }
            "while" => {
                self.i += 1;
                self.expect("(")?;
                let c = self.assign()?;
                self.expect(")")?;
                Ok(Stmt::While(c, self.sub()?))
            }
            "return" => {
                self.i += 1;
                let e = if self.is(";") { None } else { Some(self.assign()?) };
                self.expect(";")?;
                Ok(Stmt::Return(e))
            }
            "break" | "continue" | "discard" => {
                self.i += 1;
                self.expect(";")?;
                Ok(match kw.as_str() {
                    "break" => Stmt::Break,
                    "continue" => Stmt::Continue,
                    _ => Stmt::Discard,
                })
            }
            "do" | "switch" | "struct" | "typedef" | "goto" => self.err(format!("`{kw}` is not supported")),
            _ if self.is_decl_start() => self.decl(),
            _ => {
                let e = self.assign()?;
                self.expect(";")?;
                Ok(Stmt::Expr(e))
            }
        }
    }

    fn assign(&mut self) -> Result<Expr, String> {
        let lhs = self.ternary()?;
        if let T::P(p) = self.peek() {
            if let Some(op) = ASSIGN_OPS.iter().find(|o| **o == *p) {
                self.i += 1;
                let rhs = self.assign()?;
                return Ok(Expr::Assign(op, Box::new(lhs), Box::new(rhs)));
            }
        }
        Ok(lhs)
    }

    fn ternary(&mut self) -> Result<Expr, String> {
        let c = self.binary(1)?;
        if self.eat("?") {
            let a = self.assign()?;
            self.expect(":")?;
            let b = self.assign()?;
            return Ok(Expr::Ternary(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn binary(&mut self, min: u8) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let Some((prec, op)) = (match self.peek() {
                T::P(p) => binary_prec(p),
                _ => None,
            }) else {
                break;
            };
            if prec < min {
                break;
            }
            self.i += 1;
            let rhs = self.binary(prec + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        for op in ["-", "+", "!", "~"] {
            if self.eat(op) {
                return Ok(Expr::Unary(op, Box::new(self.unary()?)));
            }
        }
        for op in ["++", "--"] {
            if self.eat(op) {
                return Ok(Expr::Step(op, Box::new(self.unary()?)));
            }
        }
        // A cast: `(float3)x`.
        if self.is("(") {
            if let (T::Id(name), T::P(")")) = (self.peek_at(1).clone(), self.peek_at(2).clone()) {
                if let Some(t) = hlsl_type(&name) {
                    let t = t.map_err(|e| format!("line {}: {e}", self.t[self.i].line))?;
                    self.i += 3;
                    return Ok(Expr::Cast(t, Box::new(self.unary()?)));
                }
            }
        }
        self.postfix()
    }

    fn args(&mut self) -> Result<Vec<Expr>, String> {
        let mut v = Vec::new();
        while !self.eat(")") {
            if !v.is_empty() {
                self.expect(",")?;
            }
            v.push(self.assign()?);
        }
        Ok(v)
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        loop {
            if self.eat(".") {
                let m = self.ident()?;
                if self.eat("(") {
                    let Expr::Ident(inst) = e else {
                        return self.err(format!("method call `.{m}(...)` on an expression"));
                    };
                    e = Expr::Method(inst, m, self.args()?);
                } else {
                    e = Expr::Member(Box::new(e), m);
                }
            } else if self.eat("[") {
                let i = self.assign()?;
                self.expect("]")?;
                e = Expr::Index(Box::new(e), Box::new(i));
            } else if self.is("++") || self.is("--") {
                let op = if self.eat("++") { "++" } else { self.eat("--"); "--" };
                e = Expr::Step(op, Box::new(e));
            } else {
                return Ok(e);
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            T::Int(s) => {
                self.i += 1;
                Ok(Expr::Int(s))
            }
            T::Float(s) => {
                self.i += 1;
                Ok(Expr::Float(s))
            }
            T::Id(s) => {
                self.i += 1;
                match s.as_str() {
                    "true" => return Ok(Expr::Bool(true)),
                    "false" => return Ok(Expr::Bool(false)),
                    _ => {}
                }
                if self.is("::") {
                    return self.err(format!("`{s}::` is not supported"));
                }
                if self.eat("(") {
                    return Ok(Expr::Call(s, self.args()?));
                }
                Ok(Expr::Ident(s))
            }
            T::P("(") => {
                self.i += 1;
                let e = self.assign()?;
                self.expect(")")?;
                Ok(e)
            }
            _ => self.err(format!("unexpected {}", self.show())),
        }
    }
}

// ---- WGSL emission ----

/// WGSL keywords and reserved words, plus the built-in names the
/// translation emits (a local of that name would shadow them).
const RESERVED: &[&str] = &[
    "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default", "diagnostic", "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop", "override", "requires", "return", "struct", "switch", "true", "var", "while",
    "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment", "async", "attribute", "auto", "await", "become", "binding_array", "cast", "catch", "class", "co_await", "co_return", "co_yield", "coherent", "column_major", "common", "compile", "compile_fragment", "concept", "const_cast", "consteval", "constexpr", "constinit", "crate", "debugger", "decltype", "delete", "demote", "demote_to_helper", "do", "dynamic_cast", "enum", "explicit", "export", "extends", "extern", "external", "fallthrough", "filter", "final", "finally", "friend", "from", "fxgroup", "get", "goto", "groupshared", "highp", "impl", "implements", "import", "inline", "instanceof", "interface", "layout", "lowp", "macro", "macro_rules", "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable", "namespace", "new", "nil", "noexcept", "noinline", "nointerpolation", "noperspective", "null", "nullptr", "of", "operator", "package", "packoffset", "partition", "pass", "patch", "pixelfragment", "precise", "precision", "premerge", "priv", "protected", "pub", "public", "readonly", "ref", "regardless", "register", "reinterpret_cast", "require", "resource", "restrict", "self", "set", "shared", "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast", "std", "subroutine", "super", "target", "template", "this", "thread_local", "throw", "trait", "try", "type", "typedef", "typeid", "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized", "use", "using", "varying", "virtual", "volatile", "wgsl", "where", "with", "writeonly", "yield",
    "f32", "i32", "u32", "f16", "bool", "vec2", "vec3", "vec4", "array", "ptr", "mix", "fract", "select", "inverseSqrt", "dpdx", "dpdy", "abs", "sign", "log2", "bitcast",
];

fn wgsl_ident(name: &str) -> String {
    if name.starts_with("__") {
        format!("u{name}")
    } else if RESERVED.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// An emitted expression: WGSL text, type and precedence (12: atomic,
/// 11: unary, 10..1: binary levels as in `binary_prec`).
#[derive(Clone, Debug)]
struct E {
    code: String,
    ty: Ty,
    prec: u8,
    /// A plain integer literal (converts to a float literal).
    int_lit: bool,
}

fn e(code: impl Into<String>, ty: Ty, prec: u8) -> E {
    E { code: code.into(), ty, prec, int_lit: false }
}

fn wrap(x: &E, min: u8) -> String {
    if x.prec < min {
        format!("({})", x.code)
    } else {
        x.code.clone()
    }
}

#[derive(Clone)]
enum Local {
    Var(String, Ty),
    Inst(String),
}

struct Sig {
    wgsl: String,
    params: Vec<Ty>,
    ret: Ty,
}

struct Tr {
    prefix: String,
    /// (struct, method) -> signature.
    sigs: HashMap<(String, String), Sig>,
    helpers: BTreeMap<String, String>,
    scopes: Vec<HashMap<String, Local>>,
    cur_struct: Option<String>,
    ret: Ty,
    lines: Vec<String>,
    indent: usize,
}

fn conv(x: E, to: Ty) -> Result<E, String> {
    if x.ty == to {
        return Ok(x);
    }
    if x.ty.n == 0 || to.n == 0 {
        return Err("a void function's result used as a value".into());
    }
    let digits = x.code.trim_start_matches('-');
    if x.int_lit && to.base == Base::Float && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        let f = e(format!("{}.0", x.code), Ty::f(1), if x.code.starts_with('-') { 11 } else { 12 });
        return conv(f, to);
    }
    let mut cur = x;
    if cur.ty.n > to.n {
        let sw = &"xyzw"[..to.n as usize];
        cur = e(format!("{}.{sw}", wrap(&cur, 12)), Ty::new(cur.ty.base, to.n), 12);
    } else if cur.ty.n < to.n && cur.ty.n != 1 {
        return Err(format!("cannot convert {} to {}", cur.ty.hlsl(), to.hlsl()));
    }
    if cur.ty.base != to.base {
        let t = Ty::new(to.base, cur.ty.n);
        cur = e(format!("{}({})", t.wgsl(), cur.code), t, 12);
    }
    if cur.ty.n == 1 && to.n > 1 {
        cur = e(format!("{}({})", to.wgsl(), cur.code), to, 12);
    }
    Ok(cur)
}

/// The type operands of a component-wise operation convert to: the widest
/// base; scalars splat, vectors truncate to the shortest.
fn common(args: &[&E], float: bool) -> Ty {
    let mut base = if float { Base::Float } else { Base::Bool };
    let mut n = 0u8;
    for a in args {
        base = base.max(a.ty.base);
        if a.ty.n > 1 {
            n = if n == 0 { a.ty.n } else { n.min(a.ty.n) };
        }
    }
    Ty::new(base, n.max(1))
}

fn swizzle_index(c: char) -> Option<usize> {
    "xyzw".find(c).or_else(|| "rgba".find(c))
}

const FLOAT_UNARY: &[(&str, &str)] = &[
    ("sin", "sin"), ("cos", "cos"), ("tan", "tan"), ("asin", "asin"), ("acos", "acos"), ("atan", "atan"), ("sinh", "sinh"), ("cosh", "cosh"), ("tanh", "tanh"),
    ("exp", "exp"), ("exp2", "exp2"), ("log", "log"), ("log2", "log2"), ("sqrt", "sqrt"), ("rsqrt", "inverseSqrt"), ("floor", "floor"), ("ceil", "ceil"),
    ("round", "round"), ("trunc", "trunc"), ("frac", "fract"), ("saturate", "saturate"), ("degrees", "degrees"), ("radians", "radians"),
    ("ddx", "dpdx"), ("ddy", "dpdy"), ("ddx_coarse", "dpdxCoarse"), ("ddy_coarse", "dpdyCoarse"), ("ddx_fine", "dpdxFine"), ("ddy_fine", "dpdyFine"), ("fwidth", "fwidth"),
];

impl Tr {
    fn line(&mut self, s: impl AsRef<str>) {
        self.lines.push(format!("{}{}", "    ".repeat(self.indent), s.as_ref()));
    }

    fn lookup(&self, name: &str) -> Option<Local> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    fn declare(&mut self, name: &str, l: Local) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), l);
    }

    fn value(&mut self, x: &Expr) -> Result<E, String> {
        let r = self.expr(x)?;
        if r.ty.n == 0 {
            return Err("a void function's result used as a value".into());
        }
        Ok(r)
    }

    fn cond(&mut self, x: &Expr) -> Result<String, String> {
        let c = self.value(x)?;
        if c.ty.n != 1 {
            return Err(format!("condition of type {}", c.ty.hlsl()));
        }
        Ok(conv(c, Ty::new(Base::Bool, 1))?.code)
    }

    fn expr(&mut self, x: &Expr) -> Result<E, String> {
        Ok(match x {
            Expr::Int(s) => {
                let ty = Ty::new(if s.ends_with('u') { Base::Uint } else { Base::Int }, 1);
                E { code: s.clone(), ty, prec: 12, int_lit: true }
            }
            Expr::Float(s) => e(s.clone(), Ty::f(1), 12),
            Expr::Bool(b) => e(b.to_string(), Ty::new(Base::Bool, 1), 12),
            Expr::Ident(name) => match self.lookup(name) {
                Some(Local::Var(w, ty)) => e(w, ty, 12),
                Some(Local::Inst(_)) => return Err(format!("struct value `{name}` used as a value")),
                None if name == "PI" => e("3.1415926535897932", Ty::f(1), 12),
                None => return Err(format!("unknown identifier `{name}`")),
            },
            Expr::Unary(op, a) => {
                let a = self.value(a)?;
                match *op {
                    "+" => a,
                    "-" => {
                        if a.ty.base == Base::Bool {
                            return Err("negated bool".into());
                        }
                        let inner = wrap(&a, 11);
                        let inner = if inner.starts_with('-') { format!("({inner})") } else { inner };
                        E { code: format!("-{inner}"), ty: a.ty, prec: 11, int_lit: a.int_lit }
                    }
                    "!" => {
                        let t = Ty::new(Base::Bool, a.ty.n);
                        let a = conv(a, t)?;
                        e(format!("!{}", wrap(&a, 11)), t, 11)
                    }
                    _ => {
                        if !matches!(a.ty.base, Base::Int | Base::Uint) {
                            return Err("`~` on a non-integer".into());
                        }
                        e(format!("~{}", wrap(&a, 11)), a.ty, 11)
                    }
                }
            }
            Expr::Binary(op, a, b) => {
                let (a, b) = (self.value(a)?, self.value(b)?);
                self.binary(op, a, b)?
            }
            Expr::Ternary(c, a, b) => {
                let c = self.value(c)?;
                let (a, b) = (self.value(a)?, self.value(b)?);
                let mut t = common(&[&a, &b], false);
                if c.ty.n > 1 {
                    t.n = c.ty.n;
                }
                // A scalar condition picks whole values; a vector one, components.
                let cn = if c.ty.n > 1 { t.n } else { 1 };
                let c = conv(c, Ty::new(Base::Bool, cn))?;
                let (a, b) = (conv(a, t)?, conv(b, t)?);
                e(format!("select({}, {}, {})", b.code, a.code, c.code), t, 12)
            }
            Expr::Call(name, args) => self.call(name, args)?,
            Expr::Method(inst, m, args) => {
                let s = match self.lookup(inst) {
                    Some(Local::Inst(s)) => s,
                    _ => return Err(format!("`{inst}.{m}(...)`: `{inst}` is not a struct instance")),
                };
                self.user_call(&s, m, args)?.ok_or(format!("struct `{s}` has no method `{m}`"))?
            }
            Expr::Member(a, sw) => {
                let a = self.value(a)?;
                self.swizzle(a, sw)?
            }
            Expr::Index(a, i) => {
                let a = self.value(a)?;
                if a.ty.n < 2 {
                    return Err("indexing a scalar".into());
                }
                let i = self.value(i)?;
                let i = if matches!(i.ty.base, Base::Int | Base::Uint) && i.ty.n == 1 { i } else { conv(i, Ty::new(Base::Int, 1))? };
                e(format!("{}[{}]", wrap(&a, 12), i.code), Ty::new(a.ty.base, 1), 12)
            }
            Expr::Cast(t, a) => {
                let a = self.value(a)?;
                conv(a, *t)?
            }
            Expr::Assign(..) | Expr::Step(..) => return Err("assignment inside an expression is not supported".into()),
        })
    }

    fn swizzle(&mut self, a: E, sw: &str) -> Result<E, String> {
        let idx: Vec<usize> = sw.chars().map(swizzle_index).collect::<Option<_>>().ok_or(format!("unknown member `.{sw}`"))?;
        if idx.len() > 4 || a.ty.n > 1 && idx.iter().any(|&i| i >= a.ty.n as usize) {
            return Err(format!("swizzle `.{sw}` of a {}", a.ty.hlsl()));
        }
        let n = idx.len() as u8;
        if a.ty.n == 1 {
            // `x.x` is `x`; `x.xxx` splats. Any component of a scalar reads
            // it (HLSL allows only x/r; an input the graph translation could
            // only give as a scalar stand-in keeps working).
            let t = Ty::new(a.ty.base, n);
            return conv(a, t);
        }
        let s: String = idx.iter().map(|&i| &"xyzw"[i..i + 1]).collect();
        Ok(e(format!("{}.{s}", wrap(&a, 12)), Ty::new(a.ty.base, n), 12))
    }

    fn binary(&mut self, op: &str, a: E, b: E) -> Result<E, String> {
        let (prec, op) = binary_prec(op).ok_or(format!("operator `{op}`"))?;
        match prec {
            9 | 10 => {
                let mut t = common(&[&a, &b], false);
                if t.base == Base::Bool {
                    t.base = Base::Int;
                }
                let at = Ty::new(t.base, if a.ty.n == 1 { 1 } else { t.n });
                let bt = Ty::new(t.base, if b.ty.n == 1 { 1 } else { t.n });
                let (a, b) = (conv(a, at)?, conv(b, bt)?);
                Ok(e(format!("{} {op} {}", wrap(&a, prec), wrap(&b, prec + 1)), t, prec))
            }
            6 | 7 => {
                let t = common(&[&a, &b], false);
                let (a, b) = (conv(a, t)?, conv(b, t)?);
                Ok(e(format!("{} {op} {}", wrap(&a, 8), wrap(&b, 8)), Ty::new(Base::Bool, t.n), prec))
            }
            1 | 2 => {
                let bt = Ty::new(Base::Bool, 1);
                if a.ty.n != 1 || b.ty.n != 1 {
                    return Err(format!("`{op}` on vectors"));
                }
                let (a, b) = (conv(a, bt)?, conv(b, bt)?);
                let w = |x: &E| if x.prec < 6 && x.prec != prec { format!("({})", x.code) } else { x.code.clone() };
                Ok(e(format!("{} {op} {}", w(&a), w(&b)), bt, prec))
            }
            8 => {
                let t = Ty::new(a.ty.base, a.ty.n);
                if !matches!(t.base, Base::Int | Base::Uint) {
                    return Err(format!("`{op}` on a non-integer"));
                }
                let b = conv(b, Ty::new(Base::Uint, t.n))?;
                Ok(e(format!("{} {op} {}", wrap(&a, 11), wrap(&b, 11)), t, prec))
            }
            _ => {
                let t = common(&[&a, &b], false);
                if t.base == Base::Float {
                    return Err(format!("`{op}` on floats"));
                }
                let (a, b) = (conv(a, t)?, conv(b, t)?);
                let l = if a.prec == prec { a.code.clone() } else { wrap(&a, 11) };
                Ok(e(format!("{l} {op} {}", wrap(&b, 11)), t, prec))
            }
        }
    }

    /// A call of a sibling method or `struct_name`'s method `m`.
    fn user_call(&mut self, struct_name: &str, m: &str, args: &[Expr]) -> Result<Option<E>, String> {
        let key = (struct_name.to_string(), m.to_string());
        let Some(sig) = self.sigs.get(&key) else { return Ok(None) };
        let (wgsl, params, ret) = (sig.wgsl.clone(), sig.params.clone(), sig.ret);
        if args.len() != params.len() {
            return Err(format!("`{m}` takes {} arguments, given {}", params.len(), args.len()));
        }
        let mut a = Vec::new();
        for (x, t) in args.iter().zip(&params) {
            let v = self.value(x)?;
            a.push(conv(v, *t)?.code);
        }
        Ok(Some(e(format!("{wgsl}({})", a.join(", ")), ret, 12)))
    }

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<E, String> {
        if let Some(s) = self.cur_struct.clone() {
            if let Some(r) = self.user_call(&s, name, args)? {
                return Ok(r);
            }
        }
        let mut a = Vec::new();
        for x in args {
            a.push(self.value(x)?);
        }
        if let Some(t) = hlsl_type(name) {
            return self.construct(t?, a);
        }
        self.intrinsic(name, a)
    }

    fn construct(&mut self, t: Ty, a: Vec<E>) -> Result<E, String> {
        if a.len() == 1 && (t.n == 1 || a[0].ty.n == 1 || a[0].ty.n >= t.n) {
            return conv(a.into_iter().next().unwrap(), t);
        }
        let total: u32 = a.iter().map(|x| x.ty.n as u32).sum();
        if t.n == 1 || total != t.n as u32 {
            return Err(format!("{} constructed from {total} components", t.hlsl()));
        }
        let mut parts = Vec::new();
        for x in a {
            let n = x.ty.n;
            parts.push(conv(x, Ty::new(t.base, n))?.code);
        }
        Ok(e(format!("{}({})", t.wgsl(), parts.join(", ")), t, 12))
    }

    fn helper(&mut self, name: &str, t: Ty) -> String {
        let fname = format!("{}_{name}{}", self.prefix, t.n);
        if !self.helpers.contains_key(&fname) {
            let ty = t.wgsl();
            let body = match name {
                // Unreal's (HLSL's) smoothstep is the plain formula, which is
                // also what reversed edges rely on; WGSL leaves those undefined.
                "smoothstep" => format!("fn {fname}(a: {ty}, b: {ty}, x: {ty}) -> {ty} {{\n    let t = saturate((x - a) / (b - a));\n    return t * t * (3.0 - 2.0 * t);\n}}\n"),
                _ => unreachable!(),
            };
            self.helpers.insert(fname.clone(), body);
        }
        fname
    }

    fn intrinsic(&mut self, name: &str, a: Vec<E>) -> Result<E, String> {
        let want = |k: usize| -> Result<(), String> {
            if a.len() == k {
                Ok(())
            } else {
                Err(format!("`{name}` takes {k} arguments, given {}", a.len()))
            }
        };
        let all = |a: Vec<E>, t: Ty| -> Result<Vec<E>, String> { a.into_iter().map(|x| conv(x, t)).collect() };
        if let Some((_, w)) = FLOAT_UNARY.iter().find(|(h, _)| *h == name) {
            want(1)?;
            let t = common(&[&a[0]], true);
            let x = conv(a.into_iter().next().unwrap(), t)?;
            return Ok(e(format!("{w}({})", x.code), t, 12));
        }
        Ok(match name {
            "abs" | "sign" => {
                want(1)?;
                let mut t = a[0].ty;
                if t.base == Base::Bool {
                    t.base = Base::Int;
                }
                let x = conv(a.into_iter().next().unwrap(), t)?;
                e(format!("{name}({})", x.code), t, 12)
            }
            "log10" => {
                want(1)?;
                let t = common(&[&a[0]], true);
                let x = conv(a.into_iter().next().unwrap(), t)?;
                e(format!("log2({}) * 0.30102999566398120", x.code), t, 10)
            }
            "rcp" => {
                want(1)?;
                let t = common(&[&a[0]], true);
                let x = conv(a.into_iter().next().unwrap(), t)?;
                e(format!("1.0 / {}", wrap(&x, 11)), t, 10)
            }
            "normalize" | "length" => {
                want(1)?;
                let t = common(&[&a[0]], true);
                let x = conv(a.into_iter().next().unwrap(), t)?;
                match (name, t.n) {
                    ("normalize", 1) => e(format!("sign({})", x.code), t, 12),
                    ("normalize", _) => e(format!("normalize({})", x.code), t, 12),
                    (_, 1) => e(format!("abs({})", x.code), t, 12),
                    _ => e(format!("length({})", x.code), Ty::f(1), 12),
                }
            }
            "any" | "all" => {
                want(1)?;
                let t = Ty::new(Base::Bool, a[0].ty.n);
                let x = conv(a.into_iter().next().unwrap(), t)?;
                if t.n == 1 {
                    x
                } else {
                    e(format!("{name}({})", x.code), Ty::new(Base::Bool, 1), 12)
                }
            }
            "dot" | "distance" => {
                want(2)?;
                let t = common(&[&a[0], &a[1]], name == "distance");
                let v = all(a, t)?;
                match (name, t.n) {
                    ("dot", 1) => e(format!("{} * {}", wrap(&v[0], 10), wrap(&v[1], 11)), t, 10),
                    ("distance", 1) => e(format!("abs({} - {})", v[0].code, wrap(&v[1], 10)), t, 12),
                    _ => e(format!("{name}({}, {})", v[0].code, v[1].code), Ty::new(t.base, 1), 12),
                }
            }
            "cross" => {
                want(2)?;
                let v = all(a, Ty::f(3))?;
                e(format!("cross({}, {})", v[0].code, v[1].code), Ty::f(3), 12)
            }
            "pow" | "atan2" | "step" | "reflect" | "min" | "max" | "fmod" => {
                want(2)?;
                let float = !matches!(name, "min" | "max");
                let mut t = common(&[&a[0], &a[1]], float);
                if t.base == Base::Bool {
                    t.base = Base::Int;
                }
                let v = all(a, t)?;
                if name == "fmod" {
                    // WGSL's float % is HLSL's fmod (truncated).
                    e(format!("{} % {}", wrap(&v[0], 10), wrap(&v[1], 11)), t, 10)
                } else {
                    e(format!("{name}({}, {})", v[0].code, v[1].code), t, 12)
                }
            }
            "lerp" | "clamp" | "smoothstep" | "mad" | "fma" => {
                want(3)?;
                let float = name != "clamp" && name != "mad";
                let mut t = common(&[&a[0], &a[1], &a[2]], float);
                if t.base == Base::Bool {
                    t.base = Base::Int;
                }
                let v = all(a, t)?;
                let (x, y, z) = (&v[0].code, &v[1].code, &v[2].code);
                match name {
                    "lerp" => e(format!("mix({x}, {y}, {z})"), t, 12),
                    "clamp" => e(format!("clamp({x}, {y}, {z})"), t, 12),
                    "smoothstep" => {
                        let f = self.helper("smoothstep", t);
                        e(format!("{f}({x}, {y}, {z})"), t, 12)
                    }
                    _ => e(format!("{} * {} + {}", wrap(&v[0], 10), wrap(&v[1], 11), wrap(&v[2], 10)), t, 9),
                }
            }
            "mul" => {
                want(2)?;
                if a[0].ty.n == 1 || a[1].ty.n == 1 {
                    let mut it = a.into_iter();
                    let (x, y) = (it.next().unwrap(), it.next().unwrap());
                    return self.binary("*", x, y);
                }
                return Err("`mul` of vectors or matrices is not supported".into());
            }
            _ => return Err(format!("function `{name}` is not supported")),
        })
    }

    /// The assignable expression `x` (a variable, a member or an index of
    /// one), and the components of a multi-component swizzle target.
    fn lvalue(&mut self, x: &Expr) -> Result<(E, Option<(E, Vec<usize>)>), String> {
        match x {
            Expr::Ident(name) => match self.lookup(name) {
                Some(Local::Var(w, ty)) => Ok((e(w, ty, 12), None)),
                _ => Err(format!("cannot assign to `{name}`")),
            },
            Expr::Member(b, sw) => {
                let (base, multi) = self.lvalue(b)?;
                if multi.is_some() {
                    return Err("assignment to a swizzle of a swizzle".into());
                }
                let r = self.swizzle(base.clone(), sw)?;
                let idx: Vec<usize> = sw.chars().filter_map(swizzle_index).collect();
                if base.ty.n == 1 {
                    if idx.len() == 1 {
                        return Ok((base, None));
                    }
                    return Err("assignment to a scalar's swizzle".into());
                }
                if idx.len() == 1 {
                    Ok((r, None))
                } else {
                    let mut seen = HashSet::new();
                    if !idx.iter().all(|i| seen.insert(*i)) {
                        return Err(format!("assignment to `.{sw}` (repeated component)"));
                    }
                    Ok((r, Some((base, idx))))
                }
            }
            Expr::Index(..) => Ok((self.value(x)?, None)),
            _ => Err("assignment to an expression".into()),
        }
    }

    /// An expression statement, without its `;`.
    fn expr_stmt(&mut self, x: &Expr) -> Result<String, String> {
        match x {
            Expr::Assign(op, l, r) => {
                let (target, multi) = self.lvalue(l)?;
                let rhs = self.value(r)?;
                let val = if *op == "=" { rhs } else { self.binary(&op[..op.len() - 1], target.clone(), rhs)? };
                let val = conv(val, target.ty)?;
                Ok(match multi {
                    None => format!("{} = {}", target.code, val.code),
                    Some((base, idx)) => {
                        let mut s = format!("{{ let swz = {};", val.code);
                        for (k, i) in idx.iter().enumerate() {
                            s += &format!(" {}.{} = swz.{};", wrap(&base, 12), &"xyzw"[*i..i + 1], &"xyzw"[k..k + 1]);
                        }
                        s + " }"
                    }
                })
            }
            Expr::Step(op, l) => {
                let (target, multi) = self.lvalue(l)?;
                if multi.is_some() || target.ty.n != 1 {
                    return Err(format!("`{op}` on a vector"));
                }
                if matches!(target.ty.base, Base::Int | Base::Uint) {
                    Ok(format!("{}{op}", target.code))
                } else {
                    let one = conv(E { code: "1".into(), ty: Ty::new(Base::Int, 1), prec: 12, int_lit: true }, target.ty)?;
                    Ok(format!("{} = {} {} {}", target.code, target.code, &op[..1], one.code))
                }
            }
            Expr::Call(..) | Expr::Method(..) => {
                let r = self.expr(x)?;
                Ok(if r.ty.n == 0 { r.code } else { format!("_ = {}", r.code) })
            }
            _ => {
                let r = self.value(x)?;
                Ok(format!("_ = {}", r.code))
            }
        }
    }

    fn body(&mut self, s: &Stmt) -> Result<(), String> {
        self.indent += 1;
        self.scopes.push(HashMap::new());
        let r = match s {
            Stmt::Block(v) => v.iter().try_for_each(|s| self.stmt(s)),
            s => self.stmt(s),
        };
        self.scopes.pop();
        self.indent -= 1;
        r
    }

    fn decl_one(&mut self, ty: Ty, name: &str, init: &Option<Expr>) -> Result<String, String> {
        let w = wgsl_ident(name);
        let s = match init {
            Some(x) => {
                let v = self.value(x)?;
                format!("var {w}: {} = {}", ty.wgsl(), conv(v, ty)?.code)
            }
            None => format!("var {w}: {}", ty.wgsl()),
        };
        self.declare(name, Local::Var(w, ty));
        Ok(s)
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Decl(ty, vars) => {
                for (name, init) in vars {
                    let l = self.decl_one(*ty, name, init)?;
                    self.line(format!("{l};"));
                }
            }
            Stmt::Instance(st, names) => {
                for n in names {
                    self.declare(n, Local::Inst(st.clone()));
                }
            }
            Stmt::Expr(x) => {
                let l = self.expr_stmt(x)?;
                self.line(if l.ends_with('}') { l } else { format!("{l};") });
            }
            Stmt::If(c, a, b) => {
                let c = self.cond(c)?;
                self.line(format!("if ({c}) {{"));
                self.body(a)?;
                if let Some(b) = b {
                    self.line("} else {");
                    self.body(b)?;
                }
                self.line("}");
            }
            Stmt::For(init, c, step, b) => {
                self.scopes.push(HashMap::new());
                let init = match init.as_deref() {
                    None => String::new(),
                    Some(Stmt::Decl(ty, vars)) if vars.len() == 1 => self.decl_one(*ty, &vars[0].0, &vars[0].1)?,
                    Some(Stmt::Expr(x)) => self.expr_stmt(x)?,
                    Some(_) => return Err("unsupported `for` initializer".into()),
                };
                let c = match c {
                    Some(c) => self.cond(c)?,
                    None => String::new(),
                };
                let step = match step {
                    Some(x) => self.expr_stmt(x)?,
                    None => String::new(),
                };
                if init.contains(';') || step.contains(';') {
                    return Err("unsupported `for` initializer or step".into());
                }
                self.line(format!("for ({init}; {c}; {step}) {{"));
                self.body(b)?;
                self.line("}");
                self.scopes.pop();
            }
            Stmt::While(c, b) => {
                let c = self.cond(c)?;
                self.line(format!("while ({c}) {{"));
                self.body(b)?;
                self.line("}");
            }
            Stmt::Return(x) => match (x, self.ret.n) {
                (None, 0) => self.line("return;"),
                (Some(x), n) if n > 0 => {
                    let v = self.value(x)?;
                    let v = conv(v, self.ret)?;
                    self.line(format!("return {};", v.code));
                }
                _ => return Err("return value does not match the function".into()),
            },
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Discard => self.line("discard;"),
            Stmt::Block(v) => {
                self.line("{");
                self.body(&Stmt::Block(v.clone()))?;
                self.line("}");
            }
        }
        Ok(())
    }

    /// A WGSL function from parameters and a body.
    fn function(&mut self, wgsl: &str, params: &[(Ty, String)], ret: Ty, body: &[Stmt], strukt: Option<&str>) -> Result<String, String> {
        self.scopes = vec![HashMap::new()];
        self.cur_struct = strukt.map(String::from);
        self.ret = ret;
        self.lines.clear();
        self.indent = 1;
        let mut assigned = HashSet::new();
        for s in body {
            assigned_roots(s, &mut assigned);
        }
        let mut sig = Vec::new();
        for (t, name) in params {
            let w = wgsl_ident(name);
            if assigned.contains(name) {
                sig.push(format!("{w}_in: {}", t.wgsl()));
                self.line(format!("var {w}: {} = {w}_in;", t.wgsl()));
            } else {
                sig.push(format!("{w}: {}", t.wgsl()));
            }
            self.declare(name, Local::Var(w, *t));
        }
        for s in body {
            self.stmt(s)?;
        }
        if ret.n > 0 && !matches!(body.last(), Some(Stmt::Return(_))) {
            self.line(format!("return {}();", ret.wgsl()));
        }
        let ret_s = if ret.n == 0 { String::new() } else { format!(" -> {}", ret.wgsl()) };
        let mut out = format!("fn {wgsl}({}){ret_s} {{\n", sig.join(", "));
        for l in self.lines.drain(..) {
            out += &l;
            out.push('\n');
        }
        out += "}\n";
        Ok(out)
    }
}

fn root(x: &Expr) -> Option<&str> {
    match x {
        Expr::Ident(n) => Some(n),
        Expr::Member(b, _) | Expr::Index(b, _) => root(b),
        _ => None,
    }
}

fn assigned_in(x: &Expr, out: &mut HashSet<String>) {
    match x {
        Expr::Assign(_, l, r) => {
            if let Some(n) = root(l) {
                out.insert(n.to_string());
            }
            assigned_in(r, out);
        }
        Expr::Step(_, l) => {
            if let Some(n) = root(l) {
                out.insert(n.to_string());
            }
        }
        _ => {}
    }
}

fn assigned_roots(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Expr(x) => assigned_in(x, out),
        Stmt::If(_, a, b) => {
            assigned_roots(a, out);
            if let Some(b) = b {
                assigned_roots(b, out);
            }
        }
        Stmt::For(i, _, st, b) => {
            if let Some(i) = i {
                assigned_roots(i, out);
            }
            if let Some(st) = st {
                assigned_in(st, out);
            }
            assigned_roots(b, out);
        }
        Stmt::While(_, b) => assigned_roots(b, out),
        Stmt::Block(v) => v.iter().for_each(|s| assigned_roots(s, out)),
        _ => {}
    }
}

/// Translates a Custom node's HLSL `code` to module-level WGSL: helper
/// functions (the body's struct methods, as `<fn_name>_<Struct>_<method>`)
/// and `fn <fn_name>(<inputs>) -> <output>`. `inputs` are the node's input
/// names with their component counts (1..=4, floats); `output_components`
/// is the node's output type (CMOT_Float1..4).
pub fn translate_custom(code: &str, inputs: &[(String, usize)], output_components: usize, fn_name: &str) -> Result<String, String> {
    if !(1..=4).contains(&output_components) {
        return Err(format!("output with {output_components} components"));
    }
    let toks = tokenize(code)?;
    let mut p = Parser { t: &toks, i: 0, structs: HashSet::new() };
    let (structs, body) = p.program()?;
    let mut tr = Tr {
        prefix: fn_name.to_string(),
        sigs: HashMap::new(),
        helpers: BTreeMap::new(),
        scopes: Vec::new(),
        cur_struct: None,
        ret: Ty::void(),
        lines: Vec::new(),
        indent: 1,
    };
    for s in &structs {
        for m in &s.methods {
            let key = (s.name.clone(), m.name.clone());
            if tr.sigs.contains_key(&key) {
                return Err(format!("overloaded method `{}::{}` is not supported", s.name, m.name));
            }
            let wgsl = format!("{fn_name}_{}_{}", s.name, m.name);
            tr.sigs.insert(key, Sig { wgsl, params: m.params.iter().map(|p| p.0).collect(), ret: m.ret });
        }
    }
    let mut funcs = Vec::new();
    for s in &structs {
        for m in &s.methods {
            let wgsl = tr.sigs[&(s.name.clone(), m.name.clone())].wgsl.clone();
            let f = tr.function(&wgsl, &m.params, m.ret, &m.body, Some(&s.name)).map_err(|e| format!("{}::{}: {e}", s.name, m.name))?;
            funcs.push(f);
        }
    }
    let mut params = Vec::new();
    for (name, n) in inputs {
        if !(1..=4).contains(n) {
            return Err(format!("input `{name}` with {n} components"));
        }
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.starts_with(|c: char| c.is_ascii_digit()) {
            return Err(format!("input name `{name}`"));
        }
        params.push((Ty::f(*n as u8), name.clone()));
    }
    funcs.push(tr.function(fn_name, &params, Ty::f(output_components as u8), &body, None)?);
    let mut out = String::new();
    for h in tr.helpers.values() {
        out += h;
        out.push('\n');
    }
    out += &funcs.join("\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates `src` (plus a fragment entry point calling
    /// nothing) with naga.
    pub(crate) fn validate(src: &str) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{}\n{src}", e.emit_to_string(src)),
        };
        if let Err(e) = naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all()).validate(&module) {
            panic!("{}\n{src}", e.emit_to_string(src));
        }
    }

    fn inputs(v: &[(&str, usize)]) -> Vec<(String, usize)> {
        v.iter().map(|(n, k)| (n.to_string(), *k)).collect()
    }

    const PORTAL: &str = r#"
struct PRTN {
  float h1(float2 p) { float3 q = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973)); q += dot(q, q.yzx + 33.33); return frac((q.x + q.y) * q.z); }
  float vn2(float2 p) { float2 i = floor(p); float2 f = frac(p); f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(h1(i), h1(i + float2(1, 0)), f.x), lerp(h1(i + float2(0, 1)), h1(i + float2(1, 1)), f.x), f.y); }
  float vnp(float2 p, float per) { float2 i = floor(p); float2 f = frac(p); f = f * f * (3.0 - 2.0 * f);
    float x0 = i.x - per * floor(i.x / per); float x1 = x0 + 1.0; x1 -= per * floor(x1 / per);
    return lerp(lerp(h1(float2(x0, i.y)), h1(float2(x1, i.y)), f.x), lerp(h1(float2(x0, i.y + 1.0)), h1(float2(x1, i.y + 1.0)), f.x), f.y); }
  float fbm2(float2 p) { float s = 0.0, a = 0.5; for (int o = 0; o < 4; o++) { s += a * vn2(p); p = p * 2.03 + float2(17.1, 9.2); a *= 0.5; } return s / 0.9375; }
  float openv(float3 o, float t) { float d = o.z; float k = d > 0.001 ? saturate((t - o.x) / d) : 1.0; return o.y > 0.5 ? 1.0 - k : k; }
};
PRTN Nz;

float T = Tm;
float open = Nz.openv(O, T);
if (open <= 0.0005) return float4(0, 0, 0, 0);
float flash = sin(open * 3.14159) * (O.y > 0.5 ? 0.5 : 1.0) * (open < 0.999 ? 1.0 : 0.0);
float2 p = (UV - 0.5) * 2.0;
float r = length(p);
float a = atan2(p.y, p.x);
float wob = Nz.fbm2(p * 2.6 + float2(T * 0.8, -T * 0.55)) - 0.5;
float ring = exp(-wob * wob);
float inside = smoothstep(1.03, 0.9, r);
float s1 = sin(a), c1 = cos(a);
float arms = 0.5 + 0.5 * sin(a * 3 + log(max(r, 0.02)) * 1.1);
float wisps = pow(saturate(Nz.vnp(p, 4.0) * 1.45 - 0.38), 2.0);
float3 colIn = ColorA * wisps * 1.3 + ColorA * ring;
colIn *= inside * (1.0 - flash);
float3 hot = lerp(ColorA, float3(1, 1, 1), 0.65);
float3 zero = 0.0;
return float4(colIn + hot * s1 * c1 + zero, inside * saturate(open * 2.0));
"#;

    #[test]
    fn portal_body_translates_and_validates() {
        let w = translate_custom(PORTAL, &inputs(&[("UV", 2), ("Tm", 1), ("O", 3), ("ColorA", 3)]), 4, "custom_7").unwrap();
        validate(&w);
        for s in [
            "fn custom_7_PRTN_fbm2(p_in: vec2<f32>) -> f32 {",
            "for (var o: i32 = 0; o < 4; o++) {",
            "custom_7_PRTN_vn2(p)",
            "var a: f32 = 0.5;",
            "fn custom_7(UV: vec2<f32>, Tm: f32, O: vec3<f32>, ColorA: vec3<f32>) -> vec4<f32> {",
            "custom_7_PRTN_openv(O, T)",
            "select(1.0, 0.5, O.y > 0.5)",
            "if (open <= 0.0005) {",
            "return vec4<f32>(0.0, 0.0, 0.0, 0.0);",
            "var zero: vec3<f32> = vec3<f32>(0.0);",
            "custom_7_smoothstep1(1.03, 0.9, r)",
            "a * 3.0",
            "mix(ColorA, vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.65))",
        ] {
            assert!(w.contains(s), "missing `{s}`:\n{w}");
        }
    }

    #[test]
    fn conversions_swizzles_and_control_flow() {
        let code = r#"
// comment
float u = UV.x, v = 1.0 - UV.y;   /* block */
float3 c = 1;
c.xz = float2(u, v);
c.y += 2;
int n = 3;
float k = n / 2;
float4 big = float4(c, 1.0);
float3 back = big;
float s = back;
[unroll] for (int i = 0; i < n; ++i) { if (i == 1) continue; k += i; }
while (k > 10.0) { k -= 1.0; }
bool b = u > 0.5 && v < 0.5 || !(k > 1.0);
c = b ? c : c.zyx;
c.x = s.x + (float)n + max(n, 2) + fmod(k, 2.0) + rcp(2.0) + log10(100.0);
k++;
return c * k + PI;
"#;
        let w = translate_custom(code, &inputs(&[("UV", 2)]), 3, "custom_1").unwrap();
        validate(&w);
        for s in ["var c: vec3<f32> = vec3<f32>(1.0);", "{ let swz = vec2<f32>(u, v); c.x = swz.x; c.z = swz.y; }", "var k: f32 = f32(n / 2);", "var back: vec3<f32> = big.xyz;", "var s: f32 = back.x;"] {
            assert!(w.contains(s), "missing `{s}`:\n{w}");
        }
        // A scalar output from a float1 node; a reserved word as a name.
        let w = translate_custom("float2 p = (UV - 0.5) * 2.0; float r2 = dot(p, p); float target = exp(-r2 * 7.0); return target * saturate(1.0 - r2);", &inputs(&[("UV", 2)]), 1, "custom_2").unwrap();
        validate(&w);
        assert!(w.contains("var target_: f32"), "{w}");
        // An if block with its own return, a float3 output.
        let w = translate_custom("if (UV.y < 1.5) { float3 c = UV.x; return c * 2; } return float3(0, 0, 0);", &inputs(&[("UV", 2)]), 3, "custom_3").unwrap();
        validate(&w);
    }

    #[test]
    fn unsupported_code_is_an_error() {
        for (code, why) in [
            ("return Parameters.TexCoords[0].x;", "unknown identifier `Parameters`"),
            ("float3x3 m; return 1;", "matrix"),
            ("#define X 1\nreturn X;", "preprocessor"),
            ("float a[4]; return 1;", "arrays"),
            ("return foo(1);", "function `foo`"),
            ("struct S { float x; }; return 1;", "fields"),
        ] {
            let e = translate_custom(code, &[], 1, "c").unwrap_err();
            assert!(e.contains(why), "{code}: {e}");
        }
    }
}
