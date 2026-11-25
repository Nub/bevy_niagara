//! Best-effort translation of Niagara custom HLSL (scratch pad nodes) to a
//! WGSL custom module.
//!
//! Feasible when every pin of the HLSL node reads a particle attribute we
//! have, a module input (`Module.X`) or `Engine.DeltaTime` / `Engine.Time`,
//! every output writes a particle attribute we have, and the code calls no
//! data interface functions. The body keeps working in Unreal units: the
//! wrapper converts attributes to centimetres and Z up on the way in, and
//! back on the way out. The text translation covers declarations, vector
//! constructors, single-statement `if` / `for` bodies, `?:` and the common
//! intrinsics; the result is validated by the caller (naga), so anything
//! else fails there.

use std::fmt::Write;

use crate::fx::{quote, Fx, Lit};
use crate::niagara::{HlslNode, NInput, NSystem, NValue};

#[derive(Clone, Debug)]
pub struct Translated {
    pub name: String,
    /// The `custom_modules` entry, as RON.
    pub source: String,
    /// Inputs to pass where the module is called (links to user parameters).
    pub call_inputs: Vec<(String, Fx)>,
}

fn wgsl_type(niagara: &str) -> Option<&'static str> {
    Some(match niagara {
        "NiagaraFloat" => "f32",
        "NiagaraInt32" => "i32",
        "NiagaraBool" => "bool",
        "Vector2f" | "Vector2D" => "vec2<f32>",
        "Vector3f" | "Vector" | "NiagaraPosition" => "vec3<f32>",
        "Vector4f" | "Vector4" | "LinearColor" | "Quat4f" => "vec4<f32>",
        _ => return None,
    })
}

/// How an attribute's value converts to Unreal units.
#[derive(Clone, Copy, PartialEq)]
enum Conv {
    None,
    /// cm and Z up: swap Y/Z and scale.
    Length,
    /// Swap Y/Z only.
    Axis,
}

fn attribute(name: &str) -> Option<(String, Conv)> {
    let rest = name.strip_prefix("Particles.")?;
    let b = fx_core::params::builtins(fx_core::params::Namespace::Particles).iter().find(|b| b.name == rest)?;
    let conv = match rest {
        "Position" | "Velocity" | "Physics.Force" | "Previous.Position" | "Previous.Velocity" => Conv::Length,
        "SpriteSize" | "RibbonWidth" | "CameraOffset" | "LightRadius" => Conv::Length,
        "Scale" | "SpriteFacing" | "SpriteAlignment" => Conv::Axis,
        _ => Conv::None,
    };
    let _ = b;
    Some((rest.replace('.', "_"), conv))
}

fn sanitize(name: &str) -> String {
    let mut s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if s.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

fn to_unreal(expr: &str, ty: &str, conv: Conv) -> String {
    match (conv, ty) {
        (Conv::Length, "vec3<f32>") => format!("vec3<f32>({expr}.x, {expr}.z, {expr}.y) * 100.0"),
        (Conv::Axis, "vec3<f32>") => format!("vec3<f32>({expr}.x, {expr}.z, {expr}.y)"),
        (Conv::Length, _) => format!("{expr} * 100.0"),
        _ => expr.to_string(),
    }
}

fn from_unreal(expr: &str, ty: &str, conv: Conv) -> String {
    match (conv, ty) {
        (Conv::Length, "vec3<f32>") => format!("vec3<f32>({expr}.x, {expr}.z, {expr}.y) * 0.01"),
        (Conv::Axis, "vec3<f32>") => format!("vec3<f32>({expr}.x, {expr}.z, {expr}.y)"),
        (Conv::Length, _) => format!("{expr} * 0.01"),
        _ => expr.to_string(),
    }
}

fn lit_for(v: &NValue) -> Option<Lit> {
    Some(match v {
        NValue::Float(x) => Lit::Float(*x),
        NValue::Int(x) => Lit::Int(*x),
        NValue::Bool(x) => Lit::Bool(*x),
        NValue::Vec2(x) => Lit::Vec2(*x),
        NValue::Vec3(x) | NValue::Position(x) => Lit::Vec3(*x),
        NValue::Vec4(x) | NValue::Quat(x) => Lit::Vec4(*x),
        NValue::Color(x) => Lit::Color(*x),
        _ => return None,
    })
}

fn default_lit(wgsl: &str) -> Lit {
    match wgsl {
        "i32" => Lit::Int(0),
        "bool" => Lit::Bool(false),
        "vec2<f32>" => Lit::Vec2([0.0; 2]),
        "vec3<f32>" => Lit::Vec3([0.0; 3]),
        "vec4<f32>" => Lit::Vec4([0.0; 4]),
        _ => Lit::Float(0.0),
    }
}

/// Translates a scratch pad's custom HLSL nodes into one custom module.
pub fn translate(name: &str, nodes: &[HlslNode], inputs: &[(String, NInput)], sys: &NSystem) -> Result<Translated, String> {
    let mname = sanitize(name);
    let mut decl_inputs: Vec<(String, Lit)> = Vec::new();
    let mut call_inputs: Vec<(String, Fx)> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();
    let mut body = String::new();
    for (k, node) in nodes.iter().enumerate() {
        if node.code.contains(".Get") || node.code.contains("<Attribute=") || node.code.contains("Texture") {
            return Err("calls data interface functions (particle reads, textures)".into());
        }
        let _ = writeln!(body, "    {{ // HLSL node {}", k + 1);
        for (pin, ty, src) in &node.inputs {
            let t = wgsl_type(ty).ok_or_else(|| format!("input {pin} of type {ty}"))?;
            let p = sanitize(pin);
            let src = src.as_deref().ok_or_else(|| format!("input {pin} is not wired to a parameter"))?;
            let expr = if let Some(rest) = src.strip_prefix("Module.") {
                let iname = sanitize(rest);
                if !decl_inputs.iter().any(|(n, _)| *n == iname) {
                    let (lit, call) = match inputs.iter().find(|(n, _)| n == rest).map(|(_, v)| v) {
                        Some(NInput::Value(v)) => (lit_for(v).unwrap_or_else(|| default_lit(t)), None),
                        Some(NInput::Link(l)) if l.starts_with("User.") => {
                            let user = sys.user_parameters.iter().find(|(n, _)| n == l).map(|(_, v)| v);
                            let lit = user.and_then(lit_for).unwrap_or_else(|| default_lit(t));
                            let lt = lit.ty().name();
                            (lit, Some(Fx::Link(lt.into(), l.clone())))
                        }
                        Some(other) => return Err(format!("module input {rest} is {other:?}")),
                        None => (default_lit(t), None),
                    };
                    decl_inputs.push((iname.clone(), lit));
                    if let Some(c) = call {
                        call_inputs.push((iname.clone(), c));
                    }
                }
                iname
            } else if src == "Engine.DeltaTime" {
                "DeltaTime".into()
            } else if src == "Engine.Time" {
                "Time".into()
            } else if let Some((local, conv)) = attribute(src) {
                to_unreal(&local, t, conv)
            } else {
                return Err(format!("input {pin} reads {src}, which has no equivalent here"));
            };
            let _ = writeln!(body, "    var {p}: {t} = {expr};");
        }
        let mut writes = Vec::new();
        for (pin, ty, dst) in &node.outputs {
            let t = wgsl_type(ty).ok_or_else(|| format!("output {pin} of type {ty}"))?;
            let p = sanitize(pin);
            let dst = dst.as_deref().ok_or_else(|| format!("output {pin} is not wired to a parameter"))?;
            let (local, conv) = attribute(dst).ok_or_else(|| format!("output {pin} writes {dst}, which has no equivalent here"))?;
            let _ = writeln!(body, "    var {p}: {t};");
            writes.push(format!("    {local} = {};", from_unreal(&p, t, conv)));
            if !outputs.contains(&dst.to_string()) {
                outputs.push(dst.to_string());
            }
        }
        for line in translate_body(&node.code)?.lines() {
            let _ = writeln!(body, "    {line}");
        }
        for w in writes {
            let _ = writeln!(body, "{w}");
        }
        let _ = writeln!(body, "    }}");
    }
    if outputs.is_empty() {
        return Err("writes no particle attribute".into());
    }
    let mut src = String::new();
    let _ = writeln!(src, "(");
    let _ = writeln!(src, "    name: {},", quote(&mname));
    let ins: Vec<String> = decl_inputs.iter().map(|(n, l)| format!("{}: {}", quote(n), crate::fx::write_lit(l))).collect();
    let _ = writeln!(src, "    inputs: {{ {} }},", ins.join(", "));
    let outs: Vec<String> = outputs.iter().map(|o| quote(o)).collect();
    let _ = writeln!(src, "    outputs: [{}],", outs.join(", "));
    let _ = writeln!(src, "    wgsl: r#\"");
    src.push_str(&body);
    let _ = writeln!(src, "\"#,");
    let _ = writeln!(src, "),");
    Ok(Translated {
        name: mname,
        source: src,
        call_inputs,
    })
}

/// Translates an HLSL function body to WGSL statements.
pub fn translate_body(code: &str) -> Result<String, String> {
    let mut lines = Vec::new();
    for line in code.replace("\r\n", "\n").lines() {
        let t = line.trim();
        if t.starts_with("#pragma") || t.starts_with("#if") || t.starts_with("#endif") {
            continue;
        }
        if t.starts_with('#') {
            return Err(format!("preprocessor directive `{t}`"));
        }
        lines.push(line.to_string());
    }
    let mut s = lines.join("\n");
    for bad in ["float3x3", "float4x4", "mul(", "fmod(", "struct ", "return "] {
        if s.contains(bad) {
            return Err(format!("uses `{}`", bad.trim()));
        }
    }
    // Literal suffixes: 1.0f -> 1.0.
    s = strip_float_suffix(&s);
    for (from, to) in [
        ("float2(", "vec2<f32>("),
        ("float3(", "vec3<f32>("),
        ("float4(", "vec4<f32>("),
        ("int2(", "vec2<i32>("),
        ("int3(", "vec3<i32>("),
        ("lerp(", "mix("),
        ("frac(", "fract("),
        ("rsqrt(", "inverseSqrt("),
    ] {
        s = replace_word(&s, from, to);
    }
    s = replace_saturate(&s);
    s = s.replace("for(int ", "for (var ").replace("for (int ", "for (var ");
    s = declarations(&s);
    s = brace_bodies(&s)?;
    s = ternaries(&s);
    Ok(s)
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Replaces `from` where it is not part of a longer identifier.
fn replace_word(s: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while let Some(k) = s[i..].find(from) {
        let at = i + k;
        let prev = s[..at].chars().next_back();
        out.push_str(&s[i..at]);
        if prev.is_some_and(is_ident) {
            out.push_str(from);
        } else {
            out.push_str(to);
        }
        i = at + from.len();
    }
    out.push_str(&s[i..]);
    out
}

fn strip_float_suffix(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &c) in b.iter().enumerate() {
        if c == 'f' && i > 0 && b[i - 1].is_ascii_digit() && !b.get(i + 1).copied().is_some_and(is_ident) {
            // A digit run before it that is part of a number (not an identifier).
            let mut j = i;
            while j > 0 && (b[j - 1].is_ascii_digit() || b[j - 1] == '.') {
                j -= 1;
            }
            if j == 0 || !is_ident(b[j - 1]) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// The index of the parenthesis closing the one at `open`.
fn close_paren(s: &[char], open: usize) -> Option<usize> {
    let mut depth = 0;
    for (k, &c) in s.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(k);
                }
            }
            _ => {}
        }
    }
    None
}

fn replace_saturate(s: &str) -> String {
    let mut s = s.to_string();
    while let Some(at) = s.find("saturate(") {
        let chars: Vec<char> = s.chars().collect();
        let ci = s[..at].chars().count();
        let open = ci + "saturate".len();
        let Some(close) = close_paren(&chars, open) else { break };
        let inner: String = chars[open + 1..close].iter().collect();
        let before: String = chars[..ci].iter().collect();
        let after: String = chars[close + 1..].iter().collect();
        s = format!("{before}clamp({inner}, 0.0, 1.0){after}");
    }
    s
}

/// `float3 x = ...;` -> `var x: vec3<f32> = ...;`.
fn declarations(s: &str) -> String {
    const TYPES: &[(&str, &str)] = &[
        ("float4", "vec4<f32>"),
        ("float3", "vec3<f32>"),
        ("float2", "vec2<f32>"),
        ("float", "f32"),
        ("int", "i32"),
        ("uint", "u32"),
        ("bool", "bool"),
    ];
    let mut out = String::new();
    for line in s.lines() {
        let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
        let mut t = line.trim_start();
        let is_const = t.starts_with("const ");
        if is_const {
            t = t["const ".len()..].trim_start();
        }
        let mut done = false;
        for (h, w) in TYPES {
            if let Some(rest) = t.strip_prefix(h) {
                if !rest.starts_with(' ') {
                    continue;
                }
                let rest = rest.trim_start();
                let name: String = rest.chars().take_while(|c| is_ident(*c)).collect();
                if name.is_empty() {
                    continue;
                }
                let after = rest[name.len()..].trim_start();
                if after.starts_with('=') || after.starts_with(';') {
                    let _ = writeln!(out, "{indent}var {name}: {w}{}{after}", if after.starts_with('=') { " " } else { "" });
                    done = true;
                    break;
                }
            }
        }
        if !done {
            // Declarations as a `for` initialiser: `for (var i=0; ...`.
            let l = line.replace("for (var ", "for (var ");
            let l = fix_for_init(&l);
            let _ = writeln!(out, "{l}");
        }
    }
    out
}

/// `for (var i=0;` -> `for (var i: i32 = 0;`.
fn fix_for_init(line: &str) -> String {
    let Some(at) = line.find("for (var ") else { return line.to_string() };
    let head = &line[..at + "for (var ".len()];
    let rest = &line[at + "for (var ".len()..];
    let name: String = rest.chars().take_while(|c| is_ident(*c)).collect();
    let after = rest[name.len()..].trim_start();
    if let Some(v) = after.strip_prefix('=') {
        format!("{head}{name}: i32 = {}", v.trim_start())
    } else {
        line.to_string()
    }
}

/// Wraps single-statement bodies of `if`, `else`, `for` and `while` in braces.
fn brace_bodies(s: &str) -> Result<String, String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let word_at = |i: usize, w: &str| -> bool {
        let wc: Vec<char> = w.chars().collect();
        c.len() >= i + wc.len()
            && c[i..i + wc.len()] == wc[..]
            && (i == 0 || !is_ident(c[i - 1]))
            && !c.get(i + wc.len()).copied().is_some_and(is_ident)
    };
    // Pending closing braces at statement ends.
    let mut pending = 0usize;
    while i < c.len() {
        if word_at(i, "if") || word_at(i, "for") || word_at(i, "while") {
            let kw = if word_at(i, "if") { 2 } else if word_at(i, "for") { 3 } else { 5 };
            let mut j = i + kw;
            while j < c.len() && c[j].is_whitespace() {
                j += 1;
            }
            if c.get(j) != Some(&'(') {
                return Err("an `if` / `for` without parentheses".into());
            }
            let close = close_paren(&c, j).ok_or("unbalanced parentheses")?;
            out.extend(&c[i..=close]);
            i = close + 1;
            let mut k = i;
            while k < c.len() && c[k].is_whitespace() {
                k += 1;
            }
            if c.get(k) != Some(&'{') {
                out.push_str(" {");
                pending += 1;
            }
            continue;
        }
        if word_at(i, "else") {
            out.push_str("else");
            i += 4;
            let mut k = i;
            while k < c.len() && c[k].is_whitespace() {
                k += 1;
            }
            if c.get(k) != Some(&'{') && !word_at(k, "if") {
                out.push_str(" {");
                pending += 1;
            }
            continue;
        }
        let ch = c[i];
        out.push(ch);
        if ch == ';' && pending > 0 {
            // Inside `for (...)` headers the semicolons are not statement ends:
            // those were copied whole above.
            for _ in 0..pending {
                out.push_str(" }");
            }
            pending = 0;
        }
        i += 1;
    }
    Ok(out)
}

/// `x = c ? a : b;` -> `x = select(b, a, c);` (one top-level `?:` per statement).
fn ternaries(s: &str) -> String {
    let mut out = String::new();
    for stmt in s.split_inclusive(';') {
        let c: Vec<char> = stmt.chars().collect();
        let mut depth = 0i32;
        let mut q = None;
        let mut colon = None;
        let mut eq = None;
        for (k, &ch) in c.iter().enumerate() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                '=' if depth == 0 && eq.is_none() => {
                    let prev = if k > 0 { c[k - 1] } else { ' ' };
                    let next = c.get(k + 1).copied().unwrap_or(' ');
                    if !matches!(prev, '=' | '!' | '<' | '>') && next != '=' {
                        eq = Some(k);
                    }
                }
                '?' if depth == 0 => q = Some(k),
                ':' if depth == 0 && q.is_some() => colon = Some(k),
                _ => {}
            }
        }
        match (eq, q, colon) {
            (Some(e), Some(q), Some(col)) if e < q && q < col => {
                let end = c.iter().rposition(|&x| x == ';').unwrap_or(c.len());
                let lhs: String = c[..=e].iter().collect();
                let cond: String = c[e + 1..q].iter().collect();
                let a: String = c[q + 1..col].iter().collect();
                let b: String = c[col + 1..end].iter().collect();
                let tail: String = c[end..].iter().collect();
                let _ = write!(out, "{lhs} select({}, {}, {}){tail}", b.trim(), a.trim(), cond.trim());
            }
            _ => out.push_str(stmt),
        }
    }
    out
}
