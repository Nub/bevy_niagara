//! Module and renderer reference, generated from the definitions in
//! [`crate::modules::MODULES`] and [`crate::render::SPRITE_FIELDS`], so the
//! docs Claude and the user read never drift from the code.

use std::fmt::Write;

use crate::diag::suggest;
use crate::modules::{InputDef, MODULES, ModuleDef};
use crate::render::{RENDERER_FIELDS, RendererField};
use crate::types::Value;

/// A value in `.fx.ron` source form, e.g. `Vec3((0.0, -9.81, 0.0))`.
pub fn ron_value(v: &Value) -> String {
    fn f(x: f32) -> String {
        if x.fract() == 0.0 && x.abs() < 1e7 {
            format!("{x:.1}")
        } else {
            format!("{x}")
        }
    }
    fn tuple(xs: &[f32]) -> String {
        xs.iter().map(|&x| f(x)).collect::<Vec<_>>().join(", ")
    }
    match v {
        Value::Float(x) => format!("Float({})", f(*x)),
        Value::Int(i) => format!("Int({i})"),
        Value::Bool(b) => format!("Bool({b})"),
        Value::Vec2(a) => format!("Vec2(({}))", tuple(a)),
        Value::Vec3(a) => format!("Vec3(({}))", tuple(a)),
        Value::Vec4(a) => format!("Vec4(({}))", tuple(a)),
        Value::Color(a) => format!("Color(({}))", tuple(a)),
        Value::Quat(a) => format!("Quat(({}))", tuple(a)),
    }
}

/// "Float", or "one of Sphere | Box" for a static switch.
pub fn input_type(i: &InputDef) -> String {
    if i.choices.is_empty() {
        i.ty.name().into()
    } else {
        format!("one of {}", i.choices.join(" | "))
    }
}

/// The default in source form: a value, or a static switch's bare name.
pub fn input_default(i: &InputDef) -> String {
    match (i.choices.is_empty(), i.default) {
        (false, Value::Int(n)) => i.choices.get(n as usize).copied().unwrap_or("?").into(),
        _ => ron_value(&i.default),
    }
}

fn stages(m: &ModuleDef) -> String {
    m.stages
        .iter()
        .map(|s| s.field())
        .collect::<Vec<_>>()
        .join(", ")
}

/// One line per module: name, stages, summary.
pub fn module_list() -> String {
    let mut s = String::new();
    for m in MODULES {
        let _ = writeln!(s, "- \"{}\" [{}]: {}", m.name, stages(m), m.doc);
    }
    for (kind, _) in RENDERER_FIELDS {
        let _ = writeln!(s, "- Renderer \"{kind}\" (emitter field `renderers: [{kind}(( ... ))]`)");
    }
    s
}

/// Finds a module by name, ignoring case and surrounding spaces.
pub fn find_module(name: &str) -> Option<&'static ModuleDef> {
    let n = name.trim();
    MODULES.iter().find(|m| m.name.eq_ignore_ascii_case(n))
}

/// Markdown reference for one module, or for the `Sprite` renderer.
/// The error names the closest match.
pub fn module_doc(name: &str) -> Result<String, String> {
    if let Some((kind, fields)) = RENDERER_FIELDS
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name.trim()))
    {
        return Ok(renderer_doc(kind, fields));
    }
    let Some(m) = find_module(name) else {
        let names = MODULES
            .iter()
            .map(|m| m.name)
            .chain(RENDERER_FIELDS.iter().map(|(k, _)| *k));
        return Err(match suggest(name.trim(), names) {
            Some(s) => format!("unknown module \"{name}\"; did you mean \"{s}\"?"),
            None => format!(
                "unknown module \"{name}\"; known: {} and the renderers {}",
                MODULES
                    .iter()
                    .map(|m| format!("\"{}\"", m.name))
                    .collect::<Vec<_>>()
                    .join(", "),
                RENDERER_FIELDS.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")
            ),
        });
    };
    let mut s = String::new();
    let _ = writeln!(s, "# {}\n", m.name);
    let _ = writeln!(s, "{}\n", m.doc);
    let _ = writeln!(
        s,
        "Niagara equivalent: the core module \"{}\" (same name and inputs; units are Bevy's: metres, seconds, Y up).",
        m.name
    );
    let _ = writeln!(s, "Stages: {}", stages(m));
    let _ = writeln!(s, "Cooked id: {}\n", m.kind as u16);
    if m.inputs.is_empty() {
        s.push_str("Inputs: none.\n");
    } else {
        s.push_str("| Input | Type | Default | Meaning |\n|---|---|---|---|\n");
        for i in m.inputs {
            let _ = writeln!(
                s,
                "| {} | {} | `{}` | {} |",
                i.name,
                input_type(i).replace('|', "\\|"),
                input_default(i),
                i.doc
            );
        }
    }
    let inputs = m
        .inputs
        .iter()
        .map(|i| format!("\"{}\": {}", i.name, input_default(i)))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(s, "\nExample (in `{}`):\n```ron", m.stages[0].field());
    if inputs.is_empty() {
        let _ = writeln!(s, "(module: \"{}\"),", m.name);
    } else {
        let _ = writeln!(s, "(module: \"{}\", inputs: {{ {inputs} }}),", m.name);
    }
    s.push_str("```\nAny input also takes `Link(Type, \"Namespace.Name\")`, `RandomRange(a, b)` or `CurveOverAge(Type([(t, v), ...]))`.\n");
    Ok(s)
}

fn renderer_doc(name: &str, fields: &[RendererField]) -> String {
    let mut s = format!(
        "# {name} renderer\n\nNiagara equivalent: the {name} Renderer. Emitter field `renderers: [{name}(( ... ))]`; every field is optional.\n\n| Field | Values | Default | Meaning |\n|---|---|---|---|\n"
    );
    for f in fields {
        let _ = writeln!(s, "| {} | {} | {} | {} |", f.name, f.form, f.default, f.doc);
    }
    s
}

/// The whole reference: every module and renderer.
pub fn reference() -> String {
    let mut s = String::from("# Module reference\n\n");
    for m in MODULES {
        s.push_str(&module_doc(m.name).expect("known module"));
        s.push('\n');
    }
    for (kind, fields) in RENDERER_FIELDS {
        s.push_str(&renderer_doc(kind, fields));
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_module_has_a_doc() {
        for m in MODULES {
            let d = module_doc(m.name).unwrap();
            assert!(d.contains(m.doc), "{}", m.name);
            for i in m.inputs {
                assert!(d.contains(i.name), "{}.{}", m.name, i.name);
            }
        }
        assert!(module_doc("gravity force").is_ok());
        assert!(module_doc("Sprite").unwrap().contains("blend_mode"));
        for (kind, _) in RENDERER_FIELDS {
            assert!(module_doc(kind).is_ok(), "{kind}");
        }
        let e = module_doc("Gravity Forse").unwrap_err();
        assert!(e.contains("did you mean \"Gravity Force\""), "{e}");
    }

    #[test]
    fn values_in_source_form() {
        assert_eq!(
            ron_value(&Value::Vec3([0.0, -9.81, 0.0])),
            "Vec3((0.0, -9.81, 0.0))"
        );
        assert_eq!(ron_value(&Value::Int(10)), "Int(10)");
    }
}
