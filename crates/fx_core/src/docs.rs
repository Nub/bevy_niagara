//! Module and renderer reference, generated from the definitions in
//! [`crate::modules::MODULES`] and [`crate::render::SPRITE_FIELDS`], so the
//! docs Claude and the user read never drift from the code.

use std::fmt::Write;

use crate::diag::suggest;
use crate::dynamic::{ArgDef, ArgDefault, ArgKind, DYNAMIC_INPUTS, DynamicDef};
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
    if let Some(t) = i.di {
        format!("{} data interface", t.name())
    } else if i.choices.is_empty() {
        i.ty.name().into()
    } else {
        format!("one of {}", i.choices.join(" | "))
    }
}

/// The default in source form: a value, or a static switch's bare name.
pub fn input_default(i: &InputDef) -> String {
    if i.di.is_some() {
        return "required".into();
    }
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
    s.push_str(&dynamic_list());
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
    if let Some(d) = find_dynamic(name) {
        return Ok(dynamic_doc(d));
    }
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
    s.push_str("```\nAny input also takes `Link(Type, \"Namespace.Name\")`, `RandomRange(a, b)`, `CurveOverAge(Type([(t, v), ...]))` or a dynamic input `Dynamic(\"Name\", { inputs })` giving its type (see the dynamic input list).\n");
    Ok(s)
}

const DYNAMIC_INTRO: &str = "A dynamic input computes a module input from inputs of its own, as Niagara's dynamic inputs do: `\"Velocity\": Dynamic(\"Multiply Vec3 by Float\", { \"A\": Dynamic(\"Random Unit Vec3\"), \"B\": Float(3.0) })`. Its inputs take any input form, including further dynamic inputs, nested as deep as needed; inputs left out take their defaults. The name says the type it gives; it can go wherever an input of that type goes, in any stage (inputs reading `Particles.` only in particle stages). Random draws are per particle once in spawn and every step in update.\n\n";

/// Finds a dynamic input by name, ignoring case and surrounding spaces.
pub fn find_dynamic(name: &str) -> Option<&'static DynamicDef> {
    let n = name.trim();
    DYNAMIC_INPUTS.iter().find(|d| d.name.eq_ignore_ascii_case(n))
}

/// One line per dynamic input: name, type, summary.
pub fn dynamic_list() -> String {
    let mut s = String::from("Dynamic inputs (`Dynamic(\"Name\", { inputs })`, any module input of the type they give):\n");
    for d in DYNAMIC_INPUTS.iter() {
        let _ = writeln!(s, "- Dynamic \"{}\" → {}: {}", d.name, d.ty.name(), d.doc);
    }
    s
}

fn arg_default(a: &ArgDef) -> String {
    match (a.kind, a.default) {
        (_, ArgDefault::Required) => "required".into(),
        (_, ArgDefault::Link(p)) => format!("`Link(Float, \"{p}\")`")
            .replace("Float", match a.kind {
                ArgKind::Value(t) => t.name(),
                _ => "Float",
            }),
        (ArgKind::Choice(c), ArgDefault::Value(Value::Int(i))) => {
            format!("`{}`", c.get(i as usize).copied().unwrap_or("?"))
        }
        (_, ArgDefault::Value(v)) => format!("`{}`", ron_value(&v)),
    }
}

/// Markdown reference for one dynamic input.
pub fn dynamic_doc(d: &DynamicDef) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "## Dynamic input \"{}\" → {}\n", d.name, d.ty.name());
    let _ = writeln!(s, "{}\n", d.doc);
    let _ = writeln!(s, "Niagara equivalent: {}.\n", d.niagara);
    if d.args.is_empty() {
        s.push_str("Inputs: none.\n");
    } else {
        s.push_str("| Input | Type | Default | Meaning |\n|---|---|---|---|\n");
        for a in &d.args {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                a.name,
                crate::dynamic::arg_type(a).replace('|', "\\|"),
                arg_default(a),
                a.doc
            );
        }
    }
    let example = d
        .args
        .iter()
        .filter(|a| a.default == ArgDefault::Required)
        .map(|a| match a.kind {
            ArgKind::Curve(t) => format!(
                "\"{}\": {}([(0.0, {}), (1.0, {})])",
                a.name,
                t.name(),
                comps(t, 0.0),
                comps(t, 1.0)
            ),
            ArgKind::List(t) => format!("\"{}\": [{}, {}]", a.name, t.example(), t.example()),
            ArgKind::DataInterface(t) => format!("\"{}\": Link({}, \"User.Name\")", a.name, t.name()),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    if example.is_empty() && d.args.is_empty() {
        let _ = writeln!(s, "\nExample: `Dynamic(\"{}\")`", d.name);
    } else {
        let _ = writeln!(s, "\nExample: `Dynamic(\"{}\", {{ {example} }})`", d.name);
    }
    s
}

fn comps(t: crate::types::ValueType, x: f32) -> String {
    let n = t.components();
    let v = vec![format!("{x:.1}"); n];
    if n == 1 { v[0].clone() } else { format!("({})", v.join(", ")) }
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
    s.push_str("# Dynamic inputs\n\n");
    s.push_str(DYNAMIC_INTRO);
    for d in DYNAMIC_INPUTS.iter() {
        s.push_str(&dynamic_doc(d));
        s.push('\n');
    }
    for (kind, fields) in RENDERER_FIELDS {
        s.push_str(&renderer_doc(kind, fields));
        s.push('\n');
    }
    s.push_str(&event_handler_doc());
    s.push_str(&data_interface_doc());
    s
}

/// The data interface types and how they are declared and read.
pub fn data_interface_doc() -> String {
    let mut s = String::from(
        "# Data interfaces\n\nDeclared in `user_parameters` beside the value parameters, read by module inputs of their type written `Link(Type, \"User.Name\")`. Meshes and textures name a default asset (what previews show); the host can bind its own (fx_render: `ParticleEffect::bind_entity` / `bind_texture`). Curves are baked in and read through the \"... from Curve\" dynamic inputs (`\"Curve\": Link(ColorCurve, \"User.Tint\")`).\n\n| Type | Niagara | Declaration | Read by |\n|---|---|---|---|\n",
    );
    for t in crate::data::DiType::ALL {
        let readers: Vec<String> = MODULES
            .iter()
            .filter(|m| m.inputs.iter().any(|i| i.di == Some(t)))
            .map(|m| format!("\"{}\"", m.name))
            .chain(
                DYNAMIC_INPUTS
                    .iter()
                    .filter(|d| {
                        d.args.iter().any(|a| {
                            a.kind == ArgKind::DataInterface(t)
                                || (t.curve_type().is_some() && a.kind == ArgKind::Curve(t.curve_type().unwrap()))
                        })
                    })
                    .map(|d| format!("\"{}\"", d.name)),
            )
            .collect();
        let _ = writeln!(
            s,
            "| {} | {} | `{}` | {} |",
            t.name(),
            t.niagara_name(),
            t.example(),
            readers.join(", ")
        );
    }
    s
}

/// The `event_handlers` fields and the `Event.` parameters handler modules read.
pub fn event_handler_doc() -> String {
    let mut s = String::from(
        "# Event handlers\n\nEmitter field `event_handlers: [(source: \"Emitter\", event: Death, ...)]` (Niagara's event handlers). Events are written by Generate Death/Collision/Location Event in the source's `particle_update`.\n\n| Field | Values | Default | Meaning |\n|---|---|---|---|\n",
    );
    for (name, form, default, doc) in crate::events::HANDLER_FIELDS {
        let _ = writeln!(s, "| {name} | {form} | {default} | {doc} |");
    }
    s.push_str("\nHandler modules read:\n\n");
    for b in crate::params::EVENT {
        let _ = writeln!(s, "- `Event.{}` ({}): {}", b.name, b.ty.name(), b.doc);
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
        for d in DYNAMIC_INPUTS.iter() {
            let doc = module_doc(&d.name).unwrap();
            assert!(doc.contains(&d.doc) && doc.contains(&d.niagara), "{}", d.name);
            for a in &d.args {
                assert!(doc.contains(a.name), "{}.{}", d.name, a.name);
            }
            assert!(module_list().contains(&format!("Dynamic \"{}\"", d.name)));
            assert!(reference().contains(&format!("## Dynamic input \"{}\"", d.name)));
        }
        assert!(module_doc("float from curve").unwrap().contains("| Curve Index | Float | `Link(Float, \"Particles.NormalizedAge\")`"));
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
