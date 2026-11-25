//! Turns the parsed RON tree into the effect model, reporting shape errors
//! (wrong value kinds, unknown or missing fields) at their line.

use crate::diag::{Diags, suggest};
use crate::model::*;
use crate::render::{
    IdentEnum, MESH_BUILTINS, RENDERER_KINDS, is_mesh_asset_path, light_field_names,
    mesh_field_names, ribbon_field_names, sprite_field_names,
};
use crate::ron_parse::{Field, SKind, SValue};
use crate::types::{Curve, Value, ValueType};

pub const SYSTEM_FIELDS: &[&str] = &[
    "name",
    "user_parameters",
    "system_spawn",
    "system_update",
    "emitters",
];
pub const EMITTER_FIELDS: &[&str] = &[
    "name",
    "sim_target",
    "max_particles",
    "emitter_spawn",
    "emitter_update",
    "particle_spawn",
    "particle_update",
    "event_handlers",
    "renderers",
];
pub const MODULE_FIELDS: &[&str] = &["module", "enabled", "inputs"];

const DEFAULT_MAX_PARTICLES: u32 = 1000;

pub fn decode_system(v: &SValue, d: &mut Diags) -> Option<SystemSource> {
    let mut f = Fields::new(v, "the effect", SYSTEM_FIELDS, d)?;
    let name = f.required_string("name", d).unwrap_or_default();
    let user_parameters = f
        .take("user_parameters")
        .map(|v| decode_params(v, d))
        .unwrap_or_default();
    let system_spawn = f
        .take("system_spawn")
        .map(|v| decode_stack(v, d))
        .unwrap_or_default();
    let system_update = f
        .take("system_update")
        .map(|v| decode_stack(v, d))
        .unwrap_or_default();
    let emitters = match f.take("emitters") {
        Some(v) => list(v, "`emitters`", d)
            .iter()
            .filter_map(|e| decode_emitter(e, d))
            .collect(),
        None => Vec::new(),
    };
    f.finish(d);
    Some(SystemSource {
        name,
        user_parameters,
        system_spawn,
        system_update,
        emitters,
        span: v.span,
    })
}

fn decode_emitter(v: &SValue, d: &mut Diags) -> Option<EmitterSource> {
    let mut f = Fields::new(v, "an emitter", EMITTER_FIELDS, d)?;
    let name = f.required_string("name", d).unwrap_or_default();
    let sim_target = match f.take("sim_target") {
        None => SimTarget::default(),
        Some(SValue {
            kind: SKind::Ident(i),
            ..
        }) if i == "Gpu" => SimTarget::Gpu,
        Some(SValue {
            kind: SKind::Ident(i),
            ..
        }) if i == "Cpu" => SimTarget::Cpu,
        Some(other) => {
            d.error(
                other.span,
                format!(
                    "`sim_target` must be `Gpu` or `Cpu`, found {}",
                    other.kind.describe()
                ),
                Some("write `sim_target: Gpu` (the default) or `sim_target: Cpu`".into()),
            );
            SimTarget::default()
        }
    };
    let max_particles = match f.take("max_particles") {
        None => DEFAULT_MAX_PARTICLES,
        Some(SValue {
            kind: SKind::Number(n, true),
            ..
        }) if *n >= 1.0 && *n <= u32::MAX as f64 => *n as u32,
        Some(other) => {
            d.error(
                other.span,
                format!(
                    "`max_particles` must be a whole number of at least 1, found {}",
                    other.kind.describe()
                ),
                Some(format!(
                    "write e.g. `max_particles: {DEFAULT_MAX_PARTICLES}`"
                )),
            );
            DEFAULT_MAX_PARTICLES
        }
    };
    let mut stack =
        |name: &str, d: &mut Diags| f.take(name).map(|v| decode_stack(v, d)).unwrap_or_default();
    let emitter_spawn = stack("emitter_spawn", d);
    let emitter_update = stack("emitter_update", d);
    let particle_spawn = stack("particle_spawn", d);
    let particle_update = stack("particle_update", d);
    let event_handlers = match f.take("event_handlers") {
        Some(v) => list(v, "`event_handlers`", d)
            .iter()
            .filter_map(|h| decode_handler(h, d))
            .collect(),
        None => Vec::new(),
    };
    let renderers = match f.take("renderers") {
        Some(v) => list(v, "`renderers`", d)
            .iter()
            .filter_map(|r| decode_renderer(r, d))
            .collect(),
        None => Vec::new(),
    };
    f.finish(d);
    Some(EmitterSource {
        name,
        sim_target,
        max_particles,
        emitter_spawn,
        emitter_update,
        particle_spawn,
        particle_update,
        event_handlers,
        renderers,
        span: v.span,
    })
}

const RENDERER_HELP: &str = "write a renderer as `Kind(( field: value, ... ))` with Kind one of Sprite, Ribbon, Mesh or Light, e.g. `Sprite((blend_mode: Additive))`; every field is optional";

fn decode_renderer(v: &SValue, d: &mut Diags) -> Option<RendererSource> {
    // `Sprite((...))`, `Sprite(...fields...)`, `Sprite(())`, `Sprite()` or bare `Sprite`.
    let (tag, body): (&str, Option<&SValue>) = match &v.kind {
        SKind::Tuple(Some(tag), items) => match items.as_slice() {
            [] => (tag, None),
            [inner] => (tag, Some(inner)),
            _ => {
                d.error(
                    v.span,
                    format!("`{tag}(...)` takes one struct of settings"),
                    Some(RENDERER_HELP.into()),
                );
                return None;
            }
        },
        SKind::Struct(Some(tag), _) => (tag, Some(v)),
        SKind::Ident(tag) => (tag, None),
        other => {
            d.error(
                v.span,
                format!("expected a renderer, found {}", other.describe()),
                Some(format!(
                    "{RENDERER_HELP}. Renderers are: {}",
                    RENDERER_KINDS.join(", ")
                )),
            );
            return None;
        }
    };
    match tag {
        "Sprite" => decode_sprite(body, v, d).map(RendererSource::Sprite),
        "Ribbon" => decode_ribbon(body, v, d).map(RendererSource::Ribbon),
        "Mesh" => decode_mesh(body, v, d).map(RendererSource::Mesh),
        "Light" => decode_light(body, v, d).map(RendererSource::Light),
        _ => {
            let kinds = RENDERER_KINDS.join(", ");
            let help = match suggest(tag, RENDERER_KINDS.iter().copied()) {
                Some(s) => format!("did you mean `{s}`? Renderers are: {kinds}"),
                None => format!("renderers are: {kinds}"),
            };
            d.error(v.span, format!("unknown renderer `{tag}`"), Some(help));
            None
        }
    }
}

const MATERIAL_HELP: &str = "write e.g. `material: \"shaders/glow.wesl\"` (a WESL fragment shader), or leave it out for the built-in shader";

fn decode_sprite(body: Option<&SValue>, whole: &SValue, d: &mut Diags) -> Option<SpriteSource> {
    let mut s = SpriteSource {
        span: whole.span,
        ..SpriteSource::default()
    };
    let Some(body) = body else {
        return Some(s);
    };
    let mut f = Fields::new(body, "a Sprite renderer", sprite_field_names(), d)?;
    if let Some(v) = f.take("blend_mode") {
        s.blend_mode = ident_enum(v, "blend_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("lit") {
        s.lit = boolean(v, "lit", d).unwrap_or_default();
    }
    if let Some(v) = f.take("facing_mode") {
        s.facing_mode = ident_enum(v, "facing_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("custom_facing_vector") {
        s.custom_facing_vector = floats(v, "custom_facing_vector", "(0.0, 0.0, 1.0)", d)
            .unwrap_or(s.custom_facing_vector);
    }
    if let Some(v) = f.take("alignment") {
        s.alignment = ident_enum(v, "alignment", d).unwrap_or_default();
    }
    if let Some(v) = f.take("custom_alignment_vector") {
        s.custom_alignment_vector = floats(v, "custom_alignment_vector", "(0.0, 1.0, 0.0)", d)
            .unwrap_or(s.custom_alignment_vector);
    }
    if let Some(v) = f.take("pivot_in_uv_space") {
        s.pivot_in_uv_space =
            floats(v, "pivot_in_uv_space", "(0.5, 0.5)", d).unwrap_or(s.pivot_in_uv_space);
    }
    if let Some(v) = f.take("sub_image_size") {
        s.sub_image_size = sub_image_size(v, d).unwrap_or(s.sub_image_size);
    }
    if let Some(v) = f.take("sub_uv_blending") {
        s.sub_uv_blending = boolean(v, "sub_uv_blending", d).unwrap_or_default();
    }
    if let Some(v) = f.take("sort_mode") {
        s.sort_mode = ident_enum(v, "sort_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("soft_particle_fade_distance") {
        match v.kind {
            SKind::Number(n, _) if n >= 0.0 && n.is_finite() => {
                s.soft_particle_fade_distance = n as f32
            }
            _ => d.error(
                v.span,
                format!(
                    "`soft_particle_fade_distance` must be a distance of at least 0 metres, found {}",
                    v.kind.describe()
                ),
                Some(
                    "write e.g. `soft_particle_fade_distance: 0.5` (metres), or `0.0` to turn soft fading off"
                        .into(),
                ),
            ),
        }
    }
    if let Some(v) = f.take("ray_tracing") {
        s.ray_tracing = boolean(v, "ray_tracing", d).unwrap_or_default();
    }
    if let Some(v) = f.take("texture") {
        s.texture = asset_path(
            v,
            "texture",
            "write e.g. `texture: \"textures/smoke.png\"`, or leave it out for the built-in soft round sprite",
            d,
        );
    }
    if let Some(v) = f.take("material") {
        s.material = asset_path(v, "material", MATERIAL_HELP, d);
    }
    f.finish(d);
    Some(s)
}

fn decode_ribbon(body: Option<&SValue>, whole: &SValue, d: &mut Diags) -> Option<RibbonSource> {
    let mut r = RibbonSource {
        span: whole.span,
        ..RibbonSource::default()
    };
    let Some(body) = body else {
        return Some(r);
    };
    let mut f = Fields::new(body, "a Ribbon renderer", ribbon_field_names(), d)?;
    if let Some(v) = f.take("blend_mode") {
        r.blend_mode = ident_enum(v, "blend_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("facing_mode") {
        r.facing_mode = ident_enum(v, "facing_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("custom_facing_vector") {
        r.custom_facing_vector = floats(v, "custom_facing_vector", "(0.0, 0.0, 1.0)", d)
            .unwrap_or(r.custom_facing_vector);
    }
    if let Some(v) = f.take("uv_mode") {
        r.uv_mode = ident_enum(v, "uv_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("tiling_length") {
        match v.kind {
            SKind::Number(n, _) if n > 0.0 && n.is_finite() => r.tiling_length = n as f32,
            _ => d.error(
                v.span,
                format!(
                    "`tiling_length` must be a length of more than 0 metres, found {}",
                    v.kind.describe()
                ),
                Some(
                    "write e.g. `tiling_length: 1.0` (metres of ribbon per texture repeat)".into(),
                ),
            ),
        }
    }
    if let Some(v) = f.take("link_order") {
        r.link_order = ident_enum(v, "link_order", d).unwrap_or_default();
    }
    if let Some(v) = f.take("tessellation_factor") {
        r.tessellation_factor = whole_in_range(
            v,
            "tessellation_factor",
            1..=16,
            "write e.g. `tessellation_factor: 4` (subdivisions per segment; 1 = none)",
            d,
        )
        .unwrap_or(r.tessellation_factor);
    }
    if let Some(v) = f.take("texture") {
        r.texture = asset_path(
            v,
            "texture",
            "write e.g. `texture: \"textures/trail.png\"`, or leave it out for plain colour",
            d,
        );
    }
    if let Some(v) = f.take("material") {
        r.material = asset_path(v, "material", MATERIAL_HELP, d);
    }
    f.finish(d);
    Some(r)
}

fn decode_mesh(body: Option<&SValue>, whole: &SValue, d: &mut Diags) -> Option<MeshSource> {
    let mut m = MeshSource {
        span: whole.span,
        ..MeshSource::default()
    };
    let Some(body) = body else {
        return Some(m);
    };
    let mut f = Fields::new(body, "a Mesh renderer", mesh_field_names(), d)?;
    if let Some(v) = f.take("mesh") {
        let name = match &v.kind {
            SKind::Str(s) => Some(s.as_str()),
            SKind::Ident(i) => Some(i.as_str()),
            _ => None,
        };
        match name {
            Some(n) if matches!(v.kind, SKind::Str(_)) => {
                if let Some((msg, help)) = mesh_name_error(n) {
                    d.error(v.span, msg, Some(help));
                } else {
                    m.mesh = n.to_string();
                }
            }
            _ => d.error(
                v.span,
                format!(
                    "`mesh` must be a built-in mesh name or an asset path \"string\", found {}",
                    v.kind.describe()
                ),
                Some(match name {
                    Some(n) => format!("write it in quotes: `mesh: \"{n}\"`"),
                    None => format!(
                        "write e.g. `mesh: \"sphere\"` (built-in meshes: {}) or `mesh: \"meshes/rock.glb\"`",
                        MESH_BUILTINS.join(", ")
                    ),
                }),
            ),
        }
    }
    if let Some(v) = f.take("blend_mode") {
        m.blend_mode = ident_enum(v, "blend_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("facing_mode") {
        m.facing_mode = ident_enum(v, "facing_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("lit") {
        m.lit = boolean(v, "lit", d).unwrap_or(m.lit);
    }
    if let Some(v) = f.take("sort_mode") {
        m.sort_mode = ident_enum(v, "sort_mode", d).unwrap_or_default();
    }
    if let Some(v) = f.take("texture") {
        m.texture = asset_path(
            v,
            "texture",
            "write e.g. `texture: \"textures/rock.png\"`, or leave it out for plain colour",
            d,
        );
    }
    if let Some(v) = f.take("material") {
        m.material = asset_path(v, "material", MATERIAL_HELP, d);
    }
    f.finish(d);
    Some(m)
}

/// Why `mesh` is neither a built-in mesh nor an asset path: `(message, help)`.
pub fn mesh_name_error(mesh: &str) -> Option<(String, String)> {
    if MESH_BUILTINS.contains(&mesh) || is_mesh_asset_path(mesh) {
        return None;
    }
    let builtins = MESH_BUILTINS.join(", ");
    let help = match suggest(mesh, MESH_BUILTINS.iter().copied()) {
        Some(s) => format!(
            "did you mean `mesh: \"{s}\"`? Built-in meshes: {builtins}; an asset path contains a '.' or '/', e.g. \"meshes/rock.glb\""
        ),
        None => format!(
            "built-in meshes: {builtins}; or give an asset path (it contains a '.' or '/'), e.g. `mesh: \"meshes/rock.glb\"`"
        ),
    };
    Some((format!("unknown built-in mesh \"{mesh}\""), help))
}

fn decode_light(body: Option<&SValue>, whole: &SValue, d: &mut Diags) -> Option<LightSource> {
    let mut l = LightSource {
        span: whole.span,
        ..LightSource::default()
    };
    let Some(body) = body else {
        return Some(l);
    };
    let mut f = Fields::new(body, "a Light renderer", light_field_names(), d)?;
    if let Some(v) = f.take("radius_scale") {
        match v.kind {
            SKind::Number(n, _) if n >= 0.0 && n.is_finite() => l.radius_scale = n as f32,
            _ => d.error(
                v.span,
                format!(
                    "`radius_scale` must be a number of at least 0, found {}",
                    v.kind.describe()
                ),
                Some("write e.g. `radius_scale: 2.0` (multiplies Particles.LightRadius)".into()),
            ),
        }
    }
    if let Some(v) = f.take("color_add") {
        l.color_add = floats(v, "color_add", "(0.0, 0.0, 0.0)", d).unwrap_or(l.color_add);
    }
    if let Some(v) = f.take("intensity") {
        match v.kind {
            SKind::Number(n, _) if n >= 0.0 && n.is_finite() => l.intensity = n as f32,
            _ => d.error(
                v.span,
                format!(
                    "`intensity` must be at least 0 lumens per unit colour, found {}",
                    v.kind.describe()
                ),
                Some("write e.g. `intensity: 10000.0` (lumens for a white particle)".into()),
            ),
        }
    }
    if let Some(v) = f.take("alpha_scales_brightness") {
        l.alpha_scales_brightness =
            boolean(v, "alpha_scales_brightness", d).unwrap_or(l.alpha_scales_brightness);
    }
    if let Some(v) = f.take("max_lights") {
        l.max_lights = whole_in_range(
            v,
            "max_lights",
            1..=256,
            "write e.g. `max_lights: 16` (the lights closest to the camera are kept)",
            d,
        )
        .unwrap_or(l.max_lights);
    }
    f.finish(d);
    Some(l)
}

fn asset_path(v: &SValue, field: &str, help: &str, d: &mut Diags) -> Option<String> {
    match &v.kind {
        SKind::Str(p) => Some(p.clone()),
        other => {
            d.error(
                v.span,
                format!(
                    "`{field}` must be an asset path \"string\", found {}",
                    other.describe()
                ),
                Some(help.into()),
            );
            None
        }
    }
}

fn whole_in_range(
    v: &SValue,
    field: &str,
    range: std::ops::RangeInclusive<u32>,
    help: &str,
    d: &mut Diags,
) -> Option<u32> {
    match v.kind {
        SKind::Number(n, true) if n >= *range.start() as f64 && n <= *range.end() as f64 => {
            Some(n as u32)
        }
        _ => {
            d.error(
                v.span,
                format!(
                    "`{field}` must be a whole number from {} to {}, found {}",
                    range.start(),
                    range.end(),
                    v.kind.describe()
                ),
                Some(help.into()),
            );
            None
        }
    }
}

fn ident_enum<T: IdentEnum>(v: &SValue, field: &str, d: &mut Diags) -> Option<T> {
    let name = match &v.kind {
        SKind::Ident(i) => i.as_str(),
        SKind::Str(s) => s.as_str(),
        _ => "",
    };
    if let SKind::Ident(_) = v.kind
        && let Some(x) = T::from_name(name)
    {
        return Some(x);
    }
    let names = T::VARIANTS.iter().map(|(n, _)| *n);
    let help = match suggest(name, names).filter(|_| !name.is_empty()) {
        Some(s) => format!("did you mean `{field}: {s}`? (a bare name, no quotes)"),
        None => format!("write `{field}: {}` (the default)", T::VARIANTS[0].0),
    };
    d.error(
        v.span,
        format!(
            "`{field}` must be {}, found {}",
            T::list(),
            v.kind.describe()
        ),
        Some(help),
    );
    None
}

fn boolean(v: &SValue, field: &str, d: &mut Diags) -> Option<bool> {
    match v.kind {
        SKind::Bool(b) => Some(b),
        _ => {
            d.error(
                v.span,
                format!(
                    "`{field}` must be `true` or `false`, found {}",
                    v.kind.describe()
                ),
                Some(format!("write `{field}: true` or `{field}: false`")),
            );
            None
        }
    }
}

/// A plain tuple of `N` numbers, e.g. `(0.0, 1.0, 0.0)`.
fn floats<const N: usize>(
    v: &SValue,
    field: &str,
    example: &str,
    d: &mut Diags,
) -> Option<[f32; N]> {
    let bad = |d: &mut Diags| {
        d.error(
            v.span,
            format!(
                "`{field}` must be a tuple of {N} numbers, found {}",
                v.kind.describe()
            ),
            Some(format!("write e.g. `{field}: {example}`")),
        );
    };
    let (SKind::Tuple(None, items) | SKind::List(items)) = &v.kind else {
        bad(d);
        return None;
    };
    if items.len() != N {
        bad(d);
        return None;
    }
    let mut out = [0.0f32; N];
    for (o, x) in out.iter_mut().zip(items) {
        *o = number(x, d)?;
    }
    Some(out)
}

fn sub_image_size(v: &SValue, d: &mut Diags) -> Option<[u32; 2]> {
    let help = "write `sub_image_size: (columns, rows)` with whole numbers of at least 1, e.g. `(2, 2)`; `(1, 1)` means no flipbook";
    let (SKind::Tuple(None, items) | SKind::List(items)) = &v.kind else {
        d.error(
            v.span,
            format!(
                "`sub_image_size` must be `(columns, rows)`, found {}",
                v.kind.describe()
            ),
            Some(help.into()),
        );
        return None;
    };
    if items.len() != 2 {
        d.error(
            v.span,
            format!(
                "`sub_image_size` must be `(columns, rows)`, found {} numbers",
                items.len()
            ),
            Some(help.into()),
        );
        return None;
    }
    let mut out = [1u32; 2];
    for (o, x) in out.iter_mut().zip(items) {
        match x.kind {
            SKind::Number(n, true) if n >= 1.0 && n <= u32::MAX as f64 => *o = n as u32,
            _ => {
                d.error(
                    x.span,
                    format!(
                        "`sub_image_size` columns and rows must be whole numbers of at least 1, found {}",
                        x.kind.describe()
                    ),
                    Some(help.into()),
                );
                return None;
            }
        }
    }
    Some(out)
}

fn decode_stack(v: &SValue, d: &mut Diags) -> Vec<ModuleCall> {
    list(v, "a stage", d)
        .iter()
        .filter_map(|m| decode_module(m, d))
        .collect()
}

fn decode_module(v: &SValue, d: &mut Diags) -> Option<ModuleCall> {
    let mut f = Fields::new(v, "a module call", MODULE_FIELDS, d)?;
    let module = f.required_string("module", d)?;
    let enabled = match f.take("enabled") {
        None => true,
        Some(SValue {
            kind: SKind::Bool(b),
            ..
        }) => *b,
        Some(other) => {
            d.error(
                other.span,
                format!(
                    "`enabled` must be `true` or `false`, found {}",
                    other.kind.describe()
                ),
                None,
            );
            true
        }
    };
    let inputs = match f.take("inputs") {
        Some(m) => decode_input_map(m, "module", d),
        None => InputMap::default(),
    };
    f.finish(d);
    // Point module diagnostics at the module name, not the opening bracket.
    let span = match &v.kind {
        SKind::Struct(_, fields) => fields
            .iter()
            .find(|f| f.name == "module")
            .map_or(v.span, |f| f.value.span),
        _ => v.span,
    };
    Some(ModuleCall {
        module,
        enabled,
        inputs,
        span,
    })
}

/// `{ "Name": input, ... }` of a module or dynamic input (`owner` names it in errors).
fn decode_input_map(m: &SValue, owner: &str, d: &mut Diags) -> InputMap {
    let mut inputs = InputMap::default();
    for (k, val) in map(m, "`inputs`", d) {
        let Some(name) = key_string(k, d) else {
            continue;
        };
        if inputs.0.iter().any(|e| e.name == name) {
            d.error(
                k.span,
                format!("input \"{name}\" is set twice in this {owner}"),
                Some("keep one of the two entries".into()),
            );
            continue;
        }
        if let Some(input) = decode_input(val, &name, d) {
            inputs.0.push(InputEntry {
                name,
                input,
                span: k.span,
            });
        }
    }
    inputs
}

/// `Float([...])`, `Vec3([(t, (x, y, z)), ...])`: a typed tag around one
/// list of keys (tuples), as opposed to `Vec3([x, y, z])`.
fn is_curve_literal(tag: &str, items: &[SValue]) -> bool {
    let Some(ty) = ValueType::from_name(tag) else {
        return false;
    };
    match items {
        [SValue {
            kind: SKind::List(keys),
            ..
        }] => {
            ty.interpolable()
                && (keys.is_empty() || matches!(keys[0].kind, SKind::Tuple(None, _)))
        }
        _ => false,
    }
}

fn decode_params(v: &SValue, d: &mut Diags) -> ParamMap {
    let mut out = ParamMap::default();
    for (k, val) in map(v, "`user_parameters`", d) {
        let Some(name) = key_string(k, d) else {
            continue;
        };
        if let Some(di) = data_interface_tag(val) {
            if let Some(src) = decode_data_interface(di, val, &name, k.span, d) {
                out.1.push(src);
            }
            continue;
        }
        match decode_input(val, &name, d) {
            Some(Input::Literal(value)) => out.0.push(ParamEntry { name, value, span: k.span }),
            Some(_) => d.error(
                val.span,
                format!("user parameter \"{name}\" must have a literal default value"),
                Some("write a typed literal such as `Float(1.0)`; links, random ranges and curves go in module inputs".into()),
            ),
            None => {}
        }
    }
    out
}

const INPUT_FORMS: &str = "an input is a typed literal (`Float(1.0)`, `Int(5)`, `Bool(true)`, `Vec2((x, y))`, `Vec3((x, y, z))`, `Vec4((x, y, z, w))`, `Color((r, g, b, a))`, `Quat((x, y, z, w))`), `Link(Type, \"Namespace.Name\")`, `RandomRange(min, max)`, `CurveOverAge(Type([(time, value), ...]))` or `Dynamic(\"Name\", { inputs })` (see the dynamic input list)";

fn decode_input(v: &SValue, name: &str, d: &mut Diags) -> Option<Input> {
    match &v.kind {
        SKind::Tuple(Some(tag), items) => {
            match tag.as_str() {
                "Link" => {
                    let [
                        SValue {
                            kind: SKind::Ident(ty),
                            span: ty_span,
                        },
                        SValue {
                            kind: SKind::Str(target),
                            ..
                        },
                    ] = items.as_slice()
                    else {
                        d.error(
                            v.span,
                            format!(
                                "`Link` for \"{name}\" must be `Link(Type, \"Namespace.Name\")`"
                            ),
                            Some("e.g. `Link(Vec3, \"User.Wind\")`".into()),
                        );
                        return None;
                    };
                    if let Some(di) = crate::data::DiType::from_name(ty) {
                        return Some(Input::DataInterface(di, target.clone()));
                    }
                    let Some(t) = ValueType::from_name(ty) else {
                        d.error(
                            *ty_span,
                            format!("unknown type `{ty}`"),
                            Some(type_help(ty)),
                        );
                        return None;
                    };
                    Some(Input::Link(t, target.clone()))
                }
                "RandomRange" => {
                    let [a, b] = items.as_slice() else {
                        d.error(v.span, format!("`RandomRange` for \"{name}\" takes two values: `RandomRange(min, max)`"), Some("e.g. `RandomRange(Float(1.0), Float(2.0))`".into()));
                        return None;
                    };
                    let a = decode_literal(a, d)?;
                    let b = decode_literal(b, d)?;
                    Some(Input::RandomRange(a, b))
                }
                "CurveOverAge" => {
                    let [c] = items.as_slice() else {
                        d.error(
                            v.span,
                            format!("`CurveOverAge` for \"{name}\" takes one curve"),
                            Some("e.g. `CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)]))`".into()),
                        );
                        return None;
                    };
                    decode_curve(c, d).map(Input::CurveOverAge)
                }
                "Dynamic" => {
                    let help = "e.g. `Dynamic(\"Uniform Ranged Float\", { \"Minimum\": Float(1.0), \"Maximum\": Float(2.0) })`";
                    let (dyn_name, map_v) = match items.as_slice() {
                        [SValue { kind: SKind::Str(n), .. }] => (n, None),
                        [SValue { kind: SKind::Str(n), .. }, m] => (n, Some(m)),
                        _ => {
                            d.error(
                                v.span,
                                format!("`Dynamic` for \"{name}\" must be `Dynamic(\"Name\", {{ inputs }})`"),
                                Some(help.into()),
                            );
                            return None;
                        }
                    };
                    let inputs = match map_v {
                        Some(m) => decode_input_map(m, "dynamic input", d),
                        None => InputMap::default(),
                    };
                    Some(Input::Dynamic(DynamicCall {
                        name: dyn_name.clone(),
                        inputs,
                        span: v.span,
                    }))
                }
                t if is_curve_literal(t, items) => decode_curve(v, d).map(Input::Curve),
                _ => decode_literal(v, d).map(Input::Literal),
            }
        }
        SKind::List(items) => {
            let mut out = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let iname = format!("{name}[{i}]");
                out.push(InputEntry {
                    input: decode_input(item, &iname, d)?,
                    name: iname,
                    span: item.span,
                });
            }
            Some(Input::List(out))
        }
        SKind::Ident(choice) => Some(Input::Choice(choice.clone())),
        SKind::Number(..) => {
            d.error(
                v.span,
                format!("input \"{name}\" must state its type, found a bare number"),
                Some(format!(
                    "wrap it in its type, e.g. `Float({})` or `Int(...)`",
                    fmt_num(v)
                )),
            );
            None
        }
        _ => {
            d.error(
                v.span,
                format!(
                    "input \"{name}\" has an invalid value: found {}",
                    v.kind.describe()
                ),
                Some(INPUT_FORMS.into()),
            );
            None
        }
    }
}

fn fmt_num(v: &SValue) -> String {
    match v.kind {
        SKind::Number(n, _) if n.fract() == 0.0 => format!("{n:.1}"),
        SKind::Number(n, _) => format!("{n}"),
        _ => "1.0".into(),
    }
}

fn type_help(ty: &str) -> String {
    let names: Vec<_> = ValueType::ALL.iter().map(|t| t.name()).collect();
    match suggest(ty, names.iter().copied()) {
        Some(s) => format!("did you mean `{s}`? Types are: {}", names.join(", ")),
        None => format!(
            "types are: {} (`Color` is Niagara's Linear Color, `Vec3` its Vector)",
            names.join(", ")
        ),
    }
}

pub fn decode_literal(v: &SValue, d: &mut Diags) -> Option<Value> {
    let SKind::Tuple(Some(tag), items) = &v.kind else {
        d.error(
            v.span,
            format!("expected a typed literal, found {}", v.kind.describe()),
            Some("e.g. `Float(1.0)` or `Vec3((0.0, 1.0, 0.0))`".into()),
        );
        return None;
    };
    let Some(ty) = ValueType::from_name(tag) else {
        d.error(
            v.span,
            format!("unknown type or input form `{tag}`"),
            Some(format!("{}. {INPUT_FORMS}", type_help(tag))),
        );
        return None;
    };
    let example = ty.example();
    match ty {
        ValueType::Bool => match items.as_slice() {
            [
                SValue {
                    kind: SKind::Bool(b),
                    ..
                },
            ] => Some(Value::Bool(*b)),
            _ => {
                d.error(
                    v.span,
                    "`Bool` takes `true` or `false`",
                    Some(format!("write `{example}`")),
                );
                None
            }
        },
        ValueType::Int => match items.as_slice() {
            [
                SValue {
                    kind: SKind::Number(n, true),
                    ..
                },
            ] if n.abs() <= i32::MAX as f64 => Some(Value::Int(*n as i32)),
            [
                SValue {
                    kind: SKind::Number(n, false),
                    span,
                },
            ] => {
                d.error(
                    *span,
                    format!("`Int` needs a whole number, found {n}"),
                    Some(format!("write `Int({})` or use a `Float` input", n.round())),
                );
                None
            }
            _ => {
                d.error(
                    v.span,
                    "`Int` takes one whole number",
                    Some(format!("write `{example}`")),
                );
                None
            }
        },
        _ => {
            let n = ty.components();
            let nums: &[SValue] = match items.as_slice() {
                [
                    SValue {
                        kind: SKind::Tuple(None, inner) | SKind::List(inner),
                        ..
                    },
                ] if n > 1 => inner,
                other => other,
            };
            if nums.len() != n {
                d.error(
                    v.span,
                    format!(
                        "`{tag}` needs {n} number{}, found {}",
                        if n == 1 { "" } else { "s" },
                        nums.len()
                    ),
                    Some(format!("write `{example}`")),
                );
                return None;
            }
            let mut a = [0.0f32; 4];
            for (i, x) in nums.iter().enumerate() {
                a[i] = number(x, d)?;
            }
            Some(Value::from_array(ty, a))
        }
    }
}

fn number(v: &SValue, d: &mut Diags) -> Option<f32> {
    match v.kind {
        SKind::Number(n, _) => Some(n as f32),
        _ => {
            d.error(
                v.span,
                format!("expected a number, found {}", v.kind.describe()),
                None,
            );
            None
        }
    }
}

fn decode_curve(v: &SValue, d: &mut Diags) -> Option<Curve> {
    let help = "write a curve as `Type([(time, value), ...])`, e.g. `Float([(0.0, 1.0), (1.0, 0.0)])` or `Color([(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0, 0.0))])`";
    let SKind::Tuple(Some(tag), items) = &v.kind else {
        d.error(
            v.span,
            format!("expected a curve, found {}", v.kind.describe()),
            Some(help.into()),
        );
        return None;
    };
    let ty = match ValueType::from_name(tag) {
        Some(t) if t.interpolable() => t,
        _ => {
            d.error(
                v.span,
                format!("`{tag}` is not a curve type"),
                Some(format!(
                    "curve types are Float, Vec2, Vec3, Vec4 and Color; {help}"
                )),
            );
            return None;
        }
    };
    let [
        SValue {
            kind: SKind::List(keys),
            ..
        },
    ] = items.as_slice()
    else {
        d.error(v.span, "a curve holds one list of keys", Some(help.into()));
        return None;
    };
    let mut out: Vec<(f32, [f32; 4])> = Vec::new();
    for k in keys {
        let SKind::Tuple(None, kv) = &k.kind else {
            d.error(
                k.span,
                format!(
                    "a curve key is `(time, value)`, found {}",
                    k.kind.describe()
                ),
                Some(help.into()),
            );
            return None;
        };
        let [t, val] = kv.as_slice() else {
            d.error(k.span, "a curve key is `(time, value)`", Some(help.into()));
            return None;
        };
        let t = number(t, d)?;
        let n = ty.components();
        let comps: Vec<&SValue> = match &val.kind {
            SKind::Tuple(None, inner) | SKind::List(inner) => inner.iter().collect(),
            _ => vec![val],
        };
        if comps.len() != n {
            d.error(
                val.span,
                format!(
                    "a {} curve key value needs {n} number{}, found {}",
                    ty.name(),
                    if n == 1 { "" } else { "s" },
                    comps.len()
                ),
                Some(help.into()),
            );
            return None;
        }
        let mut a = [0.0f32; 4];
        for (i, c) in comps.iter().enumerate() {
            a[i] = number(c, d)?;
        }
        out.push((t, a));
    }
    Some(match ty {
        ValueType::Float => Curve::Float(out.iter().map(|(t, a)| (*t, a[0])).collect()),
        ValueType::Vec2 => Curve::Vec2(out.iter().map(|(t, a)| (*t, [a[0], a[1]])).collect()),
        ValueType::Vec3 => Curve::Vec3(out.iter().map(|(t, a)| (*t, [a[0], a[1], a[2]])).collect()),
        ValueType::Vec4 => Curve::Vec4(out),
        _ => Curve::Color(out),
    })
}

fn key_string(k: &SValue, d: &mut Diags) -> Option<String> {
    match &k.kind {
        SKind::Str(s) => Some(s.clone()),
        other => {
            d.error(
                k.span,
                format!(
                    "map keys must be \"quoted strings\", found {}",
                    other.describe()
                ),
                None,
            );
            None
        }
    }
}

fn list<'a>(v: &'a SValue, what: &str, d: &mut Diags) -> &'a [SValue] {
    match &v.kind {
        SKind::List(items) => items,
        other => {
            d.error(
                v.span,
                format!("{what} must be a list `[...]`, found {}", other.describe()),
                None,
            );
            &[]
        }
    }
}

fn map<'a>(v: &'a SValue, what: &str, d: &mut Diags) -> &'a [(SValue, SValue)] {
    match &v.kind {
        SKind::Map(items) => items,
        // `{}` and `()`-style empties are common slips; accept an empty tuple as an empty map.
        SKind::Tuple(None, items) if items.is_empty() => &[],
        other => {
            d.error(
                v.span,
                format!(
                    "{what} must be a map `{{ \"Name\": value, ... }}`, found {}",
                    other.describe()
                ),
                None,
            );
            &[]
        }
    }
}

/// Field access for a struct value, tracking which fields were used.
struct Fields<'a> {
    fields: &'a [Field],
    used: Vec<bool>,
    what: &'static str,
    allowed: &'static [&'static str],
    span: crate::model::Span,
}

impl<'a> Fields<'a> {
    fn new(
        v: &'a SValue,
        what: &'static str,
        allowed: &'static [&'static str],
        d: &mut Diags,
    ) -> Option<Self> {
        let fields: &[Field] = match &v.kind {
            SKind::Struct(_, f) => f,
            SKind::Tuple(_, items) if items.is_empty() => &[],
            other => {
                d.error(
                    v.span,
                    format!(
                        "{what} must be a struct `( field: value, ... )`, found {}",
                        other.describe()
                    ),
                    Some(format!("its fields are: {}", allowed.join(", "))),
                );
                return None;
            }
        };
        let mut used = vec![false; fields.len()];
        for (i, f) in fields.iter().enumerate() {
            if fields[..i].iter().any(|g| g.name == f.name) {
                d.error(
                    f.span,
                    format!("field `{}` appears twice in {what}", f.name),
                    Some("keep one of the two".into()),
                );
                used[i] = true;
            }
        }
        Some(Fields {
            fields,
            used,
            what,
            allowed,
            span: v.span,
        })
    }

    fn take(&mut self, name: &str) -> Option<&'a SValue> {
        let i = self.fields.iter().position(|f| f.name == name)?;
        self.used[i] = true;
        Some(&self.fields[i].value)
    }

    fn required_string(&mut self, name: &str, d: &mut Diags) -> Option<String> {
        match self.take(name) {
            Some(SValue {
                kind: SKind::Str(s),
                ..
            }) => Some(s.clone()),
            Some(other) => {
                d.error(
                    other.span,
                    format!(
                        "`{name}` must be a \"quoted string\", found {}",
                        other.kind.describe()
                    ),
                    None,
                );
                None
            }
            None => {
                d.error(
                    self.span,
                    format!("{} is missing the `{name}` field", self.what),
                    Some(format!("add `{name}: \"...\",`")),
                );
                None
            }
        }
    }

    fn finish(self, d: &mut Diags) {
        for (f, used) in self.fields.iter().zip(&self.used) {
            if !used {
                let help = match suggest(&f.name, self.allowed.iter().copied()) {
                    Some(s) => format!(
                        "did you mean `{s}`? Fields of {} are: {}",
                        self.what,
                        self.allowed.join(", ")
                    ),
                    None => format!("fields of {} are: {}", self.what, self.allowed.join(", ")),
                };
                d.error(
                    f.span,
                    format!("unknown field `{}` in {}", f.name, self.what),
                    Some(help),
                );
            }
        }
    }
}

const HANDLER_HELP: &str = "write a handler as `(source: \"Shell\", event: Death, spawn_number: 50, modules: [ ... ])`";

/// One entry of `event_handlers: [ ... ]`.
fn decode_handler(v: &SValue, d: &mut Diags) -> Option<crate::events::EventHandlerSource> {
    use crate::events::{EventExecutionMode, EventHandlerSource, EventKind, handler_field_names};
    if !matches!(v.kind, SKind::Struct(..)) {
        d.error(
            v.span,
            format!("an event handler must be a struct, found {}", v.kind.describe()),
            Some(HANDLER_HELP.into()),
        );
        return None;
    }
    let mut f = Fields::new(v, "an event handler", handler_field_names(), d)?;
    let source = f.required_string("source", d).unwrap_or_default();
    let event = f
        .take("event")
        .and_then(|v| ident_enum::<EventKind>(v, "event", d))
        .unwrap_or_default();
    let execution_mode = f
        .take("execution_mode")
        .and_then(|v| ident_enum::<EventExecutionMode>(v, "execution_mode", d))
        .unwrap_or_default();
    let count = |f: &mut Fields, name: &str, default: u32, help: &str, d: &mut Diags| {
        f.take(name)
            .and_then(|v| whole_in_range(v, name, 0..=1_000_000, help, d))
            .unwrap_or(default)
    };
    let spawn_number = count(
        &mut f,
        "spawn_number",
        1,
        "write e.g. `spawn_number: 50` (particles per event)",
        d,
    );
    let max_events_per_frame = count(
        &mut f,
        "max_events_per_frame",
        0,
        "write e.g. `max_events_per_frame: 10` (0 handles every event)",
        d,
    );
    let modules = f
        .take("modules")
        .map(|v| decode_stack(v, d))
        .unwrap_or_default();
    f.finish(d);
    Some(EventHandlerSource {
        source,
        event,
        execution_mode,
        spawn_number,
        max_events_per_frame,
        modules,
        span: v.span,
    })
}

/// The data interface type a `user_parameters` value declares, if it is one
/// (`SkeletalMesh("...")`, `ColorCurve([...])`, `CameraQuery`).
fn data_interface_tag(v: &SValue) -> Option<crate::data::DiType> {
    match &v.kind {
        SKind::Tuple(Some(tag), _) | SKind::Ident(tag) => crate::data::DiType::from_name(tag),
        _ => None,
    }
}

/// A data interface declaration in `user_parameters`.
fn decode_data_interface(
    ty: crate::data::DiType,
    v: &SValue,
    name: &str,
    span: Span,
    d: &mut Diags,
) -> Option<crate::data::DataInterfaceSource> {
    let mut src = crate::data::DataInterfaceSource {
        name: name.to_string(),
        ty,
        asset: None,
        curve: None,
        span,
    };
    let items: &[SValue] = match &v.kind {
        SKind::Tuple(_, items) => items,
        _ => &[],
    };
    if let Some(t) = ty.curve_type() {
        let [keys] = items else {
            d.error(
                v.span,
                format!("`{}` for \"{name}\" holds one list of `(time, value)` keys", ty.name()),
                Some(format!("e.g. `\"{name}\": {}`", ty.example())),
            );
            return None;
        };
        // Same keys as an inline curve: `FloatCurve([...])` is `Float([...])`.
        let inline = SValue {
            kind: SKind::Tuple(Some(t.name().to_string()), vec![keys.clone()]),
            span: v.span,
        };
        src.curve = Some(decode_curve(&inline, d)?);
        return Some(src);
    }
    match (ty.has_asset(), items) {
        (_, []) => {}
        (
            true,
            [SValue {
                kind: SKind::Str(path),
                ..
            }],
        ) => src.asset = Some(path.clone()),
        _ => {
            d.error(
                v.span,
                format!(
                    "`{}` for \"{name}\" is written {}",
                    ty.name(),
                    if ty.has_asset() {
                        "with one asset path, or bare when the host binds it"
                    } else {
                        "bare, with no values"
                    }
                ),
                Some(format!("e.g. `\"{name}\": {}`", ty.example())),
            );
            return None;
        }
    }
    if ty == crate::data::DiType::StaticMesh {
        if let Some((msg, help)) = src.asset.as_deref().and_then(mesh_name_error) {
            d.error(v.span, msg, Some(help));
            return None;
        }
    }
    Some(src)
}
