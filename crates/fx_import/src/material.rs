//! Unreal materials, textures and the packages they live in.
//!
//! - [`Store`] finds a referenced object's package under one or more
//!   package roots laid out like the project's mount points
//!   (`<root>/Game/VFX/M_X.uasset` for `/Game/VFX/M_X.M_X`).
//! - [`read_texture`] decodes a Texture2D's source art (editor bulk data:
//!   PNG, or raw G8 / BGRA8 / RGBA16 / RGBA16F pixels) to RGBA8.
//! - [`Material::load`] follows a material instance's parents to the base
//!   material, collecting parameter overrides, blend mode and shading model.
//! - [`compile`] translates the base material's expression graph (the
//!   common particle material nodes) into a WESL particle material for our
//!   renderers (`fx_render::particle_io`), sampling every texture it uses
//!   from one atlas, since our renderers bind one texture.
//!
//! Spaces: material math keeps Unreal's units, so positions and normals are
//! handed to it in Unreal space (centimetres, Z up: ours (x, y, z) metres ->
//! (x, z, y) * 100).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::package::{Package, PackageIndex};
use crate::property::{find, Property, Value};

// ---- packages ----

/// A loaded package and its bytes.
pub struct Loaded {
    pub file: PathBuf,
    pub bytes: Vec<u8>,
    pub pkg: Package,
}

impl Loaded {
    /// The export holding the object `name` (the asset when `name` is empty).
    pub fn export(&self, name: &str) -> Option<usize> {
        let e = &self.pkg.exports;
        (0..e.len())
            .find(|&i| !name.is_empty() && e[i].object_name == name && e[i].outer.is_null())
            .or_else(|| (0..e.len()).find(|&i| e[i].is_asset))
    }
    pub fn props(&self, i: usize) -> &[Property] {
        &self.pkg.exports[i].object.properties
    }
    /// Full object path of a reference from this package.
    pub fn path(&self, idx: PackageIndex) -> String {
        self.pkg.path(idx)
    }
}

/// Finds and caches packages under package roots.
#[derive(Default)]
pub struct Store {
    pub roots: Vec<PathBuf>,
    /// A project's `Content` folders: `/Game/A/B` is `<content>/A/B.uasset`.
    pub content_roots: Vec<PathBuf>,
    cache: HashMap<PathBuf, Option<Rc<Loaded>>>,
}

/// Splits `Class'/Game/A/B.C'`, `/Game/A/B.C` or `/Game/A/B.C:Sub` into
/// (package path, object name).
pub fn split_object_path(path: &str) -> (String, String) {
    let p = path.trim();
    let p = match (p.find('\''), p.rfind('\'')) {
        (Some(a), Some(b)) if b > a => &p[a + 1..b],
        _ => p,
    };
    let p = p.split(':').next().unwrap_or(p);
    match p.rsplit_once('.') {
        Some((pkg, obj)) if pkg.starts_with('/') => (pkg.to_string(), obj.to_string()),
        _ => (p.to_string(), p.rsplit('/').next().unwrap_or("").to_string()),
    }
}

impl Store {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Store { roots, content_roots: Vec::new(), cache: HashMap::new() }
    }

    /// Package roots for a system file: `packages/` beside it, then each
    /// folder of `FX_IMPORT_PACKAGES` (colon-separated), then the nearest
    /// enclosing `Content` folder as `/Game` (a Fab or Unreal project
    /// download: `Content/Pack/FX/NS_X.uasset`).
    pub fn for_system(file: &Path) -> Self {
        let mut roots = Vec::new();
        if let Some(dir) = file.parent() {
            roots.push(dir.join("packages"));
        }
        if let Ok(extra) = std::env::var("FX_IMPORT_PACKAGES") {
            roots.extend(extra.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
        }
        let mut store = Store::new(roots);
        if let Some(content) = file.ancestors().skip(1).find(|d| d.file_name().is_some_and(|n| n == "Content")) {
            store.content_roots.push(content.to_path_buf());
        }
        store
    }

    /// The file of an object's package, if present under a root.
    pub fn resolve(&self, object_path: &str) -> Option<PathBuf> {
        let (pkg, _) = split_object_path(object_path);
        let rel = pkg.trim_start_matches('/');
        if rel.is_empty() || rel.starts_with("Script/") {
            return None;
        }
        let game = rel.strip_prefix("Game/").map(|r| (r, &self.content_roots));
        self.roots
            .iter()
            .map(|r| r.join(format!("{rel}.uasset")))
            .chain(game.into_iter().flat_map(|(r, cs)| cs.iter().map(move |c| c.join(format!("{r}.uasset")))))
            .find(|f| f.is_file())
    }

    /// The object's package, read once.
    pub fn load(&mut self, object_path: &str) -> Option<Rc<Loaded>> {
        let file = self.resolve(object_path)?;
        if let Some(c) = self.cache.get(&file) {
            return c.clone();
        }
        let loaded = std::fs::read(&file).ok().and_then(|bytes| {
            let pkg = Package::read(&bytes).ok()?;
            Some(Rc::new(Loaded { file: file.clone(), bytes, pkg }))
        });
        self.cache.insert(file, loaded.clone());
        loaded
    }
}

// ---- textures ----

/// A decoded texture: RGBA8, rows top to bottom.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Unreal samples it as sRGB colour (false: linear data such as masks).
    pub srgb: bool,
}

fn enum_tail(v: Option<&Value>) -> Option<String> {
    v.and_then(|v| v.as_str()).map(|s| s.rsplit("::").next().unwrap_or(s).to_string())
}

/// Decodes a Texture2D's source art.
pub fn read_texture(store: &mut Store, object_path: &str) -> Result<Image, String> {
    let l = store.load(object_path).ok_or_else(|| format!("package of `{object_path}` not found"))?;
    let (_, name) = split_object_path(object_path);
    let i = l.export(&name).ok_or("no texture export")?;
    let props = l.props(i);
    let srgb = !matches!(find(props, "SRGB"), Some(Value::Bool(false)));
    let src = find(props, "Source").ok_or("texture has no Source")?;
    let get_i = |n: &str| src.field(n).and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;
    let (w, h) = (get_i("SizeX"), get_i("SizeY"));
    let format = enum_tail(src.field("Format")).unwrap_or_default();
    // UE4 (and early UE5) saves say bPNGCompressed instead of CompressionFormat.
    let png = matches!(src.field("bPNGCompressed"), Some(Value::Bool(true)));
    let compression = enum_tail(src.field("CompressionFormat"))
        .unwrap_or_else(|| if png { "TSCF_PNG" } else { "TSCF_None" }.into());
    let obj = &l.pkg.exports[i].object;
    let data = if l.pkg.summary.ue5_version == 0 {
        crate::bulk::ue4_texture_source(&l.pkg, &l.bytes, obj).map_err(|e| e.to_string())?
    } else {
        let payloads = crate::bulk::payloads_of(&l.pkg, &l.bytes, obj).map_err(|e| e.to_string())?;
        let p = payloads.first().ok_or("texture source payload not in the package")?;
        crate::bulk::read_payload(&l.bytes, p).map_err(|e| e.to_string())?
    };
    let rgba = match compression.as_str() {
        "TSCF_PNG" => decode_png(&data)?.rgba,
        "TSCF_None" => raw_to_rgba(&data, w, h, &format)?,
        other => return Err(format!("texture source compression {other} not supported")),
    };
    if rgba.len() != (w * h * 4) as usize {
        return Err(format!("texture is {} bytes for {w}x{h}", rgba.len()));
    }
    Ok(Image { width: w, height: h, rgba, srgb })
}

fn raw_to_rgba(d: &[u8], w: u32, h: u32, format: &str) -> Result<Vec<u8>, String> {
    let n = (w * h) as usize;
    let need = |b: usize| if d.len() >= n * b { Ok(()) } else { Err(format!("{format}: {} bytes for {w}x{h}", d.len())) };
    let mut out = Vec::with_capacity(n * 4);
    match format {
        "TSF_G8" => {
            need(1)?;
            for &g in &d[..n] {
                out.extend_from_slice(&[g, g, g, 255]);
            }
        }
        "TSF_BGRA8" | "TSF_BGRE8" => {
            need(4)?;
            for p in d[..n * 4].chunks(4) {
                out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
            }
        }
        "TSF_RGBA16" => {
            need(8)?;
            for p in d[..n * 8].chunks(8) {
                for c in 0..4 {
                    out.push(p[c * 2 + 1]);
                }
            }
        }
        "TSF_G16" => {
            need(2)?;
            for p in d[..n * 2].chunks(2) {
                out.extend_from_slice(&[p[1], p[1], p[1], 255]);
            }
        }
        "TSF_RGBA16F" => {
            need(8)?;
            for p in d[..n * 8].chunks(8) {
                for c in 0..4 {
                    let v = half_to_f32(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]]));
                    out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        }
        other => return Err(format!("texture source format {other} not supported")),
    }
    Ok(out)
}

fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Decodes a PNG to RGBA8.
pub fn decode_png(data: &[u8]) -> Result<Image, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(data));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let px = &buf[..info.buffer_size()];
    let n = (info.width * info.height) as usize;
    let mut rgba = Vec::with_capacity(n * 4);
    match info.color_type {
        png::ColorType::Rgba => rgba.extend_from_slice(px),
        png::ColorType::Rgb => px.chunks(3).for_each(|p| rgba.extend_from_slice(&[p[0], p[1], p[2], 255])),
        png::ColorType::GrayscaleAlpha => px.chunks(2).for_each(|p| rgba.extend_from_slice(&[p[0], p[0], p[0], p[1]])),
        png::ColorType::Grayscale => px.iter().for_each(|&g| rgba.extend_from_slice(&[g, g, g, 255])),
        png::ColorType::Indexed => return Err("indexed PNG left unexpanded".into()),
    }
    Ok(Image { width: info.width, height: info.height, rgba, srgb: true })
}

/// Encodes RGBA8 as PNG.
pub fn encode_png(img: &Image) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("png header");
        w.write_image_data(&img.rgba).expect("png data");
    }
    out
}

fn linear_to_srgb8(v: u8) -> u8 {
    let x = v as f32 / 255.0;
    let s = if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Textures packed side by side into one image (our renderers bind one
/// texture); `rects[i]` = (u, v, width, height) of texture i in UV units.
/// The atlas is loaded as sRGB, so linear textures are stored sRGB-encoded
/// and sample back as their original values.
pub struct Atlas {
    pub image: Image,
    pub rects: Vec<[f32; 4]>,
}

const PAD: u32 = 2;

pub fn atlas(images: &[Image]) -> Atlas {
    if images.len() == 1 {
        let mut img = images[0].clone();
        if !img.srgb {
            for p in img.rgba.chunks_mut(4) {
                for c in &mut p[..3] {
                    *c = linear_to_srgb8(*c);
                }
            }
            img.srgb = true;
        }
        return Atlas { image: img, rects: vec![[0.0, 0.0, 1.0, 1.0]] };
    }
    let w: u32 = images.iter().map(|i| i.width + 2 * PAD).sum();
    let h: u32 = images.iter().map(|i| i.height + 2 * PAD).max().unwrap_or(1);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut rects = Vec::new();
    let mut x0 = 0;
    for img in images {
        // Edge texels repeat into the padding (clamped bilinear at the seams).
        for y in 0..img.height + 2 * PAD {
            let sy = y.saturating_sub(PAD).min(img.height - 1);
            for x in 0..img.width + 2 * PAD {
                let sx = x.saturating_sub(PAD).min(img.width - 1);
                let s = ((sy * img.width + sx) * 4) as usize;
                let d = ((y * w + x0 + x) * 4) as usize;
                for c in 0..4 {
                    let v = img.rgba[s + c];
                    rgba[d + c] = if c < 3 && !img.srgb { linear_to_srgb8(v) } else { v };
                }
            }
        }
        rects.push([
            (x0 + PAD) as f32 / w as f32,
            PAD as f32 / h as f32,
            img.width as f32 / w as f32,
            img.height as f32 / h as f32,
        ]);
        x0 += img.width + 2 * PAD;
    }
    Atlas {
        image: Image { width: w, height: h, rgba, srgb: true },
        rects,
    }
}

// ---- materials ----

/// EBlendMode, as far as it matters here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    Opaque,
    Masked,
    Translucent,
    Additive,
    Modulate,
    AlphaComposite,
    AlphaHoldout,
}

impl Blend {
    fn parse(s: &str) -> Blend {
        match s.rsplit("::").next().unwrap_or(s) {
            "BLEND_Masked" => Blend::Masked,
            "BLEND_Translucent" => Blend::Translucent,
            "BLEND_Additive" => Blend::Additive,
            "BLEND_Modulate" => Blend::Modulate,
            "BLEND_AlphaComposite" => Blend::AlphaComposite,
            "BLEND_AlphaHoldout" => Blend::AlphaHoldout,
            _ => Blend::Opaque,
        }
    }
    /// Our renderer blend mode (sprites and ribbons have no Opaque).
    pub fn ours(self, mesh: bool) -> &'static str {
        match self {
            Blend::Additive => "Additive",
            Blend::AlphaComposite => "Premultiplied",
            Blend::Opaque | Blend::Masked if mesh => "Opaque",
            _ => "Alpha",
        }
    }
}

/// A material instance chain resolved to its base material.
pub struct Material {
    /// The object paths followed: the instance first, the base material last.
    pub chain: Vec<String>,
    pub blend: Blend,
    pub unlit: bool,
    pub two_sided: bool,
    pub clip: f32,
    pub scalars: HashMap<String, f32>,
    pub vectors: HashMap<String, [f32; 4]>,
    pub textures: HashMap<String, String>,
    /// The base material's package and its editor-only data export (the graph).
    pub base: Option<(Rc<Loaded>, usize)>,
    /// Parents that could not be loaded (the chain stops there).
    pub missing: Option<String>,
}

fn param_name(v: &Value) -> Option<String> {
    v.field("ParameterInfo").and_then(|i| i.field("Name")).and_then(|n| n.as_str()).map(String::from)
}

fn color4(v: &Value) -> Option<[f32; 4]> {
    let c = |n: &str| v.field(n).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    matches!(v, Value::Struct { .. }).then(|| [c("R"), c("G"), c("B"), c("A")])
}

impl Material {
    pub fn load(store: &mut Store, object_path: &str) -> Result<Material, String> {
        let mut m = Material {
            chain: Vec::new(),
            blend: Blend::Opaque,
            unlit: false,
            two_sided: false,
            clip: 0.3333,
            scalars: HashMap::new(),
            vectors: HashMap::new(),
            textures: HashMap::new(),
            base: None,
            missing: None,
        };
        let mut path = object_path.to_string();
        let mut blend_override: Option<Blend> = None;
        let mut unlit_override: Option<bool> = None;
        let mut two_sided_override: Option<bool> = None;
        for _ in 0..16 {
            let Some(l) = store.load(&path) else {
                if m.chain.is_empty() {
                    return Err(format!("package of `{path}` not found"));
                }
                m.missing = Some(path);
                break;
            };
            m.chain.push(path.clone());
            let (_, name) = split_object_path(&path);
            let i = l.export(&name).ok_or_else(|| format!("no export `{name}`"))?;
            let class = l.pkg.export_class(i);
            let props = l.props(i).to_vec();
            if class == "Material" {
                m.blend = blend_override.unwrap_or_else(|| find(&props, "BlendMode").and_then(|v| v.as_str()).map(Blend::parse).unwrap_or(Blend::Opaque));
                let shading = find(&props, "ShadingModel").and_then(|v| v.as_str()).unwrap_or("MSM_DefaultLit");
                m.unlit = unlit_override.unwrap_or(shading.ends_with("MSM_Unlit"));
                m.two_sided = two_sided_override.unwrap_or(matches!(find(&props, "TwoSided"), Some(Value::Bool(true))));
                if let Some(c) = find(&props, "OpacityMaskClipValue").and_then(|v| v.as_f64()) {
                    m.clip = c as f32;
                }
                let editor = match find(&props, "EditorOnlyData") {
                    Some(Value::Object(e)) => e.export(),
                    _ => None,
                };
                // Older packages keep the graph on the material itself.
                m.base = Some((l.clone(), editor.unwrap_or(i)));
                return Ok(m);
            }
            // A material instance: its overrides win over its parents'.
            for v in arr(find(&props, "ScalarParameterValues")) {
                if let (Some(n), Some(x)) = (param_name(v), v.field("ParameterValue").and_then(|x| x.as_f64())) {
                    m.scalars.entry(n).or_insert(x as f32);
                }
            }
            for v in arr(find(&props, "VectorParameterValues")) {
                if let (Some(n), Some(c)) = (param_name(v), v.field("ParameterValue").and_then(color4)) {
                    m.vectors.entry(n).or_insert(c);
                }
            }
            for v in arr(find(&props, "TextureParameterValues")) {
                if let (Some(n), Some(Value::Object(t))) = (param_name(v), v.field("ParameterValue")) {
                    if !t.is_null() {
                        m.textures.entry(n).or_insert(l.path(*t));
                    }
                }
            }
            if let Some(o) = find(&props, "BasePropertyOverrides") {
                let flag = |n: &str| matches!(o.field(n), Some(Value::Bool(true)));
                if flag("bOverride_BlendMode") && blend_override.is_none() {
                    blend_override = o.field("BlendMode").and_then(|v| v.as_str()).map(Blend::parse);
                }
                if flag("bOverride_ShadingModel") && unlit_override.is_none() {
                    unlit_override = o.field("ShadingModel").and_then(|v| v.as_str()).map(|s| s.ends_with("MSM_Unlit"));
                }
                if flag("bOverride_TwoSided") && two_sided_override.is_none() {
                    two_sided_override = Some(matches!(o.field("TwoSided"), Some(Value::Bool(true))));
                }
                if flag("bOverride_OpacityMaskClipValue") {
                    if let Some(c) = o.field("OpacityMaskClipValue").and_then(|v| v.as_f64()) {
                        m.clip = c as f32;
                    }
                }
            }
            match find(&props, "Parent") {
                Some(Value::Object(p)) if !p.is_null() => path = l.path(*p),
                _ => {
                    m.missing = Some(format!("{path} has no parent"));
                    break;
                }
            }
        }
        // No base material reached: keep what the instances said.
        if let Some(b) = blend_override {
            m.blend = b;
        }
        m.unlit = unlit_override.unwrap_or(false);
        Ok(m)
    }

    /// Short name of the instance (last path element).
    pub fn name(&self) -> String {
        self.chain.first().map(|p| split_object_path(p).1).unwrap_or_default()
    }
}

fn arr(v: Option<&Value>) -> &[Value] {
    match v {
        Some(Value::Array(a)) => a,
        _ => &[],
    }
}

// ---- graph -> WESL ----

/// What the generated material needs from the effect.
#[derive(Clone, Debug, Default)]
pub struct Needs {
    /// Which Dynamic Material Parameter indices the emitter writes; unwritten
    /// ones read the material's default value, as in Unreal.
    pub dynamic_written: [bool; 4],
    /// Dynamic parameter indices our renderers pass to materials.
    pub dynamic_available: [bool; 4],
    /// The renderer is a mesh (lit with Lambert; otherwise half-Lambert).
    pub mesh: bool,
}

/// A translated material.
pub struct Compiled {
    pub wesl: String,
    /// Object paths of the textures sampled, in atlas order.
    pub textures: Vec<String>,
    /// Graph parts approximated or dropped, with their visible effect.
    pub notes: Vec<String>,
    /// Dynamic parameter indices the material reads.
    pub dynamic_read: [bool; 4],
    /// The graph moves vertices (World Position Offset): not carried over.
    pub world_position_offset: bool,
}

#[derive(Clone)]
struct V {
    code: String,
    n: u8,
}

fn v(code: impl Into<String>, n: u8) -> V {
    V { code: code.into(), n }
}

fn lit(x: f32) -> String {
    let s = format!("{x:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

fn ty(n: u8) -> &'static str {
    match n {
        1 => "f32",
        2 => "vec2<f32>",
        3 => "vec3<f32>",
        _ => "vec4<f32>",
    }
}

/// `x` as `n` components (Unreal's rules: scalars splat, longer vectors truncate).
fn cast(x: &V, n: u8) -> V {
    if x.n == n {
        return x.clone();
    }
    if x.n == 1 {
        return v(format!("{}({})", ty(n), x.code), n);
    }
    if x.n > n {
        let sw = &"xyzw"[..n as usize];
        return v(format!("({}).{sw}", x.code), n);
    }
    let zeros = vec!["0.0"; (n - x.n) as usize].join(", ");
    v(format!("{}({}, {zeros})", ty(n), x.code), n)
}

/// Operands of a component-wise binary op, made compatible.
fn pair(a: &V, b: &V) -> (V, V, u8) {
    if a.n == b.n {
        (a.clone(), b.clone(), a.n)
    } else if a.n == 1 || b.n == 1 {
        let n = a.n.max(b.n);
        (cast(a, n), cast(b, n), n)
    } else {
        let n = a.n.min(b.n);
        (cast(a, n), cast(b, n), n)
    }
}

struct Gen<'a> {
    l: &'a Loaded,
    m: &'a Material,
    needs: &'a Needs,
    lines: Vec<String>,
    done: HashMap<usize, V>,
    textures: Vec<String>,
    notes: BTreeMap<String, ()>,
    dynamic_read: [bool; 4],
    depth: usize,
}

impl<'a> Gen<'a> {
    fn note(&mut self, s: impl Into<String>) {
        self.notes.insert(s.into(), ());
    }

    fn props(&self, e: usize) -> &'a [Property] {
        &self.l.pkg.exports[e].object.properties
    }

    fn f(&self, e: usize, name: &str, default: f32) -> f32 {
        find(self.props(e), name).and_then(|v| v.as_f64()).map(|x| x as f32).unwrap_or(default)
    }

    fn texture_slot(&mut self, path: String) -> usize {
        if let Some(i) = self.textures.iter().position(|t| *t == path) {
            return i;
        }
        self.textures.push(path);
        self.textures.len() - 1
    }

    /// The value an expression input carries (None when unconnected).
    fn input(&mut self, e: usize, name: &str) -> Option<V> {
        let inp = find(self.props(e), name)?;
        self.link(inp)
    }

    fn link(&mut self, inp: &Value) -> Option<V> {
        let idx = match inp.field("Expression") {
            Some(Value::Object(i)) => i.export()?,
            _ => return None,
        };
        let out_index = inp.field("OutputIndex").and_then(|x| x.as_f64()).unwrap_or(0.0) as usize;
        let full = self.expr(idx, out_index);
        let mi = |n: &str| inp.field(n).and_then(|x| x.as_f64()).unwrap_or(0.0) != 0.0;
        if mi("Mask") {
            let sw: String = ["MaskR", "MaskG", "MaskB", "MaskA"]
                .iter()
                .zip("xyzw".chars())
                .filter(|(n, _)| mi(n))
                .map(|(_, c)| c)
                .collect();
            if sw.is_empty() {
                return Some(full);
            }
            let need = sw.chars().map(|c| "xyzw".find(c).unwrap() as u8 + 1).max().unwrap();
            if full.n == 1 && sw.len() == 1 {
                return Some(full);
            }
            let src = if full.n < need { cast(&full, need) } else { full };
            return Some(v(format!("({}).{sw}", src.code), sw.len() as u8));
        }
        Some(full)
    }

    fn input_or(&mut self, e: usize, name: &str, cname: &str, default: f32) -> V {
        match self.input(e, name) {
            Some(x) => x,
            None => v(lit(self.f(e, cname, default)), 1),
        }
    }

    /// Expression `e`'s value (output `out` for multi-output nodes without a
    /// mask), emitted once as a `let`.
    fn expr(&mut self, e: usize, out: usize) -> V {
        let key = e * 8 + out.min(7);
        if let Some(x) = self.done.get(&key) {
            return x.clone();
        }
        self.depth += 1;
        let val = if self.depth > 200 { v("0.0", 1) } else { self.eval(e, out) };
        self.depth -= 1;
        let name = format!("e{e}_{out}");
        let simple = val.code.parse::<f32>().is_ok() || val.code.starts_with("in.") && !val.code.contains(' ');
        let res = if simple {
            val
        } else {
            self.lines.push(format!("    let {name}: {} = {};", ty(val.n), val.code));
            v(name, val.n)
        };
        self.done.insert(key, res.clone());
        res
    }

    fn sample(&mut self, slot: usize, uv: &V) -> V {
        let uv = cast(uv, 2);
        v(format!("tex({slot}u, {})", uv.code), 4)
    }

    fn eval(&mut self, e: usize, out: usize) -> V {
        let class = self.l.pkg.export_class(e);
        let class = class.strip_prefix("MaterialExpression").unwrap_or(&class).to_string();
        let p = self.props(e);
        let fprop = |n: &str, d: f32| find(p, n).and_then(|v| v.as_f64()).map(|x| x as f32).unwrap_or(d);
        let color_prop = |n: &str| find(p, n).and_then(color4);
        match class.as_str() {
            "Constant" => v(lit(fprop("R", 0.0)), 1),
            "Constant2Vector" => v(format!("vec2<f32>({}, {})", lit(fprop("R", 0.0)), lit(fprop("G", 0.0))), 2),
            "Constant3Vector" => {
                let c = color_prop("Constant").unwrap_or([0.0; 4]);
                v(format!("vec3<f32>({}, {}, {})", lit(c[0]), lit(c[1]), lit(c[2])), 3)
            }
            "Constant4Vector" => {
                let c = color_prop("Constant").unwrap_or([0.0; 4]);
                v(format!("vec4<f32>({}, {}, {}, {})", lit(c[0]), lit(c[1]), lit(c[2]), lit(c[3])), 4)
            }
            "ScalarParameter" | "CollectionParameter" => {
                let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let x = self.m.scalars.get(&name).copied().unwrap_or(fprop("DefaultValue", 0.0));
                v(lit(x), 1)
            }
            "VectorParameter" => {
                let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let c = self.m.vectors.get(&name).copied().or_else(|| color_prop("DefaultValue")).unwrap_or([0.0, 0.0, 0.0, 1.0]);
                let full = v(format!("vec4<f32>({}, {}, {}, {})", lit(c[0]), lit(c[1]), lit(c[2]), lit(c[3])), 4);
                match out {
                    1..=4 => v(lit(c[out - 1]), 1),
                    _ => full,
                }
            }
            "StaticBoolParameter" | "StaticSwitchParameter" | "StaticSwitch" => {
                let on = matches!(find(p, "DefaultValue"), Some(Value::Bool(true)));
                if class == "StaticBoolParameter" {
                    return v(if on { "1.0" } else { "0.0" }, 1);
                }
                self.note("static switches use their default values");
                let pick = if on { "A" } else { "B" };
                self.input(e, pick).unwrap_or(v("0.0", 1))
            }
            "TextureCoordinate" => {
                let (u, vv) = (fprop("UTiling", 1.0), fprop("VTiling", 1.0));
                if u == 1.0 && vv == 1.0 {
                    v("in.uv0", 2)
                } else {
                    v(format!("in.uv0 * vec2<f32>({}, {})", lit(u), lit(vv)), 2)
                }
            }
            "Time" => {
                let period = fprop("Period", 0.0);
                if period > 0.0 {
                    v(format!("(globals.time % {})", lit(period)), 1)
                } else {
                    v("globals.time", 1)
                }
            }
            "Panner" => {
                let coord = self.input(e, "Coordinate").unwrap_or(v("in.uv0", 2));
                let time = self.input(e, "Time").unwrap_or(v("globals.time", 1));
                let speed = match self.input(e, "Speed") {
                    Some(s) => cast(&s, 2),
                    None => v(format!("vec2<f32>({}, {})", lit(fprop("SpeedX", 0.0)), lit(fprop("SpeedY", 0.0))), 2),
                };
                let r = format!("{} + {} * {}", cast(&coord, 2).code, speed.code, cast(&time, 1).code);
                if matches!(find(p, "bFractionalPart"), Some(Value::Bool(true))) {
                    v(format!("fract({r})"), 2)
                } else {
                    v(r, 2)
                }
            }
            "Rotator" => {
                let coord = self.input(e, "Coordinate").unwrap_or(v("in.uv0", 2));
                let time = self.input(e, "Time").unwrap_or(v("globals.time", 1));
                let (cx, cy, speed) = (fprop("CenterX", 0.5), fprop("CenterY", 0.5), fprop("Speed", 0.25));
                v(
                    format!(
                        "rotate_uv({}, vec2<f32>({}, {}), {} * {})",
                        cast(&coord, 2).code,
                        lit(cx),
                        lit(cy),
                        cast(&time, 1).code,
                        lit(speed)
                    ),
                    2,
                )
            }
            "TextureSample" | "TextureSampleParameter2D" | "TextureObject" | "TextureObjectParameter" | "ParticleSubUV" => {
                let mut tex = match find(p, "Texture") {
                    Some(Value::Object(t)) if !t.is_null() => Some(self.l.path(*t)),
                    _ => None,
                };
                if class.ends_with("Parameter2D") || class == "TextureObjectParameter" {
                    let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    if let Some(t) = self.m.textures.get(&name) {
                        tex = Some(t.clone());
                    }
                }
                // A texture object wired into the sampler wins.
                if let Some(Value::Struct { .. }) = find(p, "TextureObject") {
                    if let Some(obj) = find(p, "TextureObject").and_then(|i| match i.field("Expression") {
                        Some(Value::Object(x)) => x.export(),
                        _ => None,
                    }) {
                        let op = self.props(obj);
                        if let Some(Value::Object(t)) = find(op, "Texture") {
                            if !t.is_null() {
                                tex = Some(self.l.path(*t));
                            }
                        }
                    }
                }
                if class.starts_with("TextureObject") {
                    return v("0.0", 1); // only meaningful through a sampler
                }
                let Some(tex) = tex else {
                    self.note("a texture sampler without a texture samples white");
                    return v("vec4<f32>(1.0)", 4);
                };
                let slot = self.texture_slot(tex);
                let full = if class == "ParticleSubUV" {
                    let a = self.sample(slot, &v("in.uv0", 2));
                    let b = self.sample(slot, &v("in.uv1", 2));
                    if matches!(find(p, "bBlend"), Some(Value::Bool(true))) {
                        v(format!("mix({}, {}, in.sub_uv_blend)", a.code, b.code), 4)
                    } else {
                        a
                    }
                } else {
                    let uv = self.input(e, "Coordinates").unwrap_or(v("in.uv0", 2));
                    self.sample(slot, &uv)
                };
                let full = self.expr_inline(full);
                match out {
                    1 => v(format!("{}.x", full.code), 1),
                    2 => v(format!("{}.y", full.code), 1),
                    3 => v(format!("{}.z", full.code), 1),
                    4 => v(format!("{}.w", full.code), 1),
                    _ => full,
                }
            }
            "Multiply" | "Add" | "Subtract" | "Divide" | "Max" | "Min" => {
                let (ca, cb) = match class.as_str() {
                    "Multiply" => (0.0, 1.0),
                    "Add" => (0.0, 1.0),
                    "Subtract" => (1.0, 1.0),
                    "Divide" => (1.0, 2.0),
                    _ => (0.0, 0.0),
                };
                let a = self.input_or(e, "A", "ConstA", ca);
                let b = self.input_or(e, "B", "ConstB", cb);
                let (a, b, n) = pair(&a, &b);
                match class.as_str() {
                    "Multiply" => v(format!("({} * {})", a.code, b.code), n),
                    "Add" => v(format!("({} + {})", a.code, b.code), n),
                    "Subtract" => v(format!("({} - {})", a.code, b.code), n),
                    "Divide" => v(format!("({} / {})", a.code, b.code), n),
                    "Max" => v(format!("max({}, {})", a.code, b.code), n),
                    _ => v(format!("min({}, {})", a.code, b.code), n),
                }
            }
            "Power" => {
                let b = self.input(e, "Base").unwrap_or(v("0.0", 1));
                let x = self.input_or(e, "Exponent", "ConstExponent", 2.0);
                let x = if x.n == 1 { cast(&x, b.n) } else { x };
                let (b, x, n) = pair(&b, &x);
                v(format!("pow(max({}, {}(0.0)), {})", b.code, ty(n), x.code), n)
            }
            "OneMinus" => {
                let x = self.input(e, "Input").unwrap_or(v("0.0", 1));
                v(format!("(1.0 - {})", x.code), x.n)
            }
            "Abs" | "Floor" | "Ceil" | "Frac" | "Saturate" | "Normalize" | "SquareRoot" => {
                let x = self.input(e, if class == "Normalize" { "VectorInput" } else { "Input" }).unwrap_or(v("0.0", 1));
                let f = match class.as_str() {
                    "Abs" => "abs",
                    "Floor" => "floor",
                    "Ceil" => "ceil",
                    "Frac" => "fract",
                    "Normalize" => "normalize",
                    "SquareRoot" => "sqrt",
                    _ => "saturate",
                };
                if f == "normalize" && x.n == 1 {
                    return v("1.0", 1);
                }
                v(format!("{f}({})", x.code), x.n)
            }
            "Sine" | "Cosine" => {
                let x = self.input(e, "Input").unwrap_or(v("0.0", 1));
                let period = fprop("Period", 1.0);
                let f = if class == "Sine" { "sin" } else { "cos" };
                if period > 0.0 {
                    v(format!("{f}({} * {})", x.code, lit(std::f32::consts::TAU / period)), x.n)
                } else {
                    v(format!("{f}({})", x.code), x.n)
                }
            }
            "Clamp" => {
                let x = self.input(e, "Input").unwrap_or(v("0.0", 1));
                let lo = self.input_or(e, "Min", "MinDefault", 0.0);
                let hi = self.input_or(e, "Max", "MaxDefault", 1.0);
                let mode = find(p, "ClampMode").and_then(|v| v.as_str()).unwrap_or("CMODE_Clamp").to_string();
                let lo = cast(&lo, x.n);
                let hi = cast(&hi, x.n);
                match mode.rsplit("::").next().unwrap_or("") {
                    "CMODE_ClampMin" => v(format!("max({}, {})", x.code, lo.code), x.n),
                    "CMODE_ClampMax" => v(format!("min({}, {})", x.code, hi.code), x.n),
                    _ => v(format!("clamp({}, {}, {})", x.code, lo.code, hi.code), x.n),
                }
            }
            "LinearInterpolate" => {
                let a = self.input_or(e, "A", "ConstA", 0.0);
                let b = self.input_or(e, "B", "ConstB", 1.0);
                let t = self.input_or(e, "Alpha", "ConstAlpha", 0.5);
                let (a, b, n) = pair(&a, &b);
                let t = if t.n == 1 { t } else { cast(&t, n) };
                v(format!("mix({}, {}, {})", a.code, b.code, t.code), n)
            }
            "ComponentMask" => {
                let x = self.input(e, "Input").unwrap_or(v("0.0", 1));
                let sw: String = ["R", "G", "B", "A"]
                    .iter()
                    .zip("xyzw".chars())
                    .filter(|(n, _)| matches!(find(p, n), Some(Value::Bool(true))))
                    .map(|(_, c)| c)
                    .collect();
                if sw.is_empty() || x.n == 1 {
                    return x;
                }
                let need = sw.chars().map(|c| "xyzw".find(c).unwrap() as u8 + 1).max().unwrap();
                let src = if x.n < need { cast(&x, need) } else { x };
                v(format!("({}).{sw}", src.code), sw.len() as u8)
            }
            "AppendVector" => {
                let a = self.input(e, "A").unwrap_or(v("0.0", 1));
                let b = self.input(e, "B").unwrap_or(v("0.0", 1));
                let n = (a.n + b.n).min(4);
                let b = if a.n + b.n > 4 { cast(&b, 4 - a.n) } else { b };
                v(format!("{}({}, {})", ty(n), a.code, b.code), n)
            }
            "Desaturation" => {
                let x = self.input(e, "Input").unwrap_or(v("vec3<f32>(0.0)", 3));
                let x3 = cast(&x, 3);
                let frac = self.input(e, "Fraction").map(|f| cast(&f, 1)).unwrap_or(v("1.0", 1));
                let lum = color_prop("LuminanceFactors").unwrap_or([0.3, 0.59, 0.11, 0.0]);
                v(
                    format!(
                        "mix({0}, vec3<f32>(dot({0}, vec3<f32>({1}, {2}, {3}))), {4})",
                        x3.code,
                        lit(lum[0]),
                        lit(lum[1]),
                        lit(lum[2]),
                        frac.code
                    ),
                    3,
                )
            }
            "DotProduct" => {
                let a = self.input(e, "A").unwrap_or(v("0.0", 1));
                let b = self.input(e, "B").unwrap_or(v("0.0", 1));
                let (a, b, n) = pair(&a, &b);
                if n == 1 {
                    v(format!("({} * {})", a.code, b.code), 1)
                } else {
                    v(format!("dot({}, {})", a.code, b.code), 1)
                }
            }
            "CrossProduct" => {
                let a = cast(&self.input(e, "A").unwrap_or(v("0.0", 1)), 3);
                let b = cast(&self.input(e, "B").unwrap_or(v("0.0", 1)), 3);
                v(format!("cross({}, {})", a.code, b.code), 3)
            }
            "Distance" => {
                let a = self.input(e, "A").unwrap_or(v("0.0", 1));
                let b = self.input(e, "B").unwrap_or(v("0.0", 1));
                let (a, b, _) = pair(&a, &b);
                v(format!("distance({}, {})", a.code, b.code), 1)
            }
            "SphereMask" => {
                let a = self.input(e, "A").unwrap_or(v("0.0", 1));
                let b = self.input(e, "B").unwrap_or(v("0.0", 1));
                let r = self.input_or(e, "Radius", "AttenuationRadius", 256.0);
                let h = self.input_or(e, "Hardness", "HardnessPercent", 100.0);
                let (a, b, _) = pair(&a, &b);
                v(
                    format!(
                        "saturate((1.0 - distance({}, {}) / {}) / max(1.0 - {} * 0.01, 0.00001))",
                        a.code,
                        b.code,
                        cast(&r, 1).code,
                        cast(&h, 1).code
                    ),
                    1,
                )
            }
            "WorldPosition" => v("ue_world_position(in)", 3),
            "ActorPositionWS" | "ObjectPositionWS" => v("ue_actor_position()", 3),
            "CameraPositionWS" => v("ue_camera_position()", 3),
            "CameraVectorWS" => v("normalize(ue_camera_position() - ue_world_position(in))", 3),
            "VertexNormalWS" | "PixelNormalWS" => v("ue_normal(in)", 3),
            "VertexColor" | "ParticleColor" => match out {
                1 => v("in.color.x", 1),
                2 => v("in.color.y", 1),
                3 => v("in.color.z", 1),
                4 => v("in.color.w", 1),
                _ => v("in.color", 4),
            },
            "ParticleRelativeTime" => v("in.normalized_age", 1),
            "DynamicParameter" => {
                let k = (fprop("ParameterIndex", 0.0) as usize).min(3);
                let d = color_prop("DefaultValue").unwrap_or([1.0; 4]);
                let full = if self.needs.dynamic_written[k] && self.needs.dynamic_available[k] {
                    self.dynamic_read[k] = true;
                    let field = if k == 0 { "dynamic_param".to_string() } else { format!("dynamic_param_{k}") };
                    v(format!("in.{field}"), 4)
                } else {
                    if self.needs.dynamic_written[k] {
                        self.note(format!("dynamic material parameter {k}: written by the emitter but not passed to materials here; its default is used"));
                    }
                    v(format!("vec4<f32>({}, {}, {}, {})", lit(d[0]), lit(d[1]), lit(d[2]), lit(d[3])), 4)
                };
                match out {
                    0..=3 => v(format!("{}.{}", full.code, &"xyzw"[out..out + 1]), 1),
                    _ => full,
                }
            }
            "Fresnel" => {
                let ex = self.input_or(e, "ExponentIn", "Exponent", 5.0);
                let base = self.input_or(e, "BaseReflectFractionIn", "BaseReflectFraction", 0.04);
                v(
                    format!(
                        "(pow(max(1.0 - max(dot(ue_normal(in), normalize(ue_camera_position() - ue_world_position(in))), 0.0), 0.0), {}) * (1.0 - {1}) + {1})",
                        cast(&ex, 1).code,
                        cast(&base, 1).code
                    ),
                    1,
                )
            }
            "DepthFade" => {
                self.note("Depth Fade: opacity is not faded near scene geometry");
                self.input(e, "InOpacity").unwrap_or_else(|| v(lit(fprop("OpacityDefault", 1.0)), 1))
            }
            "Reroute" | "NamedRerouteDeclaration" => self.input(e, "Input").unwrap_or(v("0.0", 1)),
            "NamedRerouteUsage" => match find(p, "Declaration") {
                Some(Value::Object(d)) if d.export().is_some() => self.expr(d.export().unwrap(), 0),
                _ => v("0.0", 1),
            },
            "If" => {
                let a = cast(&self.input(e, "A").unwrap_or(v("0.0", 1)), 1);
                let b = cast(&self.input_or(e, "B", "ConstB", 0.0), 1);
                let gt = self.input(e, "AGreaterThanB").unwrap_or(v("0.0", 1));
                let eq = self.input(e, "AEqualsB");
                let lt = self.input(e, "ALessThanB").unwrap_or(v("0.0", 1));
                let n = gt.n.max(lt.n);
                let (gt, lt) = (cast(&gt, n), cast(&lt, n));
                let ge = match eq {
                    Some(q) => v(format!("select({}, {}, {} == {})", gt.code, cast(&q, n).code, a.code, b.code), n),
                    None => gt,
                };
                v(format!("select({}, {}, {} >= {})", lt.code, ge.code, a.code, b.code), n)
            }
            other => {
                let what = match other {
                    "MaterialFunctionCall" => {
                        let f = match find(p, "MaterialFunction") {
                            Some(Value::Object(i)) => split_object_path(&self.l.path(*i)).1,
                            _ => "?".into(),
                        };
                        format!("material function {f}")
                    }
                    o => o.to_string(),
                };
                self.note(format!("{what}: not translated (taken as 1)"));
                v("1.0", 1)
            }
        }
    }

    /// Binds a value to a `let` so its channels can be swizzled.
    fn expr_inline(&mut self, x: V) -> V {
        let name = format!("t{}", self.lines.len());
        self.lines.push(format!("    let {name}: {} = {};", ty(x.n), x.code));
        v(name, x.n)
    }

    /// A material input (`EmissiveColor`, `Opacity`, ...) of the editor data.
    fn material_input(&mut self, editor: usize, name: &str) -> Option<V> {
        let p = &self.l.pkg.exports[editor].object.properties;
        let inp = find(p, name)?.clone();
        if let Some(x) = self.link(&inp) {
            return Some(x);
        }
        if matches!(inp.field("UseConstant"), Some(Value::Bool(true))) {
            return match inp.field("Constant") {
                Some(Value::Float(x)) => Some(v(lit(*x), 1)),
                Some(c @ Value::Struct { .. }) if c.field("R").is_some() => {
                    let b = |n: &str| c.field(n).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32 / 255.0;
                    Some(v(format!("vec3<f32>({}, {}, {})", lit(b("R")), lit(b("G")), lit(b("B"))), 3))
                }
                _ => None,
            };
        }
        None
    }
}

/// Translates the material's graph to a WESL particle material.
pub fn compile(m: &Material, needs: &Needs) -> Result<Compiled, String> {
    let (l, editor) = m.base.as_ref().ok_or("base material not loaded")?;
    let mut g = Gen {
        l,
        m,
        needs,
        lines: Vec::new(),
        done: HashMap::new(),
        textures: Vec::new(),
        notes: BTreeMap::new(),
        dynamic_read: [false; 4],
        depth: 0,
    };
    let emissive = g.material_input(*editor, "EmissiveColor").map(|x| cast(&x, 3));
    // Unreal clamps Emissive at 0 (GetMaterialEmissive) unless the material
    // sets bAllowNegativeEmissiveColor; erosion materials rely on it (mask -
    // noise goes negative where the opacity is 0 too).
    // A UMaterial property (the editor-only data export in UE5 does not hold it).
    let allow_negative = l.pkg.exports.iter().any(|e| matches!(find(&e.object.properties, "bAllowNegativeEmissiveColor"), Some(Value::Bool(true))));
    let emissive = emissive.map(|mut x| {
        if !allow_negative {
            x.code = format!("max({}, vec3<f32>(0.0))", x.code);
        }
        x
    });
    let base = if m.unlit { None } else { g.material_input(*editor, "BaseColor").map(|x| cast(&x, 3)) };
    let alpha = match m.blend {
        Blend::Opaque => None,
        Blend::Masked => g.material_input(*editor, "OpacityMask").map(|x| cast(&x, 1)),
        _ => g.material_input(*editor, "Opacity").map(|x| cast(&x, 1)),
    };
    let wpo = {
        let p = &l.pkg.exports[*editor].object.properties;
        find(p, "WorldPositionOffset").and_then(|i| i.field("Expression")).is_some_and(|x| matches!(x, Value::Object(i) if !i.is_null()))
    };
    if wpo {
        g.note("World Position Offset (vertex animation): not carried over, the mesh does not deform");
    }
    if matches!(m.blend, Blend::Modulate) {
        g.note("Modulate blend: drawn alpha blended");
    }
    if !m.unlit {
        g.note("lit material: base colour lit by the scene's ambient and directional lights (no specular)");
    }
    let mut body = std::mem::take(&mut g.lines);
    let rgb = {
        let e = emissive.map(|x| x.code).unwrap_or("vec3<f32>(0.0)".into());
        match base {
            Some(b) => format!("{} * scene_light(normalize(in.normal), {}) + {e}", b.code, !needs.mesh),
            None if m.unlit => e,
            None => format!("vec3<f32>(0.5) * scene_light(normalize(in.normal), {}) + {e}", !needs.mesh),
        }
    };
    let a = match (m.blend, alpha) {
        (Blend::Masked, Some(x)) => {
            body.push(format!("    if ({} < {}) {{ discard; }}", x.code, lit(m.clip)));
            "1.0".to_string()
        }
        (Blend::Opaque | Blend::Masked, _) => "1.0".into(),
        (_, Some(x)) => format!("saturate({})", x.code),
        (_, None) => "1.0".into(),
    };
    let textures = g.textures.clone();
    let notes: Vec<String> = g.notes.keys().cloned().collect();
    let mut w = String::new();
    let _ = writeln!(w, "// Generated by `fximport convert` from the Unreal material {}", m.chain.join(" -> "));
    let _ = writeln!(w, "// (blend {:?}, {}). Edit the source material and convert again rather than this file.", m.blend, if m.unlit { "unlit" } else { "lit" });
    let _ = writeln!(w);
    let _ = writeln!(w, "import fx_render::particle_io::{{FxVertexOutput, finish, scene_light, fx, particle_texture, particle_sampler}};");
    let _ = writeln!(w, "import bevy_pbr::render::mesh_view_bindings::{{view, globals}};");
    let _ = writeln!(w);
    let _ = writeln!(w, "{}", HELPERS);
    let _ = writeln!(w, "@fragment");
    let _ = writeln!(w, "fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {{");
    for line in &body {
        let _ = writeln!(w, "{line}");
    }
    let _ = writeln!(w, "    return finish({rgb}, {a});");
    let _ = writeln!(w, "}}");
    Ok(Compiled {
        wesl: w,
        textures,
        notes,
        dynamic_read: g.dynamic_read,
        world_position_offset: wpo,
    })
}

/// Fills the atlas rectangles into a compiled material's `tex` helper.
pub fn with_rects(wesl: &str, rects: &[[f32; 4]]) -> String {
    let mut table = String::new();
    let _ = write!(table, "const TEX_RECTS = array<vec4<f32>, {}>(", rects.len().max(1));
    if rects.is_empty() {
        table.push_str("vec4<f32>(0.0, 0.0, 1.0, 1.0)");
    }
    for (i, r) in rects.iter().enumerate() {
        if i > 0 {
            table.push_str(", ");
        }
        let _ = write!(table, "vec4<f32>({}, {}, {}, {})", lit(r[0]), lit(r[1]), lit(r[2]), lit(r[3]));
    }
    table.push_str(");");
    wesl.replace("// TEX_RECTS", &table)
}

const HELPERS: &str = r#"// TEX_RECTS

// Texture `i` of the atlas at `uv`, wrapped (Unreal samplers repeat).
fn tex(i: u32, uv: vec2<f32>) -> vec4<f32> {
    let r = TEX_RECTS[i];
    return textureSampleLevel(particle_texture, particle_sampler, r.xy + fract(uv) * r.zw, 0.0);
}

fn ue_space(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, p.z, p.y);
}

// Unreal-space (centimetres, Z up) positions and normals for material math.
fn ue_world_position(in: FxVertexOutput) -> vec3<f32> {
    return ue_space(in.world_position) * 100.0;
}

fn ue_actor_position() -> vec3<f32> {
    return ue_space(fx.world_from_local[3].xyz) * 100.0;
}

fn ue_camera_position() -> vec3<f32> {
    return ue_space(view.world_position) * 100.0;
}

fn ue_normal(in: FxVertexOutput) -> vec3<f32> {
    return ue_space(normalize(in.normal));
}

fn rotate_uv(uv: vec2<f32>, c: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let k = cos(a);
    let d = uv - c;
    return c + vec2<f32>(d.x * k - d.y * s, d.x * s + d.y * k);
}
"#;

/// Stands in for a texture whose package is missing: black, so the
/// material terms it feeds drop out (erosion and distortion read 0), which
/// is how the reference recordings of the test systems look.
pub fn stand_in_missing() -> Image {
    Image { width: 4, height: 4, rgba: [0, 0, 0, 255].repeat(16), srgb: false }
}
