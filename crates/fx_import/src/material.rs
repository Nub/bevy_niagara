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

/// `base/rel` matching each component of `rel` without regard to case.
fn find_ci(base: &Path, rel: &str) -> Option<PathBuf> {
    let mut cur = base.to_path_buf();
    for part in rel.split('/').filter(|p| !p.is_empty()) {
        let exact = cur.join(part);
        if exact.exists() {
            cur = exact;
            continue;
        }
        let found = std::fs::read_dir(&cur)
            .ok()?
            .flatten()
            .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case(part))?;
        cur = found.path();
    }
    cur.is_file().then_some(cur)
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
        let candidates: Vec<(PathBuf, String)> = self
            .roots
            .iter()
            .map(|r| (r.clone(), format!("{rel}.uasset")))
            .chain(game.into_iter().flat_map(|(r, cs)| cs.iter().map(move |c| (c.clone(), format!("{r}.uasset")))))
            .collect();
        candidates
            .iter()
            .map(|(b, r)| b.join(r))
            .find(|f| f.is_file())
            // Unreal paths are case-insensitive; packs saved on Windows can
            // name a folder `materials` that their objects call `Materials`.
            .or_else(|| candidates.iter().find_map(|(b, r)| find_ci(b, r)))
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
/// UE 5.7's TSCF_UEDELTA source art (read off Mayu's Basic VFX Pack): the raw
/// pixels with every row stored as its byte-wise difference (mod 256) from
/// the row above, except the first row of each block of rows, stored as is; the
/// payload's own compression (Oodle) is undone by the bulk reader.
fn undelta_rows(data: &[u8], w: u32, h: u32, format: &str) -> Result<Vec<u8>, String> {
    // Bytes per pixel and bytes per channel.
    let (bpp, bpc) = match format {
        "TSF_G8" => (1, 1),
        "TSF_BGRA8" | "TSF_BGRE8" => (4, 1),
        "TSF_G16" => (2, 2),
        "TSF_RGBA16" | "TSF_RGBA16F" => (8, 2),
        other => return Err(format!("TSCF_UEDELTA with source format {other} not supported")),
    };
    let row = (w * bpp) as usize;
    if data.len() != row * h as usize {
        return Err(format!("TSCF_UEDELTA payload is {} bytes for {w}x{h} {format}", data.len()));
    }
    // Rows are delta-coded in blocks of 32768 pixels, at least 32 rows (as
    // seen: 128 rows at 256 px wide, 64 at 512, 32 at 1024 and 2048, for G8
    // and BGRA8 alike), twice that for 16-bit channels (64 rows for a
    // 2048-wide RGBA16 in Loot Drops VFX, UE 5.8); each block starts raw.
    // Deltas are byte-wise for 16-bit channels too.
    let block = (32768 / w as usize).max(32) * bpc;
    let mut out = data.to_vec();
    for y in 1..h as usize {
        if y % block == 0 {
            continue;
        }
        let (prev, cur) = out.split_at_mut(y * row);
        let prev = &prev[(y - 1) * row..];
        if bpc == 2 {
            // 16-bit channels: byte-wise deltas stored + 0x80 (a delta row
            // of a flat area reads 0x80 in every byte).
            for (c, p) in cur[..row].iter_mut().zip(prev) {
                *c = c.wrapping_sub(0x80).wrapping_add(*p);
            }
        } else {
            for (c, p) in cur[..row].iter_mut().zip(prev) {
                *c = c.wrapping_add(*p);
            }
        }
    }
    Ok(out)
}

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
        // Unreal PNG-compresses BGRA8 source art byte for byte: the file's
        // "RGB" is B, G, R (normal maps decode red, leaves cyan otherwise).
        "TSCF_PNG" if matches!(format.as_str(), "TSF_BGRA8" | "TSF_BGRE8") => {
            let mut rgba = decode_png(&data)?.rgba;
            rgba.chunks_mut(4).for_each(|p| p.swap(0, 2));
            rgba
        }
        "TSCF_PNG" => decode_png(&data)?.rgba,
        "TSCF_None" => raw_to_rgba(&data, w, h, &format)?,
        "TSCF_UEDELTA" => raw_to_rgba(&undelta_rows(&data, w, h, &format)?, w, h, &format)?,
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

fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb8(v: u8) -> u8 {
    let x = v as f32 / 255.0;
    let s = if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Textures packed side by side into one image (our renderers bind one
/// texture); `rects[i]` = (u, v, width, height) of texture i in UV units.
/// `levels[i]` = the rects of texture i's mip chain, full size first: each
/// texture carries its own box-filtered mips (to the right of the full-size
/// image, stacked), so materials filter like an Unreal sampler (trilinear,
/// wrapping per texture) without mips of the whole atlas bleeding across
/// textures. The atlas is loaded as sRGB, so linear textures are stored
/// sRGB-encoded and sample back as their original values.
pub struct Atlas {
    pub image: Image,
    pub rects: Vec<[f32; 4]>,
    pub levels: Vec<Vec<[f32; 4]>>,
}

const PAD: u32 = 2;

/// Mip levels of `img` (full size first), down to 1 x 1: 2 x 2 box filtered
/// in linear light (colour) or as stored (alpha), wrapping at odd edges.
fn mip_chain(img: &Image) -> Vec<Image> {
    let to_lin = |v: u8, c: usize| if c < 3 && img.srgb { srgb_to_linear(v as f32 / 255.0) } else { v as f32 / 255.0 };
    let from_lin = |v: f32, c: usize| {
        let v = if c < 3 && img.srgb {
            let x = v.clamp(0.0, 1.0);
            if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
        } else {
            v
        };
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let mut out = vec![img.clone()];
    let (mut w, mut h) = (img.width, img.height);
    let mut cur: Vec<f32> = img.rgba.iter().enumerate().map(|(i, &v)| to_lin(v, i % 4)).collect();
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0.0f32; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut s = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx) % w;
                        let sy = (y * 2 + dy) % h;
                        s += cur[((sy * w + sx) * 4) as usize + c];
                    }
                    next[((y * nw + x) * 4) as usize + c] = s * 0.25;
                }
            }
        }
        out.push(Image {
            width: nw,
            height: nh,
            rgba: next.iter().enumerate().map(|(i, &v)| from_lin(v, i % 4)).collect(),
            srgb: img.srgb,
        });
        cur = next;
        (w, h) = (nw, nh);
    }
    out
}

pub fn atlas(images: &[Image]) -> Atlas {
    let chains: Vec<Vec<Image>> = images.iter().map(mip_chain).collect();
    // Column of texture i: the full-size image, then its mips stacked to its right.
    let col = |c: &Vec<Image>| {
        let w0 = c[0].width + 2 * PAD;
        let w1 = c.get(1).map_or(0, |m| m.width + 2 * PAD);
        let h0 = c[0].height + 2 * PAD;
        let h1: u32 = c[1..].iter().map(|m| m.height + 2 * PAD).sum();
        (w0 + w1, h0.max(h1))
    };
    let w: u32 = chains.iter().map(|c| col(c).0).sum();
    let h: u32 = chains.iter().map(|c| col(c).1).max().unwrap_or(1);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut rects = Vec::new();
    let mut levels = Vec::new();
    let mut x0 = 0;
    for chain in &chains {
        let mut lv = Vec::new();
        let mut y1 = 0;
        for (k, img) in chain.iter().enumerate() {
            let (bx, by) = if k == 0 { (x0, 0) } else { (x0 + chain[0].width + 2 * PAD, y1) };
            if k > 0 {
                y1 += img.height + 2 * PAD;
            }
            // Edge texels repeat into the padding (clamped bilinear at the seams).
            for y in 0..img.height + 2 * PAD {
                let sy = y.saturating_sub(PAD).min(img.height - 1);
                for x in 0..img.width + 2 * PAD {
                    let sx = x.saturating_sub(PAD).min(img.width - 1);
                    let s = ((sy * img.width + sx) * 4) as usize;
                    let d = (((by + y) * w + bx + x) * 4) as usize;
                    for c in 0..4 {
                        let v = img.rgba[s + c];
                        rgba[d + c] = if c < 3 && !img.srgb { linear_to_srgb8(v) } else { v };
                    }
                }
            }
            lv.push([
                (bx + PAD) as f32 / w as f32,
                (by + PAD) as f32 / h as f32,
                img.width as f32 / w as f32,
                img.height as f32 / h as f32,
            ]);
        }
        rects.push(lv[0]);
        levels.push(lv);
        x0 += col(chain).0;
    }
    Atlas {
        image: Image { width: w, height: h, rgba, srgb: true },
        rects,
        levels,
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
    /// Static switch parameter overrides (instances' `StaticSwitchParameters`).
    pub switches: HashMap<String, bool>,
    /// The base material's package and its editor-only data export (the graph).
    pub base: Option<(Rc<Loaded>, usize)>,
    /// Parents that could not be loaded (the chain stops there).
    pub missing: Option<String>,
    /// Material function packages the graph calls (directly or nested),
    /// by package path (`/Game/A/MF_X`); engine functions are not on disk.
    pub functions: HashMap<String, Rc<Loaded>>,
    /// Parameters read from dynamic material parameters instead of baked:
    /// name -> (parameter index 0..4, first component). Scalars read one
    /// component, vectors the whole vector (component 0). Set by callers
    /// (prefab meshes whose parameters animate).
    pub bound: HashMap<String, (usize, usize)>,
}

/// Loads the material functions `l`'s graph calls, and theirs, into `out`.
fn load_functions(store: &mut Store, l: &Loaded, out: &mut HashMap<String, Rc<Loaded>>) {
    for i in 0..l.pkg.exports.len() {
        if l.pkg.export_class(i) != "MaterialExpressionMaterialFunctionCall" {
            continue;
        }
        let Some(Value::Object(f)) = find(l.props(i), "MaterialFunction") else { continue };
        if f.is_null() {
            continue;
        }
        let path = l.path(*f);
        let key = split_object_path(&path).0;
        if out.contains_key(&key) {
            continue;
        }
        if let Some(fl) = store.load(&path) {
            out.insert(key, fl.clone());
            load_functions(store, &fl, out);
        }
    }
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
            switches: HashMap::new(),
            base: None,
            missing: None,
            functions: HashMap::new(),
            bound: HashMap::new(),
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
                load_functions(store, &l, &mut m.functions);
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
            // Static switches: `StaticParameters` (UE4), `StaticParametersRuntime` (UE5).
            let mut switches = Vec::new();
            static_switches(&props, &mut switches);
            for (n, on) in switches {
                m.switches.entry(n).or_insert(on);
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

/// Overridden static switch parameters (name, value) found anywhere in a
/// material instance's properties (`StaticSwitchParameters` arrays).
fn static_switches(props: &[Property], out: &mut Vec<(String, bool)>) {
    fn walk(v: &Value, out: &mut Vec<(String, bool)>) {
        match v {
            Value::Struct { fields, .. } => static_switches(fields, out),
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    for p in props {
        if p.name == "StaticSwitchParameters" {
            for s in arr(Some(&p.value)) {
                if let (Some(n), Some(Value::Bool(true))) = (param_name(s), s.field("bOverride")) {
                    out.push((n, matches!(s.field("Value"), Some(Value::Bool(true)))));
                }
            }
        } else {
            walk(&p.value, out);
        }
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
    /// A pure distortion material: draws the scene behind it, shifted; its
    /// renderer must set `distortion: true` (fx_render gives it the scene).
    pub distortion: bool,
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

/// A graph being translated: the material's own (frame 0), or a material
/// function's, inlined for one call node of its parent frame.
struct Frame<'a> {
    l: &'a Loaded,
    /// (parent frame, the MaterialFunctionCall export in it).
    call: Option<(usize, usize)>,
}

/// Nested material function calls deeper than this are not inlined.
const MAX_FUNCTION_DEPTH: usize = 16;

struct Gen<'a> {
    /// The current frame's package.
    l: &'a Loaded,
    m: &'a Material,
    needs: &'a Needs,
    lines: Vec<String>,
    /// Emitted values by (frame, export * 8 + output).
    done: HashMap<(usize, usize), V>,
    textures: Vec<String>,
    notes: BTreeMap<String, ()>,
    dynamic_read: [bool; 4],
    depth: usize,
    frames: Vec<Frame<'a>>,
    cur: usize,
    /// Frame of each inlined call: (parent frame, call export) -> frame.
    call_frames: HashMap<(usize, usize), usize>,
    /// Module-level WGSL functions (translated Custom nodes), emitted before the fragment.
    functions: Vec<String>,
}

/// Unreal's component count for a function input type (None: not a float vector).
fn input_type_width(t: Option<&str>) -> Option<u8> {
    match t.unwrap_or("FunctionInput_Vector3") {
        "FunctionInput_Scalar" => Some(1),
        "FunctionInput_Vector2" => Some(2),
        "FunctionInput_Vector3" => Some(3),
        "FunctionInput_Vector4" => Some(4),
        _ => None,
    }
}

impl<'a> Gen<'a> {
    /// Runs `op` with frame `f` current.
    fn in_frame<T>(&mut self, f: usize, op: impl FnOnce(&mut Self) -> T) -> T {
        let (l, cur) = (self.l, self.cur);
        self.l = self.frames[f].l;
        self.cur = f;
        let r = op(self);
        self.l = l;
        self.cur = cur;
        r
    }

    /// How many function calls deep frame `f` is.
    fn frame_depth(&self, mut f: usize) -> usize {
        let mut n = 0;
        while let Some((p, _)) = self.frames[f].call {
            f = p;
            n += 1;
        }
        n
    }

    /// The entry of the current frame's call node feeding Function Input `e`
    /// (matched by input Id, else by name), with the parent frame.
    fn call_input_of(&self, e: usize) -> Option<(usize, &'a Value)> {
        let (parent, call) = self.frames[self.cur].call?;
        let p = self.props(e);
        let id = find(p, "Id");
        let name = find(p, "InputName").and_then(|n| n.as_str()).unwrap_or("In");
        let entries = arr(find(self.frames[parent].l.props(call), "FunctionInputs"));
        let entry = entries
            .iter()
            .find(|i| id.is_some() && i.field("ExpressionInputId") == id)
            .or_else(|| entries.iter().find(|i| i.field("Input").and_then(|x| x.field("InputName")).and_then(|n| n.as_str()) == Some(name)))?;
        let inp = entry.field("Input")?;
        matches!(inp.field("Expression"), Some(Value::Object(x)) if x.export().is_some()).then_some((parent, inp))
    }

    /// A Function Input's value: the call's connected input, else the
    /// input's preview (connected Preview, else PreviewValue), as its type.
    fn function_input(&mut self, e: usize) -> V {
        let p = self.props(e);
        let width = input_type_width(enum_tail(find(p, "InputType")).as_deref());
        let val = match self.call_input_of(e) {
            Some((parent, inp)) => self.in_frame(parent, |g| g.link(inp)),
            None => None,
        };
        let val = match val {
            Some(x) => x,
            None => {
                let name = find(p, "InputName").and_then(|n| n.as_str()).unwrap_or("In").to_string();
                if self.frames[self.cur].call.is_some() && !matches!(find(p, "bUsePreviewValueAsDefault"), Some(Value::Bool(true))) {
                    self.note(format!("material function input {name}: required but not connected (its preview value is used)"));
                }
                match self.input(e, "Preview") {
                    Some(x) => x,
                    None => {
                        let c = find(p, "PreviewValue");
                        let c = |n: &str| c.and_then(|c| c.field(n)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                        let c = [c("X"), c("Y"), c("Z"), c("W")];
                        match width.unwrap_or(1) {
                            1 => v(lit(c[0]), 1),
                            n => v(format!("{}({})", ty(n), c[..n as usize].iter().map(|x| lit(*x)).collect::<Vec<_>>().join(", ")), n),
                        }
                    }
                }
            }
        };
        match width {
            Some(n) => cast(&val, n),
            None => val,
        }
    }

    /// A call's argument by input name (connected inputs only).
    fn call_arg(&mut self, e: usize, name: &str) -> Option<V> {
        let inp = arr(find(self.props(e), "FunctionInputs"))
            .iter()
            .filter_map(|i| i.field("Input"))
            .find(|i| i.field("InputName").and_then(|n| n.as_str()) == Some(name))?;
        self.link(inp)
    }

    /// A call's static bool argument by input name (None when unconnected).
    fn call_static_bool(&mut self, e: usize, name: &str) -> Option<bool> {
        let inp = arr(find(self.props(e), "FunctionInputs"))
            .iter()
            .filter_map(|i| i.field("Input"))
            .find(|i| i.field("InputName").and_then(|n| n.as_str()) == Some(name))?;
        match inp.field("Expression") {
            Some(Value::Object(x)) => Some(self.static_bool(x.export()?)),
            _ => None,
        }
    }

    /// Output `out` of a Material Function Call: the function's graph
    /// inlined when its package is loaded, else a built-in engine function.
    fn function_call(&mut self, e: usize, out: usize) -> V {
        let p = self.props(e);
        let path = match find(p, "MaterialFunction") {
            Some(Value::Object(i)) if !i.is_null() => self.l.path(*i),
            _ => String::new(),
        };
        let (pkg, fname) = split_object_path(&path);
        let fname = if fname.is_empty() { "?".to_string() } else { fname };
        let outputs = arr(find(p, "FunctionOutputs"));
        let out_id = outputs.get(out).and_then(|o| o.field("ExpressionOutputId"));
        let out_name = outputs.get(out).and_then(|o| o.field("Output")).and_then(|o| o.field("OutputName")).and_then(|n| n.as_str()).unwrap_or("").to_string();
        if let Some(fl) = self.m.functions.get(&pkg) {
            let fl: &'a Loaded = fl;
            if self.frame_depth(self.cur) >= MAX_FUNCTION_DEPTH {
                self.note(format!("material function {fname}: nested too deep (taken as 1)"));
                return v("1.0", 1);
            }
            let outs: Vec<usize> = (0..fl.pkg.exports.len()).filter(|&i| fl.pkg.export_class(i) == "MaterialExpressionFunctionOutput").collect();
            let oname = |o: usize| find(fl.props(o), "OutputName").and_then(|n| n.as_str()).unwrap_or("Result");
            let pick = outs
                .iter()
                .copied()
                .find(|&o| out_id.is_some() && find(fl.props(o), "Id") == out_id)
                .or_else(|| outs.iter().copied().find(|&o| !out_name.is_empty() && oname(o) == out_name))
                .or_else(|| {
                    // Unreal's order: SortPriority, then the export order.
                    let mut s = outs.clone();
                    s.sort_by_key(|&o| (find(fl.props(o), "SortPriority").and_then(|x| x.as_f64()).unwrap_or(0.0) as i64, o));
                    s.get(out).copied()
                });
            let Some(pick) = pick else {
                self.note(format!("material function {fname}: output {out} not found (taken as 1)"));
                return v("1.0", 1);
            };
            let frame = match self.call_frames.get(&(self.cur, e)) {
                Some(&f) => f,
                None => {
                    self.frames.push(Frame { l: fl, call: Some((self.cur, e)) });
                    let f = self.frames.len() - 1;
                    self.call_frames.insert((self.cur, e), f);
                    f
                }
            };
            return match self.in_frame(frame, |g| g.input(pick, "A")) {
                Some(x) => x,
                None => {
                    self.note(format!("material function {fname}: output {} is not connected (taken as 0)", oname(pick)));
                    v("0.0", 1)
                }
            };
        }
        match self.builtin_function(e, &fname, &out_name) {
            Some(x) => x,
            None => {
                let why = if pkg.starts_with("/Engine/") { "not translated" } else { "package not found, not translated" };
                self.note(format!("material function {fname}: {why} (taken as 1)"));
                v("1.0", 1)
            }
        }
    }

    /// Engine material functions (not on disk), by their documented graphs.
    fn builtin_function(&mut self, e: usize, name: &str, out_name: &str) -> Option<V> {
        let arg = |g: &mut Self, n: &str, d: V, width: u8| cast(&g.call_arg(e, n).unwrap_or(d), width);
        Some(match name {
            // Rotates UVs about the centre by angle (0-1) turns.
            "CustomRotator" => {
                let uv = arg(self, "UVs", v("in.uv0", 2), 2);
                let c = arg(self, "Rotation Center", v("vec2<f32>(0.5, 0.5)", 2), 2);
                let a = arg(self, "Rotation Angle (0-1)", v("0.0", 1), 1);
                v(format!("rotate_uv({}, {}, {} * {})", uv.code, c.code, a.code, lit(std::f32::consts::TAU)), 2)
            }
            // 1 at the centre falling to 0 at Radius, with ExponentialDensity
            // (1 - exp(-depth * density)) shaping the edge.
            "RadialGradientExponential" => {
                let uv = arg(self, "UVs", v("in.uv0", 2), 2);
                let c = arg(self, "CenterPosition", v("vec2<f32>(0.5, 0.5)", 2), 2);
                let r = arg(self, "Radius", v("0.5", 1), 1);
                let d = arg(self, "Density", v("2.25", 1), 1);
                let invert = self.call_static_bool(e, "Invert Density").unwrap_or(false);
                let g = format!("(1.0 - exp(-saturate(1.0 - distance({}, {}) / ({})) * ({})))", uv.code, c.code, r.code, d.code);
                v(if invert { format!("(1.0 - {g})") } else { g }, 1)
            }
            "CheapContrast" => {
                let x = arg(self, "In", v("0.0", 1), 1);
                let k = arg(self, "Contrast", v("0.0", 1), 1);
                v(format!("saturate(mix(0.0 - ({0}), 1.0 + ({0}), {1}))", k.code, x.code), 1)
            }
            // Temporal dithering of a masked opacity: the threshold as is
            // (no dither pattern; TAA blends it to about this).
            "DitherTemporalAA" => {
                let a = ["Alpha Threshold", "AlphaThreshold", "Alpha"].iter().find_map(|n| self.call_arg(e, n));
                cast(&a.unwrap_or(v("0.5", 1)), 1)
            }
            "MakeFloat2" | "MakeFloat3" | "MakeFloat4" => {
                let n = name.as_bytes()[9] - b'0';
                let parts: Vec<String> = ["X", "Y", "Z", "W"][..n as usize].iter().map(|c| arg(self, c, v("0.0", 1), 1).code).collect();
                v(format!("{}({})", ty(n), parts.join(", ")), n)
            }
            "BreakOutFloat2Components" | "BreakOutFloat3Components" | "BreakOutFloat4Components" => {
                let n = name.as_bytes()[13] - b'0';
                let x = arg(self, &format!("Float{n}"), v("0.0", 1), n);
                let k = ["R", "G", "B", "A"].iter().position(|c| *c == out_name)?;
                let x = self.expr_inline(x);
                v(format!("{}.{}", x.code, &"xyzw"[k..k + 1]), 1)
            }
            _ => return None,
        })
    }
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

    /// The particle field a bound parameter reads (`in.dynamic_param_k`),
    /// when bound and its dynamic parameter reaches the material.
    fn bound_field(&mut self, name: &str) -> Option<String> {
        let &(k, _) = self.m.bound.get(name)?;
        let k = k.min(3);
        if !(self.needs.dynamic_written[k] && self.needs.dynamic_available[k]) {
            self.note(format!("parameter {name}: bound to dynamic parameter {k}, which does not reach the material; baked"));
            return None;
        }
        self.dynamic_read[k] = true;
        Some(if k == 0 { "in.dynamic_param".to_string() } else { format!("in.dynamic_param_{k}") })
    }

    /// Expression `e`'s value (output `out` for multi-output nodes without a
    /// mask), emitted once as a `let`.
    fn expr(&mut self, e: usize, out: usize) -> V {
        let key = (self.cur, e * 8 + out.min(7));
        if let Some(x) = self.done.get(&key) {
            return x.clone();
        }
        self.depth += 1;
        let val = if self.depth > 200 { v("0.0", 1) } else { self.eval(e, out) };
        self.depth -= 1;
        let name = if self.cur == 0 { format!("e{e}_{out}") } else { format!("f{}_e{e}_{out}", self.cur) };
        let simple = val.code.parse::<f32>().is_ok() || val.code.starts_with("in.") && !val.code.contains(' ');
        // Another value's name (a reroute, a function input or output) is not bound again.
        let simple = simple || val.code.starts_with(|c: char| c.is_ascii_alphabetic()) && val.code.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let res = if simple {
            val
        } else {
            self.lines.push(format!("    let {name}: {} = {};", ty(val.n), val.code));
            v(name, val.n)
        };
        self.done.insert(key, res.clone());
        res
    }

    /// A static switch's or static bool's value: an instance's override of
    /// the parameter, the bool wired into a plain Static Switch's `Value`,
    /// else the node's default.
    fn static_bool(&mut self, e: usize) -> bool {
        let class = self.l.pkg.export_class(e);
        let p = self.props(e);
        let default = matches!(find(p, "DefaultValue"), Some(Value::Bool(true)));
        match class.strip_prefix("MaterialExpression").unwrap_or(&class) {
            "StaticBool" => matches!(find(p, "Value"), Some(Value::Bool(true))),
            "StaticBoolParameter" | "StaticSwitchParameter" => {
                let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("");
                self.m.switches.get(name).copied().unwrap_or(default)
            }
            "StaticSwitch" => match find(p, "Value").and_then(|i| i.field("Expression")) {
                Some(Value::Object(x)) if x.export().is_some() && self.depth < 200 => {
                    self.depth += 1;
                    let r = self.static_bool(x.export().unwrap());
                    self.depth -= 1;
                    r
                }
                _ => default,
            },
            "Reroute" | "NamedRerouteDeclaration" => match find(p, "Input").and_then(|i| i.field("Expression")) {
                Some(Value::Object(x)) if x.export().is_some() && self.depth < 200 => {
                    self.depth += 1;
                    let r = self.static_bool(x.export().unwrap());
                    self.depth -= 1;
                    r
                }
                _ => false,
            },
            "FunctionInput" => {
                if self.depth >= 200 {
                    return false;
                }
                self.depth += 1;
                let r = match self.call_input_of(e) {
                    Some((parent, inp)) => match inp.field("Expression") {
                        Some(Value::Object(x)) => self.in_frame(parent, |g| g.static_bool(x.export().unwrap())),
                        _ => false,
                    },
                    None => match find(p, "Preview").and_then(|i| i.field("Expression")) {
                        Some(Value::Object(x)) if x.export().is_some() => self.static_bool(x.export().unwrap()),
                        _ => find(p, "PreviewValue").and_then(|c| c.field("X")).and_then(|x| x.as_f64()).unwrap_or(0.0) != 0.0,
                    },
                };
                self.depth -= 1;
                r
            }
            _ => default,
        }
    }

    /// The texture an expression samples: its own, an instance's override of
    /// its parameter, or that of a texture object (parameter) wired into it.
    fn texture_of(&mut self, e: usize) -> Option<String> {
        let class = self.l.pkg.export_class(e);
        let p = self.props(e);
        if class == "MaterialExpressionFunctionInput" && self.depth < 200 {
            self.depth += 1;
            let r = match self.call_input_of(e) {
                Some((parent, inp)) => match inp.field("Expression") {
                    Some(Value::Object(x)) => self.in_frame(parent, |g| g.texture_of(x.export().unwrap())),
                    _ => None,
                },
                None => match find(p, "Preview").and_then(|i| i.field("Expression")) {
                    Some(Value::Object(x)) if x.export().is_some() => self.texture_of(x.export().unwrap()),
                    _ => None,
                },
            };
            self.depth -= 1;
            return r;
        }
        if let Some(obj) = find(p, "TextureObject").and_then(|i| match i.field("Expression") {
            Some(Value::Object(x)) => x.export(),
            _ => None,
        }) {
            if obj != e {
                if let Some(t) = self.texture_of(obj) {
                    return Some(t);
                }
            }
        }
        if class.ends_with("Parameter2D") || class.ends_with("TextureObjectParameter") || class.ends_with("ParameterSubUV") {
            let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(t) = self.m.textures.get(name) {
                return Some(t.clone());
            }
        }
        match find(p, "Texture") {
            Some(Value::Object(t)) if !t.is_null() => Some(self.l.path(*t)),
            _ => None,
        }
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
                if let Some(field) = self.bound_field(&name) {
                    let c = self.m.bound[&name].1.min(3);
                    return v(format!("{field}.{}", &"xyzw"[c..c + 1]), 1);
                }
                let x = self.m.scalars.get(&name).copied().unwrap_or(fprop("DefaultValue", 0.0));
                v(lit(x), 1)
            }
            "VectorParameter" => {
                let name = find(p, "ParameterName").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let c = self.m.vectors.get(&name).copied().or_else(|| color_prop("DefaultValue")).unwrap_or([0.0; 4]);
                if let Some(field) = self.bound_field(&name) {
                    return match out {
                        1..=4 => v(format!("{field}.{}", &"xyzw"[out - 1..out]), 1),
                        _ => v(field, 4),
                    };
                }
                let full = v(format!("vec4<f32>({}, {}, {}, {})", lit(c[0]), lit(c[1]), lit(c[2]), lit(c[3])), 4);
                match out {
                    1..=4 => v(lit(c[out - 1]), 1),
                    _ => full,
                }
            }
            "StaticBoolParameter" | "StaticSwitchParameter" | "StaticSwitch" => {
                let on = self.static_bool(e);
                if class == "StaticBoolParameter" {
                    return v(if on { "1.0" } else { "0.0" }, 1);
                }
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
                // The period applies only with bOverride_Period (0 then reads 0).
                let period = fprop("Period", 0.0);
                match find(p, "bOverride_Period") {
                    Some(Value::Bool(true)) if period > 0.0 => v(format!("(globals.time % {})", lit(period)), 1),
                    Some(Value::Bool(true)) => v("0.0", 1),
                    _ => v("globals.time", 1),
                }
            }
            "Panner" => {
                let coord = self.input(e, "Coordinate").unwrap_or(v("in.uv0", 2));
                let time = self.input(e, "Time").unwrap_or(v("globals.time", 1));
                let speed = match self.input(e, "Speed") {
                    Some(s) => cast(&s, 2),
                    None => v(format!("vec2<f32>({}, {})", lit(fprop("SpeedX", 0.0)), lit(fprop("SpeedY", 0.0))), 2),
                };
                // Unreal takes the fractional part of the offset only.
                let mut offset = format!("{} * {}", speed.code, cast(&time, 1).code);
                if matches!(find(p, "bFractionalPart"), Some(Value::Bool(true))) {
                    offset = format!("fract({offset})");
                }
                v(format!("{} + {offset}", cast(&coord, 2).code), 2)
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
            "TextureSample" | "TextureSampleParameter2D" | "TextureObject" | "TextureObjectParameter" | "ParticleSubUV" | "TextureSampleParameterSubUV" => {
                // A texture object (parameter) wired into the sampler wins.
                let tex = self.texture_of(e);
                if class.starts_with("TextureObject") {
                    return v("0.0", 1); // only meaningful through a sampler
                }
                // Sampled once for all its outputs (kept under slot 7).
                if let Some(full) = self.done.get(&(self.cur, e * 8 + 7)).cloned() {
                    return match out {
                        0 => v(format!("{}.xyz", full.code), 3),
                        1..=4 => v(format!("{}.{}", full.code, &"xyzw"[out - 1..out]), 1),
                        _ => full,
                    };
                }
                let slot = tex.map(|t| self.texture_slot(t));
                let full = if slot.is_none() {
                    self.note("a texture sampler without a texture samples white");
                    v("vec4<f32>(1.0)", 4)
                } else if class == "ParticleSubUV" || class == "TextureSampleParameterSubUV" {
                    let slot = slot.unwrap();
                    let a = self.sample(slot, &v("in.uv0", 2));
                    let b = self.sample(slot, &v("in.uv1", 2));
                    if matches!(find(p, "bBlend"), Some(Value::Bool(true))) {
                        v(format!("mix({}, {}, in.sub_uv_blend)", a.code, b.code), 4)
                    } else {
                        a
                    }
                } else {
                    let uv = self.input(e, "Coordinates").unwrap_or(v("in.uv0", 2));
                    self.sample(slot.unwrap(), &uv)
                };
                // The sampler type's lookup post-processing (grayscale
                // textures are single channel on the GPU: R, alpha 1; normal
                // maps unpack to -1..1 with Z rebuilt).
                let full = match enum_tail(find(p, "SamplerType")).as_deref() {
                    Some("SAMPLERTYPE_Grayscale" | "SAMPLERTYPE_LinearGrayscale") => {
                        let t = self.expr_inline(full);
                        v(format!("vec4<f32>({0}.x, {0}.x, {0}.x, 1.0)", t.code), 4)
                    }
                    Some("SAMPLERTYPE_Normal") => {
                        let t = self.expr_inline(full);
                        let xy = format!("({}.xy * 2.0 - 1.0)", t.code);
                        v(format!("vec4<f32>({xy}, sqrt(saturate(1.0 - dot({xy}, {xy}))), 1.0)"), 4)
                    }
                    _ => full,
                };
                let full = self.expr_inline(full);
                self.done.insert((self.cur, e * 8 + 7), full.clone());
                // Outputs: RGB, R, G, B, A, RGBA.
                match out {
                    0 => v(format!("{}.xyz", full.code), 3),
                    1 => v(format!("{}.x", full.code), 1),
                    2 => v(format!("{}.y", full.code), 1),
                    3 => v(format!("{}.z", full.code), 1),
                    4 => v(format!("{}.w", full.code), 1),
                    _ => full,
                }
            }
            "Multiply" | "Add" | "Subtract" | "Divide" | "Max" | "Min" => {
                let (ca, cb) = match class.as_str() {
                    "Subtract" => (1.0, 1.0),
                    "Divide" => (1.0, 2.0),
                    // Multiply, Add, Max, Min.
                    _ => (0.0, 1.0),
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
                // Unreal broadcasts all three to the widest (a scalar A and B
                // with a vector Alpha give a vector).
                let (a, b, n) = pair(&a, &b);
                let (a, t, n) = if t.n == 1 { (a, t, n) } else { pair(&a, &t) };
                let b = cast(&b, n);
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
            // Temporal dithering of a masked opacity: the threshold as is
            // (no dither pattern; TAA would blend it to about this).
            "DitherTemporalAA" => cast(&self.input(e, "AlphaThreshold").unwrap_or(v("0.5", 1)), 1),
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
                let n = gt.n.max(lt.n).max(eq.as_ref().map_or(1, |q| q.n));
                let (gt, lt) = (cast(&gt, n), cast(&lt, n));
                let order = v(format!("select({}, {}, {} >= {})", lt.code, gt.code, a.code, b.code), n);
                // Within EqualsThreshold of each other A and B count as equal.
                match eq {
                    Some(q) => {
                        let thr = lit(fprop("EqualsThreshold", 0.00001));
                        v(format!("select({}, {}, abs({} - {}) <= {thr})", order.code, cast(&q, n).code, a.code, b.code), n)
                    }
                    None => order,
                }
            }
            "Custom" => self.custom(e, out),
            "MaterialFunctionCall" => self.function_call(e, out),
            "FunctionInput" => self.function_input(e),
            "FunctionOutput" => self.input(e, "A").unwrap_or(v("0.0", 1)),
            other => {
                self.note(format!("{other}: not translated (taken as 1)"));
                v("1.0", 1)
            }
        }
    }

    /// A Custom expression: its HLSL translated to a WGSL function
    /// (`custom_<export>`) called with the node's inputs; untranslatable
    /// code is taken as 1, as before.
    fn custom(&mut self, e: usize, out: usize) -> V {
        let p = self.props(e);
        let desc = find(p, "Description").and_then(|v| v.as_str()).unwrap_or("Custom").to_string();
        let code = find(p, "Code").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let n = match find(p, "OutputType").and_then(|v| v.as_str()) {
            Some(s) if s.ends_with("CMOT_Float1") => 1,
            Some(s) if s.ends_with("CMOT_Float2") => 2,
            Some(s) if s.ends_with("CMOT_Float4") => 4,
            Some(s) if !s.ends_with("CMOT_Float3") => 0,
            // CMOT_Float3 is the default (not saved).
            _ => 3,
        };
        let fallback = |g: &mut Self, why: String| {
            g.note(format!("Custom node `{desc}`: not translated ({why}); taken as 1"));
            v("1.0", 1)
        };
        if n == 0 {
            return fallback(self, "output type not supported".into());
        }
        if !arr(find(p, "AdditionalOutputs")).is_empty() || out > 0 {
            return fallback(self, "additional outputs are not supported".into());
        }
        if !arr(find(p, "IncludeFilePaths")).is_empty() || !arr(find(p, "AdditionalDefines")).is_empty() {
            return fallback(self, "include files and defines are not supported".into());
        }
        let mut names = Vec::new();
        let mut args = Vec::new();
        for ci in arr(find(p, "Inputs")) {
            let name = ci.field("InputName").and_then(|v| v.as_str()).unwrap_or("");
            // Unreal skips inputs without a name.
            if name.is_empty() || name == "None" {
                continue;
            }
            let x = ci.field("Input").and_then(|inp| self.link(inp)).unwrap_or(v("0.0", 1));
            names.push((name.to_string(), x.n as usize));
            args.push(x.code);
        }
        let fn_name = if self.cur == 0 { format!("custom_{e}") } else { format!("custom_f{}_{e}", self.cur) };
        match crate::hlsl_material::translate_custom(&code, &names, n as usize, &fn_name) {
            Ok(f) => {
                self.functions.push(f);
                v(format!("{fn_name}({})", args.join(", ")), n)
            }
            Err(why) => fallback(self, why),
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
                // UE 5.7: an FLinearColor, already linear.
                Some(c @ Value::Struct { name, .. }) if name == "LinearColor" => {
                    let b = |n: &str| c.field(n).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    Some(v(format!("vec3<f32>({}, {}, {})", lit(b("R")), lit(b("G")), lit(b("B"))), 3))
                }
                Some(c @ Value::Struct { .. }) if c.field("R").is_some() => {
                    // An FColor: sRGB-encoded (Unreal linearizes it).
                    let b = |n: &str| srgb_to_linear(c.field(n).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32 / 255.0);
                    Some(v(format!("vec3<f32>({}, {}, {})", lit(b("R")), lit(b("G")), lit(b("B"))), 3))
                }
                _ => None,
            };
        }
        None
    }
}

/// Translates the material's graph to a WESL particle material.
/// A scalar or vector parameter of the base material's graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamInfo {
    pub name: String,
    pub vector: bool,
    /// Custom primitive data index when the parameter reads it.
    pub primitive_data: Option<u32>,
    /// The value with the instance chain's overrides applied.
    pub value: [f32; 4],
}

/// The scalar and vector parameters of `m`'s base graph (not its functions).
pub fn parameters(m: &Material) -> Vec<ParamInfo> {
    let Some((l, _)) = m.base.as_ref() else { return Vec::new() };
    let mut out: Vec<ParamInfo> = Vec::new();
    for (i, e) in l.pkg.exports.iter().enumerate() {
        let class = l.pkg.export_class(i);
        let vector = match class.as_str() {
            "MaterialExpressionScalarParameter" => false,
            "MaterialExpressionVectorParameter" => true,
            _ => continue,
        };
        let p = &e.object.properties;
        let Some(name) = find(p, "ParameterName").and_then(|v| v.as_str()).map(String::from) else { continue };
        if out.iter().any(|x| x.name == name) {
            continue;
        }
        let cpd = matches!(find(p, "bUseCustomPrimitiveData"), Some(Value::Bool(true)));
        let index = find(p, "PrimitiveDataIndex").and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;
        let value = if vector {
            m.vectors.get(&name).copied().or_else(|| find(p, "DefaultValue").and_then(color4)).unwrap_or([0.0; 4])
        } else {
            let x = m.scalars.get(&name).copied().unwrap_or(find(p, "DefaultValue").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32);
            [x, 0.0, 0.0, 0.0]
        };
        out.push(ParamInfo { name, vector, primitive_data: cpd.then_some(index), value });
    }
    out
}

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
        frames: vec![Frame { l, call: None }],
        cur: 0,
        call_frames: HashMap::new(),
        functions: Vec::new(),
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
    // A pure distortion material (Refraction wired, no colour): drawn as a
    // shifted copy of the scene behind it (the renderer's `distortion`).
    let distortion = if emissive.is_none() && base.is_none() {
        g.material_input(*editor, "Refraction").map(|r| {
            let r = cast(&r, 1);
            let n = g.material_input(*editor, "Normal").map(|x| cast(&x, 3)).map(|x| x.code).unwrap_or("vec3<f32>(0.0, 0.0, 1.0)".into());
            (r.code, n)
        })
    } else {
        None
    };
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
            // Unreal's default Base Color is black: only the emissive shows.
            None => e,
        }
    };
    // Unreal (index of refraction): the screen offset follows the
    // tangent-space normal times IOR - 1 (V runs down the screen).
    let rgb = match &distortion {
        Some((ior, n)) => format!("scene_behind(in, ({n}).xy * vec2<f32>(1.0, -1.0) * (({ior}) - 1.0) * DISTORTION_SCALE)"),
        None => rgb,
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
    if distortion.is_some() {
        let _ = writeln!(w, "import bevy_pbr::render::mesh_view_bindings::{{view_transmission_texture, linear_sampler}};");
        let _ = writeln!(w);
        let _ = writeln!(w, "{}", DISTORTION_HELPERS);
    }
    let _ = writeln!(w);
    let _ = writeln!(w, "{}", HELPERS);
    for f in &g.functions {
        let _ = writeln!(w, "{f}");
    }
    let _ = writeln!(w, "@fragment");
    let _ = writeln!(w, "fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {{");
    for line in &body {
        let _ = writeln!(w, "{line}");
    }
    let _ = writeln!(w, "    return finish({rgb}, {a});");
    let _ = writeln!(w, "}}");
    Ok(Compiled {
        wesl: w,
        distortion: distortion.is_some(),
        textures,
        notes,
        dynamic_read: g.dynamic_read,
        world_position_offset: wpo,
    })
}

/// Fills the atlas rectangles (one level each) into a compiled material's
/// `tex` helper.
pub fn with_rects(wesl: &str, rects: &[[f32; 4]]) -> String {
    with_levels(wesl, &rects.iter().map(|r| vec![*r]).collect::<Vec<_>>())
}

/// Fills an atlas's textures and their mip levels (`Atlas::levels`) into a
/// compiled material's `tex` helper.
pub fn with_atlas(wesl: &str, at: &Atlas) -> String {
    with_levels(wesl, &at.levels)
}

fn with_levels(wesl: &str, levels: &[Vec<[f32; 4]>]) -> String {
    let rects: Vec<[f32; 4]> = levels.iter().flatten().copied().collect();
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
    table.push_str(");\n");
    // First rect and level count of each texture.
    let n = levels.len().max(1);
    let mut first = Vec::new();
    let mut at = 0;
    for l in levels {
        first.push(format!("{at}u"));
        at += l.len();
    }
    let counts: Vec<String> = levels.iter().map(|l| format!("{}u", l.len().max(1))).collect();
    let or_one = |v: Vec<String>, d: &str| if v.is_empty() { d.to_string() } else { v.join(", ") };
    let _ = write!(table, "const TEX_FIRST = array<u32, {n}>({});\n", or_one(first, "0u"));
    let _ = write!(table, "const TEX_LEVELS = array<u32, {n}>({});", or_one(counts, "1u"));
    wesl.replace("// TEX_RECTS", &table)
}

/// For distortion materials: the opaque scene behind the fragment, shifted by
/// `offset` (screen UV), from Bevy's transmission copy.
const DISTORTION_HELPERS: &str = r#"// Screen offset per unit of (IOR - 1); matched by eye to Unreal's heat haze.
const DISTORTION_SCALE: f32 = 0.9;

fn scene_behind(in: FxVertexOutput, offset: vec2<f32>) -> vec3<f32> {
    let uv = (in.position.xy - view.viewport.xy) / view.viewport.zw;
    return textureSampleLevel(view_transmission_texture, linear_sampler, clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}
"#;

const HELPERS: &str = r#"// TEX_RECTS

// Texture `i` of the atlas at `uv`, filtered like an Unreal sampler:
// trilinear over the texture's own mip levels (picked from the screen
// derivatives of `uv`), bilinear within a level, wrapping at the texture's
// own edges (a hardware sample of the atlas would clamp into the padding
// there: uv 0 would read the corner texel instead of the mean of the four
// wrapped corners).
fn tex(i: u32, uv: vec2<f32>) -> vec4<f32> {
    let first = TEX_FIRST[i];
    let levels = TEX_LEVELS[i];
    let dims = vec2<f32>(textureDimensions(particle_texture, 0));
    let size = max(round(TEX_RECTS[first].zw * dims), vec2<f32>(1.0));
    let dx = dpdx(uv) * size;
    let dy = dpdy(uv) * size;
    let lod = clamp(0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-8)), 0.0, f32(levels - 1u));
    let l0 = u32(floor(lod));
    let l1 = min(l0 + 1u, levels - 1u);
    return mix(tex_level(first + l0, uv), tex_level(first + l1, uv), lod - f32(l0));
}

// One level (atlas rect `k`) of a texture, bilinear and wrapping.
fn tex_level(k: u32, uv: vec2<f32>) -> vec4<f32> {
    let r = TEX_RECTS[k];
    let dims = vec2<f32>(textureDimensions(particle_texture, 0));
    let size = max(round(r.zw * dims), vec2<f32>(1.0));
    let origin = vec2<i32>(round(r.xy * dims));
    let p = fract(uv) * size - 0.5;
    let f = fract(p);
    let a = (floor(p) + size) % size;
    let b = (a + 1.0) % size;
    let ia = origin + vec2<i32>(a);
    let ib = origin + vec2<i32>(b);
    let t00 = textureLoad(particle_texture, ia, 0);
    let t10 = textureLoad(particle_texture, vec2<i32>(ib.x, ia.y), 0);
    let t01 = textureLoad(particle_texture, vec2<i32>(ia.x, ib.y), 0);
    let t11 = textureLoad(particle_texture, ib, 0);
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
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
/// A generated stand-in for an engine texture packs use without shipping
/// (engine content is not in Fab downloads), with what it is. Only noise
/// textures, whose exact pixels do not matter, qualify.
pub fn engine_stand_in(object_path: &str) -> Option<(Image, &'static str)> {
    let (_, name) = split_object_path(object_path);
    if !object_path.starts_with("/Engine/") {
        return None;
    }
    match name.as_str() {
        // Engine_MaterialFunctions02/ExampleContent/Textures: 128x128,
        // independent soft noise in each channel, tiling.
        "LowResBlurredNoise" => Some((value_noise(128, 8), "tiling soft value noise")),
        _ => None,
    }
}

/// Tiling value noise: a `cells` x `cells` lattice of random values per
/// channel, smoothstep-interpolated, alpha 1.
fn value_noise(size: u32, cells: u32) -> Image {
    let hash = |x: u32, y: u32, c: u32| -> f32 {
        let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ c.wrapping_mul(0xcb1a_b31f) ^ 0x2545_f491;
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0
    };
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    for py in 0..size {
        for px in 0..size {
            let fx = px as f32 * cells as f32 / size as f32;
            let fy = py as f32 * cells as f32 / size as f32;
            let (x0, y0) = (fx as u32, fy as u32);
            let (tx, ty) = (s(fx - x0 as f32), s(fy - y0 as f32));
            let (x1, y1) = ((x0 + 1) % cells, (y0 + 1) % cells);
            for c in 0..3 {
                let a = hash(x0, y0, c) + (hash(x1, y0, c) - hash(x0, y0, c)) * tx;
                let b = hash(x0, y1, c) + (hash(x1, y1, c) - hash(x0, y1, c)) * tx;
                rgba.push(((a + (b - a) * ty) * 255.0).round() as u8);
            }
            rgba.push(255);
        }
    }
    Image { width: size, height: size, rgba, srgb: false }
}

pub fn stand_in_missing() -> Image {
    Image { width: 4, height: 4, rgba: [0, 0, 0, 255].repeat(16), srgb: false }
}

#[cfg(test)]
mod tests {
    //! The graph translator on small hand-built graphs (export 0 is the
    //! material, holding the material inputs; node `k` is export `k`).
    use super::*;
    use crate::package::{Export, Import, Object, Package};
    use crate::property::TypeName;

    fn prop(name: &str, value: Value) -> Property {
        Property { name: name.into(), ty: TypeName::new("Test"), array_index: 0, value }
    }
    fn f(name: &str, x: f32) -> Property {
        prop(name, Value::Float(x))
    }
    fn s(name: &str, x: &str) -> Property {
        prop(name, Value::Name(x.into()))
    }
    fn st(name: &str, fields: Vec<Property>) -> Property {
        prop(name, Value::Struct { name: "S".into(), fields })
    }
    /// An expression input linked to node `k`, output `out`, mask "rgba" letters.
    fn link(name: &str, k: usize, out: i32, mask: &str) -> Property {
        let bit = |c: char| Value::Int(mask.contains(c) as i32);
        st(
            name,
            vec![
                prop("Expression", Value::Object(PackageIndex(k as i32 + 1))),
                prop("OutputIndex", Value::Int(out)),
                prop("Mask", Value::Int(!mask.is_empty() as i32)),
                prop("MaskR", bit('r')),
                prop("MaskG", bit('g')),
                prop("MaskB", bit('b')),
                prop("MaskA", bit('a')),
            ],
        )
    }

    /// A material whose node 0 holds `inputs` and nodes 1.. are `nodes`.
    fn material(inputs: Vec<Property>, nodes: Vec<(&str, Vec<Property>)>) -> Material {
        from_package(graph(inputs, nodes, &[]))
    }

    /// A graph package: export 0 holds `inputs`, exports 1.. are `nodes`;
    /// `refs` are object paths imported first (see [`obj_ref`]).
    fn graph(inputs: Vec<Property>, nodes: Vec<(&str, Vec<Property>)>, refs: &[&str]) -> Package {
        let class = Import {
            class_package: "/Script/CoreUObject".into(),
            class_name: "Class".into(),
            outer: PackageIndex(0),
            object_name: "Material".into(),
            package_name: String::new(),
            optional: false,
        };
        let mut imports = Vec::new();
        for r in refs {
            let (pkg, obj) = split_object_path(r);
            let outer = PackageIndex(-(imports.len() as i32 + 1));
            imports.push(Import { object_name: pkg, class_name: "Package".into(), ..class.clone() });
            imports.push(Import { object_name: obj, class_name: "MaterialFunction".into(), outer, ..class.clone() });
        }
        imports.push(class.clone());
        let export = |class: i32, name: String, properties: Vec<Property>| Export {
            class: PackageIndex(class),
            super_: PackageIndex(0),
            template: PackageIndex(0),
            outer: PackageIndex(0),
            object_name: name,
            object_flags: 0,
            serial_size: 0,
            serial_offset: 0,
            is_asset: false,
            script_range: None,
            object: Object { properties, ..Default::default() },
        };
        let mut exports = vec![export(-(imports.len() as i32), "M".into(), inputs)];
        for (k, (class, props)) in nodes.into_iter().enumerate() {
            imports.push(Import { object_name: format!("MaterialExpression{class}"), ..imports[refs.len() * 2].clone() });
            exports.push(export(-(imports.len() as i32), format!("N{k}"), props));
        }
        Package { summary: Default::default(), names: vec![], imports, exports, soft_object_paths: vec![], problems: vec![] }
    }

    /// The `i`th of a graph's `refs`.
    fn obj_ref(i: usize) -> Value {
        Value::Object(PackageIndex(-(2 * i as i32 + 2)))
    }

    fn loaded(pkg: Package) -> Rc<Loaded> {
        Rc::new(Loaded { file: PathBuf::new(), bytes: vec![], pkg })
    }

    fn from_package(pkg: Package) -> Material {
        Material {
            chain: vec!["/Game/M.M".into()],
            blend: Blend::Translucent,
            unlit: true,
            two_sided: false,
            clip: 0.3333,
            scalars: HashMap::new(),
            vectors: HashMap::new(),
            textures: HashMap::new(),
            switches: HashMap::new(),
            base: Some((loaded(pkg), 0)),
            missing: None,
            functions: HashMap::new(),
            bound: HashMap::new(),
        }
    }

    fn wesl(m: &Material) -> String {
        compile(m, &Needs::default()).unwrap().wesl
    }

    fn tex_node(sampler: &str) -> (&'static str, Vec<Property>) {
        ("TextureSample", vec![prop("Texture", Value::Object(PackageIndex(-1))), prop("SamplerType", Value::Enum(sampler.into()))])
    }

    #[test]
    fn atlas_sampling_wraps_at_each_texture_edge() {
        // uv 0 must blend the four wrapped corners, as an Unreal Wrap sampler does.
        let m = material(vec![link("EmissiveColor", 1, 0, "rgb")], vec![tex_node("SAMPLERTYPE_Color")]);
        let w = wesl(&m);
        assert!(w.contains("let a = (floor(p) + size) % size;") && w.contains("textureLoad(particle_texture"), "{w}");
        assert!(!w.contains("textureSampleLevel"), "{w}");
    }

    #[test]
    fn texture_outputs_and_sampler_types() {
        // Output 0 is RGB; a grayscale sampler reads R with alpha 1.
        let m = material(
            vec![link("EmissiveColor", 1, 0, ""), link("Opacity", 1, 4, "a")],
            vec![tex_node("SAMPLERTYPE_LinearGrayscale")],
        );
        let w = wesl(&m);
        assert!(w.contains("vec4<f32>(t0.x, t0.x, t0.x, 1.0)"), "{w}");
        assert!(w.contains("vec3<f32> = t1.xyz;") && w.contains("f32 = t1.w;"), "{w}");
        assert_eq!(w.matches("tex(0u").count(), 1, "one sample for both outputs: {w}");
        // Normal maps unpack.
        let n = material(vec![link("EmissiveColor", 1, 0, "")], vec![tex_node("SAMPLERTYPE_Normal")]);
        assert!(wesl(&n).contains("* 2.0 - 1.0)"));
    }

    #[test]
    fn panner_fraction_applies_to_the_offset_only() {
        let m = material(
            vec![link("EmissiveColor", 1, 0, "")],
            vec![("Panner", vec![f("SpeedX", 0.5), prop("bFractionalPart", Value::Bool(true))])],
        );
        assert!(wesl(&m).contains("in.uv0 + fract(vec2<f32>(0.5, 0.0) * globals.time)"), "{}", wesl(&m));
    }

    #[test]
    fn time_period_needs_its_override_flag() {
        let plain = material(vec![link("Opacity", 1, 0, "")], vec![("Time", vec![f("Period", 2.0)])]);
        assert!(wesl(&plain).contains("= globals.time;") && !wesl(&plain).contains("globals.time %"), "{}", wesl(&plain));
        let periodic = material(vec![link("Opacity", 1, 0, "")], vec![("Time", vec![f("Period", 2.0), prop("bOverride_Period", Value::Bool(true))])]);
        assert!(wesl(&periodic).contains("(globals.time % 2.0)"));
    }

    #[test]
    fn max_and_min_default_b_is_one() {
        let m = material(vec![link("Opacity", 1, 0, "")], vec![("Max", vec![link("A", 2, 0, "")]), ("Time", vec![])]);
        assert!(wesl(&m).contains("max(e2_0, 1.0)"), "{}", wesl(&m));
    }

    #[test]
    fn lerp_widens_to_a_vector_alpha() {
        let m = material(
            vec![link("EmissiveColor", 1, 0, "")],
            vec![("LinearInterpolate", vec![link("Alpha", 2, 0, "")]), ("Constant3Vector", vec![])],
        );
        let w = wesl(&m);
        assert!(w.contains("mix(vec3<f32>(0.0), vec3<f32>(1.0), "), "{w}");
    }

    #[test]
    fn if_uses_the_equals_threshold() {
        let m = material(
            vec![link("Opacity", 1, 0, "")],
            vec![
                ("If", vec![link("A", 2, 0, ""), f("ConstB", 0.5), link("AGreaterThanB", 2, 0, ""), link("AEqualsB", 2, 0, ""), link("ALessThanB", 2, 0, ""), f("EqualsThreshold", 0.1)]),
                ("Time", vec![]),
            ],
        );
        let w = wesl(&m);
        assert!(w.contains("abs(e2_0 - 0.5) <= 0.1)"), "{w}");
    }

    #[test]
    fn static_switches_take_instance_overrides() {
        let nodes = vec![
            ("StaticSwitchParameter", vec![s("ParameterName", "UseTime"), link("A", 2, 0, ""), link("B", 3, 0, "")]),
            ("Constant", vec![f("R", 0.75)]),
            ("Constant", vec![f("R", 0.25)]),
        ];
        let mut m = material(vec![link("Opacity", 1, 0, "")], nodes);
        assert!(wesl(&m).contains("saturate(0.25)"), "default false picks B");
        m.switches.insert("UseTime".into(), true);
        assert!(wesl(&m).contains("saturate(0.75)"));

        // The overrides as an instance stores them (UE4 StaticParameters).
        let entry = |name: &str, on: bool, over: bool| Value::Struct {
            name: "StaticSwitchParameter".into(),
            fields: vec![
                st("ParameterInfo", vec![s("Name", name)]),
                prop("Value", Value::Bool(on)),
                prop("bOverride", Value::Bool(over)),
            ],
        };
        let props = vec![st("StaticParameters", vec![prop("StaticSwitchParameters", Value::Array(vec![entry("A", true, true), entry("B", true, false)]))])];
        let mut out = Vec::new();
        static_switches(&props, &mut out);
        assert_eq!(out, vec![("A".to_string(), true)]);
    }

    #[test]
    fn constant_inputs_and_defaults_match_unreal() {
        // A constant Emissive is an sRGB FColor; an unconnected Base Color is black.
        let c = st("Constant", vec![prop("R", Value::Byte(128)), prop("G", Value::Byte(0)), prop("B", Value::Byte(255))]);
        let mut m = material(vec![st("EmissiveColor", vec![prop("UseConstant", Value::Bool(true)), c])], vec![]);
        m.unlit = false;
        let w = wesl(&m);
        assert!(w.contains("vec3<f32>(0.21586") && w.contains(", 0.0, 1.0)"), "{w}");
        assert!(!w.contains("scene_light(normalize"), "{w}");
        // A vector parameter without a saved default is zero.
        let v0 = material(vec![link("Opacity", 1, 4, "a")], vec![("VectorParameter", vec![s("ParameterName", "C")])]);
        assert!(wesl(&v0).contains("saturate(0.0)"), "{}", wesl(&v0));
    }

    #[test]
    fn texture_object_parameters_take_instance_overrides() {
        let mut m = material(
            vec![link("EmissiveColor", 1, 0, "")],
            vec![
                ("TextureSample", vec![link("TextureObject", 2, 0, "")]),
                ("TextureObjectParameter", vec![s("ParameterName", "T"), prop("Texture", Value::Object(PackageIndex(-1)))]),
            ],
        );
        m.textures.insert("T".into(), "/Game/T_Over.T_Over".into());
        assert_eq!(compile(&m, &Needs::default()).unwrap().textures, vec!["/Game/T_Over.T_Over".to_string()]);
    }

    /// The Fab "Perfect Fire" erosion instances (local-only test files;
    /// skipped when testdata/fab is absent).
    #[test]
    fn fab_erosion_instances() {
        let content = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fab/perfect_fire/Content");
        if !content.is_dir() {
            eprintln!("skipped: {} is absent (sh testdata/fab_assets.sh)", content.display());
            return;
        }
        let mut store = Store::new(vec![]);
        store.content_roots.push(content);
        let dir = "/Game/Vefects/Free_Fire/Shared/Materials";
        let needs = Needs { dynamic_written: [true, false, false, false], dynamic_available: [true, true, false, false], mesh: false };
        for (name, blend, unlit) in [
            ("MI_VFX_Fire_02", Blend::Additive, true),
            ("MI_VFX_Fire_01", Blend::Additive, true),
            ("MI_VFX_Glow_01", Blend::Translucent, false),
            ("MI_VFX_Smoke_01", Blend::Translucent, false),
        ] {
            let m = Material::load(&mut store, &format!("{dir}/{name}.{name}")).unwrap();
            assert_eq!((m.blend, m.unlit), (blend, unlit), "{name}");
            let c = compile(&m, &needs).unwrap();
            assert_eq!(c.textures.len(), 3, "{name}: {:?}", c.textures);
            assert!(c.wesl.contains("in.dynamic_param.x"), "{name}: dissolve");
            assert!(c.notes.iter().any(|n| n.contains("Depth Fade")), "{name}");
        }
        // Fire_02: emissive 40, noise 1 panning at (0.5, 0.5) over a 2x tiling.
        let m = Material::load(&mut store, &format!("{dir}/MI_VFX_Fire_02.MI_VFX_Fire_02")).unwrap();
        let w = compile(&m, &needs).unwrap().wesl;
        for s in ["vec3<f32>(40.0)", "vec2<f32>(0.5, 0.5);", "vec2<f32>(2.0, 2.0)", "* 1.2)"] {
            assert!(w.contains(s), "{s}\n{w}");
        }
    }

    /// The Fab portal materials' Custom HLSL nodes translate to WGSL that
    /// naga accepts (local-only test files; skipped when absent).
    #[test]
    fn fab_portal_custom_nodes() {
        let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fab/portals/Content/PortalFX/Materials/M_PRT_Portal.uasset");
        if !file.is_file() {
            eprintln!("skipped: {} is absent", file.display());
            return;
        }
        let mut store = Store::for_system(&file);
        let needs = Needs { dynamic_written: [true; 4], dynamic_available: [true; 4], mesh: true };
        for name in ["MI_PRT_Portal_Red", "MI_PRT_Rays_Red", "MI_PRT_Spark_Red"] {
            let m = Material::load(&mut store, &format!("/Game/PortalFX/Materials/Instances/{name}.{name}")).unwrap();
            let c = compile(&m, &needs).unwrap();
            assert!(!c.notes.iter().any(|n| n.contains("Custom")), "{name}: {:?}", c.notes);
            // The translated functions sit between the helpers and the fragment.
            let start = c.wesl.find("fn custom_").unwrap_or_else(|| panic!("{name}: no Custom function\n{}", c.wesl));
            let end = c.wesl.find("@fragment").unwrap();
            let functions = &c.wesl[start..end];
            let module = naga::front::wgsl::parse_str(functions).unwrap_or_else(|e| panic!("{name}: {}\n{functions}", e.emit_to_string(functions)));
            naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {}\n{functions}", e.emit_to_string(functions)));
            assert!(c.wesl.contains("= custom_"), "{name}: not called\n{}", c.wesl);
        }
    }

    fn guid(k: u32) -> Value {
        Value::Struct { name: "Guid".into(), fields: vec![prop("Guid", Value::Guid(crate::reader::Guid([k, 0, 0, 0])))] }
    }
    fn unlinked(name: &str) -> Property {
        st(name, vec![prop("Expression", Value::Object(PackageIndex(0))), prop("OutputIndex", Value::Int(0))])
    }
    /// A call's FunctionInputs entry: (input id, the input as linked).
    fn call_input(id: u32, mut input: Property, name: &str) -> Value {
        if let Value::Struct { fields, .. } = &mut input.value {
            fields.push(s("InputName", name));
        }
        Value::Struct { name: "FunctionExpressionInput".into(), fields: vec![prop("ExpressionInputId", guid(id)), Property { name: "Input".into(), ..input }] }
    }
    fn call_output(id: u32, name: &str) -> Value {
        Value::Struct {
            name: "FunctionExpressionOutput".into(),
            fields: vec![prop("ExpressionOutputId", guid(id)), st("Output", vec![s("OutputName", name)])],
        }
    }

    /// MF_T: `UV` (vector2, input id 1) * `Scale` (scalar, id 2, preview 3)
    /// as output `Out` (id 3); `Scale` alone as output `S` (id 4).
    fn function_package() -> Package {
        let input = |name: &str, id: u32, ty: &str, preview: [f32; 2]| {
            vec![
                s("InputName", name),
                prop("Id", guid(id)),
                prop("InputType", Value::Enum(format!("EFunctionInputType::{ty}"))),
                st("PreviewValue", vec![f("X", preview[0]), f("Y", preview[1]), f("Z", 0.0), f("W", 1.0)]),
                prop("bUsePreviewValueAsDefault", Value::Bool(true)),
            ]
        };
        graph(
            vec![],
            vec![
                ("FunctionInput", input("UV", 1, "FunctionInput_Vector2", [0.25, 0.75])),
                ("FunctionInput", input("Scale", 2, "FunctionInput_Scalar", [3.0, 0.0])),
                ("Multiply", vec![link("A", 1, 0, ""), link("B", 2, 0, "")]),
                ("FunctionOutput", vec![link("A", 3, 0, ""), prop("Id", guid(3)), s("OutputName", "Out")]),
                ("FunctionOutput", vec![link("A", 2, 0, ""), prop("Id", guid(4)), s("OutputName", "S")]),
            ],
            &[],
        )
    }

    #[test]
    fn material_functions_inline_with_call_inputs_and_previews() {
        // The call lists outputs S then Out; UV is wired, Scale takes its preview.
        let call = vec![
            prop("MaterialFunction", obj_ref(0)),
            prop("FunctionInputs", Value::Array(vec![call_input(1, link("Input", 2, 0, ""), "UV"), call_input(2, unlinked("Input"), "Scale")])),
            prop("FunctionOutputs", Value::Array(vec![call_output(4, "S"), call_output(3, "Out")])),
        ];
        let pkg = graph(
            vec![link("EmissiveColor", 1, 1, ""), link("Opacity", 1, 0, "")],
            vec![("MaterialFunctionCall", call), ("TextureCoordinate", vec![f("UTiling", 2.0)])],
            &["/Game/MF_T.MF_T"],
        );
        let mut m = from_package(pkg);
        // Without the package: the call is not translated.
        let c = compile(&m, &Needs::default()).unwrap();
        assert!(c.notes.iter().any(|n| n == "material function MF_T: package not found, not translated (taken as 1)"), "{:?}", c.notes);
        m.functions.insert("/Game/MF_T".into(), loaded(function_package()));
        let c = compile(&m, &Needs::default()).unwrap();
        assert!(c.notes.is_empty(), "{:?}", c.notes);
        let w = c.wesl;
        assert!(w.contains("let e2_0: vec2<f32> = in.uv0 * vec2<f32>(2.0, 1.0);"), "{w}");
        assert!(w.contains("let f1_e3_0: vec2<f32> = (e2_0 * vec2<f32>(3.0));") && w.contains("finish(max(vec3<f32>(f1_e3_0, 0.0)"), "{w}");
        assert!(w.contains("saturate(3.0)"), "output S is Scale's preview: {w}");
        assert_eq!(w.matches("let f1_e3_0").count(), 1, "{w}");
    }

    #[test]
    fn unconnected_function_inputs_use_their_preview_value() {
        let call = vec![
            prop("MaterialFunction", obj_ref(0)),
            prop("FunctionInputs", Value::Array(vec![call_input(1, unlinked("Input"), "UV"), call_input(2, link("Input", 2, 0, ""), "Scale")])),
            prop("FunctionOutputs", Value::Array(vec![call_output(3, "Out")])),
        ];
        let pkg = graph(vec![link("EmissiveColor", 1, 0, "")], vec![("MaterialFunctionCall", call), ("Constant", vec![f("R", 0.5)])], &["/Game/MF_T.MF_T"]);
        let mut m = from_package(pkg);
        m.functions.insert("/Game/MF_T".into(), loaded(function_package()));
        let w = wesl(&m);
        assert!(w.contains("let f1_e1_0: vec2<f32> = vec2<f32>(0.25, 0.75);") && w.contains("(f1_e1_0 * vec2<f32>(0.5))"), "{w}");
    }

    #[test]
    fn recursive_material_functions_stop() {
        // MF_R calls itself: inlining stops at the depth limit with a note.
        let call = |k: usize| vec![prop("MaterialFunction", obj_ref(0)), prop("FunctionOutputs", Value::Array(vec![call_output(9, "Result")])), link("Unused", k, 0, "")];
        let f = graph(vec![], vec![("MaterialFunctionCall", call(0)), ("FunctionOutput", vec![link("A", 1, 0, ""), prop("Id", guid(9))])], &["/Game/MF_R.MF_R"]);
        let mut m = from_package(graph(vec![link("Opacity", 1, 0, "")], vec![("MaterialFunctionCall", call(0))], &["/Game/MF_R.MF_R"]));
        m.functions.insert("/Game/MF_R".into(), loaded(f));
        let c = compile(&m, &Needs::default()).unwrap();
        assert!(c.notes.iter().any(|n| n.contains("MF_R: nested too deep")), "{:?}", c.notes);
    }

    #[test]
    fn built_in_engine_functions() {
        let engine = "/Engine/Functions/Engine_MaterialFunctions02/Texturing/CustomRotator.CustomRotator";
        let call = vec![
            prop("MaterialFunction", obj_ref(0)),
            prop(
                "FunctionInputs",
                Value::Array(vec![
                    call_input(1, link("Input", 2, 0, ""), "UVs"),
                    call_input(2, unlinked("Input"), "Rotation Center"),
                    call_input(3, link("Input", 3, 0, ""), "Rotation Angle (0-1)"),
                ]),
            ),
            prop("FunctionOutputs", Value::Array(vec![call_output(4, "Rotated Values")])),
        ];
        let m = from_package(graph(
            vec![link("EmissiveColor", 4, 0, "")],
            vec![("MaterialFunctionCall", call), ("TextureCoordinate", vec![]), ("Constant", vec![f("R", 0.25)]), tex_node_with_uv(1)],
            &[engine],
        ));
        let c = compile(&m, &Needs::default()).unwrap();
        assert!(c.notes.is_empty(), "{:?}", c.notes);
        assert!(c.wesl.contains("rotate_uv(in.uv0, vec2<f32>(0.5, 0.5), 0.25 * 6.2831855)"), "{}", c.wesl);

        // BreakOutFloat3Components picks its output by name; MakeFloat2 appends.
        let brk = vec![
            prop("MaterialFunction", obj_ref(0)),
            prop("FunctionInputs", Value::Array(vec![call_input(1, link("Input", 3, 0, ""), "Float3")])),
            prop("FunctionOutputs", Value::Array(vec![call_output(2, "R"), call_output(3, "G"), call_output(4, "B")])),
        ];
        let make = vec![
            prop("MaterialFunction", obj_ref(1)),
            prop("FunctionInputs", Value::Array(vec![call_input(1, link("Input", 1, 2, ""), "X"), call_input(2, unlinked("Input"), "Y")])),
            prop("FunctionOutputs", Value::Array(vec![call_output(5, "Result")])),
        ];
        let m = from_package(graph(
            vec![link("EmissiveColor", 2, 0, "")],
            vec![("MaterialFunctionCall", brk), ("MaterialFunctionCall", make), ("VertexColor", vec![])],
            &["/Engine/Functions/Engine_MaterialFunctions02/Utility/BreakOutFloat3Components.BreakOutFloat3Components", "/Engine/Functions/Engine_MaterialFunctions02/Utility/MakeFloat2.MakeFloat2"],
        ));
        let w = wesl(&m);
        assert!(w.contains("let e1_2: f32 = t0.z;") && w.contains("vec2<f32>(e1_2, 0.0)"), "{w}");

        // Unknown engine functions keep the note.
        let m = from_package(graph(vec![link("Opacity", 1, 0, "")], vec![("MaterialFunctionCall", vec![prop("MaterialFunction", obj_ref(0))])], &["/Engine/Functions/X/Fancy.Fancy"]));
        let c = compile(&m, &Needs::default()).unwrap();
        assert_eq!(c.notes, vec!["material function Fancy: not translated (taken as 1)".to_string()]);
    }

    fn tex_node_with_uv(k: usize) -> (&'static str, Vec<Property>) {
        ("TextureSample", vec![prop("Texture", Value::Object(PackageIndex(-1))), link("Coordinates", k, 0, "")])
    }

    /// The Fab "Free Arrow Trail" materials call MF_Texcoord (in the pack)
    /// and engine functions (local-only test files; skipped when absent).
    #[test]
    fn fab_arrow_trail_material_functions() {
        let content = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fab/arrow_trail/Content");
        if !content.is_dir() {
            eprintln!("skipped: {} is absent (sh testdata/fab_assets.sh)", content.display());
            return;
        }
        let mut store = Store::new(vec![]);
        store.content_roots.push(content);
        let needs = Needs { dynamic_written: [true, false, false, false], dynamic_available: [true, true, false, false], mesh: false };
        for path in [
            "/Game/ArrowTrail/Material/M_Master/M_SimpleTrail.M_SimpleTrail",
            "/Game/ArrowTrail/Material/MI_Instance/MI_HolyRibbon.MI_HolyRibbon",
            "/Game/ArrowTrail/Material/M_Master/M_SimpleParticle.M_SimpleParticle",
            "/Game/ArrowTrail/Demo/Material/M_PrototypeGrid.M_PrototypeGrid",
        ] {
            let m = Material::load(&mut store, path).unwrap();
            let c = compile(&m, &needs).unwrap();
            assert!(!c.notes.iter().any(|n| n.contains("material function")), "{path}: {:?}", c.notes);
        }
        let m = Material::load(&mut store, "/Game/ArrowTrail/Material/M_Master/M_SimpleTrail.M_SimpleTrail").unwrap();
        assert!(m.functions.contains_key("/Game/ArrowTrail/Material/MF_Function/MF_Texcoord"));
        let w = compile(&m, &needs).unwrap().wesl;
        assert!(w.contains("rotate_uv(e30_0, vec2<f32>(0.5, 0.5), 0.0 * 6.2831855)"), "{w}");
        // MF_Texcoord inlined (Holy ribbon: UV * tile + time * speed).
        let m = Material::load(&mut store, "/Game/ArrowTrail/Material/MI_Instance/MI_HolyRibbon.MI_HolyRibbon").unwrap();
        let w = compile(&m, &needs).unwrap().wesl;
        assert!(w.contains("= vec2<f32>(f1_e3_0, f1_e2_0);") && w.contains("tex(1u, f1_e4_0)"), "{w}");
    }
}
