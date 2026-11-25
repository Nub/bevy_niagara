//! Static meshes: the editor `FMeshDescription` of a `UStaticMesh` (the
//! payload of its `StaticMeshDescriptionBulkData` export, see
//! [`crate::bulk`]) read into triangle lists, and written out as glTF 2.0
//! binary (`.glb`).
//!
//! The serialized mesh description (UE 5.4/5.5), as found in the files:
//!
//! ```text
//! u32 element type count, then per element type:
//!   FString name ("Vertices", "VertexInstances", "UVs", "Edges",
//!                 "Triangles", "Polygons", "PolygonGroups")
//!   u32 channel count (1, or one per UV channel for "UVs"), per channel:
//!     TBitArray allocation (u32 bit count, u32 words)
//!     u32 (0 in every file seen)
//!     attribute set:                           (FAttributesSetBase)
//!     u32 element count
//!     u32 attribute count, per attribute:
//!       FString name ("Position " has a trailing space in UE)
//!       u32 type: 0 FVector4f, 1 FVector3f, 2 FVector2f, 3 float,
//!                 4 int32 (element IDs), 5 bool, 6 FName
//!       u32 extent (values per element), u32 element count
//!       u32 channel count, per channel:
//!         u32 extent
//!         FName: TArray<FString>; others: u32 element size, u32 count, data
//!       default value (one value; bool as u32, FName as FString)
//!       u32 flags
//! ```
//!
//! Element IDs index the attribute arrays directly; the allocation bits say
//! which IDs are live (holes are skipped).
//!
//! Unreal triangles are clockwise-front in its left-handed space; the
//! mirroring (X, Y, Z) -> (X, Z, Y) swap turns them counter-clockwise, so
//! the corner order is kept as is (checked against the stored normals).

use crate::package::Package;
use crate::reader::{Reader, Result};

/// One attribute's values per channel.
#[derive(Clone, Debug)]
pub enum Values {
    /// Little-endian values, `size` bytes each (`extent` per element).
    Raw { size: usize, channels: Vec<Vec<u8>> },
    Names(Vec<Vec<String>>),
}

#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: String,
    pub ty: u32,
    pub extent: usize,
    pub num_elements: usize,
    pub values: Values,
}

impl Attribute {
    fn raw(&self, channel: usize) -> Option<(usize, &[u8])> {
        match &self.values {
            Values::Raw { size, channels } => channels.get(channel).map(|c| (*size, c.as_slice())),
            Values::Names(_) => None,
        }
    }

    /// Channel `channel` as f32s (any float type).
    pub fn f32s(&self, channel: usize) -> Vec<f32> {
        match self.raw(channel) {
            Some((_, b)) if self.ty <= 3 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            _ => Vec::new(),
        }
    }

    /// Channel `channel` as i32s (element IDs).
    pub fn i32s(&self, channel: usize) -> Vec<i32> {
        match self.raw(channel) {
            Some((4, b)) if self.ty == 4 => b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            _ => Vec::new(),
        }
    }

    pub fn names(&self, channel: usize) -> Vec<String> {
        match &self.values {
            Values::Names(c) => c.get(channel).cloned().unwrap_or_default(),
            Values::Raw { .. } => Vec::new(),
        }
    }
}

/// One channel of an element type: its allocation bits and attributes.
/// Most element types have one channel; "UVs" has one per UV channel.
#[derive(Clone, Debug)]
pub struct Channel {
    pub allocated: Vec<bool>,
    pub num_elements: usize,
    pub attributes: Vec<Attribute>,
}

/// One element type ("Vertices", "Triangles", ...).
#[derive(Clone, Debug)]
pub struct Elements {
    pub name: String,
    pub channels: Vec<Channel>,
}

impl Elements {
    /// An attribute of channel 0.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.channels.first()?.attributes.iter().find(|a| a.name == name)
    }

    /// The live element IDs of channel 0.
    pub fn live(&self) -> Vec<usize> {
        let Some(c) = self.channels.first() else { return Vec::new() };
        (0..c.num_elements).filter(|&i| c.allocated.get(i).copied().unwrap_or(false)).collect()
    }
}

/// A serialized FMeshDescription.
#[derive(Clone, Debug)]
pub struct MeshDescription {
    pub elements: Vec<Elements>,
}

impl MeshDescription {
    pub fn get(&self, name: &str) -> Option<&Elements> {
        self.elements.iter().find(|e| e.name == name)
    }
}

fn bit_array(r: &mut Reader) -> Result<Vec<bool>> {
    let n = r.u32()? as usize;
    let words = n.div_ceil(32);
    let b = r.bytes(words * 4)?;
    Ok((0..n).map(|i| b[i / 8] >> (i % 8) & 1 == 1).collect())
}

fn attribute(r: &mut Reader) -> Result<Attribute> {
    let name = r.fstring()?.trim_end().to_string();
    let ty = r.u32()?;
    if ty > 6 {
        return Err(r.err(format!("attribute {name}: type {ty} not supported")));
    }
    let extent = r.u32()? as usize;
    let num_elements = r.u32()? as usize;
    let nch = r.count(4)?;
    let values = if ty == 6 {
        let mut chans = Vec::with_capacity(nch);
        for _ in 0..nch {
            let _extent = r.u32()?;
            let n = r.count(4)?;
            chans.push((0..n).map(|_| r.fstring()).collect::<Result<Vec<_>>>()?);
        }
        let _default = r.fstring()?;
        Values::Names(chans)
    } else {
        let size = [16, 12, 8, 4, 4, 1][ty as usize];
        let mut chans = Vec::with_capacity(nch);
        for _ in 0..nch {
            let _extent = r.u32()?;
            let elem = r.u32()? as usize;
            if elem != size {
                return Err(r.err(format!("attribute {name}: element size {elem}, expected {size}")));
            }
            let n = r.count(elem)?;
            chans.push(r.bytes(n * elem)?.to_vec());
        }
        let _default = r.bytes(if ty == 5 { 4 } else { size })?;
        Values::Raw { size, channels: chans }
    };
    let _flags = r.u32()?;
    Ok(Attribute {
        name,
        ty,
        extent,
        num_elements,
        values,
    })
}

/// Parses a serialized FMeshDescription (the decompressed bulk payload).
pub fn read_mesh_description(data: &[u8]) -> Result<MeshDescription> {
    let mut r = Reader::new(data);
    let n = r.count(8)?;
    let mut elements = Vec::with_capacity(n);
    for _ in 0..n {
        let name = r.fstring()?;
        let nch = r.count(16)?;
        let mut channels = Vec::with_capacity(nch);
        for _ in 0..nch {
            let allocated = bit_array(&mut r)?;
            let _ = r.u32()?;
            let num_elements = r.u32()? as usize;
            let na = r.count(8)?;
            let attributes = (0..na).map(|_| attribute(&mut r)).collect::<Result<Vec<_>>>()?;
            channels.push(Channel {
                allocated,
                num_elements,
                attributes,
            });
        }
        elements.push(Elements { name, channels });
    }
    if r.remaining() != 0 {
        return Err(r.err(format!("{} bytes after the mesh description", r.remaining())));
    }
    Ok(MeshDescription { elements })
}

/// A triangle mesh in our convention: metres, Y up (Unreal X, Y, Z ->
/// X, Z, Y), counter-clockwise front faces. One vertex per Unreal vertex
/// instance.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uv0: Vec<[f32; 2]>,
    /// Linear RGBA, only when the mesh has non-white vertex colours.
    pub colors: Option<Vec<[f32; 4]>>,
    /// Three per triangle.
    pub indices: Vec<u32>,
    /// Material slot of each triangle (index into `slot_names`).
    pub triangle_slots: Vec<u32>,
    pub slot_names: Vec<String>,
}

impl MeshData {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned bounds (min, max); zeros when empty.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        if self.positions.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    }
}

/// Reads the editor mesh description of the static mesh in `pkg` (the
/// package's file bytes in `bytes`) and converts it to [`MeshData`].
pub fn read_static_mesh(pkg: &Package, bytes: &[u8]) -> std::result::Result<MeshData, String> {
    let export = pkg
        .exports
        .iter()
        .find(|e| e.object_name.starts_with("StaticMeshDescriptionBulkData"))
        .ok_or("no StaticMeshDescriptionBulkData export (not an editor static mesh)")?;
    let payloads = crate::bulk::payloads_of(pkg, bytes, &export.object).map_err(|e| e.to_string())?;
    let payload = payloads.first().ok_or("the mesh description has no payload in the package trailer")?;
    let data = crate::bulk::read_payload(bytes, payload).map_err(|e| e.to_string())?;
    let desc = read_mesh_description(&data).map_err(|e| format!("mesh description: {e}"))?;
    mesh_from_description(&desc)
}

/// Builds [`MeshData`] from a parsed mesh description.
pub fn mesh_from_description(desc: &MeshDescription) -> std::result::Result<MeshData, String> {
    let need = |name: &str| desc.get(name).ok_or_else(|| format!("no {name} in the mesh description"));
    let attr = |e: &'_ Elements, name: &str| -> std::result::Result<Attribute, String> {
        e.attribute(name).cloned().ok_or_else(|| format!("no {}.{name}", e.name))
    };
    let verts = need("Vertices")?;
    let vis = need("VertexInstances")?;
    let tris = need("Triangles")?;

    let pos = attr(verts, "Position")?.f32s(0);
    let vi_vertex = attr(vis, "VertexIndex")?.i32s(0);
    let normal = vis.attribute("Normal").map(|a| a.f32s(0)).unwrap_or_default();
    let uv = vis.attribute("TextureCoordinate").map(|a| a.f32s(0)).unwrap_or_default();
    let color = vis.attribute("Color").map(|a| a.f32s(0)).unwrap_or_default();
    let tri_vi = attr(tris, "VertexInstanceIndex")?.i32s(0);
    let tri_group = tris.attribute("PolygonGroupIndex").map(|a| a.i32s(0)).unwrap_or_default();

    // Material slots: live polygon groups, in ID order.
    let mut group_slot = std::collections::HashMap::new();
    let mut slot_names = Vec::new();
    if let Some(groups) = desc.get("PolygonGroups") {
        let names = groups.attribute("ImportedMaterialSlotName").map(|a| a.names(0)).unwrap_or_default();
        for id in groups.live() {
            group_slot.insert(id as i32, slot_names.len() as u32);
            slot_names.push(names.get(id).cloned().unwrap_or_else(|| format!("slot{id}")));
        }
    }

    let mut out = MeshData::default();
    let mut vi_map: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
    let mut colors = Vec::new();
    for t in tris.live() {
        let Some(ids) = tri_vi.get(3 * t..3 * t + 3) else {
            return Err(format!("triangle {t} past the VertexInstanceIndex array"));
        };
        let mut corner = [0u32; 3];
        for (k, &vi) in ids.iter().enumerate() {
            if let Some(&i) = vi_map.get(&vi) {
                corner[k] = i;
                continue;
            }
            let v = vi as usize;
            let vert = *vi_vertex.get(v).ok_or_else(|| format!("triangle {t}: vertex instance {vi} out of range"))?;
            let p = pos
                .get(3 * vert as usize..3 * vert as usize + 3)
                .ok_or_else(|| format!("vertex instance {vi}: vertex {vert} out of range"))?;
            out.positions.push([p[0] * 0.01, p[2] * 0.01, p[1] * 0.01]);
            let n = normal.get(3 * v..3 * v + 3).map(|n| [n[0], n[2], n[1]]).unwrap_or([0.0; 3]);
            out.normals.push(n);
            out.uv0.push(uv.get(2 * v..2 * v + 2).map(|u| [u[0], u[1]]).unwrap_or([0.0; 2]));
            colors.push(color.get(4 * v..4 * v + 4).map(|c| [c[0], c[1], c[2], c[3]]).unwrap_or([1.0; 4]));
            let i = vi_map.len() as u32;
            vi_map.insert(vi, i);
            corner[k] = i;
        }
        // Unreal is left-handed with clockwise front faces; the mirroring
        // axis swap alone makes them counter-clockwise, so the order stays.
        out.indices.extend_from_slice(&corner);
        let g = tri_group.get(t).copied().unwrap_or(0);
        let slot = match group_slot.get(&g) {
            Some(&s) => s,
            None => {
                let s = slot_names.len() as u32;
                group_slot.insert(g, s);
                slot_names.push(format!("slot{g}"));
                s
            }
        };
        out.triangle_slots.push(slot);
    }
    if slot_names.is_empty() {
        slot_names.push("default".to_string());
    }
    out.slot_names = slot_names;
    if colors.iter().any(|c| *c != [1.0; 4]) {
        out.colors = Some(colors);
    }
    fix_normals(&mut out);
    Ok(out)
}

/// Normalizes the normals; missing (zero or non-finite) ones get the
/// area-weighted normal of the faces around their position.
fn fix_normals(m: &mut MeshData) {
    let bad = |n: &[f32; 3]| {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        !l.is_finite() || l < 1e-6
    };
    if m.normals.iter().any(bad) {
        let key = |p: &[f32; 3]| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        let mut acc: std::collections::HashMap<[u32; 3], [f32; 3]> = std::collections::HashMap::new();
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            for &i in t {
                let s = acc.entry(key(&m.positions[i as usize])).or_insert([0.0; 3]);
                for k in 0..3 {
                    s[k] += n[k];
                }
            }
        }
        for i in 0..m.normals.len() {
            if bad(&m.normals[i]) {
                m.normals[i] = acc.get(&key(&m.positions[i])).copied().unwrap_or([0.0, 1.0, 0.0]);
            }
        }
    }
    for n in &mut m.normals {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        *n = if l.is_finite() && l > 1e-12 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0, 1.0, 0.0] };
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        "0.0".to_string()
    }
}

/// Writes `mesh` as a glTF 2.0 binary: one node, one mesh, one primitive
/// per material slot that has triangles (shared vertex accessors, u32
/// indices), one default material per slot named after it.
pub fn write_glb(mesh: &MeshData) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<String> = Vec::new();
    let mut accessors: Vec<String> = Vec::new();
    let mut push_view = |bin: &mut Vec<u8>, data: &[u8], target: u32| -> usize {
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let off = bin.len();
        bin.extend_from_slice(data);
        views.push(format!(
            "{{\"buffer\":0,\"byteOffset\":{off},\"byteLength\":{},\"target\":{target}}}",
            data.len()
        ));
        views.len() - 1
    };
    let floats = |v: &mut dyn Iterator<Item = f32>| -> Vec<u8> { v.flat_map(|f| f.to_le_bytes()).collect() };
    let n = mesh.positions.len();

    let (lo, hi) = mesh.bounds();
    let v = push_view(&mut bin, &floats(&mut mesh.positions.iter().flatten().copied()), 34962);
    accessors.push(format!(
        "{{\"bufferView\":{v},\"componentType\":5126,\"count\":{n},\"type\":\"VEC3\",\"min\":[{},{},{}],\"max\":[{},{},{}]}}",
        json_f32(lo[0]),
        json_f32(lo[1]),
        json_f32(lo[2]),
        json_f32(hi[0]),
        json_f32(hi[1]),
        json_f32(hi[2])
    ));
    let pos_acc = accessors.len() - 1;
    let v = push_view(&mut bin, &floats(&mut mesh.normals.iter().flatten().copied()), 34962);
    accessors.push(format!("{{\"bufferView\":{v},\"componentType\":5126,\"count\":{n},\"type\":\"VEC3\"}}"));
    let nrm_acc = accessors.len() - 1;
    let v = push_view(&mut bin, &floats(&mut mesh.uv0.iter().flatten().copied()), 34962);
    accessors.push(format!("{{\"bufferView\":{v},\"componentType\":5126,\"count\":{n},\"type\":\"VEC2\"}}"));
    let uv_acc = accessors.len() - 1;
    let col_acc = mesh.colors.as_ref().map(|c| {
        let v = push_view(&mut bin, &floats(&mut c.iter().flatten().copied()), 34962);
        accessors.push(format!("{{\"bufferView\":{v},\"componentType\":5126,\"count\":{n},\"type\":\"VEC4\"}}"));
        accessors.len() - 1
    });

    let mut prims = Vec::new();
    let mut materials = Vec::new();
    for (slot, name) in mesh.slot_names.iter().enumerate() {
        let idx: Vec<u8> = mesh
            .indices
            .chunks_exact(3)
            .zip(&mesh.triangle_slots)
            .filter(|(_, s)| **s as usize == slot)
            .flat_map(|(t, _)| t.iter().flat_map(|i| i.to_le_bytes()))
            .collect();
        if idx.is_empty() {
            continue;
        }
        let count = idx.len() / 4;
        let v = push_view(&mut bin, &idx, 34963);
        accessors.push(format!("{{\"bufferView\":{v},\"componentType\":5125,\"count\":{count},\"type\":\"SCALAR\"}}"));
        let ia = accessors.len() - 1;
        materials.push(format!(
            "{{\"name\":{},\"pbrMetallicRoughness\":{{\"baseColorFactor\":[0.8,0.8,0.8,1.0],\"metallicFactor\":0.0,\"roughnessFactor\":0.5}}}}",
            json_str(name)
        ));
        let col = col_acc.map(|c| format!(",\"COLOR_0\":{c}")).unwrap_or_default();
        prims.push(format!(
            "{{\"attributes\":{{\"POSITION\":{pos_acc},\"NORMAL\":{nrm_acc},\"TEXCOORD_0\":{uv_acc}{col}}},\"indices\":{ia},\"material\":{},\"mode\":4}}",
            materials.len() - 1
        ));
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let mut json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"fx_import\"}},\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\"nodes\":[{{\"mesh\":0}}],\"meshes\":[{{\"primitives\":[{}]}}],\"materials\":[{}],\"accessors\":[{}],\"bufferViews\":[{}],\"buffers\":[{{\"byteLength\":{}}}]}}",
        prims.join(","),
        materials.join(","),
        accessors.join(","),
        views.join(","),
        bin.len()
    )
    .into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }

    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}
