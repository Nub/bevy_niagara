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

fn attribute(r: &mut Reader, ue4: bool) -> Result<Attribute> {
    let name = r.fstring()?.trim_end().to_string();
    let ty = r.u32()?;
    if ty > 6 {
        return Err(r.err(format!("attribute {name}: type {ty} not supported")));
    }
    let extent = if ue4 { 1 } else { r.u32()? as usize };
    let num_elements = r.u32()? as usize;
    let nch = r.count(4)?;
    let values = if ty == 6 {
        let mut chans = Vec::with_capacity(nch);
        for _ in 0..nch {
            if !ue4 {
                r.u32()?; // extent
            }
            let n = r.count(4)?;
            chans.push((0..n).map(|_| r.fstring()).collect::<Result<Vec<_>>>()?);
        }
        let _default = r.fstring()?;
        Values::Names(chans)
    } else {
        let size = [16, 12, 8, 4, 4, 1][ty as usize];
        let mut chans = Vec::with_capacity(nch);
        for _ in 0..nch {
            if !ue4 {
                r.u32()?; // extent
            }
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
            let attributes = (0..na).map(|_| attribute(&mut r, false)).collect::<Result<Vec<_>>>()?;
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
/// package's file bytes in `bytes`) and converts it to [`MeshData`]. UE4
/// packages (no `StaticMeshDescriptionBulkData` export) are read from the
/// LOD0 source mesh in the StaticMesh native data instead
/// ([`read_ue4_source_mesh`]).
pub fn read_static_mesh(pkg: &Package, bytes: &[u8]) -> std::result::Result<MeshData, String> {
    let Some(export) = pkg.exports.iter().find(|e| e.object_name.starts_with("StaticMeshDescriptionBulkData")) else {
        return read_ue4_source_mesh(pkg, bytes).map_err(|e| {
            format!("no StaticMeshDescriptionBulkData export and no UE4 source mesh ({e}): not an editor static mesh")
        });
    };
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

// ---- UE4 source meshes (FRawMesh, UE4 FMeshDescription) ----

/// The UE4 editor source geometry of the static mesh in `pkg`, LOD0.
/// UStaticMesh's native data (after the tagged properties) holds strip
/// flags, bCooked, the body setup and nav collision references and, per
/// source model, FRawMeshBulkData (FByteBulkData, FGuid, bool32
/// bGuidIsHash) and, from UE 4.2x, bool32 bIsValid plus an
/// FMeshDescriptionBulkData (FByteBulkData, FGuid, bool32 bGuidIsHash,
/// ...) whose payload is a UE4 FMeshDescription ([`read_ue4_mesh_description`]);
/// the raw mesh is then empty. What precedes them varies with the engine
/// version, so the native data is scanned for bulk data headers, in
/// order, and the first payload that reads as a whole [`RawMesh`] or UE4
/// mesh description is LOD0's.
pub fn read_ue4_source_mesh(pkg: &Package, bytes: &[u8]) -> std::result::Result<MeshData, String> {
    let i = pkg.find_export_by_class("StaticMesh").ok_or("no StaticMesh export")?;
    let obj = &pkg.exports[i].object;
    let end = (obj.native_offset + obj.native_size).min(bytes.len());
    // Bulk data flags this reader knows (0x1..0x10000); anything else is
    // not a header.
    const KNOWN_FLAGS: u32 = 0x1_FFFF;
    let le32 = |o: usize| bytes.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let mut last = String::from("no source mesh bulk data in the StaticMesh native data");
    for at in obj.native_offset..end.saturating_sub(20) {
        let flags = le32(at).unwrap_or(u32::MAX);
        if flags & !KNOWN_FLAGS != 0 || flags & 0x20 != 0 {
            continue;
        }
        // Element count and size on disk: positive and within the file.
        let (count, disk) = if flags & 0x2000 != 0 {
            (le32(at + 4).unwrap_or(0), le32(at + 12).unwrap_or(0))
        } else {
            (le32(at + 4).unwrap_or(0), le32(at + 8).unwrap_or(0))
        };
        if count < 16 || disk == 0 || disk as usize > bytes.len() || (flags & 0x2 == 0 && count != disk) {
            continue;
        }
        let Ok(mut r) = Reader::new(bytes).window(at, end) else { continue };
        let Ok(data) = crate::bulk::ue4_bulk_data(pkg, bytes, &mut r) else { continue };
        let raw_err = match read_raw_mesh(&data) {
            Ok(m) => return Ok(mesh_from_raw(&m)),
            Err(e) => e,
        };
        match read_ue4_mesh_description(&data) {
            Ok(d) => return mesh_from_description(&d),
            Err(e) => last = format!("bulk data at {at:#x}: not a raw mesh ({raw_err}) nor a UE4 mesh description ({e})"),
        }
    }
    Err(last)
}

/// Parses a UE 4.2x FMeshDescription (MeshDescriptionNewSerialization), as
/// found in the files:
///
/// ```text
/// element arrays Vertices, VertexInstances, Edges, Polygons, PolygonGroups:
///   TBitArray allocation (u32 bit count, then the u32 words), then per
///   live element: vertex nothing, vertex instance i32 VertexID, edge
///   2 x i32 VertexID, polygon 8 bytes (zeros in every file seen), polygon
///   group nothing
/// one attribute set per element array, in the same order:
///   u32 element count, u32 attribute count, per attribute:
///     FString name, u32 type (as UE5), u32 element count,
///     u32 channel count, per channel: FName: TArray<FString>;
///       others: u32 element size, u32 count, data
///     default value (bool as u32, FName as FString), u32 flags
/// triangle array: allocation, per live triangle 3 x i32 VertexInstanceID
///   and i32 PolygonID; then its attribute set
/// ```
///
/// The result uses the UE5 names [`mesh_from_description`] reads: the
/// vertex instance's vertex becomes `VertexInstances.VertexIndex`, the
/// triangle corners `Triangles.VertexInstanceIndex` and the triangle's
/// polygon group `Triangles.PolygonGroupIndex` (polygon groups are taken
/// from the polygon's second i32 when it is a valid group, else 0).
pub fn read_ue4_mesh_description(data: &[u8]) -> Result<MeshDescription> {
    fn ids(name: &str, n: usize, per: usize, values: &[i32]) -> Attribute {
        Attribute {
            name: name.to_string(),
            ty: 4,
            extent: per,
            num_elements: n,
            values: Values::Raw { size: 4, channels: vec![values.iter().flat_map(|v| v.to_le_bytes()).collect()] },
        }
    }
    fn live(bits: &[bool]) -> impl Iterator<Item = usize> + '_ {
        bits.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i)
    }
    let mut r = Reader::new(data);
    let verts = bit_array(&mut r)?;
    let vis = bit_array(&mut r)?;
    let mut vi_vertex = vec![-1i32; vis.len()];
    for i in live(&vis) {
        vi_vertex[i] = r.i32()?;
    }
    let edges = bit_array(&mut r)?;
    r.bytes(8 * live(&edges).count())?;
    let polys = bit_array(&mut r)?;
    let mut poly_group = vec![0i32; polys.len()];
    for i in live(&polys) {
        let _ = r.i32()?;
        poly_group[i] = r.i32()?;
    }
    let groups = bit_array(&mut r)?;
    if vi_vertex.iter().zip(&vis).any(|(&v, &l)| l && (v < 0 || v as usize >= verts.len())) {
        return Err(r.err("vertex instance with a bad vertex"));
    }
    let mut sets = Vec::new();
    for bits in [&verts, &vis, &edges, &polys, &groups] {
        let n = r.u32()? as usize;
        if n != bits.len() {
            return Err(r.err(format!("attribute set of {n} elements for {} allocated", bits.len())));
        }
        let na = r.count(8)?;
        sets.push((0..na).map(|_| attribute(&mut r, true)).collect::<Result<Vec<_>>>()?);
    }
    let tris = bit_array(&mut r)?;
    let mut tri_vi = vec![0i32; 3 * tris.len()];
    let mut tri_group = vec![0i32; tris.len()];
    for t in live(&tris) {
        for k in 0..3 {
            tri_vi[3 * t + k] = r.i32()?;
        }
        let p = r.i32()?;
        let g = poly_group.get(p as usize).copied().unwrap_or(0);
        tri_group[t] = if g >= 0 && groups.get(g as usize) == Some(&true) { g } else { 0 };
    }
    let n = r.u32()? as usize;
    if n != tris.len() {
        return Err(r.err(format!("triangle attribute set of {n} elements for {}", tris.len())));
    }
    let na = r.count(8)?;
    let mut tri_attrs = (0..na).map(|_| attribute(&mut r, true)).collect::<Result<Vec<_>>>()?;
    if r.remaining() != 0 {
        return Err(r.err(format!("{} bytes after the mesh description", r.remaining())));
    }
    tri_attrs.push(ids("VertexInstanceIndex", tris.len(), 3, &tri_vi));
    tri_attrs.push(ids("PolygonGroupIndex", tris.len(), 1, &tri_group));
    let mut sets = sets.into_iter();
    let mut elements = Vec::new();
    for (name, bits) in [("Vertices", verts), ("VertexInstances", vis), ("Edges", edges), ("Polygons", polys), ("PolygonGroups", groups)] {
        let mut attributes = sets.next().unwrap_or_default();
        if name == "VertexInstances" {
            attributes.push(ids("VertexIndex", bits.len(), 1, &vi_vertex));
        }
        elements.push(Elements { name: name.to_string(), channels: vec![Channel { num_elements: bits.len(), allocated: bits, attributes }] });
    }
    elements.push(Elements {
        name: "Triangles".to_string(),
        channels: vec![Channel { num_elements: tris.len(), allocated: tris, attributes: tri_attrs }],
    });
    Ok(MeshDescription { elements })
}

/// FRawMesh (UE4 editor source geometry): faces of three wedges (corners).
#[derive(Clone, Debug, Default)]
pub struct RawMesh {
    pub version: i32,
    pub face_material_indices: Vec<i32>,
    pub face_smoothing_masks: Vec<u32>,
    pub vertex_positions: Vec<[f32; 3]>,
    pub wedge_indices: Vec<u32>,
    pub wedge_tangent_x: Vec<[f32; 3]>,
    pub wedge_tangent_y: Vec<[f32; 3]>,
    pub wedge_tangent_z: Vec<[f32; 3]>,
    /// MAX_MESH_TEXTURE_COORDS (8) channels, each empty or one per wedge.
    pub wedge_tex_coords: Vec<Vec<[f32; 2]>>,
    /// FColor bytes as B, G, R, A.
    pub wedge_colors: Vec<[u8; 4]>,
    pub material_index_to_import_index: Vec<i32>,
}

/// Reads an FRawMesh: i32 version, i32 licensee version, then TArrays
/// FaceMaterialIndices (i32), FaceSmoothingMasks (u32), VertexPositions
/// (FVector, 3 x f32), WedgeIndices (u32), WedgeTangentX/Y/Z (FVector),
/// 8 x WedgeTexCoords (FVector2D, 2 x f32), WedgeColors (FColor) and,
/// from version 1, MaterialIndexToImportIndex (i32). The whole buffer
/// must be consumed and the arrays must agree in size.
pub fn read_raw_mesh(data: &[u8]) -> Result<RawMesh> {
    let mut r = Reader::new(data);
    let version = r.i32()?;
    let _licensee = r.i32()?;
    if !(0..=16).contains(&version) {
        return Err(r.err(format!("raw mesh version {version}")));
    }
    fn arr<T>(r: &mut Reader, size: usize, f: impl Fn(&mut Reader) -> Result<T>) -> Result<Vec<T>> {
        let n = r.count(size)?;
        (0..n).map(|_| f(r)).collect()
    }
    let v3 = |r: &mut Reader| Ok([r.f32()?, r.f32()?, r.f32()?]);
    let mut m = RawMesh { version, ..Default::default() };
    m.face_material_indices = arr(&mut r, 4, |r| r.i32())?;
    m.face_smoothing_masks = arr(&mut r, 4, |r| r.u32())?;
    m.vertex_positions = arr(&mut r, 12, v3)?;
    m.wedge_indices = arr(&mut r, 4, |r| r.u32())?;
    m.wedge_tangent_x = arr(&mut r, 12, v3)?;
    m.wedge_tangent_y = arr(&mut r, 12, v3)?;
    m.wedge_tangent_z = arr(&mut r, 12, v3)?;
    for _ in 0..8 {
        m.wedge_tex_coords.push(arr(&mut r, 8, |r| Ok([r.f32()?, r.f32()?]))?);
    }
    m.wedge_colors = arr(&mut r, 4, |r| r.array::<4>())?;
    if version >= 1 {
        m.material_index_to_import_index = arr(&mut r, 4, |r| r.i32())?;
    }
    if r.remaining() != 0 {
        return Err(r.err(format!("{} bytes after the raw mesh", r.remaining())));
    }
    let wedges = m.wedge_indices.len();
    let faces = m.face_material_indices.len();
    if wedges == 0 || wedges != 3 * faces {
        return Err(r.err(format!("{wedges} wedges for {faces} faces")));
    }
    let per_wedge = |n: usize| n == 0 || n == wedges;
    if !per_wedge(m.wedge_tangent_z.len())
        || !per_wedge(m.wedge_colors.len())
        || !m.wedge_tex_coords.iter().all(|c| per_wedge(c.len()))
    {
        return Err(r.err("wedge arrays disagree in size"));
    }
    if m.wedge_indices.iter().any(|&i| i as usize >= m.vertex_positions.len()) {
        return Err(r.err("wedge index past the vertex positions"));
    }
    Ok(m)
}

/// Builds [`MeshData`] from an FRawMesh, the same way
/// [`mesh_from_description`] does: one vertex per wedge, centimetres to
/// metres, Unreal X, Y, Z -> X, Z, Y with the winding kept, normals from
/// TangentZ, UV channel 0, vertex colours (sRGB FColor -> linear) only when
/// not all white, one slot per material index.
pub fn mesh_from_raw(raw: &RawMesh) -> MeshData {
    let mut out = MeshData::default();
    let mut slot_of = std::collections::BTreeMap::new();
    for &mi in &raw.face_material_indices {
        slot_of.insert(mi, 0u32);
    }
    for (s, (mi, slot)) in slot_of.iter_mut().enumerate() {
        *slot = s as u32;
        out.slot_names.push(format!("slot{mi}"));
    }
    if out.slot_names.is_empty() {
        out.slot_names.push("default".to_string());
    }
    let srgb = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let mut colors = Vec::with_capacity(raw.wedge_indices.len());
    for (w, &vi) in raw.wedge_indices.iter().enumerate() {
        let p = raw.vertex_positions[vi as usize];
        out.positions.push([p[0] * 0.01, p[2] * 0.01, p[1] * 0.01]);
        out.normals.push(raw.wedge_tangent_z.get(w).map(|n| [n[0], n[2], n[1]]).unwrap_or([0.0; 3]));
        out.uv0.push(raw.wedge_tex_coords.first().and_then(|c| c.get(w)).copied().unwrap_or([0.0; 2]));
        colors.push(
            raw.wedge_colors.get(w).map(|c| [srgb(c[2]), srgb(c[1]), srgb(c[0]), c[3] as f32 / 255.0]).unwrap_or([1.0; 4]),
        );
        out.indices.push(w as u32);
    }
    out.triangle_slots = raw.face_material_indices.iter().map(|mi| slot_of[mi]).collect();
    if colors.iter().any(|c| *c != [1.0; 4]) {
        out.colors = Some(colors);
    }
    fix_normals(&mut out);
    out
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
