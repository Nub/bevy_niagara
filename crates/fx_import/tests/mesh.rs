//! Static meshes from the free Niagara test packages in
//! testdata/niagara/packages (fetched by `sh testdata/niagara_assets.sh`,
//! never committed): read to triangles and written as glTF binary.
//! Skipped when absent.

use std::path::PathBuf;

use fx_import::mesh::{read_static_mesh, write_glb, MeshData};
use fx_import::Package;

/// (file, the largest plausible bounds extent in metres)
const MESHES: &[(&str, f32)] = &[
    ("ST_NiagaraCSIntegration/Meshes/SM_Boid.uasset", 2.0),
    ("Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample.uasset", 2.0),
    ("Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample2.uasset", 2.0),
    ("Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample3.uasset", 2.0),
    ("NiagaraDestructionDriver/Defaults/SM_PreviewMesh_NDD.uasset", 10.0),
];

fn meshes() -> Vec<(PathBuf, f32, MeshData)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/niagara/packages");
    if !dir.is_dir() {
        eprintln!("skipped: {} is absent (sh testdata/niagara_assets.sh)", dir.display());
        return Vec::new();
    }
    MESHES
        .iter()
        .map(|(f, max)| {
            let path = dir.join(f);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let pkg = Package::read(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let mesh = read_static_mesh(&pkg, &bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            (path, *max, mesh)
        })
        .collect()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn static_meshes_read_to_valid_triangles() {
    for (path, max, m) in meshes() {
        let name = path.display();
        let n = m.positions.len();
        assert!(n > 0 && m.triangle_count() > 0, "{name}: empty");
        assert_eq!(m.indices.len() % 3, 0, "{name}");
        assert_eq!(m.normals.len(), n, "{name}");
        assert_eq!(m.uv0.len(), n, "{name}");
        if let Some(c) = &m.colors {
            assert_eq!(c.len(), n, "{name}");
        }
        assert_eq!(m.triangle_slots.len(), m.triangle_count(), "{name}");
        assert!(m.indices.iter().all(|&i| (i as usize) < n), "{name}: index out of range");
        assert!(m.triangle_slots.iter().all(|&s| (s as usize) < m.slot_names.len()), "{name}: slot out of range");
        for (i, nr) in m.normals.iter().enumerate() {
            let l = dot(*nr, *nr).sqrt();
            assert!((l - 1.0).abs() < 1e-3, "{name}: normal {i} has length {l}");
        }
        assert!(m.positions.iter().flatten().all(|v| v.is_finite()), "{name}: non-finite position");
        assert!(m.uv0.iter().flatten().all(|v| v.is_finite()), "{name}: non-finite uv");

        let (lo, hi) = m.bounds();
        let ext = sub(hi, lo);
        assert!(ext.iter().all(|&e| e >= 0.0 && e < max), "{name}: bounds {lo:?}..{hi:?}");
        assert!(ext.iter().any(|&e| e > 0.01), "{name}: degenerate bounds {lo:?}..{hi:?}");

        // Winding: after the axis swap and the flip, the geometric normal of
        // (almost) every triangle agrees with its vertex normals.
        let mut agree = 0;
        let mut total = 0;
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            let g = cross(sub(b, a), sub(c, a));
            if dot(g, g) < 1e-14 {
                continue;
            }
            let vn = [0, 1, 2].map(|k| m.normals[t[k] as usize]);
            let s = [vn[0][0] + vn[1][0] + vn[2][0], vn[0][1] + vn[1][1] + vn[2][1], vn[0][2] + vn[1][2] + vn[2][2]];
            total += 1;
            if dot(g, s) > 0.0 {
                agree += 1;
            }
        }
        assert!(total > 0 && agree as f32 >= 0.95 * total as f32, "{name}: only {agree}/{total} triangles face their normals");
        eprintln!("{name}: {n} vertices, {} triangles, slots {:?}, bounds {lo:?}..{hi:?}", m.triangle_count(), m.slot_names);
    }
}

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

/// The number after `"key":` at the first occurrence of the key.
fn json_number(json: &str, key: &str) -> usize {
    let at = json.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key}")) + key.len() + 3;
    json[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
}

#[test]
fn static_meshes_write_parseable_glb() {
    for (path, _, m) in meshes() {
        let name = path.display();
        let glb = write_glb(&m);
        assert_eq!(&glb[0..4], b"glTF", "{name}");
        assert_eq!(u32_at(&glb, 4), 2, "{name}: version");
        assert_eq!(u32_at(&glb, 8), glb.len(), "{name}: total length");
        let json_len = u32_at(&glb, 12);
        assert_eq!(&glb[16..20], b"JSON", "{name}");
        assert_eq!(json_len % 4, 0, "{name}");
        let json = std::str::from_utf8(&glb[20..20 + json_len]).unwrap();
        assert!(json.starts_with('{') && json.trim_end().ends_with('}'), "{name}");
        assert_eq!(json.matches('{').count(), json.matches('}').count(), "{name}: unbalanced JSON");
        assert_eq!(json.matches('[').count(), json.matches(']').count(), "{name}: unbalanced JSON");
        for key in ["\"POSITION\"", "\"NORMAL\"", "\"TEXCOORD_0\"", "\"indices\"", "\"min\"", "\"max\"", "\"materials\""] {
            assert!(json.contains(key), "{name}: no {key}");
        }
        let bin = 20 + json_len;
        let bin_len = u32_at(&glb, bin);
        assert_eq!(&glb[bin + 4..bin + 8], b"BIN\0", "{name}");
        assert_eq!(bin + 8 + bin_len, glb.len(), "{name}: BIN chunk length");
        assert_eq!(json_number(json, "buffers\":[{\"byteLength"), bin_len, "{name}: buffer byteLength");
        // Index accessors together cover every triangle.
        let index_count: usize = json
            .match_indices("\"componentType\":5125,\"count\":")
            .map(|(at, s)| json[at + s.len()..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<usize>().unwrap())
            .sum();
        assert_eq!(index_count, m.indices.len(), "{name}: index count");
    }
}
