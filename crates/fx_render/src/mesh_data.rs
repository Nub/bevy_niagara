//! Meshes for the mesh renderer: built-in shapes and the conversion of any
//! Bevy [`Mesh`] (triangle list) to the vertex-pulling layout `mesh.wesl`
//! reads (position, normal, uv: 8 floats per vertex, u32 indices).

use bevy::mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

/// A mesh's data in the layout `mesh.wesl` reads.
pub struct MeshData {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    /// Largest distance of a vertex from the origin (for culling).
    pub radius: f32,
}

/// The built-in shape `name` (see `fx_core::render::MESH_BUILTINS`), about one
/// metre across.
pub fn builtin(name: &str) -> Option<Mesh> {
    Some(match name {
        "cube" => Cuboid::from_length(1.0).mesh().build(),
        "sphere" => Sphere::new(0.5).mesh().uv(32, 18),
        "torus" => Torus::new(0.25, 0.5).mesh().build(),
        "cylinder" => Cylinder::new(0.5, 1.0).mesh().build(),
        "cone" => Cone::new(0.5, 1.0).mesh().build(),
        "capsule" => Capsule3d::new(0.25, 0.5).mesh().build(),
        _ => return None,
    })
}

/// Converts a triangle-list mesh; `Err` says why it cannot be drawn.
pub fn convert(mesh: &Mesh) -> Result<MeshData, String> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(format!(
            "the mesh renderer draws triangle lists, not {:?}",
            mesh.primitive_topology()
        ));
    }
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p.clone(),
        _ => return Err("the mesh has no Float32x3 positions".into()),
    };
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(n)) if n.len() == positions.len() => n.clone(),
        _ => vec![[0.0, 1.0, 0.0]; positions.len()],
    };
    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(u)) if u.len() == positions.len() => u.clone(),
        _ => vec![[0.0, 0.0]; positions.len()],
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U16(i)) => i.iter().map(|&x| x as u32).collect(),
        Some(Indices::U32(i)) => i.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    if indices.is_empty() {
        return Err("the mesh has no triangles".into());
    }
    let mut vertices = Vec::with_capacity(positions.len() * 8);
    let mut radius = 0.0f32;
    for ((p, n), uv) in positions.iter().zip(&normals).zip(&uvs) {
        vertices.extend_from_slice(p);
        vertices.extend_from_slice(n);
        vertices.extend_from_slice(uv);
        radius = radius.max(Vec3::from_array(*p).length());
    }
    Ok(MeshData {
        vertices,
        indices,
        radius,
    })
}
