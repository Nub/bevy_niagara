//! Browser workaround (wasm32 only): Safari 26/27 rejects the fork's
//! `pbr_fragment.wesl`, which binds textures with `let`, for any
//! StandardMaterial mesh with UVs, and the bad pipeline blanks the frame.
//! Until the fork is fixed, the web build strips UVs from every mesh and
//! textures from every StandardMaterial (the fox gets a flat fox orange).
//! Desktop keeps everything.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;

pub fn plugin(app: &mut App) {
    if cfg!(target_arch = "wasm32") || std::env::var_os("WIZARD_TD_NO_UV").is_some() {
        app.add_systems(PostUpdate, strip);
    }
}

fn strip(
    new_meshes: Query<&Mesh3d, Added<Mesh3d>>,
    new_mats: Query<&MeshMaterial3d<StandardMaterial>, Added<MeshMaterial3d<StandardMaterial>>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut done_meshes: Local<HashSet<AssetId<Mesh>>>,
    mut done_mats: Local<HashSet<AssetId<StandardMaterial>>>,
) {
    for m in &new_meshes {
        if done_meshes.insert(m.id())
            && let Some(mut mesh) = meshes.get_mut(m.id())
        {
            mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
            mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
            mesh.remove_attribute(Mesh::ATTRIBUTE_TANGENT);
        }
    }
    for m in &new_mats {
        if !done_mats.insert(m.id()) {
            continue;
        }
        let textured = mats.get(m.id()).is_some_and(|s| {
            s.base_color_texture.is_some()
                || s.normal_map_texture.is_some()
                || s.metallic_roughness_texture.is_some()
                || s.emissive_texture.is_some()
                || s.occlusion_texture.is_some()
        });
        if textured && let Some(mut s) = mats.get_mut(m.id()) {
            if s.base_color_texture.take().is_some() {
                s.base_color = Color::srgb(0.85, 0.42, 0.16);
            }
            s.normal_map_texture = None;
            s.metallic_roughness_texture = None;
            s.emissive_texture = None;
            s.occlusion_texture = None;
        }
    }
}
