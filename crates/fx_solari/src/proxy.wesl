// Writes one emitter's ray-traced proxy geometry from its GPU particles: per
// particle slot k two quads (facing the camera and away from it, so emissive
// sprites light both sides) into the proxy mesh's vertices, which Solari's
// deform pass copies and builds the emitter's one BLAS from; and the
// particle's colour and age into the data texture the hit function reads.
// Slots past the living particles collapse to a point (zero area: never hit,
// never emit).

import fx_render::particle_surface::sprite_corner;

struct ProxyParams {
    world_from_local: mat4x4<f32>,
    cam_position: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_back: vec4<f32>,
    custom_facing: vec4<f32>,
    custom_alignment: vec4<f32>,
    pivot: vec2<f32>,
    capacity: u32,
    slots: u32,
    facing_mode: u32,
    alignment: u32,
    // Mesh vertices in the slab: first word, words per vertex, attribute offsets.
    base: u32,
    stride: u32,
    normal_offset: u32,
    tangent_offset: u32,
    data_width: u32,
    _pad: u32,
}

const NONE = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> p: ProxyParams;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> draw_args: array<u32>;
@group(0) @binding(4) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(5) var data: texture_storage_2d<rgba16float, write>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * p.capacity + i];
}

fn write_vertex(v: u32, pos: vec3<f32>, n: vec3<f32>, t: vec3<f32>) {
    let b = p.base + v * p.stride;
    vertices[b] = pos.x;
    vertices[b + 1u] = pos.y;
    vertices[b + 2u] = pos.z;
    let nb = b + p.normal_offset;
    vertices[nb] = n.x;
    vertices[nb + 1u] = n.y;
    vertices[nb + 2u] = n.z;
    if p.tangent_offset != NONE {
        let tb = b + p.tangent_offset;
        vertices[tb] = t.x;
        vertices[tb + 1u] = t.y;
        vertices[tb + 2u] = t.z;
        vertices[tb + 3u] = 1.0;
    }
}

fn write_texel(t: u32, value: vec4<f32>) {
    textureStore(data, vec2<u32>(t % p.data_width, t / p.data_width), value);
}

@compute @workgroup_size(64, 1, 1)
fn proxy(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if k >= p.slots { return; }
    if k >= draw_args[1] {
        for (var v = 0u; v < 8u; v++) {
            write_vertex(8u * k + v, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
        }
        write_texel(2u * k, vec4<f32>(0.0));
        write_texel(2u * k + 1u, vec4<f32>(0.0));
        return;
    }
    let i = indices[k];
    let m = p.world_from_local;
    let pos = (m * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz;
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
    let size = vec2<f32>(pw(10u, i), pw(11u, i));
    let rotation = radians(pw(12u, i));
    let facing = (m * vec4<f32>(p.custom_facing.xyz, 0.0)).xyz;
    let alignment = (m * vec4<f32>(p.custom_alignment.xyz, 0.0)).xyz;
    // Corner UVs in the order of the proxy mesh's vertices.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    for (var c = 0u; c < 4u; c++) {
        let corner = sprite_corner(
            pos, vel, size, rotation, corners[c], p.pivot, p.facing_mode, p.alignment,
            facing, alignment,
            p.cam_position.xyz, p.cam_right.xyz, p.cam_up.xyz, p.cam_back.xyz,
        );
        write_vertex(8u * k + c, corner.world, corner.normal, corner.right);
        // (2 mm behind: coplanar twins would let rays commit the one facing away, which emits nothing.)
        write_vertex(8u * k + 4u + c, corner.world - corner.normal * 0.002, -corner.normal, corner.right);
    }
    write_texel(2u * k, vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i)));
    write_texel(2u * k + 1u, vec4<f32>(pw(16u, i), 0.0, 0.0, 0.0));
}
