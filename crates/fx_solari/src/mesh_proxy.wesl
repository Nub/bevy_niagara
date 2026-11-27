// Writes one mesh emitter's ray-traced proxy geometry from its GPU
// particles: the proxy mesh holds one copy of the renderer's mesh per slot,
// and thread (v, k) places vertex v of slot k's copy where the raster mesh
// shader draws it (`fx_render::particle_surface::mesh_point`). Solari's
// deform pass copies the vertices and rebuilds the emitter's one BLAS from
// them. The particle's colour and age go into the data texture the hit
// function reads (two texels per slot, as sprites). Slots past the traced
// particles collapse to a point at the effect's origin (never hit).

import fx_render::particle_surface::mesh_point;

struct MeshProxyParams {
    world_from_local: mat4x4<f32>,
    cam_position: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_back: vec4<f32>,
    capacity: u32,
    slots: u32,
    // Vertices of the renderer's mesh (one copy per slot).
    mesh_vertices: u32,
    facing_mode: u32,
    // Proxy vertices in the slab: first word, words per vertex, attribute offsets.
    base: u32,
    stride: u32,
    normal_offset: u32,
    tangent_offset: u32,
    data_width: u32,
    // The emitter stores the extended words (Scale 28..30, MeshOrientation 31..34).
    extended: u32,
    // Opaque blend: alpha does not matter (faint particles stay).
    opaque: u32,
}

const NONE = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> p: MeshProxyParams;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> draw_args: array<u32>;
// The renderer's mesh: 8 floats per vertex (position, normal, uv).
@group(0) @binding(4) var<storage, read> source: array<f32>;
@group(0) @binding(5) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(6) var data: texture_storage_2d<rgba16float, write>;

fn pw(w: u32, i: u32) -> f32 {
    return particles[w * p.capacity + i];
}

fn write_vertex(v: u32, pos: vec3<f32>, n: vec3<f32>) {
    let b = p.base + v * p.stride;
    vertices[b] = pos.x;
    vertices[b + 1u] = pos.y;
    vertices[b + 2u] = pos.z;
    let nb = b + p.normal_offset;
    vertices[nb] = n.x;
    vertices[nb + 1u] = n.y;
    vertices[nb + 2u] = n.z;
    if p.tangent_offset != NONE {
        // Any unit vector across the normal (particle meshes carry no tangents).
        var t = cross(n, vec3<f32>(0.0, 1.0, 0.0));
        if dot(t, t) < 1e-6 {
            t = cross(n, vec3<f32>(1.0, 0.0, 0.0));
        }
        t = normalize(t);
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
fn mesh_proxy(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x;
    let k = id.y;
    if v >= p.mesh_vertices || k >= p.slots { return; }
    // Past `p.slots` living particles an even subset is traced (slot k takes
    // every `weight`-th one), as for sprites; meshes keep their own opacity.
    let alive = draw_args[1];
    let used = min(alive, p.slots);
    let weight = f32(alive) / f32(max(used, 1u));
    var faint = false;
    var i = 0u;
    if k < used {
        i = indices[min(u32((f32(k) + 0.5) * weight), alive - 1u)];
        faint = p.opaque == 0u && pw(9u, i) < 0.02;
    }
    let out_v = k * p.mesh_vertices + v;
    if k >= used || faint {
        // At the effect's origin, not the world's: keeps the BLAS bounds tight.
        write_vertex(out_v, p.world_from_local[3].xyz, vec3<f32>(0.0, 1.0, 0.0));
        if v == 0u {
            write_texel(2u * k, vec4<f32>(0.0));
            write_texel(2u * k + 1u, vec4<f32>(0.0));
        }
        return;
    }
    let m = p.world_from_local;
    let center = (m * vec4<f32>(pw(0u, i), pw(1u, i), pw(2u, i), 1.0)).xyz;
    let vel = (m * vec4<f32>(pw(3u, i), pw(4u, i), pw(5u, i), 0.0)).xyz;
    var scale = vec3<f32>(1.0);
    var q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if p.extended != 0u {
        scale = vec3<f32>(pw(28u, i), pw(29u, i), pw(30u, i));
        q = vec4<f32>(pw(31u, i), pw(32u, i), pw(33u, i), pw(34u, i));
    }
    let b = v * 8u;
    let pos = vec3<f32>(source[b], source[b + 1u], source[b + 2u]);
    let nrm = vec3<f32>(source[b + 3u], source[b + 4u], source[b + 5u]);
    let point = mesh_point(
        m, center, vel, scale, q, pos, nrm, p.facing_mode, p.cam_position.xyz, p.cam_back.xyz,
    );
    var n = point.normal;
    let l = length(n);
    n = select(vec3<f32>(0.0, 1.0, 0.0), n / l, l > 1e-6);
    write_vertex(out_v, point.world, n);
    if v == 0u {
        write_texel(2u * k, vec4<f32>(pw(6u, i), pw(7u, i), pw(8u, i), pw(9u, i)));
        write_texel(2u * k + 1u, vec4<f32>(pw(16u, i), 1.0, 0.0, 0.0));
    }
}
