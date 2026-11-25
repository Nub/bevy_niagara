
// ---- scene collisions: the depth buffer (bind group 2) ----
// The view is the previous frame's, as the depth buffer is; points are in
// the effect's simulation space. Bevy's depth is reverse-Z (0 = far).

struct Scene {
    clip_from_local: mat4x4<f32>,
    local_from_clip: mat4x4<f32>,
    world_from_local: mat4x4<f32>,
    local_from_world: mat4x4<f32>,
    camera: vec4<f32>,    // camera position (simulation space)
    size: vec2<f32>,      // depth texture size in pixels
    flags: u32,           // 1: depth valid
    pad: u32,
}

@group(2) @binding(0) var<uniform> scene: Scene;
@group(2) @binding(1) var scene_depth: texture_2d<f32>;

struct Contact {
    hit: bool,
    n: vec3<f32>,
    o: vec3<f32>,
}

fn no_contact() -> Contact {
    return Contact(false, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0));
}

// The scene point seen at pixel `px` (w = 0 off-screen or sky).
fn scene_point(px: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(scene.size);
    if (any(px < vec2<i32>(0)) || any(px >= dims)) {
        return vec4<f32>(0.0);
    }
    let z = textureLoad(scene_depth, px, 0).r;
    if (!(z > 0.0)) {
        return vec4<f32>(0.0);
    }
    let uv = (vec2<f32>(px) + 0.5) / scene.size;
    let h = scene.local_from_clip * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    return vec4<f32>(h.xyz / h.w, 1.0);
}

// Tangent at `s` from its neighbours `a` (before) and `b` (after), using the
// one closer in depth so edges do not bend the normal.
fn scene_tangent(s: vec3<f32>, a: vec4<f32>, b: vec4<f32>) -> vec3<f32> {
    let cam = scene.camera.xyz;
    let ds = length(s - cam);
    let da = select(1e30, abs(length(a.xyz - cam) - ds), a.w > 0.0);
    let db = select(1e30, abs(length(b.xyz - cam) - ds), b.w > 0.0);
    if (da < db) {
        return s - a.xyz;
    }
    if (b.w > 0.0) {
        return b.xyz - s;
    }
    return vec3<f32>(0.0);
}

// The surface plane a particle at `pos` is pushed out through, if it is
// behind the depth buffer (by at most `thickness` metres).
fn scene_depth_contact(pos: vec3<f32>, radius: f32, thickness: f32) -> Contact {
    if ((scene.flags & 1u) == 0u) {
        return no_contact();
    }
    let clip = scene.clip_from_local * vec4<f32>(pos, 1.0);
    if (!(clip.w > 0.0)) {
        return no_contact();
    }
    let ndc = clip.xy / clip.w;
    if (any(abs(ndc) >= vec2<f32>(1.0))) {
        return no_contact();
    }
    let px = vec2<i32>(floor(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * scene.size));
    let s = scene_point(px);
    if (s.w == 0.0) {
        return no_contact();
    }
    let cam = scene.camera.xyz;
    let dp = length(pos - cam);
    let ds = length(s.xyz - cam);
    if (dp + radius < ds || dp > ds + thickness) {
        return no_contact();
    }
    let tx = scene_tangent(s.xyz, scene_point(px - vec2<i32>(1, 0)), scene_point(px + vec2<i32>(1, 0)));
    let ty = scene_tangent(s.xyz, scene_point(px - vec2<i32>(0, 1)), scene_point(px + vec2<i32>(0, 1)));
    let c = cross(tx, ty);
    let l = length(c);
    if (!(l > 0.0)) {
        return no_contact();
    }
    var n = c / l;
    if (dot(n, cam - s.xyz) < 0.0) {
        n = -n;
    }
    return Contact(true, n, s.xyz);
}
