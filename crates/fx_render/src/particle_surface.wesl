// Sprite geometry and appearance as pure functions, shared by the raster
// sprite shader (`sprite.wesl`) and the ray-tracing side (`fx_solari`: the
// proxy geometry pass and the Solari hit function), so a sprite looks the same
// seen directly and in a reflection. No bindings: safe to import anywhere.

// A flipbook frame: cell UVs of the current and next frame, the blend between
// them, and the (fractional) frame index.
struct Flipbook {
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    blend: f32,
    index: f32,
}

// UV of `uv` inside cell `frame` of a `sub_image` (columns, rows) grid.
fn flipbook_cell_uv(frame: f32, uv: vec2<f32>, sub_image: vec2<f32>) -> vec2<f32> {
    let col = frame % sub_image.x;
    let row = floor(frame / sub_image.x);
    return (vec2<f32>(col, row) + uv) / sub_image;
}

// The flipbook frame at normalized age `nage` for quad UV `uv`.
fn flipbook(nage: f32, uv: vec2<f32>, sub_image: vec2<f32>, blending: bool) -> Flipbook {
    let frames = sub_image.x * sub_image.y;
    let f = clamp(nage, 0.0, 0.99999) * frames;
    let f0 = floor(f);
    let f1 = min(f0 + 1.0, frames - 1.0);
    return Flipbook(
        flipbook_cell_uv(f0, uv, sub_image),
        flipbook_cell_uv(f1, uv, sub_image),
        select(0.0, f - f0, blending),
        f,
    );
}

// Colour and opacity of a sprite texel: particle colour times texture.
fn sprite_texel(color: vec4<f32>, tex: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * tex.rgb, color.a * tex.a);
}

// Unit vector, or `fallback` when `v` is (nearly) zero.
fn surface_safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(fallback, v / l, l > 1e-6);
}

// A sprite corner in world space, with the sprite plane's axes.
struct SpriteCorner {
    world: vec3<f32>,
    // Facing the viewer.
    normal: vec3<f32>,
    right: vec3<f32>,
}

// Where corner `uv` ((0,0) top left) of a sprite lies.
// `pos`, `vel`: particle position and velocity (world); `size` metres;
// `rotation` radians; `custom_facing` / `custom_alignment` in world space;
// the camera's position and axes (right, up, back) in world space.
// Facing modes: 0 FaceCamera (camera position, camera roll), 1
// FaceCameraPlane, 2 FaceCameraPosition (world up), 3 CustomFacingVector.
// Alignment: 0 unaligned, 1 velocity, 2 custom vector.
fn sprite_corner(
    pos: vec3<f32>,
    vel: vec3<f32>,
    size: vec2<f32>,
    rotation: f32,
    uv: vec2<f32>,
    pivot: vec2<f32>,
    facing_mode: u32,
    alignment: u32,
    custom_facing: vec3<f32>,
    custom_alignment: vec3<f32>,
    cam_position: vec3<f32>,
    cam_right: vec3<f32>,
    cam_up: vec3<f32>,
    cam_back: vec3<f32>,
) -> SpriteCorner {
    var n = cam_back;
    var up_ref = cam_up;
    switch facing_mode {
        case 0u: { n = surface_safe_normalize(cam_position - pos, cam_back); }
        case 2u: {
            n = surface_safe_normalize(cam_position - pos, cam_back);
            up_ref = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 3u: { n = surface_safe_normalize(custom_facing, cam_back); }
        default: {}
    }
    if (alignment == 1u) {
        up_ref = surface_safe_normalize(vel, up_ref);
    } else if (alignment == 2u) {
        up_ref = surface_safe_normalize(custom_alignment, up_ref);
    }
    var right = cross(up_ref, n);
    if (length(right) < 1e-6) {
        right = cam_right;
    }
    right = normalize(right);
    let up = cross(n, right);

    let q = (vec2<f32>(uv.x, 1.0 - uv.y) - vec2<f32>(pivot.x, 1.0 - pivot.y)) * size;
    let c = cos(rotation);
    let s = sin(rotation);
    let r = vec2<f32>(q.x * c - q.y * s, q.x * s + q.y * c);
    return SpriteCorner(pos + right * r.x + up * r.y, n, right);
}
