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
    return flipbook_frame(clamp(nage, 0.0, 0.99999) * sub_image.x * sub_image.y, uv, sub_image, blending);
}

// Flipbook frame `frame` (fractional between frames, wrapping past the last)
// for quad UV `uv`: Particles.SubImageIndex.
fn flipbook_frame(frame: f32, uv: vec2<f32>, sub_image: vec2<f32>, blending: bool) -> Flipbook {
    let frames = sub_image.x * sub_image.y;
    let f = frame - frames * floor(frame / frames);
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

// ---------------------------------------------------------------- ribbons
// Shared by the raster ribbon shader (`ribbon.wesl`) and fx_solari's ribbon
// proxy pass and hit function.

// U at sorted position k of the ribbon spanning sorted positions s..=e, from
// the distances along the sorted polyline at k, s and e. `uv_mode`: 0 scaled
// uniformly, 1 by segment length, 2 tiled every `tiling_length` metres.
fn ribbon_u(
    k: u32, s: u32, e: u32, dk: f32, ds: f32, de: f32,
    uv_mode: u32, from_end: bool, tiling_length: f32,
) -> f32 {
    switch uv_mode {
        case 1u: {
            let total = max(de - ds, 1e-6);
            return select(dk - ds, de - dk, from_end) / total;
        }
        case 2u: {
            return select(dk - ds, de - dk, from_end) / tiling_length;
        }
        default: {
            let span = f32(max(e - s, 1u));
            return select(f32(k - s), f32(e - k), from_end) / span;
        }
    }
}

// A point on a ribbon's surface.
struct RibbonPoint {
    world: vec3<f32>,
    // cross(side, tangent): one fixed side of the surface (not flipped to the viewer).
    normal: vec3<f32>,
}

// The point at `t` in 0..1 along the segment p1 → p2 (uniform Catmull-Rom with
// neighbours p0 and p3), `side_sign` (-1 or 1) half a `width` across it. The
// width runs across the tangent, seen face-on from `cam_position` (facing
// mode 0, Screen) or facing `custom_facing` (1), then turned by `twist`
// radians. `cam_up` / `cam_right` are fallbacks for degenerate directions.
fn ribbon_point(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>,
    t: f32, side_sign: f32, width: f32, twist: f32,
    facing_mode: u32, custom_facing: vec3<f32>,
    cam_position: vec3<f32>, cam_right: vec3<f32>, cam_up: vec3<f32>,
) -> RibbonPoint {
    let a = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    let b = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
    let pos = 0.5 * (2.0 * p1 + (p2 - p0) * t + a * t * t + b * t * t * t);
    let d = 0.5 * ((p2 - p0) + 2.0 * a * t + 3.0 * b * t * t);
    let tangent = surface_safe_normalize(d, surface_safe_normalize(p2 - p1, vec3<f32>(0.0, 1.0, 0.0)));

    var facing = cam_position - pos;
    if (facing_mode == 1u) {
        facing = custom_facing;
    }
    var side = cross(tangent, facing);
    if (length(side) < 1e-6) {
        side = cross(tangent, cam_up);
    }
    side = surface_safe_normalize(side, cam_right);
    side = side * cos(twist) + cross(tangent, side) * sin(twist);
    return RibbonPoint(pos + side * (0.5 * width * side_sign), cross(side, tangent));
}

// The built-in ribbon look (no texture): soft edges across the width, `v` in 0..1.
fn ribbon_builtin_texel(v: f32) -> vec4<f32> {
    let x = v * 2.0 - 1.0;
    return vec4<f32>(1.0, 1.0, 1.0, 1.0 - x * x);
}

// ----------------------------------------------------------------- meshes
// Shared by the raster mesh shader (`mesh.wesl`) and fx_solari's mesh proxy
// pass, so a mesh particle sits in reflections where it is drawn.

// q * v * q⁻¹ for unit quaternion q = (x, y, z, w).
fn mesh_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Rotation taking the mesh's X axis to `x` (world), keeping Y as close to
// world up as it can.
fn mesh_facing_basis(x: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(x, up)) > 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let z = normalize(cross(x, up));
    let y = cross(z, x);
    return mat3x3<f32>(x, y, z);
}

// A mesh particle's vertex in world space, with its (unnormalized) normal.
struct MeshPoint {
    world: vec3<f32>,
    normal: vec3<f32>,
}

// Where mesh vertex `p` (normal `nrm`) of a particle at world `center` lies:
// scaled by `scale`, rotated by quaternion `q` (Particles.MeshOrientation),
// then turned by the facing mode (0 Default: the effect transform `m`; 1
// Velocity: X along world velocity `vel`; 2 CameraPosition: X toward
// `cam_position`; 3 CameraPlane: X along the view direction, -`cam_back`).
fn mesh_point(
    m: mat4x4<f32>,
    center: vec3<f32>,
    vel: vec3<f32>,
    scale: vec3<f32>,
    q: vec4<f32>,
    p: vec3<f32>,
    nrm: vec3<f32>,
    facing_mode: u32,
    cam_position: vec3<f32>,
    cam_back: vec3<f32>,
) -> MeshPoint {
    var offset = mesh_rotate(q, p * scale);
    // Normals transform by the inverse scale.
    var normal = mesh_rotate(q, nrm / select(scale, vec3<f32>(1.0), scale == vec3<f32>(0.0)));
    var x = vec3<f32>(0.0);
    switch facing_mode {
        case 1u: { x = vel; }
        case 2u: { x = cam_position - center; }
        case 3u: { x = -cam_back; }
        default: {}
    }
    if (facing_mode != 0u && length(x) > 1e-6) {
        let f = mesh_facing_basis(normalize(x));
        offset = f * offset;
        normal = f * normal;
    } else {
        offset = (m * vec4<f32>(offset, 0.0)).xyz;
        normal = (m * vec4<f32>(normal, 0.0)).xyz;
    }
    return MeshPoint(center + offset, normal);
}
