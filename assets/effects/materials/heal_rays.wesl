// CGHOW healing aura vertical rays: an unlit additive cylinder whose
// gradient is brightest at the ground and fades upward; the caps are hidden.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let side = 1.0 - smoothstep(0.6, 0.9, abs(in.normal.y));
    let up = clamp(in.uv.y, 0.0, 1.0);
    let grad = pow(1.0 - up, 2.2);
    let streak = 0.9 + 0.1 * sin(in.uv.x * 6.2831853 * 9.0);
    let a = grad * streak * side;
    return finish(in.color.rgb * a, in.color.a * a);
}
