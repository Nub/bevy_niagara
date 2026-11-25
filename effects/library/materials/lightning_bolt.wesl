// CGHOW stylized lightning material, rebuilt procedurally: a channel-packed
// core (sharp white line) and outer glow (soft falloff) across the ribbon,
// times the particle colour, pinched to nothing at both ends of the beam
// (the tutorial's "Material UV Pinching for Tapered Ends").

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let across = abs(in.uv.y - 0.5) * 2.0;                   // 0 centre line, 1 edge
    let pinch = clamp(sin(in.uv.x * 3.14159265) * 1.6, 0.0, 1.0);
    let core = smoothstep(0.22 * pinch + 0.02, 0.0, across);
    let glow = exp(-across * 3.5 / max(pinch, 0.05)) * 0.55;
    let rgb = in.color.rgb * glow + vec3<f32>(1.0, 1.0, 1.0) * core * 3.0;
    return finish(rgb, in.color.a * clamp(core + glow, 0.0, 1.0));
}
