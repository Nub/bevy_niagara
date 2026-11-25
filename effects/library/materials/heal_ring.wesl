// CGHOW healing aura ground ring: a procedural ring whose thickness comes
// from Dynamic Material Parameter x, with a soft inner glow.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5)) * 2.0;
    let thickness = max(in.dynamic_param.x, 0.01);
    let ring = smoothstep(1.0, 1.0 - thickness * 0.4, d) * smoothstep(1.0 - thickness * 1.6, 1.0 - thickness * 0.6, d);
    let inner = smoothstep(1.0, 0.0, d) * 0.25;
    let a = clamp(ring + inner, 0.0, 1.0);
    return finish(in.color.rgb * (ring * 1.5 + inner), in.color.a * a);
}
