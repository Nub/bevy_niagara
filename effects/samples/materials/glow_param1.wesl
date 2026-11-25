// Custom particle material for Dynamic Material Parameters 2: brightness from
// Particles.DynamicMaterialParameter1.x and a hue shift from .y.

import fx_render::particle_io::{FxVertexOutput, sample_particle_texture, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let tex = sample_particle_texture(in);
    let glow = max(in.dynamic_param_1.x, 0.0);
    let hue = clamp(in.dynamic_param_1.y, 0.0, 1.0);
    let tint = mix(vec3<f32>(1.0, 0.45, 0.1), vec3<f32>(0.2, 0.6, 1.0), hue);
    return finish(in.color.rgb * tint * tex.rgb * glow, in.color.a * tex.a);
}
