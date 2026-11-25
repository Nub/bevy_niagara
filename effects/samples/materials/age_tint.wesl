// Custom particle material: tints the particle colour from cyan to magenta
// with Particles.DynamicMaterialParameter.x (set by Dynamic Material
// Parameters over the particle's life) and adds a rim from the UV.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let t = clamp(in.dynamic_param.x, 0.0, 1.0);
    let tint = mix(vec3<f32>(0.3, 1.0, 1.2), vec3<f32>(1.4, 0.3, 1.0), t);
    let rim = 0.6 + 0.4 * sin(in.uv.x * 6.2831853 * 4.0);
    return finish(in.color.rgb * tint * rim, in.color.a);
}
