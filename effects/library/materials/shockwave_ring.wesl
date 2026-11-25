// CGHOW "Radial to Shockwave" material: Radial Gradient Exponential, 1-Minus
// (bright at the rim, dark in the middle), multiplied by a hard outer mask
// and raised to a power for a sharp edge. Particle colour (HDR) gives the
// brightness, its alpha the fade over life.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5)) * 2.0;             // 0 centre, 1 edge
    let radial = clamp(1.0 - d, 0.0, 1.0);                    // Radial Gradient
    let inv = pow(1.0 - radial, 3.5);                         // 1-Minus, exponent: rim-heavy
    let edge = smoothstep(1.0, 0.93, d);                      // radius/density: sharp outer edge
    let ring = inv * edge;
    let hot = pow(ring, 6.0);                                 // power: hot yellow rim line
    let rgb = in.color.rgb * ring + vec3<f32>(1.0, 0.75, 0.3) * hot * 3.0;
    return finish(rgb, in.color.a * ring);
}
