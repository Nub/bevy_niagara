// CGHOW stylized tornado material: a streaky noise texture panned around the
// torus, eroded by Dynamic Material Parameter x (rises over the particle's
// life, so each ring frays away as it climbs), additive green with a
// brighter edge where the erosion cuts.

import fx_render::particle_io::{FxVertexOutput, finish};

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    // Streaks: stretched along the ring (u), panned with age.
    let p = vec2<f32>(in.uv.x * 14.0 + in.normalized_age * 3.0, in.uv.y * 3.0);
    let n = vnoise(p) * 0.65 + vnoise(p * 2.3) * 0.35;
    let erode = in.dynamic_param.x;
    let mask = smoothstep(erode, erode + 0.12, n);
    let edge = smoothstep(erode + 0.2, erode + 0.05, n) * mask;
    let rgb = in.color.rgb * (0.6 + n) + vec3<f32>(0.6, 1.0, 0.5) * edge * 1.5;
    return finish(rgb, in.color.a * mask);
}
