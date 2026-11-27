// Gabriel Aguiar's muzzle cone texture, procedural: bright at the muzzle,
// fading to the tip, streaked around the cone, cap hidden.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let along = clamp(in.uv.y, 0.0, 1.0);
    let streak = 0.55 + 0.45 * sin(in.uv.x * 6.2831853 * 7.0 + in.normalized_age * 9.0);
    let fade = pow(along, 1.5) * streak;
    return finish(in.color.rgb * fade, in.color.a * fade);
}
