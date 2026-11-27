// The tutorial's stylized dark outline: the same beam shape, wider, as a dark
// alpha-blended band behind the bright core.

import fx_render::particle_io::{FxVertexOutput, finish};

@fragment
fn fragment(in: FxVertexOutput) -> @location(0) vec4<f32> {
    let across = abs(in.uv.y - 0.5) * 2.0;
    let pinch = clamp(sin(in.uv.x * 3.14159265) * 1.6, 0.0, 1.0);
    let band = smoothstep(0.55 * pinch + 0.02, 0.2 * pinch, across);
    return finish(in.color.rgb, in.color.a * band);
}
