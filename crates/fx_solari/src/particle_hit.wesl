// Solari hit function of ray-traced particles (`ParticleRtMaterial`): sprites
// and ribbons.
//
// The proxy geometry gives each slot k two quads with UV (2k + u, v).
// Sprites (`proxy.wesl`): the particle's colour (texel 2k) and normalized age
// (texel 2k + 1) are in the material's data texture; the surface is the raster
// sprite's: particle colour times the (flipbook) sprite texture.
// Ribbons (`ribbon_proxy.wesl`): a slot is one ribbon sub-segment, u running
// along it; texels 3k and 3k + 1 hold the colour at its ends, texel 3k + 2 the
// ribbon U at its ends; the surface is the raster ribbon's: colour times the
// texture at (fract(U), v), or the built-in soft edges.
// Both through the same `fx_render::particle_surface` functions the raster
// shaders use.
//
// params[0]: flipbook columns, rows, sub-UV blending (0/1), flags
// params[1]: radiance per unit of raster colour (1 / camera exposure), data texture width
// textures:  0 sprite / ribbon texture (ribbons: none = built-in look), 1 particle data

import bevy_solari::scene::raytracing_material::{RaytracingHitInput, RaytracingSurface, sample_material_texture, textures};
import fx_render::particle_surface::{flipbook, sprite_texel, ribbon_builtin_texel};

// Unlit or additive: the surface emits its colour (and lights the scene).
const FLAG_EMISSIVE = 1u;
// Additive blending: emission scales with opacity, as the raster blend does.
const FLAG_ADDITIVE = 2u;
// A ribbon proxy (three texels per slot).
const FLAG_RIBBON = 4u;

fn texel_at(id: u32, t: u32, width: u32) -> vec4<f32> {
    return textureLoad(textures[id], vec2<u32>(t % width, t / width), 0);
}

fn raytracing_hit(input: RaytracingHitInput) -> RaytracingSurface {
    var surface = input.fallback;
    let data_id = input.texture_ids[1];
    if data_id == 0xFFFFFFFFu {
        surface.alpha = 0.0;
        return surface;
    }
    let slot = u32(max(floor(input.uv.x * 0.5), 0.0));
    let quad_uv = clamp(vec2<f32>(input.uv.x - 2.0 * f32(slot), input.uv.y), vec2<f32>(0.0), vec2<f32>(1.0));
    let width = max(u32(input.params[1].y), 1u);
    let flags = u32(input.params[0].w);

    var texel: vec4<f32>;
    // LOD weight: how many particles this traced quad stands for (proxy.wesl).
    var weight = 1.0;
    if (flags & FLAG_RIBBON) != 0u {
        let t = 3u * slot;
        let color = mix(texel_at(data_id, t, width), texel_at(data_id, t + 1u, width), quad_uv.x);
        let u = texel_at(data_id, t + 2u, width);
        var tex = ribbon_builtin_texel(quad_uv.y);
        if input.texture_ids[0] != 0xFFFFFFFFu {
            tex = sample_material_texture(input, 0u, vec2<f32>(fract(mix(u.x, u.y, quad_uv.x)), quad_uv.y));
        }
        texel = sprite_texel(color, tex);
    } else {
        let t = 2u * slot;
        let color = texel_at(data_id, t, width);
        let extra = texel_at(data_id, t + 1u, width);
        weight = max(extra.y, 1.0);
        let frame = flipbook(extra.x, quad_uv, input.params[0].xy, input.params[0].z > 0.5);
        let tex = mix(
            sample_material_texture(input, 0u, frame.uv0),
            sample_material_texture(input, 0u, frame.uv1),
            frame.blend,
        );
        texel = sprite_texel(color, tex);
    }

    // Opacity is stochastic (as Solari's transparency pass treats blended
    // surfaces): a hit is kept with probability `alpha`, so shadows,
    // reflections, light samples and what shows behind a sprite match its
    // blend on average (soft smoke shadows, no hard cut-out discs). The draw
    // depends only on the hit point and the frame time, so the alpha test
    // and the shading of one hit agree. (Solari's alpha test and its light
    // sampling both call this with a zero view direction: no special case.)
    // Additive sprites occlude too (by their alpha), unlike in raster: Solari
    // stops shadow and path rays alike, so a surface reflections can see also
    // casts shadows. (Kept 4x less often but 4x brighter, they doubled the
    // per-frame noise for 3% more light past them.)
    let a1 = clamp(texel.a, 0.0, 1.0);
    // A quad standing for `weight` particles covers as `weight` layers of
    // them would, and emits `weight` times their light over that coverage.
    let a = 1.0 - pow(1.0 - a1, weight);
    let gain = select(1.0, weight * a1 / a, a > 0.0);
    let kept = a > 1.0 / 255.0 && hit_random(input.world_position, input.time) < a;
    surface.alpha = select(0.0, 1.0, kept);
    if (flags & FLAG_EMISSIVE) != 0u {
        // Kept hits emit the full colour: on average colour x alpha, as the
        // raster blend adds (additive) or covers (alpha) it.
        surface.base_color = vec3<f32>(0.0);
        surface.emissive = texel.rgb * (input.params[1].x * gain);
    } else {
        surface.base_color = texel.rgb;
        surface.emissive = vec3<f32>(0.0);
    }
    return surface;
}

// Uniform [0, 1) from the hit point (quantized to 0.5 mm) and the frame time.
fn hit_random(p: vec3<f32>, time: f32) -> f32 {
    let q = vec3<i32>(floor(p * 2000.0));
    var h = (bitcast<u32>(q.x) * 0x8da6b343u) ^ (bitcast<u32>(q.y) * 0xd8163841u) ^ (bitcast<u32>(q.z) * 0xcb1ab31fu);
    h ^= (bitcast<u32>(time) * 0x9e3779b9u);
    h = pcg(h);
    h = pcg(h);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
