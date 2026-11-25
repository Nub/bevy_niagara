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
        let frame = flipbook(extra.x, quad_uv, input.params[0].xy, input.params[0].z > 0.5);
        let tex = mix(
            sample_material_texture(input, 0u, frame.uv0),
            sample_material_texture(input, 0u, frame.uv1),
            frame.blend,
        );
        texel = sprite_texel(color, tex);
    }

    surface.alpha = texel.a;
    if (flags & FLAG_EMISSIVE) != 0u {
        // Sampled as a light (no view direction), alpha cuts nothing away:
        // weight the emission by coverage so the soft edge emits less.
        let as_light = all(input.view_direction == vec3<f32>(0.0));
        let coverage = select(1.0, texel.a, (flags & FLAG_ADDITIVE) != 0u || as_light);
        surface.base_color = vec3<f32>(0.0);
        surface.emissive = texel.rgb * coverage * input.params[1].x;
    } else {
        surface.base_color = texel.rgb;
        surface.emissive = vec3<f32>(0.0);
    }
    return surface;
}
