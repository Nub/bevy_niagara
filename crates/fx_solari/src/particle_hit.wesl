// Solari hit function of ray-traced particle sprites (`ParticleRtMaterial`).
//
// The proxy geometry (`proxy.wesl`) gives each particle slot k two quads with
// UV (2k + u, v), and writes the particle's colour (texel 2k) and normalized
// age (texel 2k + 1) to the material's data texture. The surface is the raster
// sprite's: particle colour times the (flipbook) sprite texture, through the
// same `fx_render::particle_surface` functions `sprite.wesl` uses.
//
// params[0]: flipbook columns, rows, sub-UV blending (0/1), flags
// params[1]: radiance per unit of raster colour (1 / camera exposure), data texture width
// textures:  0 sprite texture, 1 particle data

import bevy_solari::scene::raytracing_material::{RaytracingHitInput, RaytracingSurface, sample_material_texture, textures};
import fx_render::particle_surface::{flipbook, sprite_texel};

// Unlit or additive: the sprite emits its colour (and lights the scene).
const FLAG_EMISSIVE = 1u;
// Additive blending: emission scales with opacity, as the raster blend does.
const FLAG_ADDITIVE = 2u;

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
    let t = 2u * slot;
    let color = textureLoad(textures[data_id], vec2<u32>(t % width, t / width), 0);
    let extra = textureLoad(textures[data_id], vec2<u32>((t + 1u) % width, (t + 1u) / width), 0);

    let frame = flipbook(extra.x, quad_uv, input.params[0].xy, input.params[0].z > 0.5);
    let tex = mix(
        sample_material_texture(input, 0u, frame.uv0),
        sample_material_texture(input, 0u, frame.uv1),
        frame.blend,
    );
    let texel = sprite_texel(color, tex);
    let flags = u32(input.params[0].w);

    surface.alpha = texel.a;
    if (flags & FLAG_EMISSIVE) != 0u {
        // Sampled as a light (no view direction), alpha cuts nothing away:
        // weight the emission by coverage so the sprite's soft edge emits less.
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
