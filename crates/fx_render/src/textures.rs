//! Built-in sprite textures used when a sprite renderer names no texture:
//! a soft round dot, or for a flipbook grid a soft puff per cell whose
//! billows change from frame to frame, so sub-UV animation is visible.

use bevy::platform::collections::HashMap;
use fx_gpu::wgpu;

const CELL: u32 = 64;

pub struct Builtin {
    textures: HashMap<(u32, u32), (wgpu::Texture, wgpu::TextureView)>,
    sampler: wgpu::Sampler,
}

fn hash(x: u32) -> u32 {
    fx_core::sim::rng::pcg(x)
}

/// Smooth value noise in [0, 1].
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - xi as f32, y - yi as f32);
    let v = |dx: i32, dy: i32| {
        let h = hash(hash(hash(seed) ^ (xi + dx) as u32) ^ (yi + dy) as u32);
        (h >> 8) as f32 / 16_777_216.0
    };
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let a = v(0, 0) + (v(1, 0) - v(0, 0)) * sx;
    let b = v(0, 1) + (v(1, 1) - v(0, 1)) * sx;
    a + (b - a) * sy
}

/// RGBA8 pixels of a `cols` × `rows` grid of cells.
pub fn pixels(cols: u32, rows: u32) -> Vec<u8> {
    let (w, h) = (cols * CELL, rows * CELL);
    let flipbook = cols * rows > 1;
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let frame = (y / CELL) * cols + x / CELL;
            let u = ((x % CELL) as f32 + 0.5) / CELL as f32 * 2.0 - 1.0;
            let v = ((y % CELL) as f32 + 0.5) / CELL as f32 * 2.0 - 1.0;
            let r2 = u * u + v * v;
            let mut a = (1.0 - r2).max(0.0);
            a *= a;
            let mut l = 1.0;
            if flipbook {
                let n = 0.5 * value_noise(u * 3.0 + 7.0, v * 3.0 + 7.0, frame * 2 + 1)
                    + 0.5 * value_noise(u * 6.0 + 3.0, v * 6.0 + 3.0, frame * 2 + 2);
                a *= (0.35 + 1.1 * n).min(1.0);
                l = 0.75 + 0.25 * n;
            }
            let i = ((y * w + x) * 4) as usize;
            let c = (l * 255.0).round() as u8;
            px[i..i + 4].copy_from_slice(&[c, c, c, (a * 255.0).round() as u8]);
        }
    }
    px
}

impl Builtin {
    pub fn new(device: &wgpu::Device) -> Self {
        Builtin {
            textures: HashMap::default(),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fx builtin sprite"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    pub fn get(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cols: u32,
        rows: u32,
    ) -> (&wgpu::TextureView, &wgpu::Sampler) {
        let (cols, rows) = (cols.max(1), rows.max(1));
        let (_, view) = self.textures.entry((cols, rows)).or_insert_with(|| {
            let size = wgpu::Extent3d {
                width: cols * CELL,
                height: rows * CELL,
                depth_or_array_layers: 1,
            };
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fx builtin sprite"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                tex.as_image_copy(),
                &pixels(cols, rows),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.width * 4),
                    rows_per_image: None,
                },
                size,
            );
            let view = tex.create_view(&Default::default());
            (tex, view)
        });
        (view, &self.sampler)
    }
}
