//! Opt-in particle counts for tools (the viewer's stats overlay): each
//! emitter's living-particle count is copied to a small staging buffer after
//! the frame's simulation and mapped asynchronously, so it arrives a few
//! frames late and never stalls the GPU. The sim itself still never reads
//! counts back. Add [`FxStatsPlugin`] after [`crate::FxPlugin`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bevy::core_pipeline::schedule::camera_driver;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph};
use bevy::render::{Render, RenderApp, RenderSystems};
use fx_gpu::wgpu;

use crate::FxState;

/// One emitter's count, as last read back.
#[derive(Clone, Debug, Default)]
pub struct EmitterStats {
    pub name: String,
    pub alive: u32,
    pub capacity: u32,
}

/// One effect instance, keyed by its main-world entity in [`FxStatsData`].
#[derive(Clone, Debug, Default)]
pub struct EffectStats {
    pub name: String,
    pub emitters: Vec<EmitterStats>,
    /// GPU memory held by its particle and state buffers.
    pub gpu_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct FxStatsData {
    pub effects: HashMap<Entity, EffectStats>,
    /// Readbacks completed so far.
    pub readbacks: u64,
}

/// Shared between the main and render worlds; read it from the main world.
#[derive(Resource, Clone, Default)]
pub struct FxStats(pub Arc<Mutex<FxStatsData>>);

impl FxStats {
    pub fn snapshot(&self) -> FxStatsData {
        self.0.lock().unwrap().clone()
    }
}

pub struct FxStatsPlugin;

impl Plugin for FxStatsPlugin {
    fn build(&self, app: &mut App) {
        let stats = FxStats::default();
        app.insert_resource(stats.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(stats)
            .init_resource::<CountReadback>()
            .add_systems(
                RenderGraph,
                copy_counts.after(crate::simulate).before(camera_driver),
            )
            .add_systems(Render, map_counts.in_set(RenderSystems::Cleanup));
    }
}

#[derive(Resource, Default)]
struct CountReadback {
    buffer: Option<wgpu::Buffer>,
    /// What the copied words mean, in order.
    layout: Vec<(Entity, EffectStats)>,
    bytes: u64,
    copied: bool,
    busy: Arc<AtomicBool>,
}

fn copy_counts(
    mut ctx: RenderContext,
    state: Option<Res<FxState>>,
    device: Res<RenderDevice>,
    mut rb: ResMut<CountReadback>,
    stats: Res<FxStats>,
) {
    let Some(state) = state else { return };
    if rb.busy.load(Ordering::Acquire) || rb.copied {
        return;
    }
    let n: usize = state.effects.values().map(|e| e.gpu.emitters.len()).sum();
    if n == 0 {
        let mut s = stats.0.lock().unwrap();
        s.effects.clear();
        return;
    }
    let bytes = (n * 4) as u64;
    if rb.buffer.as_ref().is_none_or(|b| b.size() < bytes) {
        rb.buffer = Some(device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx stats readback"),
            size: bytes.next_power_of_two().max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    let rb = &mut *rb;
    let buffer = rb.buffer.as_ref().unwrap();
    let encoder = ctx.command_encoder();
    rb.layout.clear();
    let mut off = 0;
    for (entity, inst) in &state.effects {
        let mut es = EffectStats {
            name: inst.system.name.clone(),
            emitters: Vec::new(),
            gpu_bytes: inst.gpu.gpu_bytes(),
        };
        for em in &inst.gpu.emitters {
            encoder.copy_buffer_to_buffer(&em.state.buffer, em.state.offset + fx_gpu::STATE_COUNT * 4, buffer, off, 4);
            off += 4;
            es.emitters.push(EmitterStats {
                name: em.name.clone(),
                alive: 0,
                capacity: em.capacity,
            });
        }
        rb.layout.push((*entity, es));
    }
    rb.bytes = bytes;
    rb.copied = true;
}

/// After the frame's submit: map what `copy_counts` copied.
fn map_counts(mut rb: ResMut<CountReadback>, stats: Res<FxStats>) {
    if !rb.copied {
        return;
    }
    rb.copied = false;
    rb.busy.store(true, Ordering::Release);
    let buffer = rb.buffer.clone().unwrap();
    let layout = std::mem::take(&mut rb.layout);
    let busy = rb.busy.clone();
    let shared = stats.0.clone();
    let bytes = rb.bytes;
    buffer.clone().slice(..bytes).map_async(wgpu::MapMode::Read, move |res| {
        if res.is_ok() {
            let words: Vec<u32> = {
                let data = buffer.slice(..bytes).get_mapped_range().unwrap();
                bytemuck_words(&data)
            };
            buffer.unmap();
            let mut it = words.into_iter();
            let mut s = shared.lock().unwrap();
            s.effects.clear();
            for (e, mut es) in layout {
                for em in &mut es.emitters {
                    em.alive = it.next().unwrap_or(0).min(em.capacity);
                }
                s.effects.insert(e, es);
            }
            s.readbacks += 1;
        }
        busy.store(false, Ordering::Release);
    });
}

fn bytemuck_words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
