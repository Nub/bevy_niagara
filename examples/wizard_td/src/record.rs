//! The scripted headless run: the bot plays on a fixed clock, every frame is
//! read back and streamed to ffmpeg (DIR/wizard_td.mp4), with three stills.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use fx_render::{FxClock, FxPlugin, FxPoolStats};

use crate::{Mode, Phase, Run, RunMode, SIM_HZ};

pub fn headless_plugins(app: &mut App, assets: AssetPlugin) {
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(assets)
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::winit::WinitPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        FxPlugin::default(),
    ))
    .insert_resource(FxClock::manual(1.0 / SIM_HZ as f32));
}

#[derive(Resource)]
struct Target(Handle<Image>);

pub fn attach_target(commands: &mut Commands, cam: Entity, (w, h): (u32, u32), images: &mut Assets<Image>) {
    let mut img = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(img);
    commands.entity(cam).insert(RenderTarget::from(handle.clone()));
    commands.insert_resource(Target(handle));
}

pub fn add(app: &mut App) {
    app.init_resource::<Recorder>().add_systems(Update, (clock, record).chain());
}

/// Frames read back (by index), written in order as they complete.
#[derive(Default)]
struct Sink {
    pending: BTreeMap<u32, Vec<u8>>,
    next: u32,
    ffmpeg: Option<Child>,
    stills: Vec<u32>,
    size: (u32, u32),
    out: std::path::PathBuf,
    error: Option<String>,
}

#[derive(Resource, Default)]
struct Recorder {
    requested: u32,
    sink: Arc<Mutex<Sink>>,
    max_live: u64,
}

/// Two simulation steps per 30 fps frame once play has started.
fn clock(mut clock: ResMut<FxClock>, mode: Res<RunMode>, run: Res<Run>) {
    let Mode::Record { fps, .. } = mode.0 else { return };
    // While loading, single steps so warm-up effects build and compile.
    clock.request_steps(if run.phase == Phase::Loading { 1 } else { (SIM_HZ / fps).max(1) });
}

fn unpad(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4) as usize;
    let padded = row.div_ceil(256) * 256;
    if data.len() == row * h as usize {
        return data.to_vec();
    }
    (0..h as usize).flat_map(|y| data[y * padded..y * padded + row].iter().copied()).collect()
}

fn write_png(path: &Path, (w, h): (u32, u32), rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())
}

impl Sink {
    fn push(&mut self, i: u32, data: Vec<u8>) {
        self.pending.insert(i, data);
        while let Some(f) = self.pending.remove(&self.next) {
            let (w, h) = self.size;
            let f = unpad(&f, w, h);
            if let Some(k) = self.stills.iter().position(|&s| s == self.next) {
                let p = self.out.join(format!("wizard_td_{}.png", k + 1));
                if let Err(e) = write_png(&p, self.size, &f) {
                    self.error.get_or_insert(e);
                }
            }
            if let Some(stdin) = self.ffmpeg.as_mut().and_then(|c| c.stdin.as_mut())
                && let Err(e) = stdin.write_all(&f)
            {
                self.error.get_or_insert(format!("ffmpeg: {e}"));
            }
            self.next += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record(
    mut commands: Commands,
    mut rec: ResMut<Recorder>,
    mode: Res<RunMode>,
    target: Res<Target>,
    run: Res<Run>,
    pool: Res<FxPoolStats>,
    mut exit: MessageWriter<AppExit>,
) {
    let Mode::Record { out, seconds, fps, size } = &mode.0 else { return };
    if run.phase == Phase::Loading {
        return;
    }
    let n = (*seconds * *fps as f32).round() as u32;
    if rec.requested == 0 {
        if let Err(e) = std::fs::create_dir_all(out) {
            eprintln!("wizard_td: {}: {e}", out.display());
            exit.write(AppExit::error());
            return;
        }
        let mp4 = out.join("wizard_td.mp4");
        let child = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{}x{}", size.0, size.1))
            .args(["-r", &fps.to_string(), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart"])
            .arg(&mp4)
            .stdin(Stdio::piped())
            .spawn();
        let mut s = rec.sink.lock().unwrap();
        match child {
            Ok(c) => s.ffmpeg = Some(c),
            Err(e) => {
                eprintln!("wizard_td: ffmpeg: {e}");
                exit.write(AppExit::error());
                return;
            }
        }
        s.size = *size;
        s.out = out.clone();
        s.stills = vec![n / 4, n / 2, 3 * n / 4];
    }
    rec.max_live = rec.max_live.max(pool.get().live);
    if rec.requested < n {
        let i = rec.requested;
        let sink = rec.sink.clone();
        commands
            .spawn(Readback::texture(target.0.clone()))
            .observe(move |ev: On<ReadbackComplete>, mut commands: Commands| {
                sink.lock().unwrap().push(i, ev.data.clone());
                commands.entity(ev.entity).despawn();
            });
        rec.requested += 1;
        if rec.requested % (*fps * 10) == 0 {
            println!(
                "{:.0} s: wave {}, tower {:.0}, score {}, {} kills, {} effects live",
                run.time, run.wave, run.tower_hp, run.score, run.kills, pool.get().live
            );
        }
    }
    let mut s = rec.sink.lock().unwrap();
    if s.next < n {
        return;
    }
    let mut result = s.error.take().map(Err).unwrap_or(Ok(()));
    if let Some(mut c) = s.ffmpeg.take() {
        drop(c.stdin.take());
        match c.wait() {
            Ok(st) if !st.success() => result = Err(format!("ffmpeg failed: {st}")),
            Err(e) => result = Err(e.to_string()),
            _ => {}
        }
    }
    match result {
        Ok(()) => {
            let p = pool.get();
            println!(
                "{}: {n} frames at {fps} fps\nwave {}, tower {:.0}, score {}, {} kills, {} casts\npool: {} programs, {} instances built, {} reused, up to {} live",
                out.join("wizard_td.mp4").display(),
                run.wave,
                run.tower_hp,
                run.score,
                run.kills,
                run.casts,
                p.programs,
                p.instances_created,
                p.instances_reused,
                rec.max_live
            );
            exit.write(AppExit::Success);
        }
        Err(e) => {
            eprintln!("wizard_td: {e}");
            exit.write(AppExit::error());
        }
    }
}
