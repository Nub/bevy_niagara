//! `fxbench`: GPU time of a renderer path at scale, headless at 1080p.
//!
//! Spawns `--particles N` (default 1M) particles drawn by `--renderer` in one burst that live for the
//! whole run, simulates one fixed step per frame and reads Bevy's render
//! diagnostics (GPU timestamp queries): `fx_sim` (spawn/update/compaction),
//! `fx_cull_sort` (frustum culling, depth sort, indirect args) and the main
//! transparent pass (the sprite, ribbon or mesh draw); ribbons add
//! `fx_ribbon_build` (sort by ribbon and link order is in `fx_cull_sort`),
//! lights add `fx_lights` (cull, nearest-first sort, gather) and light a
//! ground plane in the main opaque pass. Prints medians over the measured frames.
//!
//! Ribbon particles join 255 ribbons at random (a worst case for overdraw);
//! mesh particles are lit opaque cubes; light particles are 256 point lights
//! (the cap) out of N, with additive sprites.
//!
//! ```text
//! fxbench [--renderer sprite|ribbon|mesh|light] [--particles N] [--sort none|depth] [--blend additive|alpha] [--frames N] [--size WxH]
//! ```

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::camera::{Hdr, RenderTarget};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use fx_render::{FxClock, FxPlugin, FxReady, ParticleEffect};

struct Opts {
    renderer: String,
    particles: u32,
    sort: String,
    blend: String,
    frames: u32,
    width: u32,
    height: u32,
}

fn effect_source(o: &Opts) -> String {
    let sort = if o.sort == "depth" {
        "ViewDepth"
    } else {
        "None"
    };
    let blend = if o.blend == "alpha" {
        "Alpha"
    } else {
        "Additive"
    };
    let (renderers, extra_spawn) = match o.renderer.as_str() {
        "sprite" => (
            format!("Sprite((blend_mode: {blend}, sort_mode: {sort}))"),
            String::new(),
        ),
        "ribbon" => (
            format!("Ribbon((blend_mode: {blend}))"),
            r#"(module: "Initialize Ribbon", inputs: { "Ribbon Width": Float(0.01), "Ribbon ID": RandomRange(Int(0), Int(254)) }),"#
                .to_string(),
        ),
        "mesh" => (
            format!("Mesh((mesh: \"cube\", sort_mode: {sort}))"),
            r#"(module: "Initialize Mesh", inputs: { "Mesh Scale": Vec3((0.02, 0.02, 0.02)), "Mesh Rotation": RandomRange(Vec3((0.0, 0.0, 0.0)), Vec3((360.0, 360.0, 360.0))) }),"#
                .to_string(),
        ),
        "light" => (
            format!(
                "Sprite((blend_mode: {blend}, sort_mode: {sort})), Light((max_lights: 256, intensity: 2000.0))"
            ),
            r#"(module: "Initialize Light", inputs: { "Light Radius": Float(0.6) }),"#.to_string(),
        ),
        r => panic!("--renderer is sprite, ribbon, mesh or light, not {r}"),
    };
    format!(
        r#"(
    name: "Bench",
    emitters: [(
        name: "Sparks",
        max_particles: {n},
        emitter_update: [
            (module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int({n}) }}),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{
                "Lifetime": Float(1000.0),
                "Position": RandomRange(Vec3((-2.0, 0.0, -2.0)), Vec3((2.0, 4.0, 2.0))),
                "Color": Color((1.0, 0.6, 0.2, 1.0)),
                "Sprite Size": RandomRange(Vec2((0.01, 0.01)), Vec2((0.03, 0.03))),
            }}),
            (module: "Add Velocity", inputs: {{ "Velocity": RandomRange(Vec3((-0.2, -0.2, -0.2)), Vec3((0.2, 0.2, 0.2))) }}),
            {extra_spawn}
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Gravity Force", inputs: {{ "Gravity": Vec3((0.0, -0.05, 0.0)) }}),
            (module: "Drag", inputs: {{ "Drag": Float(0.1) }}),
            (module: "Solve Forces and Velocity"),
            (module: "Scale Color", inputs: {{
                "Scale Alpha": CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)])),
            }}),
        ],
        renderers: [{renderers}],
    )],
)"#,
        n = o.particles
    )
}

#[derive(Resource)]
struct Bench {
    frames: u32,
    warm: u32,
    measured: u32,
    samples: Arc<Mutex<Vec<(String, Vec<f64>)>>>,
}

fn main() {
    let mut o = Opts {
        renderer: "sprite".into(),
        particles: 1_000_000,
        sort: "none".into(),
        blend: "additive".into(),
        frames: 240,
        width: 1920,
        height: 1080,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_default();
        match a.as_str() {
            "--renderer" => o.renderer = v,
            "--particles" => o.particles = v.parse().expect("--particles N"),
            "--sort" => o.sort = v,
            "--blend" => o.blend = v,
            "--frames" => o.frames = v.parse().expect("--frames N"),
            "--size" => {
                let (w, h) = v.split_once('x').expect("--size WxH");
                o.width = w.parse().unwrap();
                o.height = h.parse().unwrap();
            }
            _ => panic!("unknown option {a}"),
        }
    }
    let src = effect_source(&o);
    let checked = fx_core::check_source("bench.fx.ron", &src);
    for d in &checked.diagnostics {
        eprintln!("{d}");
    }
    let system = checked.cooked.expect("bench effect compiles");
    let samples = Arc::new(Mutex::new(Vec::new()));
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::winit::WinitPlugin>(),
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        RenderDiagnosticsPlugin,
        FxPlugin::default(),
    ))
    .insert_resource(FxClock::manual(1.0 / 60.0))
    .insert_resource(Bench {
        frames: o.frames,
        warm: 0,
        measured: 0,
        samples: samples.clone(),
    });
    let (w, h) = (o.width, o.height);
    app.add_systems(
        Startup,
        move |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
            let mut img = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
            img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            commands.spawn((
                Camera3d::default(),
                RenderTarget::from(images.add(img)),
                Hdr,
                DepthPrepass,
                Msaa::Off,
                Transform::from_xyz(0.0, 2.0, 7.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
            ));
            commands.spawn((ParticleEffect::new(system.clone(), 1), Transform::IDENTITY));
        },
    )
    .add_systems(Update, measure);
    app.run();

    let samples = samples.lock().unwrap();
    println!(
        "fxbench: {} particles, sort {}, blend {}, {}x{}, {} frames measured",
        o.particles,
        o.sort,
        o.blend,
        o.width,
        o.height,
        samples.first().map_or(0, |s| s.1.len())
    );
    for (path, v) in samples.iter() {
        let mut v = v.clone();
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        println!(
            "{path}: median {med:.3} ms (min {:.3}, max {:.3})",
            v[0],
            v[v.len() - 1]
        );
    }
}

fn measure(
    mut bench: ResMut<Bench>,
    mut clock: ResMut<FxClock>,
    ready: Res<FxReady>,
    store: Res<DiagnosticsStore>,
    mut exit: MessageWriter<AppExit>,
) {
    if !ready.get() {
        return;
    }
    clock.request_steps(1);
    // Let the burst spawn and timings settle before measuring.
    if bench.warm < 60 {
        bench.warm += 1;
        return;
    }
    let mut samples = bench.samples.lock().unwrap();
    for d in store.iter() {
        let path = d.path().as_str();
        if !path.ends_with("elapsed_gpu") {
            continue;
        }
        let Some(v) = d.measurement().map(|m| m.value) else {
            continue;
        };
        match samples.iter_mut().find(|(p, _)| p == path) {
            Some((_, list)) => list.push(v),
            None => samples.push((path.to_string(), vec![v])),
        }
    }
    drop(samples);
    bench.measured += 1;
    if bench.measured >= bench.frames {
        exit.write(AppExit::Success);
    }
}
