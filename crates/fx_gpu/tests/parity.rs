//! CPU/GPU parity: the same sample effect and seed give the same particles
//! on the GPU as fx_core's CPU reference, within float tolerance.
//!
//! Integer work (random hashes, spawn counts, UniqueIDs, kills) must match
//! exactly; float attributes may differ by the GPU's FMA contraction and its
//! exp/sin/cos approximations. One inline system covers the extended
//! attributes (Ribbon, Mesh, Light, material parameters).

use std::path::PathBuf;

use fx_core::sim::{Particle, Simulation};
use fx_gpu::{GpuEffect, WORDS_PER_PARTICLE, WORDS_PER_PARTICLE_EXTENDED};

const DT: f32 = 1.0 / 60.0;

fn sample(name: &str) -> fx_core::CookedSystem {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../effects/samples")
        .join(name);
    let c = fx_core::check_file(&path).unwrap();
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("{name}: {errs:?}"))
}

const LIBRARY: &[&str] = &[
    "campfire.fx.ron",
    "sparks_burst.fx.ron",
    "rising_smoke.fx.ron",
    "magic_swirl.fx.ron",
    "sword_trail.fx.ron",
    "debris.fx.ron",
];

fn library(name: &str) -> fx_core::CookedSystem {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../effects/library")
        .join(name);
    fx_core::check_file(&path).unwrap().cooked.unwrap_or_else(|| panic!("{name} does not cook"))
}

fn fields(p: &Particle) -> Vec<(&'static str, f32)> {
    let mut v = Vec::new();
    let mut put = |n: &'static str, xs: &[f32]| xs.iter().for_each(|x| v.push((n, *x)));
    put("position", &p.position.to_array());
    put("velocity", &p.velocity.to_array());
    put("color", &p.color.to_array());
    put("sprite_size", &p.sprite_size.to_array());
    put("sprite_rotation", &[p.sprite_rotation]);
    put("mass", &[p.mass]);
    put("lifetime", &[p.lifetime]);
    put("age", &[p.age]);
    put("normalized_age", &[p.normalized_age]);
    put("initial_color", &p.initial_color.to_array());
    put("initial_sprite_size", &p.initial_sprite_size.to_array());
    put("force", &p.force.to_array());
    put("drag", &[p.drag]);
    put("scale", &p.scale.to_array());
    put("mesh_orientation", &p.mesh_orientation.to_array());
    put("ribbon_width", &[p.ribbon_width]);
    put("ribbon_twist", &[p.ribbon_twist]);
    put("ribbon_link_order", &[p.ribbon_link_order]);
    put("light_radius", &[p.light_radius]);
    put(
        "dynamic_material_parameter",
        &p.dynamic_material_parameter.to_array(),
    );
    put("sub_image_index", &[p.sub_image_index]);
    put("initial_scale", &p.initial_scale.to_array());
    put("initial_ribbon_width", &[p.initial_ribbon_width]);
    v
}

/// Largest absolute and relative difference seen.
#[derive(Default, Debug)]
struct Diff {
    max_abs: f32,
    worst: String,
}

fn compare(name: &str, step: u32, cpu: &[Particle], gpu: &[Particle], diff: &mut Diff) {
    let mut cpu: Vec<_> = cpu.to_vec();
    let mut gpu: Vec<_> = gpu.to_vec();
    cpu.sort_by_key(|p| p.unique_id);
    gpu.sort_by_key(|p| p.unique_id);
    let ids = |v: &[Particle]| v.iter().map(|p| p.unique_id).collect::<Vec<_>>();
    assert_eq!(
        ids(&cpu),
        ids(&gpu),
        "{name} step {step}: living particle ids differ ({} on CPU, {} on GPU)",
        cpu.len(),
        gpu.len()
    );
    for (a, b) in cpu.iter().zip(&gpu) {
        assert_eq!(
            a.ribbon_id, b.ribbon_id,
            "{name} step {step} particle {}: ribbon_id",
            a.unique_id
        );
        for ((field, x), (_, y)) in fields(a).into_iter().zip(fields(b)) {
            let d = (x - y).abs();
            let tol = 1e-3 * (1.0 + x.abs());
            assert!(
                d <= tol,
                "{name} step {step} particle {}: {field} CPU {x} GPU {y}",
                a.unique_id
            );
            if d > diff.max_abs {
                diff.max_abs = d;
                diff.worst = format!("{field} of particle {} at step {step}", a.unique_id);
            }
        }
    }
}

/// Every Ribbon/Mesh/Light/material attribute and the modules that write
/// them, plus a panning curl field; drawn by Mesh and Light renderers, so the
/// emitter stores the extended attributes.
const EXTENDED: &str = r#"(
    name: "Extended",
    emitters: [(
        name: "E",
        max_particles: 2000,
        emitter_update: [
            (module: "Spawn Rate", inputs: { "Spawn Rate": Float(300.0) }),
            (module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(50) }),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {
                "Lifetime": RandomRange(Float(0.5), Float(1.5)),
                "Position": RandomRange(Vec3((-1.0, -1.0, -1.0)), Vec3((1.0, 1.0, 1.0))),
            }),
            (module: "Initialize Ribbon", inputs: {
                "Ribbon ID": RandomRange(Int(0), Int(7)),
                "Ribbon Width": RandomRange(Float(0.05), Float(0.2)),
                "Ribbon Twist": Float(30.0),
                "Ribbon Link Order": Link(Float, "Engine.Time"),
            }),
            (module: "Initialize Mesh", inputs: {
                "Mesh Scale": RandomRange(Vec3((0.5, 0.5, 0.5)), Vec3((2.0, 2.0, 2.0))),
                "Mesh Rotation": RandomRange(Vec3((-180.0, -180.0, -180.0)), Vec3((180.0, 180.0, 180.0))),
            }),
            (module: "Add Velocity", inputs: { "Velocity": Link(Vec3, "Particles.Scale") }),
            (module: "Initialize Light", inputs: { "Light Radius": RandomRange(Float(0.5), Float(3.0)) }),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Update Mesh Orientation", inputs: { "Rotation Rate": Vec3((90.0, -45.0, 200.0)) }),
            (module: "Dynamic Material Parameters", inputs: {
                "Dynamic Material Parameter": CurveOverAge(Vec4([(0.0, (0.0, 0.0, 0.0, 1.0)), (1.0, (1.0, 2.0, 3.0, 0.0))])),
            }),
            (module: "Curl Noise Force", inputs: {
                "Noise Strength": Float(2.0),
                "Pan Noise Field": Vec3((0.5, 1.0, -0.25)),
            }),
            (module: "Initialize Light", inputs: { "Light Radius": Link(Float, "Particles.NormalizedAge") }),
            (module: "Solve Forces and Velocity"),
        ],
        renderers: [Mesh((mesh: "sphere")), Light],
    )],
)"#;

fn extended_system() -> fx_core::CookedSystem {
    let c = fx_core::check_source("extended.fx.ron", EXTENDED);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("extended: {errs:?}"))
}

/// Niagara's common modules (forces, every Shape Location shape, rotation,
/// SubUV modes, scaling, Kill Particles in Volume): one emitter per shape,
/// with the other modules spread over them. The owner moves, so Inherit
/// Velocity has something to inherit.
fn common_modules_system() -> fx_core::CookedSystem {
    let shapes = ["Sphere", "Box", "Cylinder", "Torus", "Cone", "Ring"];
    let modes = ["Linear", "Random", "InfiniteLoop"];
    let emitters: Vec<String> = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let surface = i % 2 == 1;
            let mode = modes[i % 3];
            format!(
                r#"(
        name: "{shape}",
        max_particles: 1500,
        emitter_update: [(module: "Spawn Rate", inputs: {{ "Spawn Rate": Float(250.0) }})],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {{ "Lifetime": RandomRange(Float(0.5), Float(1.5)), "Mass": RandomRange(Float(0.5), Float(2.0)) }}),
            (module: "Shape Location", inputs: {{
                "Shape Primitive": {shape}, "Surface Only": Bool({surface}), "Offset": Vec3((0.25, 0.5, -0.25)),
                "Sphere Radius": Float(1.5), "Box Size": Vec3((1.0, 2.0, 3.0)), "Cylinder Height": Float(2.0),
                "Cylinder Radius": RandomRange(Float(0.5), Float(1.0)), "Large Radius": Float(1.5), "Handle Radius": Float(0.3),
                "Cone Length": Float(2.0), "Cone Angle": Float(40.0), "Ring Radius": Float(1.2), "Disc Coverage": Float(0.5) }}),
            (module: "Inherit Velocity", inputs: {{ "Velocity Scale": Float(0.5) }}),
            (module: "Vortex Velocity", inputs: {{ "Vortex Axis": Vec3((0.2, 1.0, 0.1)), "Velocity Amount": Float(1.5) }}),
            (module: "Initialize Mesh", inputs: {{ "Mesh Scale": RandomRange(Vec3((0.5, 0.5, 0.5)), Vec3((1.5, 1.5, 1.5))) }}),
            (module: "Initial Mesh Orientation", inputs: {{ "Rotation": RandomRange(Vec3((-180.0, -180.0, -180.0)), Vec3((180.0, 180.0, 180.0))) }}),
            (module: "Initialize Ribbon", inputs: {{ "Ribbon Width": RandomRange(Float(0.05), Float(0.2)) }}),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Point Attraction Force", inputs: {{ "Attraction Strength": Float(3.0), "Attraction Radius": Float(2.5), "Falloff Exponent": Float(1.5), "Kill Radius": Float(0.05) }}),
            (module: "Vortex Force", inputs: {{ "Vortex Force Amount": Float(2.0), "Origin Pull Amount": Float(0.5) }}),
            (module: "Wind Force", inputs: {{ "Wind Velocity": Vec3((1.0, 0.0, 0.5)), "Wind Speed": Float(1.5), "Air Resistance": Float(0.3) }}),
            (module: "Linear Force", inputs: {{ "Linear Force": Vec3((0.0, 0.5, 0.0)) }}),
            (module: "Acceleration Force", inputs: {{ "Acceleration": Vec3((0.0, -1.0, 0.0)) }}),
            (module: "Point Force", inputs: {{ "Point Position": Vec3((0.5, 0.0, 0.5)), "Point Force Strength": Float(4.0), "Falloff Radius": Float(1.5), "Falloff Exponent": Float(2.0) }}),
            (module: "Limit Force", inputs: {{ "Force Limit": Float(3.0) }}),
            (module: "Solve Forces and Velocity"),
            (module: "Sprite Rotation Rate", inputs: {{ "Rotation Rate": RandomRange(Float(-90.0), Float(90.0)) }}),
            (module: "SubUV Animation", inputs: {{ "SubUV Animation Mode": {mode}, "Start Frame": Int(1), "Frames Per Second": Float(12.0) }}),
            (module: "Color", inputs: {{ "Color": CurveOverAge(Color([(0.0, (1.0, 0.5, 0.0, 1.0)), (1.0, (0.0, 0.5, 1.0, 0.0))])) }}),
            (module: "Scale Color by Speed", inputs: {{ "Velocity Threshold": Float(3.0) }}),
            (module: "Scale Sprite Size by Speed", inputs: {{ "Velocity Threshold": Float(2.0) }}),
            (module: "Scale Mesh Size", inputs: {{ "Scale Factor": CurveOverAge(Vec3([(0.0, (1.0, 1.0, 1.0)), (1.0, (0.0, 2.0, 0.5))])) }}),
            (module: "Scale Ribbon Width", inputs: {{ "Ribbon Width Scale": CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)])) }}),
            (module: "Kill Particles in Volume", inputs: {{ "Volume Shape": Box, "Box Size": Vec3((8.0, 8.0, 8.0)), "Invert Volume": Bool(true) }}),
            (module: "Kill Particles in Volume", inputs: {{ "Volume Center": Vec3((3.0, 0.0, 0.0)), "Sphere Radius": Float(1.0) }}),
        ],
        renderers: [Sprite((sub_image_size: (4, 4)))],
    )"#
            )
        })
        .collect();
    let src = format!(
        "(name: \"Common modules\", emitters: [{}])",
        emitters.join(",\n")
    );
    let c = fx_core::check_source("common.fx.ron", &src);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    c.cooked.unwrap_or_else(|| panic!("common modules: {errs:?}"))
}

/// Dynamic inputs of the types the sample does not use (Vec4, Vec2 and Int
/// families, Break of each vector), nested, on extended attributes.
const DYNAMIC_TYPES: &str = r#"(
    name: "Dynamic types",
    emitters: [(
        name: "E",
        max_particles: 400,
        emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Add Float", { "A": Float(60.0), "B": Dynamic("Random Float from List", { "Values": [Float(0.0), Float(40.0)] }) }) })],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {
                "Lifetime": Dynamic("Uniform Ranged Float", { "Minimum": Float(0.5), "Maximum": Float(1.5) }),
                "Sprite Size": Dynamic("Lerp Vec2", { "A": Vec2((0.1, 0.2)), "B": Dynamic("Uniform Ranged Vec2", {}), "Alpha": Float(0.25) }),
                "Mass": Dynamic("Break Vec2", { "Value": Dynamic("Random Range Vec2 by Uniform", { "Minimum": Vec2((1.0, 2.0)), "Maximum": Vec2((3.0, 5.0)) }), "Component": Y }),
                "Sprite Rotation": Dynamic("Break Color", { "Value": Dynamic("Random Color from List", { "Values": [Color((1.0, 2.0, 3.0, 4.0)), Color((5.0, 6.0, 7.0, 8.0))] }), "Component": B }),
            }),
            (module: "Initialize Ribbon", inputs: {
                "Ribbon ID": Dynamic("Select Int by Int", { "Index": Dynamic("Uniform Ranged Int", { "Minimum": Int(-1), "Maximum": Int(3) }), "Values": [Int(4), Int(5), Int(6)] }),
                "Ribbon Width": Dynamic("Break Vec4", { "Value": Dynamic("Make Vec4 from Floats", { "X": Float(1.0), "W": Dynamic("Uniform Ranged Float", {}) }), "Component": W }),
            }),
            (module: "Add Velocity", inputs: { "Velocity": Dynamic("Uniform Ranged Vec3", { "Minimum": Vec3((-1.0, 1.0, -1.0)), "Maximum": Vec3((1.0, 3.0, 1.0)) }) }),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Solve Forces and Velocity"),
            (module: "Dynamic Material Parameters", inputs: { "Dynamic Material Parameter": Dynamic("Lerp Vec4", {
                "A": Dynamic("Select Vec4 by Int", { "Index": Link(Int, "Particles.RibbonID"), "Values": [Vec4((1.0, 0.0, 0.0, 0.0)), Vec4((0.0, 1.0, 0.0, 0.0)), Vec4((0.0, 0.0, 1.0, 0.0)), Vec4((0.0, 0.0, 0.0, 1.0)), Vec4((1.0, 1.0, 1.0, 1.0)), Vec4((2.0, 2.0, 2.0, 2.0)), Vec4((3.0, 3.0, 3.0, 3.0))] }),
                "B": Dynamic("Multiply Vec4", { "A": Dynamic("Vec4 from Curve", { "Curve": Vec4([(0.0, (0.0, 1.0, 2.0, 3.0)), (1.0, (3.0, 2.0, 1.0, 0.0))]), "Curve Index": Link(Float, "Particles.NormalizedAge") }), "B": Dynamic("Uniform Ranged Vec4", {}) }),
                "Alpha": Dynamic("Sine over Time", { "Time": Link(Float, "Particles.Age"), "Amplitude": Float(0.5), "Offset": Float(0.5) }),
            }) }),
            (module: "Scale Color", inputs: { "Scale RGB": Dynamic("Subtract Vec3", { "A": Vec3((1.0, 1.0, 1.0)), "B": Dynamic("Random Range Vec3 by Uniform", { "Maximum": Vec3((0.5, 0.5, 0.5)) }) }) }),
            (module: "Kill Particles", inputs: { "Kill Particles": Dynamic("Compare Floats", { "A": Link(Float, "Particles.Age"), "B": Float(1.3), "Operation": GreaterThan }) }),
        ],
    )],
)"#;

#[test]
fn gpu_matches_cpu_reference() {
    let (device, queue, info) = match fx_gpu::headless_device() {
        Ok(d) => d,
        Err(e) => panic!("parity test needs a GPU: {e}"),
    };
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    let mut systems: Vec<_> = ["fountain.fx.ron", "burst.fx.ron", "smoke_curl.fx.ron"]
        .into_iter()
        .map(|n| (n, sample(n), WORDS_PER_PARTICLE))
        .collect();
    systems.push(("extended", extended_system(), WORDS_PER_PARTICLE_EXTENDED));
    systems.push(("common modules", common_modules_system(), WORDS_PER_PARTICLE_EXTENDED));
    // The reference effect library (effects/library); words per particle vary
    // by emitter there, so 0 skips that check.
    // Samples of the common modules (forces, shapes, rotation, sub-UV).
    for n in ["forces.fx.ron", "shapes.fx.ron", "rotation.fx.ron", "subuv.fx.ron", "dynamic_inputs.fx.ron"] {
        systems.push((n, sample(n), 0));
    }
    // Nested dynamic inputs of every type.
    let c = fx_core::check_source("dynamic_types.fx.ron", DYNAMIC_TYPES);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    systems.push(("dynamic types", c.cooked.unwrap_or_else(|| panic!("{errs:?}")), WORDS_PER_PARTICLE_EXTENDED));
    for n in LIBRARY {
        systems.push((n, library(n), 0));
    }
    for (name, sys, words) in systems {
        let seed = 42;
        let mut cpu = Simulation::new(sys.clone(), seed);
        let mut gpu = GpuEffect::new(&device, &sys, seed).unwrap();
        // A moving owner, for Inherit Velocity.
        cpu.owner_velocity = bevy_math::Vec3::new(0.5, 0.25, -1.0);
        gpu.owner_velocity = cpu.owner_velocity.to_array();
        assert!(
            words == 0 || gpu.emitters.iter().all(|e| e.words == words),
            "{name}: words per particle"
        );
        let mut diff = Diff::default();
        let mut most_alive = 0usize;
        // 3 seconds: bursts, steady spawning, deaths and the curl field.
        for step in 1..=180u32 {
            cpu.step(DT);
            let mut enc = device.create_command_encoder(&Default::default());
            gpu.step(&queue, &mut enc, DT);
            queue.submit([enc.finish()]);
            gpu.submitted();
            if step % 15 == 0 {
                let g = gpu.read_particles(&device, &queue);
                for (ei, ps) in g.iter().enumerate() {
                    most_alive = most_alive.max(ps.len());
                    compare(name, step, cpu.particles(ei), ps, &mut diff);
                }
                let counts = gpu.read_counts(&device, &queue);
                for (ei, (_, next_id)) in counts.iter().enumerate() {
                    assert_eq!(*next_id, cpu.emitters[ei].next_id, "{name}: next UniqueID");
                }
            }
        }
        assert!(most_alive > 0, "{name}: nothing alive to compare");
        eprintln!(
            "{name}: up to {} particles compared at 12 checkpoints; largest difference {:.2e} ({})",
            most_alive, diff.max_abs, diff.worst
        );
    }
}

#[test]
fn several_steps_per_submit_match_one_per_submit() {
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let sys = sample("fountain.fx.ron");
    let mut a = GpuEffect::new(&device, &sys, 7).unwrap();
    let mut b = GpuEffect::new(&device, &sys, 7).unwrap();
    for _ in 0..8 {
        let mut enc = device.create_command_encoder(&Default::default());
        for _ in 0..4 {
            a.step(&queue, &mut enc, DT);
        }
        queue.submit([enc.finish()]);
        a.submitted();
        for _ in 0..4 {
            let mut enc = device.create_command_encoder(&Default::default());
            b.step(&queue, &mut enc, DT);
            queue.submit([enc.finish()]);
            b.submitted();
        }
    }
    let mut pa = a.read_particles(&device, &queue).remove(0);
    let mut pb = b.read_particles(&device, &queue).remove(0);
    pa.sort_by_key(|p| p.unique_id);
    pb.sort_by_key(|p| p.unique_id);
    assert!(!pa.is_empty());
    assert_eq!(pa.len(), pb.len());
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(
            x.bits(),
            y.bits(),
            "GPU results depend on how steps are submitted"
        );
    }
}
