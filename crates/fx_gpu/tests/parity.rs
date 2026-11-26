//! CPU/GPU parity: the same sample effect and seed give the same particles
//! on the GPU as fx_core's CPU reference, within float tolerance.
//!
//! Integer work (random hashes, spawn counts, UniqueIDs, kills) must match
//! exactly; float attributes may differ by the GPU's FMA contraction and its
//! exp/sin/cos approximations. One inline system covers the extended
//! attributes (Ribbon, Mesh, Light, material parameters).

use std::path::PathBuf;

use fx_core::sim::{Particle, Simulation};
use fx_gpu::{GpuEffect, WORDS_PER_PARTICLE, WORDS_PER_PARTICLE_EXTENDED, WORDS_PER_PARTICLE_EXTENDED_2};

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
    // Tutorial batch 2 (lightning_strike and meteor_rain use custom WGSL
    // modules, which only the GPU runs, so they are not compared here).
    "shockwave_ring.fx.ron",
    "portal_vortex.fx.ron",
    "stylized_tornado.fx.ron",
    "rain_splash.fx.ron",
    "muzzle_flash.fx.ron",
    "footstep_dust.fx.ron",
    "heal_aura.fx.ron",
    "fireworks_show.fx.ron",
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
    put("camera_offset", &[p.camera_offset]);
    put("sprite_facing", &p.sprite_facing.to_array());
    put("sprite_alignment", &p.sprite_alignment.to_array());
    put("dynamic_material_parameter_1", &p.dynamic_material_parameter_1.to_array());
    put("rotational_velocity", &p.rotational_velocity.to_array());
    put("rotational_force", &p.rotational_force.to_array());
    put("rotational_drag", &[p.rotational_drag]);
    put("dynamic_material_parameter_2", &p.dynamic_material_parameter_2.to_array());
    put("dynamic_material_parameter_3", &p.dynamic_material_parameter_3.to_array());
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

/// What the samples leave out: Set Particle Position, burst probability, a
/// non-deterministic emitter (CPU and GPU given the same nonce), Emitter State
/// with a delay only before the first loop, loop values read by modules.
const BATCH2: &str = r#"(
    name: "Batch 2",
    warmup_time: 0.5,
    emitters: [(
        name: "E",
        max_particles: 2000,
        deterministic: false,
        emitter_update: [
            (module: "Emitter State", inputs: { "Loop Duration": Float(0.7), "Loop Delay": Float(0.2), "Delay First Loop Only": Bool(true) }),
            (module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(40), "Spawn Time": Float(0.1), "Spawn Probability": Float(0.6) }),
            (module: "Spawn Burst Interval", inputs: { "Spawn Count": Int(5), "Spawn Interval": Float(0.05), "Spawn Probability": Float(0.5) }),
            (module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Multiply Float", { "A": Float(200.0), "B": Link(Float, "Emitter.NormalizedLoopAge") }) }),
        ],
        particle_spawn: [
            (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.4), Float(1.2)) }),
            (module: "Set Particle Position", inputs: { "Position": RandomRange(Vec3((-1.0, 0.0, -1.0)), Vec3((1.0, 2.0, 1.0))) }),
            (module: "Set Particle Velocity", inputs: { "Velocity": RandomRange(Vec3((-2.0, -2.0, -2.0)), Vec3((2.0, 2.0, 2.0))) }),
            (module: "Initialize Mesh", inputs: { "Mesh Scale": Vec3((0.5, 0.5, 0.5)) }),
            (module: "Set Particle Sprite Size", inputs: { "Sprite Size": Vec2((0.2, 0.3)), "Set Initial": Bool(true) }),
            (module: "Set Particle Scale", inputs: { "Scale": RandomRange(Vec3((0.5, 0.5, 0.5)), Vec3((1.0, 2.0, 1.0))), "Set Initial": Bool(true) }),
            (module: "Add Velocity from Point", inputs: { "Velocity Strength": Float(1.5), "Velocity Scale": Vec3((1.0, 0.5, 1.0)) }),
            (module: "Initialize Ribbon", inputs: { "Ribbon Width": Float(0.2) }),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Mesh Rotation Force", inputs: { "Rotation Force": Vec3((30.0, 90.0, -45.0)) }),
            (module: "Rotational Drag", inputs: { "Rotational Drag": Float(0.5) }),
            (module: "Noise Force", inputs: { "Noise Strength": Float(2.0), "Noise Octaves": Int(4), "Pan Noise Field": Vec3((0.3, 0.0, 0.1)) }),
            (module: "Solve Forces and Velocity"),
            (module: "Limit Velocity", inputs: { "Velocity Limit": Float(2.5) }),
            (module: "Orient Mesh to Vector", inputs: { "Orientation Vector": Link(Vec3, "Particles.Velocity") }),
            (module: "Align Sprite to Mesh Orientation"),
            (module: "Scale Color by Distance", inputs: { "Reference Position": Vec3((0.0, 1.0, 0.0)), "Far Distance": Float(2.0) }),
            (module: "Scale Sprite Size by Distance", inputs: { "Near Distance": Float(0.5), "Far Distance": Float(1.5) }),
            (module: "Scale Mesh Size by Speed", inputs: { "Velocity Threshold": Float(2.0) }),
            (module: "Scale Ribbon Width by Speed", inputs: { "Velocity Threshold": Float(2.0) }),
            (module: "Ribbon Twist Rate", inputs: { "Twist Rate": Link(Float, "Emitter.LoopedAge") }),
            (module: "Dynamic Material Parameters 2", inputs: { "Dynamic Material Parameter 1": Dynamic("Make Vec4 from Floats", { "X": Link(Float, "Emitter.LoopedAge"), "Y": Dynamic("Multiply Float", { "A": Float(1.0), "B": Link(Float, "Emitter.Age") }) }) }),
            (module: "Jitter Position", inputs: { "Jitter Amount": Vec3((0.01, 0.02, 0.01)), "Jitter Delay": Float(0.1) }),
            (module: "Kill Particles by Speed", inputs: { "Minimum Speed": Float(0.05) }),
        ],
    )],
)"#;

/// Inputs added for Niagara imports: HSV colour, cone inner angle,
/// distribution and speed falloff, box midpoint, cone shape options (a
/// random input past slot 16), Scale RGBA, wind turbulence, the attracted
/// position offset, and Dynamic Material Parameters 3 and 4 (which store 8
/// more words per particle). A second emitter keeps the base layout.
// Interpolated spawning: spawned particles run Particle Update in their spawn
// step over part of it (random draws in update, curves, forces, deaths on the
// spawn step), next to an emitter without it.
const INTERPOLATED: &str = r#"(
    name: "Interpolated spawning",
    emitters: [
        (
            name: "Stream",
            max_particles: 2000,
            interpolated_spawning: true,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(137.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.5), Float(1.5)), "Sprite Size": Vec2((0.2, 0.2)) }),
                (module: "Add Velocity", inputs: { "Velocity": RandomRange(Vec3((-1.0, 2.0, -1.0)), Vec3((1.0, 4.0, 1.0))) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Gravity Force"),
                (module: "Drag", inputs: { "Drag": RandomRange(Float(0.0), Float(2.0)) }),
                (module: "Scale Color", inputs: { "Scale Alpha": CurveOverAge(Float([(0.0, 0.0), (0.2, 1.0), (1.0, 0.0)])) }),
                (module: "Scale Sprite Size", inputs: { "Scale Factor": CurveOverAge(Vec2([(0.0, (0.0, 0.0)), (1.0, (2.0, 2.0))])) }),
                (module: "Solve Forces and Velocity"),
            ],
        ),
        (
            name: "Burst",
            max_particles: 500,
            interpolated_spawning: true,
            emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.001), Float(1.0)) }),
                (module: "Add Velocity", inputs: { "Velocity": RandomRange(Vec3((-2.0, -2.0, -2.0)), Vec3((2.0, 2.0, 2.0))) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Scale Color", inputs: { "Scale RGB": CurveOverAge(Vec3([(0.0, (4.0, 2.0, 1.0)), (1.0, (0.0, 0.0, 0.0))])) }),
                (module: "Solve Forces and Velocity"),
            ],
        ),
        (
            name: "Plain",
            max_particles: 500,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(90.0) })],
            particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(1.0) })],
            particle_update: [
                (module: "Particle State"),
                (module: "Scale Color", inputs: { "Scale Alpha": CurveOverAge(Float([(0.0, 0.0), (1.0, 1.0)])) }),
            ],
        ),
    ],
)"#;

const GAPS: &str = r#"(
    name: "Import gaps",
    emitters: [(
        name: "Cone",
        max_particles: 2000,
        emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(300.0) })],
        particle_spawn: [
            (module: "Initialize Particle", inputs: {
                "Lifetime": RandomRange(Float(0.5), Float(1.5)),
                "Color": Color((0.8, 0.4, 0.2, 0.9)),
                "Hue Shift": RandomRange(Float(-36.0), Float(36.0)),
                "Saturation Scale": RandomRange(Float(0.5), Float(1.0)),
                "Value Scale": Float(1.5),
                "Alpha Scale": RandomRange(Float(0.25), Float(1.0)),
            }),
            (module: "Shape Location", inputs: {
                "Shape Primitive": Cone, "Cone Length": Float(2.0), "Cone Angle": Float(40.0),
                "Cone Inner Angle": Float(15.0), "Cone Radial Angle": Float(270.0),
                "Cone Surface Distribution": RandomRange(Float(0.0), Float(0.5)),
            }),
            (module: "Add Velocity in Cone", inputs: {
                "Cone Angle": Float(50.0), "Velocity Strength": Float(3.0), "Inner Cone Angle": Float(10.0),
                "Distribution Along Cone Axis": Float(0.6), "Speed Falloff": Float(0.4),
            }),
        ],
        particle_update: [
            (module: "Particle State"),
            (module: "Scale Color", inputs: { "Scale RGB": Vec3((1.0, 0.5, 1.0)), "Scale RGBA": CurveOverAge(Vec4([(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (0.5, 2.0, 1.0, 0.0))])) }),
            (module: "Wind Force", inputs: {
                "Wind Velocity": Vec3((2.0, 0.0, 1.0)), "Air Resistance": Float(0.5),
                "Turbulence Scale": Float(0.5), "Turbulence Frequency": Float(0.8),
                "Turbulence Offset By Time": Vec3((0.0, -1.0, 0.0)), "Turbulence Position": Link(Vec3, "Particles.Position"),
            }),
            (module: "Point Attraction Force", inputs: {
                "Attractor Position": Vec3((0.0, 2.0, 0.0)), "Attraction Strength": Float(2.0), "Attraction Radius": Float(4.0),
                "Attracted Position Offset": Dynamic("Multiply Vec3 by Float", { "A": Vec3((0.5, 0.0, 0.25)), "B": Dynamic("Sine over Time", { "Frequency": Float(0.5) }) }),
            }),
            (module: "Solve Forces and Velocity"),
            (module: "Dynamic Material Parameters 3", inputs: { "Dynamic Material Parameter 2": Dynamic("Make Vec4 from Floats", { "X": Link(Float, "Particles.NormalizedAge"), "Y": Float(2.0) }) }),
            (module: "Dynamic Material Parameters 4", inputs: { "Dynamic Material Parameter 3": Link(Vec4, "Particles.DynamicMaterialParameter2") }),
        ],
    ), (
        name: "Box",
        max_particles: 1000,
        emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(200.0) })],
        particle_spawn: [
            (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.5), Float(1.0)) }),
            (module: "Shape Location", inputs: {
                "Shape Primitive": Box, "Surface Only": Bool(true), "Box Size": Vec3((1.0, 2.0, 3.0)),
                "Box Midpoint": RandomRange(Vec3((0.0, 0.0, 0.0)), Vec3((1.0, 1.0, 1.0))),
            }),
        ],
        particle_update: [(module: "Particle State"), (module: "Scale Color", inputs: { "Scale RGBA": Vec4((0.5, 0.5, 0.5, 0.5)) })],
        renderers: [Sprite(())],
    )],
)"#;

// World-space emitters (`local_space: false`): spawned particles move to world
// space with the owner pose at their birth (interpolated through the step or
// at its end), next to a local-space emitter. The owner moves and turns.
const WORLD_SPACE: &str = r#"(
    name: "World space",
    emitters: [
        (
            name: "Interpolated",
            max_particles: 2000,
            local_space: false,
            interpolated_spawning: true,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(237.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.5), Float(1.5)), "Position": Vec3((0.2, 0.5, -0.3)) }),
                (module: "Add Velocity", inputs: { "Velocity": RandomRange(Vec3((-1.0, 2.0, -1.0)), Vec3((1.0, 4.0, 1.0))) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Gravity Force"),
                (module: "Solve Forces and Velocity"),
            ],
        ),
        (
            name: "At step end",
            max_particles: 2000,
            local_space: false,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(97.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": Float(1.0), "Position": Vec3((1.0, 0.0, 0.0)) }),
                (module: "Add Velocity", inputs: { "Velocity": Vec3((0.0, 0.0, 2.0)) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Solve Forces and Velocity"),
            ],
        ),
        (
            name: "Local",
            max_particles: 500,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(50.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": Float(1.0), "Position": Vec3((1.0, 0.0, 0.0)) }),
            ],
            particle_update: [(module: "Particle State")],
        ),
    ],
)"#;

/// Where the parity test's owner is at time `t`: flying, turning about a
/// tilted axis and scaled.
fn owner_at(t: f32) -> bevy_math::Affine3A {
    let axis = bevy_math::Vec3::new(0.3, 1.0, 0.2).normalize();
    bevy_math::Affine3A::from_scale_rotation_translation(
        bevy_math::Vec3::new(1.2, 0.8, 1.0),
        bevy_math::Quat::from_axis_angle(axis, 1.7 * t),
        bevy_math::Vec3::new(4.0 * t, (2.0 * t).sin(), -1.5 * t),
    )
}

#[test]
fn material_parameters_2_3_widen_only_their_emitter() {
    let (device, _, _) = fx_gpu::headless_device().unwrap();
    let c = fx_core::check_source("gaps.fx.ron", GAPS);
    let sys = c.cooked.unwrap();
    let gpu = GpuEffect::new(&device, &sys, 1).unwrap();
    assert_eq!(gpu.emitters[0].words, WORDS_PER_PARTICLE_EXTENDED_2);
    assert_eq!(gpu.emitters[1].words, WORDS_PER_PARTICLE);
    assert_eq!(fx_gpu::extended_level(gpu.emitters[0].words), 2);
    assert_eq!(fx_gpu::extended_level(WORDS_PER_PARTICLE_EXTENDED), 1);
    assert_eq!(fx_gpu::extended_level(gpu.emitters[1].words), 0);
}

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
    // Batch 2: emitter options, spawn modules and the further particle modules.
    for n in [
        "loops.fx.ron",
        "motion.fx.ron",
        "trail.fx.ron",
        "mesh_spin.fx.ron",
        "sprite_facing.fx.ron",
        "ribbon_motion.fx.ron",
    ] {
        systems.push((n, sample(n), 0));
    }
    // Inputs added for Niagara imports, and material parameters 2 and 3.
    let c = fx_core::check_source("gaps.fx.ron", GAPS);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    systems.push(("import gaps", c.cooked.unwrap_or_else(|| panic!("{errs:?}")), 0));
    let c = fx_core::check_source("interpolated.fx.ron", INTERPOLATED);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    systems.push(("interpolated spawning", c.cooked.unwrap_or_else(|| panic!("{errs:?}")), 0));
    let c = fx_core::check_source("world_space.fx.ron", WORLD_SPACE);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    systems.push(("world space", c.cooked.unwrap_or_else(|| panic!("{errs:?}")), 0));
    let c = fx_core::check_source("batch2.fx.ron", BATCH2);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    systems.push(("batch 2 extras", c.cooked.unwrap_or_else(|| panic!("{errs:?}")), WORDS_PER_PARTICLE_EXTENDED));
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
        // And a moving, turning owner, for world-space emitters.
        cpu.teleport_owner(owner_at(0.0));
        gpu.teleport_owner(owner_at(0.0));
        // Same per-instance value for `deterministic: false` emitters.
        gpu.nonce = cpu.nonce;
        // The CPU warms up inside its first step; the GPU before it.
        gpu.warm_up(&device, &queue);
        assert!(
            words == 0 || gpu.emitters.iter().all(|e| e.words == words),
            "{name}: words per particle"
        );
        let mut diff = Diff::default();
        let mut most_alive = 0usize;
        // 3 seconds: bursts, steady spawning, deaths and the curl field.
        for step in 1..=180u32 {
            cpu.move_owner(owner_at(step as f32 * DT));
            gpu.move_owner(owner_at(step as f32 * DT));
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

/// Every event path: a receiver before its source (Stop collisions, capped
/// events), bounces on a tilted plane writing collision, death and location
/// events, an EveryParticle handler, and an emitter spawning from its own
/// deaths (Kill collisions) until its capacity caps the spawns.
const EVENTS: &str = r#"(
    name: "Events",
    emitters: [
        (
            name: "Before",
            max_particles: 300,
            event_handlers: [(source: "Drops", event: Collision, spawn_number: 3, max_events_per_frame: 7, modules: [
                (module: "Receive Collision Event", inputs: { "Inherit Velocity Scale": Float(0.5) }),
                (module: "Add Velocity", inputs: { "Velocity": Dynamic("Multiply Vec3 by Float", { "A": Link(Vec3, "Event.Normal"), "B": RandomRange(Float(0.5), Float(1.5)) }) }),
            ])],
            particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.3), Float(0.8)) })],
            particle_update: [
                (module: "Particle State"),
                (module: "Gravity Force"),
                (module: "Solve Forces and Velocity"),
                (module: "Collision", inputs: { "Collision Response": Stop, "Plane Origin": Vec3((0.0, -0.2, 0.0)) }),
            ],
        ),
        (
            name: "Drops",
            max_particles: 500,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(150.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(1.0), Float(2.0)), "Position": RandomRange(Vec3((-0.5, 1.0, -0.5)), Vec3((0.5, 1.5, 0.5))) }),
                (module: "Add Velocity", inputs: { "Velocity": RandomRange(Vec3((-1.0, -1.0, -1.0)), Vec3((1.0, 2.0, 1.0))) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Gravity Force"),
                (module: "Solve Forces and Velocity"),
                (module: "Collision", inputs: {
                    "Plane Normal": Vec3((0.3, 1.0, 0.1)),
                    "Particle Radius": Float(0.02),
                    "Restitution": RandomRange(Float(0.3), Float(0.8)),
                    "Friction": Float(0.3),
                    "Rest Speed": Float(0.3),
                }),
                (module: "Generate Collision Event"),
                (module: "Generate Death Event"),
                (module: "Generate Location Event", inputs: { "Event Probability": Float(0.1) }),
            ],
        ),
        (
            name: "Listeners",
            max_particles: 50,
            emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(50) })],
            particle_spawn: [(module: "Initialize Particle", inputs: { "Lifetime": Float(100.0) })],
            particle_update: [(module: "Particle State"), (module: "Solve Forces and Velocity")],
            event_handlers: [
                (source: "Drops", event: Location, execution_mode: EveryParticle, max_events_per_frame: 5, modules: [
                    (module: "Add Velocity", inputs: { "Velocity": Dynamic("Multiply Vec3 by Float", { "A": Link(Vec3, "Event.Velocity"), "B": RandomRange(Float(-0.01), Float(0.01)) }) }),
                ]),
                (source: "Drops", event: Death, spawn_number: 0),
            ],
        ),
        (
            name: "Self",
            max_particles: 200,
            emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(20) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.2), Float(0.5)) }),
                (module: "Add Velocity", inputs: { "Velocity": RandomRange(Vec3((-1.0, -3.0, -1.0)), Vec3((1.0, 0.0, 1.0))) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Gravity Force"),
                (module: "Solve Forces and Velocity"),
                (module: "Collision", inputs: { "Collision Response": Kill, "Plane Origin": Vec3((0.0, -0.5, 0.0)) }),
                (module: "Generate Death Event"),
            ],
            event_handlers: [(source: "Self", event: Death, spawn_number: 2, modules: [
                (module: "Receive Death Event", inputs: { "Position Offset": Vec3((0.0, 1.0, 0.0)) }),
            ])],
        ),
    ],
)"#;

fn compare_events(name: &str, step: u32, cpu: &[fx_core::events::Event], gpu: &[fx_core::events::Event]) {
    let ids = |v: &[fx_core::events::Event]| v.iter().map(|e| e.particle_id).collect::<Vec<_>>();
    assert_eq!(ids(cpu), ids(gpu), "{name} step {step}: event ids or order differ");
    for (a, b) in cpu.iter().zip(gpu) {
        let fa = a.position.iter().chain(&a.velocity).chain(&a.normal).chain([&a.age]);
        let fb = b.position.iter().chain(&b.velocity).chain(&b.normal).chain([&b.age]);
        for (x, y) in fa.zip(fb) {
            assert!(
                (x - y).abs() <= 1e-3 * (1.0 + x.abs()),
                "{name} step {step}: event of particle {}: CPU {a:?} GPU {b:?}",
                a.particle_id
            );
        }
    }
}

#[test]
fn gpu_events_and_collisions_match_cpu() {
    use fx_core::events::EventKind;
    let (device, queue, _) = fx_gpu::headless_device().expect("parity test needs a GPU");
    let c = fx_core::check_source("events.fx.ron", EVENTS);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    let events = c.cooked.unwrap_or_else(|| panic!("events: {errs:?}"));
    let systems = [
        ("bouncing_sparks.fx.ron", sample("bouncing_sparks.fx.ron")),
        ("fireworks.fx.ron", sample("fireworks.fx.ron")),
        ("events", events),
    ];
    for (name, sys) in systems {
        let mut cpu = Simulation::new(sys.clone(), 9);
        let mut gpu = GpuEffect::new(&device, &sys, 9).unwrap();
        let mut diff = Diff::default();
        let mut delivered = [0usize; 3];
        for step in 1..=180u32 {
            cpu.step(DT);
            let mut enc = device.create_command_encoder(&Default::default());
            gpu.step(&queue, &mut enc, DT);
            queue.submit([enc.finish()]);
            gpu.submitted();
            // Events every step.
            {
                for ei in 0..sys.emitters.len() {
                    for kind in EventKind::ALL {
                        let c = cpu.events(ei, kind);
                        compare_events(name, step, c, &gpu.read_events(&device, &queue, ei, kind));
                        delivered[kind.index()] += c.len();
                    }
                }
            }
            if step % 15 == 0 {
                let g = gpu.read_particles(&device, &queue);
                for (ei, ps) in g.iter().enumerate() {
                    compare(name, step, cpu.particles(ei), ps, &mut diff);
                }
                let counts = gpu.read_counts(&device, &queue);
                for (ei, (_, next_id)) in counts.iter().enumerate() {
                    assert_eq!(*next_id, cpu.emitters[ei].next_id, "{name}: next UniqueID of emitter {ei}");
                }
            }
        }
        eprintln!(
            "{name}: events compared (death {}, collision {}, location {}); largest particle difference {:.2e} ({})",
            delivered[0], delivered[1], delivered[2], diff.max_abs, diff.worst
        );
        assert!(delivered.iter().sum::<usize>() > 0, "{name}: no events");
    }
}

/// Every data interface: skinned surface and bone sampling of an animated
/// two-joint tube, a moving static mesh, a texture sampled at a UV built from
/// the position (Replace in spawn, Multiply in update), a colour curve asset,
/// the camera (position, forward, scene depth with nothing in the depth
/// buffer) and Camera Offset.
const DATA_INTERFACES: &str = r#"(
    name: "Data interfaces",
    user_parameters: {
        "User.Body": SkeletalMesh,
        "User.Rock": StaticMesh,
        "User.Picture": Texture,
        "User.Tint": ColorCurve([(0.0, (1.0, 0.2, 0.1, 1.0)), (0.5, (0.2, 1.0, 0.3, 0.8)), (1.0, (0.1, 0.2, 1.0, 0.0))]),
        "User.Camera": CameraQuery,
        "User.Depth": SceneDepth,
    },
    emitters: [
        (
            name: "Skin",
            max_particles: 2000,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(400.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.3), Float(1.0)) }),
                (module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body"), "Surface Offset": RandomRange(Float(0.0), Float(0.1)), "Inherit Velocity Scale": Float(0.5) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Color", inputs: { "Color": Dynamic("Color from Curve", { "Curve": Link(ColorCurve, "User.Tint") }) }),
                (module: "Solve Forces and Velocity"),
            ],
        ),
        (
            name: "Bones",
            max_particles: 500,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(100.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": Float(0.5) }),
                (module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body"), "Sampling Mode": Bones }),
            ],
            particle_update: [(module: "Particle State"), (module: "Solve Forces and Velocity")],
        ),
        (
            name: "Rock",
            max_particles: 1000,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(200.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.3), Float(1.0)) }),
                (module: "Static Mesh Location", inputs: { "Static Mesh": Link(StaticMesh, "User.Rock"), "Surface Offset": Float(0.05) }),
            ],
            particle_update: [(module: "Particle State"), (module: "Gravity Force"), (module: "Solve Forces and Velocity")],
        ),
        (
            name: "Picture",
            max_particles: 1500,
            emitter_update: [(module: "Spawn Rate", inputs: { "Spawn Rate": Float(300.0) })],
            particle_spawn: [
                (module: "Initialize Particle", inputs: { "Lifetime": RandomRange(Float(0.3), Float(1.0)) }),
                (module: "Shape Location", inputs: { "Shape Primitive": Box, "Box Size": Vec3((2.0, 2.0, 0.5)) }),
                (module: "Sample Texture", inputs: {
                    "Texture": Link(Texture, "User.Picture"),
                    "UV": Dynamic("Make Vec2 from Floats", {
                        "X": Dynamic("Break Vec3", { "Value": Link(Vec3, "Particles.Position"), "Component": X }),
                        "Y": Dynamic("Break Vec3", { "Value": Link(Vec3, "Particles.Position"), "Component": Y }),
                    }),
                }),
                (module: "Add Velocity", inputs: { "Velocity": Dynamic("Add Vec3", {
                    "A": Dynamic("Get Camera Forward Vector", { "Camera": Link(CameraQuery, "User.Camera") }),
                    "B": Dynamic("Multiply Vec3 by Float", { "A": Dynamic("Get Camera Position", { "Camera": Link(CameraQuery, "User.Camera") }), "B": Float(0.1) }),
                }) }),
                (module: "Camera Offset", inputs: { "Camera Offset Amount": RandomRange(Float(0.0), Float(0.3)) }),
            ],
            particle_update: [
                (module: "Particle State"),
                (module: "Solve Forces and Velocity"),
                (module: "Sample Texture", inputs: { "Texture": Link(Texture, "User.Picture"), "UV": Vec2((0.3, 0.7)), "Blend": Multiply, "Color Scale": Color((1.0, 1.0, 1.0, 0.98)) }),
                (module: "Camera Offset", inputs: { "Camera Offset Amount": Float(0.01), "Mode": Additive }),
                (module: "Dynamic Material Parameters", inputs: { "Dynamic Material Parameter": Dynamic("Make Vec4 from Floats", {
                    "X": Dynamic("Sample Scene Depth", { "Scene Depth": Link(SceneDepth, "User.Depth") }),
                    "Y": Dynamic("Vec3 Distance", { "A": Link(Vec3, "Particles.Position"), "B": Dynamic("Get Camera Position", { "Camera": Link(CameraQuery, "User.Camera") }) }),
                }) }),
            ],
        ),
    ],
)"#;

/// A tube along +Y from 0 to 2 m: the lower ring follows joint 0, the upper
/// joint 1 (bind pose at y = 1), blended in between.
fn tube() -> fx_core::data::MeshData {
    use bevy_math::{Mat4, Vec3};
    let (rings, segs) = (9u32, 12u32);
    let (mut pos, mut nrm, mut joints, mut weights, mut idx) = (vec![], vec![], vec![], vec![], vec![]);
    for r in 0..rings {
        let y = 2.0 * r as f32 / (rings - 1) as f32;
        let w1 = (y - 0.5).clamp(0.0, 1.0);
        for s in 0..segs {
            let a = std::f32::consts::TAU * s as f32 / segs as f32;
            pos.push([0.3 * a.cos(), y, 0.3 * a.sin()]);
            nrm.push([a.cos(), 0.0, a.sin()]);
            joints.push([0, 1, 0, 0]);
            weights.push([1.0 - w1, w1, 0.0, 0.0]);
        }
    }
    for r in 0..rings - 1 {
        for s in 0..segs {
            let (a, b) = (r * segs + s, r * segs + (s + 1) % segs);
            let (c, d) = (a + segs, b + segs);
            idx.extend([a, c, b, b, c, d]);
        }
    }
    let binds = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0))];
    fx_core::data::MeshData::skinned(pos, nrm, idx, joints, weights, &binds)
}

/// The tube's joints at time `t`: the whole tube turns, the upper half bends.
fn tube_pose(t: f32) -> Vec<bevy_math::Mat4> {
    use bevy_math::{Mat4, Vec3};
    let j0 = Mat4::from_rotation_y(t * 1.5) * Mat4::from_translation(Vec3::new(0.2 * t, 0.0, 0.0));
    let j1 = j0 * Mat4::from_translation(Vec3::Y) * Mat4::from_rotation_z((t * 3.0).sin() * 0.8);
    vec![j0, j1 * Mat4::from_translation(Vec3::NEG_Y)]
}

fn octahedron() -> fx_core::data::MeshData {
    let p = vec![
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ];
    let i = vec![0, 2, 4, 4, 2, 1, 1, 2, 5, 5, 2, 0, 0, 4, 3, 4, 1, 3, 1, 5, 3, 5, 0, 3];
    fx_core::data::MeshData::new(p.clone(), p, i)
}

fn picture() -> fx_core::data::TextureData {
    let (w, h) = (16u32, 8u32);
    let texels = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            [x / 15.0, y / 7.0, ((x * 7.0 + y * 3.0) % 5.0) / 4.0, 0.5 + 0.5 * ((x + y) % 2.0)]
        })
        .collect();
    fx_core::data::TextureData::new(w, h, texels)
}

#[test]
fn gpu_data_interfaces_match_cpu() {
    use bevy_math::{Mat4, Vec3};
    use fx_core::data::MeshPose;
    use std::sync::Arc;
    let (device, queue, _) = fx_gpu::headless_device().unwrap();
    let c = fx_core::check_source("data.fx.ron", DATA_INTERFACES);
    let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
    let sys = c.cooked.unwrap_or_else(|| panic!("{errs:?}"));
    let seed = 9;
    let mut cpu = Simulation::new(sys.clone(), seed);
    let mut gpu = GpuEffect::new(&device, &sys, seed).unwrap();
    assert!(gpu.uses_scene(), "camera and depth read the scene bindings");

    let (tube, rock, pic) = (Arc::new(tube()), Arc::new(octahedron()), Arc::new(picture()));
    let rock_pose = |t: f32| vec![Mat4::from_translation(Vec3::new(t.sin(), 1.0, 0.0)) * Mat4::from_rotation_x(t)];
    assert!(cpu.set_mesh("User.Body", tube.clone(), MeshPose::still(tube_pose(0.0))));
    assert!(gpu.set_mesh(&device, "User.Body", &tube, MeshPose::still(tube_pose(0.0))));
    assert!(cpu.set_mesh("User.Rock", rock.clone(), MeshPose::still(rock_pose(0.0))));
    assert!(gpu.set_mesh(&device, "User.Rock", &rock, MeshPose::still(rock_pose(0.0))));
    assert!(cpu.set_texture("User.Picture", pic.clone()));
    assert!(gpu.set_texture(&device, "User.Picture", &pic));
    assert!(!gpu.set_texture(&device, "User.Body", &pic), "wrong type");

    // A camera looking at the effect; no depth texture is bound.
    let eye = Vec3::new(3.0, 2.0, 6.0);
    // look_at_rh and perspective_infinite_reverse_rh, written out.
    let f = (-eye).normalize();
    let s = f.cross(Vec3::Y).normalize();
    let u = s.cross(f);
    let view = Mat4::from_cols(
        bevy_math::Vec4::new(s.x, u.x, -f.x, 0.0),
        bevy_math::Vec4::new(s.y, u.y, -f.y, 0.0),
        bevy_math::Vec4::new(s.z, u.z, -f.z, 0.0),
        bevy_math::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    );
    let fl = 1.0 / (0.5f32).tan();
    let proj = Mat4::from_cols(
        bevy_math::Vec4::new(fl / (16.0 / 9.0), 0.0, 0.0, 0.0),
        bevy_math::Vec4::new(0.0, fl, 0.0, 0.0),
        bevy_math::Vec4::new(0.0, 0.0, 0.0, -1.0),
        bevy_math::Vec4::new(0.0, 0.0, 0.1, 0.0),
    );
    gpu.set_scene_view(
        &queue,
        Mat4::IDENTITY,
        Some(&fx_gpu::SceneView {
            clip_from_world: proj * view,
            camera_position: eye,
            size: [64, 36],
        }),
    );
    cpu.camera = Some(fx_core::data::CameraData {
        position: eye,
        forward: (-eye).normalize(),
    });

    let mut diff = Diff::default();
    let mut seen = vec![0usize; sys.emitters.len()];
    for step in 1..=120u32 {
        let t = step as f32 * DT;
        cpu.set_mesh_pose("User.Body", &tube_pose(t), DT);
        gpu.set_mesh_pose(&queue, "User.Body", &tube_pose(t), DT);
        cpu.set_mesh_pose("User.Rock", &rock_pose(t), DT);
        gpu.set_mesh_pose(&queue, "User.Rock", &rock_pose(t), DT);
        cpu.step(DT);
        let mut enc = device.create_command_encoder(&Default::default());
        gpu.step(&queue, &mut enc, DT);
        queue.submit([enc.finish()]);
        gpu.submitted();
        if step % 20 == 0 {
            for (ei, ps) in gpu.read_particles(&device, &queue).iter().enumerate() {
                seen[ei] = seen[ei].max(ps.len());
                compare("data interfaces", step, cpu.particles(ei), ps, &mut diff);
            }
        }
    }
    assert!(seen.iter().all(|&n| n > 20), "every emitter compared: {seen:?}");
    // The samples really used the data: particles off the origin, coloured.
    assert!(cpu.particles(0).iter().any(|p| p.position.y > 1.5), "skinned surface");
    assert!(cpu.particles(3).iter().any(|p| p.color.x > 0.5 && p.color.y < 0.5), "texture");
    // Nothing in the depth buffer: "nothing there" (0 for particles not yet updated).
    assert!(cpu.particles(3).iter().all(|p| [0.0, fx_core::data::NO_SCENE_DEPTH].contains(&p.dynamic_material_parameter.x)));
    assert!(cpu.particles(3).iter().any(|p| p.dynamic_material_parameter.x == fx_core::data::NO_SCENE_DEPTH));
    eprintln!(
        "data interfaces: up to {seen:?} particles compared; largest difference {:.2e} ({})",
        diff.max_abs, diff.worst
    );
}
