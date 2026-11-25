//! Data interfaces (skeletal and static meshes, textures, curves, camera,
//! scene depth): declaration and validation, and one test per interface
//! and module on the CPU.

mod common;

use std::sync::Arc;

use bevy_math::{Mat4, Vec3, Vec4};
use common::*;
use fx_core::data::{self, CameraData, DiType, MeshData, MeshPose, TextureData};
use fx_core::sim::{Collider, Particle, Simulation};

const DT: f32 = 0.25;

/// Every particle after one step of a burst of `count` with `spawn` modules.
fn burst(user: &str, spawn: &str, count: u32, bind: impl FnOnce(&mut Simulation)) -> Vec<Particle> {
    let src = one_emitter(
        user,
        &format!(r#"(module: "Spawn Burst Instantaneous", inputs: {{ "Spawn Count": Int({count}) }})"#),
        spawn,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 7);
    bind(&mut sim);
    sim.step(DT);
    sim.particles(0).to_vec()
}

fn errors(src: &str) -> Vec<String> {
    fx_core::check_source("t.fx.ron", src)
        .errors()
        .map(|d| d.to_string())
        .collect()
}

/// A 2 x 2 metre square in the XZ plane (normal +Y), two triangles.
fn square() -> MeshData {
    MeshData::new(
        vec![[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        vec![[0.0, 1.0, 0.0]; 4],
        vec![0, 2, 1, 0, 3, 2],
    )
}

const MESH_USER: &str = r#""User.Rock": StaticMesh("cube"), "User.Body": SkeletalMesh"#;

#[test]
fn every_data_interface_type_declares_and_cooks() {
    let src = one_emitter(
        r#""User.Body": SkeletalMesh("models/character.glb"),
           "User.Rock": StaticMesh("sphere"),
           "User.Pattern": Texture("pattern.png"),
           "User.Fade": FloatCurve([(0.0, 1.0), (1.0, 0.0)]),
           "User.Size": Vec2Curve([(0.0, (1.0, 1.0))]),
           "User.Path": VectorCurve([(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 2.0, 0.0))]),
           "User.Param": Vec4Curve([(0.0, (0.0, 0.0, 0.0, 0.0))]),
           "User.Tint": ColorCurve([(0.0, (1.0, 0.0, 0.0, 1.0)), (1.0, (0.0, 0.0, 1.0, 1.0))]),
           "User.Camera": CameraQuery,
           "User.Depth": SceneDepth,
           "User.Rate": Float(10.0)"#,
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Link(Float, "User.Rate") })"#,
        r#"(module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body") })"#,
        "",
    );
    let sys = compile(&src);
    let names: Vec<_> = sys.data_interfaces.iter().map(|d| (d.name.as_str(), d.ty)).collect();
    assert_eq!(names.len(), 10);
    assert_eq!(names[0], ("User.Body", DiType::SkeletalMesh));
    assert_eq!(sys.data_interfaces[0].asset.as_deref(), Some("models/character.glb"));
    assert_eq!(sys.data_interfaces[3].curve.as_ref().unwrap().values.len(), 2);
    assert_eq!(sys.user_params.len(), 1, "value parameters stay apart");
    // The module input cooks to the data interface's index.
    let m = &sys.emitters[0].particle_spawn[0];
    assert_eq!(m.inputs[0], fx_core::cook::CookedInput::Const(fx_core::types::Value::Int(0)));
    // The source model serialises back to RON that parses to the same model.
    let model = fx_core::check_source("t.fx.ron", &src).source.unwrap();
    let text = ron::ser::to_string(&model).unwrap();
    let again = fx_core::check_source("t.fx.ron", &text);
    assert!(again.ok(), "{:?}\n{text}", again.diagnostics);
    assert_eq!(ron::ser::to_string(&again.source.unwrap()).unwrap(), text);
}

#[test]
fn data_interface_errors_name_the_fix() {
    let spawn = |user: &str, inputs: &str| {
        one_emitter(
            user,
            r#"(module: "Spawn Rate")"#,
            &format!(r#"(module: "Static Mesh Location", inputs: {{ {inputs} }})"#),
            "",
        )
    };
    let e = errors(&spawn(MESH_USER, ""));
    assert!(e.iter().any(|m| m.contains("needs its \"Static Mesh\" input") && m.contains("Link(StaticMesh, \"User.Rock\")")), "{e:?}");
    let e = errors(&spawn(MESH_USER, r#""Static Mesh": Link(StaticMesh, "User.Rocks")"#));
    assert!(e.iter().any(|m| m.contains("\"User.Rocks\" is not declared") && m.contains("did you mean \"User.Rock\"")), "{e:?}");
    let e = errors(&spawn(MESH_USER, r#""Static Mesh": Link(StaticMesh, "User.Body")"#));
    assert!(e.iter().any(|m| m.contains("is a SkeletalMesh, but the link says StaticMesh")), "{e:?}");
    let e = errors(&spawn(MESH_USER, r#""Static Mesh": Link(SkeletalMesh, "User.Body")"#));
    assert!(e.iter().any(|m| m.contains("takes a StaticMesh data interface")), "{e:?}");
    let e = errors(&spawn(MESH_USER, r#""Static Mesh": Int(0)"#));
    assert!(e.iter().any(|m| m.contains("takes a StaticMesh data interface")), "{e:?}");
    let e = errors(&spawn(MESH_USER, r#""Surface Offset": Link(StaticMesh, "User.Rock")"#));
    assert!(e.iter().any(|m| m.contains("is Float, but a StaticMesh data interface is given")), "{e:?}");
    let e = errors(&spawn(r#""User.Rock": StaticMesh("rock")"#, ""));
    assert!(e.iter().any(|m| m.contains("unknown built-in mesh \"rock\"")), "{e:?}");
    let e = errors(&spawn(r#""User.Rock": StaticMesh(1.0)"#, ""));
    assert!(e.iter().any(|m| m.contains("with one asset path")), "{e:?}");
    let e = errors(&spawn(r#""Rock": StaticMesh("cube")"#, ""));
    assert!(e.iter().any(|m| m.contains("must be in the User namespace")), "{e:?}");
    let e = errors(&spawn(r#""User.Rock": StaticMesh("cube"), "User.Rock": Float(1.0)"#, ""));
    assert!(e.iter().any(|m| m.contains("declared twice")), "{e:?}");
    let e = errors(&spawn(r#""User.Fade": FloatCurve([(1.0, 1.0), (0.0, 0.0)])"#, ""));
    assert!(e.iter().any(|m| m.contains("ascending")), "{e:?}");
    // A curve of the wrong type, and the camera outside particle stages.
    let e = errors(&one_emitter(
        r#""User.Fade": FloatCurve([(0.0, 1.0)]), "User.Camera": CameraQuery"#,
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Break Vec3", { "Value": Dynamic("Get Camera Position", { "Camera": Link(CameraQuery, "User.Camera") }) }) })"#,
        r#"(module: "Initialize Particle", inputs: { "Color": Dynamic("Color from Curve", { "Curve": Link(FloatCurve, "User.Fade") }) })"#,
        "",
    ));
    assert!(e.iter().any(|m| m.contains("takes a ColorCurve data interface, but \"User.Fade\" is a FloatCurve")), "{e:?}");
    assert!(e.iter().any(|m| m.contains("reads the camera, which `emitter_update` does not have")), "{e:?}");
}

#[test]
fn static_mesh_location_samples_the_surface_evenly_and_moves_with_the_mesh() {
    let user = r#""User.Rock": StaticMesh("cube")"#;
    let spawn = r#"(module: "Static Mesh Location", inputs: { "Static Mesh": Link(StaticMesh, "User.Rock"), "Surface Offset": Float(0.25), "Inherit Velocity Scale": Float(0.5) })"#;
    // Unbound: particles stay where they are.
    for p in burst(user, spawn, 10, |_| {}) {
        assert_eq!(p.position, Vec3::ZERO);
    }
    let ps = burst(user, spawn, 1000, |sim| {
        let mut pose = MeshPose::still(vec![Mat4::IDENTITY]);
        pose.advance(&[Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0))], 0.5);
        assert!(sim.set_mesh("User.Rock", Arc::new(square()), pose));
        assert!(!sim.set_mesh("User.None", Arc::new(square()), MeshPose::default()));
    });
    assert_eq!(ps.len(), 1000);
    let mut quadrants = [0u32; 4];
    for p in &ps {
        // On the square, moved up 2 m, plus 0.25 m along the normal.
        assert_close(p.position.y, 2.25);
        assert!(p.position.x.abs() <= 1.0 && p.position.z.abs() <= 1.0, "{p:?}");
        // The mesh moved 2 m in 0.5 s: 4 m/s, half inherited.
        assert_vec_close(&p.velocity.to_array(), &[0.0, 2.0, 0.0]);
        quadrants[(p.position.x > 0.0) as usize * 2 + (p.position.z > 0.0) as usize] += 1;
    }
    for q in quadrants {
        assert!((200..300).contains(&q), "uneven spread {quadrants:?}");
    }
}

#[test]
fn surface_samples_are_spread_by_area() {
    // A 1 m² triangle and a 0.01 m² one: about 1% of samples on the small one.
    let mesh = MeshData::new(
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [10.0, 0.0, 0.1],
        ],
        Vec::new(),
        vec![0, 2, 1, 3, 5, 4],
    );
    assert_eq!(mesh.normals[0], [0.0, 1.0, 0.0], "flat normals are made when none are given");
    let pose = MeshPose::still(vec![Mat4::IDENTITY]);
    let n = 20000;
    let small = (0..n)
        .filter(|i| {
            let u = [(*i as f32 + 0.5) / n as f32, 0.3, 0.3];
            data::sample_surface(&mesh, &pose, u).unwrap().position.x >= 10.0
        })
        .count();
    assert!((150..250).contains(&small), "{small}");
}

/// A skinned strip: vertices 0..2 follow joint 0, 3..5 joint 1, 6..8 half
/// each; three triangles, one per group.
fn skinned_strip() -> MeshData {
    let mut positions = Vec::new();
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    for (g, w) in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]].iter().enumerate() {
        let x = g as f32 * 2.0;
        positions.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 0.0, 1.0]]);
        for _ in 0..3 {
            joints.push([0, 1, 0, 0]);
            weights.push([w[0], w[1], 0.0, 0.0]);
        }
    }
    let binds = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(-3.0, 0.0, 0.0))];
    MeshData::skinned(positions, Vec::new(), vec![0, 2, 1, 3, 5, 4, 6, 8, 7], joints, weights, &binds)
}

#[test]
fn skeletal_mesh_location_follows_the_skinned_pose() {
    let mesh = skinned_strip();
    assert!(mesh.is_skinned());
    assert_eq!(mesh.bone_positions, vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
    let user = r#""User.Body": SkeletalMesh"#;
    let up = Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0));
    let pose = MeshPose {
        current: vec![Mat4::IDENTITY, up],
        previous: vec![Mat4::IDENTITY, Mat4::IDENTITY],
        interval: 2.0,
    };
    let spawn = r#"(module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body") })"#;
    let ps = burst(user, spawn, 600, |sim| {
        assert!(sim.set_mesh("User.Body", Arc::new(skinned_strip()), pose.clone()));
    });
    let mut seen = [0; 3];
    for p in &ps {
        // Joint 0's triangle stays, joint 1's is lifted 4 m, the shared one 2 m.
        let g = (p.position.x / 2.0).floor() as usize;
        let want = [0.0, 4.0, 2.0][g];
        assert_close(p.position.y, want);
        assert_vec_close(&p.velocity.to_array(), &[0.0, want / 2.0, 0.0]);
        seen[g] += 1;
    }
    assert!(seen.iter().all(|&n| n > 150), "{seen:?}");

    // Bones: each joint's bind position, posed.
    let spawn = r#"(module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body"), "Sampling Mode": Bones, "Inherit Velocity Scale": Float(0.0) })"#;
    let ps = burst(user, spawn, 200, |sim| {
        sim.set_mesh("User.Body", Arc::new(skinned_strip()), pose.clone());
    });
    let mut bones = [0; 2];
    for p in &ps {
        let b = (p.position.x > 1.0) as usize;
        let want = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]][b];
        assert_vec_close(&p.position.to_array(), &want);
        assert_eq!(p.velocity, Vec3::ZERO);
        bones[b] += 1;
    }
    assert!(bones.iter().all(|&n| n > 50), "{bones:?}");

    // A new pose moves later spawns; the interval gives the velocity.
    let src = one_emitter(
        user,
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Float(40.0) })"#,
        r#"(module: "Skeletal Mesh Location", inputs: { "Skeletal Mesh": Link(SkeletalMesh, "User.Body") })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 3);
    sim.set_mesh("User.Body", Arc::new(skinned_strip()), MeshPose::still(vec![Mat4::IDENTITY; 2]));
    sim.step(DT);
    assert!(sim.set_mesh_pose("User.Body", &[Mat4::IDENTITY, Mat4::from_translation(Vec3::Z)], DT));
    sim.step(DT);
    let newest = sim.particles(0).last().unwrap();
    if newest.position.x >= 2.0 && newest.position.x < 4.0 {
        assert_close(newest.velocity.z, 4.0);
    }
    assert!(sim.particles(0).iter().any(|p| p.position.z > 1.0), "joint 1 moved +Z");
}

#[test]
fn sample_texture_reads_texels_bilinearly_and_wraps() {
    // 2 x 2: red, green / blue, white.
    let tex = TextureData::new(
        2,
        2,
        vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
    );
    assert_eq!(data::sample_texture(&tex, [0.25, 0.25]), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(data::sample_texture(&tex, [0.75, 0.75]), [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(data::sample_texture(&tex, [1.25, -0.75]), [1.0, 0.0, 0.0, 1.0], "wraps");
    assert_eq!(data::sample_texture(&tex, [0.5, 0.25]), [0.5, 0.5, 0.0, 1.0], "between red and green");

    let user = r#""User.Pattern": Texture("pattern.png")"#;
    let spawn = |uv: &str, blend: &str| {
        format!(
            r#"(module: "Initialize Particle", inputs: {{ "Color": Color((0.5, 0.5, 0.5, 0.5)) }}),
               (module: "Sample Texture", inputs: {{ "Texture": Link(Texture, "User.Pattern"), "UV": {uv}, "Color Scale": Color((1.0, 1.0, 1.0, 0.5)), "Blend": {blend} }})"#
        )
    };
    let tex = Arc::new(tex);
    let p = burst(user, &spawn("Vec2((0.75, 0.25))", "Replace"), 1, |sim| {
        assert!(sim.set_texture("User.Pattern", tex.clone()));
    })[0];
    assert_eq!(p.color, Vec4::new(0.0, 1.0, 0.0, 0.5));
    let p = burst(user, &spawn("Vec2((0.25, 0.75))", "Multiply"), 1, |sim| {
        sim.set_texture("User.Pattern", tex.clone());
    })[0];
    assert_eq!(p.color, Vec4::new(0.0, 0.0, 0.5, 0.25));
    // Unbound: white times the scale.
    let p = burst(user, &spawn("Vec2((0.25, 0.75))", "Replace"), 1, |_| {})[0];
    assert_eq!(p.color, Vec4::new(1.0, 1.0, 1.0, 0.5));
}

#[test]
fn curve_data_interfaces_sample_like_inline_curves_in_any_stage() {
    let user = r#""User.Tint": ColorCurve([(0.0, (1.0, 0.0, 0.0, 1.0)), (1.0, (0.0, 0.0, 1.0, 1.0))]),
                  "User.Rate": FloatCurve([(0.0, 4.0), (1.0, 4.0)])"#;
    let src = one_emitter(
        user,
        r#"(module: "Spawn Rate", inputs: { "Spawn Rate": Dynamic("Float from Curve", { "Curve": Link(FloatCurve, "User.Rate"), "Curve Index": Link(Float, "Emitter.Age") }) })"#,
        r#"(module: "Initialize Particle", inputs: { "Color": Dynamic("Color from Curve", { "Curve": Link(ColorCurve, "User.Tint"), "Curve Index": Float(0.25) }) })"#,
        "",
    );
    let mut sim = Simulation::new(compile(&src), 0);
    sim.step(0.5);
    assert_eq!(sim.particle_count(), 2, "4 per second from the curve");
    for p in sim.particles(0) {
        assert_vec_close(&p.color.to_array(), &[0.75, 0.0, 0.25, 1.0]);
    }
}

#[test]
fn camera_query_and_scene_depth_read_the_host_camera() {
    let user = r#""User.Camera": CameraQuery, "User.Depth": SceneDepth"#;
    let spawn = r#"(module: "Initialize Particle", inputs: {
            "Position": Dynamic("Get Camera Position", { "Camera": Link(CameraQuery, "User.Camera") }),
            "Lifetime": Dynamic("Sample Scene Depth", { "Scene Depth": Link(SceneDepth, "User.Depth"), "Position": Vec3((0.0, 0.0, -1.0)) }),
        }),
        (module: "Add Velocity", inputs: { "Velocity": Dynamic("Get Camera Forward Vector", { "Camera": Link(CameraQuery, "User.Camera") }) })"#;
    // No camera: origin, -Z, nothing in depth.
    let p = burst(user, spawn, 1, |_| {})[0];
    assert_eq!(p.position, Vec3::ZERO);
    assert_eq!(p.velocity, Vec3::NEG_Z);
    assert_eq!(p.lifetime, data::NO_SCENE_DEPTH);
    let p = burst(user, spawn, 1, |sim| {
        sim.camera = Some(CameraData {
            position: Vec3::new(0.0, 0.0, 5.0),
            forward: Vec3::NEG_Z,
        });
        // A wall 3 m in front of the camera and a sphere further on.
        sim.colliders.push(Collider::Plane {
            normal: Vec3::Z,
            origin: Vec3::new(0.0, 0.0, 2.0),
        });
        sim.colliders.push(Collider::Sphere {
            center: Vec3::new(0.0, 0.0, -4.0),
            radius: 1.0,
        });
    })[0];
    assert_eq!(p.position, Vec3::new(0.0, 0.0, 5.0));
    assert_eq!(p.velocity, Vec3::NEG_Z);
    assert_close(p.lifetime, 3.0);
}

#[test]
fn camera_offset_sets_or_adds_the_attribute() {
    let mut sim = single_particle(
        "",
        r#"(module: "Camera Offset", inputs: { "Camera Offset Amount": Float(0.5) })"#,
        r#"(module: "Camera Offset", inputs: { "Camera Offset Amount": Float(0.25), "Mode": Additive })"#,
    );
    sim.step(DT);
    assert_eq!(sim.particles(0)[0].camera_offset, 0.5);
    sim.step(DT);
    assert_eq!(sim.particles(0)[0].camera_offset, 0.75);
    assert!(sim.system.emitters[0].uses_extended_attributes());
}

#[test]
fn the_reference_lists_every_data_interface() {
    let r = fx_core::docs::reference();
    for t in DiType::ALL {
        assert!(r.contains(&format!("| {} |", t.name())), "{}", t.name());
    }
    assert!(r.contains("\"Skeletal Mesh Location\""));
    let doc = fx_core::docs::module_doc("Sample Texture").unwrap();
    assert!(doc.contains("Texture data interface"), "{doc}");
}
