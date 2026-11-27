//! `fximport blueprint` on the Fab packs' VFX Blueprints in testdata/fab
//! (local only, never committed; `sh testdata/fab_assets.sh`): PortalFX's 11
//! portals (UE 5.0 saves; ten colour variants of BP_PRT_Portal), Mixed VFX's
//! three actors (UE 4.26), the arrow trail demo (UE 5.4) and Perfect Fire's
//! BP_Fire (UE 4.27). Skipped when absent.

use std::path::PathBuf;

use fx_core::prefab::{Channel, Do, NodeKind, Num, Prefab, Property};
use fx_import::blueprint::{import, Effects, Imported};

fn fab(rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fab").join(rel);
    if !p.is_file() {
        eprintln!("skipped: {} is absent (sh testdata/fab_assets.sh)", p.display());
        return None;
    }
    Some(p)
}

fn node<'a>(p: &'a Prefab, name: &str) -> &'a fx_core::prefab::Node {
    p.nodes.iter().find(|n| n.name == name).unwrap_or_else(|| panic!("{}: no node {name}", p.name))
}

fn effect_of(p: &Prefab, name: &str) -> String {
    match &node(p, name).kind {
        NodeKind::Effect { effect, .. } => effect.clone(),
        k => panic!("{}: {name} is {k:?}", p.name),
    }
}

fn text(im: &Imported, file: &str) -> String {
    String::from_utf8(im.files.get(file).unwrap_or_else(|| panic!("no file {file}")).clone()).unwrap()
}

const PORTALS: &[(&str, &str)] = &[
    // (Blueprint suffix, colour its systems and materials are named for)
    ("", "Violet"),
    ("_Blue", "Blue"),
    ("_Cyan", "Cyan"),
    ("_Gold", "Gold"),
    ("_Green", "Green"),
    ("_Lime", "Lime"),
    ("_Orange", "Orange"),
    ("_Pink", "Pink"),
    ("_Red", "Red"),
    ("_Violet", "Violet"),
    ("_White", "White"),
];

#[test]
fn portal_variants_import_with_their_own_colour() {
    let mut effects = Effects::default();
    let mut lights = Vec::new();
    for (suffix, colour) in PORTALS {
        let Some(f) = fab(&format!("portals/Content/PortalFX/Blueprints/BP_PRT_Portal{suffix}.uasset")) else { return };
        let im = import(&f, &mut effects).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let p = &im.prefab;
        // The child variants inherit the parent's components and construction script.
        for n in ["DefaultSceneRoot", "Face", "Rays", "Sparks", "Glow"] {
            node(p, n);
        }
        assert_eq!(effect_of(p, "Sparks"), format!("NS_PRT_Sparks_{colour}.fx.ron"), "{}", p.name);
        // The meshes are read (UE 5.0 inline editor bulk data), with the variant's material.
        let face = text(&im, &effect_of(p, "Face"));
        assert!(face.contains("SM_PRT_Face.glb"), "{face}");
        assert!(face.contains(&format!("MI_PRT_Portal_{colour}.wesl")), "{face}");
        let rays = text(&im, &effect_of(p, "Rays"));
        assert!(rays.contains(&format!("MI_PRT_Rays_{colour}.wesl")), "{rays}");
        // 170 cm up, turned -90 degrees in yaw (the construction script).
        let t = node(p, "Face").transform;
        assert!((t.translation[1] - 1.7).abs() < 1e-4, "{t:?}");
        match node(p, "Glow").kind {
            NodeKind::PointLight { color, .. } => lights.push(color),
            ref k => panic!("{k:?}"),
        }
        // BeginPlay opens the portal: the material's custom primitive data
        // (OpenStart = game time, Closing = 0, OpenDuration = OpenTime).
        let set = |name: &str| {
            p.actions
                .iter()
                .filter(|a| a.node == "Face")
                .filter_map(|a| match &a.what {
                    Do::SetMaterial { name: n, value } if n == name => Some(value.clone()),
                    _ => None,
                })
                .last()
                .unwrap_or_else(|| panic!("{}: Face {name} never set", p.name))
        };
        assert_eq!(set("OpenStart"), vec![Num::GameTime { plus: 0.0 }]);
        assert_eq!(set("Closing"), vec![Num::Value(0.0)]);
        assert_eq!(set("OpenDuration"), vec![Num::Value(1.2)]);
        assert!(p.actions.iter().any(|a| a.node == "Sparks" && matches!(a.what, Do::Activate { .. })));
        // The demo loop and toggle events are listed as not imported.
        assert!(im.report.contains("TogglePortal"), "{}", im.report);
    }
    // Every colour has its own light colour, except the base (its defaults are Violet's).
    let mut distinct = lights.clone();
    distinct.dedup_by(|a, b| a == b);
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup();
    assert_eq!(distinct.len(), PORTALS.len() - 1, "{lights:?}");
}

#[test]
fn main_camera_timeline_orbits_the_spring_arm() {
    let Some(f) = fab("mixed_vfx/Content/MixedVFX/Blueprints/BP_MainCamera.uasset") else { return };
    let im = import(&f, &mut Effects::default()).unwrap();
    let p = &im.prefab;
    assert!(matches!(node(p, "Camera").kind, NodeKind::Camera { .. }));
    // The camera hangs at the end of the 12 m arm (its socket node).
    assert_eq!(node(p, "Camera").parent.as_deref(), Some("SpringArm.Socket"));
    let t = p.timelines.iter().find(|t| t.name == "Timeline_0").expect("Timeline_0");
    assert_eq!((t.length, t.looping), (100.0, true));
    let tr = t.tracks.iter().find(|t| t.node == "SpringArm").expect("SpringArm track");
    assert_eq!(tr.property, Property::Rotation);
    // Yaw from the curve, pitch and roll constant.
    assert!(matches!(tr.channels[0], Channel::Const(_)));
    assert_eq!(tr.channels[1].eval(50.0), 180.0);
    // Started from Tick and input events, which are listed.
    assert!(im.report.contains("ReceiveTick"), "{}", im.report);
}

#[test]
fn mixed_vfx_arrow_and_fire_actors_import() {
    let mut effects = Effects::default();
    let cases = [
        ("mixed_vfx/Content/MixedVFX/Blueprints/BP_Projectile.uasset", Some(("Niagara", "NS_FireArrow.fx.ron"))),
        ("perfect_fire/Content/Vefects/Free_Fire/Shared/Particles/BP_Fire.uasset", Some(("NS_Fire", "NS_Fire_Big.fx.ron"))),
        ("arrow_trail/Content/ArrowTrail/Demo/BP_Demo.uasset", None),
        ("mixed_vfx/Content/MixedVFX/Blueprints/BP_ParticleSpawner.uasset", None),
    ];
    for (rel, niagara) in cases {
        let Some(f) = fab(rel) else { return };
        let im = import(&f, &mut effects).unwrap_or_else(|e| panic!("{rel}: {e}"));
        if let Some((n, e)) = niagara {
            assert_eq!(effect_of(&im.prefab, n), e);
            assert!(im.files.contains_key(e), "{rel}: {e} not written");
        }
        // Gameplay components are listed, not imported.
        if rel.contains("Projectile") || rel.contains("BP_Demo") {
            assert!(im.report.contains("ProjectileMovement: ProjectileMovementComponent: left out"), "{}", im.report);
        }
    }
    // BP_Demo's system is set on its placed instance, and its Move event is logic.
    let im = import(&fab("arrow_trail/Content/ArrowTrail/Demo/BP_Demo.uasset").unwrap(), &mut effects).unwrap();
    assert!(matches!(node(&im.prefab, "NS_Niagara").kind, NodeKind::Empty));
    assert!(im.report.contains("custom event Move"), "{}", im.report);
}

#[test]
fn prefab_files_round_trip() {
    let Some(f) = fab("portals/Content/PortalFX/Blueprints/BP_PRT_Portal_Red.uasset") else { return };
    let im = import(&f, &mut Effects::default()).unwrap();
    let text = fx_core::prefab::to_string(&im.prefab);
    assert_eq!(fx_core::prefab::from_str(&text).unwrap(), im.prefab);
    // Every effect it names is written and checks.
    for n in &im.prefab.nodes {
        if let NodeKind::Effect { effect, .. } = &n.kind {
            let src = text_of(&im, effect);
            let checked = fx_core::check_source(effect, &src);
            assert!(checked.ok(), "{effect}: {:?}", checked.diagnostics);
        }
    }
}

fn text_of(im: &Imported, file: &str) -> String {
    text(im, file)
}
