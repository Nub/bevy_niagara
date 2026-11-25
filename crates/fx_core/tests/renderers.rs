//! Emitter `renderers: [ ... ]`: decoding, defaults, cooking and errors.

mod common;

use common::*;
use fx_core::cook::{self, CookedLight, CookedMesh, CookedRenderer, CookedRibbon, CookedSprite};
use fx_core::model::{
    BlendMode, FacingMode, MeshBlendMode, MeshFacingMode, RibbonFacingMode, RibbonLinkOrderMode,
    RibbonUvMode, SortMode, SpriteAlignment,
};
use fx_core::render::{MESH_BUILTINS, RENDERER_FIELDS, RENDERER_KINDS};

fn with_renderers(renderers: &str) -> String {
    format!(
        "(
    name: \"Test\",
    emitters: [(
        name: \"E\",
        emitter_update: [(module: \"Spawn Rate\")],
        particle_update: [(module: \"Particle State\")],
        renderers: [ {renderers} ],
    )],
)"
    )
}

fn sprites(renderers: &str) -> Vec<CookedSprite> {
    compile(&with_renderers(renderers)).emitters[0]
        .renderers
        .iter()
        .map(|r| match r {
            CookedRenderer::Sprite(s) => s.clone(),
            other => panic!("expected a sprite, got {other:?}"),
        })
        .collect()
}

/// The single error for these renderers, rendered with its help.
fn error(renderers: &str) -> String {
    let c = fx_core::check_source("test.fx.ron", &with_renderers(renderers));
    assert!(!c.ok());
    let errs: Vec<_> = c.errors().collect();
    assert_eq!(errs.len(), 1, "{errs:#?}");
    assert_eq!(errs[0].span.line, 7, "{}", errs[0]);
    errs[0].to_string()
}

#[test]
fn niagara_defaults() {
    let d = CookedSprite::default();
    assert_eq!(d.blend_mode, BlendMode::Alpha);
    assert!(!d.lit);
    assert_eq!(d.facing_mode, FacingMode::FaceCamera);
    assert_eq!(d.custom_facing_vector, [0.0, 0.0, 1.0]);
    assert_eq!(d.alignment, SpriteAlignment::Unaligned);
    assert_eq!(d.custom_alignment_vector, [0.0, 1.0, 0.0]);
    assert_eq!(d.pivot_in_uv_space, [0.5, 0.5]);
    assert_eq!(d.sub_image_size, [1, 1]);
    assert!(!d.sub_uv_blending);
    assert_eq!(d.sort_mode, SortMode::None);
    assert_eq!(d.soft_particle_fade_distance, 0.0);
    assert_eq!(d.texture, None);
    assert_eq!(d.material, None);
    // Every spelling of an empty sprite gives the defaults.
    assert_eq!(
        sprites("Sprite(()), Sprite(), Sprite"),
        vec![d.clone(), d.clone(), d]
    );
}

#[test]
fn no_renderers_field_means_no_renderers() {
    let src = one_emitter("", r#"(module: "Spawn Rate")"#, "", "");
    assert!(compile(&src).emitters[0].renderers.is_empty());
    assert!(sprites("").is_empty());
}

#[test]
fn every_field_decodes_and_cooks() {
    let s = sprites(
        r#"Sprite((
            blend_mode: Premultiplied,
            lit: true,
            facing_mode: CustomFacingVector,
            custom_facing_vector: (1.0, 0.0, 0),
            alignment: CustomAlignment,
            custom_alignment_vector: (0.0, 0.0, -1.0),
            pivot_in_uv_space: (0.5, 1.0),
            sub_image_size: (8, 4),
            sub_uv_blending: true,
            sort_mode: ViewDistance,
            soft_particle_fade_distance: 0.25,
            ray_tracing: true,
            texture: "textures/spark.png",
            material: "shaders/spark.wesl",
        ))"#,
    );
    assert_eq!(
        s,
        vec![CookedSprite {
            blend_mode: BlendMode::Premultiplied,
            lit: true,
            facing_mode: FacingMode::CustomFacingVector,
            custom_facing_vector: [1.0, 0.0, 0.0],
            alignment: SpriteAlignment::CustomAlignment,
            custom_alignment_vector: [0.0, 0.0, -1.0],
            pivot_in_uv_space: [0.5, 1.0],
            sub_image_size: [8, 4],
            sub_uv_blending: true,
            sort_mode: SortMode::ViewDistance,
            soft_particle_fade_distance: 0.25,
            ray_tracing: true,
            texture: Some("textures/spark.png".into()),
            material: Some("shaders/spark.wesl".into()),
        }]
    );
}

#[test]
fn unset_fields_keep_defaults_and_order_is_kept() {
    let s = sprites(
        "Sprite((blend_mode: Additive, alignment: VelocityAligned)), Sprite(sort_mode: ViewDepth)",
    );
    assert_eq!(
        s,
        vec![
            CookedSprite {
                blend_mode: BlendMode::Additive,
                alignment: SpriteAlignment::VelocityAligned,
                ..CookedSprite::default()
            },
            CookedSprite {
                sort_mode: SortMode::ViewDepth,
                ..CookedSprite::default()
            },
        ]
    );
}

#[test]
fn renderers_survive_the_cooked_binary() {
    let sys = compile(&with_renderers(
        r#"Sprite((blend_mode: Additive, sub_image_size: (2, 3), texture: "a.png"))"#,
    ));
    let bytes = cook::to_bytes(&sys);
    assert_eq!(cook::from_bytes(&bytes).unwrap(), sys);
    assert_eq!(cook::COOKED_FORMAT_VERSION, 9);
}

#[test]
fn samples_have_their_renderers() {
    let load = |name: &str| {
        let c = fx_core::check_file(&effects_dir().join("samples").join(name)).unwrap();
        let sys = c.cooked.unwrap();
        let CookedRenderer::Sprite(s) = sys.emitters[0].renderers[0].clone() else {
            panic!("{name}: expected a sprite renderer");
        };
        assert_eq!(sys.emitters[0].renderers.len(), 1);
        s
    };
    assert_eq!(
        load("burst.fx.ron"),
        CookedSprite {
            blend_mode: BlendMode::Additive,
            alignment: SpriteAlignment::VelocityAligned,
            ..CookedSprite::default()
        }
    );
    assert_eq!(
        load("fountain.fx.ron"),
        CookedSprite {
            sort_mode: SortMode::ViewDepth,
            soft_particle_fade_distance: 0.2,
            ..CookedSprite::default()
        }
    );
    assert_eq!(
        load("smoke_curl.fx.ron"),
        CookedSprite {
            lit: true,
            sort_mode: SortMode::ViewDepth,
            soft_particle_fade_distance: 0.5,
            sub_image_size: [2, 2],
            sub_uv_blending: true,
            ..CookedSprite::default()
        }
    );
}

#[test]
fn errors_point_at_the_line_and_suggest_fixes() {
    let e = error("Sprit((blend_mode: Additive))");
    assert!(e.contains("unknown renderer `Sprit`"), "{e}");
    assert!(e.contains("did you mean `Sprite`?"), "{e}");

    let e = error("Sprite((blend_mod: Additive))");
    assert!(
        e.contains("unknown field `blend_mod` in a Sprite renderer"),
        "{e}"
    );
    assert!(e.contains("did you mean `blend_mode`?"), "{e}");

    let e = error("Sprite((blend_mode: Addative))");
    assert!(
        e.contains("`blend_mode` must be `Alpha`, `Additive` or `Premultiplied`, found `Addative`"),
        "{e}"
    );
    assert!(e.contains("did you mean `blend_mode: Additive`?"), "{e}");

    let e = error(r#"Sprite((sort_mode: "ViewDepth"))"#);
    assert!(e.contains("did you mean `sort_mode: ViewDepth`?"), "{e}");

    let e = error("Sprite((sub_image_size: (0, 2)))");
    assert!(e.contains("at least 1, found the integer 0"), "{e}");
    let e = error("Sprite((sub_image_size: (2.5, 2)))");
    assert!(e.contains("whole numbers"), "{e}");
    let e = error("Sprite((sub_image_size: (2, 2, 2)))");
    assert!(e.contains("found 3 numbers"), "{e}");

    let e = error("Sprite((soft_particle_fade_distance: -1.0))");
    assert!(e.contains("at least 0 metres"), "{e}");

    let e = error("Sprite((lit: 1))");
    assert!(e.contains("`lit` must be `true` or `false`"), "{e}");
    let e = error("Sprite((custom_facing_vector: (1.0, 0.0)))");
    assert!(e.contains("tuple of 3 numbers"), "{e}");
    let e = error("Sprite((texture: Some(\"a.png\")))");
    assert!(e.contains("asset path"), "{e}");
    let e = error("Sprite((blend_mode: Additive), (lit: true))");
    assert!(e.contains("takes one struct"), "{e}");

    let e = error("Ribon(())");
    assert!(e.contains("unknown renderer `Ribon`"), "{e}");
    assert!(e.contains("did you mean `Ribbon`?"), "{e}");
    let e = error("Particles(())");
    assert!(
        e.contains("renderers are: Sprite, Ribbon, Mesh, Light"),
        "{e}"
    );
    assert!(!e.contains("not supported"), "{e}");
    let e = error("Sprite((material: 3))");
    assert!(e.contains("`material` must be an asset path"), "{e}");
}

#[test]
fn raw_identifiers_parse() {
    // `ron` writes `SortMode::None` as `r#None`; tools' output must read back.
    let s = sprites("Sprite((sort_mode: r#ViewDepth)), Sprite((sort_mode: r#None))");
    assert_eq!(s[0].sort_mode, SortMode::ViewDepth);
    assert_eq!(s[1].sort_mode, SortMode::None);
    // A raw `r#true` is an identifier, not a bool.
    let e = error("Sprite((lit: r#true))");
    assert!(
        e.contains("`lit` must be `true` or `false`, found `true`"),
        "{e}"
    );
}

fn renderers(renderers: &str) -> Vec<CookedRenderer> {
    compile(&with_renderers(renderers)).emitters[0]
        .renderers
        .clone()
}

#[test]
fn ribbon_mesh_light_defaults() {
    let r = CookedRibbon::default();
    assert_eq!(r.blend_mode, BlendMode::Alpha);
    assert_eq!(r.facing_mode, RibbonFacingMode::Screen);
    assert_eq!(r.custom_facing_vector, [0.0, 0.0, 1.0]);
    assert_eq!(r.uv_mode, RibbonUvMode::ScaledUniformly);
    assert_eq!(r.tiling_length, 1.0);
    assert_eq!(r.link_order, RibbonLinkOrderMode::SpawnOrder);
    assert_eq!(r.tessellation_factor, 1);
    assert_eq!((r.texture.clone(), r.material.clone()), (None, None));

    let m = CookedMesh::default();
    assert_eq!(m.mesh, "cube");
    assert_eq!(m.blend_mode, MeshBlendMode::Opaque);
    assert_eq!(m.facing_mode, MeshFacingMode::Default);
    assert!(m.lit);
    assert_eq!(m.sort_mode, SortMode::None);
    assert_eq!((m.texture.clone(), m.material.clone()), (None, None));

    let l = CookedLight::default();
    assert_eq!(l.radius_scale, 1.0);
    assert_eq!(l.color_add, [0.0; 3]);
    assert_eq!(l.intensity, 10000.0);
    assert!(l.alpha_scales_brightness);
    assert_eq!(l.max_lights, 16);

    assert_eq!(
        renderers("Ribbon, Mesh(()), Light()"),
        vec![
            CookedRenderer::Ribbon(r),
            CookedRenderer::Mesh(m),
            CookedRenderer::Light(l)
        ]
    );
}

#[test]
fn every_ribbon_mesh_light_field_decodes() {
    let rs = renderers(
        r#"Ribbon((
            blend_mode: Additive,
            facing_mode: CustomFacingVector,
            custom_facing_vector: (0.0, 1.0, 0.0),
            uv_mode: TiledOverRibbonLength,
            tiling_length: 2.5,
            link_order: Custom,
            tessellation_factor: 16,
            ray_tracing: true,
            texture: "textures/trail.png",
            material: "shaders/trail.wesl",
        )),
        Mesh((
            mesh: "meshes/rock.glb",
            blend_mode: Premultiplied,
            facing_mode: CameraPlane,
            lit: false,
            sort_mode: ViewDistance,
            texture: "textures/rock.png",
            material: "shaders/rock.wesl",
        )),
        Mesh((mesh: "torus")),
        Light((
            radius_scale: 0.0,
            color_add: (0.1, 0.2, 0.3),
            intensity: 500,
            alpha_scales_brightness: false,
            max_lights: 256,
        ))"#,
    );
    assert_eq!(
        rs,
        vec![
            CookedRenderer::Ribbon(CookedRibbon {
                blend_mode: BlendMode::Additive,
                facing_mode: RibbonFacingMode::CustomFacingVector,
                custom_facing_vector: [0.0, 1.0, 0.0],
                uv_mode: RibbonUvMode::TiledOverRibbonLength,
                tiling_length: 2.5,
                link_order: RibbonLinkOrderMode::Custom,
                tessellation_factor: 16,
                ray_tracing: true,
                texture: Some("textures/trail.png".into()),
                material: Some("shaders/trail.wesl".into()),
            }),
            CookedRenderer::Mesh(CookedMesh {
                mesh: "meshes/rock.glb".into(),
                blend_mode: MeshBlendMode::Premultiplied,
                facing_mode: MeshFacingMode::CameraPlane,
                lit: false,
                sort_mode: SortMode::ViewDistance,
                texture: Some("textures/rock.png".into()),
                material: Some("shaders/rock.wesl".into()),
            }),
            CookedRenderer::Mesh(CookedMesh {
                mesh: "torus".into(),
                ..CookedMesh::default()
            }),
            CookedRenderer::Light(CookedLight {
                radius_scale: 0.0,
                color_add: [0.1, 0.2, 0.3],
                intensity: 500.0,
                alpha_scales_brightness: false,
                max_lights: 256,
            }),
        ]
    );
    // And they survive the cooked binary.
    let sys = compile(&with_renderers(
        r#"Ribbon((link_order: Custom)), Mesh((mesh: "sphere")), Light((max_lights: 4))"#,
    ));
    assert_eq!(cook::from_bytes(&cook::to_bytes(&sys)).unwrap(), sys);
}

#[test]
fn every_builtin_mesh_is_accepted() {
    for name in MESH_BUILTINS {
        let rs = renderers(&format!("Mesh((mesh: \"{name}\"))"));
        assert!(matches!(&rs[0], CookedRenderer::Mesh(m) if m.mesh == *name));
    }
}

#[test]
fn field_tables_list_every_renderer() {
    let kinds: Vec<_> = RENDERER_FIELDS.iter().map(|(k, _)| *k).collect();
    assert_eq!(kinds, RENDERER_KINDS);
    // Every listed field decodes (its absence of an "unknown field" error).
    for (kind, fields) in RENDERER_FIELDS {
        for f in *fields {
            let c = fx_core::check_source(
                "test.fx.ron",
                &with_renderers(&format!("{kind}(({}: ()))", f.name)),
            );
            assert!(
                c.errors().all(|e| !e.message.contains("unknown field")),
                "{kind}.{}",
                f.name
            );
        }
    }
}

#[test]
fn ribbon_errors() {
    let e = error("Ribbon((tessellation_factor: 0))");
    assert!(
        e.contains(
            "`tessellation_factor` must be a whole number from 1 to 16, found the integer 0"
        ),
        "{e}"
    );
    let e = error("Ribbon((tessellation_factor: 17))");
    assert!(e.contains("from 1 to 16"), "{e}");
    let e = error("Ribbon((tiling_length: 0.0))");
    assert!(
        e.contains("`tiling_length` must be a length of more than 0 metres"),
        "{e}"
    );
    let e = error("Ribbon((link_order: Spawn))");
    assert!(
        e.contains("`link_order` must be `SpawnOrder` or `Custom`"),
        "{e}"
    );
    let e = error("Ribbon((uv_mode: TiledOverRibonLength))");
    assert!(
        e.contains("did you mean `uv_mode: TiledOverRibbonLength`?"),
        "{e}"
    );
    let e = error("Ribbon((facing_mod: Screen))");
    assert!(
        e.contains("unknown field `facing_mod` in a Ribbon renderer"),
        "{e}"
    );
    assert!(e.contains("did you mean `facing_mode`?"), "{e}");
}

#[test]
fn mesh_errors() {
    let e = error(r#"Mesh((mesh: "sphre"))"#);
    assert!(e.contains("unknown built-in mesh \"sphre\""), "{e}");
    assert!(e.contains("did you mean `mesh: \"sphere\"`?"), "{e}");
    let e = error(r#"Mesh((mesh: "blob"))"#);
    assert!(e.contains("built-in meshes: cube, sphere"), "{e}");
    let e = error("Mesh((mesh: cube))");
    assert!(e.contains("write it in quotes: `mesh: \"cube\"`"), "{e}");
    let e = error("Mesh((blend_mode: Translucent))");
    assert!(
        e.contains("`blend_mode` must be `Opaque`, `Alpha`, `Additive` or `Premultiplied`"),
        "{e}"
    );
    let e = error("Mesh((facing_mode: FaceCamera))");
    assert!(
        e.contains("`facing_mode` must be `Default`, `Velocity`"),
        "{e}"
    );
}

#[test]
fn light_errors() {
    let e = error("Light((max_lights: 0))");
    assert!(
        e.contains("`max_lights` must be a whole number from 1 to 256"),
        "{e}"
    );
    let e = error("Light((max_lights: 300))");
    assert!(e.contains("from 1 to 256"), "{e}");
    let e = error("Light((intensity: -1.0))");
    assert!(
        e.contains("`intensity` must be at least 0 lumens per unit colour"),
        "{e}"
    );
    let e = error("Light((radius_scale: -0.5))");
    assert!(
        e.contains("`radius_scale` must be a number of at least 0"),
        "{e}"
    );
    let e = error("Light((color_add: (1.0, 1.0)))");
    assert!(
        e.contains("`color_add` must be a tuple of 3 numbers"),
        "{e}"
    );
    let e = error("Light((texture: \"a.png\"))");
    assert!(
        e.contains("unknown field `texture` in a Light renderer"),
        "{e}"
    );
}

#[test]
fn renderers_decide_extended_attributes() {
    let ext = |r: &str| compile(&with_renderers(r)).emitters[0].uses_extended_attributes();
    assert!(!ext(""));
    assert!(!ext("Sprite"));
    assert!(ext("Sprite, Ribbon"));
    assert!(ext("Mesh"));
    assert!(ext("Light"));
}

#[test]
fn new_renderers_serialise_to_ron_that_parses_back() {
    let text = with_renderers(
        r#"Sprite((material: "m.wesl")),
        Ribbon((uv_mode: ScaledUsingRibbonSegmentLength, tiling_length: 0.5, texture: "t.png")),
        Mesh((mesh: "capsule", blend_mode: Alpha, facing_mode: Velocity, sort_mode: ViewDepth)),
        Light((color_add: (0.5, 0.0, 0.25), intensity: 250.0, max_lights: 32))"#,
    );
    let c = fx_core::check_source("test.fx.ron", &text);
    let src = c.source.unwrap();
    let ron_text = ron::ser::to_string_pretty(&src, ron::ser::PrettyConfig::default()).unwrap();
    let again = fx_core::check_source("roundtrip.fx.ron", &ron_text);
    assert!(again.ok(), "{ron_text}");
    assert_eq!(again.cooked.unwrap(), c.cooked.unwrap());
}
