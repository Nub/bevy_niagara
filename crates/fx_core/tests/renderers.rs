//! Emitter `renderers: [ ... ]`: decoding, defaults, cooking and errors.

mod common;

use common::*;
use fx_core::cook::{self, CookedRenderer, CookedSprite};
use fx_core::model::{BlendMode, FacingMode, SortMode, SpriteAlignment};

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
            texture: "textures/spark.png",
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
            texture: Some("textures/spark.png".into()),
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
    assert_eq!(cook::COOKED_FORMAT_VERSION, 2);
}

#[test]
fn samples_have_their_renderers() {
    let load = |name: &str| {
        let c = fx_core::check_file(&effects_dir().join("samples").join(name)).unwrap();
        let sys = c.cooked.unwrap();
        let CookedRenderer::Sprite(s) = sys.emitters[0].renderers[0].clone();
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

    let e = error("Ribbon(())");
    assert!(e.contains("unknown renderer `Ribbon`"), "{e}");
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
