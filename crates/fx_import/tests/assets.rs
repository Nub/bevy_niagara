//! Materials, textures and their packages: unit checks that need no test
//! files, then the real dependency packages in testdata/niagara/packages
//! (fetched by `sh testdata/niagara_assets.sh`; skipped when absent).

use std::path::PathBuf;

use fx_import::material::{self, Blend, Material, Needs, Store};

fn packages() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/niagara/packages");
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipped: {} is absent (sh testdata/niagara_assets.sh)", dir.display());
        None
    }
}

#[test]
fn object_paths_split_into_package_and_object() {
    assert_eq!(
        material::split_object_path("Material'/Game/VFX/M_A.M_A'"),
        ("/Game/VFX/M_A".to_string(), "M_A".to_string())
    );
    assert_eq!(
        material::split_object_path("/ST_Plugin/Meshes/SM_Boid.SM_Boid"),
        ("/ST_Plugin/Meshes/SM_Boid".to_string(), "SM_Boid".to_string())
    );
    assert_eq!(material::split_object_path("/Game/A/B.B:Sub").1, "B");
}

#[test]
fn uncompressed_buffers_decode() {
    // FCompressedBuffer, method 0 (none): 64-byte big-endian header, then the bytes.
    let raw = b"hello payload";
    let mut b = Vec::new();
    b.extend_from_slice(&0xB775_6362u32.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes()); // crc
    b.extend_from_slice(&[0, 0, 0, 0]); // method none, compressor, level, block exponent
    b.extend_from_slice(&1u32.to_be_bytes()); // blocks
    b.extend_from_slice(&(raw.len() as u64).to_be_bytes());
    b.extend_from_slice(&(raw.len() as u64).to_be_bytes());
    b.extend_from_slice(&[0; 32]); // hash
    b.extend_from_slice(raw);
    assert_eq!(fx_import::bulk::decompress(&b).unwrap(), raw);
    b[0] = 0;
    assert!(fx_import::bulk::decompress(&b).is_err());
}

#[test]
fn png_round_trips_and_atlases_pack_side_by_side() {
    let img = |w, h, v: u8, srgb| material::Image { width: w, height: h, rgba: vec![v; (w * h * 4) as usize], srgb };
    let a = img(4, 2, 200, true);
    let back = material::decode_png(&material::encode_png(&a)).unwrap();
    assert_eq!((back.width, back.height, back.rgba.clone()), (4, 2, a.rgba.clone()));

    // One texture: stored as is (linear data sRGB-encoded so it samples back the same).
    let one = material::atlas(&[img(4, 4, 64, false)]);
    assert_eq!(one.rects, vec![[0.0, 0.0, 1.0, 1.0]]);
    assert!(one.image.rgba[0] > 64, "linear 64 is stored sRGB-encoded");
    assert_eq!(one.image.rgba[3], 64, "alpha stays linear");

    let at = material::atlas(&[img(8, 8, 10, true), img(16, 4, 20, true)]);
    assert_eq!(at.rects.len(), 2);
    assert_eq!(at.image.width, 8 + 16 + 4 * 2);
    assert_eq!(at.image.height, 8 + 2 * 2);
    for r in &at.rects {
        assert!(r[0] >= 0.0 && r[1] >= 0.0 && r[0] + r[2] <= 1.0 && r[1] + r[3] <= 1.0, "{r:?}");
    }
    assert!(at.rects[0][0] + at.rects[0][2] < at.rects[1][0], "tiles do not overlap");

    let n = material::stand_in_noise();
    assert_eq!(n.rgba.len(), (n.width * n.height * 4) as usize);
    assert!(n.rgba.chunks(4).map(|p| p[0]).collect::<std::collections::HashSet<_>>().len() > 50);
}

#[test]
fn texture_source_art_decodes() {
    let Some(root) = packages() else { return };
    let mut store = Store::new(vec![root]);
    let t = material::read_texture(&mut store, "/Game/VFX/Environment/_Global/Fire/Texture/T_FireParticlesSprite.T_FireParticlesSprite").unwrap();
    assert_eq!((t.width, t.height), (288, 288));
    let lit = t.rgba.chunks(4).filter(|p| p[0] > 128).count();
    assert!(lit > 100 && lit < 288 * 288, "a sprite mask, not blank or solid: {lit}");
    let s = material::read_texture(&mut store, "/Game/VFX/Environment/Clifftop/ShinyGlimmer/Texture/T_StarSpriteHighRes.T_StarSpriteHighRes").unwrap();
    assert_eq!((s.width, s.height), (1024, 1024));
    assert!(material::read_texture(&mut store, "/Game/Missing/T_Nope.T_Nope").is_err());
}

#[test]
fn material_instances_resolve_and_translate() {
    let Some(root) = packages() else { return };
    let mut store = Store::new(vec![root]);
    let m = Material::load(&mut store, "/Game/VFX/Environment/_Global/Fire/Materials/MI_FireParticlesInstance.MI_FireParticlesInstance").unwrap();
    assert_eq!(m.chain.len(), 2);
    assert_eq!(m.blend, Blend::Additive);
    assert!(m.unlit);
    let needs = Needs { dynamic_written: [true, false, false, false], dynamic_available: [true, true, false, false], mesh: false };
    let c = material::compile(&m, &needs).unwrap();
    assert_eq!(c.textures.len(), 3, "{:?}", c.textures);
    assert!(c.wesl.contains("fn fragment(in: FxVertexOutput)"));
    assert!(c.wesl.contains("in.dynamic_param"), "Power / Erosion / Distortion come from dynamic parameter 0");
    assert!(c.wesl.contains("globals.time"), "the panners move with time");
    assert!(c.wesl.contains("in.color"), "particle colour");
    assert!(c.dynamic_read[0]);

    // Unwritten dynamic parameters read the material's default value.
    let quiet = Needs { dynamic_written: [false; 4], ..needs.clone() };
    let c = material::compile(&m, &quiet).unwrap();
    assert!(!c.wesl.contains("in.dynamic_param"));
    assert!(c.wesl.contains("0.296138"), "default Erosion");

    // Masked, with a scalar parameter and the particle colour's alpha as mask.
    let b = Material::load(&mut store, "/ST_NiagaraCSIntegration/Materials/MI_Boid.MI_Boid").unwrap();
    assert_eq!(b.blend, Blend::Masked);
    let c = material::compile(&b, &Needs { mesh: true, ..needs.clone() }).unwrap();
    assert!(c.wesl.contains("discard"));
    assert!(c.textures.is_empty());

    // Lit fish: base colour lit by the scene, vertex animation listed.
    let f = Material::load(&mut store, "/Game/VFX/Environment/Clifftop/Fish/Material/MI_FishExample.MI_FishExample").unwrap();
    assert!(!f.unlit);
    let c = material::compile(&f, &Needs { mesh: true, ..needs }).unwrap();
    assert!(c.wesl.contains("scene_light"));
    assert!(c.world_position_offset);
}

#[test]
fn converted_systems_use_their_real_assets() {
    let Some(root) = packages() else { return };
    let dir = root.parent().unwrap().to_path_buf();
    let out = std::env::temp_dir().join(format!("fx_import_assets_{}", std::process::id()));
    for (name, clean) in [
        ("NS_Fire", true),
        ("NS_FireSmokeOnly", true),
        ("NS_ShinyGlimmer", true),
        ("NS_Fish", true),
        ("NS_Boids", true),
        ("NS_FluidSimulation", true),
        ("NS_GeyserEruption", false), // SM_SmokeParticle and MI_GeyserSmoke_VFX are not in the repository
        ("NS_DestructibleRig", false), // DefaultSpriteMaterial_DEBUG is not in the repository
    ] {
        let c = fx_import::convert_file(&dir.join(format!("{name}.uasset")), &out).unwrap();
        assert!(c.report.validation.is_empty(), "{name}: {:?}", c.report.validation);
        assert_eq!(c.report.placeholders.is_empty(), clean, "{name}: {:#?}", c.report.placeholders);
        let ron = std::fs::read_to_string(&c.ron_path).unwrap();
        // Every asset path in the effect exists beside it.
        for part in ron.split('"').skip(1).step_by(2) {
            if part.starts_with(&format!("{name}_assets/")) {
                assert!(out.join(part).is_file(), "{name}: {part} not written");
            }
        }
    }
    let fire = std::fs::read_to_string(out.join("NS_Fire.fx.ron")).unwrap();
    assert!(fire.contains("material: \"NS_Fire_assets/MI_FireParticlesInstance.wesl\""));
    assert!(fire.contains("texture: \"NS_Fire_assets/MI_FireParticles2Instance.png\""));
    let fish = std::fs::read_to_string(out.join("NS_Fish.fx.ron")).unwrap();
    assert!(fish.contains("mesh: \"NS_Fish_assets/SM_FishExample.glb\""));
    let report = std::fs::read_to_string(out.join("NS_Boids.report.md")).unwrap();
    assert!(!report.contains("not in the package folders"), "NM_Boids is read from its package");
    let _ = std::fs::remove_dir_all(&out);
}
