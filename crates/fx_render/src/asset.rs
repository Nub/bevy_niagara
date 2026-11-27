//! Effects as Bevy assets: `.fx.ron` source (checked and cooked on load) or
//! cooked `.fxb` bytes, read through the asset server, so they come from
//! disk natively and over HTTP in the browser. Texture, material and mesh
//! paths in the effect resolve next to the effect file.

use std::sync::Arc;

use bevy::asset::io::Reader;
use bevy::asset::{AssetLoader, LoadContext};
use bevy::prelude::*;
use fx_core::CookedSystem;
use fx_core::cook::CookedRenderer;

/// A loaded effect. Spawn it with [`ParticleEffect::shared`](crate::ParticleEffect::shared)
/// (`effect.system.clone()`), so instances share one program and GPU pool.
#[derive(Asset, TypePath, Clone)]
pub struct FxEffectAsset {
    pub system: Arc<CookedSystem>,
}

#[derive(Default, TypePath)]
pub struct FxEffectLoader;

impl AssetLoader for FxEffectLoader {
    type Asset = FxEffectAsset;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<FxEffectAsset, std::io::Error> {
        let err = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let path = load_context.path().clone();
        let name = path.to_string();
        let mut system = if name.ends_with(fx_core::cook::COOKED_EXTENSION) {
            fx_core::cook::from_bytes(&bytes).map_err(|e| err(format!("{name}: {e}")))?
        } else {
            let text = String::from_utf8(bytes).map_err(|e| err(format!("{name}: {e}")))?;
            let checked = fx_core::check_source(&name, &text);
            for d in &checked.diagnostics {
                warn!("{d}");
            }
            checked
                .cooked
                .ok_or_else(|| err(format!("{name} has errors")))?
        };
        let resolve = |p: &mut String| {
            if let Ok(full) = path.resolve_embed_str(p) {
                *p = full.to_string();
            }
        };
        for r in system.emitters.iter_mut().flat_map(|e| e.renderers.iter_mut()) {
            match r {
                CookedRenderer::Sprite(s) => {
                    s.texture.as_mut().map(resolve);
                    s.material.as_mut().map(resolve);
                }
                CookedRenderer::Ribbon(s) => {
                    s.texture.as_mut().map(resolve);
                    s.material.as_mut().map(resolve);
                }
                CookedRenderer::Mesh(s) => {
                    s.texture.as_mut().map(resolve);
                    s.material.as_mut().map(resolve);
                    if !fx_core::render::MESH_BUILTINS.contains(&s.mesh.as_str()) {
                        resolve(&mut s.mesh);
                    }
                }
                CookedRenderer::Decal(s) => {
                    s.texture.as_mut().map(resolve);
                }
                CookedRenderer::Light(_) | CookedRenderer::Component(_) | CookedRenderer::Volumetric(_) => {}
            }
        }
        Ok(FxEffectAsset {
            system: Arc::new(system),
        })
    }

    fn extensions(&self) -> &[&str] {
        &["fx.ron", fx_core::cook::COOKED_EXTENSION]
    }
}
