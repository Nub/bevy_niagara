//! Renderer settings: how an emitter's particles are drawn (Niagara's
//! renderer properties). Only the Sprite renderer exists so far; Ribbon, Mesh
//! and Light renderers become further variants of
//! [`RendererSource`](crate::model::RendererSource) and
//! [`CookedRenderer`](crate::cook::CookedRenderer).
//!
//! The option enums here are shared by the source model (serde) and the
//! cooked asset (rkyv).

use serde::Serialize;

/// A unit enum written as a bare identifier in `.fx.ron`.
pub trait IdentEnum: Copy + 'static {
    /// Every variant with its source name, default first.
    const VARIANTS: &'static [(&'static str, Self)];

    fn from_name(name: &str) -> Option<Self> {
        Self::VARIANTS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
    }

    /// `` `A`, `B` or `C` ``, for messages.
    fn list() -> String {
        let names: Vec<_> = Self::VARIANTS
            .iter()
            .map(|(n, _)| format!("`{n}`"))
            .collect();
        match names.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
            _ => names.join(""),
        }
    }
}

macro_rules! ident_enum {
    ($(#[$m:meta])* $name:ident { $(#[$vm:meta])* $default:ident, $($(#[$om:meta])* $variant:ident),* $(,)? }) => {
        $(#[$m])*
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize,
            rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
        )]
        pub enum $name {
            $(#[$vm])*
            #[default]
            $default,
            $($(#[$om])* $variant,)*
        }

        impl IdentEnum for $name {
            const VARIANTS: &'static [(&'static str, Self)] = &[
                (stringify!($default), $name::$default),
                $((stringify!($variant), $name::$variant),)*
            ];
        }
    };
}

ident_enum! {
    /// How a sprite combines with what is behind it (Niagara's material blend mode).
    BlendMode {
        /// Standard alpha blending (translucent).
        Alpha,
        /// Colour is added; alpha scales it.
        Additive,
        /// Colour is already multiplied by alpha.
        Premultiplied,
    }
}

ident_enum! {
    /// Which way a sprite faces (Niagara's Facing Mode).
    FacingMode {
        /// Faces the camera's view direction.
        FaceCamera,
        /// Parallel to the camera's near plane.
        FaceCameraPlane,
        /// Faces the camera's position (rotates as it moves across the view).
        FaceCameraPosition,
        /// Faces `custom_facing_vector`.
        CustomFacingVector,
    }
}

ident_enum! {
    /// How a sprite's up axis is chosen (Niagara's Alignment).
    SpriteAlignment {
        /// Free; rotated by `Particles.SpriteRotation`.
        Unaligned,
        /// Up axis follows the particle velocity.
        VelocityAligned,
        /// Up axis follows `custom_alignment_vector`.
        CustomAlignment,
    }
}

ident_enum! {
    /// Draw order of translucent sprites (Niagara's Sort Mode).
    SortMode {
        /// Unsorted (fine for additive).
        None,
        /// Back to front by depth along the view direction.
        ViewDepth,
        /// Back to front by distance to the camera.
        ViewDistance,
    }
}

/// Names of the renderer variants, as written in `renderers: [ ... ]`.
pub const RENDERER_KINDS: &[&str] = &["Sprite"];

/// One field of a renderer, for `fxc modules` and error help.
pub struct RendererField {
    pub name: &'static str,
    pub form: &'static str,
    pub default: &'static str,
    pub doc: &'static str,
}

/// Fields of `Sprite(( ... ))`, all optional.
pub const SPRITE_FIELDS: &[RendererField] = &[
    RendererField {
        name: "blend_mode",
        form: "Alpha | Additive | Premultiplied",
        default: "Alpha",
        doc: "how sprites blend with the scene",
    },
    RendererField {
        name: "lit",
        form: "bool",
        default: "false",
        doc: "lit by scene lights instead of emissive-only",
    },
    RendererField {
        name: "facing_mode",
        form: "FaceCamera | FaceCameraPlane | FaceCameraPosition | CustomFacingVector",
        default: "FaceCamera",
        doc: "which way sprites face",
    },
    RendererField {
        name: "custom_facing_vector",
        form: "(x, y, z)",
        default: "(0.0, 0.0, 1.0)",
        doc: "facing for CustomFacingVector",
    },
    RendererField {
        name: "alignment",
        form: "Unaligned | VelocityAligned | CustomAlignment",
        default: "Unaligned",
        doc: "how the sprite's up axis is chosen",
    },
    RendererField {
        name: "custom_alignment_vector",
        form: "(x, y, z)",
        default: "(0.0, 1.0, 0.0)",
        doc: "up axis for CustomAlignment",
    },
    RendererField {
        name: "pivot_in_uv_space",
        form: "(u, v)",
        default: "(0.5, 0.5)",
        doc: "sprite pivot in UV space",
    },
    RendererField {
        name: "sub_image_size",
        form: "(columns, rows)",
        default: "(1, 1)",
        doc: "flipbook grid, each at least 1",
    },
    RendererField {
        name: "sub_uv_blending",
        form: "bool",
        default: "false",
        doc: "blend between flipbook frames",
    },
    RendererField {
        name: "sort_mode",
        form: "None | ViewDepth | ViewDistance",
        default: "None",
        doc: "draw order of sprites",
    },
    RendererField {
        name: "soft_particle_fade_distance",
        form: "metres",
        default: "0.0",
        doc: "fade near opaque surfaces; 0 = off",
    },
    RendererField {
        name: "texture",
        form: "\"asset/path.png\"",
        default: "none",
        doc: "sprite texture; none = built-in soft round sprite",
    },
];

pub fn sprite_field_names() -> &'static [&'static str] {
    const N: usize = SPRITE_FIELDS.len();
    const NAMES: [&str; N] = {
        let mut a = [""; N];
        let mut i = 0;
        while i < N {
            a[i] = SPRITE_FIELDS[i].name;
            i += 1;
        }
        a
    };
    &NAMES
}
