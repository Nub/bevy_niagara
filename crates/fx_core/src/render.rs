//! Renderer settings: how an emitter's particles are drawn (Niagara's
//! renderer properties): Sprite, Ribbon, Mesh, Light, Decal, Component and Volumetric renderers, the
//! variants of [`RendererSource`](crate::model::RendererSource) and
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

pub(crate) use ident_enum;

ident_enum! {
    /// How a sprite or ribbon combines with what is behind it (Niagara's material blend mode).
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
ident_enum! {
    /// Which way a ribbon's surface faces (Niagara's Ribbon Facing Mode).
    RibbonFacingMode {
        /// Turned toward the camera around the ribbon direction.
        Screen,
        /// Its normal follows `custom_facing_vector`.
        CustomFacingVector,
    }
}

ident_enum! {
    /// How texture U runs along a ribbon (Niagara's UV0 Leading/Trailing mode).
    RibbonUvMode {
        /// U goes 0..1 over the whole ribbon, the same step per segment.
        ScaledUniformly,
        /// U goes 0..1 over the whole ribbon, each segment's share by its length.
        ScaledUsingRibbonSegmentLength,
        /// U repeats every `tiling_length` metres.
        TiledOverRibbonLength,
    }
}

ident_enum! {
    /// Order in which a ribbon links its particles (Niagara's Ribbon Link Order).
    RibbonLinkOrderMode {
        /// By `Particles.UniqueID` (spawn order).
        SpawnOrder,
        /// By `Particles.RibbonLinkOrder`, ascending.
        Custom,
    }
}

ident_enum! {
    /// How a mesh combines with the scene.
    MeshBlendMode {
        /// Opaque, depth-written.
        Opaque,
        /// Standard alpha blending (translucent).
        Alpha,
        /// Colour is added; alpha scales it.
        Additive,
        /// Colour is already multiplied by alpha.
        Premultiplied,
    }
}

ident_enum! {
    /// Which way a mesh's X axis faces (Niagara's Mesh Facing Mode).
    MeshFacingMode {
        /// `Particles.MeshOrientation` only.
        Default,
        /// X axis along the particle velocity.
        Velocity,
        /// X axis toward the camera's position.
        CameraPosition,
        /// X axis against the camera's view direction.
        CameraPlane,
    }
}

/// Built-in meshes a Mesh renderer can name instead of an asset path.
pub const MESH_BUILTINS: &[&str] = &["cube", "sphere", "torus", "cylinder", "cone", "capsule"];

/// Whether `mesh` names an asset file rather than a built-in shape.
pub fn is_mesh_asset_path(mesh: &str) -> bool {
    mesh.contains('.') || mesh.contains('/')
}

/// Names of the renderer variants, as written in `renderers: [ ... ]`.
pub const RENDERER_KINDS: &[&str] = &[
    "Sprite",
    "Ribbon",
    "Mesh",
    "Light",
    "Decal",
    "Component",
    "Volumetric",
];

/// One field of a renderer, for `fxc modules` and error help.
pub struct RendererField {
    pub name: &'static str,
    pub form: &'static str,
    pub default: &'static str,
    pub doc: &'static str,
}

const fn field(
    name: &'static str,
    form: &'static str,
    default: &'static str,
    doc: &'static str,
) -> RendererField {
    RendererField {
        name,
        form,
        default,
        doc,
    }
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
        name: "distortion",
        form: "bool",
        default: "false",
        doc: "draw after the opaque scene; the material can sample it (view_transmission_texture) to refract it",
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
        name: "ray_tracing",
        form: "bool",
        default: "false",
        doc: "Solari: seen in reflections and bounce light, casts shadows, emissive sprites light the scene (needs fx_solari)",
    },
    RendererField {
        name: "ray_tracing_max_particles",
        form: "1..=16383",
        default: "24",
        doc: "Solari: most particles rays see; an even subset with raised opacity stands for the rest",
    },
    RendererField {
        name: "ray_tracing_lights",
        form: "bool",
        default: "false",
        doc: "Solari: also sample emissive sprites as lights (less noise, costs more); off = they light only via bounce rays",
    },
    RendererField {
        name: "texture",
        form: "\"asset/path.png\"",
        default: "none",
        doc: "sprite texture; none = built-in soft round sprite",
    },
    RendererField {
        name: "material",
        form: "\"asset/path.wesl\"",
        default: "none",
        doc: "custom WESL fragment shader; none = built-in",
    },
];

/// Fields of `Ribbon(( ... ))`, all optional.
pub const RIBBON_FIELDS: &[RendererField] = &[
    field(
        "blend_mode",
        "Alpha | Additive | Premultiplied",
        "Alpha",
        "how the ribbon blends with the scene",
    ),
    field(
        "facing_mode",
        "Screen | CustomFacingVector",
        "Screen",
        "which way the ribbon surface faces",
    ),
    field(
        "custom_facing_vector",
        "(x, y, z)",
        "(0.0, 0.0, 1.0)",
        "surface normal for CustomFacingVector",
    ),
    field(
        "uv_mode",
        "ScaledUniformly | ScaledUsingRibbonSegmentLength | TiledOverRibbonLength",
        "ScaledUniformly",
        "how texture U runs along the ribbon",
    ),
    field(
        "tiling_length",
        "metres",
        "1.0",
        "texture repeat length for TiledOverRibbonLength; more than 0",
    ),
    field(
        "link_order",
        "SpawnOrder | Custom",
        "SpawnOrder",
        "particle order along the ribbon; Custom = by Particles.RibbonLinkOrder",
    ),
    field(
        "tessellation_factor",
        "1..=16",
        "1",
        "subdivisions per segment",
    ),
    field(
        "ray_tracing",
        "bool",
        "false",
        "Solari: seen in reflections and bounce light, casts shadows, lights the scene (needs fx_solari)",
    ),
    field(
        "ray_tracing_lights",
        "bool",
        "false",
        "Solari: also sample the ribbon as a light (less noise, costs more); off = it lights only via bounce rays",
    ),
    field(
        "texture",
        "\"asset/path.png\"",
        "none",
        "ribbon texture; none = plain colour",
    ),
    field(
        "material",
        "\"asset/path.wesl\"",
        "none",
        "custom WESL fragment shader; none = built-in",
    ),
];

/// Fields of `Mesh(( ... ))`, all optional.
pub const MESH_FIELDS: &[RendererField] = &[
    field(
        "mesh",
        "\"cube\" | \"sphere\" | \"torus\" | \"cylinder\" | \"cone\" | \"capsule\" | \"asset/path.glb\"",
        "\"cube\"",
        "a built-in shape or a mesh asset path",
    ),
    field(
        "blend_mode",
        "Opaque | Alpha | Additive | Premultiplied",
        "Opaque",
        "how meshes blend with the scene",
    ),
    field(
        "facing_mode",
        "Default | Velocity | CameraPosition | CameraPlane",
        "Default",
        "which way the mesh's X axis faces",
    ),
    field(
        "lit",
        "bool",
        "true",
        "lit by scene lights instead of emissive-only",
    ),
    field(
        "sort_mode",
        "None | ViewDepth | ViewDistance",
        "None",
        "draw order of translucent meshes",
    ),
    field(
        "ray_tracing",
        "bool",
        "false",
        "Solari: seen in reflections and bounce light, casts shadows; unlit/additive meshes light the scene (needs fx_solari)",
    ),
    field(
        "ray_tracing_max_particles",
        "1..=16383",
        "256",
        "Solari: most meshes rays see (an even subset past it; capped at 65535 triangles per emitter)",
    ),
    field(
        "ray_tracing_lights",
        "bool",
        "false",
        "Solari: also sample unlit/additive meshes as lights (less noise, costs more)",
    ),
    field(
        "texture",
        "\"asset/path.png\"",
        "none",
        "base colour texture; none = plain colour",
    ),
    field(
        "material",
        "\"asset/path.wesl\"",
        "none",
        "custom WESL fragment shader; none = built-in",
    ),
];

/// Fields of `Light(( ... ))`, all optional.
pub const LIGHT_FIELDS: &[RendererField] = &[
    field(
        "radius_scale",
        "number >= 0",
        "1.0",
        "multiplies Particles.LightRadius",
    ),
    field(
        "color_add",
        "(r, g, b)",
        "(0.0, 0.0, 0.0)",
        "added to the particle colour",
    ),
    field(
        "intensity",
        "lumens per unit colour, >= 0",
        "10000.0",
        "brightness of a white particle",
    ),
    field(
        "alpha_scales_brightness",
        "bool",
        "true",
        "particle alpha multiplies brightness",
    ),
    field(
        "max_lights",
        "1..=256",
        "16",
        "most lights drawn per emitter (the closest ones win)",
    ),
];

/// Fields of `Decal(( ... ))`, all optional.
pub const DECAL_FIELDS: &[RendererField] = &[
    field(
        "size",
        "(x, y, z) metres",
        "(1.0, 0.5, 1.0)",
        "decal box (Y = projection depth), times Particles.Scale; turned by Particles.MeshOrientation",
    ),
    field(
        "blend_mode",
        "Alpha | Additive | Premultiplied",
        "Alpha",
        "how the decal blends with the surface",
    ),
    field(
        "lit",
        "bool",
        "false",
        "lit by scene lights (normal = the box's up axis) instead of emissive-only",
    ),
    field(
        "fade_start",
        "0..=1",
        "0.75",
        "Particles.NormalizedAge at which the decal fades out; 1 = no fade with age",
    ),
    field(
        "texture",
        "\"asset/path.png\"",
        "none",
        "decal texture (projected on the box's XZ); none = built-in soft round sprite",
    ),
];

/// Fields of `Component(( ... ))`, all optional.
pub const COMPONENT_FIELDS: &[RendererField] = &[
    field(
        "component",
        "\"Name\"",
        "\"Particle\"",
        "name a game registers its component under (fx_render add_particle_component)",
    ),
    field(
        "max_components",
        "1..=1024",
        "32",
        "most entities driven per effect instance (the oldest particles win)",
    ),
    field(
        "transform",
        "bool",
        "true",
        "the entity's Transform follows Position, MeshOrientation and Scale",
    ),
];

/// Fields of `Volumetric(( ... ))`, all optional.
pub const VOLUMETRIC_FIELDS: &[RendererField] = &[
    field(
        "resolution",
        "8..=128",
        "48",
        "density grid cells per axis",
    ),
    field(
        "bounds",
        "((min x, y, z), (max x, y, z))",
        "emitter fixed_bounds, else ±bounds_radius",
        "volume box in effect space",
    ),
    field(
        "density_scale",
        "number >= 0",
        "4.0",
        "extinction per metre at a particle's centre (times Color alpha)",
    ),
    field(
        "radius_scale",
        "number >= 0",
        "0.5",
        "splat radius as a fraction of Particles.SpriteSize.x",
    ),
    field(
        "lit",
        "bool",
        "true",
        "scatters the scene's ambient and directional light",
    ),
    field(
        "emissive",
        "number >= 0",
        "0.0",
        "particle colour emitted per unit density (fire, glowing smoke)",
    ),
    field(
        "steps",
        "8..=256",
        "64",
        "ray-march steps across the volume",
    ),
];

/// `(renderer name, its fields)`, in [`RENDERER_KINDS`] order.
pub const RENDERER_FIELDS: &[(&str, &[RendererField])] = &[
    ("Sprite", SPRITE_FIELDS),
    ("Ribbon", RIBBON_FIELDS),
    ("Mesh", MESH_FIELDS),
    ("Light", LIGHT_FIELDS),
    ("Decal", DECAL_FIELDS),
    ("Component", COMPONENT_FIELDS),
    ("Volumetric", VOLUMETRIC_FIELDS),
];

macro_rules! field_names {
    ($fn_name:ident, $table:ident) => {
        #[doc = concat!("Names of [`", stringify!($table), "`].")]
        pub fn $fn_name() -> &'static [&'static str] {
            const N: usize = $table.len();
            const NAMES: [&str; N] = {
                let mut a = [""; N];
                let mut i = 0;
                while i < N {
                    a[i] = $table[i].name;
                    i += 1;
                }
                a
            };
            &NAMES
        }
    };
}

field_names!(sprite_field_names, SPRITE_FIELDS);
field_names!(ribbon_field_names, RIBBON_FIELDS);
field_names!(mesh_field_names, MESH_FIELDS);
field_names!(light_field_names, LIGHT_FIELDS);
field_names!(decal_field_names, DECAL_FIELDS);
field_names!(component_field_names, COMPONENT_FIELDS);
field_names!(volumetric_field_names, VOLUMETRIC_FIELDS);
