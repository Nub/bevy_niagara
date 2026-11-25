//! System-level lifecycle and scalability settings (Niagara's Effect Type):
//! distance and visibility culling, an instance budget, spawn-count scaling by
//! distance, and per-quality-level overrides.
//!
//! The effect file holds an optional `scalability: ( ... )` block
//! ([`ScalabilitySource`], decoded in [`crate::decode`]); cooking gives
//! [`CookedScalability`]. The game runtime picks a [`QualityLevel`] and calls
//! [`CookedScalability::resolve`] and [`CookedScalability::spawn_scale_at`].

use serde::Serialize;
use serde::ser::{SerializeStruct, Serializer};

use crate::model::Span;

/// Default `bounds_radius` (m).
pub const DEFAULT_BOUNDS_RADIUS: f32 = 2.0;
/// Default `visibility_grace` (s).
pub const DEFAULT_VISIBILITY_GRACE: f32 = 0.5;

/// Fields of the `scalability: ( ... )` block.
pub const SCALABILITY_FIELDS: &[&str] = &[
    "max_distance",
    "spawn_count_scale_by_distance",
    "cull_when_not_visible",
    "visibility_grace",
    "bounds_radius",
    "max_instances",
    "quality",
];
/// Fields of one quality level entry.
pub const QUALITY_FIELDS: &[&str] = &["spawn_count_scale", "max_distance", "max_instances"];
/// Quality level names, the fields of `quality: ( ... )`.
pub const QUALITY_NAMES: &[&str] = &["Low", "Medium", "High", "Epic", "Cinematic"];

/// A scalability quality level, picked by the game at runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QualityLevel {
    Low,
    Medium,
    #[default]
    High,
    Epic,
    Cinematic,
}

impl QualityLevel {
    pub const ALL: [QualityLevel; 5] = [
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Epic,
        QualityLevel::Cinematic,
    ];

    /// Position in [`Self::ALL`] (and in [`CookedScalability::quality`]).
    pub fn index(self) -> usize {
        self as usize
    }

    /// The name written in `.fx.ron`.
    pub fn name(self) -> &'static str {
        QUALITY_NAMES[self.index()]
    }

    pub fn from_name(name: &str) -> Option<QualityLevel> {
        QUALITY_NAMES
            .iter()
            .position(|n| *n == name)
            .map(|i| QualityLevel::ALL[i])
    }
}

/// One entry of `quality: ( Low: ( ... ), ... )`; unset fields keep the base value.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualitySource {
    /// Multiplies every spawn request; at least 0.
    #[serde(skip_serializing_if = "is_one")]
    pub spawn_count_scale: f32,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_plain"
    )]
    pub max_distance: Option<f32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_plain"
    )]
    pub max_instances: Option<u32>,
    #[serde(skip)]
    pub span: Span,
}

impl Default for QualitySource {
    fn default() -> Self {
        QualitySource {
            spawn_count_scale: 1.0,
            max_distance: None,
            max_instances: None,
            span: Span::default(),
        }
    }
}

impl QualitySource {
    fn is_default(&self) -> bool {
        self.spawn_count_scale == 1.0 && self.max_distance.is_none() && self.max_instances.is_none()
    }
}

/// The `scalability: ( ... )` block, with defaults filled in for fields the
/// source leaves out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScalabilitySource {
    /// Metres from the nearest camera; `f32::INFINITY` when unset.
    #[serde(skip_serializing_if = "is_infinite")]
    pub max_distance: f32,
    /// (distance m, scale), distances strictly increasing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spawn_count_scale_by_distance: Vec<(f32, f32)>,
    pub cull_when_not_visible: bool,
    pub visibility_grace: f32,
    pub bounds_radius: f32,
    /// 0 = unlimited.
    pub max_instances: u32,
    /// Indexed by [`QualityLevel::index`].
    #[serde(serialize_with = "serialize_quality")]
    pub quality: [QualitySource; 5],
    #[serde(skip)]
    pub span: Span,
}

impl Default for ScalabilitySource {
    fn default() -> Self {
        ScalabilitySource {
            max_distance: f32::INFINITY,
            spawn_count_scale_by_distance: Vec::new(),
            cull_when_not_visible: false,
            visibility_grace: DEFAULT_VISIBILITY_GRACE,
            bounds_radius: DEFAULT_BOUNDS_RADIUS,
            max_instances: 0,
            quality: Default::default(),
            span: Span::default(),
        }
    }
}

fn is_one(x: &f32) -> bool {
    *x == 1.0
}

fn is_infinite(x: &f32) -> bool {
    x.is_infinite()
}

/// Written as the bare value (the source form), not `Some(...)`.
pub(crate) fn serialize_plain<T: Serialize, S: Serializer>(
    t: &Option<T>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match t {
        Some(v) => v.serialize(s),
        None => s.serialize_none(),
    }
}

/// Only the levels that override something, as `(Low: ( ... ), ...)`.
fn serialize_quality<S: Serializer>(q: &[QualitySource; 5], s: S) -> Result<S::Ok, S::Error> {
    let set: Vec<usize> = (0..5).filter(|&i| !q[i].is_default()).collect();
    let mut st = s.serialize_struct("Quality", set.len())?;
    for i in set {
        st.serialize_field(QUALITY_NAMES[i], &q[i])?;
    }
    st.end()
}

/// Per-quality-level overrides, cooked.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedQuality {
    /// Multiplies every spawn request.
    pub spawn_count_scale: f32,
    /// Replaces [`CookedScalability::max_distance`] when set.
    pub max_distance: Option<f32>,
    /// Replaces [`CookedScalability::max_instances`] when set.
    pub max_instances: Option<u32>,
}

impl Default for CookedQuality {
    fn default() -> Self {
        CookedQuality {
            spawn_count_scale: 1.0,
            max_distance: None,
            max_instances: None,
        }
    }
}

/// System scalability settings in the cooked asset.
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CookedScalability {
    /// Metres from the nearest camera to the bounds sphere beyond which an
    /// instance is culled; `f32::INFINITY` = never.
    pub max_distance: f32,
    /// (distance m, scale) keys, distances strictly increasing; piecewise
    /// linear, clamped at the ends. Empty = scale 1.
    pub spawn_count_scale_by_distance: Vec<(f32, f32)>,
    /// Cull instances that have been out of every view for `visibility_grace`.
    pub cull_when_not_visible: bool,
    /// Seconds out of every view before the visibility cull applies.
    pub visibility_grace: f32,
    /// Metres; sphere around the effect origin used for distance and visibility.
    pub bounds_radius: f32,
    /// Most live instances of this system, nearest kept; 0 = unlimited.
    pub max_instances: u32,
    /// Indexed by [`QualityLevel::index`].
    pub quality: [CookedQuality; 5],
}

impl Default for CookedScalability {
    fn default() -> Self {
        CookedScalability {
            max_distance: f32::INFINITY,
            spawn_count_scale_by_distance: Vec::new(),
            cull_when_not_visible: false,
            visibility_grace: DEFAULT_VISIBILITY_GRACE,
            bounds_radius: DEFAULT_BOUNDS_RADIUS,
            max_instances: 0,
            quality: Default::default(),
        }
    }
}

/// The settings that depend on the quality level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedScalability {
    /// Metres; `f32::INFINITY` = no distance cull.
    pub max_distance: f32,
    /// Multiplies every spawn request (times [`CookedScalability::spawn_scale_at`]).
    pub spawn_count_scale: f32,
    /// 0 = unlimited.
    pub max_instances: u32,
}

impl CookedScalability {
    /// The values for `level`: its overrides over the base values.
    pub fn resolve(&self, level: QualityLevel) -> ResolvedScalability {
        let q = &self.quality[level.index()];
        ResolvedScalability {
            max_distance: q.max_distance.unwrap_or(self.max_distance),
            spawn_count_scale: q.spawn_count_scale,
            max_instances: q.max_instances.unwrap_or(self.max_instances),
        }
    }

    /// The spawn-count scale at `distance` metres from the camera: the
    /// piecewise-linear curve, clamped at its ends; 1 when there is none.
    pub fn spawn_scale_at(&self, distance: f32) -> f32 {
        let k = &self.spawn_count_scale_by_distance;
        let (Some(first), Some(last)) = (k.first(), k.last()) else {
            return 1.0;
        };
        if distance <= first.0 {
            return first.1;
        }
        if distance >= last.0 {
            return last.1;
        }
        let i = k.partition_point(|p| p.0 <= distance); // 1..len
        let ((d0, s0), (d1, s1)) = (k[i - 1], k[i]);
        let f = if d1 > d0 { (distance - d0) / (d1 - d0) } else { 0.0 };
        s0 + (s1 - s0) * f
    }

    /// No culling, no budget, no scaling at any level.
    pub fn is_default(&self) -> bool {
        *self == CookedScalability::default()
    }
}

/// Cooks the source block (already validated by the decoder).
pub(crate) fn cook(src: Option<&ScalabilitySource>) -> CookedScalability {
    let Some(s) = src else {
        return CookedScalability::default();
    };
    CookedScalability {
        max_distance: s.max_distance,
        spawn_count_scale_by_distance: s.spawn_count_scale_by_distance.clone(),
        cull_when_not_visible: s.cull_when_not_visible,
        visibility_grace: s.visibility_grace,
        bounds_radius: s.bounds_radius,
        max_instances: s.max_instances,
        quality: std::array::from_fn(|i| {
            let q = &s.quality[i];
            CookedQuality {
                spawn_count_scale: q.spawn_count_scale,
                max_distance: q.max_distance,
                max_instances: q.max_instances,
            }
        }),
    }
}
