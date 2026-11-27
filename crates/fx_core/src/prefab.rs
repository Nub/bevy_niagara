//! Prefabs: a tree of effects, meshes, lights and cameras with their
//! animations, as `fximport blueprint` writes them from Unreal Blueprint
//! actors (`<name>.prefab.ron`) and `fx_render::prefab` spawns and plays.
//!
//! Units and axes are ours (metres, Y up) except where an animation channel
//! targets an Unreal-space property ([`Property`] says which); the player
//! converts those with [`ue_location`], [`ue_rotator`] and [`ue_scale`] so a
//! track keeps the Blueprint's own numbers.
//!
//! - **Nodes** carry a transform relative to their parent and what they
//!   draw: an effect file (Niagara components, and static meshes as one
//!   mesh particle), a point light, a camera, or nothing.
//! - **Material parameters**: a mesh node's effect exposes the material
//!   parameters something animates as `User.` Vec4 parameters (four per
//!   parameter vector, read by the material through dynamic material
//!   parameters); [`MaterialParam`] says which component holds which.
//! - **Timelines** (Unreal Timeline components): a length, looping, a rate
//!   and tracks, each driving one node property through up to four
//!   channels, every channel a constant or a curve.
//! - **Actions**: what the actor's BeginPlay does, at the time it does it
//!   (set a material parameter, show or hide, activate, play a timeline).

use serde::{Deserialize, Serialize};

/// The file extension of prefab files.
pub const EXTENSION: &str = "prefab.ron";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Prefab {
    pub name: String,
    /// Where it came from (the Blueprint file), for people.
    #[serde(default)]
    pub source: String,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub timelines: Vec<Timeline>,
    #[serde(default)]
    pub actions: Vec<Action>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    /// Parent node name; `None`: the prefab root.
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub transform: Transform,
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default)]
    pub kind: NodeKind,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: [f32; 3],
    /// Quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Transform { translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum NodeKind {
    #[default]
    Empty,
    /// An effect file (relative to the prefab file).
    Effect {
        effect: String,
        #[serde(default = "yes")]
        auto_activate: bool,
        /// `User.` parameter values set on spawn.
        #[serde(default)]
        params: Vec<(String, ParamValue)>,
        /// Material parameters exposed as `User.` Vec4 parameters.
        #[serde(default)]
        material_params: Vec<MaterialParam>,
    },
    /// A point light: linear colour, lumens, range in metres.
    PointLight { color: [f32; 3], intensity: f32, range: f32 },
    /// A camera: vertical field of view in degrees. Not spawned as a Bevy
    /// camera; the app may look through it.
    Camera { fov: f32 },
}

/// A material parameter the node's effect reads from a `User.` parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialParam {
    /// The Unreal material parameter name.
    pub name: String,
    /// Its custom primitive data index, when the material reads it from
    /// there (`SetCustomPrimitiveData*` writes it).
    #[serde(default)]
    pub primitive_data: Option<u32>,
    /// The `User.` Vec4 parameter holding it.
    pub user: String,
    /// First component (0..4) and component count (1 scalar, 4 vector).
    pub component: u32,
    pub count: u32,
    /// Its value before anything sets it.
    pub default: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub name: String,
    /// Seconds.
    pub length: f32,
    #[serde(default)]
    pub looping: bool,
    /// Plays from spawn (the template's auto play).
    #[serde(default)]
    pub autoplay: bool,
    #[serde(default = "one")]
    pub rate: f32,
    pub tracks: Vec<Track>,
}

fn one() -> f32 {
    1.0
}

/// One node property driven by a timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub node: String,
    pub property: Property,
    /// One channel per component of the property (3 for transforms, 1 for
    /// a scalar, 4 for a colour).
    pub channels: Vec<Channel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Property {
    /// Unreal relative location (centimetres, Unreal axes).
    Location,
    /// Unreal rotator (pitch, yaw, roll; degrees).
    Rotation,
    /// Unreal relative scale (Unreal axes).
    Scale,
    /// A material parameter (scalar: 1 channel; vector: 4).
    Material(String),
    /// An effect `User.` parameter (Float: 1 channel, Vec3: 3, Color: 4).
    Param(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Channel {
    Const(f32),
    /// `scale * curve(t) + offset`.
    Curve { keys: Vec<Key>, #[serde(default = "one")] scale: f32, #[serde(default)] offset: f32 },
}

impl Channel {
    pub fn eval(&self, t: f32) -> f32 {
        match self {
            Channel::Const(v) => *v,
            Channel::Curve { keys, scale, offset } => scale * eval_keys(keys, t) + offset,
        }
    }
}

/// A curve key (Unreal rich curve key).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub time: f32,
    pub value: f32,
    #[serde(default)]
    pub interp: Interp,
    #[serde(default)]
    pub arrive: f32,
    #[serde(default)]
    pub leave: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Interp {
    #[default]
    Linear,
    Constant,
    Cubic,
}

/// Evaluates a key list: constant before the first key and after the last,
/// linear, stepped or Hermite (Unreal's cubic, tangents in value/second).
pub fn eval_keys(keys: &[Key], t: f32) -> f32 {
    let (Some(first), Some(last)) = (keys.first(), keys.last()) else { return 0.0 };
    if t <= first.time {
        return first.value;
    }
    if t >= last.time {
        return last.value;
    }
    let i = keys.partition_point(|k| k.time <= t).max(1) - 1;
    let (a, b) = (keys[i], keys[i + 1]);
    let dt = b.time - a.time;
    if dt <= 0.0 {
        return b.value;
    }
    let s = (t - a.time) / dt;
    match a.interp {
        Interp::Constant => a.value,
        Interp::Linear => a.value + (b.value - a.value) * s,
        Interp::Cubic => {
            let (s2, s3) = (s * s, s * s * s);
            let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
            let h10 = s3 - 2.0 * s2 + s;
            let h01 = -2.0 * s3 + 3.0 * s2;
            let h11 = s3 - s2;
            h00 * a.value + h10 * dt * a.leave + h01 * b.value + h11 * dt * b.arrive
        }
    }
}

/// Something BeginPlay does, at `time` seconds after spawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub time: f32,
    pub node: String,
    pub what: Do,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Do {
    /// Sets material parameter components (the scalar, or the vector's
    /// components) on the node's effect.
    SetMaterial { name: String, value: Vec<Num> },
    SetParam { name: String, value: ParamValue },
    SetVisible(bool),
    Activate { reset: bool },
    Deactivate,
    /// Node: the timeline name.
    PlayTimeline { from_start: bool },
    StopTimeline,
    SetTimelineRate(f32),
}

/// A number set by an action: a value, or the game time when it runs (the
/// time materials read, Unreal's `GetGameTimeInSeconds`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Num {
    Value(f32),
    GameTime { #[serde(default)] plus: f32 },
}

/// Unreal location (centimetres) to ours (metres, (X, Y, Z) -> (X, Z, Y)).
pub fn ue_location(v: [f32; 3]) -> [f32; 3] {
    [v[0] * 0.01, v[2] * 0.01, v[1] * 0.01]
}

/// Unreal scale to ours.
pub fn ue_scale(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], v[1]]
}

/// Unreal rotator (pitch, yaw, roll in degrees) to our quaternion (x, y, z,
/// w): Unreal's FRotator -> FQuat, then mirrored by the Y/Z swap (the axis
/// swaps and the angle changes sign).
pub fn ue_rotator(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    let h = std::f32::consts::PI / 360.0;
    let (sp, cp) = (pitch * h).sin_cos();
    let (sy, cy) = (yaw * h).sin_cos();
    let (sr, cr) = (roll * h).sin_cos();
    let x = cr * sp * sy - sr * cp * cy;
    let y = -cr * sp * cy - sr * cp * sy;
    let z = cr * cp * sy - sr * sp * cy;
    let w = cr * cp * cy + sr * sp * sy;
    [-x, -z, -y, w]
}

/// Parses a prefab file.
pub fn from_str(text: &str) -> Result<Prefab, String> {
    ron::from_str(text).map_err(|e| e.to_string())
}

/// Writes a prefab file.
pub fn to_string(p: &Prefab) -> String {
    let cfg = ron::ser::PrettyConfig::new().depth_limit(4).indentor("    ".to_string());
    ron::ser::to_string_pretty(p, cfg).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        let (qx, qy, qz, w) = (q[0], q[1], q[2], q[3]);
        // v + 2w(q×v) + 2q×(q×v)
        let c1 = [qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]];
        let c2 = [qy * c1[2] - qz * c1[1], qz * c1[0] - qx * c1[2], qx * c1[1] - qy * c1[0]];
        [v[0] + 2.0 * (w * c1[0] + c2[0]), v[1] + 2.0 * (w * c1[1] + c2[1]), v[2] + 2.0 * (w * c1[2] + c2[2])]
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn rotators_match_unreal_after_the_axis_swap() {
        // Unreal yaw 90 turns +X to +Y; ours: +X to +Z.
        assert!(close(rotate(ue_rotator(0.0, 90.0, 0.0), [1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]));
        // Unreal pitch 90 turns +X up (+Z); ours: up is +Y.
        assert!(close(rotate(ue_rotator(90.0, 0.0, 0.0), [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        // Unreal roll 90 turns +Y to -Z (FRotationMatrix); ours: +Z to -Y.
        assert!(close(rotate(ue_rotator(0.0, 0.0, 90.0), [0.0, 0.0, 1.0]), [0.0, -1.0, 0.0]));
    }

    #[test]
    fn keys_interpolate() {
        let k = |time, value| Key { time, value, interp: Interp::Linear, arrive: 0.0, leave: 0.0 };
        let keys = [k(0.0, 0.0), k(100.0, 360.0)];
        assert_eq!(eval_keys(&keys, 50.0), 180.0);
        assert_eq!(eval_keys(&keys, -1.0), 0.0);
        assert_eq!(eval_keys(&keys, 200.0), 360.0);
        let step = [Key { interp: Interp::Constant, ..k(0.0, 1.0) }, k(1.0, 2.0)];
        assert_eq!(eval_keys(&step, 0.5), 1.0);
    }

    #[test]
    fn round_trips_through_ron() {
        let p = Prefab {
            name: "P".into(),
            nodes: vec![Node {
                name: "Face".into(),
                parent: None,
                transform: Transform::default(),
                visible: true,
                kind: NodeKind::Effect { effect: "a.fx.ron".into(), auto_activate: true, params: vec![], material_params: vec![] },
            }],
            timelines: vec![Timeline {
                name: "T".into(),
                length: 1.0,
                looping: true,
                autoplay: false,
                rate: 1.0,
                tracks: vec![Track { node: "Face".into(), property: Property::Rotation, channels: vec![Channel::Const(0.0), Channel::Curve { keys: vec![], scale: 1.0, offset: 0.0 }, Channel::Const(0.0)] }],
            }],
            actions: vec![Action { time: 0.0, node: "Face".into(), what: Do::SetMaterial { name: "X".into(), value: vec![Num::GameTime { plus: 0.0 }] } }],
            ..Default::default()
        };
        assert_eq!(from_str(&to_string(&p)).unwrap(), p);
    }
}
