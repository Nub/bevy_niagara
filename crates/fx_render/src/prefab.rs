//! Prefabs as Bevy assets (`.prefab.ron`, written by `fximport blueprint`):
//! a tree of effects, point lights, cameras and empty nodes, with timelines
//! and timed actions (see [`fx_core::prefab`]).
//!
//! Spawn [`FxPrefabInstance`] with a `Transform`. Once the prefab and its
//! effects have loaded, the node entities are spawned as its children
//! (each with an [`FxPrefabNode`]; cameras get an [`FxPrefabCamera`] for the
//! app to look through), and [`FxPrefabPlayer`] on the instance plays it:
//! - **Actions** run at their time: material parameters (written into the
//!   mesh effect's `User.` Vec4 parameters; `GameTime` reads the time
//!   materials see, `Time::elapsed_secs_wrapped`), effect parameters,
//!   visibility, activation, timeline play / stop / rate.
//! - **Timelines** advance at their rate (looping or stopping at the end)
//!   and set their tracks every frame (time: the effects' [`FxClock`]): Unreal-space location, rotation and
//!   scale (converted, replacing that part of the node's transform),
//!   material parameters and effect parameters.

use bevy::asset::io::Reader;
use bevy::asset::{AssetLoader, LoadContext};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use fx_core::prefab::{self, Do, Num, ParamValue, Prefab, Property};
use fx_core::types::Value;

use crate::{EffectControl, FxClock, FxEffectAsset, ParticleEffect};

/// A loaded prefab and its effects (by the path written in the prefab).
#[derive(Asset, TypePath, Clone)]
pub struct FxPrefab {
    pub prefab: Prefab,
    pub effects: HashMap<String, Handle<FxEffectAsset>>,
}

#[derive(Default, TypePath)]
pub struct FxPrefabLoader;

impl AssetLoader for FxPrefabLoader {
    type Asset = FxPrefab;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(&self, reader: &mut dyn Reader, _settings: &(), load_context: &mut LoadContext<'_>) -> Result<FxPrefab, std::io::Error> {
        let err = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let path = load_context.path().clone();
        let text = String::from_utf8(bytes).map_err(|e| err(format!("{path}: {e}")))?;
        let prefab = prefab::from_str(&text).map_err(|e| err(format!("{path}: {e}")))?;
        let mut effects = HashMap::default();
        for n in &prefab.nodes {
            if let prefab::NodeKind::Effect { effect, .. } = &n.kind {
                if effects.contains_key(effect) {
                    continue;
                }
                let full = path.resolve_embed_str(effect).map_err(|e| err(format!("{path}: {effect}: {e}")))?;
                effects.insert(effect.clone(), load_context.load::<FxEffectAsset>(full));
            }
        }
        Ok(FxPrefab { prefab, effects })
    }

    fn extensions(&self) -> &[&str] {
        &[prefab::EXTENSION]
    }
}

/// A prefab to spawn here (its nodes become children of this entity).
#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct FxPrefabInstance {
    pub prefab: Handle<FxPrefab>,
    pub seed: u32,
}

impl FxPrefabInstance {
    pub fn new(prefab: Handle<FxPrefab>) -> Self {
        FxPrefabInstance { prefab, seed: 0 }
    }
}

/// A spawned prefab node.
#[derive(Component, Clone, Debug)]
pub struct FxPrefabNode {
    pub name: String,
}

/// A prefab camera node: look through it with the vertical field of view
/// `fov` (degrees).
#[derive(Component, Clone, Copy, Debug)]
pub struct FxPrefabCamera {
    pub fov: f32,
}

struct TimelineState {
    playing: bool,
    time: f32,
    rate: f32,
}

/// Plays a spawned prefab: its clock, timelines and actions.
#[derive(Component)]
pub struct FxPrefabPlayer {
    /// Seconds since the prefab spawned.
    pub time: f32,
    prefab: Handle<FxPrefab>,
    nodes: HashMap<String, Entity>,
    base: HashMap<String, Transform>,
    timelines: Vec<TimelineState>,
    next_action: usize,
}

impl FxPrefabPlayer {
    /// The prefab it plays.
    pub fn prefab(&self) -> &Handle<FxPrefab> {
        &self.prefab
    }
    /// The entity of node `name`.
    pub fn node(&self, name: &str) -> Option<Entity> {
        self.nodes.get(name).copied()
    }
    /// Starts timeline `name` (from the start, or where it stopped).
    pub fn play(&mut self, prefab: &Prefab, name: &str, from_start: bool) {
        if let Some(i) = prefab.timelines.iter().position(|t| t.name == name) {
            let s = &mut self.timelines[i];
            s.playing = true;
            if from_start {
                s.time = 0.0;
            }
        }
    }
    /// Sets timeline `name`'s play rate.
    pub fn set_rate(&mut self, prefab: &Prefab, name: &str, rate: f32) {
        if let Some(i) = prefab.timelines.iter().position(|t| t.name == name) {
            self.timelines[i].rate = rate;
        }
    }
}

pub struct FxPrefabPlugin;

impl Plugin for FxPrefabPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<FxPrefab>()
            .init_asset_loader::<FxPrefabLoader>()
            .add_systems(
                PostUpdate,
                (spawn_prefabs, play_prefabs)
                    .chain()
                    .after(crate::advance_clock)
                    .before(bevy::transform::TransformSystems::Propagate),
            );
    }
}

fn transform_of(t: &prefab::Transform) -> Transform {
    Transform {
        translation: Vec3::from_array(t.translation),
        rotation: Quat::from_array(t.rotation).normalize(),
        scale: Vec3::from_array(t.scale),
    }
}

fn spawn_prefabs(
    mut commands: Commands,
    new: Query<(Entity, &FxPrefabInstance), Without<FxPrefabPlayer>>,
    prefabs: Res<Assets<FxPrefab>>,
    effects: Res<Assets<FxEffectAsset>>,
) {
    for (root, inst) in &new {
        let Some(p) = prefabs.get(&inst.prefab) else { continue };
        if p.effects.values().any(|h| effects.get(h).is_none()) {
            continue;
        }
        let mut nodes = HashMap::default();
        let mut base = HashMap::default();
        for (k, n) in p.prefab.nodes.iter().enumerate() {
            let t = transform_of(&n.transform);
            base.insert(n.name.clone(), t);
            let vis = if n.visible { Visibility::Inherited } else { Visibility::Hidden };
            let mut e = commands.spawn((FxPrefabNode { name: n.name.clone() }, t, vis));
            match &n.kind {
                prefab::NodeKind::Empty => {}
                prefab::NodeKind::Effect { effect, auto_activate, params, material_params } => {
                    let asset = effects.get(&p.effects[effect]).unwrap();
                    let mut fx = ParticleEffect::shared(asset.system.clone(), inst.seed.wrapping_add(k as u32));
                    for (name, v) in params {
                        if let Err(err) = fx.set_param(name, param_value(*v)) {
                            warn!("prefab {}: node {}: {err}", p.prefab.name, n.name);
                        }
                    }
                    for m in material_params {
                        set_material(&mut fx, m, &m.default[..m.count as usize]);
                    }
                    let mut control = EffectControl::looping();
                    if !auto_activate {
                        control.deactivate();
                    }
                    e.insert((fx, control));
                }
                prefab::NodeKind::PointLight { color, intensity, range } => {
                    e.insert(PointLight {
                        color: Color::linear_rgb(color[0], color[1], color[2]),
                        intensity: *intensity,
                        range: *range,
                        ..default()
                    });
                }
                prefab::NodeKind::Camera { fov } => {
                    e.insert(FxPrefabCamera { fov: *fov });
                }
            }
            nodes.insert(n.name.clone(), e.id());
        }
        for n in &p.prefab.nodes {
            let child = nodes[&n.name];
            let parent = n.parent.as_ref().and_then(|p| nodes.get(p)).copied().unwrap_or(root);
            commands.entity(parent).add_child(child);
        }
        let timelines = p.prefab.timelines.iter().map(|t| TimelineState { playing: t.autoplay, time: 0.0, rate: t.rate }).collect();
        commands.entity(root).insert(FxPrefabPlayer { time: 0.0, prefab: inst.prefab.clone(), nodes, base, timelines, next_action: 0 });
    }
}

fn param_value(v: ParamValue) -> Value {
    match v {
        ParamValue::Float(x) => Value::Float(x),
        ParamValue::Int(x) => Value::Int(x),
        ParamValue::Bool(x) => Value::Bool(x),
        ParamValue::Vec2(x) => Value::Vec2(x),
        ParamValue::Vec3(x) => Value::Vec3(x),
        ParamValue::Vec4(x) => Value::Vec4(x),
        ParamValue::Color(x) => Value::Color(x),
    }
}

/// Writes material parameter `m`'s components into its `User.` Vec4.
fn set_material(fx: &mut ParticleEffect, m: &prefab::MaterialParam, value: &[f32]) {
    let mut v = match fx.param(&m.user) {
        Some(Value::Vec4(v)) => v,
        _ => [0.0; 4],
    };
    for (i, x) in value.iter().enumerate().take(m.count as usize) {
        if let Some(s) = v.get_mut(m.component as usize + i) {
            *s = *x;
        }
    }
    let _ = fx.set_param(&m.user, Value::Vec4(v));
}

fn material_param<'a>(p: &'a Prefab, node: &str, name: &str) -> Option<&'a prefab::MaterialParam> {
    let n = p.nodes.iter().find(|n| n.name == node)?;
    match &n.kind {
        prefab::NodeKind::Effect { material_params, .. } => material_params.iter().find(|m| m.name == name),
        _ => None,
    }
}

fn play_prefabs(
    time: Res<Time>,
    clock: Res<FxClock>,
    prefabs: Res<Assets<FxPrefab>>,
    mut players: Query<&mut FxPrefabPlayer>,
    mut nodes: Query<(&mut Transform, &mut Visibility, Option<&mut ParticleEffect>, Option<&mut EffectControl>), With<FxPrefabNode>>,
) {
    // Prefabs run on the effects' clock (simulated time).
    let dt = clock.steps_this_frame() as f32 * clock.step;
    let game_time = time.elapsed_secs_wrapped();
    for mut pl in &mut players {
        let Some(asset) = prefabs.get(&pl.prefab) else { continue };
        let p = &asset.prefab;
        // Actions due by now (the spawn frame runs the time-0 ones).
        while let Some(a) = p.actions.get(pl.next_action).filter(|a| a.time <= pl.time) {
            pl.next_action += 1;
            match &a.what {
                Do::PlayTimeline { from_start } => pl.play(p, &a.node, *from_start),
                Do::StopTimeline => {
                    if let Some(i) = p.timelines.iter().position(|t| t.name == a.node) {
                        pl.timelines[i].playing = false;
                    }
                }
                Do::SetTimelineRate(r) => pl.set_rate(p, &a.node, *r),
                what => {
                    let Some(e) = pl.nodes.get(&a.node).copied() else { continue };
                    let Ok((_, mut vis, fx, control)) = nodes.get_mut(e) else { continue };
                    match what {
                        Do::SetMaterial { name, value } => {
                            if let (Some(mut fx), Some(m)) = (fx, material_param(p, &a.node, name)) {
                                let v: Vec<f32> = value
                                    .iter()
                                    .map(|n| match n {
                                        Num::Value(x) => *x,
                                        Num::GameTime { plus } => game_time + plus,
                                    })
                                    .collect();
                                set_material(&mut fx, m, &v);
                            }
                        }
                        Do::SetParam { name, value } => {
                            if let Some(mut fx) = fx {
                                let _ = fx.set_param(name, param_value(*value));
                            }
                        }
                        Do::SetVisible(v) => *vis = if *v { Visibility::Inherited } else { Visibility::Hidden },
                        Do::Activate { reset } => {
                            if let Some(mut c) = control {
                                if *reset {
                                    c.reset();
                                } else {
                                    c.activate();
                                }
                            }
                        }
                        Do::Deactivate => {
                            if let Some(mut c) = control {
                                c.deactivate();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        // Timelines.
        for (i, t) in p.timelines.iter().enumerate() {
            let s = &mut pl.timelines[i];
            if !s.playing {
                continue;
            }
            let at = s.time;
            s.time += dt * s.rate;
            if t.length > 0.0 && s.time > t.length {
                if t.looping {
                    s.time %= t.length;
                } else {
                    s.time = t.length;
                    s.playing = false;
                }
            }
            for tr in &t.tracks {
                let Some(e) = pl.nodes.get(&tr.node).copied() else { continue };
                let Ok((mut tf, _, fx, _)) = nodes.get_mut(e) else { continue };
                let v: Vec<f32> = tr.channels.iter().map(|c| c.eval(at)).collect();
                let v3 = || [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
                let base = pl.base.get(&tr.node).copied().unwrap_or_default();
                match &tr.property {
                    Property::Location => tf.translation = Vec3::from_array(prefab::ue_location(v3())),
                    Property::Rotation => {
                        let r = v3();
                        tf.rotation = Quat::from_array(prefab::ue_rotator(r[0], r[1], r[2]));
                    }
                    Property::Scale => tf.scale = Vec3::from_array(prefab::ue_scale(v3())),
                    Property::Material(name) => {
                        if let (Some(mut fx), Some(m)) = (fx, material_param(p, &tr.node, name)) {
                            set_material(&mut fx, m, &v);
                        }
                    }
                    Property::Param(name) => {
                        if let Some(mut fx) = fx {
                            let val = match v.len() {
                                1 => Value::Float(v[0]),
                                3 => Value::Vec3(v3()),
                                _ => Value::Color([v[0], v[1], v[2], v.get(3).copied().unwrap_or(1.0)]),
                            };
                            let _ = fx.set_param(name, val);
                        }
                    }
                }
                let _ = base;
            }
        }
        pl.time += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::prefab::{Channel, Interp, Key, MaterialParam, Node, NodeKind, Timeline, Track};
    use std::sync::Arc;

    fn line(t0: f32, v0: f32, t1: f32, v1: f32) -> Channel {
        let k = |time, value| Key { time, value, interp: Interp::Linear, arrive: 0.0, leave: 0.0 };
        Channel::Curve { keys: vec![k(t0, v0), k(t1, v1)], scale: 1.0, offset: 0.0 }
    }

    /// A timeline turning, scaling and fading a mesh node, as `fximport
    /// blueprint` writes one from a Timeline driving a static mesh
    /// component's relative rotation, scale and a material scalar.
    #[test]
    fn timelines_drive_mesh_rotation_scale_and_material_scalars() {
        let src = r#"(name: "Mesh", emitters: [], user_parameters: { "User.MaterialParams0": Vec4((0.0, 0.0, 0.0, 0.0)) })"#;
        let system = fx_core::check_source("mesh.fx.ron", src).cooked.expect("cooks");
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), bevy::transform::TransformPlugin))
            .init_asset::<FxEffectAsset>()
            .init_asset::<FxPrefab>()
            .insert_resource(FxClock::manual(1.0 / 60.0))
            .add_systems(PostUpdate, (crate::advance_clock, spawn_prefabs, play_prefabs).chain());
        let effect = app.world_mut().resource_mut::<Assets<FxEffectAsset>>().add(FxEffectAsset { system: Arc::new(system) });
        let glow = MaterialParam { name: "Glow".into(), primitive_data: None, user: "User.MaterialParams0".into(), component: 1, count: 1, default: [0.25, 0.0, 0.0, 0.0] };
        let p = Prefab {
            name: "Spinner".into(),
            nodes: vec![Node {
                name: "Blade".into(),
                parent: None,
                transform: Default::default(),
                visible: true,
                kind: NodeKind::Effect { effect: "mesh.fx.ron".into(), auto_activate: true, params: vec![], material_params: vec![glow] },
            }],
            timelines: vec![Timeline {
                name: "Spin".into(),
                length: 2.0,
                looping: false,
                autoplay: true,
                rate: 1.0,
                tracks: vec![
                    // Unreal yaw 0 -> 90 degrees over 1 s.
                    Track { node: "Blade".into(), property: Property::Rotation, channels: vec![Channel::Const(0.0), line(0.0, 0.0, 1.0, 90.0), Channel::Const(0.0)] },
                    Track { node: "Blade".into(), property: Property::Scale, channels: vec![line(0.0, 1.0, 1.0, 3.0), Channel::Const(1.0), Channel::Const(1.0)] },
                    Track { node: "Blade".into(), property: Property::Material("Glow".into()), channels: vec![line(0.0, 0.0, 1.0, 2.0)] },
                ],
            }],
            ..Default::default()
        };
        let mut effects = HashMap::default();
        effects.insert("mesh.fx.ron".to_string(), effect);
        let prefab = app.world_mut().resource_mut::<Assets<FxPrefab>>().add(FxPrefab { prefab: p, effects });
        app.world_mut().spawn(FxPrefabInstance::new(prefab));
        app.update(); // spawns
        // On the spawn frame the auto-playing timeline already sets its value
        // at time 0 (over the default 0.25), in component 1 of the Vec4.
        let fx = node_effect(&mut app);
        assert_eq!(fx.param("User.MaterialParams0"), Some(Value::Vec4([0.0, 0.0, 0.0, 0.0])));
        // Half a second of simulated time (30 steps), then one more frame to sample it.
        run_steps(&mut app, 30);
        let (tf, fx) = node_state(&mut app);
        let expect = Quat::from_array(prefab::ue_rotator(0.0, 45.0, 0.0));
        assert!(tf.rotation.angle_between(expect) < 1e-4, "{:?} vs {expect:?}", tf.rotation);
        // Unreal X scale is our X.
        assert!((tf.scale.x - 2.0).abs() < 1e-4, "{:?}", tf.scale);
        let Some(Value::Vec4(v)) = fx.param("User.MaterialParams0") else { panic!() };
        assert!((v[1] - 1.0).abs() < 1e-4, "{v:?}");
        // Past its end a non-looping timeline holds its last value.
        run_steps(&mut app, 240);
        let (tf, _) = node_state(&mut app);
        assert!((tf.scale.x - 3.0).abs() < 1e-4, "{:?}", tf.scale);
    }

    /// Runs `n` simulation steps (spread over frames), then a frame with
    /// none, which samples the timelines at the time reached.
    fn run_steps(app: &mut App, n: u32) {
        let target = app.world().resource::<FxClock>().total_steps + n as u64;
        app.world_mut().resource_mut::<FxClock>().request_steps(n);
        while app.world().resource::<FxClock>().total_steps < target {
            app.update();
        }
        app.update();
    }

    fn node_effect(app: &mut App) -> ParticleEffect {
        node_state(app).1
    }

    fn node_state(app: &mut App) -> (Transform, ParticleEffect) {
        let w = app.world_mut();
        let mut q = w.query_filtered::<(&Transform, &ParticleEffect), With<FxPrefabNode>>();
        let (t, f) = q.single(w).expect("one node");
        (*t, f.clone())
    }

    #[test]
    fn material_params_pack_into_user_vec4s() {
        let src = r#"(name: "M", emitters: [], user_parameters: { "User.MaterialParams0": Vec4((0.0, 0.0, 0.0, 0.0)) })"#;
        let checked = fx_core::check_source("m.fx.ron", src);
        let Some(system) = checked.cooked else { panic!("{:?}", checked.diagnostics) };
        let mut fx = ParticleEffect::new(system, 0);
        let m = prefab::MaterialParam { name: "Open".into(), primitive_data: Some(2), user: "User.MaterialParams0".into(), component: 2, count: 1, default: [0.0; 4] };
        set_material(&mut fx, &m, &[1.5]);
        assert_eq!(fx.param("User.MaterialParams0"), Some(Value::Vec4([0.0, 0.0, 1.5, 0.0])));
    }
}
