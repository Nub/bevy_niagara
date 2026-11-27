//! Input: keyboard and mouse, touch (virtual stick, look drag, spell
//! buttons), and the bot of the scripted run. All three fill [`Intent`].

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::input::touch::Touches;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::hud::{STICK_RADIUS, button_rect, restart_rect};
use crate::{Enemy, Intent, Phase, Player, Run, RunMode, Spell, Turret, enemy_center};

#[derive(Resource, Default)]
pub struct TouchState {
    /// The touch steering the stick and where it started (logical px).
    pub stick: Option<(u64, Vec2)>,
    pub stick_pos: Vec2,
    pub look: Option<u64>,
    /// A touch was seen: the touch controls show, the mouse is ignored.
    pub seen: bool,
}

/// Mouse look only while the pointer is captured (in the browser, the page
/// releases it on Esc without telling Bevy).
fn pointer_locked(cursor: &CursorOptions) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cursor;
        web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.pointer_lock_element())
            .is_some()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        cursor.grab_mode != CursorGrabMode::None
    }
}

#[allow(clippy::too_many_arguments)]
pub fn keyboard_mouse(
    mode: Res<RunMode>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    touch: Res<TouchState>,
    mut intent: ResMut<Intent>,
    mut windows: Query<(&Window, &mut CursorOptions)>,
) {
    if mode.bot() {
        return;
    }
    let mut mv = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        mv.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        mv.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        mv.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        mv.x -= 1.0;
    }
    intent.mv += mv;
    intent.run |= keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    for (k, s) in [
        (KeyCode::Digit1, Spell::Fireball),
        (KeyCode::Digit2, Spell::Lightning),
        (KeyCode::Digit3, Spell::Frost),
        (KeyCode::Digit4, Spell::Turret),
    ] {
        if keys.just_pressed(k) {
            intent.select = Some(s);
        }
    }
    intent.restart |= keys.just_pressed(KeyCode::KeyR);

    let Ok((window, mut cursor)) = windows.single_mut() else { return };
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
    if touch.seen {
        return;
    }
    let locked = pointer_locked(&cursor);
    if locked {
        intent.look += motion.delta * 0.0022;
    }
    if buttons.just_pressed(MouseButton::Left) {
        // Over a button (pointer free): press it.
        if !locked && let Some(p) = window.cursor_position() {
            let size = window.size();
            if restart_rect(size).contains(p) {
                intent.restart = true;
                return;
            }
            if let Some(s) = Spell::ALL.into_iter().find(|s| button_rect(s.index(), size).contains(p)) {
                intent.select = Some(s);
                return;
            }
        }
        if locked {
            intent.cast_selected = true;
        } else {
            cursor.grab_mode = CursorGrabMode::Locked;
            cursor.visible = false;
        }
    }
}

pub fn touch(
    mode: Res<RunMode>,
    touches: Res<Touches>,
    mut st: ResMut<TouchState>,
    mut intent: ResMut<Intent>,
    windows: Query<&Window>,
) {
    if mode.bot() {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let size = window.size();
    for t in touches.iter_just_pressed() {
        st.seen = true;
        let p = t.position();
        if restart_rect(size).contains(p) {
            intent.restart = true;
        } else if let Some(s) = Spell::ALL.into_iter().find(|s| button_rect(s.index(), size).contains(p)) {
            // Spell buttons cast at the crosshair.
            intent.select = Some(s);
            intent.cast = Some(s);
        } else if p.x < size.x * 0.45 && st.stick.is_none() {
            st.stick = Some((t.id(), p));
            st.stick_pos = p;
        } else if st.look.is_none() {
            st.look = Some(t.id());
        }
    }
    for t in touches.iter_just_released().chain(touches.iter_just_canceled()) {
        if st.stick.is_some_and(|(id, _)| id == t.id()) {
            st.stick = None;
        }
        if st.look == Some(t.id()) {
            st.look = None;
        }
    }
    if let Some((id, origin)) = st.stick {
        if let Some(t) = touches.get_pressed(id) {
            let d = (t.position() - origin).clamp_length_max(STICK_RADIUS);
            st.stick_pos = origin + d;
            let v = d / STICK_RADIUS;
            // Screen y grows downwards: up is forward.
            intent.mv += Vec2::new(v.x, -v.y);
            intent.run |= v.length() > 0.95;
        }
    }
    if let Some(id) = st.look
        && let Some(t) = touches.get_pressed(id)
    {
        intent.look += t.delta() * 0.0045;
    }
}

/// The scripted player: aims at the most advanced fox, casts fireballs,
/// lightning and frost novas, builds turrets beside the path at set times
/// and strafes around near the tower.
#[allow(clippy::type_complexity)]
pub fn bot(
    mode: Res<RunMode>,
    run: Res<Run>,
    player: Res<Player>,
    mut intent: ResMut<Intent>,
    enemies: Query<(&Transform, &Enemy)>,
    turrets: Query<(), With<Turret>>,
    mut next_cast: Local<f32>,
    clock: Res<fx_render::FxClock>,
    mut over: Local<f32>,
) {
    if !mode.bot() {
        return;
    }
    // After a lost round, a few seconds on the score, then play again.
    if run.phase == Phase::GameOver {
        *over += crate::game_dt(&clock);
        if *over > 4.0 {
            *over = 0.0;
            *next_cast = 0.0;
            intent.restart = true;
        }
        return;
    }
    if run.phase != Phase::Playing {
        return;
    }
    // Turret sites beside the path, built in order once there is mana.
    const SITES: [Vec3; 4] = [
        Vec3::new(-7.0, 0.0, -21.0),
        Vec3::new(7.0, 0.0, -20.0),
        Vec3::new(-6.0, 0.0, -6.0),
        Vec3::new(6.0, 0.0, -6.0),
    ];
    let built = turrets.iter().count();
    let want_turret = built < SITES.len() && run.time > 6.0 + 11.0 * built as f32;
    let target = if want_turret {
        Some((SITES[built], true))
    } else {
        enemies
            .iter()
            .filter(|(t, _)| t.translation.distance(player.pos) < 42.0)
            .max_by(|a, b| a.1.dist.total_cmp(&b.1.dist))
            .map(|(t, e)| {
                // Lead the target a little.
                let (_, dir) = crate::world::path_point(e.dist);
                (enemy_center(t, e) + dir * e.speed * 0.25, false)
            })
    };
    // Wander in a slow figure eight in front of the tower.
    let t = run.time * 0.35;
    let goal = Vec3::new(2.5 * t.sin(), 0.0, -19.0 + 1.5 * (2.0 * t).sin());
    let to = goal - player.pos;
    if to.length() > 0.3 {
        let (s, c) = player.yaw.sin_cos();
        let fwd = Vec3::new(-s, 0.0, -c);
        let right = Vec3::new(c, 0.0, -s);
        let d = to.normalize();
        intent.mv = Vec2::new(d.dot(right), d.dot(fwd)) * to.length().min(1.0) * 0.6;
    }
    let Some((aim, building)) = target else {
        // Nothing to shoot: look down the path.
        let want_yaw = 0.0;
        intent.look.x = -(want_yaw - player.yaw) * 0.05;
        intent.look.y = -(-0.15 - player.pitch) * 0.05;
        return;
    };
    let eye = player.eye();
    let d = aim - eye;
    let want_yaw = (-d.x).atan2(-d.z);
    let want_pitch = d.y.atan2(Vec2::new(d.x, d.z).length());
    let mut dy = want_yaw - player.yaw;
    while dy > std::f32::consts::PI {
        dy -= std::f32::consts::TAU;
    }
    while dy < -std::f32::consts::PI {
        dy += std::f32::consts::TAU;
    }
    let dp = want_pitch - player.pitch;
    // Smooth turning (look is subtracted from yaw/pitch).
    intent.look = -Vec2::new(dy, dp) * 0.18;
    let on_target = dy.abs() < 0.04 && dp.abs() < 0.04;
    if !on_target || run.time < *next_cast {
        return;
    }
    if building {
        if run.mana >= Spell::Turret.cost() {
            intent.cast = Some(Spell::Turret);
            *next_cast = run.time + 0.8;
        }
        return;
    }
    let near_aim = enemies
        .iter()
        .filter(|(t, _)| Vec2::new(t.translation.x - aim.x, t.translation.z - aim.z).length() < 5.0)
        .count();
    let spell = if near_aim >= 2 && run.mana >= Spell::Frost.cost() + 10.0 && run.cooldown[Spell::Frost.index()] <= 0.0 {
        Spell::Frost
    } else if run.mana >= 40.0 && run.cooldown[Spell::Lightning.index()] <= 0.0 {
        Spell::Lightning
    } else {
        Spell::Fireball
    };
    if run.mana >= spell.cost() {
        intent.select = Some(spell);
        intent.cast = Some(spell);
        *next_cast = run.time + 0.45;
    }
}
