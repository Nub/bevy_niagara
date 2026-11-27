//! The HUD: wave, score, tower HP and mana bars, the restart button, the
//! spell buttons, the virtual stick, messages. Button rectangles are
//! computed here so input hit-tests the same layout the UI draws.

use bevy::prelude::*;

use crate::controls::TouchState;
use crate::{Enemy, Intent, MAX_MANA, MAX_TURRETS, Phase, Player, Quality, Run, Spell, Turret, LEVELS};

pub const BTN: f32 = 104.0;
pub const GAP: f32 = 12.0;
pub const MARGIN: f32 = 20.0;
pub const STICK_RADIUS: f32 = 70.0;
const RESTART: Vec2 = Vec2::new(130.0, 46.0);

/// Spell button `i` (2x2 grid at the bottom right), logical px, top-left origin.
pub fn button_rect(i: usize, screen: Vec2) -> Rect {
    let (col, row) = ((i % 2) as f32, (i / 2) as f32);
    let right = MARGIN + (1.0 - col) * (BTN + GAP);
    let bottom = MARGIN + (1.0 - row) * (BTN + GAP);
    let min = Vec2::new(screen.x - right - BTN, screen.y - bottom - BTN);
    Rect::from_corners(min, min + Vec2::splat(BTN))
}

pub fn restart_rect(screen: Vec2) -> Rect {
    let min = Vec2::new(screen.x - MARGIN - RESTART.x, MARGIN);
    Rect::from_corners(min, min + RESTART)
}

#[derive(Component)]
pub struct StatusText;
#[derive(Component)]
pub struct HpFill;
#[derive(Component)]
pub struct ManaFill;
#[derive(Component)]
pub struct FpsText;
#[derive(Component)]
pub struct Toast;
#[derive(Component)]
pub struct SpellButton(pub Spell);
#[derive(Component)]
pub struct SpellLabel(pub Spell);
#[derive(Component)]
pub struct StickBase;
#[derive(Component)]
pub struct StickKnob;

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn bar(p: &mut ChildSpawnerCommands, label: &str, color: Color, marker: impl Component) {
    p.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: px(8),
        ..default()
    })
    .with_children(|r| {
        r.spawn((Text::new(label), font(15.0), TextColor(Color::srgb(0.9, 0.9, 0.95)), Node { width: px(48), ..default() }));
        r.spawn((
            Node {
                width: px(200),
                height: px(14),
                border_radius: BorderRadius::all(px(7)),
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        ))
        .with_child((
            Node {
                width: percent(100),
                height: percent(100),
                ..default()
            },
            BackgroundColor(color),
            marker,
        ));
    });
}

pub fn build_hud(mut commands: Commands) {
    // Top left: status and bars.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: px(MARGIN),
                left: px(MARGIN),
                padding: UiRect::all(px(10)),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                border_radius: BorderRadius::all(px(10)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.1, 0.55)),
        ))
        .with_children(|p| {
            p.spawn((Text::new("Loading"), font(20.0), TextColor(Color::WHITE), StatusText));
            bar(p, "Tower", Color::srgb(0.85, 0.25, 0.2), HpFill);
            bar(p, "Mana", Color::srgb(0.3, 0.55, 1.0), ManaFill);
        });
    // Top right: restart and the frame rate.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: px(MARGIN),
                right: px(MARGIN),
                width: px(RESTART.x),
                height: px(RESTART.y),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(px(2)),
                border_radius: BorderRadius::all(px(10)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.12, 0.1, 0.2, 0.75)),
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.5)),
        ))
        .with_child((Text::new("Restart"), font(19.0), TextColor(Color::WHITE)));
    commands.spawn((
        Text::new(""),
        font(13.0),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            top: px(MARGIN + RESTART.y + 6.0),
            right: px(MARGIN),
            ..default()
        },
        FpsText,
    ));
    // Crosshair.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: percent(50),
            top: percent(50),
            width: px(8),
            height: px(8),
            margin: UiRect::new(px(-4), px(0), px(-4), px(0)),
            border: UiRect::all(px(1.5)),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.85)),
    ));
    // Messages, upper middle.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: percent(22),
            width: percent(100),
            justify_content: JustifyContent::Center,
            ..default()
        })
        .with_child((
            Text::new(""),
            font(34.0),
            TextColor(Color::srgb(1.0, 0.95, 0.8)),
            TextShadow::default(),
            TextLayout::justify(Justify::Center),
            Toast,
        ));
    // Spell buttons.
    let colors = [
        Color::srgb(0.85, 0.35, 0.1),
        Color::srgb(0.55, 0.35, 0.95),
        Color::srgb(0.3, 0.7, 0.95),
        Color::srgb(0.6, 0.45, 0.8),
    ];
    for s in Spell::ALL {
        let i = s.index();
        let (col, row) = ((i % 2) as f32, (i / 2) as f32);
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    right: px(MARGIN + (1.0 - col) * (BTN + GAP)),
                    bottom: px(MARGIN + (1.0 - row) * (BTN + GAP)),
                    width: px(BTN),
                    height: px(BTN),
                    flex_direction: FlexDirection::Column,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(px(3)),
                    border_radius: BorderRadius::all(px(18)),
                    ..default()
                },
                BackgroundColor(colors[i].with_alpha(0.55)),
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.3)),
                SpellButton(s),
            ))
            .with_children(|b| {
                b.spawn((Text::new(s.name()), font(15.0), TextColor(Color::WHITE), TextLayout::justify(Justify::Center)));
                b.spawn((Text::new(""), font(13.0), TextColor(Color::srgb(0.85, 0.9, 1.0)), SpellLabel(s)));
            });
    }
    // Virtual stick (touch only).
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: px(2.0 * STICK_RADIUS),
                height: px(2.0 * STICK_RADIUS),
                border: UiRect::all(px(3)),
                border_radius: BorderRadius::MAX,
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.08)),
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.35)),
            StickBase,
        ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            width: px(64),
            height: px(64),
            border_radius: BorderRadius::MAX,
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.4)),
        StickKnob,
    ));
}

#[allow(clippy::too_many_arguments)]
pub fn update_hud(
    run: Res<Run>,
    q: Res<Quality>,
    fxq: Res<fx_render::FxQuality>,
    enemies: Query<(), With<Enemy>>,
    mut status: Query<&mut Text, (With<StatusText>, Without<Toast>, Without<FpsText>)>,
    mut toast: Query<&mut Text, (With<Toast>, Without<StatusText>, Without<FpsText>)>,
    mut fps: Query<&mut Text, (With<FpsText>, Without<StatusText>, Without<Toast>)>,
    mut hp: Query<&mut Node, (With<HpFill>, Without<ManaFill>)>,
    mut mana: Query<&mut Node, (With<ManaFill>, Without<HpFill>)>,
) {
    let left = enemies.iter().count() as u32 + run.to_spawn;
    for mut t in &mut status {
        t.0 = match run.phase {
            Phase::Loading => "Loading".into(),
            _ if run.wave == 0 => format!("Get ready    Score {}", run.score),
            _ => format!("Wave {}   {} foxes    Score {}", run.wave, left, run.score),
        };
    }
    for mut t in &mut toast {
        t.0 = match run.phase {
            Phase::Loading => "Loading spells...".into(),
            Phase::GameOver => format!(
                "The tower has fallen\nWave {}, {} foxes, score {}\nPress Restart",
                run.wave, run.kills, run.score
            ),
            Phase::Playing => run.toast.as_ref().map(|t| t.0.clone()).unwrap_or_default(),
        };
    }
    for mut t in &mut fps {
        let (scale, _, _) = LEVELS[q.level];
        t.0 = format!("{:.0} fps, {:?}, scale {scale}", q.fps, fxq.0);
    }
    for mut n in &mut hp {
        n.width = percent(run.tower_hp.max(0.0));
    }
    for mut n in &mut mana {
        n.width = percent(100.0 * run.mana / MAX_MANA);
    }
}

pub fn update_buttons(
    run: Res<Run>,
    player: Res<Player>,
    turrets: Query<(), With<Turret>>,
    mut buttons: Query<(&SpellButton, &mut BorderColor, &mut BackgroundColor)>,
    mut labels: Query<(&SpellLabel, &mut Text)>,
) {
    for (b, mut border, mut bg) in &mut buttons {
        let selected = player.selected == b.0;
        *border = BorderColor::all(if selected { Color::WHITE } else { Color::srgba(1.0, 1.0, 1.0, 0.3) });
        let ready = run.mana >= b.0.cost() && run.cooldown[b.0.index()] <= 0.0;
        bg.0 = bg.0.with_alpha(if ready { 0.7 } else { 0.25 });
    }
    let n = turrets.iter().count();
    for (l, mut t) in &mut labels {
        let key = l.0.index() + 1;
        t.0 = if l.0 == Spell::Turret {
            format!("[{key}] {:.0}, {n}/{MAX_TURRETS}", l.0.cost())
        } else {
            format!("[{key}] {:.0}", l.0.cost())
        };
    }
}

pub fn update_stick(
    st: Res<TouchState>,
    windows: Query<&Window>,
    _intent: Res<Intent>,
    mut base: Query<&mut Node, (With<StickBase>, Without<StickKnob>)>,
    mut knob: Query<&mut Node, (With<StickKnob>, Without<StickBase>)>,
) {
    let screen = windows.single().map(|w| w.size()).unwrap_or(Vec2::new(1280.0, 720.0));
    // Where the stick is: under the thumb, or resting at the lower left.
    let rest = Vec2::new(MARGIN + 30.0 + STICK_RADIUS, screen.y - MARGIN - 30.0 - STICK_RADIUS);
    let (origin, pos, active) = match st.stick {
        Some((_, o)) => (o, st.stick_pos, true),
        None => (rest, rest, false),
    };
    let display = if st.seen { Display::Flex } else { Display::None };
    for mut n in &mut base {
        n.display = display;
        n.left = px(origin.x - STICK_RADIUS);
        n.top = px(origin.y - STICK_RADIUS);
    }
    for mut n in &mut knob {
        n.display = display;
        n.left = px(pos.x - 32.0);
        n.top = px(pos.y - 32.0);
        let _ = active;
    }
}
