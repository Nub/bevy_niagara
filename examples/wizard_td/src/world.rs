//! The map, made from primitives: grass, a dirt path, the tower, trees,
//! rocks, distant hills and a spawn gate; turrets placed by the player.

use bevy::asset::RenderAssetUsages;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;

use crate::{Rng, Round, Turret, TurretCrystal};

/// The tower the foxes walk to.
pub const TOWER: Vec3 = Vec3::new(0.0, 0.0, 3.0);

/// The path the foxes walk, (x, z) waypoints from the gate to the tower.
pub const PATH: &[(f32, f32)] = &[
    (-30.0, -34.0),
    (-30.0, -16.0),
    (-12.0, -16.0),
    (-12.0, -28.0),
    (12.0, -28.0),
    (12.0, -12.0),
    (0.0, -12.0),
    (0.0, 0.4),
];

fn wp(i: usize) -> Vec3 {
    Vec3::new(PATH[i].0, 0.0, PATH[i].1)
}

pub fn path_length() -> f32 {
    (1..PATH.len()).map(|i| wp(i).distance(wp(i - 1))).sum()
}

/// Point and direction `d` metres along the path.
pub fn path_point(mut d: f32) -> (Vec3, Vec3) {
    for i in 1..PATH.len() {
        let (a, b) = (wp(i - 1), wp(i));
        let len = a.distance(b);
        let dir = (b - a) / len;
        if d <= len || i == PATH.len() - 1 {
            return (a + dir * d.min(len), dir);
        }
        d -= len;
    }
    (wp(0), Vec3::Z)
}

/// Horizontal distance from `p` to the path.
pub fn path_distance(p: Vec3) -> f32 {
    let p = Vec2::new(p.x, p.z);
    (1..PATH.len())
        .map(|i| {
            let a = Vec2::new(PATH[i - 1].0, PATH[i - 1].1);
            let b = Vec2::new(PATH[i].0, PATH[i].1);
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            p.distance(a + ab * t)
        })
        .fold(f32::MAX, f32::min)
}

/// Flat-shaded (low-poly look).
pub fn flat(mesh: impl Into<Mesh>) -> Mesh {
    let mut m: Mesh = mesh.into();
    m.duplicate_vertices();
    m.compute_flat_normals();
    m
}

fn mat(mats: &mut Assets<StandardMaterial>, c: Color, rough: f32) -> Handle<StandardMaterial> {
    mats.add(StandardMaterial {
        base_color: c,
        perceptual_roughness: rough,
        ..default()
    })
}

/// A flat-shaded grass plane with a little height noise at the edges.
fn ground(size: f32, n: u32, rng: &mut Rng) -> Mesh {
    let mut m = Plane3d::default().mesh().size(size, size).subdivisions(n).build();
    if let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos)) = m.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
        for p in pos.iter_mut() {
            let r = Vec2::new(p[0], p[2]).length();
            // Flat where the game is played, rolling hills beyond.
            let k = ((r - 42.0) / 20.0).clamp(0.0, 1.0);
            p[1] = k * k * rng.range(0.0, 9.0) - 0.02 * (1.0 - k) * rng.next();
        }
    }
    let _ = RenderAssetUsages::default();
    flat(m)
}

pub fn build_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    let mut rng = Rng(0x1234_5678);
    let grass = mat(&mut mats, Color::srgb(0.30, 0.47, 0.22), 0.95);
    let dirt = mat(&mut mats, Color::srgb(0.52, 0.40, 0.26), 1.0);
    let stone = mat(&mut mats, Color::srgb(0.55, 0.54, 0.56), 0.85);
    let dark_stone = mat(&mut mats, Color::srgb(0.36, 0.35, 0.38), 0.9);
    let roof = mat(&mut mats, Color::srgb(0.25, 0.28, 0.55), 0.7);
    let bark = mat(&mut mats, Color::srgb(0.35, 0.24, 0.14), 0.9);
    let leaves = [
        mat(&mut mats, Color::srgb(0.16, 0.36, 0.17), 0.9),
        mat(&mut mats, Color::srgb(0.22, 0.42, 0.16), 0.9),
        mat(&mut mats, Color::srgb(0.12, 0.30, 0.20), 0.9),
    ];
    let rock = mat(&mut mats, Color::srgb(0.47, 0.46, 0.44), 0.95);
    let crystal = mats.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.8, 1.0),
        emissive: LinearRgba::rgb(1.5, 3.0, 6.0),
        ..default()
    });

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.8, 1.0),
        brightness: 500.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            color: Color::srgb(1.0, 0.93, 0.82),
            shadow_maps_enabled: !cfg!(target_arch = "wasm32"),
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 15.0,
            maximum_distance: 70.0,
            num_cascades: 2,
            ..default()
        }
        .build(),
        Transform::from_xyz(-20.0, 30.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((Mesh3d(meshes.add(ground(150.0, 40, &mut rng))), MeshMaterial3d(grass)));

    // The path: one flat slab per segment, a square at each corner.
    let corner = meshes.add(Cuboid::new(3.0, 0.04, 3.0));
    for i in 0..PATH.len() {
        let b = wp(i);
        commands.spawn((Mesh3d(corner.clone()), MeshMaterial3d(dirt.clone()), Transform::from_translation(b + Vec3::Y * 0.02)));
        if i == 0 {
            continue;
        }
        let a = wp(i - 1);
        let len = a.distance(b);
        let dir = (b - a) / len;
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(3.0, 0.04, len))),
            MeshMaterial3d(dirt.clone()),
            Transform::from_translation((a + b) / 2.0 + Vec3::Y * 0.02)
                .with_rotation(Quat::from_rotation_y(dir.x.atan2(dir.z))),
        ));
    }

    // The tower: an octagonal keep, battlements, a roof and a floating crystal.
    commands.spawn((
        Mesh3d(meshes.add(flat(Cylinder::new(2.3, 7.0).mesh().resolution(8)))),
        MeshMaterial3d(stone.clone()),
        Transform::from_translation(TOWER + Vec3::Y * 3.5),
    ));
    commands.spawn((
        Mesh3d(meshes.add(flat(Cylinder::new(2.8, 0.9).mesh().resolution(8)))),
        MeshMaterial3d(dark_stone.clone()),
        Transform::from_translation(TOWER + Vec3::Y * 7.3),
    ));
    let merlon = meshes.add(Cuboid::new(0.6, 0.7, 0.6));
    for k in 0..8 {
        let a = k as f32 / 8.0 * std::f32::consts::TAU;
        commands.spawn((
            Mesh3d(merlon.clone()),
            MeshMaterial3d(dark_stone.clone()),
            Transform::from_translation(TOWER + Vec3::new(2.45 * a.cos(), 8.1, 2.45 * a.sin())),
        ));
    }
    commands.spawn((
        Mesh3d(meshes.add(flat(Cone { radius: 2.2, height: 3.2 }.mesh().resolution(8)))),
        MeshMaterial3d(roof),
        Transform::from_translation(TOWER + Vec3::Y * 9.6),
    ));
    commands.spawn((
        Mesh3d(meshes.add(flat(Sphere::new(0.6).mesh().ico(0).unwrap()))),
        MeshMaterial3d(crystal.clone()),
        TurretCrystal(TOWER.y + 12.0),
        Transform::from_translation(TOWER + Vec3::Y * 12.0),
    ));
    commands.spawn((
        PointLight {
            color: Color::srgb(0.6, 0.8, 1.0),
            intensity: 300_000.0,
            range: 20.0,
            ..default()
        },
        Transform::from_translation(TOWER + Vec3::Y * 12.0),
    ));
    // A door facing the path.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.2, 2.0, 0.3))),
        MeshMaterial3d(bark.clone()),
        Transform::from_translation(TOWER + Vec3::new(0.0, 1.0, -2.25)),
    ));

    // The gate the foxes come from.
    let g = wp(0);
    let pillar = meshes.add(flat(Cylinder::new(0.6, 4.5).mesh().resolution(6)));
    for x in [-2.2, 2.2] {
        commands.spawn((Mesh3d(pillar.clone()), MeshMaterial3d(dark_stone.clone()), Transform::from_translation(g + Vec3::new(x, 2.25, 0.0))));
    }
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 0.9, 1.3))),
        MeshMaterial3d(dark_stone.clone()),
        Transform::from_translation(g + Vec3::Y * 4.8),
    ));

    // Trees and rocks, off the path and away from the tower.
    let trunk = meshes.add(flat(Cylinder::new(0.22, 1.4).mesh().resolution(5)));
    let cone_lo = meshes.add(flat(Cone { radius: 1.4, height: 2.4 }.mesh().resolution(7)));
    let cone_hi = meshes.add(flat(Cone { radius: 1.0, height: 1.9 }.mesh().resolution(7)));
    let boulder = meshes.add(flat(Sphere::new(1.0).mesh().ico(0).unwrap()));
    let mut placed: Vec<Vec3> = Vec::new();
    let mut tries = 0;
    while placed.len() < 110 && tries < 4000 {
        tries += 1;
        let r = 60.0;
        let p = Vec3::new(rng.range(-r, r), 0.0, rng.range(-r, r));
        if path_distance(p) < 3.5
            || Vec2::new(p.x - TOWER.x, p.z - TOWER.z).length() < 9.0
            || placed.iter().any(|q| q.distance(p) < 2.6)
        {
            continue;
        }
        // Fewer in the middle of the field, so the turrets have room.
        if p.length() < 26.0 && rng.next() < 0.7 {
            continue;
        }
        placed.push(p);
        if rng.next() < 0.8 {
            let s = rng.range(0.8, 1.5);
            let leaf = leaves[(rng.next() * 3.0) as usize % 3].clone();
            commands
                .spawn((
                    Transform::from_translation(p).with_scale(Vec3::splat(s)).with_rotation(Quat::from_rotation_y(rng.range(0.0, 6.3))),
                    Visibility::default(),
                ))
                .with_children(|c| {
                    c.spawn((Mesh3d(trunk.clone()), MeshMaterial3d(bark.clone()), Transform::from_xyz(0.0, 0.7, 0.0)));
                    c.spawn((Mesh3d(cone_lo.clone()), MeshMaterial3d(leaf.clone()), Transform::from_xyz(0.0, 2.3, 0.0)));
                    c.spawn((Mesh3d(cone_hi.clone()), MeshMaterial3d(leaf), Transform::from_xyz(0.0, 3.5, 0.0)));
                });
        } else {
            let s = Vec3::new(rng.range(0.5, 1.3), rng.range(0.4, 0.9), rng.range(0.5, 1.3));
            commands.spawn((
                Mesh3d(boulder.clone()),
                MeshMaterial3d(rock.clone()),
                Transform::from_translation(p).with_scale(s).with_rotation(Quat::from_rotation_y(rng.range(0.0, 6.3))),
            ));
        }
    }
}

/// A turret: a hexagonal stone plinth with a floating crystal that casts.
pub fn spawn_turret(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    at: Vec3,
) {
    let stone = mat(mats, Color::srgb(0.45, 0.43, 0.5), 0.85);
    let glow = mats.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.6, 1.0),
        emissive: LinearRgba::rgb(4.0, 1.5, 7.0),
        ..default()
    });
    commands
        .spawn((
            Turret { cooldown: 0.5, shots: 0 },
            Transform::from_translation(at),
            Visibility::default(),
            Round,
        ))
        .with_children(|c| {
            c.spawn((
                Mesh3d(meshes.add(flat(Cylinder::new(0.75, 0.5).mesh().resolution(6)))),
                MeshMaterial3d(stone.clone()),
                Transform::from_xyz(0.0, 0.25, 0.0),
            ));
            c.spawn((
                Mesh3d(meshes.add(flat(Cylinder::new(0.4, 2.0).mesh().resolution(6)))),
                MeshMaterial3d(stone),
                Transform::from_xyz(0.0, 1.5, 0.0),
            ));
            c.spawn((
                Mesh3d(meshes.add(flat(Sphere::new(0.35).mesh().ico(0).unwrap()))),
                MeshMaterial3d(glow),
                TurretCrystal(3.1),
                Transform::from_xyz(0.0, 3.1, 0.0).with_scale(Vec3::new(0.8, 1.4, 0.8)),
            ));
            c.spawn((
                PointLight {
                    color: Color::srgb(0.8, 0.5, 1.0),
                    intensity: 40_000.0,
                    range: 6.0,
                    ..default()
                },
                Transform::from_xyz(0.0, 3.1, 0.0),
            ));
        });
}
