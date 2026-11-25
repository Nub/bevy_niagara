# Effect source files (`.fx.ron`)

One Niagara-style **System** per file. Check and cook with:

```sh
cargo run -p fx_core --bin fxc -- check effects/samples/fountain.fx.ron
cargo run -p fx_core --bin fxc -- cook  effects/samples/fountain.fx.ron   # writes fountain.fxb
cargo run -p fx_core --bin fxc -- modules                                  # modules, stages, inputs
```

Shape: `( name, duration, scalability: ( ... ), user_parameters: { "User.X": <literal> }, system_spawn, system_update, emitters: [ ( name, sim_target: Gpu|Cpu, max_particles, emitter_spawn, emitter_update, particle_spawn, particle_update, event_handlers: [ ... ], renderers: [ ... ] ) ] )`.
Each stage is a list of `(module: "Niagara Module Name", enabled: true, inputs: { "Input Name": <input> })`.

**Renderers** (Niagara's emitter renderers) say how particles are drawn. `renderers` is a list of `Sprite(( ... ))`, `Ribbon(( ... ))`, `Mesh(( ... ))` and `Light(( ... ))`; an emitter without it draws nothing. Every field is optional and defaults to Niagara's value. `Sprite`:

| Field | Values | Default |
|---|---|---|
| `blend_mode` | `Alpha`, `Additive`, `Premultiplied` | `Alpha` |
| `lit` | `true`, `false` | `false` |
| `facing_mode` | `FaceCamera`, `FaceCameraPlane`, `FaceCameraPosition`, `CustomFacingVector` | `FaceCamera` |
| `custom_facing_vector` | `(x, y, z)` | `(0.0, 0.0, 1.0)` |
| `alignment` | `Unaligned`, `VelocityAligned`, `CustomAlignment` | `Unaligned` |
| `custom_alignment_vector` | `(x, y, z)` | `(0.0, 1.0, 0.0)` |
| `pivot_in_uv_space` | `(u, v)` | `(0.5, 0.5)` |
| `sub_image_size` | `(columns, rows)` flipbook, whole numbers >= 1; played over each particle's life, or as `SubUV Animation` sets `Particles.SubImageIndex` | `(1, 1)` |
| `sub_uv_blending` | `true`, `false` | `false` |
| `sort_mode` | `None`, `ViewDepth`, `ViewDistance` | `None` |
| `soft_particle_fade_distance` | metres >= 0 (0 = off) | `0.0` |
| `texture` | `"asset/path.png"` | none (built-in soft round sprite) |
| `material` | `"asset/path.wesl"` (custom WESL fragment shader) | none (built-in) |

`Ribbon` joins the particles of each `Particles.RibbonID` (set by `Initialize Ribbon`; width and twist from `Particles.RibbonWidth`, `Particles.RibbonTwist`):

| Field | Values | Default |
|---|---|---|
| `blend_mode` | `Alpha`, `Additive`, `Premultiplied` | `Alpha` |
| `facing_mode` | `Screen`, `CustomFacingVector` | `Screen` |
| `custom_facing_vector` | `(x, y, z)` | `(0.0, 0.0, 1.0)` |
| `uv_mode` | `ScaledUniformly`, `ScaledUsingRibbonSegmentLength`, `TiledOverRibbonLength` | `ScaledUniformly` |
| `tiling_length` | metres > 0 per texture repeat | `1.0` |
| `link_order` | `SpawnOrder`, `Custom` (by `Particles.RibbonLinkOrder`, ascending) | `SpawnOrder` |
| `tessellation_factor` | subdivisions per segment, 1..=16 | `1` |
| `ray_tracing` | Solari: in reflections and shadows, lights the scene (needs `FxSolariPlugin`) | `false` |
| `texture`, `material` | asset paths | none |

`Mesh` draws a mesh per particle, scaled by `Particles.Scale` and rotated by `Particles.MeshOrientation` (`Initialize Mesh`, `Update Mesh Orientation`):

| Field | Values | Default |
|---|---|---|
| `mesh` | `"cube"`, `"sphere"`, `"torus"`, `"cylinder"`, `"cone"`, `"capsule"`, or an asset path (contains `.` or `/`) | `"cube"` |
| `blend_mode` | `Opaque`, `Alpha`, `Additive`, `Premultiplied` | `Opaque` |
| `facing_mode` | `Default`, `Velocity`, `CameraPosition`, `CameraPlane` | `Default` |
| `lit` | `true`, `false` | `true` |
| `sort_mode` | `None`, `ViewDepth`, `ViewDistance` | `None` |
| `texture`, `material` | asset paths | none |

`Light` makes each particle a point light of radius `Particles.LightRadius * radius_scale` (`Initialize Light`) and the particle's colour:

| Field | Values | Default |
|---|---|---|
| `radius_scale` | >= 0 | `1.0` |
| `color_add` | `(r, g, b)` added to the particle colour | `(0.0, 0.0, 0.0)` |
| `intensity` | lumens per unit colour, >= 0 | `10000.0` |
| `alpha_scales_brightness` | `true`, `false` | `true` |
| `max_lights` | 1..=256 per emitter | `16` |

Custom materials can read `Particles.DynamicMaterialParameter` (set by `Dynamic Material Parameters`).

E.g. `renderers: [Sprite((blend_mode: Additive, alignment: VelocityAligned))]`. `fxc modules` lists these too.

Every input states its type:

| Form | Example |
|---|---|
| literal | `Float(1.0)`, `Int(5)`, `Bool(true)`, `Vec2((x, y))`, `Vec3((x, y, z))`, `Vec4(..)`, `Color((r, g, b, a))` (linear), `Quat((x, y, z, w))` |
| parameter link | `Link(Vec3, "User.Wind")`, `Link(Float, "Engine.DeltaTime")`, `Link(Vec3, "Particles.Position")` |
| random range | `RandomRange(Float(1.0), Float(2.0))` (per component, uniform) |
| curve over normalised age | `CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)]))` (particle stages only, linear keys) |
| static switch (enum) | `"Shape Primitive": Box`, `"SubUV Animation Mode": InfiniteLoop` (a bare name; `fxc modules` lists the choices) |
| dynamic input (nestable) | `Dynamic("Multiply Vec3 by Float", { "A": Dynamic("Random Unit Vec3"), "B": Dynamic("Uniform Ranged Float", { "Minimum": Float(1.0), "Maximum": Float(3.0) }) })` |

Dynamic inputs are Niagara's (Uniform Ranged, Random Range by Uniform, `<Type> from Curve` at any normalised index, Add/Subtract/Multiply/Lerp, Make/Break, Normalize, Length, Distance, Sine/Cosine over Time, Select by Bool/Int, Random from List, Random Unit Vec3, Normalized Emitter/System Age, Compare Floats). Their inputs take any form above, nested; left-out inputs take defaults; lists are `[Float(1.0), Float(2.0)]`; a curve input is a bare curve `Float([(0.0, 1.0), (1.0, 0.0)])`. `fxc modules` lists them with their types; `samples/dynamic_inputs.fx.ron` uses most of them.

**Collisions and events** (Niagara's Collision module, event generators and event handlers):

- `Collision` (in `particle_update`, after `Solve Forces and Velocity`) moves particles that crossed a surface back onto it and bounces (`Restitution`, `Friction`; slower than `Rest Speed` rests), stops or kills them. `"Collision Type": Plane` is an analytic plane (`Plane Normal`, `Plane Origin`; same on CPU and GPU); `SceneDepth` reads the first 3D camera's depth buffer (GPU, on-screen only, one frame late); `RayTraced` traces rays in Solari's scene (with `FxSolariPlugin`; opaque geometry, previous frame's TLAS), else falls back to `SceneDepth`. On the CPU, `SceneDepth` and `RayTraced` hit `Simulation::colliders` set by the host.
- `Generate Death Event`, `Generate Collision Event` (after `Collision`) and `Generate Location Event` (with `Event Probability`) in `particle_update` write events: position, velocity, normal (collisions), age and the particle's UniqueID.
- A receiving emitter lists `event_handlers: [(source: "Shells", event: Death, execution_mode: SpawnedParticles, spawn_number: 150, max_events_per_frame: 0, modules: [ ... ])]`. `SpawnedParticles` spawns `spawn_number` particles per event, which run `particle_spawn` then `modules`; `EveryParticle` runs `modules` on every living particle once per event (not on its own events). `modules` read `Event.Position`, `Event.Velocity`, `Event.Normal`, `Event.Age`, `Event.ParticleID`; `Receive Death/Collision/Location Event` puts the particle at the event (`Position Offset`, `Inherit Velocity Scale`). Events are ordered by source UniqueID; `max_events_per_frame` (0 = all) keeps the first ones. A receiver gets an earlier emitter's (or its own) events the same step, a later emitter's the next step.

`samples/bouncing_sparks.fx.ron` (floor collisions, impact flashes from collision events) and `samples/fireworks.fx.ron` (trails from location events, bursts from death events) show both.

**Data interfaces** (Niagara's) are typed `User.` parameters giving modules scene and asset data. Declare them in `user_parameters` beside the values:

| Declaration | What it is |
|---|---|
| `"User.Character": SkeletalMesh("models/Fox.glb")` | a skinned mesh; the default asset is a glTF file whose scene 0 is spawned under the effect, playing its first clip (`"models/Fox.glb#Animation2"` plays clip 2) |
| `"User.Rock": StaticMesh("sphere")` | a mesh: a built-in name (`cube`, `sphere`, ...) or an asset path (a `.glb` without a label reads its first primitive) |
| `"User.Pattern": Texture("textures/pattern.png")` | an image (Rgba8 sRGB/linear, Rgba16Float, Rgba32Float), sampled in linear colour |
| `"User.Fade": FloatCurve([(0.0, 1.0), (1.0, 0.0)])` (also `Vec2Curve`, `VectorCurve`, `Vec4Curve`, `ColorCurve`) | a curve baked into the effect, read with `Dynamic("Float from Curve", { "Curve": Link(FloatCurve, "User.Fade") })` |
| `"User.Camera": CameraQuery` | the camera the effect is seen from (`Get Camera Position`, `Get Camera Forward Vector`) |
| `"User.Depth": SceneDepth` | the scene depth buffer (`Sample Scene Depth`) |

Modules take one through an input of its type: `"Skeletal Mesh": Link(SkeletalMesh, "User.Character")` (`Skeletal Mesh Location`, `Sampling Mode: Surface | Bones`), `Link(StaticMesh, "User.Rock")` (`Static Mesh Location`), `Link(Texture, "User.Pattern")` (`Sample Texture`). `Camera Offset` sets `Particles.CameraOffset`: sprites and meshes are drawn that many metres towards the camera. Asset paths are relative to the effect file's folder in `fxpreview`. In a game, bind your own data on the `ParticleEffect` component (fx_render) instead of the default asset:

```rust
let mut effect = ParticleEffect::new(cooked, seed);
effect.bind_entity("User.Character", player);    // an entity with Mesh3d (+ SkinnedMesh: sampled in its animated pose)
effect.bind_mesh("User.Rock", meshes.add(Sphere::new(0.5)));  // a mesh at the effect's origin
effect.bind_texture("User.Pattern", asset_server.load("pattern.png"));
```

Meshes and images must keep their main-world data (`RenderAssetUsages::MAIN_WORLD`, the default for loaded assets). `samples/skeletal_mesh.fx.ron` (embers off the running fox) and `samples/texture_color.fx.ron` (a picture dissolving into coloured particles) use them.

**Custom WGSL modules** (Niagara's scratch pad / custom HLSL) are declared once in the effect and placed in stacks by name like built-in modules:

```ron
custom_modules: [
    (
        name: "Swirl",
        inputs: { "Spin": Float(2.0), "Center": Vec3((0.0, 0.0, 0.0)) },   // typed, with defaults
        outputs: ["Particles.Velocity"],                                    // attributes it writes
        wgsl: r#"
    let r = Position - Center;
    Velocity += vec3<f32>(-r.z, 0.0, r.x) * Spin * DeltaTime;
"#,
    ),
],
// in an emitter: particle_update: [ ..., (module: "Swirl", inputs: { "Spin": Link(Float, "User.Spin") }) ]
```

The body is a WGSL function body. It sees its inputs by name, every particle attribute by its name without `Particles.` (dots become `_`: `Initial_Color`; the `outputs` are `var`s written back, the rest read-only), `DeltaTime`, `Time`, `EmitterAge`, `rand(slot)` (uniform in [0, 1), per particle, module and step) and the helper library (`curl_noise(p)`, `noise_grad(p, channel)`, `euler_quat`, `quat_mul`, `quat_normalize`, `unit_or_y`, ...). Call inputs take every input form (links, random ranges, curves, dynamic inputs). Bodies are validated with naga when the effect is checked: errors give the line and column in the `.fx.ron` file (write bodies as raw strings `r#"..."#` so lines match). Custom modules, grids and simulation stages run on the GPU only (`sim_target: Gpu`); the CPU reference simulation skips them.

**Grids and simulation stages** (Niagara's Grid2D/Grid3D collections and simulation stages) are per emitter:

```ron
grids: [Grid2D(name: "Smoke", cells: (64, 96), min: (-1.0, 0.0), max: (1.0, 3.0),
               attributes: { "Velocity": Vec2, "Density": Float })],   // Grid3D: 3 numbers each
simulation_stages: [
    (name: "Advect", iterate: Grid("Smoke"), modules: [(module: "Advect")]),
    (name: "Pressure", iterate: Grid("Smoke"), iterations: 24, modules: [(module: "Jacobi Pressure")]),
    (name: "Relax", iterate: Particles, modules: [(module: "My Particle Module")]),
],
```

A 2D grid lies in the world XY plane. Stages run each step after the spawn, in order, each `iterations` times (1..=64). A grid stage runs custom modules once per cell; the body sees `Cell` (`vec2<i32>`/`vec3<i32>`), `CellPosition` (world) and every attribute of the grid as a `var`; it reads the previous values (stages ping-pong two copies), so neighbour reads are race-free. Every body of the emitter can read its grids: `Smoke_Density(cell)` (clamped to the grid), `Smoke_sample_Velocity(pos)` (bilinear/trilinear at a world position), `Smoke_cell(pos)`, `Smoke_cell_position(cell)` and the constants `Smoke_NumCells`, `Smoke_Min`, `Smoke_CellSize`. Particle stages run built-in and custom particle modules in place (no kills, collisions or events there). `samples/wgsl_swirl.fx.ron` (a custom particle module) and `samples/grid_smoke.fx.ron` (a 64x96 stable-fluids smoke grid moving the particles) use them.

**Lifecycle and scalability** (Niagara's Effect Type) are two optional system-level fields:

```ron
duration: 0.6,                 // seconds a one-shot keeps spawning
scalability: (                 // every field optional
    max_distance: 40.0,                 // m from the nearest camera to the bounds sphere; beyond it the instance is culled (default: none)
    spawn_count_scale_by_distance: [(10.0, 1.0), (40.0, 0.25)],  // (distance m, scale), piecewise linear, clamped at the ends
    cull_when_not_visible: true,        // default false
    visibility_grace: 0.5,              // s out of every view before the visibility cull applies; default 0.5
    bounds_radius: 2.0,                 // m, sphere around the effect origin used for distance and visibility; default 2.0
    max_instances: 32,                  // per system, the nearest are kept; 0 = unlimited (default)
    quality: (                          // per level, each optional, each field optional
        Low: (spawn_count_scale: 0.25, max_distance: 20.0, max_instances: 8),
        Medium: (spawn_count_scale: 0.5),
    ),
),
```

Without `duration`, the cook derives it: an emitter whose spawn modules are only `Spawn Burst Instantaneous` with constant Spawn Times stops spawning after its latest Spawn Time (no spawn modules: at once); if every emitter stops, the effect is a one-shot lasting the latest of them, otherwise (any `Spawn Rate`, or a Spawn Time that is not a constant) it loops. The game runtime stops spawning a one-shot after `duration` and despawns it when its particles are gone. A culled looping effect pauses until it is back in range or view; a culled one-shot ends. The spawn-count scale (the quality level's `spawn_count_scale` times the distance curve) multiplies every spawn request. The game picks the quality level (`Low`, `Medium`, `High` (default), `Epic`, `Cinematic`) at runtime; a level's `max_distance` and `max_instances` replace the base values. Distances, scales, `visibility_grace` and `bounds_radius` must be finite and at least 0 (`max_distance` and `duration` more than 0), curve distances strictly increasing.

**Using effects from a game** (`fx_render::game`; full example in `examples/arena`):

```rust
let impact: Arc<CookedSystem> = /* cook once, share the Arc: one program and one pool per system */;
let mut fx = ParticleEffect::shared(impact.clone(), seed);
fx.set_param("User.Tint", Color::linear_rgb(1.0, 0.4, 0.1))?;   // Err(ParamError::WrongType / Unknown)
commands.spawn_one_shot(fx, Transform::from_translation(hit));     // despawns itself when done
commands.attach_effect(hand_bone, ParticleEffect::shared(glow, 1), EffectMode::Looping, Transform::IDENTITY);
// EffectControl (on every effect): activate(), deactivate(), reset(), fire(), is_finished(), culled()
// Resources: FxQuality(QualityLevel::Low), FxScalability { enabled }, FxPoolStats::get()
```

Attach to a skeleton joint with `fx_render::find_joint(root, "b_Head", &children, &names)`. Particles simulate in the effect's local space, so attached effects carry their particles with them.

Units are Bevy's: metres, seconds, Y up (Niagara's -980 cm/s² Z gravity is `(0.0, -9.81, 0.0)` here).
Put `Particle State` first in `particle_update` (ages and kills particles) and `Solve Forces and Velocity` after the force modules.

`samples/` hold valid effects; `invalid/` hold broken ones whose first line `// expect: <line>: <text>` is the error the validator must report (tested).
