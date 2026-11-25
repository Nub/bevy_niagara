# Effect source files (`.fx.ron`)

One Niagara-style **System** per file. Check and cook with:

```sh
cargo run -p fx_core --bin fxc -- check effects/samples/fountain.fx.ron
cargo run -p fx_core --bin fxc -- cook  effects/samples/fountain.fx.ron   # writes fountain.fxb
cargo run -p fx_core --bin fxc -- modules                                  # modules, stages, inputs
```

Shape: `( name, user_parameters: { "User.X": <literal> }, system_spawn, system_update, emitters: [ ( name, sim_target: Gpu|Cpu, max_particles, emitter_spawn, emitter_update, particle_spawn, particle_update, event_handlers: [ ... ], renderers: [ ... ] ) ] )`.
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

Units are Bevy's: metres, seconds, Y up (Niagara's -980 cm/s² Z gravity is `(0.0, -9.81, 0.0)` here).
Put `Particle State` first in `particle_update` (ages and kills particles) and `Solve Forces and Velocity` after the force modules.

`samples/` hold valid effects; `invalid/` hold broken ones whose first line `// expect: <line>: <text>` is the error the validator must report (tested).
