# Effect source files (`.fx.ron`)

One Niagara-style **System** per file. Check and cook with:

```sh
cargo run -p fx_core --bin fxc -- check effects/samples/fountain.fx.ron
cargo run -p fx_core --bin fxc -- cook  effects/samples/fountain.fx.ron   # writes fountain.fxb
cargo run -p fx_core --bin fxc -- modules                                  # modules, stages, inputs
```

Shape: `( name, user_parameters: { "User.X": <literal> }, system_spawn, system_update, emitters: [ ( name, sim_target: Gpu|Cpu, max_particles, emitter_spawn, emitter_update, particle_spawn, particle_update, renderers: [ ... ] ) ] )`.
Each stage is a list of `(module: "Niagara Module Name", enabled: true, inputs: { "Input Name": <input> })`.

**Renderers** (Niagara's emitter renderers) say how particles are drawn. `renderers` is a list; an emitter without it draws nothing. Only `Sprite(( ... ))` exists so far (Ribbon, Mesh and Light will follow). Every field is optional and defaults to Niagara's value:

| Field | Values | Default |
|---|---|---|
| `blend_mode` | `Alpha`, `Additive`, `Premultiplied` | `Alpha` |
| `lit` | `true`, `false` | `false` |
| `facing_mode` | `FaceCamera`, `FaceCameraPlane`, `FaceCameraPosition`, `CustomFacingVector` | `FaceCamera` |
| `custom_facing_vector` | `(x, y, z)` | `(0.0, 0.0, 1.0)` |
| `alignment` | `Unaligned`, `VelocityAligned`, `CustomAlignment` | `Unaligned` |
| `custom_alignment_vector` | `(x, y, z)` | `(0.0, 1.0, 0.0)` |
| `pivot_in_uv_space` | `(u, v)` | `(0.5, 0.5)` |
| `sub_image_size` | `(columns, rows)` flipbook, whole numbers >= 1 | `(1, 1)` |
| `sub_uv_blending` | `true`, `false` | `false` |
| `sort_mode` | `None`, `ViewDepth`, `ViewDistance` | `None` |
| `soft_particle_fade_distance` | metres >= 0 (0 = off) | `0.0` |
| `texture` | `"asset/path.png"` | none (built-in soft round sprite) |

E.g. `renderers: [Sprite((blend_mode: Additive, alignment: VelocityAligned))]`. `fxc modules` lists these too.

Every input states its type:

| Form | Example |
|---|---|
| literal | `Float(1.0)`, `Int(5)`, `Bool(true)`, `Vec2((x, y))`, `Vec3((x, y, z))`, `Vec4(..)`, `Color((r, g, b, a))` (linear), `Quat((x, y, z, w))` |
| parameter link | `Link(Vec3, "User.Wind")`, `Link(Float, "Engine.DeltaTime")`, `Link(Vec3, "Particles.Position")` |
| random range | `RandomRange(Float(1.0), Float(2.0))` (per component, uniform) |
| curve over normalised age | `CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)]))` (particle stages only, linear keys) |

Units are Bevy's: metres, seconds, Y up (Niagara's -980 cm/s² Z gravity is `(0.0, -9.81, 0.0)` here).
Put `Particle State` first in `particle_update` (ages and kills particles) and `Solve Forces and Velocity` after the force modules.

`samples/` hold valid effects; `invalid/` hold broken ones whose first line `// expect: <line>: <text>` is the error the validator must report (tested).
