# Effect source files (`.fx.ron`)

One Niagara-style **System** per file. Check and cook with:

```sh
cargo run -p fx_core --bin fxc -- check effects/samples/fountain.fx.ron
cargo run -p fx_core --bin fxc -- cook  effects/samples/fountain.fx.ron   # writes fountain.fxb
cargo run -p fx_core --bin fxc -- modules                                  # modules, stages, inputs
```

Shape: `( name, user_parameters: { "User.X": <literal> }, system_spawn, system_update, emitters: [ ( name, sim_target: Gpu|Cpu, max_particles, emitter_spawn, emitter_update, particle_spawn, particle_update ) ] )`.
Each stage is a list of `(module: "Niagara Module Name", enabled: true, inputs: { "Input Name": <input> })`.

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
