# Solari cost of ray-traced particles (task_34ea655cd4)

RTX 3080 (graybeard), 1920x1080, `fxsolari bench` default scene (campfire:
Flames 1024 + Embers 128 slots, sparks 512; 3 traced sprite emitters), 300
measured frames, medians. "Extra" is the sum of the Solari spans
(`solari_lighting/lighting`, `presample_light_tiles`, `world_cache`,
`light_selection`, `blas_build`, `solari_deform/blas_build`, `tlas_build`,
`setup_tlas_instances`, `fx_rt_proxy`) minus the same sum with `--mode off`.
Run-to-run noise is about 0.05 ms.

```sh
nix develop /home/zach/helm-projects/bevy-solari-opt -c cargo build --release -p fx_solari --bin fxsolari
B=target/release/fxsolari   # run through the nix shell (Vulkan loader)
$B bench --mode off
$B bench --mode on                              # new default: 24 traced, bounce-only
$B bench --mode on --rt-max 16383 --rt-lights   # old behaviour
FXSOLARI_RESTIR=1 $B bench --mode on            # ReSTIR
```

| Setup | Lighting ms | Extra ms |
|---|---|---|
| RT off | 1.03 | 0 |
| Before: every particle, sampled as lights | 6.27 | 5.82 |
| **Default: 24 per emitter, bounce-only** | 1.40 | **0.54** |
| 32 per emitter, bounce-only | 1.46 | 0.59 |
| 16 per emitter, bounce-only | 1.32 | 0.45 |
| 32 per emitter + `ray_tracing_lights` | 2.45 | 1.66 |
| 16 per emitter + `ray_tracing_lights` | 2.01 | 1.22 |
| ReSTIR, RT off | 2.97 | 0 |
| ReSTIR, before | 19.86 | 17.4 |
| ReSTIR, 32 bounce-only | 4.29 | 1.49 |
| ReSTIR, 32 + `ray_tracing_lights` | 10.37 | 7.6 |

Where the cost went (uncapped): rays crossing the fire run Solari's alpha
test (`candidate_surface`: full triangle resolve + the hit function) on every
flame quad; made invisible but still traced, the fire cost 7.5 ms (rays never
stop inside it). Light sampling aims shadow rays into the fire, ~0.7 ms more
even at 16 particles.

Raw span output: `raw.txt` (first sweep, 32/16 and light modes), `raw2.txt`
(24 and 32, and a second RT-off run).

## Mesh particles (task_b0bcd13f30)

`fxsolari bench --effect crates/fx_solari/scenes/debris_rt.fx.ron --mode off|on [--copies 4]`,
1920x1080, RTX 3080, 300 frames, median GPU ms. The debris chunks (80 lit opaque
cubes per burst, `ray_tracing: true`, one merged proxy mesh + opaque BLAS per emitter):

| span | off, 1 | on, 1 | off, 4 | on, 4 |
|---|---|---|---|---|
| fx_rt_proxy | - | 0.034 | - | 0.103 |
| solari_deform/blas_build | - | 0.191 | - | 0.196 |
| lighting/initial_path | 1.039 | 1.088 | 1.104 | 1.216 |

Extra cost: ~0.27 ms for one emitter, ~0.41 ms for four (BLAS rebuild is a
fixed ~0.19 ms per frame for any number of deformed meshes; opaque BLAS hits
skip the hit function's alpha test).
