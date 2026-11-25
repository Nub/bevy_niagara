# Reference effect library

Claude-authored effects that each follow a common Niagara tutorial recipe, so
they double as templates and as proof that Niagara tutorials carry over. Each
file's header lists the tutorial steps it follows, step by step, and what the
model cannot express yet. They are checked by `fx_core` (`tests/assets.rs`:
validate, cook, simulate, header and preview present), compared CPU vs GPU in
`fx_gpu/tests/parity.rs`, and benchmarked as the `lib_*` scenes of `fx_bench`
(budget 0.2 ms GPU each on the 3080 at 1080p).

| Effect | Tutorial | Renderers | Preview |
|---|---|---|---|
| `campfire.fx.ron` | Fire (flames, embers, smoke, light) | Sprite, Light | `previews/campfire_strip.png` |
| `sparks_burst.fx.ron` | Sparks burst | Sprite (velocity aligned) | `previews/sparks_burst_strip.png` |
| `rising_smoke.fx.ron` | Smoke | Sprite (lit, sorted, soft) | `previews/rising_smoke_strip.png` |
| `magic_swirl.fx.ron` | Magic swirl | Sprite, Ribbon | `previews/magic_swirl_strip.png` |
| `sword_trail.fx.ron` | Sword / weapon trail | Ribbon | `previews/sword_trail_strip.png` |
| `debris.fx.ron` | Mesh explosion debris | Mesh, Sprite | `previews/debris_strip.png` |
| `shockwave_ring.fx.ron` | CGHOW, Radial to Shockwave | Sprite (custom material) | `previews/shockwave_ring_strip.png` |
| `lightning_strike.fx.ron` | CGHOW, Stylized Lightning | Ribbon (custom materials), Sprite | `previews/lightning_strike_strip.png` |
| `portal_vortex.fx.ron` | CGHOW, Portal Gate | Sprite, Light | `previews/portal_vortex_strip.png` |
| `stylized_tornado.fx.ron` | CGHOW, Stylized Tornado | Mesh (torus, erode material), Sprite | `previews/stylized_tornado_strip.png` |
| `rain_splash.fx.ron` | Matt Aspland, Rain Splashback | Sprite (collision events) | `previews/rain_splash_strip.png` |
| `muzzle_flash.fx.ron` | Gabriel Aguiar, Muzzle Flash | Sprite (flipbook), Mesh (cone) | `previews/muzzle_flash_strip.png` |
| `footstep_dust.fx.ron` | Vive Games, Footsteps Dust | Sprite (sorted, soft) | `previews/footstep_dust_strip.png` |
| `heal_aura.fx.ron` | CGHOW, Healing Aura | Sprite, Mesh (cylinder), Light | `previews/heal_aura_strip.png` |
| `fireworks_show.fx.ron` | Enrique Ventura, Advanced Fireworks ch. 3 | Sprite (death/location events) | `previews/fireworks_show_strip.png` |
| `meteor_rain.fx.ron` | Gabriel Aguiar, Meteor Rain | Sprite, Ribbon (location/collision events) | `previews/meteor_rain_strip.png` |

Regenerate the previews (needs the GPU; camera per effect as in `fx_bench`'s
`SCENES`):

```sh
nix develop /home/zach/helm-projects/bevy-solari-opt -c sh effects/library/previews.sh
```

## GPU cost on the RTX 3080 (fx_bench, 1080p, commit 312ed48)

| Effect | Emitters | Sim | Sort | Draw | Total GPU ms | Budget 0.2 |
|---|---|---|---|---|---|---|
| campfire | 4 | 0.147 | 0.067 | 0.054 | 0.268 | over by 0.068 |
| sparks_burst | 2 | 0.065 | 0.034 | 0.004 | 0.102 | ok |
| rising_smoke | 1 | 0.038 | 0.035 | 0.081 | 0.154 | ok |
| magic_swirl | 5 | 0.180 | 0.034 | 0.013 | 0.227 | over by 0.027 |
| sword_trail | 2 | 0.070 | 0.001 | 0.008 | 0.079 | ok |
| debris | 2 | 0.062 | 0.048 | 0.010 | 0.121 | ok |

Sim time grows with the number of emitters (about 0.035 ms each), not with
particle count (all are under 1000 particles), so the two effects over budget
are over because of fixed per-emitter cost.

## Tutorial batch 2: assets and materials

The batch-2 effects each name their tutorial (YouTube link) and map its
chapters to modules in the file header, with the steps that could not be
followed listed as gaps. Their textures are recreated procedurally
(`textures/make_textures.sh`) and their tutorial materials are WESL files in
`materials/`. The one tutorial that gives its assets away (the fireworks
flares) is fetched, with the reference frames used for comparisons, by
`sh testdata/tutorial_assets.sh` into the git-ignored
`testdata/tutorial_assets/` (manifest and terms in the script);
`tutorial_assets` here links to it. Without it, `fireworks_show` draws the
built-in soft sprite.
