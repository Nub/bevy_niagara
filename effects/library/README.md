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

Regenerate the previews (needs the GPU; camera per effect as in `fx_bench`'s
`SCENES`):

```sh
nix develop /home/zach/helm-projects/bevy-solari-opt -c sh effects/library/previews.sh
```
