# Benchmarks

Fixed headless scenes on the real renderer (fx_render in Bevy's raster
pipeline), 1920x1080 offscreen, one fixed 1/60 s sim step per frame. Each
scene waits for pipelines, warms up 180 frames (spawn-rate effects reach steady
state), then measures 300 frames. GPU ms per pass come from Bevy's render
diagnostics (wgpu timestamp queries); medians are reported.

Run on the RTX 3080 host (claim it as a heavy job, it is shared):

```sh
nix develop /home/zach/helm-projects/bevy-solari-opt -c cargo run -p fx_bench --release
```

This writes `bench/results/<commit>.json` (`<commit>-dirty.json`, git-ignored,
when the tree has uncommitted changes), prints a table and compares with the
newest earlier committed result. Commit the JSON so the history stays in the
repo. Exit code 3 means a regression.

| Scene | File | Budget (total GPU ms) |
|---|---|---|
| `sprites_1m` | `scenes/sprites_1m.fx.ron` | 2.0 (1M sprites simulated and drawn) |
| `sprites_1m_sorted` | `scenes/sprites_1m_sorted.fx.ron` | none yet (radix sort + sorted-sprite payload: 2.3 ms, was 5.0 with the bitonic sort) |
| `fire` | `scenes/fire.fx.ron` | 0.2 |
| `sparks` | `scenes/sparks.fx.ron` | 0.2 |
| `lib_*` (6) | `../effects/library/*.fx.ron` | 0.2 each (reference effect library) |

Passes: `sim` = `fx_sim` (spawn/update/compaction), `sort` = `fx_cull_sort`
(cull, depth sort, indirect args), `draw` = `main_transparent_pass_3d`,
`rt_build` = `fx_rt_build` (null until particles join Solari). Every span is
also kept raw under `passes`. CPU: `frame` is the median wall time of a frame
(no pipelined rendering), `fx_encode` the CPU time recording the fx passes.
`allocs_per_frame` counts the whole process (Bevy + wgpu + fx).

Other commands:

```sh
fx_bench run --scenes fire,sparks --frames 300 --out FILE --base FILE --no-compare
fx_bench compare BASE.json NEW.json   # exit 3 on regression
fx_bench dashboard                    # tile data for the Performance dashboard
fx_bench list
```

Showcase systems that cannot be committed (licensed imports, git-ignored)
are benched opt-in from a folder: every `*.fx.ron` under it becomes a scene
`show_<stem>` (camera at 0,2,6 looking at 0,1,0). The result must go to
`--out` (never `bench/results`, so the dashboard history only holds
committed scenes):

```sh
fx_bench run --scenes-dir DIR --out FILE [--scenes a,b] [--base FILE]
```

A regression is a metric over 10% slower than the base AND over a noise floor
(0.005 ms GPU, 0.01 ms CPU encode, 2 allocations), so microsecond jitter on
tiny passes is not flagged. CPU frame time is reported but not compared (it
depends on what else runs on the shared host).
