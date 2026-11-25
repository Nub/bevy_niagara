#!/bin/sh
# Renders every library effect's preview strip into effects/library/previews.
# Run from the repository root inside the fork's nix dev shell.
set -e
cargo build -q --release -p fx_preview --bin fxpreview
P=target/release/fxpreview
L=effects/library
O=$L/previews
$P $L/campfire.fx.ron     --out $O --times 0.5,1,2,3      --camera 0,1.0,3.2 --target 0,0.7,0 --no-mp4
$P $L/sparks_burst.fx.ron --out $O --times 0.05,0.2,0.4,0.7 --camera 0,1.2,4 --target 0,0.9,0 --no-mp4
$P $L/rising_smoke.fx.ron --out $O --times 1,3,5,7        --camera 0,1.6,5.5 --target 0,1.4,0 --backdrop grey --no-mp4
$P $L/magic_swirl.fx.ron  --out $O --times 0.5,1,2,3      --camera 0,1.2,3.5 --target 0,0.9,0 --no-mp4
$P $L/sword_trail.fx.ron  --out $O --times 0.2,0.4,0.6,1  --camera 0,1.3,3.5 --target 0,1.2,0 --no-mp4
$P $L/debris.fx.ron       --out $O --times 0.1,0.4,0.8,1.4 --camera 0,1.3,5 --target 0,1.0,0 --no-mp4
