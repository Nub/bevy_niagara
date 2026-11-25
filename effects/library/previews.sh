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
# Tutorial batch 2 (fireworks_show uses the tutorial's flares when
# testdata/tutorial_assets.sh has fetched them).
$P $L/shockwave_ring.fx.ron   --out $O --times 0.05,0.12,0.25,0.45 --camera 0,1,4.5 --target 0,1,0 --no-mp4
$P $L/lightning_strike.fx.ron --out $O --times 0.02,0.1,0.22,0.85 --camera 0,1.7,6 --target 0,1.6,0 --no-mp4
$P $L/portal_vortex.fx.ron    --out $O --times 0.5,1,2,3          --camera 0,1.3,4.5 --target 0,1.3,0 --no-mp4
$P $L/stylized_tornado.fx.ron --out $O --times 1,2,3,4            --camera 0,2.3,9 --target 0,2.1,0 --no-mp4
$P $L/rain_splash.fx.ron      --out $O --times 1,1.5,2,3          --camera 0,1.7,6 --target 0,1,0 --backdrop grey --no-mp4
$P $L/muzzle_flash.fx.ron     --out $O --times 0.02,0.04,0.06,0.42 --camera 0,2.2,2.4 --target 0,1.35,0 --backdrop grey --no-mp4
$P $L/footstep_dust.fx.ron    --out $O --times 0.5,1,1.5,2.5      --camera -0.6,1.1,3 --target -0.6,0.15,0 --backdrop grey --no-mp4
$P $L/heal_aura.fx.ron        --out $O --times 1,1.5,2,3          --camera 0,1.6,4 --target 0,0.6,0 --no-mp4
$P $L/fireworks_show.fx.ron   --out $O --times 1.5,2,2.5,3.5      --camera 0,6,18 --target 0,6,0 --no-mp4
$P $L/meteor_rain.fx.ron      --out $O --times 1.5,2,2.5,3        --camera 0,2.5,9 --target 0,1.5,0 --backdrop grey --no-mp4
