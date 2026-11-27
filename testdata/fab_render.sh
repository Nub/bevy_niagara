#!/bin/sh
# Converts and renders every system of the local Fab packs (testdata/fab,
# git-ignored: see fab_assets.sh) with fxpreview's Unreal-like look, for
# side-by-side comparisons with the listings' reference images and video.
#
# Output (git-ignored, never commit or share): testdata/fab/_look/<pack>/<system>/
# (.fx.ron, report, materials, strip, mp4, logs), and per pack a labelled grid
# video <pack>_grid.mp4, a still <pack>_grid.png (frame at GRID_T s: 1.5, arrows 0.9 as their trails die after ~1.5 s), and the
# whole pack in one scene laid out like the listing shots: <pack>_lineup/.
#
# Run from the repository root inside the fork's dev shell, after
#   cargo build --release -p fx_import --bin fximport -p fx_preview --bin fxpreview
#   nix develop /home/zach/helm-projects/bevy-solari-opt -c sh testdata/fab_render.sh [pack ...]
# Packs: perfect_fire arrow_trail lightning portals mixed_vfx mayu_basic
# niagara_examples m5_fire fire_explosion sparkles loot_drops (default: the first two). GPU: claim graybeard heavy.
set -eu
out=${OUT:-testdata/fab/_look}
bin=${BIN:-target/release}
packs=${*:-perfect_fire arrow_trail}

# Camera and motion per pack, matched to the listings: fires sit on the floor
# seen from standing height a few metres away; arrows fly past along +X just
# above the floor, seen from above and to the side.
view() {
	case $1 in
	perfect_fire) echo "--camera 0,2.0,5.75 --target 0,0.7,0 --fov 59 --duration 3" ;;
	arrow_trail) echo "--camera -7,3,-3.5 --target -1,0,0 --fov 59 --duration 1.6 --fly 20,0,0 --ground -3.5" ;;
	# Lightning and portals stand about 2 m tall, seen from the front; Mixed
	# VFX (projectiles in place, fires, explosions) and Mayu from a few metres.
	# Rings, spheres and the saw blade are centred on the system origin: lifted
	# 1 m so they clear the floor, as placed in the listings.
	lightning) echo "--camera 0,1.5,6 --target 0,1,0 --fov 59 --duration 3 --at 0,1,0" ;;
	portals) echo "--camera -3.5,1.6,5 --target 0,1.4,0 --fov 59 --duration 3" ;;
	mixed_vfx) echo "--camera 0,2.5,7 --target 0,1,0 --fov 59 --duration 3" ;;
	mayu_basic) echo "--camera 0,1.5,5 --target 0,1,0 --fov 59 --duration 3 --at 0,1,0" ;;
	# Epic's Niagara Examples: impacts, explosions, buffs at human scale; M5
	# fires on the floor like Perfect Fire; DeepBlues explosions up to ~8 m
	# tall; SoftTofu sparkles and Targi loot beams around a 1 m pickup.
	niagara_examples) echo "--camera 0,2,7 --target 0,1,0 --fov 59 --duration 3" ;;
	m5_fire) echo "--camera 0,2.0,5.75 --target 0,0.9,0 --fov 59 --duration 3" ;;
	fire_explosion) echo "--camera 0,4,16 --target 0,3,0 --fov 59 --duration 4" ;;
	sparkles) echo "--camera 0,1.5,5 --target 0,1,0 --fov 59 --duration 3 --at 0,1,0" ;;
	loot_drops) echo "--camera 0,2,6 --target 0,1.2,0 --fov 59 --duration 3" ;;
	esac
}
# The Unreal-like look. Exposure matched by eye to the listings; arrows fly at
# 20 m/s, the demo blueprint's Speed, over the demo map's blue-grey prototype
# grid (`--floor proto`, shown the same at any exposure), so the exposure only
# sets the trails against it: +5.5 stops makes the faint translucent ribbons
# (Basic, Basic_02, Holy) read as in the listing, with a tight bloom.
# BEFORE=1 renders with the plain default look (and an
# older fxpreview via BIN) for before/after shots.
look() {
	[ -n "${BEFORE:-}" ] && return
	case $1 in
	perfect_fire) echo "--look ue" ;;
	arrow_trail) echo "--look ue --floor proto --bloom 0.1 --exposure 5.5" ;;
	*) echo "--look ue" ;;
	esac
}
src() {
	case $1 in
	perfect_fire) echo testdata/fab/perfect_fire/Content/Vefects/Free_Fire/Shared/Particles ;;
	arrow_trail) echo testdata/fab/arrow_trail/Content/ArrowTrail/FX ;;
	lightning) echo testdata/fab/lightning/Content/GrimzaFX_simple_lightning/Particles ;;
	portals) echo testdata/fab/portals/Content/PortalFX/FX ;;
	mixed_vfx) echo testdata/fab/mixed_vfx/Content/MixedVFX/Particles ;;
	mayu_basic) echo testdata/fab/mayu_basic/Content/MayuAssets ;;
	niagara_examples) echo testdata/fab/niagara_examples/Content/NiagaraExamples ;;
	m5_fire) echo testdata/fab/m5_fire/Content/M5VFXVOL2/Niagara ;;
	fire_explosion) echo testdata/fab/fire_explosion/Content/Fire_EXP_Vol01_Free/Niagara ;;
	sparkles) echo testdata/fab/sparkles/Content/FreeParticle_SoftTofu/Niagara ;;
	loot_drops) echo testdata/fab/loot_drops/Content/LootDropVFX/Systems ;;
	esac
}

for pack in $packs; do
	dir=$(src "$pack")
	[ -d "$dir" ] || { echo "skip $pack: $dir missing (sh testdata/fab_assets.sh)"; continue; }
	n=0
	inputs="" labels=""
	# Systems in subfolders too (Mixed VFX, Mayu), in name order. M5 names
	# them NFire_*; its _for_BP / _for_Dir copies and Epic's sprite-baking
	# utilities are left out.
	for f in $(find "$dir" -name 'NS_*.uasset' -o -name 'NFire_*.uasset' | grep -v -e _for_ -e SpriteGeneration | sort); do
		s=$(basename "$f" .uasset)
		o=$out/$pack/$s
		mkdir -p "$o"
		"$bin/fximport" convert "$f" --out "$o" >"$o/convert.log" 2>&1 || true
		# An effect the GPU path refuses keeps fxpreview logging forever: give up after 5 min.
		# shellcheck disable=SC2046
		timeout 300 "$bin/fxpreview" "$o/$s.fx.ron" --out "$o" --size 640x360 \
			--times ${TIMES:-0.5,1,1.5,2} $(view "$pack") $(look "$pack") >"$o/render.log" 2>&1 ||
			echo "render failed: $s (see $o/render.log)"
		echo "HELM-PROGRESS $((n + 1)) $s"
		if [ -f "$o/$s.mp4" ]; then
			inputs="$inputs -i $o/$s.mp4"
			labels="$labels $s"
			n=$((n + 1))
		fi
	done
	[ "$n" -gt 0 ] || continue
	# Labelled grid, 4 columns (missing cells black).
	cols=4
	rows=$(((n + cols - 1) / cols))
	filter="" layout="" i=0
	for l in $labels; do
		filter="$filter[$i:v]drawtext=text='$l':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5,tpad=stop_mode=clone:stop_duration=10[v$i];"
		layout="$layout${layout:+|}$(((i % cols) * 640))_$(((i / cols) * 360))"
		i=$((i + 1))
	done
	stack=""
	j=0
	while [ $j -lt $n ]; do stack="$stack[v$j]"; j=$((j + 1)); done
	# shellcheck disable=SC2086
	ffmpeg -y -loglevel error $inputs -filter_complex \
		"$filter${stack}xstack=inputs=$n:layout=$layout:fill=black,pad=$((cols * 640)):$((rows * 360)):0:0:black,scale=iw/2:-2[out]" \
		-map "[out]" -t 3 -c:v libx264 -pix_fmt yuv420p -crf 20 "$out/${pack}_grid.mp4"
	# Mixed VFX hits, slashes and explosions are one-shot bursts over by ~1 s.
	case $pack in arrow_trail) grid_t=${GRID_T:-0.9} ;; mixed_vfx) grid_t=${GRID_T:-0.35} ;; *) grid_t=${GRID_T:-1.5} ;; esac
	ffmpeg -y -loglevel error -ss "$grid_t" -i "$out/${pack}_grid.mp4" -frames:v 1 "$out/${pack}_grid.png"
	echo "$out/${pack}_grid.mp4 ($n systems)"
	# The whole pack in one scene, laid out like the listing's showcase shot:
	# fires in three rows on the floor; arrows side by side, flying together.
	case $pack in
	perfect_fire)
		order="NS_Fire_Big NS_Fire_Big_Simple NS_Fire_Big_Smoke NS_Fire_Medium NS_Fire_Medium_Smoke NS_Fire_Simple NS_Fire_Small NS_Fire_Small_Simple NS_Fire_Small_Smoke NS_Fire_Floor_01 NS_Fire_Floor_01_Simple NS_Fire_Floor_01_Smoke NS_Fire_Floor_02 NS_Fire_Floor_02_Simple NS_Fire_Floor_02_Smoke"
		lview="--camera 0,7,11 --target 0,0,-1.5 --fov 59 --duration 3"
		pos() { echo "$(((($1 % 5) - 2) * 3)),0,$((2 - ($1 / 5) * 4))"; } ;;
	arrow_trail)
		# The listing's showcase shot (testdata/fab/_cmp/at_ref.png): the trails in
		# its order, nearest first, 1.5 m apart as in the demo map (BP_Demo
		# actors 150-160 cm apart), heads level; camera fitted to the shot's
		# trail heads and directions (vertical fov 38.6, yaw 48, pitch -35).
		order="NS_ArrowTrail_Basic_02 NS_ArrowTrail_Basic NS_ArrowTrail_Fire NS_ArrowTrail_Ice NS_ArrowTrail_Magic NS_ArrowTrail_Magic_2 NS_ArrowTrail_Holy NS_ArrowTrail_Water NS_ArrowTrail_Nature NS_ArrowTrail_Fire_02"
		lview="--camera -13.493,9.095,-11.783 --target -8.064,3.308,-5.697 --fov 38.6 --duration 2 --fly 20,0,0 --ground -3.5"
		pos() { echo "0,0,$(echo "$1" | awk '{print -6.75 + $1 * 1.5}')"; } ;;
	*) order="" ;;
	esac
	first="" also="" i=0
	for s in $order; do
		f=$out/$pack/$s/$s.fx.ron
		[ -f "$f" ] || continue
		if [ -z "$first" ]; then first="$f --at $(pos $i)"; else also="$also --also $f@$(pos $i)"; fi
		i=$((i + 1))
	done
	if [ -n "$first" ]; then
		# shellcheck disable=SC2086
		"$bin/fxpreview" $first $also --out "$out/${pack}_lineup" --size 1280x720 \
			--times 1,2 $lview $(look "$pack") >"$out/${pack}_lineup.log" 2>&1 ||
			echo "lineup render failed: see $out/${pack}_lineup.log"
	fi
done
