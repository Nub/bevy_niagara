#!/bin/sh
# One GPU job for the showcase packs (task_53567c1789): builds the release
# tools, converts and renders every system with fab_render.sh (UE look), then
# times each converted effect with `fx_bench scene show --file` (GPU ms per pass,
# 1080p, fx_bench's showcase camera) into testdata/fab/_look/bench.jsonl, one line per
# effect: {"pack", "result": fx_bench's SceneResult (max_particles, gpu_ms)}.
#
# Run from the repository root (claim graybeard heavy; ~40 min):
#   nix develop /home/zach/helm-projects/bevy-solari-opt -c sh testdata/fab_showcase.sh [pack ...]
# BENCH_ONLY=1 skips the build and renders.
set -eu
packs=${*:-loot_drops sparkles fire_explosion m5_fire niagara_examples}
out=${OUT:-testdata/fab/_look}
bin=target/release
if [ -z "${BENCH_ONLY:-}" ]; then
	cargo build --release -p fx_import --bin fximport -p fx_preview --bin fxpreview -p fx_bench
	# shellcheck disable=SC2086
	sh testdata/fab_render.sh $packs
fi
bench=$out/bench.jsonl
: >"$bench"
n=0
for pack in $packs; do
	for f in "$out/$pack"/*/*.fx.ron; do
		[ -f "$f" ] || continue
		n=$((n + 1))
		echo "HELM-PROGRESS $n bench $(basename "$f" .fx.ron)"
		# shellcheck disable=SC2046
		r=$(timeout 180 "$bin/fx_bench" scene show --file "$(realpath "$f")" --frames 120 --warm 120 2>/dev/null | sed -n 's/^FXBENCH_RESULT //p') || true
		[ -n "$r" ] && echo "{\"pack\":\"$pack\",\"result\":$r}" >>"$bench"
	done
done
echo "$bench ($n effects)"
