#!/bin/sh
# Source manifest for testdata/fab/ (git-ignored, never committed, shipped or
# shared): Fab marketplace Niagara packs the user owns, used to test
# crates/fx_import locally. Both listings are flagged NoAI on Fab; the user
# chose to use them for local import testing only. Tests that read these
# files (crates/fx_import/tests/fab_files.rs) skip when they are absent.
#
# Item                  Seller            Fab listing                            Engine build    Licence
# perfect_fire          Vefects           d2136c01-8d35-403f-a637-146e8173bdb3   4.27 packages   Fab Standard
#   "Perfect Fire VFX": 15 systems (NS_Fire_*), 6 emitters (NE_*); the UE_5.8 build
#   still holds UE 4.27.2 saves (legacy file version -7).
# arrow_trail           CodePhase Games   b8ff3ab4-0e81-4335-bbf0-fea15f6fcdfc   5.8 packages    Fab Standard
#   "Free Arrow Trail": 10 systems (NS_ArrowTrail_*), ribbons + sprites.
#
# Needs: the user's Fab library to hold both items, and a fab-downloader
# session (the user signs in once; ~/.config/fab-downloader/session.bin,
# never print or copy it). fab-downloader: github.com/Fobal/fab-downloader
# @36e7dae (GPL-3.0, standard-library Python), cloned to testdata/tools/.
#
# Layout: testdata/fab/<item>/Content/... (as the pack ships; `fximport
# convert` finds /Game/... dependencies in the enclosing Content folder),
# <item>/MANIFEST.md, <item>/reference/ (listing.json + the listing's
# images and video, the Unreal look to compare against).
#
# Run from the repository root: sh testdata/fab_assets.sh
set -eu
root=$(pwd)/testdata
tool=$root/tools/fab-downloader
if [ ! -d "$tool" ]; then
	git clone https://github.com/Fobal/fab-downloader "$tool"
	git -C "$tool" checkout -q 36e7dae
fi

# item  library asset id  listing id
fetch() {
	item=$1 asset=$2 listing=$3
	out=$root/fab/$item
	if [ ! -d "$out/Content" ]; then
		mkdir -p "$root/fab"
		(cd "$tool" && nix shell nixpkgs#python3 -c python3 -m fabdl download "$asset" --engine 5.8 --out "$out" </dev/null)
	fi
	ref=$out/reference
	mkdir -p "$ref"
	curl -sfL -A "Mozilla/5.0" -H "Accept: application/json" "https://www.fab.com/i/listings/$listing" -o "$ref/listing.json"
	i=0
	for u in $(nix shell nixpkgs#jq -c jq -r '.medias[] | .mediaUrl' "$ref/listing.json"); do
		i=$((i + 1))
		f=$ref/media_$i.${u##*.}
		[ -f "$f" ] || curl -sfL -A "Mozilla/5.0" -o "$f" "$u"
	done
}

fetch perfect_fire 6a60a866c89e4583bfed4f2d40a0900f d2136c01-8d35-403f-a637-146e8173bdb3
fetch arrow_trail 477458f2e7a349e39d219eed300e2bd7 b8ff3ab4-0e81-4335-bbf0-fea15f6fcdfc
echo "testdata/fab ready"
