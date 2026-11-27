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
# Showcase packs (task_53567c1789; each <item>/MANIFEST.md has the details):
# niagara_examples (Epic, Standard, NoAI, 5.7), m5_fire (M5VFX, Standard, NoAI,
# 5.7, systems named NFire_*), fire_explosion (DeepBlues, Standard, NoAI, 5.7),
# sparkles (SoftTofuVFX, CC BY 4.0), loot_drops (Targi VFX, Standard, 5.8).
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

# item  library asset id  listing id  [engine, default 5.8]
fetch() {
	item=$1 asset=$2 listing=$3 engine=${4:-5.8}
	out=$root/fab/$item
	if [ ! -d "$out/Content" ]; then
		mkdir -p "$root/fab"
		(cd "$tool" && nix shell nixpkgs#python3 -c python3 -m fabdl download "$asset" --engine "$engine" --out "$out" </dev/null)
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
fetch lightning 3ae510e26b08480ba417b1433973818c 90f37b0c-f672-4750-869a-5fa4452d5676
fetch portals 3962952c2e944ef6b3475b3d89dbd4ba aeb9fe88-41d6-4ba7-947d-0bfa5c1e74e6
fetch mixed_vfx f2fc93aa275241bf93c2b9fb66453962 6518971c-c8ca-47d7-8611-1ff068b78dd5 5.4
fetch mayu_basic 091bf691081042308370fcaa9fb4e9a0 35636320-5ff4-42cc-a233-34ee16a4a7a4 5.7
fetch niagara_examples 373b1bf893914d468128839ea8eeb954 0e188eca-4e54-4fb2-a9ed-d8b8a565e600 5.7
fetch m5_fire 4d66bc76bff54641a73cb8576b48fe69 c5b0270a-a295-4644-a4be-42cb1e56a197 5.7
fetch fire_explosion c39771778e274c1fa4cbd34f95def578 2adf4e4e-5f1d-4543-9e8c-192fe3c9a8ed 5.7
fetch sparkles e79ac126f89d4e2bbd518692f1617a3e 183732bc-c2fb-465c-9453-f70a1ce7ba2c
fetch loot_drops f7b599308b6541b1926ed08d0681c0fc 9036618c-f389-4551-9494-93b4bad2d1a3
echo "testdata/fab ready"
