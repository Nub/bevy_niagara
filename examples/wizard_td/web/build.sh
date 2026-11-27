#!/bin/sh
# Builds Wizard TD for the browser into examples/wizard_td/dist: wasm32
# build (profile `web`), wasm-bindgen, wasm-opt, the page and the assets.
# Toolchain from web/nix (as web/build.sh). Serve the folder over HTTPS or
# localhost in a WebGPU browser, e.g. `sh examples/wizard_td/web/serve.sh`.
#   sh examples/wizard_td/web/build.sh [--no-opt] [--out DIR]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
crate=$(dirname "$here")
root=$(cd "$crate/../.." && pwd)
if [ -z "${FX_WEB_SHELL:-}" ]; then
    exec env FX_WEB_SHELL=1 nix develop "path:$root/web/nix" -c sh "$0" "$@"
fi

dist="$crate/dist"
opt=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-opt) opt=0 ;;
        --out) shift; dist="$1" ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$root"
cargo build --profile web --target wasm32-unknown-unknown -p wizard_td
wasm="$root/target/wasm32-unknown-unknown/web/wizard_td.wasm"

rm -rf "$dist"
mkdir -p "$dist"
wasm-bindgen --target web --no-typescript --out-dir "$dist" --out-name wizard_td "$wasm"
if [ "$opt" = 1 ]; then
    wasm-opt -Oz --enable-bulk-memory --enable-mutable-globals --enable-nontrapping-float-to-int \
        --enable-sign-ext --enable-reference-types --enable-multivalue "$dist/wizard_td_bg.wasm" -o "$dist/wizard_td_bg.opt.wasm"
    mv "$dist/wizard_td_bg.opt.wasm" "$dist/wizard_td_bg.wasm"
fi
cp "$here/index.html" "$dist/"
cp -r "$crate/assets" "$dist/assets"
size=$(wc -c < "$dist/wizard_td_bg.wasm")
gz=$(gzip -9 -c "$dist/wizard_td_bg.wasm" | wc -c)
echo "wizard_td bundle: $dist ($(du -sb "$dist" | cut -f1) bytes); wasm $size bytes, $gz gzipped"
