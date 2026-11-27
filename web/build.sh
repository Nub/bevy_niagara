#!/bin/sh
# Builds the browser demo into web/dist: wasm32 release build of fx_web,
# wasm-bindgen (web target), wasm-opt, index.html and the library effects.
# Toolchain from web/nix (Rust 1.100 + wasm32 target, wasm-bindgen-cli
# matching Cargo.lock, binaryen). Serve with `sh web/serve.sh` and open
# http://localhost:8080 in a WebGPU browser (HTTPS or localhost only).
#   sh web/build.sh [--no-opt]
set -eu
web=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$web")
dist="$web/dist"
opt=1
[ "${1:-}" = "--no-opt" ] && opt=0

if [ -z "${FX_WEB_SHELL:-}" ]; then
    exec env FX_WEB_SHELL=1 nix develop "path:$web/nix" -c sh "$0" "$@"
fi

cd "$root"
cargo build --profile web --target wasm32-unknown-unknown -p fx_web
wasm="$root/target/wasm32-unknown-unknown/web/fx_web.wasm"

rm -rf "$dist"
mkdir -p "$dist/assets/effects"
wasm-bindgen --target web --no-typescript --out-dir "$dist" --out-name fx_web "$wasm"
if [ "$opt" = 1 ]; then
    # Exactly the features rustc's wasm32 target emits (--all-features adds
    # proposals browsers reject, e.g. compact imports).
    wasm-opt -Oz --enable-bulk-memory --enable-mutable-globals --enable-nontrapping-float-to-int \
        --enable-sign-ext --enable-reference-types --enable-multivalue "$dist/fx_web_bg.wasm" -o "$dist/fx_web_bg.opt.wasm"
    mv "$dist/fx_web_bg.opt.wasm" "$dist/fx_web_bg.wasm"
fi
cp "$web/index.html" "$dist/"
# The demo's effects, and the library's textures/ and materials/ for effects that name them.
cp effects/library/campfire.fx.ron effects/library/sparks_burst.fx.ron "$dist/assets/effects/"
cp -r effects/library/textures effects/library/materials "$dist/assets/effects/"

size=$(wc -c < "$dist/fx_web_bg.wasm")
gz=$(gzip -9 -c "$dist/fx_web_bg.wasm" | wc -c)
total=$(du -sb "$dist" | cut -f1)
echo "web bundle: $dist ($total bytes in total)"
echo "fx_web_bg.wasm: $size bytes, $gz gzipped"
