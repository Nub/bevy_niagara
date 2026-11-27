#!/bin/sh
# Publishes the browser demo to GitHub Pages: builds web/dist with
# web/build.sh (locally: CI cannot reach the private Bevy fork), then pushes
# it as a single orphan commit to the `gh-pages` branch of origin, replacing
# what was there. master's history is not touched. The page is served at
# https://<owner>.github.io/<repo>/; every path in it is relative, so the
# subpath needs no configuration.
#   sh web/deploy.sh [--no-build]
set -eu
web=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$web")
dist="$web/dist"
remote=${FX_DEPLOY_REMOTE:-origin}

[ "${1:-}" = "--no-build" ] || sh "$web/build.sh"
[ -f "$dist/index.html" ] && [ -f "$dist/fx_web_bg.wasm" ] || { echo "no bundle in $dist" >&2; exit 1; }

# Pages runs Jekyll unless told not to.
touch "$dist/.nojekyll"
rev=$(git -C "$root" rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q -b gh-pages
cp -r "$dist/." "$tmp/"
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git -C "$root" config user.name)" -c user.email="$(git -C "$root" config user.email)" \
    commit -q -m "Web demo built from $rev"
url=$(git -C "$root" remote get-url "$remote")
git -C "$tmp" push -f "$url" gh-pages:gh-pages
echo "pushed $(du -sh "$dist" | cut -f1) to gh-pages of $url (built from $rev)"
