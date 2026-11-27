#!/bin/sh
# Publishes Wizard TD to GitHub Pages under wizard_td/ of the gh-pages
# branch (https://<owner>.github.io/<repo>/wizard_td/), next to the web demo
# that web/deploy.sh puts at the root: fetches gh-pages, replaces only the
# wizard_td/ folder, pushes one commit on top. Builds first unless --no-build.
#   sh examples/wizard_td/web/deploy.sh [--no-build]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
crate=$(dirname "$here")
root=$(cd "$crate/../.." && pwd)
dist="$crate/dist"
remote=${FX_DEPLOY_REMOTE:-origin}

[ "${1:-}" = "--no-build" ] || sh "$here/build.sh"
[ -f "$dist/index.html" ] && [ -f "$dist/wizard_td_bg.wasm" ] || { echo "no bundle in $dist" >&2; exit 1; }

url=$(git -C "$root" remote get-url "$remote")
rev=$(git -C "$root" rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if git clone -q --depth 1 --branch gh-pages "$url" "$tmp/site" 2>/dev/null; then :; else
    git -C "$tmp" init -q -b gh-pages site
fi
rm -rf "$tmp/site/wizard_td"
mkdir -p "$tmp/site/wizard_td"
cp -r "$dist/." "$tmp/site/wizard_td/"
touch "$tmp/site/.nojekyll"
git -C "$tmp/site" add -A
git -C "$tmp/site" -c user.name="$(git -C "$root" config user.name)" -c user.email="$(git -C "$root" config user.email)" \
    commit -q -m "Wizard TD built from $rev"
git -C "$tmp/site" push "$url" HEAD:gh-pages
echo "pushed wizard_td/ ($(du -sh "$dist" | cut -f1)) to gh-pages of $url (built from $rev)"
