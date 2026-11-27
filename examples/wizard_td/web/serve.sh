#!/bin/sh
# Serves examples/wizard_td/dist on http://localhost:${PORT:-8080}.
set -eu
dir=$(cd "$(dirname "$0")/.." && pwd)/dist
exec nix shell nixpkgs#static-web-server -c static-web-server --port "${PORT:-8080}" --root "$dir" --host 0.0.0.0 --compression true
