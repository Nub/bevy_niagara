#!/bin/sh
# Serves web/dist on http://localhost:${PORT:-8080} (WebGPU needs localhost or HTTPS).
set -eu
dir=$(cd "$(dirname "$0")" && pwd)/dist
exec nix shell nixpkgs#static-web-server -c static-web-server --port "${PORT:-8080}" --root "$dir" --host 0.0.0.0 --compression true
