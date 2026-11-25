#!/bin/sh
# Starts the fx MCP server (stdio) for Claude Code; see .mcp.json.
# Runs inside the Bevy fork's nix dev shell so cargo, ffmpeg and the Vulkan
# loader are available to the server and to the fxpreview/fxviewer binaries
# it builds on first use. Build output goes to stderr: stdout is the protocol.
root="$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
cd "$root" || exit 1
exec nix develop /home/zach/helm-projects/bevy-solari-opt --command sh -c \
  'cargo build --quiet --release -p fx_mcp 1>&2 && exec target/release/fx_mcp'
