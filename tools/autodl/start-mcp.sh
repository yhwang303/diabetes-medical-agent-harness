#!/bin/bash
set -euo pipefail
umask 077
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUTODL_REMOTE_MCP_CLI="$task_dir/autodl-remote"
export AUTODL_REMOTE_MCP_MAX_OUTPUT_BYTES=32768
exec /usr/local/bin/node "$task_dir/vendor/AutoDL-Remote/plugins/autodl-remote/mcp/server.mjs"
