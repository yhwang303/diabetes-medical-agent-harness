#!/bin/zsh
set -euo pipefail
PROJECT_DIR="${0:A:h:h}"
export CARGO_HOME="$PROJECT_DIR/.cache/cargo"
export RUSTUP_HOME="$PROJECT_DIR/.cache/rustup"
export CARGO_TARGET_DIR="$PROJECT_DIR/app/src-tauri/target"
export npm_config_cache="$PROJECT_DIR/.cache/npm"
export PATH="$CARGO_HOME/bin:$PATH"
cd "$PROJECT_DIR/app"
case "${1:-build}" in
  build) npm run tauri -- build --debug --bundles app ;;
  open) open "$CARGO_TARGET_DIR/debug/bundle/macos/Medical Harness.app" ;;
  *) print -u2 'Usage: desktop.sh [build|open]'; exit 2 ;;
esac
