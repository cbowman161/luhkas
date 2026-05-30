#!/usr/bin/env bash
# Launcher for the physical display service.
# luhkas_node owns /ui and /chat; display_node owns what the kiosk screen shows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$NODE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$NODE_DIR/.env"
  set +a
fi

cd "$NODE_DIR"
exec python3 display_node/service.py
