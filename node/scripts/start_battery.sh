#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$NODE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$NODE_DIR/.env"
  set +a
fi

# The uart_proxy backend reads from ${XDG_RUNTIME_DIR}/luhkas-battery.json
# (or /tmp fallback). systemd creates the per-user runtime dir automatically;
# no setup needed.

cd "$NODE_DIR"
exec python3 battery_node/service.py
