#!/usr/bin/env bash
# Launch the pantilt_node service. Subscribes to vision_service /meta and
# drives the pan/tilt servos via robot_api. Only run on nodes with the
# pantilt_node module.
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
exec python3 pantilt_node/service.py "$@"
