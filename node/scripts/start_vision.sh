#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"

cd "${HOME}/hailo-apps"
set +u
source setup_env.sh
set -u
cd "$NODE_DIR"
python3 services/vision_service.py "$@"
