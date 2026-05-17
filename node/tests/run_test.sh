#!/usr/bin/env bash
# Run the chair-tracking calibration test on port 5010
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${HOME}/hailo-apps"
source setup_env.sh

cd "${SCRIPT_DIR}"
echo "[TEST] Starting chair-tracking calibration service on port 5010..." >&2
exec python3 vision_service_test.py \
    --port 5010 \
    --robot-api-url http://127.0.0.1:5001 \
    --no-face-detection \
    --no-face-recognition \
    "$@"
