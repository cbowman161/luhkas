#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/scout_runtime"
cd "${HOME}/hailo-apps"
set +u
source setup_env.sh
set -u
cd "${HOME}/scout_runtime"
python3 services/robot_api.py
