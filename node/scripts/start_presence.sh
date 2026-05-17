#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/scout_runtime"
python3 services/presence_client_service.py "$@"
