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

cd "$NODE_DIR"

if [ "${AUDIO_AUTO_CONFIGURE_HAT:-1}" != "0" ] \
  && [ -r /proc/asound/cards ] \
  && grep -Eq "wm8960soundcard|wm8960-soundcard" /proc/asound/cards; then
  "$NODE_DIR/audio_node/configure_raspiaudio_mic_ultra_3.sh"
  export AUDIO_INPUT_DEVICE="${AUDIO_INPUT_DEVICE:-plughw:CARD=wm8960soundcard,DEV=0}"
  export AUDIO_OUTPUT_DEVICE="${AUDIO_OUTPUT_DEVICE:-plughw:CARD=wm8960soundcard,DEV=0}"
fi

if [ -z "${AUDIO_VOSK_MODEL:-}" ] \
  && [ -d "$HOME/.local/share/luhkas/audio_node/vosk-model-en-us-0.22-lgraph" ]; then
  export AUDIO_VOSK_MODEL="$HOME/.local/share/luhkas/audio_node/vosk-model-en-us-0.22-lgraph"
fi

AUDIO_PYTHON="${AUDIO_PYTHON:-$HOME/.local/share/luhkas/audio_node/venv/bin/python}"
if [ ! -x "$AUDIO_PYTHON" ]; then
  AUDIO_PYTHON="python3"
fi

exec "$AUDIO_PYTHON" audio_node/service.py
