#!/usr/bin/env bash
# Install everything audio_node needs:
#   * alsa-utils (arecord/aplay), espeak-ng fallback TTS)
#   * an audio_node-owned Python venv with Vosk STT and Piper TTS
#   * dtoverlay for the audio HAT (RaspAudio Mic Ultra 3 uses WM8960)
#   * Vosk English STT model
#
# Idempotent. Run as root; orchestrator/firstboot supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../scripts/lib_install.sh"

: "${NODE_USER:?NODE_USER is required}"
USER_HOME="$(getent passwd "$NODE_USER" | cut -d: -f6)"
[ -n "$USER_HOME" ] || { echo "ERROR: user $NODE_USER not found"; exit 2; }
BOOT_CONFIG="/boot/firmware/config.txt"
AUDIO_OVERLAY="${AUDIO_OVERLAY:-wm8960-soundcard}"
AUDIO_VENV="${AUDIO_VENV:-${USER_HOME}/.local/share/luhkas/audio_node/venv}"
VOSK_MODEL_URL="${VOSK_MODEL_URL:-https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip}"
VOSK_MODEL_NAME="${VOSK_MODEL_NAME:-vosk-model-en-us-0.22-lgraph}"
VOSK_MODEL_DEST="${VOSK_MODEL_DEST:-${USER_HOME}/.local/share/luhkas/audio_node/${VOSK_MODEL_NAME}}"
PIPER_VOICE_URL="${PIPER_VOICE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx}"
PIPER_VOICE_CONFIG_URL="${PIPER_VOICE_CONFIG_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json}"
PIPER_VOICE_DEST="${PIPER_VOICE_DEST:-${USER_HOME}/.local/share/luhkas/audio_node/piper/en_US-joe-medium.onnx}"

echo "[audio_node/install] starting (user=${NODE_USER}, overlay=${AUDIO_OVERLAY})"

ensure_apt_updated
apt_install \
  alsa-utils \
  espeak-ng \
  python3-venv \
  python3-pip \
  unzip \
  curl

echo "[audio_node/install] ensuring Python venv at ${AUDIO_VENV}"
sudo -u "$NODE_USER" -H mkdir -p "$(dirname "$AUDIO_VENV")"
if [ ! -x "${AUDIO_VENV}/bin/python" ]; then
  sudo -u "$NODE_USER" -H python3 -m venv "$AUDIO_VENV"
fi

sudo -u "$NODE_USER" -H "${AUDIO_VENV}/bin/python" -m pip install --upgrade pip wheel
sudo -u "$NODE_USER" -H "${AUDIO_VENV}/bin/python" -m pip install --upgrade vosk piper-tts gpiod

if [ -n "$PIPER_VOICE_URL" ] && [ ! -f "$PIPER_VOICE_DEST" ]; then
  echo "[audio_node/install] downloading Piper voice to ${PIPER_VOICE_DEST}"
  sudo -u "$NODE_USER" -H mkdir -p "$(dirname "$PIPER_VOICE_DEST")"
  sudo -u "$NODE_USER" -H curl -fL "$PIPER_VOICE_URL" -o "$PIPER_VOICE_DEST" || \
    echo "[audio_node/install] WARN: Piper voice download failed"
fi
if [ -n "$PIPER_VOICE_CONFIG_URL" ] && [ ! -f "${PIPER_VOICE_DEST}.json" ]; then
  sudo -u "$NODE_USER" -H curl -fL "$PIPER_VOICE_CONFIG_URL" -o "${PIPER_VOICE_DEST}.json" || \
    echo "[audio_node/install] WARN: Piper voice config download failed"
fi

if [ -f "$BOOT_CONFIG" ]; then
  if ! grep -qE "^dtoverlay=${AUDIO_OVERLAY}\b" "$BOOT_CONFIG"; then
    echo "[audio_node/install] adding 'dtoverlay=${AUDIO_OVERLAY}' to ${BOOT_CONFIG}"
    printf '\n# LUHKAS audio_node: HAT overlay\ndtoverlay=%s\n' "$AUDIO_OVERLAY" >> "$BOOT_CONFIG"
  fi
fi

# Vosk English model. Create the tmp dir AS the node user so the subsequent
# curl/unzip (also as that user) can write to it. mktemp -d run by root
# produces a 700 dir owned by root.
if [ ! -d "$VOSK_MODEL_DEST" ]; then
  echo "[audio_node/install] downloading Vosk model ${VOSK_MODEL_NAME}"
  TMP=$(sudo -u "$NODE_USER" -H mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  if sudo -u "$NODE_USER" -H curl -fL "$VOSK_MODEL_URL" -o "${TMP}/vosk.zip"; then
    sudo -u "$NODE_USER" -H unzip -q "${TMP}/vosk.zip" -d "$TMP"
    sudo -u "$NODE_USER" -H mv "${TMP}/${VOSK_MODEL_NAME}" "${VOSK_MODEL_DEST}"
    echo "[audio_node/install] vosk model installed at ${VOSK_MODEL_DEST}"
  else
    echo "[audio_node/install] WARN: vosk model download failed; STT will be disabled until ${VOSK_MODEL_DEST} exists"
  fi
fi

sudo -u "$NODE_USER" -H "${AUDIO_VENV}/bin/python" - <<'PY'
import importlib.util
import pathlib

checks = ["vosk", "piper"]
missing = [name for name in checks if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python packages after install: {missing}")
print("audio_node Python packages OK:", pathlib.Path(__import__("sys").executable))
PY

echo "[audio_node/install] done"
