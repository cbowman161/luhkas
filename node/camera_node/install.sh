#!/usr/bin/env bash
# Install everything camera_node needs on a fresh Pi 5 + AI HAT+:
#   * Hailo-10 runtime, PCIe driver, models, Python bindings, TAPPAS
#   * libcamera + picamera2 + rpicam-apps + OpenCV
#   * PCIe gen3 dtparam (required by the AI HAT+ at full bandwidth)
#   * hailo-ai/hailo-apps repo cloned to ~$NODE_USER/hailo-apps with its
#     own venv (sourced by start_vision.sh / start_robot_api.sh)
#
# Idempotent — safe to re-run. Run as root; firstboot supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../scripts/lib_install.sh"

: "${NODE_USER:?NODE_USER is required}"
USER_HOME="$(getent passwd "$NODE_USER" | cut -d: -f6)"
[ -n "$USER_HOME" ] || { echo "ERROR: user $NODE_USER not found"; exit 2; }
BOOT_CONFIG="/boot/firmware/config.txt"
HAILO_APPS_REPO="${HAILO_APPS_REPO:-https://github.com/hailo-ai/hailo-apps.git}"
HAILO_APPS_DIR="${USER_HOME}/hailo-apps"

echo "[camera_node/install] starting (user=${NODE_USER})"
echo "[camera_node/install] Hailo SDK + hailo-apps build can take 5–10 minutes"

# ── apt packages ─────────────────────────────────────────────────────────────
ensure_apt_updated
apt_install \
  hailo-h10-all \
  hailo-models \
  python3-h10-hailort \
  python3-hailo-tappas \
  rpicam-apps-hailo-postprocess \
  rpicam-apps \
  rpicam-apps-core \
  libcamera0.7 \
  libcamera-ipa \
  libcamera-tools \
  libcamera-v4l2 \
  python3-libcamera \
  python3-picamera2 \
  python3-opencv \
  python3-gi \
  python3-numpy

# ── PCIe gen3 (AI HAT+ needs this for full bandwidth) ────────────────────────
# These dtparam values take effect on the NEXT reboot. hailo-apps/install.sh
# below builds a venv and installs Python bindings — it does not need the
# chip live, so we don't need to reboot mid-firstboot.
if [ -f "$BOOT_CONFIG" ]; then
  for line in "dtparam=pciex1" "dtparam=pciex1_gen=3"; do
    if ! grep -qE "^${line}\b" "$BOOT_CONFIG"; then
      echo "[camera_node/install] adding '${line}' to ${BOOT_CONFIG} (takes effect after reboot)"
      printf '\n# LUHKAS camera_node: AI HAT+ PCIe link\n%s\n' "$line" >> "$BOOT_CONFIG"
    fi
  done
fi

# ── hailo-apps repo + venv (the vision service sources its setup_env.sh) ────
if [ ! -d "$HAILO_APPS_DIR/.git" ]; then
  echo "[camera_node/install] cloning ${HAILO_APPS_REPO} -> ${HAILO_APPS_DIR}"
  sudo -u "$NODE_USER" -H git clone --recurse-submodules "$HAILO_APPS_REPO" "$HAILO_APPS_DIR"
else
  echo "[camera_node/install] hailo-apps already cloned; pulling latest"
  sudo -u "$NODE_USER" -H git -C "$HAILO_APPS_DIR" pull --ff-only || true
  sudo -u "$NODE_USER" -H git -C "$HAILO_APPS_DIR" submodule update --init --recursive || true
fi

if [ -x "$HAILO_APPS_DIR/install.sh" ] && [ ! -d "$HAILO_APPS_DIR/venv_hailo_apps" ]; then
  echo "[camera_node/install] running hailo-apps/install.sh (creates venv_hailo_apps)"
  # hailo-apps install.sh REQUIRES root (it apt-installs, drops files into
  # /usr/local/hailo, etc.) and uses SUDO_USER to discover the target user.
  # We're already running as root in this script (orchestrator invoked us
  # with sudo), so set SUDO_USER explicitly and invoke directly. stdin from
  # /dev/null prevents any unexpected prompt from hanging the install.
  (
    cd "$HAILO_APPS_DIR"
    SUDO_USER="$NODE_USER" ./install.sh </dev/null
  ) || echo "[camera_node/install] WARN: hailo-apps/install.sh exited non-zero; check ${HAILO_APPS_DIR}/logs/"
fi

echo "[camera_node/install] done"
