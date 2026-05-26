#!/usr/bin/env bash
# Install everything display_node needs:
#   * chromium-browser (kiosk launcher)
#   * unclutter (hides idle cursor)
#   * KMS display driver overlay (Pi 5 default; idempotent)
#   * desktop autologin (so chromium-browser unit can start under the
#     user's graphical-session.target)
#
# Idempotent. Run as root; firstboot supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../scripts/lib_install.sh"

: "${NODE_USER:?NODE_USER is required}"
BOOT_CONFIG="/boot/firmware/config.txt"

echo "[display_node/install] starting (user=${NODE_USER})"

ensure_apt_updated
apt_install chromium-browser unclutter

# ── Lite vs Desktop detection ────────────────────────────────────────────────
# kiosk-browser.service requires graphical-session.target, which is only
# active on Desktop images. On Lite, install the minimum X stack so the
# graphical session can come up.
NEEDS_X_STACK=0
if ! command -v startx >/dev/null 2>&1 && ! dpkg -l raspberrypi-ui-mods 2>/dev/null | grep -q '^ii'; then
  NEEDS_X_STACK=1
  echo "[display_node/install] Pi OS Lite detected — installing minimum X stack for chromium kiosk"
  apt_install \
    xserver-xorg-core \
    xserver-xorg-input-libinput \
    xserver-xorg-video-fbdev \
    xserver-xorg-video-modesetting \
    xinit \
    openbox \
    lightdm \
    libgles2 \
    libgl1
fi

# KMS overlay — Pi OS Desktop has this enabled by default; Lite images don't.
if [ -f "$BOOT_CONFIG" ]; then
  if ! grep -qE "^dtoverlay=vc4-kms-v3d\b" "$BOOT_CONFIG"; then
    echo "[display_node/install] adding 'dtoverlay=vc4-kms-v3d' to ${BOOT_CONFIG}"
    printf '\n# LUHKAS display_node: KMS display driver\ndtoverlay=vc4-kms-v3d\n' >> "$BOOT_CONFIG"
  fi
fi

# Desktop autologin — required so the kiosk-browser user service has an X
# session to attach to. raspi-config exposes this as B4 (Desktop autologin).
if command -v raspi-config >/dev/null 2>&1; then
  raspi-config nonint do_boot_behaviour B4 || true
fi

# On Lite we also need lightdm autologin pointed at $NODE_USER so X starts
# without a manual login.
if [ "$NEEDS_X_STACK" = "1" ] && [ -d /etc/lightdm ]; then
  install -d -m 0755 /etc/lightdm/lightdm.conf.d
  cat > /etc/lightdm/lightdm.conf.d/90-luhkas-autologin.conf <<EOF
[Seat:*]
autologin-user=${NODE_USER}
autologin-session=openbox
user-session=openbox
EOF
  systemctl enable lightdm.service 2>/dev/null || true
fi

echo "[display_node/install] done"
