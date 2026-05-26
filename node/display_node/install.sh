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
CMDLINE="/boot/firmware/cmdline.txt"
# Optional: one of normal | right | inverted | left. Unset => no rotation.
DISPLAY_ROTATION="${DISPLAY_ROTATION:-}"

echo "[display_node/install] starting (user=${NODE_USER})"

# Pi OS Trixie packages chromium as ``chromium`` (no ``-browser`` suffix).
# Older releases used chromium-browser. ``start_kiosk_browser.sh`` finds
# whichever binary is installed.
ensure_apt_updated
apt_install chromium unclutter

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

# ── optional: rotate the active HDMI output ──────────────────────────────────
# KMS-level rotation via the kernel command line so it applies to the
# console AND the X/Wayland desktop, and survives every boot.
if [ -n "$DISPLAY_ROTATION" ]; then
  case "$DISPLAY_ROTATION" in
    normal)   ROTATE_DEG=0  ;;
    right)    ROTATE_DEG=90 ;;
    inverted) ROTATE_DEG=180 ;;
    left)     ROTATE_DEG=270 ;;
    *)
      echo "[display_node/install] WARN: ignoring unknown DISPLAY_ROTATION='${DISPLAY_ROTATION}'"
      ROTATE_DEG=""
      ;;
  esac

  if [ -n "$ROTATE_DEG" ] && [ -f "$CMDLINE" ]; then
    # Pick the first connected HDMI port reported by DRM (HDMI-A-1, HDMI-A-2).
    HDMI_PORT=""
    for s in /sys/class/drm/card*-HDMI*/status; do
      if [ -f "$s" ] && [ "$(cat "$s")" = "connected" ]; then
        HDMI_PORT="$(basename "$(dirname "$s")" | sed 's/^card[0-9]*-//')"
        break
      fi
    done
    HDMI_PORT="${HDMI_PORT:-HDMI-A-1}"

    # Kernel-level rotation hint for the console + X11 fallback. Wayland
    # compositors (labwc on Pi OS Trixie Desktop, wayfire on Bookworm)
    # ignore this and manage their own transforms — see labwc autostart
    # below for the Wayland-side rotation.
    VIDEO_ARG="video=${HDMI_PORT}:rotate=${ROTATE_DEG}"
    sed -i -E "s| ?video=HDMI-A-[0-9]+:[^ ]*||g" "$CMDLINE"
    sed -i -E "s|$| ${VIDEO_ARG}|" "$CMDLINE"
    tr -d '\n' < "$CMDLINE" > "${CMDLINE}.tmp" && mv "${CMDLINE}.tmp" "$CMDLINE"
    printf '\n' >> "$CMDLINE"
    echo "[display_node/install] applied console rotation: ${DISPLAY_ROTATION} (${VIDEO_ARG})"

    # Wayland-side rotation via kanshi. Kanshi is the Wayland output
    # config daemon Pi OS Trixie's labwc-pi launches by default; it reads
    # ~/.config/kanshi/config and applies output transforms on every
    # session start AND on hotplug. This is the canonical persistent
    # mechanism — using labwc autostart instead would race with kanshi.
    USER_HOME="$(getent passwd "$NODE_USER" | cut -d: -f6)"
    KANSHI_DIR="${USER_HOME}/.config/kanshi"
    KANSHI_CONF="${KANSHI_DIR}/config"
    install -d -m 0755 -o "$NODE_USER" -g "$NODE_USER" "$KANSHI_DIR"
    cat > "$KANSHI_CONF" <<EOF
profile {
    output ${HDMI_PORT} transform ${ROTATE_DEG} enable
}
EOF
    chown "$NODE_USER:$NODE_USER" "$KANSHI_CONF"
    chmod 0644 "$KANSHI_CONF"
    # Remove any obsolete labwc autostart from earlier installs so it
    # doesn't fight kanshi.
    rm -f "${USER_HOME}/.config/labwc/autostart"
    echo "[display_node/install] wrote kanshi rotation: ${HDMI_PORT} transform ${ROTATE_DEG}"
  fi
fi

echo "[display_node/install] done"
