#!/usr/bin/env bash
# Install what rover_node needs (also covers pantilt_node + light_node,
# which share the same UART under robot_api):
#   * UART enabled in /boot/firmware/config.txt
#   * Serial console DISABLED on UART (otherwise robot_api can't talk to
#     the rover controller)
#   * pyserial + inputs are in node/requirements.txt
#
# Idempotent. Run as root; firstboot supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../scripts/lib_install.sh"

BOOT_CONFIG="/boot/firmware/config.txt"
CMDLINE="/boot/firmware/cmdline.txt"

echo "[rover_node/install] starting"

# Enable UART hardware.
if [ -f "$BOOT_CONFIG" ]; then
  for line in "dtparam=uart0=on" "enable_uart=1"; do
    if ! grep -qE "^${line}\b" "$BOOT_CONFIG"; then
      echo "[rover_node/install] adding '${line}' to ${BOOT_CONFIG}"
      printf '\n# LUHKAS rover_node: serial to rover controller\n%s\n' "$line" >> "$BOOT_CONFIG"
    fi
  done
fi

# Disable serial console on UART so robot_api owns the line.
if [ -f "$CMDLINE" ]; then
  if grep -q "console=serial0" "$CMDLINE"; then
    echo "[rover_node/install] removing serial console from ${CMDLINE}"
    sed -i 's|console=serial0,[0-9]*\s*||g' "$CMDLINE"
  fi
fi

# Disable serial-getty so it doesn't grab the port.
systemctl disable --now serial-getty@ttyAMA0.service 2>/dev/null || true
systemctl disable --now serial-getty@ttyS0.service 2>/dev/null || true

echo "[rover_node/install] done"
