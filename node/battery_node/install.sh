#!/usr/bin/env bash
# Install everything battery_node needs:
#   * i2c-tools (for i2cdetect / debugging the UPS HAT)
#   * I2C bus enabled in /boot/firmware/config.txt (so MAX17040 / INA219
#     are addressable by the auto-detect backend)
#   * smbus2 is in node/requirements.txt — installed by bootstrap_node.sh
#
# Idempotent. Run as root; firstboot supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/../scripts/lib_install.sh"

BOOT_CONFIG="/boot/firmware/config.txt"

echo "[battery_node/install] starting"

ensure_apt_updated
apt_install i2c-tools python3-smbus2

if [ -f "$BOOT_CONFIG" ]; then
  if ! grep -qE "^dtparam=i2c_arm=on" "$BOOT_CONFIG"; then
    echo "[battery_node/install] enabling I2C in ${BOOT_CONFIG}"
    printf '\n# LUHKAS battery_node: I2C for UPS HATs (MAX17040 / INA219)\ndtparam=i2c_arm=on\n' >> "$BOOT_CONFIG"
  fi
fi

echo "[battery_node/install] done"
