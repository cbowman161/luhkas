#!/usr/bin/env bash
set -euo pipefail

UNIT_DIR="${HOME}/.config/systemd/user"
PROJECT_DIR="${HOME}/scout_runtime"

mkdir -p "${UNIT_DIR}"
cp "${PROJECT_DIR}/systemd/scout-robot-api.service" "${UNIT_DIR}/"
cp "${PROJECT_DIR}/systemd/scout-vision.service" "${UNIT_DIR}/"
cp "${PROJECT_DIR}/systemd/scout-presence.service" "${UNIT_DIR}/"

systemctl --user daemon-reload
systemctl --user enable --now scout-robot-api.service
systemctl --user enable --now scout-vision.service
systemctl --user enable --now scout-presence.service

if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "${USER}" || true
fi

echo "Installed user services."
echo
echo "Robot API, vision, and presence proxy are enabled and started."
echo "User lingering was requested so services can start at boot before login."
echo
echo "Useful commands:"
echo "  systemctl --user status scout-robot-api.service"
echo "  systemctl --user status scout-vision.service"
echo "  systemctl --user status scout-presence.service"
echo "  journalctl --user -u scout-robot-api.service -f"
echo "  journalctl --user -u scout-vision.service -f"
echo "  journalctl --user -u scout-presence.service -f"
