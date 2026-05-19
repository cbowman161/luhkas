#!/usr/bin/env bash
set -euo pipefail

UNIT_DIR="${HOME}/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"
NODE_ID="${LUHKAS_NODE_ID:-scout}"
VAULT_URL="${VAULT_CHAT_URL:-http://luhkas-vault.local:7000}"

mkdir -p "${UNIT_DIR}"

for svc in scout-robot-api.service scout-vision.service scout-presence.service scout-controller.service; do
  sed \
    -e "s|{NODE_DIR}|${NODE_DIR}|g" \
    -e "s|{NODE_ID}|${NODE_ID}|g" \
    -e "s|{VAULT_URL}|${VAULT_URL}|g" \
    "${NODE_DIR}/systemd/${svc}" > "${UNIT_DIR}/${svc}"
done

systemctl --user daemon-reload
systemctl --user enable --now scout-robot-api.service
systemctl --user enable --now scout-vision.service
systemctl --user enable --now scout-presence.service
systemctl --user enable scout-controller.service >/dev/null 2>&1 || true

if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "${USER}" || true
fi

echo "Installed user services."
echo "Node directory: ${NODE_DIR}"
echo "Node id: ${NODE_ID}"
echo "Vault URL: ${VAULT_URL}"
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
