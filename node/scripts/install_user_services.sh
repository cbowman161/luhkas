#!/usr/bin/env bash
set -euo pipefail

UNIT_DIR="${HOME}/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"
BOOTSTRAP_ENV="${LUHKAS_BOOTSTRAP_ENV:-$HOME/.config/luhkas/bootstrap.env}"
if [ -f "${BOOTSTRAP_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${BOOTSTRAP_ENV}"
  set +a
fi

: "${LUHKAS_NODE_ID:?LUHKAS_NODE_ID is required (set it in ~/.config/luhkas/bootstrap.env or via env)}"
: "${VAULT_CHAT_URL:?VAULT_CHAT_URL is required}"
NODE_ID="${LUHKAS_NODE_ID}"
VAULT_URL="${VAULT_CHAT_URL}"
LUHKAS_TAILSCALE="${LUHKAS_TAILSCALE:-1}"

mkdir -p "${UNIT_DIR}"

# ── make user systemd usable ───────────────────────────────────────────────
# When this script is invoked under ``sudo -u luhkas`` from firstboot, the
# environment lacks the bits that ``systemctl --user`` needs (DBUS socket,
# XDG_RUNTIME_DIR). Enable linger FIRST so the user manager starts now and
# also at every boot, then make sure the right env vars are exported.
USER_UID="$(id -u)"
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "${USER}" 2>/dev/null || true
  # Give the user manager a couple of seconds to come up.
  for i in 1 2 3 4 5; do
    if [ -S "/run/user/${USER_UID}/bus" ]; then break; fi
    sleep 1
  done
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${USER_UID}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

if [ "${LUHKAS_TAILSCALE}" = "1" ]; then
  LUHKAS_NODE_ID="${NODE_ID}" "${NODE_DIR}/scripts/setup_tailscale.sh" || {
    echo "WARNING: Tailscale setup failed. Re-run:"
    echo "  LUHKAS_NODE_ID=${NODE_ID} ${NODE_DIR}/scripts/setup_tailscale.sh"
  }
fi

# Remove any LUHKAS-rendered unit files that DON'T belong to this node.
# A unit is recognizable as LUHKAS-rendered by its ExecStart path pointing
# at a script in ${NODE_DIR}/scripts/start_*.sh. Anything matching that but
# not prefixed with "${NODE_ID}-" is a stale rendering from a previous run
# with a different NODE_ID; disable and remove it so the cleanup is
# idempotent across reconfigurations.
PREFIX="${NODE_ID}-"
shopt -s nullglob
for unit in "${UNIT_DIR}"/*.service; do
  name="$(basename "${unit}")"
  if grep -qE "^ExecStart=${NODE_DIR}/scripts/start_" "${unit}" 2>/dev/null; then
    if [[ "${name}" != ${PREFIX}* ]]; then
      echo "Removing stale unit (different NODE_ID): ${name}"
      systemctl --user disable --now "${name}" 2>/dev/null || true
      rm -f "${unit}"
    fi
  fi
done
shopt -u nullglob

# Render this node's systemd units from the shared templates + profile.
# render_units.py is the single source of truth for what unit files exist.
python3 "${NODE_DIR}/scripts/render_units.py" \
  "${NODE_ID}" \
  --dest "${UNIT_DIR}" \
  --node-dir "${NODE_DIR}" \
  --vault-url "${VAULT_URL}"

PREFIX="${NODE_ID}-"
shopt -s nullglob
RENDERED=()
for svc_path in "${UNIT_DIR}/${PREFIX}"*.service; do
  RENDERED+=("$(basename "${svc_path}")")
done
shopt -u nullglob

systemctl --user daemon-reload

# Always-on services for every node (start with --now). Controller, if present,
# is enabled but not auto-started (gamepad is optional).
ALWAYS_ON_SUFFIXES=("robot-api" "vision" "presence" "battery" "audio" "display")
for svc in "${RENDERED[@]}"; do
  suffix="${svc#${PREFIX}}"
  suffix="${suffix%.service}"
  match=0
  for on in "${ALWAYS_ON_SUFFIXES[@]}"; do
    if [ "$suffix" = "$on" ]; then
      match=1; break
    fi
  done
  if [ "$match" = "1" ]; then
    systemctl --user enable --now "${svc}"
  else
    systemctl --user enable "${svc}" >/dev/null 2>&1 || true
  fi
done

echo "Installed user services for node '${NODE_ID}'."
echo "Node directory: ${NODE_DIR}"
echo "Vault URL: ${VAULT_URL}"
echo
echo "Installed units:"
for svc in "${RENDERED[@]}"; do
  echo "  ${svc}"
done
echo
echo "User lingering was requested so services can start at boot before login."
echo
echo "Useful commands:"
for svc in "${RENDERED[@]}"; do
  echo "  systemctl --user status ${svc}"
done
for svc in "${RENDERED[@]}"; do
  echo "  journalctl --user -u ${svc} -f"
done
