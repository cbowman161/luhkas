#!/usr/bin/env bash
# First-boot bootstrap for a LUHKAS edge node.
#
# Copied by ``prep_node_sd.sh`` onto the SD card's boot partition as
# ``/boot/firmware/luhkas-firstboot.sh``. Pi Imager invokes it via either
# its legacy firstrun.sh hook (patched) or cloud-init's runcmd. Runs as
# root.
#
# Flow:
#   1. wait for network
#   2. install baseline apt deps (git, python3-pip, etc.)
#   3. clone repo as $NODE_USER
#   4. for each module in node/profiles/${NODE_ID}.json's "modules" list,
#      run ``node/<module>/install.sh`` if it exists (Hailo, audio HAT,
#      chromium, etc. — entirely declarative on profile modules)
#   5. run bootstrap_node.sh (renders systemd units from profile +
#      templates, sets up Tailscale, enables user services)
#   6. clean up SD-card files and reboot
#
# Reads ``/boot/firmware/luhkas-bootstrap.env`` for NODE_ID, NODE_USER,
# VAULT_URL, REPO_URL, TAILSCALE_AUTHKEY.

set -euo pipefail

LOG=/var/log/luhkas-firstboot.log
exec >>"$LOG" 2>&1
echo "[$(date -Is)] luhkas-firstboot starting"

on_err() {
  local rc=$?
  echo "[$(date -Is)] luhkas-firstboot FAILED with exit code ${rc}"
  echo "[firstboot] recovery:  ssh into the Pi and run:"
  echo "             sudo /boot/firmware/luhkas-firstboot.sh   # if SD-side files still present"
  echo "             OR (after first reboot):"
  echo "             sudo -u luhkas bash ~/luhkas/scripts/bootstrap_node.sh"
  echo "             sudo bash ~/luhkas/node/<module>/install.sh   # per failed module"
  exit "$rc"
}
trap on_err ERR

BOOT_DIR="/boot/firmware"
ENV_FILE="${BOOT_DIR}/luhkas-bootstrap.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE missing"
  exit 1
fi
# shellcheck disable=SC1090
. "$ENV_FILE"

NODE_ID="${NODE_ID:?NODE_ID required in bootstrap.env}"
NODE_USER="${NODE_USER:-luhkas}"
VAULT_URL="${VAULT_URL:-http://luhkas-vault:7000}"
REPO_URL="${REPO_URL:-https://github.com/cbowman161/luhkas.git}"
TAILSCALE_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
INSTALL_DIR="/home/${NODE_USER}/luhkas"

echo "[firstboot] node_id=${NODE_ID} user=${NODE_USER} vault=${VAULT_URL}"

# ── 1. wait for network ────────────────────────────────────────────────────
for i in $(seq 1 60); do
  if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
    echo "[firstboot] network up after ${i} attempts"
    break
  fi
  sleep 5
done

# ── 2. baseline apt deps (everything else is per-module install.sh) ─────────
# Wait for cloud-init / unattended-upgrades to release the dpkg lock before
# we start. Also run ``apt-get update`` ONCE here and let each module's
# install.sh skip its own update via the LUHKAS_APT_UPDATED flag.
export DEBIAN_FRONTEND=noninteractive

wait_for_apt_lock() {
  for i in $(seq 1 100); do
    if ! fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; then
      return 0
    fi
    echo "[firstboot] dpkg locked; waiting (${i})..."
    sleep 3
  done
  echo "[firstboot] WARN: dpkg lock still held after 5 min; proceeding anyway"
}

wait_for_apt_lock
apt-get update -y
export LUHKAS_APT_UPDATED=1

wait_for_apt_lock
apt-get install -y --no-install-recommends \
  git curl ca-certificates \
  python3 python3-pip python3-venv

# ── 3. user + secrets ──────────────────────────────────────────────────────
if ! id -u "$NODE_USER" >/dev/null 2>&1; then
  echo "ERROR: user $NODE_USER not present"
  exit 1
fi
USER_HOME=$(getent passwd "$NODE_USER" | cut -d: -f6)
USER_CFG="${USER_HOME}/.config/luhkas"
install -d -m 0755 -o "$NODE_USER" -g "$NODE_USER" "$USER_CFG"

AUTHKEY_PATH=""
if [ -n "${TAILSCALE_AUTHKEY}" ]; then
  AUTHKEY_PATH="${USER_CFG}/tailscale.authkey"
  printf '%s\n' "${TAILSCALE_AUTHKEY}" > "${AUTHKEY_PATH}"
  chown "$NODE_USER:$NODE_USER" "${AUTHKEY_PATH}"
  chmod 0600 "${AUTHKEY_PATH}"
fi

cat > "${USER_CFG}/bootstrap.env" <<EOF
LUHKAS_NODE_ID=${NODE_ID}
VAULT_CHAT_URL=${VAULT_URL}
${AUTHKEY_PATH:+TAILSCALE_AUTHKEY_FILE=${AUTHKEY_PATH}}
EOF
chown "$NODE_USER:$NODE_USER" "${USER_CFG}/bootstrap.env"
chmod 0600 "${USER_CFG}/bootstrap.env"

# ── 4. clone repo ──────────────────────────────────────────────────────────
sudo -u "$NODE_USER" -H bash -lc "
  set -e
  if [ -d '${INSTALL_DIR}/.git' ]; then
    git -C '${INSTALL_DIR}' pull --ff-only || true
  else
    git clone '${REPO_URL}' '${INSTALL_DIR}'
  fi
"

# ── 5. run each module's install.sh (modules drive everything) ─────────────
PROFILE_PATH="${INSTALL_DIR}/node/profiles/${NODE_ID}.json"
if [ ! -f "$PROFILE_PATH" ]; then
  echo "ERROR: profile not found at ${PROFILE_PATH}"
  exit 1
fi

MODULES=$(python3 -c "
import json, pathlib
p = pathlib.Path('${PROFILE_PATH}')
data = json.loads(p.read_text())
for m in data.get('modules', []):
    print(m)
")

for mod in $MODULES; do
  INSTALL="${INSTALL_DIR}/node/${mod}/install.sh"
  if [ -x "$INSTALL" ]; then
    echo "[firstboot] >> running ${mod}/install.sh"
    NODE_USER="$NODE_USER" NODE_ID="$NODE_ID" \
      LUHKAS_APT_UPDATED="${LUHKAS_APT_UPDATED:-0}" \
      bash "$INSTALL" \
      || echo "[firstboot] WARN: ${mod}/install.sh exited non-zero — see log; firstboot continues"
  else
    echo "[firstboot] (no install.sh for ${mod})"
  fi
done

# ── 6. bootstrap_node.sh: render systemd units + Tailscale + enable units ──
sudo -u "$NODE_USER" -H bash -lc "
  set -e
  NODE_ID='${NODE_ID}' VAULT_URL='${VAULT_URL}' INSTALL_DIR='${INSTALL_DIR}' \
    bash '${INSTALL_DIR}/scripts/bootstrap_node.sh'
  bash '${INSTALL_DIR}/node/scripts/install_user_services.sh' || true
"

# ── 7. cleanup + reboot ────────────────────────────────────────────────────
rm -f "${BOOT_DIR}/luhkas-bootstrap.env" "${BOOT_DIR}/luhkas-firstboot.sh"

echo "[$(date -Is)] luhkas-firstboot done — rebooting"
sync
sleep 2
shutdown -r +1 "luhkas-firstboot complete" || reboot
