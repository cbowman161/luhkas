#!/usr/bin/env bash
# Minimal first-boot bootstrap for a LUHKAS edge node.
#
# Copied by ``prep_node_sd.sh`` onto the SD card's boot partition as
# ``/boot/firmware/luhkas-firstboot.sh``. Invoked once by cloud-init's runcmd
# after the user, SSH, hostname and WiFi have been configured.
#
# This script does as little as possible:
#   1. Wait for the network.
#   2. POST /node/register to the vault with our LAN IP.
#   3. Self-clean the SD-side files.
#
# After that, vault's ``node_orchestrator`` takes over: it SSHes into the
# Pi (using the public key Pi Imager seeded into authorized_keys via
# user-data) and runs the actual installation. Anything that fails can be
# retried by re-running the orchestrator on vault — no SD reflash needed.
#
# Reads ``/boot/firmware/luhkas-bootstrap.env`` for NODE_ID, VAULT_URL.

set -euo pipefail

LOG=/var/log/luhkas-firstboot.log
exec >>"$LOG" 2>&1
echo "[$(date -Is)] luhkas-firstboot starting (minimal mode)"

BOOT_DIR="/boot/firmware"
ENV_FILE="${BOOT_DIR}/luhkas-bootstrap.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE missing"
  exit 1
fi
# shellcheck disable=SC1090
. "$ENV_FILE"

: "${NODE_ID:?NODE_ID required in bootstrap.env}"
: "${NODE_USER:?NODE_USER required in bootstrap.env}"
: "${VAULT_URL:?VAULT_URL required in bootstrap.env}"

echo "[firstboot] node_id=${NODE_ID} user=${NODE_USER} vault=${VAULT_URL}"

# ── 1. wait for network ────────────────────────────────────────────────────
for i in $(seq 1 60); do
  if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
    echo "[firstboot] network up after ${i} attempts"
    break
  fi
  sleep 5
done

# ── 2. find our LAN IP and tell vault we exist ─────────────────────────────
LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
HOSTNAME=$(hostname)

if [ -z "$LAN_IP" ]; then
  echo "ERROR: could not determine LAN IP"
  exit 1
fi

PAYLOAD=$(printf '{"node_id":"%s","node_name":"%s","ip":"%s","network":{"lan_ip":"%s","preferred":"lan"},"bootstrap_phase":"pre-install"}' \
  "$NODE_ID" "$HOSTNAME" "$LAN_IP" "$LAN_IP")

# Retry registration for ~5 minutes — vault may take a moment to be reachable
# (e.g. WiFi DHCP still settling, mDNS not yet resolving the vault hostname).
registered=0
for i in $(seq 1 30); do
  if curl -fsS --max-time 10 -X POST \
       -H 'Content-Type: application/json' \
       -d "$PAYLOAD" \
       "${VAULT_URL}/node/register" >>"$LOG" 2>&1; then
    echo "[firstboot] registered with vault on attempt ${i} (ip=${LAN_IP})"
    registered=1
    break
  fi
  echo "[firstboot] registration attempt ${i} failed; retrying..."
  sleep 10
done

if [ "$registered" != "1" ]; then
  echo "[firstboot] WARN: vault unreachable after retries — orchestrator can be invoked manually from vault:"
  echo "             python3 -m vault.node_orchestrator ${NODE_ID} ${LAN_IP}"
fi

# ── 3. clean up SD-side files ──────────────────────────────────────────────
# Leave the env file in place if registration failed so a manual rerun can
# read it; otherwise clean up to keep the boot partition tidy.
if [ "$registered" = "1" ]; then
  rm -f "${BOOT_DIR}/luhkas-bootstrap.env" "${BOOT_DIR}/luhkas-firstboot.sh"
fi

echo "[$(date -Is)] luhkas-firstboot done; awaiting orchestration from vault"
