# Shared helpers for per-module install.sh scripts.
# Source (not exec) this file at the top of any install.sh:
#   . "$(dirname "$(readlink -f "$0")")/../scripts/lib_install.sh"
# or from firstboot:
#   . "${INSTALL_DIR}/node/scripts/lib_install.sh"

# Wait until cloud-init / unattended-upgrades / another apt process releases
# the dpkg lock before we try to install anything. Up to ~5 minutes.
wait_for_apt() {
  local i
  for i in $(seq 1 100); do
    if ! fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; then
      return 0
    fi
    echo "[lib_install] dpkg lock held; waiting (${i})..."
    sleep 3
  done
  echo "[lib_install] WARN: dpkg lock still held after 5 min; proceeding anyway"
  return 0
}

# Run ``apt-get update`` once per firstboot. Subsequent calls are no-ops as
# long as ``LUHKAS_APT_UPDATED=1`` is set in the env. Each install.sh that
# does apt installs should call ``ensure_apt_updated`` instead of running
# ``apt-get update`` directly.
ensure_apt_updated() {
  if [ "${LUHKAS_APT_UPDATED:-0}" = "1" ]; then
    return 0
  fi
  wait_for_apt
  echo "[lib_install] apt-get update"
  apt-get update -y
  export LUHKAS_APT_UPDATED=1
}

# Idempotent apt install with the lock wait baked in.
apt_install() {
  wait_for_apt
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}
