#!/usr/bin/env bash
# Prepare a freshly-flashed Pi OS SD card so a LUHKAS edge node boots
# completely autonomously on first power-on.
#
# Run on the vault PC. The SD card can be anywhere — a USB reader, a CSI
# adapter, the built-in slot — as long as it shows up as a block device
# with a vfat partition labeled "bootfs" containing Pi Imager's firstrun.sh.
#
# Usage:
#   scripts/prep_node_sd.sh <node-id>                       # auto-detect SD
#   scripts/prep_node_sd.sh <node-id> --device /dev/sda     # explicit device
#   scripts/prep_node_sd.sh <node-id> --boot /path/to/bootfs  # pre-mounted
#
# Optional flags:
#   --vault-url URL     vault chat URL (default: http://luhkas-vault:7000)
#   --repo-url URL      git repo to clone on the Pi
#   --user NAME         node service user (default: luhkas)
#   --no-tailscale      skip Tailscale auth-key provisioning
#
# Pre-reqs:
#   1. node/profiles/<node-id>.json exists in this repo (defines modules,
#      services, ports — the only file you need to author per node).
#   2. Pi Imager was used to flash with advanced options set (hostname,
#      SSH+pubkey, user, WiFi).
#   3. A Tailscale auth key exists at vault/secrets/tailscale.authkey
#      (the rotation timer maintains this).
#
# The script self-elevates with sudo if needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stash original args so we can re-exec under sudo after parsing.
ORIG_ARGS=("$@")

NODE_ID=""
BOOT_DIR=""
DEVICE=""
VAULT_URL="http://luhkas-vault:7000"
REPO_URL=""
NODE_USER="luhkas"
PROVISION_TAILSCALE=1

# Parse args BEFORE elevating so --help works as a normal user.
while [ $# -gt 0 ]; do
  case "$1" in
    --boot)         BOOT_DIR="$2"; shift 2;;
    --device)       DEVICE="$2"; shift 2;;
    --vault-url)    VAULT_URL="$2"; shift 2;;
    --repo-url)     REPO_URL="$2"; shift 2;;
    --user)         NODE_USER="$2"; shift 2;;
    --no-tailscale) PROVISION_TAILSCALE=0; shift;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0;;
    -*)             echo "unknown flag: $1" >&2; exit 2;;
    *)              if [ -z "$NODE_ID" ]; then NODE_ID="$1"; else echo "extra arg: $1" >&2; exit 2; fi; shift;;
  esac
done

# ── self-elevate ───────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
  echo "[prep] elevating (sudo)..."
  exec sudo -- "$0" "${ORIG_ARGS[@]}"
fi

# When sudo elevated us, SUDO_USER points at the original caller; otherwise
# we were invoked directly as root and fall back to the current user.
INVOKING_USER="${SUDO_USER:-$(id -un)}"
INVOKING_HOME="$(getent passwd "$INVOKING_USER" | cut -d: -f6)"
if [ -z "$INVOKING_HOME" ]; then
  INVOKING_HOME="$HOME"
fi

if [ -z "$NODE_ID" ]; then
  echo "ERROR: NODE_ID is required (e.g. 'kiosk')" >&2
  exit 2
fi

PROFILE_PATH="${REPO_ROOT}/node/profiles/${NODE_ID}.json"
if [ ! -f "$PROFILE_PATH" ]; then
  echo "ERROR: no profile at ${PROFILE_PATH}" >&2
  echo "       Create one — see node/profiles/scout.json for an example." >&2
  exit 2
fi

echo "[prep] node profile: ${PROFILE_PATH}"

# Resolve and print everything the loader will infer from the profile, so
# you can verify what the node is going to do before ejecting the SD card.
python3 - "$REPO_ROOT" "$NODE_ID" <<'PY'
import json, sys, pathlib
repo_root, node_id = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo_root / "node"))
from profile_loader import load_profile
p = load_profile(node_id)

modules = p.get("modules") or []
services = p.get("services") or {}
extras = p.get("extra_units") or []

def _mark(name):
    install = repo_root / "node" / name / "install.sh"
    return "install.sh" if install.exists() else "(no install.sh)"

print(f"[prep] node_id        : {p.get('node_id')}")
print(f"[prep] display        : has_display={p['display'].get('has_display')}")
print(f"[prep] sync host      : {p['sync'].get('host')}")
print(f"[prep] modules ({len(modules)}):")
for m in modules:
    print(f"          - {m:<14}  {_mark(m)}")
print(f"[prep] services ({len(services)}):")
for name, spec in services.items():
    port = spec.get("port") if isinstance(spec, dict) else spec
    print(f"          - {name:<10}  port={port}")
if extras:
    print(f"[prep] extra units    : {', '.join(str(e) for e in extras)}")
PY


# ── locate / mount the SD bootfs partition ─────────────────────────────────
MOUNTED_BY_US=0
TMP_MOUNT=""

cleanup() {
  if [ "$MOUNTED_BY_US" = "1" ] && [ -n "$TMP_MOUNT" ] && mountpoint -q "$TMP_MOUNT"; then
    sync
    umount "$TMP_MOUNT" || true
    rmdir "$TMP_MOUNT" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

resolve_bootfs_part() {
  local dev="$1"
  # ``-l`` keeps output flat (no tree-drawing chars); ``-p`` returns full paths.
  lsblk -lnp -o NAME,LABEL "$dev" 2>/dev/null | awk '$2 == "bootfs" { print $1 }' | head -1
}

discover_sd() {
  local candidates=()
  while read -r name rest; do
    local dev="/dev/$name"
    [ -b "$dev" ] || continue
    local part
    part=$(resolve_bootfs_part "$dev" || true)
    [ -n "$part" ] || continue
    candidates+=("$dev")
  done < <(lsblk -d -no NAME 2>/dev/null)

  if [ ${#candidates[@]} -eq 0 ]; then
    echo "ERROR: no block device with a 'bootfs'-labeled partition found." >&2
    echo "       Re-flash with Pi Imager, or pass --device /dev/sdX explicitly." >&2
    return 2
  fi
  if [ ${#candidates[@]} -gt 1 ]; then
    echo "ERROR: multiple SD-card candidates found:" >&2
    for c in "${candidates[@]}"; do echo "  $c" >&2; done
    echo "       Disambiguate with --device /dev/sdX" >&2
    return 2
  fi
  echo "${candidates[0]}"
}

if [ -z "$BOOT_DIR" ]; then
  if [ -z "$DEVICE" ]; then
    DEVICE=$(discover_sd)
    echo "[prep] auto-detected SD device: ${DEVICE}"
  fi
  BOOT_PART=$(resolve_bootfs_part "$DEVICE")
  if [ -z "$BOOT_PART" ]; then
    echo "ERROR: no 'bootfs'-labeled partition on ${DEVICE}" >&2
    exit 2
  fi

  EXISTING_MOUNT=$(findmnt -n -o TARGET "$BOOT_PART" 2>/dev/null | head -1 || true)
  if [ -n "$EXISTING_MOUNT" ]; then
    BOOT_DIR="$EXISTING_MOUNT"
    echo "[prep] bootfs already mounted at ${BOOT_DIR}"
  else
    TMP_MOUNT=$(mktemp -d -t luhkas-bootfs.XXXXXX)
    mount "$BOOT_PART" "$TMP_MOUNT"
    BOOT_DIR="$TMP_MOUNT"
    MOUNTED_BY_US=1
    echo "[prep] mounted ${BOOT_PART} -> ${BOOT_DIR}"
  fi
fi

# Pi Imager writes EITHER ``firstrun.sh`` (legacy) OR ``user-data`` (cloud-init,
# newer builds). We hook into whichever is present.
PI_FIRSTRUN="${BOOT_DIR}/firstrun.sh"
CLOUD_USERDATA="${BOOT_DIR}/user-data"
PROVISION_MODE=""
if [ -f "$PI_FIRSTRUN" ]; then
  PROVISION_MODE="firstrun"
elif [ -f "$CLOUD_USERDATA" ]; then
  PROVISION_MODE="cloud-init"
else
  echo "ERROR: neither ${PI_FIRSTRUN} nor ${CLOUD_USERDATA} found." >&2
  echo "       Re-flash with Pi Imager and set hostname/SSH/WiFi/user in advanced options." >&2
  exit 2
fi
echo "[prep] provisioning mode: ${PROVISION_MODE}"

# ── Tailscale auth key ─────────────────────────────────────────────────────
TAILSCALE_AUTHKEY=""
if [ "$PROVISION_TAILSCALE" = "1" ]; then
  AUTHKEY_FILE="${REPO_ROOT}/vault/secrets/tailscale.authkey"
  ROTATE_IF_NEEDED="${REPO_ROOT}/scripts/rotate_tailscale_authkey_if_needed.py"
  if [ -x "$ROTATE_IF_NEEDED" ]; then
    sudo -u "$INVOKING_USER" -H python3 "$ROTATE_IF_NEEDED" \
      || echo "[prep] WARN: rotation check failed; using existing key" >&2
  fi
  if [ ! -f "$AUTHKEY_FILE" ]; then
    echo "ERROR: ${AUTHKEY_FILE} missing. Provision via scripts/rotate_tailscale_authkey.sh." >&2
    exit 2
  fi
  TAILSCALE_AUTHKEY=$(tr -d '\r\n' < "$AUTHKEY_FILE")
fi

# ── write bootstrap env + firstboot script onto the SD card ────────────────
ENV_OUT="${BOOT_DIR}/luhkas-bootstrap.env"
{
  echo "NODE_ID=${NODE_ID}"
  echo "NODE_USER=${NODE_USER}"
  echo "VAULT_URL=${VAULT_URL}"
  [ -n "$REPO_URL" ]          && echo "REPO_URL=${REPO_URL}"
  [ -n "$TAILSCALE_AUTHKEY" ] && echo "TAILSCALE_AUTHKEY=${TAILSCALE_AUTHKEY}"
} > "${ENV_OUT}"
chmod 0600 "${ENV_OUT}" || true

FIRSTBOOT_SRC="${REPO_ROOT}/scripts/luhkas_firstboot.sh"
FIRSTBOOT_DST="${BOOT_DIR}/luhkas-firstboot.sh"
install -m 0755 "$FIRSTBOOT_SRC" "$FIRSTBOOT_DST"

# ── hook our firstboot into Pi Imager's provisioning ───────────────────────
PATCH_MARKER="luhkas-bootstrap-hook"

if [ "$PROVISION_MODE" = "firstrun" ]; then
  if ! grep -q "$PATCH_MARKER" "$PI_FIRSTRUN"; then
    python3 - "$PI_FIRSTRUN" "# $PATCH_MARKER" <<'PY'
import sys, re, pathlib
path, marker = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
hook = f"""
{marker}
if [ -x /boot/firmware/luhkas-firstboot.sh ]; then
  /boot/firmware/luhkas-firstboot.sh >>/var/log/luhkas-firstboot.log 2>&1 &
fi
"""
patched, n = re.subn(r"(\nrm -f /boot/firstrun\.sh)", hook + r"\1", text, count=1)
if n == 0:
    patched, n = re.subn(r"(\nrm -f /boot/firmware/firstrun\.sh)", hook + r"\1", text, count=1)
if n == 0:
    patched = re.sub(r"(\nexit 0\s*$)", hook + r"\1", text, count=1)
path.write_text(patched)
PY
    echo "[prep] patched ${PI_FIRSTRUN}"
  else
    echo "[prep] firstrun.sh already patched (marker found)"
  fi
else
  # cloud-init: merge a runcmd entry into user-data via PyYAML.
  if grep -q "$PATCH_MARKER" "$CLOUD_USERDATA"; then
    echo "[prep] user-data already patched (marker found)"
  else
    python3 - "$CLOUD_USERDATA" "$PATCH_MARKER" <<'PY'
import sys, pathlib
import yaml

path = pathlib.Path(sys.argv[1])
marker = sys.argv[2]
text = path.read_text()

# Cloud-init expects a "#cloud-config" header — preserve whatever's at the top.
first_line = text.split("\n", 1)[0].strip()
header = first_line if first_line.startswith("#") else "#cloud-config"

data = yaml.safe_load(text) or {}
if not isinstance(data, dict):
    raise SystemExit(f"unexpected user-data root: {type(data).__name__}")

# Drop a marker comment so prep is idempotent.
data.setdefault("_luhkas_marker", marker)

runcmd = data.setdefault("runcmd", [])
if not isinstance(runcmd, list):
    raise SystemExit("user-data runcmd is not a list")
runcmd.append(
    "/boot/firmware/luhkas-firstboot.sh >>/var/log/luhkas-firstboot.log 2>&1 &"
)

out = header + "\n" + yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
path.write_text(out)
PY
    echo "[prep] merged runcmd into ${CLOUD_USERDATA}"
  fi
fi

sync

echo
echo "[prep] ✔ SD card prepared for node '${NODE_ID}'"
echo "[prep]   profile        : ${PROFILE_PATH}"
echo "[prep]   boot partition : ${BOOT_DIR}"
[ -n "${DEVICE:-}" ] && echo "[prep]   device         : ${DEVICE}"
echo "[prep]   env file       : ${ENV_OUT}"
echo "[prep]   firstboot      : ${FIRSTBOOT_DST}"
if [ "$PROVISION_MODE" = "firstrun" ]; then
  echo "[prep]   firstrun.sh    : ${PI_FIRSTRUN} (patched)"
else
  echo "[prep]   user-data      : ${CLOUD_USERDATA} (runcmd merged)"
fi
echo "[prep]   tailscale key  : $([ -n "$TAILSCALE_AUTHKEY" ] && echo provisioned || echo skipped)"
echo
echo "Eject the SD, insert into the Pi, power on. First-boot log on the Pi:"
echo "  /var/log/luhkas-firstboot.log"
