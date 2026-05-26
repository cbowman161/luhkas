#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_FILE="${TAILSCALE_AUTHKEY_FILE:-$REPO_ROOT/vault/secrets/tailscale.authkey}"

mkdir -p "$(dirname "$KEY_FILE")"
chmod 700 "$(dirname "$KEY_FILE")" 2>/dev/null || true

if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
  key="$TAILSCALE_AUTHKEY"
else
  if [ ! -t 0 ]; then
    echo "No TTY available. Run interactively or set TAILSCALE_AUTHKEY." >&2
    exit 1
  fi
  printf "New Tailscale auth key: "
  stty -echo
  trap 'stty echo' EXIT
  read -r key
  stty echo
  trap - EXIT
  printf "\n"
fi

if [ -z "$key" ]; then
  echo "No key provided; leaving existing file unchanged." >&2
  exit 1
fi

tmp="$(mktemp)"
printf "%s\n" "$key" > "$tmp"
install -m 600 "$tmp" "$KEY_FILE"
rm -f "$tmp"

echo "Saved Tailscale auth key to $KEY_FILE"
