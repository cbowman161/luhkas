#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="$REPO_ROOT/vault/secrets"
OAUTH_FILE="${TAILSCALE_OAUTH_FILE:-$SECRETS_DIR/tailscale_oauth.env}"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR" 2>/dev/null || true

if [ ! -t 0 ]; then
  echo "Run interactively so credentials are not saved in shell history." >&2
  exit 1
fi

read -r -p "Tailscale OAuth client ID: " client_id
printf "Tailscale OAuth client secret: "
stty -echo
trap 'stty echo' EXIT
read -r client_secret
stty echo
trap - EXIT
printf "\n"
read -r -p "Auth key tag [tag:luhkas-node]: " tag
tag="${tag:-tag:luhkas-node}"
read -r -p "Tailnet API id [-]: " tailnet
tailnet="${tailnet:--}"

if [ -z "$client_id" ] || [ -z "$client_secret" ]; then
  echo "Client ID and secret are required." >&2
  exit 1
fi

tmp="$(mktemp)"
cat > "$tmp" <<EOF
TAILSCALE_OAUTH_CLIENT_ID='$client_id'
TAILSCALE_OAUTH_CLIENT_SECRET='$client_secret'
TAILSCALE_AUTHKEY_TAG='$tag'
TAILSCALE_TAILNET='$tailnet'
TAILSCALE_AUTHKEY_EXPIRY_SECONDS='7776000'
TAILSCALE_AUTHKEY_DESCRIPTION='luhkas-node-bootstrap'
TAILSCALE_AUTHKEY_REUSABLE='1'
TAILSCALE_AUTHKEY_EPHEMERAL='0'
TAILSCALE_AUTHKEY_PREAUTHORIZED='1'
EOF
install -m 600 "$tmp" "$OAUTH_FILE"
rm -f "$tmp"

echo "Saved Tailscale OAuth credentials to $OAUTH_FILE"
