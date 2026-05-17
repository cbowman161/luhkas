#!/usr/bin/env bash
# Bootstrap a new luhkas node from the GitHub repo.
# Run as the node's service user (e.g. scout/luhkas).
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/cbowman161/luhkas/main/scripts/bootstrap_node.sh | \
#     NODE_ID=scout bash
#
# Required env:
#   NODE_ID   - unique name for this node (e.g. scout, kitchen, office)
#
# Optional env:
#   REPO_URL  - override repo URL (default: https://github.com/cbowman161/luhkas.git)
#   INSTALL_DIR - where to clone (default: ~/luhkas)
#   VAULT_URL - brain URL (default: http://luhkas-vault.local:7000)

set -euo pipefail

NODE_ID="${NODE_ID:?NODE_ID is required}"
REPO_URL="${REPO_URL:-https://github.com/cbowman161/luhkas.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/luhkas}"
VAULT_URL="${VAULT_URL:-http://luhkas-vault.local:7000}"
NODE_DIR="$INSTALL_DIR/node"

echo "[bootstrap] Node ID : $NODE_ID"
echo "[bootstrap] Repo    : $REPO_URL"
echo "[bootstrap] Install : $INSTALL_DIR"
echo "[bootstrap] Vault   : $VAULT_URL"

# ── clone or update ────────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[bootstrap] Updating existing repo..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "[bootstrap] Cloning repo..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ── python dependencies ────────────────────────────────────────────────────────
if [ -f "$NODE_DIR/requirements.txt" ]; then
    echo "[bootstrap] Installing python dependencies..."
    pip3 install -q -r "$NODE_DIR/requirements.txt"
fi

# ── write node env file ────────────────────────────────────────────────────────
ENV_FILE="$NODE_DIR/.env"
cat > "$ENV_FILE" << EOF
LUHKAS_NODE_ID=$NODE_ID
VAULT_CHAT_URL=$VAULT_URL
EOF
echo "[bootstrap] Wrote $ENV_FILE"

# ── install systemd services ───────────────────────────────────────────────────
SYSTEMD_SRC="$NODE_DIR/systemd"
SYSTEMD_DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_DST"

if [ -d "$SYSTEMD_SRC" ]; then
    for svc in "$SYSTEMD_SRC"/*.service; do
        name=$(basename "$svc")
        sed "s|{NODE_DIR}|$NODE_DIR|g; s|{NODE_ID}|$NODE_ID|g; s|{VAULT_URL}|$VAULT_URL|g" \
            "$svc" > "$SYSTEMD_DST/$name"
        echo "[bootstrap] Installed systemd unit: $name"
    done
    systemctl --user daemon-reload
    systemctl --user enable scout-vision scout-robot-api scout-presence scout-controller 2>/dev/null || true
    echo "[bootstrap] Services enabled. Start with: systemctl --user start scout-vision scout-robot-api scout-presence scout-controller"
fi

echo "[bootstrap] Done. Node '$NODE_ID' is ready."
