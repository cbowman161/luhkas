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
#   LUHKAS_TAILSCALE - install/start Tailscale tunnel (default: 1)
#   TAILSCALE_AUTHKEY - optional reusable/ephemeral auth key for unattended setup
#   TAILSCALE_HOSTNAME - tailnet hostname (default: luhkas-$NODE_ID)

set -euo pipefail

BOOTSTRAP_ENV="${LUHKAS_BOOTSTRAP_ENV:-$HOME/.config/luhkas/bootstrap.env}"
if [ -f "$BOOTSTRAP_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$BOOTSTRAP_ENV"
    set +a
fi

: "${NODE_ID:?NODE_ID is required}"
: "${VAULT_URL:?VAULT_URL is required}"
: "${REPO_URL:?REPO_URL is required}"
: "${INSTALL_DIR:?INSTALL_DIR is required}"
LUHKAS_TAILSCALE="${LUHKAS_TAILSCALE:-1}"
NODE_DIR="$INSTALL_DIR/node"

echo "[bootstrap] Node ID : $NODE_ID"
echo "[bootstrap] Repo    : $REPO_URL"
echo "[bootstrap] Install : $INSTALL_DIR"
echo "[bootstrap] Vault   : $VAULT_URL"
echo "[bootstrap] Tailnet : $LUHKAS_TAILSCALE"

# ── clone or update ────────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[bootstrap] Updating existing repo..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "[bootstrap] Cloning repo..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ── python dependencies ────────────────────────────────────────────────────────
# --break-system-packages: Pi OS Trixie+ marks system Python as
# externally-managed (PEP 668). The Pi is dedicated to this purpose, so
# installing into system site-packages is the right answer here.
if [ -f "$NODE_DIR/requirements.txt" ]; then
    echo "[bootstrap] Installing python dependencies..."
    pip3 install -q --break-system-packages -r "$NODE_DIR/requirements.txt"
fi

# ── write node env file ────────────────────────────────────────────────────────
ENV_FILE="$NODE_DIR/.env"
cat > "$ENV_FILE" << EOF
LUHKAS_NODE_ID=$NODE_ID
VAULT_CHAT_URL=$VAULT_URL
EOF
echo "[bootstrap] Wrote $ENV_FILE"

# ── private node tunnel ───────────────────────────────────────────────────────
if [ "$LUHKAS_TAILSCALE" = "1" ]; then
    echo "[bootstrap] Setting up Tailscale tunnel..."
    LUHKAS_NODE_ID="$NODE_ID" "$NODE_DIR/scripts/setup_tailscale.sh" || {
        echo "[bootstrap] WARNING: Tailscale setup failed. Re-run:"
        echo "  LUHKAS_NODE_ID=$NODE_ID $NODE_DIR/scripts/setup_tailscale.sh"
    }
fi

echo "[bootstrap] If this node registered with the vault before joining the tailnet,"
echo "[bootstrap] the vault can also push/rotate the Tailscale auth key automatically."

# ── install systemd services ───────────────────────────────────────────────────
# Units are rendered from node/systemd/templates/<svc>.service.tmpl and the
# node profile (node/profiles/<NODE_ID>.json). The profile is the single
# source of truth: no per-node unit files to maintain.
SYSTEMD_DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_DST"

# Make sure ``systemctl --user`` will work in this shell. When invoked from
# the firstboot ``sudo -u luhkas`` context the environment lacks the user
# manager's DBUS socket — enabling linger spins up user@<uid>.service and
# its /run/user/<uid>/bus.
USER_UID="$(id -u)"
if command -v loginctl >/dev/null 2>&1; then
    sudo loginctl enable-linger "$USER" 2>/dev/null || loginctl enable-linger "$USER" 2>/dev/null || true
    for i in 1 2 3 4 5; do
        [ -S "/run/user/${USER_UID}/bus" ] && break
        sleep 1
    done
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${USER_UID}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

if python3 "$NODE_DIR/scripts/render_units.py" \
    "$NODE_ID" \
    --dest "$SYSTEMD_DST" \
    --node-dir "$NODE_DIR" \
    --vault-url "$VAULT_URL"; then
    shopt -s nullglob
    installed=()
    for svc in "$SYSTEMD_DST/${NODE_ID}-"*.service; do
        installed+=("$(basename "$svc")")
    done
    shopt -u nullglob
    if [ ${#installed[@]} -gt 0 ]; then
        systemctl --user daemon-reload
        systemctl --user enable "${installed[@]}" 2>/dev/null || true
        echo "[bootstrap] Services enabled: ${installed[*]}"
        echo "[bootstrap] Start with: systemctl --user start ${installed[*]}"
    else
        echo "[bootstrap] No systemd units rendered for node '$NODE_ID'."
    fi
else
    echo "[bootstrap] WARNING: render_units.py failed for node '$NODE_ID'."
fi

echo "[bootstrap] Done. Node '$NODE_ID' is ready."
