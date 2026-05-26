#!/usr/bin/env bash
# Install the world-vault user-level systemd units:
#   * luhkas-world-watchdog.timer  — fires the ingest-health check every 2 min
#   * luhkas-world-watchdog.service — the oneshot watchdog itself
#   * luhkas-chat.service           — persistent tmux session running ui_client
#                                      (push notifications in the chat feed)
#
# Also drops a `chat` symlink into ~/bin so you can type `chat` from
# anywhere to attach.
#
# Idempotent: re-run any time to refresh the unit files + re-enable.

set -eu

UNIT_DIR="$HOME/.config/systemd/user"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/systemd"
BIN_DIR="$HOME/bin"
mkdir -p "$UNIT_DIR" "$BIN_DIR"

for unit in luhkas-world-watchdog.service luhkas-world-watchdog.timer luhkas-chat.service; do
  cp "$SRC_DIR/$unit" "$UNIT_DIR/$unit"
  echo "installed $UNIT_DIR/$unit"
done

# `chat` convenience launcher
ln -sf "$(cd "$(dirname "$0")" && pwd)/chat.sh" "$BIN_DIR/chat"
chmod +x "$(cd "$(dirname "$0")" && pwd)/chat.sh"
echo "installed $BIN_DIR/chat -> $(readlink "$BIN_DIR/chat")"

systemctl --user daemon-reload
systemctl --user enable --now luhkas-world-watchdog.timer
systemctl --user enable --now luhkas-chat.service

echo
echo "=== watchdog timer ==="
systemctl --user status luhkas-world-watchdog.timer --no-pager | head -6
echo
echo "=== chat session ==="
systemctl --user status luhkas-chat.service --no-pager | head -6

cat <<'EOF'

----------------------------------------------------------------------
Attach to chat:        chat        (or: tmux attach -t luhkas-chat)
Detach (keep running): Ctrl-B then D
Tail watchdog log:     tail -f /home/vault/world_data/logs/ingest_watchdog.log
Watchdog manual run:   systemctl --user start luhkas-world-watchdog.service
Disable everything:    systemctl --user disable --now luhkas-world-watchdog.timer luhkas-chat.service
----------------------------------------------------------------------
EOF
