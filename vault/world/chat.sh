#!/usr/bin/env bash
# Attach to the persistent LUHKAS chat session.
#
# The session is managed by `luhkas-chat.service` (user-level systemd
# unit, starts at boot via Linger). Push notifications from the
# watchdog land in this feed automatically.
#
# Detach without ending the session: Ctrl-B then D.
# View the session's scrollback even while detached: just attach again.

set -eu

SESSION="${LUHKAS_CHAT_SESSION:-luhkas-chat}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is not running. Starting it via systemd..." >&2
  systemctl --user start luhkas-chat.service
  # Give tmux a moment to bring the session up.
  for _ in 1 2 3 4 5; do
    sleep 0.5
    if tmux has-session -t "$SESSION" 2>/dev/null; then break; fi
  done
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "error: chat session still not up. Check: systemctl --user status luhkas-chat.service" >&2
  exit 1
fi

exec tmux attach -t "$SESSION"
