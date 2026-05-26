#!/usr/bin/env bash
# Long-running ingest of the full English Wikipedia ZIM into world.lance.
#
# Survives terminal disconnect (setsid + nohup), logs to disk, writes a
# resume cursor after every batch flush, and re-running the script picks
# up where the last invocation stopped.
#
# Usage:
#   bash run_full_ingest.sh              # start or resume
#   bash run_full_ingest.sh status       # show state file + counts
#   bash run_full_ingest.sh tail         # tail the live log
#   bash run_full_ingest.sh stop         # graceful stop (next flush exits)
#
# Concurrent chat is unaffected — vault-runtime keeps using Ollama. The
# ingest holds its own bge-m3 instance on the GPU; both fit in 24 GB.

set -u

ROOT="/home/vault/world_data"
ZIM="${WORLD_INGEST_ZIM:-$ROOT/zim/wikipedia_en_all_nopic_2026-03.zim}"
STATE_FILE="$ROOT/logs/ingest_wiki.state.json"
LOG_FILE="$ROOT/logs/ingest_wiki.log"
PID_FILE="$ROOT/logs/ingest_wiki.pid"
VAULT="/home/vault/luhkas/vault"

mkdir -p "$ROOT/logs"

CMD="${1:-start}"

case "$CMD" in
  status)
    if [[ -f "$STATE_FILE" ]]; then
      echo "=== state file ==="
      cat "$STATE_FILE"
    else
      echo "(no state file yet at $STATE_FILE)"
    fi
    echo
    echo "=== process ==="
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "RUNNING pid=$(cat "$PID_FILE")"
    else
      echo "NOT RUNNING"
    fi
    echo
    echo "=== vault /world/status ==="
    curl -sS http://127.0.0.1:7000/world/status 2>&1 | head -30
    exit 0
    ;;
  tail)
    exec tail -f "$LOG_FILE"
    ;;
  stop)
    if [[ -f "${PID_FILE}.scope" ]]; then
      scope=$(cat "${PID_FILE}.scope")
      if systemctl --user is-active --quiet "$scope.scope"; then
        echo "stopping systemd scope $scope (current batch will flush)"
        systemctl --user stop "$scope.scope"
        exit 0
      fi
    fi
    if [[ -f "$PID_FILE" ]]; then
      pid=$(cat "$PID_FILE")
      if kill -0 "$pid" 2>/dev/null; then
        echo "sending SIGTERM to $pid (resume cursor will be valid)"
        kill -TERM "$pid"
        exit 0
      fi
    fi
    echo "not running"
    exit 0
    ;;
  start|resume|"")
    ;;
  *)
    echo "unknown command: $CMD" >&2
    exit 2
    ;;
esac

# Block double-starts.
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID_FILE"); use \`$0 status\` or \`$0 stop\`" >&2
  exit 1
fi

cd "$VAULT"

# Evict the qwen2.5vl vision model from Ollama before we load native
# bge-m3 on the GPU. The vision model eats ~13.5 GB of VRAM (much
# bigger than its Q4 7B param count suggests because of the image
# encoder), and combined with the other warm chat models leaves no
# room for native bge-m3. Vision will lazy-reload on first /vision
# request — slow first hit, but the rest of chat keeps working
# during the multi-day ingest.
curl -sS -X POST http://127.0.0.1:11434/api/generate \
  -H "content-type: application/json" \
  -d "{\"model\":\"qwen2.5vl:7b\",\"keep_alive\":0}" \
  >/dev/null 2>&1 || true

# Launch in its own systemd-user transient scope so the ingest lives
# outside vault-runtime.service's cgroup. Without this, a restart of
# vault-runtime (which is the parent when the runner is invoked from
# the chat admin command) wipes the whole cgroup — including the ingest.
# `--scope` runs in the foreground; we still `&` it so the runner returns.
SCOPE_NAME="luhkas-wiki-ingest-$(date +%s)"
systemd-run --user --scope --quiet --unit="$SCOPE_NAME" \
  bash -c "exec python3 -u -m world.ingest_wiki \
    \"$ZIM\" \
    --embedder native \
    --batch 64 \
    --state-file \"$STATE_FILE\" \
    --resume-from-state \
    >> \"$LOG_FILE\" 2>&1 < /dev/null" &

echo "$SCOPE_NAME" > "${PID_FILE}.scope"

# Wait up to 10s for the scope to register and report its MainPID.
PID=""
for i in 1 2 3 4 5 6 7 8 9 10; do
  PID=$(systemctl --user show "$SCOPE_NAME.scope" -p MainPID --value 2>/dev/null || echo "")
  if [[ -n "$PID" && "$PID" != "0" ]]; then break; fi
  sleep 1
done
if [[ -z "$PID" || "$PID" == "0" ]]; then
  # Last-resort: scan our user processes for the ingest module.
  PID=$(pgrep -u "$USER" -f "world.ingest_wiki" | head -1)
fi
echo "${PID:-unknown}" > "$PID_FILE"

echo "started ingest pid=${PID:-unknown} scope=$SCOPE_NAME"
echo "  zim:       $ZIM"
echo "  state:     $STATE_FILE"
echo "  log:       $LOG_FILE"
echo
echo "monitor:   bash $0 status"
echo "tail:      bash $0 tail"
echo "stop:      bash $0 stop   # next flush exits cleanly, resume from cursor"
