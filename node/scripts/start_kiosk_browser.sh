#!/usr/bin/env bash
# Launch chromium in kiosk mode pointing at the local display_node /ui.
# Runs under the user's X session — make sure the kiosk Pi is set to
# autologin into a desktop session before enabling the systemd unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$NODE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$NODE_DIR/.env"
  set +a
fi

DISPLAY_URL="${KIOSK_URL:-http://127.0.0.1:${DISPLAY_PORT:-5005}/ui}"

# Wait up to ~60s for display_node to come up before launching chromium.
for _ in $(seq 1 60); do
  if curl -fsS "${DISPLAY_URL}" -o /dev/null --max-time 2; then
    break
  fi
  sleep 1
done

# Pick whichever chromium binary the OS image provides.
BROWSER=""
for candidate in chromium-browser chromium google-chrome-stable; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    BROWSER="${candidate}"
    break
  fi
done

if [ -z "${BROWSER}" ]; then
  echo "[kiosk-browser] no chromium binary found; install chromium-browser" >&2
  exit 1
fi

USER_DATA_DIR="${KIOSK_USER_DATA_DIR:-${HOME}/.cache/luhkas-kiosk-chromium}"
mkdir -p "${USER_DATA_DIR}"

# Hide the mouse cursor when idle if unclutter is available (nice for kiosks).
if command -v unclutter >/dev/null 2>&1; then
  unclutter -idle 1 -root &
fi

exec "${BROWSER}" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-features=TranslateUI \
  --check-for-update-interval=31536000 \
  --user-data-dir="${USER_DATA_DIR}" \
  --no-first-run \
  --start-fullscreen \
  --app="${DISPLAY_URL}"
