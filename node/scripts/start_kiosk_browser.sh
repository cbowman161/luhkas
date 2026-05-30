#!/usr/bin/env bash
# Launch chromium in kiosk mode pointing at the configured kiosk URL.
# Runs under the user's X session - make sure the kiosk Pi is set to
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

# Pick a browser. Firefox is more reliable on this kiosk image; Chromium is
# retained as a fallback for images that do not ship Firefox.
BROWSER=""
for candidate in firefox firefox-esr chromium-browser chromium google-chrome-stable /usr/lib/chromium/chromium; do
  if [ -x "${candidate}" ] || command -v "${candidate}" >/dev/null 2>&1; then
    BROWSER="${candidate}"
    break
  fi
done

if [ -z "${BROWSER}" ]; then
  echo "[kiosk-browser] no kiosk browser found; install firefox or chromium-browser" >&2
  exit 1
fi

USER_DATA_DIR="${KIOSK_USER_DATA_DIR:-${HOME}/.cache/luhkas-kiosk-chromium}"
mkdir -p "${USER_DATA_DIR}"

# Hide the mouse cursor when idle if unclutter is available (nice for kiosks).
if command -v unclutter >/dev/null 2>&1; then
  unclutter -idle 1 -root &
fi

case "$(basename "${BROWSER}")" in
  firefox|firefox-esr)
    export MOZ_ENABLE_WAYLAND=0
    exec "${BROWSER}" \
      --no-remote \
      --profile "${KIOSK_FIREFOX_PROFILE:-${HOME}/.cache/luhkas-kiosk-firefox}" \
      --kiosk \
      "${DISPLAY_URL}"
    ;;
esac

exec "${BROWSER}" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-extensions \
  --disable-component-extensions-with-background-pages \
  --disable-background-networking \
  --disable-gpu \
  --disable-software-rasterizer=false \
  --ozone-platform=x11 \
  --disable-features=TranslateUI \
  --check-for-update-interval=31536000 \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port="${KIOSK_DEBUG_PORT:-9222}" \
  --user-data-dir="${USER_DATA_DIR}" \
  --no-first-run \
  --start-fullscreen \
  "${DISPLAY_URL}"
