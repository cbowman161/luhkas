#!/usr/bin/env bash
set -euo pipefail

: "${LUHKAS_NODE_ID:?LUHKAS_NODE_ID is required}"
NODE_ID="${LUHKAS_NODE_ID}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-luhkas-${NODE_ID}}"
TAILSCALE_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
TAILSCALE_AUTHKEY_FILE="${TAILSCALE_AUTHKEY_FILE:-}"
TAILSCALE_UP_FLAGS="${TAILSCALE_UP_FLAGS:-}"
ENABLE_TAILSCALE_SSH="${ENABLE_TAILSCALE_SSH:-1}"
TAILSCALE_ALLOW_INTERACTIVE="${TAILSCALE_ALLOW_INTERACTIVE:-0}"

need_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

wait_for_apt_lock() {
  for i in $(seq 1 60); do
    if ! fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
}

install_tailscale_apt() {
  wait_for_apt_lock
  . /etc/os-release
  repo_os="${ID:-}"
  if [ "$repo_os" = "raspbian" ]; then
    repo_os="debian"
  fi
  case "$repo_os" in
    debian|ubuntu) ;;
    *)
      if printf '%s\n' "${ID_LIKE:-}" | grep -qw debian; then
        repo_os="debian"
      else
        echo "[tailscale] Unsupported distro for apt setup: ${PRETTY_NAME:-unknown}" >&2
        return 1
      fi
      ;;
  esac

  if [ "$repo_os" != "debian" ] && [ "$repo_os" != "ubuntu" ]; then
    echo "[tailscale] Unsupported distro for apt setup: ${PRETTY_NAME:-unknown}" >&2
    return 1
  fi

  codename="${VERSION_CODENAME:-}"
  if [ -z "$codename" ] && command -v lsb_release >/dev/null 2>&1; then
    codename="$(lsb_release -cs)"
  fi
  if [ -z "$codename" ]; then
    echo "[tailscale] Could not determine Ubuntu/Debian codename" >&2
    return 1
  fi

  echo "[tailscale] Installing Tailscale package repo for $repo_os $codename..."
  tmp_key="$(mktemp)"
  tmp_list="$(mktemp)"
  curl -fsSL "https://pkgs.tailscale.com/stable/${repo_os}/${codename}.noarmor.gpg" -o "$tmp_key"
  curl -fsSL "https://pkgs.tailscale.com/stable/${repo_os}/${codename}.tailscale-keyring.list" -o "$tmp_list"
  need_sudo install -m 0755 -d /usr/share/keyrings
  need_sudo install -m 0644 "$tmp_key" /usr/share/keyrings/tailscale-archive-keyring.gpg
  need_sudo install -m 0644 "$tmp_list" /etc/apt/sources.list.d/tailscale.list
  rm -f "$tmp_key" "$tmp_list"
  need_sudo apt-get update
  need_sudo apt-get install -y tailscale
}

if ! command -v tailscale >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    install_tailscale_apt
  else
    echo "[tailscale] No supported package manager found; install Tailscale manually." >&2
    exit 1
  fi
fi

need_sudo systemctl enable --now tailscaled

if tailscale status --json 2>/dev/null | grep -q '"BackendState"[[:space:]]*:[[:space:]]*"Running"'; then
  echo "[tailscale] Already connected."
  tailscale ip -4 2>/dev/null || true
  exit 0
fi

up_args=(--hostname "$TAILSCALE_HOSTNAME")
if tailscale status --json 2>/dev/null | grep -q '"BackendState"[[:space:]]*:[[:space:]]*"NeedsLogin"'; then
  up_args+=(--reset --force-reauth)
fi
if [ "$ENABLE_TAILSCALE_SSH" = "1" ]; then
  up_args+=(--ssh)
fi
if [ -n "$TAILSCALE_AUTHKEY_FILE" ]; then
  up_args+=(--auth-key "file:$TAILSCALE_AUTHKEY_FILE")
elif [ -n "$TAILSCALE_AUTHKEY" ]; then
  up_args+=(--auth-key "$TAILSCALE_AUTHKEY")
elif [ "$TAILSCALE_ALLOW_INTERACTIVE" != "1" ]; then
  echo "[tailscale] No auth key available yet; skipping interactive login."
  echo "[tailscale] The vault will push ~/.config/luhkas/tailscale.authkey after node registration."
  exit 0
fi
if [ -n "$TAILSCALE_UP_FLAGS" ]; then
  # shellcheck disable=SC2206
  extra_flags=($TAILSCALE_UP_FLAGS)
  up_args+=("${extra_flags[@]}")
fi

echo "[tailscale] Starting tunnel as $TAILSCALE_HOSTNAME..."
if [ -n "$TAILSCALE_AUTHKEY" ] || [ -n "$TAILSCALE_AUTHKEY_FILE" ]; then
  need_sudo tailscale up "${up_args[@]}"
else
  echo "[tailscale] No TAILSCALE_AUTHKEY was provided; an interactive login URL may be printed."
  need_sudo tailscale up "${up_args[@]}"
fi

tailscale ip -4 2>/dev/null || true
