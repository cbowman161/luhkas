#!/usr/bin/env bash
# Install xrdp + xfce4 so a node accepts Remote Desktop connections from
# the Windows App / Microsoft RDP client, restricted to the tailscale0
# interface (matches the vault-side pattern from
# HANDOFF_TAILSCALE_REMOTE_DESKTOP.md).
#
# Triggered by the orchestrator when profile.rdp.enabled is true.
# Idempotent. Run as root; orchestrator supplies NODE_USER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/lib_install.sh"

: "${NODE_USER:?NODE_USER is required}"
USER_HOME="$(getent passwd "$NODE_USER" | cut -d: -f6)"
[ -n "$USER_HOME" ] || { echo "ERROR: user $NODE_USER not found"; exit 2; }

echo "[install_rdp] starting (user=${NODE_USER})"

# ── packages ─────────────────────────────────────────────────────────────────
# xfce4 keeps the RDP session lightweight. dbus-x11 lets xfce4 talk to its
# session bus. xfce4-terminal so the desktop has a usable terminal app.
ensure_apt_updated
apt_install \
  xrdp \
  xfce4 \
  xfce4-terminal \
  dbus-x11

# ── ~/.xsession launches xfce4 when xrdp opens a session ─────────────────────
SESSION_FILE="${USER_HOME}/.xsession"
if [ ! -f "$SESSION_FILE" ] || ! grep -qx "startxfce4" "$SESSION_FILE"; then
  echo "[install_rdp] writing ${SESSION_FILE} -> startxfce4"
  printf 'startxfce4\n' > "$SESSION_FILE"
  chown "$NODE_USER:$NODE_USER" "$SESSION_FILE"
  chmod 0755 "$SESSION_FILE"
fi

# ── xrdp.ini sanity (regression in HANDOFF doc) ──────────────────────────────
# A previous edit on vault accidentally set every ``port=`` line to 3389,
# which made xrdp connect to itself and produce a black RDP session. Be
# defensive: keep [Globals] at 3389; force [Xorg]/[Xvnc] to -1.
XRDP_INI=/etc/xrdp/xrdp.ini
if [ -f "$XRDP_INI" ]; then
  python3 - "$XRDP_INI" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines()
section = None
changed = False
for i, line in enumerate(lines):
    m = re.match(r"^\[([^\]]+)\]\s*$", line)
    if m:
        section = m.group(1)
        continue
    if line.strip().startswith("port="):
        want = "3389" if section == "Globals" else ("-1" if section in {"Xorg", "Xvnc"} else None)
        if want is not None and line.strip() != f"port={want}":
            lines[i] = f"port={want}"
            changed = True
if changed:
    path.write_text("\n".join(lines) + "\n")
    print("[install_rdp] normalized port= entries in xrdp.ini")
PY
fi

# ── allow xrdp users to start X ─────────────────────────────────────────────
# Debian/Pi OS's X wrapper script defaults to ``allowed_users=console``,
# which blocks any non-console user (xrdp sessions count as non-console)
# from opening /dev/tty0. Without this file Xorg fails with
# ``parse_vt_settings: Cannot open /dev/tty0 (Permission denied)`` and the
# RDP session dies at "creating session - X server could not be started".
XWRAPPER=/etc/X11/Xwrapper.config
if [ ! -f "$XWRAPPER" ] || ! grep -q '^allowed_users=anybody' "$XWRAPPER"; then
  echo "[install_rdp] writing ${XWRAPPER} (allowed_users=anybody)"
  install -d -m 0755 /etc/X11
  cat > "$XWRAPPER" <<'EOF'
allowed_users=anybody
needs_root_rights=yes
EOF
  chmod 0644 "$XWRAPPER"
fi

# ── enable + start xrdp services ────────────────────────────────────────────
systemctl enable --now xrdp.service
systemctl enable --now xrdp-sesman.service

# ── tailscale-only firewall (matches vault pattern) ─────────────────────────
# Only allow inbound TCP 3389 from the tailnet. LAN clients can't connect
# unless you explicitly punch a hole — by design.
FIREWALL_UNIT=/etc/systemd/system/luhkas-rdp-tailscale-only.service
if [ ! -f "$FIREWALL_UNIT" ]; then
  cat > "$FIREWALL_UNIT" <<'EOF'
[Unit]
Description=Restrict xrdp (TCP 3389) to the tailscale0 interface
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  /usr/sbin/iptables -C INPUT -p tcp --dport 3389 -i tailscale0 -j ACCEPT 2>/dev/null \
    || /usr/sbin/iptables -I INPUT -p tcp --dport 3389 -i tailscale0 -j ACCEPT; \
  /usr/sbin/iptables -C INPUT -p tcp --dport 3389 -j DROP 2>/dev/null \
    || /usr/sbin/iptables -A INPUT -p tcp --dport 3389 -j DROP'
ExecStop=/bin/bash -c '\
  /usr/sbin/iptables -D INPUT -p tcp --dport 3389 -i tailscale0 -j ACCEPT 2>/dev/null || true; \
  /usr/sbin/iptables -D INPUT -p tcp --dport 3389 -j DROP 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
fi
systemctl enable --now luhkas-rdp-tailscale-only.service 2>/dev/null \
  || echo "[install_rdp] WARN: firewall unit start deferred (tailscale0 may come up later)"

echo "[install_rdp] done — connect to ${NODE_USER}@<tailnet-name> port 3389"
