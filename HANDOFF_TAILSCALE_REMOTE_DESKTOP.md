# LUHKAS Vault Handoff: Tailscale, Node Provisioning, Remote Desktop

Date: 2026-05-25

## Goal

Make LUHKAS remote nodes private/autonomous through Tailscale, with the vault owning node auth-key distribution/rotation, and provide a private remote desktop path to the vault PC.

## Current State

- Vault PC: `luhkas-vault`
- Vault Tailscale IP: `100.70.245.116`
- Scout node: `luhkas-scout`
- Scout Tailscale IP: `100.112.87.59`
- Tailscale MagicDNS works from the vault: `luhkas-scout` resolves to `100.112.87.59`
- Scout registered with vault using preferred network `tailscale`
- Domain: `luhkas-vault.net`
- Email: `admin@luhkas-vault.net` works via Proton Mail

## Tailscale Auth-Key Automation

Vault owns the node auth key.

Secret files on vault:

- `~/luhkas/vault/secrets/tailscale.authkey`
- `~/luhkas/vault/secrets/tailscale_oauth.env`
- `~/luhkas/vault/secrets/tailscale_authkey_state.json`

These are ignored by git via `vault/secrets/`.

Installed vault scripts:

- `~/luhkas/scripts/update_tailscale_authkey.sh`
- `~/luhkas/scripts/update_tailscale_oauth_credentials.sh`
- `~/luhkas/scripts/rotate_tailscale_authkey.sh`
- `~/luhkas/scripts/rotate_tailscale_authkey.py`
- `~/luhkas/scripts/rotate_tailscale_authkey_if_needed.py`

Manual rotation:

```bash
~/luhkas/scripts/rotate_tailscale_authkey.sh
```

Daily auto-rotation:

- User systemd timer: `tailscale-authkey-rotate.timer`
- Runs daily at `03:20`
- Rotates only if current key expires within 24 hours

Check timer:

```bash
systemctl --user list-timers tailscale-authkey-rotate.timer
```

Current generated key state after setup:

- Key id: `kpnRFSkVS621CNTRL`
- Expires: `2026-08-24T00:02:40Z`
- Tag: `tag:luhkas-node`
- Reusable/preauthorized/non-ephemeral

## Node Provisioning Flow

Desired autonomous flow:

1. Node starts/restarts presence service.
2. Node registers with vault over LAN or existing network.
3. Vault receives `/node/register`.
4. Vault SSHes back to node.
5. Vault writes `~/.config/luhkas/tailscale.authkey`.
6. Vault writes `~/.config/luhkas/bootstrap.env`.
7. Vault runs node `setup_tailscale.sh`.
8. Node joins Tailscale automatically.
9. Node re-registers using Tailscale IP.

Important code paths:

- `vault/vault_service.py`
  - calls `_provision_tailscale_after_register(...)` after `/node/register`
- `vault/sync_manager.py`
  - `provision_tailscale_for_node(...)`
  - `push_tailscale_authkey(...)`
  - `push_tailscale_authkeys(...)`
- `node/scripts/setup_tailscale.sh`
  - supports `TAILSCALE_AUTHKEY_FILE`
  - skips interactive login if no key is present
  - uses `--reset --force-reauth` if node is stuck in `NeedsLogin`
- `scripts/bootstrap_node.sh` and `node/scripts/install_user_services.sh`
  - load `~/.config/luhkas/bootstrap.env` if present

## Scout Over Tailscale

Verified from vault:

```bash
curl http://100.112.87.59:5002/health
curl http://100.112.87.59:5001/health
curl http://100.112.87.59:5000/health
```

All responded OK.

Vault runtime now uses MagicDNS defaults:

- `VAULT_SCOUT_URL=http://luhkas-scout:5000`
- `VAULT_SCOUT_ROBOT_URL=http://luhkas-scout:5001`

Files changed for MagicDNS:

- `node/profiles/scout.json`
- `vault/config.py`
- `vault/systemd/vault-runtime.service`
- `vault/VAULT_RUNTIME.md`

Verified node sync over Tailscale:

```bash
cd ~/luhkas/vault
python3 -c 'from sync_manager import push_all; import json; print(json.dumps(push_all(node_id="scout"), indent=2))'
```

Result OK with host `luhkas-scout`.

## Remote Desktop Status

GNOME Remote Desktop initially connected but gave black screen due to GNOME RDP redirection/session behavior.

Switched to:

- `xrdp`
- `xfce4`

Configured:

- `~/.xsession` contains `startxfce4`
- `xrdp` active
- `xrdp-sesman` active
- GNOME Remote Desktop disabled
- RDP target: `100.70.245.116:3389`
- Login: `luhkas` / `luhkas`

Firewall:

- Persistent service: `/etc/systemd/system/luhkas-rdp-tailscale-only.service`
- Drops TCP `3389` unless interface is `tailscale0`
- Tailscale path tested open
- LAN path tested blocked

Important RDP bug fixed:

Earlier, all `port=` lines in `/etc/xrdp/xrdp.ini` were accidentally changed to `3389`, including `[Xorg]` and `[Xvnc]`, causing xrdp to connect to itself and show a black screen.

Fixed values:

```text
[Globals] port=3389
[Xorg]    port=-1
[Xvnc]    port=-1
```

Verified after fix:

```text
xrdp-sesman active
xrdp active
listen: *:3389
listen: [::1]:3350
```

Next step:

- User should fully close the black RDP session/app and reconnect to `100.70.245.116:3389`.
- If still black, inspect:
  - `/var/log/xrdp.log`
  - `/var/log/xrdp-sesman.log`
  - `~/.xsession-errors`
  - `ps -ef | grep -E "xrdp|Xorg|xfce|startxfce"`

## Validation Already Run

```bash
bash -n node/scripts/setup_tailscale.sh scripts/bootstrap_node.sh node/scripts/install_user_services.sh
python3 -m py_compile vault/sync_manager.py vault/vault_service.py node/services/presence_client_service.py node/luhkas_node/node.py
python3 -m unittest tests.node_registry_test tests.smoke_api_test
```

Tests passed:

```text
Ran 5 tests
OK (skipped=1)
```

## Caveats

- Do not print or paste secret auth keys.
- Existing joined Tailscale nodes do not need the auth key unless reset/rebuilt/logged out.
- Auto-rotation is mainly for future registrations/re-provisioning.
- Repo had unrelated existing dirty files. Do not revert unrelated changes.

