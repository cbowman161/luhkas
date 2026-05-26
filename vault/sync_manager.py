"""Pull from git and push node updates to registered node devices."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILES_DIR = _REPO_ROOT / "node" / "profiles"
_NODE_DIR = _REPO_ROOT / "node"

# Use the canonical profile loader so sync.host / user / node_dir / services
# are filled in from node_id + modules when the profile doesn't spell them out.
if str(_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(_NODE_DIR))
from profile_loader import load_profile as _load_profile  # noqa: E402
_SECRETS_DIR = _REPO_ROOT / "vault" / "secrets"
_TAILSCALE_AUTHKEY_FILE = _SECRETS_DIR / "tailscale.authkey"
_SSH_KEY = Path.home() / ".ssh" / "id_ed25519_nodes"

_SSH_OPTS = [
    "-i", str(_SSH_KEY),
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]

_last_result: dict = {}
_last_sync_at: float = 0.0
_auto_synced: set[str] = set()  # nodes pushed this session; reset on vault restart


def pubkey() -> str:
    """Return the vault's node-sync SSH public key, or empty string if missing."""
    try:
        return (_SSH_KEY.parent / (_SSH_KEY.name + ".pub")).read_text().strip()
    except Exception:
        return ""


def pull() -> dict:
    """Pull latest commits from the remote on the vault's own repo."""
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (result.stdout + result.stderr).strip()
    changed = result.returncode == 0 and "Already up to date." not in output
    return {
        "ok": result.returncode == 0,
        "changed": changed,
        "output": output,
    }


def push_node(profile: dict) -> dict:
    """Rsync node/ to one node device and restart its services."""
    sync = profile.get("sync") or {}
    host = sync.get("host", "")
    user = sync.get("user", "luhkas")
    node_dir = sync.get("node_dir", "luhkas/node")
    services = sync.get("services") or []
    node_id = profile.get("node_id", "?")

    if not host:
        return {"ok": False, "error": "no sync.host configured in profile"}

    dest = f"{user}@{host}:{node_dir}/"
    rsync = subprocess.run(
        [
            "rsync", "-a", "--delete", "--itemize-changes",
            # Build artifacts and editor cruft (never canonical)
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--exclude=._*",
            "--exclude=.DS_Store",
            # Logs (runtime, per-node)
            "--exclude=*.log",
            "--exclude=*.log.*",
            # Runtime data that lives only on the node — must be preserved
            # across syncs, never replaced from the vault's canonical tree.
            "--exclude=captures/",
            "--exclude=config/faces/",
            "--exclude=config/people/",
            "--exclude=config/vault_faces/",
            "--exclude=config/unknown_faces/",
            "--exclude=config/brain_faces/",
            "--exclude=config/telemetry.db",
            "--exclude=config/telemetry.db-shm",
            "--exclude=config/telemetry.db-wal",
            "--exclude=luhkas_node/data/",
            # Node-specific deterministic-command overrides (carried per-node).
            "--exclude=data/deterministic_commands.json",
            "-e", "ssh " + " ".join(_SSH_OPTS),
            str(_NODE_DIR) + "/",
            dest,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if rsync.returncode != 0:
        return {"ok": False, "node_id": node_id, "error": rsync.stderr.strip()}

    files_changed = bool(rsync.stdout.strip())
    restarted: list[str] = []
    if files_changed and services:
        restart = subprocess.run(
            ["ssh"] + _SSH_OPTS + [f"{user}@{host}", f"systemctl --user restart {' '.join(services)}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if restart.returncode != 0:
            return {
                "ok": False,
                "node_id": node_id,
                "error": f"service restart failed: {restart.stderr.strip()}",
            }
        restarted = services

    return {
        "ok": True,
        "node_id": node_id,
        "host": host,
        "files_changed": files_changed,
        "services_restarted": restarted,
    }


def push_tailscale_authkey(profile: dict) -> dict:
    """Copy the vault's current Tailscale auth key to one node."""
    sync = profile.get("sync") or {}
    host = sync.get("host", "")
    user = sync.get("user", "luhkas")
    node_id = profile.get("node_id", "?")

    if not host:
        return {"ok": False, "node_id": node_id, "error": "no sync.host configured in profile"}
    if not _TAILSCALE_AUTHKEY_FILE.exists():
        return {
            "ok": False,
            "node_id": node_id,
            "error": f"missing auth key file: {_TAILSCALE_AUTHKEY_FILE}",
        }

    key = _TAILSCALE_AUTHKEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        return {"ok": False, "node_id": node_id, "error": "auth key file is empty"}

    node_id_q = node_id.replace("'", "'\\''")
    remote = (
        "set -euo pipefail; "
        "install -m 700 -d ~/.config/luhkas; "
        "tmp=$(mktemp); "
        "cat > \"$tmp\"; "
        "install -m 600 \"$tmp\" ~/.config/luhkas/tailscale.authkey; "
        "rm -f \"$tmp\"; "
        f"cat > ~/.config/luhkas/bootstrap.env <<EOF\n"
        f"export LUHKAS_NODE_ID='{node_id_q}'\n"
        "export TAILSCALE_AUTHKEY_FILE=\"$HOME/.config/luhkas/tailscale.authkey\"\n"
        "export LUHKAS_TAILSCALE=1\n"
        "export LUHKAS_PREFER_TAILSCALE=1\n"
        "EOF\n"
        "chmod 600 ~/.config/luhkas/bootstrap.env"
    )
    result = subprocess.run(
        ["ssh"] + _SSH_OPTS + [f"{user}@{host}", remote],
        input=key + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {"ok": False, "node_id": node_id, "host": host, "error": result.stderr.strip()}
    return {"ok": True, "node_id": node_id, "host": host, "installed": True}


def push_tailscale_authkeys(node_id: str | None = None) -> dict:
    """Push the current Tailscale auth key secret to all configured nodes."""
    nodes: dict[str, dict] = {}
    for profile_path in sorted(p for p in _PROFILES_DIR.glob("*.json") if not p.name.startswith(".")):
        try:
            profile = _load_profile(profile_path)
        except Exception as exc:
            nodes[profile_path.stem] = {"ok": False, "error": f"bad profile: {exc}"}
            continue
        nid = profile.get("node_id", profile_path.stem)
        if node_id and nid != node_id:
            continue
        if not profile.get("sync"):
            continue
        nodes[nid] = push_tailscale_authkey(profile)
    return {
        "ok": all(v.get("ok") for v in nodes.values()) if nodes else True,
        "nodes": nodes,
        "synced_at": time.time(),
    }


def provision_tailscale_for_node(
    *,
    node_id: str,
    host: str,
    user: str = "luhkas",
    node_dir: str = "luhkas/node",
) -> dict:
    """Install the current Tailscale auth key on a registered node and run setup.

    This is called after /node/register. The node already fetched the vault's
    SSH public key during registration, so the vault can push the secret over
    SSH without exposing the auth key through the HTTP API.
    """
    node_id = str(node_id or "").strip()
    host = str(host or "").strip()
    user = str(user or "luhkas").strip()
    node_dir = str(node_dir or "luhkas/node").strip().strip("/")
    if not node_id:
        return {"ok": False, "error": "missing node_id"}
    if not host:
        return {"ok": False, "node_id": node_id, "error": "missing host"}
    if not _TAILSCALE_AUTHKEY_FILE.exists():
        return {
            "ok": False,
            "node_id": node_id,
            "host": host,
            "error": f"missing auth key file: {_TAILSCALE_AUTHKEY_FILE}",
        }
    key = _TAILSCALE_AUTHKEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        return {"ok": False, "node_id": node_id, "host": host, "error": "auth key file is empty"}

    hostname = f"luhkas-{node_id}"
    remote = (
        "set -euo pipefail; "
        "install -m 700 -d ~/.config/luhkas; "
        "tmp=$(mktemp); "
        "cat > \"$tmp\"; "
        "install -m 600 \"$tmp\" ~/.config/luhkas/tailscale.authkey; "
        "rm -f \"$tmp\"; "
        f"cat > ~/.config/luhkas/bootstrap.env <<EOF\n"
        f"export LUHKAS_NODE_ID='{node_id}'\n"
        "export TAILSCALE_AUTHKEY_FILE=\"$HOME/.config/luhkas/tailscale.authkey\"\n"
        "export LUHKAS_TAILSCALE=1\n"
        "export LUHKAS_PREFER_TAILSCALE=1\n"
        "EOF\n"
        "chmod 600 ~/.config/luhkas/bootstrap.env; "
        f"if [ -x ~/{node_dir}/scripts/setup_tailscale.sh ]; then "
        f"LUHKAS_NODE_ID={node_id!r} TAILSCALE_HOSTNAME={hostname!r} "
        f"~/{node_dir}/scripts/setup_tailscale.sh; "
        "else echo 'setup_tailscale.sh not found' >&2; exit 2; fi"
    )
    result = subprocess.run(
        ["ssh"] + _SSH_OPTS + [f"{user}@{host}", remote],
        input=key + "\n",
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "node_id": node_id,
            "host": host,
            "error": (result.stderr or result.stdout).strip(),
        }
    return {
        "ok": True,
        "node_id": node_id,
        "host": host,
        "hostname": hostname,
        "output": result.stdout.strip(),
    }


def push_all(node_id: str | None = None) -> dict:
    """Rsync node/ to every node with a sync profile, without pulling from git.

    Intended for the periodic auto-sync timer: cheap to run (rsync no-ops when
    nothing has changed, and push_node only restarts services when files
    actually changed). Pass *node_id* to limit to a single node.
    """
    global _last_result, _last_sync_at

    nodes: dict[str, dict] = {}

    for profile_path in sorted(p for p in _PROFILES_DIR.glob("*.json") if not p.name.startswith(".")):
        try:
            profile = _load_profile(profile_path)
        except Exception as exc:
            nodes[profile_path.stem] = {"ok": False, "error": f"bad profile: {exc}"}
            continue

        nid = profile.get("node_id", profile_path.stem)
        if node_id and nid != node_id:
            continue
        if not profile.get("sync"):
            continue

        nodes[nid] = push_node(profile)

    all_nodes_ok = all(v.get("ok") for v in nodes.values()) if nodes else True
    result = {
        "ok": all_nodes_ok,
        "pull": None,
        "nodes": nodes,
        "synced_at": time.time(),
    }
    _last_result = result
    _last_sync_at = result["synced_at"]
    return result


def sync_all(node_id: str | None = None) -> dict:
    """Pull from git then push to all nodes (or just *node_id* if given)."""
    global _last_result, _last_sync_at

    pull_result = pull()
    nodes: dict[str, dict] = {}

    for profile_path in sorted(p for p in _PROFILES_DIR.glob("*.json") if not p.name.startswith(".")):
        try:
            profile = _load_profile(profile_path)
        except Exception as exc:
            nodes[profile_path.stem] = {"ok": False, "error": f"bad profile: {exc}"}
            continue

        nid = profile.get("node_id", profile_path.stem)
        if node_id and nid != node_id:
            continue
        if not profile.get("sync"):
            continue

        nodes[nid] = push_node(profile)

    all_nodes_ok = all(v.get("ok") for v in nodes.values()) if nodes else True
    result = {
        "ok": pull_result["ok"] and all_nodes_ok,
        "pull": pull_result,
        "nodes": nodes,
        "synced_at": time.time(),
    }
    _last_result = result
    _last_sync_at = result["synced_at"]
    return result


def auto_push_if_new(node_id: str) -> None:
    """Push to node_id once per vault session (called on node registration).

    Skips nodes already pushed this session and nodes with no sync profile.
    Runs rsync but only restarts services if files actually changed.
    """
    if node_id in _auto_synced:
        return
    profile_path = _PROFILES_DIR / f"{node_id}.json"
    if not profile_path.exists():
        return
    try:
        profile = _load_profile(profile_path)
    except Exception:
        return
    if not profile.get("sync"):
        return
    _auto_synced.add(node_id)
    result = push_node(profile)
    restarted = result.get("services_restarted", [])
    status = "ok" if result.get("ok") else f"failed: {result.get('error')}"
    print(
        f"[sync] auto-push to {node_id}: {status}"
        + (f" | {len(restarted)} service(s) restarted" if restarted else ""),
        flush=True,
    )
    global _last_result, _last_sync_at
    import time as _time
    _last_sync_at = _time.time()
    _last_result = {
        "ok": result.get("ok"),
        "trigger": "auto",
        "pull": None,
        "nodes": {node_id: result},
        "synced_at": _last_sync_at,
    }


def last_result() -> dict:
    return {**_last_result, "last_sync_at": _last_sync_at} if _last_result else {
        "ok": None,
        "last_sync_at": None,
        "message": "no sync has run yet",
    }
