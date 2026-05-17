"""Pull from git and push node updates to registered node devices."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILES_DIR = _REPO_ROOT / "node" / "profiles"
_NODE_DIR = _REPO_ROOT / "node"
_SSH_KEY = Path.home() / ".ssh" / "id_ed25519_nodes"

_SSH_OPTS = [
    "-i", str(_SSH_KEY),
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]

_last_result: dict = {}
_last_sync_at: float = 0.0


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
            "rsync", "-a", "--delete",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--exclude=*.db",
            "--exclude=._*",
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

    restarted: list[str] = []
    if services:
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

    return {"ok": True, "node_id": node_id, "host": host, "services_restarted": restarted}


def sync_all(node_id: str | None = None) -> dict:
    """Pull from git then push to all nodes (or just *node_id* if given)."""
    global _last_result, _last_sync_at

    pull_result = pull()
    nodes: dict[str, dict] = {}

    for profile_path in sorted(p for p in _PROFILES_DIR.glob("*.json") if not p.name.startswith(".")):
        try:
            profile = json.loads(profile_path.read_text())
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


def last_result() -> dict:
    return {**_last_result, "last_sync_at": _last_sync_at} if _last_result else {
        "ok": None,
        "last_sync_at": None,
        "message": "no sync has run yet",
    }
