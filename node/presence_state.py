"""Small local state bus for node presence surfaces.

Services on the same node run as separate user systemd units. This module gives
them a shared, low-latency JSON state file without introducing another daemon.
It is intentionally best-effort: display animation should improve when this is
available, but no service should fail because the state file cannot be written.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import fcntl
from pathlib import Path
from typing import Any


def _state_path() -> Path:
    configured = os.environ.get("LUHKAS_PRESENCE_STATE_FILE")
    if configured:
        return Path(configured).expanduser()
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    node_id = os.environ.get("LUHKAS_NODE_ID", "node")
    return Path(runtime_dir) / f"luhkas-presence-{node_id}.json"


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_state(max_age_seconds: float | None = None) -> dict[str, Any]:
    path = _state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if max_age_seconds is not None:
        updated_at = float(payload.get("updated_at") or 0.0)
        if updated_at and time.time() - updated_at > max_age_seconds:
            return {}
    return payload


def update_state(patch: dict[str, Any]) -> None:
    if not isinstance(patch, dict):
        return
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_path(path).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(current, dict):
                        current = {}
                except Exception:
                    current = {}
                now = time.time()
                payload = _deep_merge(current, patch)
                payload["updated_at"] = now
                payload.setdefault("node_id", os.environ.get("LUHKAS_NODE_ID", "node"))
                fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                        json.dump(payload, tmp, separators=(",", ":"), sort_keys=True)
                    os.replace(tmp_name, path)
                finally:
                    try:
                        if os.path.exists(tmp_name):
                            os.unlink(tmp_name)
                    except OSError:
                        pass
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except Exception:
        return
