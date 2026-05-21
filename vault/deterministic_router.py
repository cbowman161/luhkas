"""Confirmed fast-path routing cache.

Stores normalized input → route mappings so the routing LLM is skipped for
inputs the system has already classified and the user has confirmed. Each
entry also keeps the examples that created or reinforced that route.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

_PATH = Path(__file__).parent / "data" / "deterministic_routes.json"
_lock = threading.Lock()
_cache: dict | None = None
_cache_mtime: float = 0.0


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _load() -> dict:
    global _cache, _cache_mtime
    try:
        mtime = _PATH.stat().st_mtime
        if _cache is not None and mtime == _cache_mtime:
            return _cache
        data = json.loads(_PATH.read_text())
        _cache = data
        _cache_mtime = mtime
        return data
    except Exception:
        return {}


def _save(store: dict) -> None:
    global _cache, _cache_mtime
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True))
    tmp.replace(_PATH)
    _cache = store
    _cache_mtime = _PATH.stat().st_mtime


def lookup(text: str) -> dict | None:
    """Return a cached route dict if this input has been seen before, else None."""
    key = _normalize(text)
    if not key:
        return None
    with _lock:
        store = _load()
    entry = store.get(key)
    if entry is None:
        return None
    threading.Thread(target=_hit, args=(key,), daemon=True).start()
    route = entry.get("route") or {}
    if not isinstance(route, dict):
        return None
    return dict(route)


def learn(text: str, route: dict, *, confirmed_by: str = "user_confirmation") -> None:
    """Persist a text → route mapping after explicit user confirmation."""
    key = _normalize(text)
    if not key:
        return
    route_copy = _route_snapshot(route)
    example = {
        "input": str(text or ""),
        "normalized_input": key,
        "decided_route": route_copy.get("route"),
        "self_route": (route_copy.get("self_route") or {}).get("route")
        if isinstance(route_copy.get("self_route"), dict)
        else route_copy.get("self_route"),
        "confidence": route_copy.get("confidence"),
        "reason": route_copy.get("reason"),
        "confirmed_by": confirmed_by,
        "confirmed_at": time.time(),
    }
    with _lock:
        store = _load()
        entry = store.get(key) or {}
        examples = entry.get("examples") if isinstance(entry.get("examples"), list) else []
        examples.append(example)
        store[key] = {
            "input": str(text or ""),
            "normalized_input": key,
            "route": route_copy,
            "decided_route": route_copy.get("route"),
            "self_route": example["self_route"],
            "confidence": route_copy.get("confidence"),
            "reason": route_copy.get("reason"),
            "confirmed_by": confirmed_by,
            "confirmed_at": example["confirmed_at"],
            "hits": int(entry.get("hits") or 0),
            "examples": examples[-20:],
        }
        _save(store)


def unlearn(text: str) -> bool:
    """Remove a learned mapping. Returns True if an entry was deleted."""
    key = _normalize(text)
    if not key:
        return False
    with _lock:
        store = _load()
        if key not in store:
            return False
        del store[key]
        _save(store)
    return True


def _hit(key: str) -> None:
    with _lock:
        store = _load()
        if key in store:
            store[key]["hits"] = store[key].get("hits", 0) + 1
            store[key]["last_hit_at"] = time.time()
            _save(store)


def entries() -> dict:
    """Return the raw learned route store for diagnostics and tests."""
    with _lock:
        return dict(_load())


def _route_snapshot(route: dict) -> dict:
    result = {
        "ok": bool(route.get("ok", True)),
        "route": route.get("route"),
        "confidence": route.get("confidence"),
        "reason": route.get("reason"),
        "attempts": route.get("attempts", 0),
    }
    if isinstance(route.get("self_route"), dict):
        result["self_route"] = {
            "ok": bool(route["self_route"].get("ok", True)),
            "route": route["self_route"].get("route"),
            "confidence": route["self_route"].get("confidence"),
            "reason": route["self_route"].get("reason"),
            "attempts": route["self_route"].get("attempts", 0),
        }
    for key in ("request_owner", "target_node", "explicit_target", "keywords"):
        if key in route:
            result[key] = route[key]
    return result
