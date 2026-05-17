"""Fast-path routing cache.

Stores normalized input → route mappings so the routing LLM is skipped for
inputs the system has already classified. Written by the system after each
successful LLM route; readable and editable as plain JSON.
"""
from __future__ import annotations

import json
import re
import threading
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
    return entry["route"]


def learn(text: str, route: dict) -> None:
    """Persist a text → route mapping after a successful LLM classification."""
    key = _normalize(text)
    if not key:
        return
    with _lock:
        store = _load()
        if key not in store:
            store[key] = {"route": route, "hits": 0}
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
            _save(store)
