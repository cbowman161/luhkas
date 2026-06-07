"""Confirmed fast-path routing cache.

Stores normalized input → route mappings so the routing LLM is skipped for
inputs the system has already classified and the user has confirmed. Each
entry also keeps the examples that created or reinforced that route.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

try:
    from models import get_model
    from semantic_route_store import SemanticRouteStore
except Exception:  # pragma: no cover - optional semantic sidecar
    get_model = None
    SemanticRouteStore = None

_PATH = Path(__file__).parent / "data" / "deterministic_routes.json"
_lock = threading.Lock()
_cache: dict | None = None
_cache_mtime: float = 0.0
_semantic_store = None
_semantic_signature: tuple[float, int] | None = None
_VECTOR_DISTANCE_MAX = float(os.environ.get("DETERMINISTIC_ROUTE_VECTOR_DISTANCE_MAX", "0.14"))


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
    """Return a cached route dict if this input has been seen before.

    Exact normalized lookup wins. If that misses, try the LanceDB semantic
    sidecar built from confirmed routes; the returned route is still the
    confirmed route snapshot, never an invented route.
    """
    key = _normalize(text)
    if not key:
        return None
    with _lock:
        store = _load()
    entry = store.get(key)
    if entry is not None:
        threading.Thread(target=_hit, args=(key,), daemon=True).start()
        route = entry.get("route") or {}
        if not isinstance(route, dict):
            return None
        return dict(route)
    return semantic_lookup(text, store=store)


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
        _semantic_upsert(key, store[key])


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


def semantic_lookup(text: str, *, store: dict | None = None) -> dict | None:
    sem = _semantic_store_instance()
    if sem is None:
        return None
    store = store if store is not None else _load()
    _sync_semantic_store(sem, store)
    try:
        hits = sem.search_deterministic_routes(text, top_k=3)
    except Exception as exc:
        print(f"[deterministic_router] semantic lookup failed: {exc}", flush=True)
        return None
    for hit in hits:
        dist = hit.get("distance")
        if dist is None or float(dist) > _VECTOR_DISTANCE_MAX:
            continue
        key = str(hit.get("normalized_input") or hit.get("id") or "")
        entry = store.get(key)
        if not isinstance(entry, dict):
            continue
        route = entry.get("route") or {}
        if not isinstance(route, dict):
            continue
        result = dict(route)
        result["from_vector_cache"] = True
        result["vector_match"] = {"normalized_input": key, "distance": float(dist)}
        threading.Thread(target=_hit, args=(key,), daemon=True).start()
        return result
    return None


def _semantic_store_instance():
    global _semantic_store
    if _semantic_store is False:
        return None
    if _semantic_store is not None:
        return _semantic_store
    if SemanticRouteStore is None or get_model is None:
        return None
    try:
        _semantic_store = SemanticRouteStore(embedder=get_model("embed"))
    except Exception as exc:
        print(f"[deterministic_router] semantic index disabled: {exc}", flush=True)
        _semantic_store = False
    return _semantic_store or None


def _sync_semantic_store(sem, store: dict) -> None:
    global _semantic_signature
    try:
        signature = (_PATH.stat().st_mtime if _PATH.exists() else 0.0, len(store))
        if signature == _semantic_signature:
            return
        for key, entry in store.items():
            if isinstance(entry, dict):
                sem.upsert_deterministic_route(str(key), entry)
        _semantic_signature = signature
    except Exception as exc:
        print(f"[deterministic_router] semantic sync failed: {exc}", flush=True)


def _semantic_upsert(key: str, entry: dict) -> None:
    sem = _semantic_store_instance()
    if sem is None:
        return
    try:
        sem.upsert_deterministic_route(key, entry)
    except Exception as exc:
        print(f"[deterministic_router] semantic upsert failed: {exc}", flush=True)



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
