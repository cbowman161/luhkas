"""Generic pre-vault deterministic router for LUHKAS nodes.

Reusable packages contribute read-only mappings with this convention:

    *_node/deterministic_mappings.json

The node runtime overlays a writable learned cache on top of those package
mappings. This lets a composed node route obvious local commands before sending
anything to the vault.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path


_NODE_DIR = Path(__file__).parent
_RUNTIME_ROOT = _NODE_DIR.parent
_DEFAULT_CACHE = _NODE_DIR / "data" / "deterministic_commands.json"
_CACHE_PATH = Path(os.environ.get("LUHKAS_DETERMINISTIC_CACHE", str(_DEFAULT_CACHE))).expanduser()
if not _CACHE_PATH.is_absolute():
    _CACHE_PATH = _RUNTIME_ROOT / _CACHE_PATH
_MAPPING_FILENAME = "deterministic_mappings.json"
_lock = threading.Lock()
_cache: dict | None = None
_cache_signature: tuple | None = None


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _mapping_paths() -> list[Path]:
    return sorted(
        path for path in _RUNTIME_ROOT.glob(f"*_node/{_MAPPING_FILENAME}")
        if path.is_file()
    )


def _signature(paths: list[Path]) -> tuple:
    signature = []
    for path in [*paths, _CACHE_PATH]:
        try:
            signature.append((str(path), path.stat().st_mtime))
        except OSError:
            signature.append((str(path), None))
    return tuple(signature)


def _load_json_mapping(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    normalized = {}
    for phrase, entry in data.items():
        key = _normalize(str(phrase))
        if not key:
            continue
        if isinstance(entry, str):
            normalized[key] = {"type": entry}
        elif isinstance(entry, dict) and entry.get("type"):
            normalized[key] = dict(entry)
    return normalized


def _load_package_mappings(paths: list[Path]) -> dict:
    mappings = {}
    for path in paths:
        mappings.update(_load_json_mapping(path))
    return mappings


def _load() -> dict:
    global _cache, _cache_signature
    try:
        package_paths = _mapping_paths()
        signature = _signature(package_paths)
        if _cache is not None and signature == _cache_signature:
            return _cache
        data = _load_package_mappings(package_paths)
        data.update(_load_json_mapping(_CACHE_PATH))
        _cache = data
        _cache_signature = signature
        return data
    except Exception:
        return {}


def _save(store: dict) -> None:
    global _cache, _cache_signature
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(_CACHE_PATH)
    _cache = None
    _cache_signature = None


def _load_learned() -> dict:
    return _load_json_mapping(_CACHE_PATH)


def lookup(text: str) -> str | None:
    key = _normalize(text)
    if not key:
        return None
    with _lock:
        store = _load()
    entry = store.get(key)
    if entry is None:
        return None
    threading.Thread(target=_hit, args=(key,), daemon=True).start()
    return entry["type"]


def learn(text: str, dispatch_type: str) -> None:
    key = _normalize(text)
    if not key:
        return
    with _lock:
        store = _load_learned()
        if key not in store:
            store[key] = {"type": dispatch_type, "hits": 0}
            _save(store)


def _hit(key: str) -> None:
    with _lock:
        store = _load_learned()
        if key in store:
            store[key]["hits"] = store[key].get("hits", 0) + 1
            _save(store)
