"""Generic local command composition for LUHKAS nodes."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

try:
    from .deterministic import learn as _det_learn
    from .deterministic import lookup as _det_lookup
    from .routing_intent import classify_request_target as _classify_request_target
    from .routing_intent import should_attempt_local as _should_attempt_local
except ImportError:
    from deterministic import learn as _det_learn
    from deterministic import lookup as _det_lookup
    from routing_intent import classify_request_target as _classify_request_target
    from routing_intent import should_attempt_local as _should_attempt_local


_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

_PROFILES_DIR = _RUNTIME_ROOT / "profiles"
_FALLBACK_PACKAGES = "camera_node,pantilt_node,rover_node,light_node"


def _load_profile_modules(node_id: str) -> list[str]:
    """Return the module list for *node_id* from its profile file.

    Falls back to LUHKAS_NODE_PACKAGES env var, then the built-in default.
    """
    if node_id:
        profile_path = _PROFILES_DIR / f"{node_id}.json"
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text())
                modules = [m.strip() for m in data.get("modules", []) if m.strip()]
                if modules:
                    return modules
            except Exception:
                pass
    return [
        name.strip()
        for name in os.environ.get("LUHKAS_NODE_PACKAGES", _FALLBACK_PACKAGES).split(",")
        if name.strip()
    ]


def _all_known_modules() -> list[str]:
    """Union of all modules declared across every profile in the profiles dir."""
    modules: set[str] = set()
    if _PROFILES_DIR.exists():
        for path in _PROFILES_DIR.glob("*.json"):
            if path.name.startswith("."):  # skip ._* macOS resource forks
                continue
            try:
                data = json.loads(path.read_text())
                modules.update(m.strip() for m in data.get("modules", []) if m.strip())
            except Exception:
                pass
    return sorted(modules)


_NODE_ID = os.environ.get("LUHKAS_NODE_ID", "")
_PACKAGE_NAMES: list[str] = _load_profile_modules(_NODE_ID)
_KNOWN_NODE_PACKAGES: list[str] = _all_known_modules() or _PACKAGE_NAMES

_handlers: dict[str, object] | None = None
_module_status: dict[str, dict | None] | None = None


def _load_handlers() -> dict[str, object]:
    global _handlers, _module_status
    if _handlers is not None:
        return _handlers
    handlers = {}
    status: dict[str, dict | None] = {name: None for name in _PACKAGE_NAMES}
    for package_name in sorted(_PACKAGE_NAMES):
        try:
            module = importlib.import_module(f"{package_name}.commands")
        except Exception:
            status[package_name] = None
            continue
        caps = _module_capabilities(module)
        status[package_name] = {"available": True, "commands": len(caps)}
        for command in caps:
            dispatch_type = command.get("dispatch_type")
            if dispatch_type and hasattr(module, "handle"):
                handlers.setdefault(str(dispatch_type), module)
    _handlers = handlers
    _module_status = status
    return handlers


def _module_capabilities(module) -> list[dict]:
    try:
        caps = module.capabilities()
    except Exception:
        return []
    return caps if isinstance(caps, list) else []


def handle(message: str) -> dict | None:
    node_id = os.environ.get("LUHKAS_NODE_ID", "scout")
    if not _should_attempt_local(message, node_id):
        return None
    handlers = _load_handlers()
    known = _det_lookup(message)
    if known == "vault":
        return None
    if known in handlers:
        response = handlers[known].handle(message)
        if response is not None:
            return response
    for dispatch_type, module in handlers.items():
        response = module.handle(message)
        if response is not None:
            _det_learn(message, dispatch_type)
            return response
    return None


def request_target(message: str) -> dict:
    return _classify_request_target(message, os.environ.get("LUHKAS_NODE_ID", "scout"))


def capabilities() -> dict:
    commands = []
    modules = []
    for module in _load_handlers().values():
        name = getattr(module, "__package__", None) or getattr(module, "__name__", "unknown")
        modules.append(str(name).split(".")[0])
        commands.extend(_module_capabilities(module))
    node_id = os.environ.get("LUHKAS_NODE_ID", "scout")
    return {
        "ok": True,
        "capability": "luhkas_node_local_commands",
        "owner_node": node_id,
        "target_node": node_id,
        "scope": "node_local",
        "description": "Node-local deterministic commands compiled from installed *_node packages.",
        "commands": commands,
        "modules": sorted(set(modules)),
        "module_status": module_status(),
    }


def module_status() -> dict[str, dict | None]:
    _load_handlers()
    return dict(_module_status or {name: None for name in _PACKAGE_NAMES})


def selftest() -> dict:
    results: dict[str, dict] = {}
    for package_name, module in _load_handlers().items():
        pkg = str(getattr(module, "__package__", None) or package_name).split(".")[0]
        try:
            if hasattr(module, "health"):
                result = module.health()
                results[pkg] = {"ok": bool(result.get("ok", True)), **result}
            else:
                caps = _module_capabilities(module)
                results[pkg] = {"ok": True, "commands": len(caps)}
        except Exception as exc:
            results[pkg] = {"ok": False, "error": str(exc)}
    all_ok = all(v.get("ok", False) for v in results.values()) if results else True
    return {"ok": all_ok, "modules": results}
