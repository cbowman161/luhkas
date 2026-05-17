"""Generic local command composition for LUHKAS nodes."""
from __future__ import annotations

import importlib
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

_DEFAULT_PACKAGES = "camera_node,pantilt_node,rover_node,light_node"
_KNOWN_NODE_PACKAGES = [
    "camera_node",
    "pantilt_node",
    "rover_node",
    "light_node",
    "display_node",
    "speech_node",
]
_PACKAGE_NAMES = [
    name.strip()
    for name in os.environ.get("LUHKAS_NODE_PACKAGES", _DEFAULT_PACKAGES).split(",")
    if name.strip()
]
_handlers: dict[str, object] | None = None
_module_status: dict[str, dict | None] | None = None


def _load_handlers() -> dict[str, object]:
    global _handlers, _module_status
    if _handlers is not None:
        return _handlers
    handlers = {}
    status: dict[str, dict | None] = {name: None for name in _KNOWN_NODE_PACKAGES}
    for package_name in sorted(set(_KNOWN_NODE_PACKAGES + _PACKAGE_NAMES)):
        if package_name not in _PACKAGE_NAMES:
            continue
        try:
            module = importlib.import_module(f"{package_name}.commands")
        except Exception as exc:
            status[package_name] = None
            continue
        caps = _module_capabilities(module)
        status[package_name] = {
            "available": True,
            "commands": len(caps),
        }
        for command in _module_capabilities(module):
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
    return dict(_module_status or {name: None for name in _KNOWN_NODE_PACKAGES})


def selftest() -> dict:
    """Run a quick health check of each loaded *_node module.

    For each module, calls `health()` if it exists, otherwise checks that
    `capabilities()` succeeds. Returns a structured report.
    """
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
