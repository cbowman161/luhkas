"""Canonical loader for node profiles.

A profile only has to declare ``node_id`` + ``modules`` + ``services``.
Everything else (``display``, ``sync``, ``extra_units``) is derived from
those — but the profile may override any field if it needs to.

Used by:
  * node/scripts/render_units.py     — to materialize systemd unit files
  * node/services/presence_client_service.py — for display capability + selftest
  * vault/sync_manager.py            — to know where to rsync code
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

# Module → HTTP service it implies. Modules not in this map (pantilt_node,
# light_node, ...) don't have their own service — they piggyback on another
# module's service (e.g. pantilt + light commands ride on robot_api).
_MODULE_SERVICES = {
    "camera_node": "vision",
    "pantilt_node": "pantilt",
    "rover_node": "robot-api",
    "battery_node": "battery",
    "audio_node": "audio",
    "display_node": "display",
}

# Additional services implied by a module beyond its primary one. Used when
# a single module needs both a hardware-proxy service and a higher-level
# logic service. rover_node maps to robot-api (UART proxy) AND rover
# (wheel-drive logic that polls vision /meta + dispatches via robot-api).
_MODULE_EXTRA_SERVICES = {
    "rover_node": ["rover"],
}

# Module → extra systemd unit (no network port). Chromium kiosk, gamepad
# client, etc. Added to ``extra_units`` automatically.
_MODULE_EXTRA_UNITS = {
    "display_node": "browser",
    "rover_node": "controller",
}

# Every node runs the presence proxy. Other "always on" services would go
# here too, if any.
_ALWAYS_SERVICES = ["presence"]

# Default port for each service. Profiles can override per service via
# ``services.<name>.port``.
_DEFAULT_PORTS = {
    "vision": 5000,
    "robot-api": 5001,
    "presence": 5002,
    "battery": 5003,
    "audio": 5004,
    "display": 5005,
    "pantilt": 5006,
    "rover": 5007,
}


# Combination defaults: when *all* modules in ``requires`` are present, apply
# the listed ``env`` / ``after`` / ``wants`` onto the named ``service``.
# Lets us encode "vision tracks a person when camera + pantilt + rover are
# all on the same node" without those settings leaking into kiosk's vision.
_SERVICE_COMBO_DEFAULTS: list[dict] = [
    # Pantilt service polls vision /meta and dispatches commands to robot_api;
    # start it after both are up on the same node.
    {
        "requires": ["pantilt_node"],
        "service": "pantilt",
        "after": ["{NODE_ID}-vision.service", "{NODE_ID}-robot-api.service"],
        "wants": ["{NODE_ID}-vision.service", "{NODE_ID}-robot-api.service"],
    },
    # Rover service polls vision /meta and dispatches wheel commands to robot_api.
    {
        "requires": ["rover_node"],
        "service": "rover",
        "after": ["{NODE_ID}-vision.service", "{NODE_ID}-robot-api.service"],
        "wants": ["{NODE_ID}-vision.service", "{NODE_ID}-robot-api.service"],
    },
    {
        "requires": ["camera_node", "pantilt_node", "rover_node"],
        "service": "vision",
        "after": ["{NODE_ID}-robot-api.service"],
        "wants": ["{NODE_ID}-robot-api.service"],
        "env": {
            "SCOUT_TRACKING_ENABLED": "1",
            "SCOUT_TARGET_LABEL": "person",
            "SCOUT_SCORE_THRESHOLD": "0.45",
            "SCOUT_COMMAND_INTERVAL": "0.12",
            "SCOUT_ABSOLUTE_PAN_GAIN": "28",
            "SCOUT_ABSOLUTE_TILT_GAIN": "22",
            "SCOUT_ABSOLUTE_MAX_STEP": "8",
            "SCOUT_ABSOLUTE_DISTANCE_GAIN": "0.6",
            "SCOUT_ABSOLUTE_DISTANCE_MAX_MULTIPLIER": "1.1",
            "SCOUT_WHEEL_ENABLED": "0",
            "SCOUT_POSE_JOINT_THRESHOLD": "0.3",
            "SCOUT_GUARD_ALERT_URL": "http://luhkas-vault.local:7000/alerts",
            "SCOUT_GUARD_SNAPSHOT": "1",
            "SCOUT_BEHAVIOR_ENABLED": "1",
            "SCOUT_SEARCH_TIMEOUT": "30",
            "SCOUT_SEARCH_SWEEP_DURATION": "3.2",
            "SCOUT_SEARCH_SWEEP_PAN_AMOUNT": "60",
            "SCOUT_SEARCH_SCAN_PAN_PERIOD": "11.0",
            "SCOUT_SEARCH_SCAN_TILT_PERIOD": "18.0",
            "SCOUT_SEARCH_SCAN_COMMAND_INTERVAL": "0.20",
            "SCOUT_SEARCH_COMMAND_INTERVAL": "0.20",
            "SCOUT_AVOID_DURATION": "3",
            "SCOUT_EGO_MOTION_ENABLED": "0",
            "SCOUT_GYRO_PAN_SCALE": "0.0",
            "SCOUT_GYRO_TILT_SCALE": "0.0",
            "SCOUT_TELEMETRY_POLL_INTERVAL": "0.2",
            "SCOUT_VAULT_MEMORY_ENABLED": "1",
            "SCOUT_VAULT_MEMORY_URL": "http://luhkas-vault.local:7000",
            "SCOUT_TARGET_LOST_GRACE_SECONDS": "1.5",
            "SCOUT_BYTETRACKER_MATCH_THRESH": "0.75",
        },
    },
]


def _expand_node_id(value: str, node_id: str) -> str:
    return value.replace("{NODE_ID}", node_id)


def iter_profile_paths(profiles_dir: Path | None = None) -> list[Path]:
    """List real ``*.json`` profile files in a directory.

    Skips ``.`` / ``._`` files (macOS SMB resource forks) and anything that
    isn't a plain file.
    """
    base = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
    return sorted(
        p for p in base.glob("*.json")
        if p.is_file() and not p.name.startswith(".")
    )


def load_profile(node_id_or_path: Any, *, profiles_dir: Path | None = None) -> dict:
    """Load and resolve a node profile.

    Pass either a node id (``"kiosk"``) or a path to a profile JSON file.
    """
    text_arg = str(node_id_or_path)
    if text_arg.endswith(".json"):
        path = Path(text_arg)
        node_id_hint = path.stem
    else:
        node_id_hint = text_arg
        base = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        path = base / f"{node_id_hint}.json"
    raw = json.loads(path.read_text())
    return resolve(raw, default_node_id=node_id_hint)


def resolve(raw: dict, *, default_node_id: str = "") -> dict:
    """Fill in derived defaults for a parsed profile dict.

    The truth is in ``modules``. ``services`` is optional — when omitted, the
    set of services to run is inferred from modules using ``_MODULE_SERVICES``
    plus ``_ALWAYS_SERVICES`` (presence). Default ports come from
    ``_DEFAULT_PORTS``. Anything declared in ``services`` overrides the
    inference (port, env, after, wants).
    """
    node_id = str(raw.get("node_id") or default_node_id).strip()
    if not node_id:
        raise ValueError("profile is missing 'node_id'")

    modules = list(raw.get("modules") or [])
    overrides = dict(raw.get("services") or {})

    # ── services inferred from modules ─────────────────────────────────────
    inferred_names: list[str] = list(_ALWAYS_SERVICES)
    for m in modules:
        name = _MODULE_SERVICES.get(m)
        if name and name not in inferred_names:
            inferred_names.append(name)
        for extra in _MODULE_EXTRA_SERVICES.get(m, []):
            if extra not in inferred_names:
                inferred_names.append(extra)
    # Profile may declare extra services not implied by any module.
    for name in overrides.keys():
        if name not in inferred_names:
            inferred_names.append(name)

    services: dict[str, dict] = {}
    for name in inferred_names:
        services[name] = {
            "port": _DEFAULT_PORTS.get(name),
            "env": {},
            "after": [],
            "wants": [],
        }

    # Apply combo defaults whose required modules are all present.
    module_set = set(modules)
    for combo in _SERVICE_COMBO_DEFAULTS:
        if not set(combo.get("requires") or []).issubset(module_set):
            continue
        svc_name = combo.get("service")
        if svc_name not in services:
            continue
        target = services[svc_name]
        for unit in combo.get("after") or []:
            unit = _expand_node_id(unit, node_id)
            if unit not in target["after"]:
                target["after"].append(unit)
        for unit in combo.get("wants") or []:
            unit = _expand_node_id(unit, node_id)
            if unit not in target["wants"]:
                target["wants"].append(unit)
        for key, value in (combo.get("env") or {}).items():
            target["env"].setdefault(key, value)

    # Apply profile-level overrides last (they win).
    for name, override in overrides.items():
        target = services.setdefault(name, {
            "port": _DEFAULT_PORTS.get(name),
            "env": {},
            "after": [],
            "wants": [],
        })
        if isinstance(override, dict):
            if "port" in override:
                target["port"] = override["port"]
            for unit in override.get("after") or []:
                if unit not in target["after"]:
                    target["after"].append(unit)
            for unit in override.get("wants") or []:
                if unit not in target["wants"]:
                    target["wants"].append(unit)
            target["env"].update(override.get("env") or {})
        elif isinstance(override, int):
            target["port"] = override

    # ── display ────────────────────────────────────────────────────────────
    inferred_has_display = "display_node" in modules
    display = dict(raw.get("display") or {})
    display.setdefault("has_display", inferred_has_display)
    if display.get("has_display") and "kind" not in display:
        display["kind"] = "hdmi_touch"

    # ── rdp ────────────────────────────────────────────────────────────────
    # Opt-in only. RDP isn't installed unless a profile explicitly sets
    # ``rdp.enabled: true``. (Earlier this auto-inferred from camera_node
    # / display_node, but the only host we actually RDP into is the vault,
    # which manages its own xrdp stack — nodes don't need it.)
    rdp = dict(raw.get("rdp") or {})
    rdp.setdefault("enabled", False)

    # ── extra units (inferred from modules; profile may add more) ──────────
    inferred_extras = [
        _MODULE_EXTRA_UNITS[m] for m in modules if m in _MODULE_EXTRA_UNITS
    ]
    explicit_extras = list(raw.get("extra_units") or [])
    # Preserve order; dedupe.
    seen = set()
    extras: list = []
    for name in inferred_extras + explicit_extras:
        key = name if isinstance(name, str) else name.get("name", "")
        if key in seen:
            continue
        seen.add(key)
        extras.append(name)

    # ── sync ───────────────────────────────────────────────────────────────
    sync_raw = dict(raw.get("sync") or {})
    sync = {
        "host": sync_raw.get("host") or f"luhkas-{node_id}",
        "user": sync_raw.get("user") or "luhkas",
        "node_dir": sync_raw.get("node_dir") or "luhkas/node",
    }
    if sync_raw.get("services"):
        sync["services"] = sync_raw["services"]
    else:
        service_names = [f"{node_id}-{name}.service" for name in services.keys()]
        extra_names = [
            f"{node_id}-{(extra if isinstance(extra, str) else extra.get('name', ''))}.service"
            for extra in extras
            if (extra if isinstance(extra, str) else extra.get("name", "")) != "controller"
        ]
        sync["services"] = [name for name in service_names + extra_names if name != f"{node_id}-.service"]

    resolved = dict(raw)
    resolved.update({
        "node_id": node_id,
        "modules": modules,
        "services": services,
        "display": display,
        "rdp": rdp,
        "sync": sync,
        "extra_units": extras,
    })
    return resolved
