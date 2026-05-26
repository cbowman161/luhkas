"""Portable deterministic commands for node-local battery state.

Provides natural-language answers to questions like "what's my battery?"
without having to round-trip to vault. Talks to the local battery_node
service over HTTP.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class BatteryCommandConfig:
    service_url: str = os.environ.get("BATTERY_SERVICE_URL", "http://127.0.0.1:5003")
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "scout")
    scope: str = "node_local"
    dispatch_type: str = "local_battery"

    @property
    def base_url(self) -> str:
        return self.service_url.rstrip("/")


_BATTERY_PATTERNS = [
    r"\bbattery\b",
    r"\bcharge\b",
    r"\bhow (much )?(charge|power)\b",
    r"\bpercent(age)? (left|remaining)\b",
    r"\bhow .* (charge|power) (do (i|we) have|is left|remaining)\b",
]


def capabilities() -> list[dict]:
    cfg = BatteryCommandConfig()
    return [
        {
            "name": "report_battery",
            "description": "Report current battery percent/voltage on this node.",
            "scope": cfg.scope,
            "dispatch_type": cfg.dispatch_type,
            "owner_node": cfg.node_id,
            "target_node": cfg.node_id,
            "examples": [
                "what's my battery?",
                "how much charge is left?",
                "battery percentage",
            ],
        }
    ]


def handle(user_input: str, config: BatteryCommandConfig | None = None) -> dict | None:
    cfg = config or BatteryCommandConfig()
    text = (user_input or "").lower().strip()
    if not text:
        return None
    if not any(re.search(p, text) for p in _BATTERY_PATTERNS):
        return None
    reading = _read_battery(cfg)
    if reading is None or not reading.get("ok"):
        message = "I can't read the battery right now."
        return {
            "ok": False,
            "capability": "report_battery",
            "message": message,
            "tts": message,
            "data": reading or {},
        }
    percent = reading.get("percent", 0)
    voltage = reading.get("voltage")
    state = "charging" if reading.get("charging") else "discharging"
    parts = [f"Battery is at {percent} percent"]
    if voltage:
        parts.append(f"({voltage:.2f}V)")
    if "charging" in reading:
        parts.append(f"and {state}")
    message = " ".join(parts) + "."
    return {
        "ok": True,
        "capability": "report_battery",
        "message": message,
        "tts": message,
        "data": reading,
    }


def health(config: BatteryCommandConfig | None = None) -> dict:
    cfg = config or BatteryCommandConfig()
    reading = _read_battery(cfg)
    if reading is None:
        return {"ok": False, "error": "battery_service_unreachable"}
    return {"ok": bool(reading.get("ok")), "battery": reading}


def _read_battery(cfg: BatteryCommandConfig) -> dict | None:
    try:
        req = Request(f"{cfg.base_url}/battery", headers={"Accept": "application/json"})
        with urlopen(req, timeout=2.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
