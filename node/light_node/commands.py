"""Portable deterministic commands for a node-local light."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from urllib.request import Request, urlopen


@dataclass
class LightCommandConfig:
    service_url: str = os.environ.get("LIGHT_SERVICE_URL", os.environ.get("CAMERA_SERVICE_URL", "http://127.0.0.1:5000"))
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "scout")
    scope: str = "node_local"
    dispatch_type: str = "local_light"

    @property
    def base_url(self) -> str:
        return self.service_url.rstrip("/")


_LIGHT_ON_WORDS = {"turn on the light", "light on", "lamp on"}
_LIGHT_OFF_WORDS = {"turn off the light", "light off", "lamp off"}


def handle(user_input: str, config: LightCommandConfig | None = None) -> dict | None:
    cfg = config or LightCommandConfig()
    text = user_input.lower().strip()
    brightness = _extract_light_brightness(text)
    if brightness is not None:
        result = _post_json(f"{cfg.base_url}/settings", {"camera_light_brightness": brightness})
        return _response("control_light", result, f"Light brightness is set to {brightness}.", "I could not set the light brightness.")
    if _matches(text, _LIGHT_ON_WORDS):
        result = _post_json(f"{cfg.base_url}/settings", {"camera_light_enabled": True})
        return _response("control_light", result, "The light is on.", "I could not turn the light on.")
    if _matches(text, _LIGHT_OFF_WORDS):
        result = _post_json(f"{cfg.base_url}/settings", {"camera_light_enabled": False})
        return _response("control_light", result, "The light is off.", "I could not turn the light off.")
    return None


def capabilities(config: LightCommandConfig | None = None) -> list[dict]:
    cfg = config or LightCommandConfig()
    common = {
        "capability": "light_node",
        "owner_node": cfg.node_id,
        "target_node": cfg.node_id,
        "scope": cfg.scope,
        "target_required": False,
        "subsystem": "light",
        "dispatch_type": cfg.dispatch_type,
        "requires": ["light"],
    }
    return [
        {**common, "action": "light_on", "description": "Turn on this node's light.", "triggers": sorted(_LIGHT_ON_WORDS)},
        {**common, "action": "light_off", "description": "Turn off this node's light.", "triggers": sorted(_LIGHT_OFF_WORDS)},
        {**common, "action": "light_brightness", "description": "Set light brightness from 0-255, or percent when the command uses a percent sign.", "triggers": ["brightness {value}", "light {value}", "light to {value}"]},
    ]


def _post_json(url: str, payload: dict, timeout: float = 1.0) -> dict:
    try:
        req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except Exception:
                return {"ok": 200 <= response.status < 300}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _extract_light_brightness(text: str) -> int | None:
    match = re.search(r"\b(?:brightness|light)\s+(?:to\s+)?(?P<value>\d{1,3})(?:\s*%|\b)", text)
    if not match:
        return None
    value = int(match.group("value"))
    if "%" in match.group(0):
        return max(0, min(255, round(value * 255 / 100)))
    return max(0, min(255, value))


def _matches(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _response(action: str, result: dict, ok_message: str, fail_message: str) -> dict:
    ok = bool(result.get("ok"))
    message = ok_message if ok else fail_message
    return {"ok": ok, "action": action, "message": message, "tts": message, "mode": "direct", "local": True, "result": result}
