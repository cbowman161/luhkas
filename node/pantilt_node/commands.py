"""Pan/tilt and pan/tilt-tracking deterministic commands."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from urllib.request import Request, urlopen


@dataclass
class PanTiltCommandConfig:
    camera_service_url: str = os.environ.get("PANTILT_CAMERA_SERVICE_URL", os.environ.get("CAMERA_SERVICE_URL", "http://127.0.0.1:5000"))
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "scout")
    scope: str = "node_local"
    dispatch_type: str = "local_pantilt"
    pan_step: int = int(os.environ.get("PANTILT_PAN_STEP", "60"))
    tilt_step: int = int(os.environ.get("PANTILT_TILT_STEP", "40"))

    @property
    def camera_base_url(self) -> str:
        return self.camera_service_url.rstrip("/")


_LOOK_LEFT_WORDS = {"look left", "pan left", "face left", "glance left", "turn camera left"}
_LOOK_RIGHT_WORDS = {"look right", "pan right", "face right", "glance right", "turn camera right"}
_LOOK_UP_WORDS = {"look up", "tilt up", "look higher", "glance up"}
_LOOK_DOWN_WORDS = {"look down", "tilt down", "look lower", "glance down"}
_CENTER_WORDS = {"look straight", "center", "center camera", "face forward", "look ahead", "reset camera", "look forward", "straighten"}
_TRACKING_ON_WORDS = {"enable tracking", "track person", "track people", "start tracking"}
_TRACKING_OFF_WORDS = {"disable tracking", "turn off tracking", "stop tracking"}
_SEARCH_CAMERA_ON_WORDS = {"search camera on", "enable search camera", "turn on search camera"}
_SEARCH_CAMERA_OFF_WORDS = {"search camera off", "disable search camera", "turn off search camera"}


def handle(user_input: str, config: PanTiltCommandConfig | None = None) -> dict | None:
    cfg = config or PanTiltCommandConfig()
    text = user_input.lower().strip()
    if _matches(text, _SEARCH_CAMERA_OFF_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/settings", {"search_movement_enabled": False})
        return _response("control_search_camera", result, "Search camera is off.", "I could not disable search camera.")
    if _matches(text, _SEARCH_CAMERA_ON_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/settings", {"search_movement_enabled": True})
        return _response("control_search_camera", result, "Search camera is on.", "I could not enable search camera.")
    if _matches(text, _TRACKING_OFF_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/tracking", {"enabled": False})
        return _response("control_tracking", result, "Tracking is off.", "I could not turn tracking off.")
    if _matches(text, _TRACKING_ON_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/tracking", {"enabled": True})
        return _response("control_tracking", result, "Tracking is on.", "I could not turn tracking on.")
    if _matches(text, _LOOK_LEFT_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/pantilt", {"pan": -cfg.pan_step, "tilt": 0})
        return _response("control_pantilt", result, "Looking left.", "I could not move the camera left.")
    if _matches(text, _LOOK_RIGHT_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/pantilt", {"pan": cfg.pan_step, "tilt": 0})
        return _response("control_pantilt", result, "Looking right.", "I could not move the camera right.")
    if _matches(text, _LOOK_UP_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/pantilt", {"pan": 0, "tilt": cfg.tilt_step})
        return _response("control_pantilt", result, "Looking up.", "I could not move the camera up.")
    if _matches(text, _LOOK_DOWN_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/pantilt", {"pan": 0, "tilt": -cfg.tilt_step})
        return _response("control_pantilt", result, "Looking down.", "I could not move the camera down.")
    if _matches(text, _CENTER_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/pantilt", {"center": True})
        return _response("control_pantilt", result, "Centering camera.", "I could not center the camera.")
    return None


def capabilities(config: PanTiltCommandConfig | None = None) -> list[dict]:
    cfg = config or PanTiltCommandConfig()
    common = {
        "capability": "pantilt_node",
        "owner_node": cfg.node_id,
        "target_node": cfg.node_id,
        "scope": cfg.scope,
        "target_required": False,
        "subsystem": "pantilt",
        "dispatch_type": cfg.dispatch_type,
    }
    return [
        {**common, "action": "tracking_on", "description": "Enable pan/tilt visual tracking.", "triggers": sorted(_TRACKING_ON_WORDS), "requires": ["camera", "pantilt", "tracking"]},
        {**common, "action": "tracking_off", "description": "Disable pan/tilt visual tracking.", "triggers": sorted(_TRACKING_OFF_WORDS), "requires": ["camera", "pantilt", "tracking"]},
        {**common, "action": "search_camera_on", "description": "Enable autonomous pan/tilt search sweep.", "triggers": sorted(_SEARCH_CAMERA_ON_WORDS), "requires": ["camera", "pantilt", "tracking"]},
        {**common, "action": "search_camera_off", "description": "Disable autonomous pan/tilt search sweep.", "triggers": sorted(_SEARCH_CAMERA_OFF_WORDS), "requires": ["camera", "pantilt", "tracking"]},
        {**common, "action": "look_left", "description": "Pan the camera left.", "triggers": sorted(_LOOK_LEFT_WORDS), "requires": ["pantilt"]},
        {**common, "action": "look_right", "description": "Pan the camera right.", "triggers": sorted(_LOOK_RIGHT_WORDS), "requires": ["pantilt"]},
        {**common, "action": "look_up", "description": "Tilt the camera up.", "triggers": sorted(_LOOK_UP_WORDS), "requires": ["pantilt"]},
        {**common, "action": "look_down", "description": "Tilt the camera down.", "triggers": sorted(_LOOK_DOWN_WORDS), "requires": ["pantilt"]},
        {**common, "action": "center_camera", "description": "Center or stop the camera pan/tilt.", "triggers": sorted(_CENTER_WORDS), "requires": ["pantilt"]},
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


def _matches(text: str, phrases: set[str]) -> bool:
    for phrase in phrases:
        if " " in phrase:
            if phrase in text:
                return True
        elif re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return True
    return False


def _response(action: str, result: dict, ok_message: str, fail_message: str) -> dict:
    ok = bool(result.get("ok"))
    message = ok_message if ok else fail_message
    return {"ok": ok, "action": action, "message": message, "tts": message, "mode": "direct", "local": True, "result": result}
