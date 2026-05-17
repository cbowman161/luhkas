"""Wheel rover and follow commands.

This package assumes a camera_node exists for tracking/follow state. Gamepad
manual routing also belongs with the rover package, though the current live
USB loop still runs inside the Scout vision service until it is peeled out.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from urllib.request import Request, urlopen


@dataclass
class RoverCommandConfig:
    robot_service_url: str = os.environ.get("ROVER_ROBOT_SERVICE_URL", "http://127.0.0.1:5001")
    camera_service_url: str = os.environ.get("ROVER_CAMERA_SERVICE_URL", os.environ.get("CAMERA_SERVICE_URL", "http://127.0.0.1:5000"))
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "scout")
    scope: str = "node_local"
    dispatch_type: str = "local_rover"
    drive_speed: int = int(os.environ.get("ROVER_DRIVE_SPEED", "600"))
    turn_speed: int = int(os.environ.get("ROVER_TURN_SPEED", "500"))
    turn_around_duration: float = float(os.environ.get("ROVER_TURN_AROUND_SECONDS", "1.5"))

    @property
    def robot_base_url(self) -> str:
        return self.robot_service_url.rstrip("/")

    @property
    def camera_base_url(self) -> str:
        return self.camera_service_url.rstrip("/")


_STOP_WORDS = {"stop", "halt", "freeze", "stay", "hold", "pause", "don't move", "dont move", "do not move"}
_FORWARD_WORDS = {"forward", "forwards", "ahead", "advance", "go forward", "move forward", "drive forward", "go ahead", "move ahead"}
_BACKWARD_WORDS = {"back", "backward", "backwards", "reverse", "go back", "move back", "back up", "go backward", "drive back"}
_TURN_LEFT_WORDS = {"turn left", "spin left", "rotate left", "go left", "strafe left"}
_TURN_RIGHT_WORDS = {"turn right", "spin right", "rotate right", "go right", "strafe right"}
_TURN_AROUND_WORDS = {"turn around", "u-turn", "uturn", "rotate 180", "spin around", "face the other way", "reverse direction", "about face"}
_FOLLOW_WORDS = {"follow me", "track me", "come here", "come with me", "follow"}
_STOP_FOLLOW_WORDS = {"stop following", "don't follow", "dont follow", "do not follow", "stay put", "hold position"}


def handle(user_input: str, config: RoverCommandConfig | None = None) -> dict | None:
    cfg = config or RoverCommandConfig()
    text = user_input.lower().strip()
    if _matches(text, _STOP_FOLLOW_WORDS):
        _post_json(f"{cfg.camera_base_url}/tracking", {"enabled": False, "follow": False})
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": 0})
        return _response("control_rover", result, "Stopped following.", "I could not stop following.")
    if _matches(text, _FOLLOW_WORDS):
        result = _post_json(f"{cfg.camera_base_url}/tracking", {"enabled": True, "follow": True})
        return _response("control_rover", result, "Following.", "I could not turn following on.")
    if _matches(text, _STOP_WORDS):
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": 0})
        return _response("control_rover", result, "Stopped.", "I could not stop.")
    if _matches(text, _FORWARD_WORDS):
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": cfg.drive_speed, "z": 0})
        return _response("control_rover", result, "Moving forward.", "I could not move forward.")
    if _matches(text, _BACKWARD_WORDS):
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": -cfg.drive_speed, "z": 0})
        return _response("control_rover", result, "Moving backward.", "I could not move backward.")
    if _matches(text, _TURN_AROUND_WORDS):
        deadline = time.monotonic() + cfg.turn_around_duration
        result = {"ok": True}
        while time.monotonic() < deadline:
            result = _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": cfg.turn_speed})
            time.sleep(0.1)
        _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": 0})
        return _response("control_rover", result, "Turned around.", "I could not turn around.")
    if _matches(text, _TURN_LEFT_WORDS):
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": -cfg.turn_speed})
        return _response("control_rover", result, "Turning left.", "I could not turn left.")
    if _matches(text, _TURN_RIGHT_WORDS):
        result = _post_json(f"{cfg.robot_base_url}/move", {"x": 0, "z": cfg.turn_speed})
        return _response("control_rover", result, "Turning right.", "I could not turn right.")
    return None


def capabilities(config: RoverCommandConfig | None = None) -> list[dict]:
    cfg = config or RoverCommandConfig()
    common = {
        "capability": "rover_node",
        "owner_node": cfg.node_id,
        "target_node": cfg.node_id,
        "scope": cfg.scope,
        "subsystem": "rover",
        "dispatch_type": cfg.dispatch_type,
    }
    return [
        {**common, "action": "stop", "target_required": False, "description": "Stop wheel movement.", "triggers": sorted(_STOP_WORDS), "requires": ["wheels"]},
        {**common, "action": "stop_following", "target_required": False, "description": "Disable follow/tracking and stop wheel movement.", "triggers": sorted(_STOP_FOLLOW_WORDS), "requires": ["camera", "tracking", "wheels"]},
        {**common, "action": "follow", "target_required": True, "description": "Enable person tracking and wheel following.", "triggers": sorted(_FOLLOW_WORDS), "requires": ["camera", "tracking", "wheels"]},
        {**common, "action": "drive_forward", "target_required": False, "description": "Move the wheel base forward.", "triggers": sorted(_FORWARD_WORDS), "requires": ["wheels"]},
        {**common, "action": "drive_backward", "target_required": False, "description": "Move the wheel base backward.", "triggers": sorted(_BACKWARD_WORDS), "requires": ["wheels"]},
        {**common, "action": "turn_around", "target_required": False, "description": "Rotate the wheel base roughly 180 degrees.", "triggers": sorted(_TURN_AROUND_WORDS), "requires": ["wheels"]},
        {**common, "action": "turn_left", "target_required": False, "description": "Turn the wheel base left (use 'look left' to pan the camera instead).", "triggers": sorted(_TURN_LEFT_WORDS), "requires": ["wheels"]},
        {**common, "action": "turn_right", "target_required": False, "description": "Turn the wheel base right (use 'look right' to pan the camera instead).", "triggers": sorted(_TURN_RIGHT_WORDS), "requires": ["wheels"]},
    ]


def _post_json(url: str, payload: dict, timeout: float = 0.5) -> dict:
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
