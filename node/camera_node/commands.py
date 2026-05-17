"""Portable deterministic commands for any LUHKAS node with a camera.

This package owns camera-only commands such as snapshots and short video clips.
It deliberately does not import Scout movement, wheels, tracking, pan/tilt, or
robot APIs, so it can be copied into another camera node as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from urllib.request import Request, urlopen


@dataclass
class CameraCommandConfig:
    camera_url: str = os.environ.get("CAMERA_SERVICE_URL", "http://127.0.0.1:5000")
    node_id: str = os.environ.get("CAMERA_NODE_ID", os.environ.get("LUHKAS_NODE_ID", "scout"))
    scope: str = os.environ.get("CAMERA_COMMAND_SCOPE", "node_local")
    dispatch_type: str = os.environ.get("CAMERA_COMMAND_DISPATCH_TYPE", "local_media")
    clip_seconds: float = float(os.environ.get("CAMERA_CLIP_SECONDS", "8.0"))
    snapshot_timeout: float = float(os.environ.get("CAMERA_SNAPSHOT_TIMEOUT", "5.0"))
    clip_timeout: float = float(os.environ.get("CAMERA_CLIP_TIMEOUT", "12.0"))

    @property
    def base_url(self) -> str:
        return self.camera_url.rstrip("/")


_CLIP_WORDS = {
    "record a clip",
    "save a clip",
    "video clip",
    "record video",
    "record a video",
    "take a video",
}
_SNAPSHOT_WORDS = {
    "take a picture",
    "take a photo",
    "save a snapshot",
    "snapshot",
}
_GUARD_ON_WORDS = {"guard on", "enable guard", "start guard", "start guarding"}
_GUARD_OFF_WORDS = {"guard off", "disable guard", "stop guard", "stop guarding"}


def handle(user_input: str, config: CameraCommandConfig | None = None) -> dict | None:
    cfg = config or CameraCommandConfig()
    text = user_input.lower().strip()
    if _matches(text, _SNAPSHOT_WORDS):
        result = _capture_snapshot(cfg)
        return _response(result, "capture_snapshot", "Saved snapshot", "I could not save a snapshot.")
    if _matches(text, _CLIP_WORDS):
        result = _record_clip(cfg)
        return _response(result, "record_clip", "Recorded clip", "I could not record a clip.")
    if _matches(text, _GUARD_ON_WORDS):
        result = _post_json(f"{cfg.base_url}/guard", {"enabled": True}, timeout=cfg.snapshot_timeout)
        return _response(result, "control_guard", "Guard mode is on", "I could not enable guard mode.")
    if _matches(text, _GUARD_OFF_WORDS):
        result = _post_json(f"{cfg.base_url}/guard", {"enabled": False}, timeout=cfg.snapshot_timeout)
        return _response(result, "control_guard", "Guard mode is off", "I could not disable guard mode.")
    return None


def capabilities(config: CameraCommandConfig | None = None) -> list[dict]:
    cfg = config or CameraCommandConfig()
    common = {
        "capability": "camera_node_media",
        "owner_node": cfg.node_id,
        "target_node": cfg.node_id,
        "scope": cfg.scope,
        "target_required": False,
        "subsystem": "camera",
        "dispatch_type": cfg.dispatch_type,
    }
    return [
        {
            **common,
            "action": "capture_snapshot",
            "description": "Save a still image from this node's camera.",
            "triggers": sorted(_SNAPSHOT_WORDS),
            "requires": ["camera"],
        },
        {
            **common,
            "action": "record_clip",
            "description": "Record a short video clip from this node's camera.",
            "triggers": sorted(_CLIP_WORDS),
            "requires": ["camera", "video_buffer"],
        },
        {
            **common,
            "action": "guard_on",
            "description": "Enable camera guard mode alerts.",
            "triggers": sorted(_GUARD_ON_WORDS),
            "requires": ["camera"],
        },
        {
            **common,
            "action": "guard_off",
            "description": "Disable camera guard mode alerts.",
            "triggers": sorted(_GUARD_OFF_WORDS),
            "requires": ["camera"],
        },
    ]


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
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


def _record_clip(config: CameraCommandConfig) -> dict:
    return _post_json(
        f"{config.base_url}/clip",
        {"seconds": config.clip_seconds},
        timeout=config.clip_timeout,
    )


def _capture_snapshot(config: CameraCommandConfig) -> dict:
    return _post_json(f"{config.base_url}/snapshot", {}, timeout=config.snapshot_timeout)


def _response(result: dict, action: str, ok_prefix: str, fail_message: str) -> dict:
    ok = bool(result.get("ok"))
    path = result.get("path")
    message = f"{ok_prefix} to {path}." if ok and path else ok_prefix + "." if ok else fail_message
    return {
        "ok": ok,
        "action": action,
        "message": message,
        "tts": message,
        "mode": "direct",
        "local": True,
        "result": result,
    }
