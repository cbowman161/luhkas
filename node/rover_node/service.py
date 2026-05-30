#!/usr/bin/env python3
"""rover_node service — owns wheel drive (follow + recovery).

Architecture mirror of pantilt_service: polls vision_service /meta for
tracking_state, branches on behavior_state, dispatches wheel commands
via robot_api.

    vision_service (camera_node)  ->  /meta {tracking_state}
                                          |
                                          v
    rover_service (this file)  ->  PanTiltController.wheel_follow_command
                                          |
                                          v
    robot_api  ->  UART  ->  wheel motors

Only runs on nodes with rover_node in modules. Pantilt servo control
lives in pantilt_service; this only deals with the wheels.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scout.config import MotionConfig
from scout.motion import PanTiltController
from scout.robot_client import RobotClient
from scout.types import Detection
from rover_node.runtime import apply_settings as _apply_rover_settings


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] rover: %(message)s")
log = logging.getLogger("rover_service")


_VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:5000").rstrip("/")
_ROBOT_API_URL = os.environ.get("ROBOT_API_URL", "http://127.0.0.1:5001").rstrip("/")
_ROVER_PORT = int(os.environ.get("ROVER_PORT", "5007"))
_POLL_INTERVAL_S = float(os.environ.get("ROVER_POLL_INTERVAL", "0.05"))


_state_lock = threading.Lock()
_controller = PanTiltController(MotionConfig())
_robot = RobotClient(_ROBOT_API_URL)

_latest_tracking_state: dict | None = None
_last_meta_fetch_ts: float = 0.0
_last_behavior_state: str | None = None
_state_entered_at: float = time.time()
_status: dict = {
    "started_at": time.time(),
    "polls_ok": 0,
    "polls_fail": 0,
    "commands_sent": 0,
    "vision_url": _VISION_URL,
    "robot_api_url": _ROBOT_API_URL,
}


def _fetch_tracking_state(timeout_s: float = 0.2) -> dict | None:
    try:
        with urlopen(_VISION_URL + "/meta", timeout=timeout_s) as r:
            meta = json.loads(r.read().decode("utf-8"))
        return meta.get("tracking_state")
    except Exception as exc:
        log.debug("meta fetch failed: %s", exc)
        return None


def _target_from_tracking_state(t: dict) -> Detection:
    bbox = list(t["bbox"])
    vx, vy = (t.get("velocity") or [0.0, 0.0])
    det = Detection(
        id=t.get("id"),
        label=t.get("label", ""),
        class_id=int(t.get("class_id", 0)),
        bbox=bbox,
        confidence=float(t.get("confidence", 0.0)),
    )
    det.vx = float(vx)
    det.vy = float(vy)
    det.predicted = bool(t.get("predicted", False))
    det.identity = t.get("identity")
    aim = t.get("aim") or {}
    det.aim_x = aim.get("x")
    det.aim_y = aim.get("y")
    det.aim_source = aim.get("source", "bbox")
    return det


def _act_on_tracking_state(ts: dict | None) -> None:
    global _last_behavior_state, _state_entered_at
    if ts is None:
        return

    behavior = ts.get("behavior_state", "IDLE")
    target_dict = ts.get("target")
    fshape = ts.get("frame_shape") or [640, 640, 3]
    fh, fw = int(fshape[0]), int(fshape[1])
    collision = bool(ts.get("collision_blocked", False))

    if behavior != _last_behavior_state:
        _last_behavior_state = behavior
        _state_entered_at = time.time()

    target = None
    if target_dict is not None:
        try:
            target = _target_from_tracking_state(target_dict)
        except Exception as exc:
            log.warning("could not build target from tracking_state: %s", exc)

    def _bump_sent():
        with _state_lock:
            _status["commands_sent"] += 1

    if behavior == "FOLLOWING":
        if (
            target is not None
            and _controller.config.follow_enabled
            and _controller.config.wheel_enabled
            and not collision
        ):
            follow_cmd = _controller.wheel_follow_command(target, fw, fh)
            if follow_cmd is not None:
                _robot.move(follow_cmd["x"], follow_cmd["z"])
                _bump_sent()
            else:
                _robot.move(0, 0)
        else:
            _robot.move(0, 0)
    elif behavior == "AVOIDING":
        _robot.move(-200, 0)
        _bump_sent()
    else:
        # SEARCHING / IDLE / GUARDING / MANUAL: wheels stopped.
        _robot.move(0, 0)


def _poll_loop() -> None:
    global _latest_tracking_state, _last_meta_fetch_ts
    log.info("poll loop start (vision=%s, interval=%.3fs)", _VISION_URL, _POLL_INTERVAL_S)
    while True:
        ts = _fetch_tracking_state()
        with _state_lock:
            if ts is None:
                _status["polls_fail"] += 1
            else:
                _status["polls_ok"] += 1
                _last_meta_fetch_ts = time.time()
                _latest_tracking_state = ts
        if ts is not None:
            try:
                _act_on_tracking_state(ts)
            except Exception as exc:
                log.warning("act_on_tracking_state failed: %s", exc)
        time.sleep(_POLL_INTERVAL_S)


class Handler(BaseHTTPRequestHandler):
    # CORS so UI page served from vision_service (:5000) can POST settings
    # directly to rover_service (:5007) without violating same-origin.
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:
            self._json({"ok": False, "error": f"bad request: {exc}"}, status=400)
            return None

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            with _state_lock:
                age = (time.time() - _last_meta_fetch_ts) if _last_meta_fetch_ts > 0 else None
                self._json({
                    "ok": True,
                    "service": "rover_node",
                    "vision_url": _VISION_URL,
                    "robot_api_url": _ROBOT_API_URL,
                    "poll_interval_s": _POLL_INTERVAL_S,
                    "last_meta_fetch_age_s": age,
                    "polls_ok": _status["polls_ok"],
                    "polls_fail": _status["polls_fail"],
                    "commands_sent": _status["commands_sent"],
                    "current_behavior": _last_behavior_state,
                })
        elif self.path == "/status":
            with _state_lock:
                self._json({
                    "ok": True,
                    "tracking_state": _latest_tracking_state,
                    "status": dict(_status),
                    "controller_config": {
                        "wheel_enabled": _controller.config.wheel_enabled,
                        "follow_enabled": _controller.config.follow_enabled,
                        "follow_forward_speed": _controller.config.follow_forward_speed,
                        "follow_steer_gain": _controller.config.follow_steer_gain,
                        "follow_target_bbox_ratio": _controller.config.follow_target_bbox_ratio,
                        "close_target_bbox_ratio": _controller.config.close_target_bbox_ratio,
                        "follow_deadzone_ratio": _controller.config.follow_deadzone_ratio,
                    },
                })
        elif self.path == "/":
            self._json({"service": "rover_node", "endpoints": ["/health", "/status", "/settings"]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/settings":
            body = self._read_json()
            if body is None:
                return
            try:
                _apply_rover_settings(body, _controller)
                self._json({"ok": True, "applied": list(body.keys())})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
        elif self.path == "/move":
            body = self._read_json()
            if body is None:
                return
            try:
                x = float(body.get("x", 0))
                z = float(body.get("z", 0))
                _robot.move(x, z)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)


def main() -> None:
    log.info(
        "rover service starting on http://0.0.0.0:%d (vision=%s, robot_api=%s)",
        _ROVER_PORT, _VISION_URL, _ROBOT_API_URL,
    )
    threading.Thread(target=_poll_loop, daemon=True, name="rover-poll").start()
    server = ThreadingHTTPServer(("0.0.0.0", _ROVER_PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
