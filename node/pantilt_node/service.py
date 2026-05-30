#!/usr/bin/env python3
"""pantilt_node service — owns the physical pan/tilt servos.

Architecture:
    vision_service (camera_node)
        computes tracking_state per frame, exposes on /meta
                |
                v
    pantilt_service (this file)
        polls /meta, instantiates PanTiltController + SearchController,
        branches on tracking_state.behavior_state, dispatches commands
                |
                v
    robot_api (UART proxy)

Only runs on nodes whose profile lists pantilt_node in modules.

Settings (max_command, min_command, edge_reacquire_enabled, settle_*,
pan/tilt limits, etc.) are currently sourced from MotionConfig defaults.
The UI still POSTs to vision_service's /settings; mirroring those into
pantilt-service is a future refinement (proxy or direct).
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

from scout.config import MotionConfig, SearchConfig
from scout.motion import PanTiltController
from scout.search import SearchController
from scout.robot_client import RobotClient
from scout.types import Detection
from pantilt_node.runtime import dispatch_robot_command


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] pantilt: %(message)s")
log = logging.getLogger("pantilt_service")


_VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:5000").rstrip("/")
_ROBOT_API_URL = os.environ.get("ROBOT_API_URL", "http://127.0.0.1:5001").rstrip("/")
_PANTILT_PORT = int(os.environ.get("PANTILT_PORT", "5006"))
_POLL_INTERVAL_S = float(os.environ.get("PANTILT_POLL_INTERVAL", "0.05"))  # 20 Hz


# ---- Module-level state -----------------------------------------------------
_state_lock = threading.Lock()

# Pantilt controller objects. Configs are kept at defaults for now; later
# slices will sync them from /meta or accept their own /settings POSTs.
_controller = PanTiltController(MotionConfig())
_search_controller = SearchController(SearchConfig())
_robot = RobotClient(_ROBOT_API_URL)

_latest_tracking_state: dict | None = None
_last_meta_fetch_ts: float = 0.0
_last_target_vx: float = 0.0
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
    """Reconstruct a Detection-shaped object from the tracking_state.target
    dict so PanTiltController.command_for_target can consume it directly."""
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
    """Per-poll dispatch: branch on behavior_state, drive servos."""
    global _last_target_vx, _last_behavior_state, _state_entered_at

    if ts is None:
        return

    behavior = ts.get("behavior_state", "IDLE")
    target_dict = ts.get("target")
    fshape = ts.get("frame_shape") or [640, 640, 3]
    fh, fw = int(fshape[0]), int(fshape[1])
    tracking_enabled = bool(ts.get("tracking_enabled", False))
    guard_enabled = bool(ts.get("guard_enabled", False))

    # Track behavior transitions so we know when we just entered a state.
    if behavior != _last_behavior_state:
        _last_behavior_state = behavior
        _state_entered_at = time.time()
    time_in_state = time.time() - _state_entered_at

    target = None
    if target_dict is not None:
        try:
            target = _target_from_tracking_state(target_dict)
        except Exception as exc:
            log.warning("could not build target from tracking_state: %s", exc)

    # Maintain search controller's notion of last-seen velocity.
    if target is not None:
        _last_target_vx = target.vx
        _search_controller.on_target_acquired()
    elif guard_enabled or tracking_enabled:
        _search_controller.on_target_lost(_controller._estimated_pan, _last_target_vx)

    def _bump_sent():
        with _state_lock:
            _status["commands_sent"] += 1

    if behavior == "FOLLOWING" and target is not None:
        cmd = _controller.command_for_target(target, fw, fh)
        if cmd is not None:
            dispatch_robot_command(_robot, cmd)
            _bump_sent()
    elif behavior == "SEARCHING":
        search_cmd = _search_controller.search_command(
            _controller._estimated_pan, _controller._estimated_tilt
        )
        if search_cmd is not None:
            _robot.pantilt(search_cmd)
            _controller.notify_external_pantilt(search_cmd["pan"], search_cmd["tilt"])
            _bump_sent()
    elif behavior == "AVOIDING":
        _robot.pantilt(_controller.center_command())
        _bump_sent()
    elif behavior in ("IDLE", "GUARDING"):
        # Only recenter when we just entered the state — avoid spamming
        # center commands every poll. Mirrors vision_service's previous
        # behavior.time_in_state() < 0.1 check.
        if time_in_state < 0.2:
            _robot.pantilt(_controller.center_command())
            _bump_sent()
    # MANUAL or unknown state: no-op (user is driving via gamepad / dpad)


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
    # directly to pantilt_service (:5006) without violating same-origin.
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

    def do_POST(self) -> None:
        if self.path == "/settings":
            body = self._read_json()
            if body is None:
                return
            try:
                from pantilt_node.runtime import apply_settings as _apply_pt
                from scout.config import TrackingConfig, SearchConfig
                # tracking_config + search_config are throwaway: tracking-side
                # settings (score_threshold etc.) belong to camera, not here.
                _apply_pt(body, _controller, TrackingConfig(), SearchConfig())
                self._json({"ok": True, "applied": list(body.keys())})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
        elif self.path == "/pantilt":
            body = self._read_json()
            if body is None:
                return
            try:
                from pantilt_node.runtime import handle_manual_pantilt
                result, status = handle_manual_pantilt(body, _robot, _controller)
                self._json(result, status=status)
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        if self.path == "/health":
            with _state_lock:
                age = (time.time() - _last_meta_fetch_ts) if _last_meta_fetch_ts > 0 else None
                self._json({
                    "ok": True,
                    "service": "pantilt_node",
                    "vision_url": _VISION_URL,
                    "robot_api_url": _ROBOT_API_URL,
                    "poll_interval_s": _POLL_INTERVAL_S,
                    "last_meta_fetch_age_s": age,
                    "polls_ok": _status["polls_ok"],
                    "polls_fail": _status["polls_fail"],
                    "commands_sent": _status["commands_sent"],
                    "controller_pan_est": _controller._estimated_pan,
                    "controller_tilt_est": _controller._estimated_tilt,
                    "current_behavior": _last_behavior_state,
                })
        elif self.path == "/status":
            with _state_lock:
                self._json({
                    "ok": True,
                    "tracking_state": _latest_tracking_state,
                    "status": dict(_status),
                    "controller": {
                        "estimated_pan": _controller._estimated_pan,
                        "estimated_tilt": _controller._estimated_tilt,
                        "commanded_pan": _controller._commanded_pan,
                        "commanded_tilt": _controller._commanded_tilt,
                    },
                })
        elif self.path == "/":
            self._json({"service": "pantilt_node", "endpoints": ["/health", "/status"]})
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)


def main() -> None:
    log.info(
        "pantilt service starting on http://0.0.0.0:%d (vision=%s, robot_api=%s)",
        _PANTILT_PORT, _VISION_URL, _ROBOT_API_URL,
    )
    threading.Thread(target=_poll_loop, daemon=True, name="pantilt-poll").start()
    server = ThreadingHTTPServer(("0.0.0.0", _PANTILT_PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
