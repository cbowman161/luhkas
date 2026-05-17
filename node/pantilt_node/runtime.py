"""Pan/tilt runtime helpers for service endpoints and tracking commands."""
from __future__ import annotations

import time
from collections.abc import Callable


def dispatch_robot_command(robot, command: dict) -> bool:
    if command.get("mode") == "edge_reacquire":
        pantilt_ok = robot.pantilt(command["pantilt"])
        move = command.get("move", {})
        move_ok = True
        if move:
            move_ok = robot.move(move.get("x", 0), move.get("z", 0))
            stop_after = float(command.get("move_stop_after", 0))
            if stop_after > 0:
                time.sleep(stop_after)
                move_ok = robot.move(0, 0) and move_ok
        return pantilt_ok and move_ok
    return robot.pantilt(command)


def handle_manual_pantilt(body: dict, robot, controller) -> tuple[dict, int]:
    if body.get("center"):
        if robot and controller:
            robot.pantilt(controller.center_command())
        return {"ok": True}, 200
    if robot is None:
        return {"ok": False, "error": "not_initialized"}, 503
    pan = int(body.get("pan", 0))
    tilt = int(body.get("tilt", 0))
    if pan == 0 and tilt == 0:
        robot.pantilt({"mode": "stop"})
    elif controller:
        next_pan = controller._clamp_pan(controller._estimated_pan + pan)
        next_tilt = controller._clamp_tilt(controller._estimated_tilt + tilt)
        robot.pantilt({"mode": "absolute", "pan": int(round(next_pan)), "tilt": int(round(next_tilt)), "spd": 0, "acc": 0})
        controller.notify_external_pantilt(next_pan, next_tilt)
    else:
        robot.pantilt({"mode": "relative", "pan": pan, "tilt": tilt, "sx": 800, "sy": 450})
    return {"ok": True}, 200


def set_tracking(
    body: dict,
    controller,
    tracking_config,
    disable_manual: Callable[[], None],
) -> tuple[dict, int]:
    if controller is None:
        return {"ok": False, "error": "not_initialized"}, 503
    if "enabled" in body:
        controller.config.enabled = bool(body["enabled"])
        if controller.config.enabled:
            disable_manual()
    if "follow" in body:
        controller.config.follow_enabled = bool(body["follow"])
    if "target_identity" in body:
        tracking_config.target_identity = body["target_identity"] or None
    return {"ok": True, "tracking_enabled": controller.config.enabled, "follow_enabled": controller.config.follow_enabled}, 200


def apply_settings(body: dict, controller, tracking_config, search_config) -> None:
    if controller is not None:
        for key in ("edge_reacquire_enabled",):
            if key in body:
                setattr(controller.config, key, bool(body[key]))

        for key in (
            "max_command",
            "min_command",
            "max_command_step",
            "absolute_max_step",
            "absolute_min_step",
            "estimated_pan_min",
            "estimated_pan_max",
            "estimated_tilt_min",
            "estimated_tilt_max",
            "pan_limit_margin",
        ):
            if key in body:
                setattr(controller.config, key, int(body[key]))

        for key in (
            "command_interval_seconds",
            "settle_enter_degrees",
            "settle_exit_degrees",
            "pan_estimate_scale",
            "tilt_estimate_scale",
            "absolute_pan_gain",
            "absolute_tilt_gain",
            "absolute_distance_gain",
            "absolute_distance_max_multiplier",
        ):
            if key in body:
                setattr(controller.config, key, float(body[key]))

    if "score_threshold" in body:
        tracking_config.score_threshold = float(body["score_threshold"])
    if "target_label" in body:
        tracking_config.target_label = str(body["target_label"]).strip() or "person"
    if "person_score_threshold" in body:
        tracking_config.person_score_threshold = float(body["person_score_threshold"])
    if "search_movement_enabled" in body:
        search_config.enabled = bool(body["search_movement_enabled"])
