"""Rover runtime helpers for wheel and collision endpoints."""
from __future__ import annotations


def handle_manual_move(body: dict, robot) -> tuple[dict, int]:
    if robot is None:
        return {"ok": False, "error": "not_initialized"}, 503
    x = max(-500, min(500, int(body.get("x", 0))))
    z = max(-500, min(500, int(body.get("z", 0))))
    ok = robot.move(x, z)
    return {"ok": ok, "x": x, "z": z}, 200


def set_collision(body: dict, collision_guard) -> tuple[dict, int]:
    if "enabled" in body:
        collision_guard.enabled = bool(body["enabled"])
    if "height_threshold" in body:
        collision_guard.height_threshold = float(body["height_threshold"])
    if "center_zone_fraction" in body:
        collision_guard.center_zone_fraction = float(body["center_zone_fraction"])
    return {"ok": True}, 200


def apply_settings(body: dict, controller) -> None:
    if controller is None:
        return
    if "wheel_enabled" in body:
        controller.config.wheel_enabled = bool(body["wheel_enabled"])
    for key in (
        "follow_forward_speed",
        "follow_steer_gain",
        "follow_target_bbox_ratio",
        "close_target_bbox_ratio",
        "follow_deadzone_ratio",
    ):
        if key in body:
            setattr(controller.config, key, float(body[key]))
