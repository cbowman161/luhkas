from __future__ import annotations

import math
import time

from .config import MotionConfig
from .types import Detection


class PanTiltController:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self._last_command_at = 0.0
        self._last_edge_reacquire_at = 0.0
        self._moving = False
        self._last_pan = 0
        self._last_tilt = 0
        self._estimated_pan = 0
        self._estimated_tilt = 0
        self._commanded_pan = 0.0
        self._commanded_tilt = 0.0
        self._recent_base_turn = 0
        self._aim_x: float | None = None
        self._aim_y: float | None = None
        self._err_x: float | None = None
        self._err_y: float | None = None
        self._last_reverse_at_x = 0.0
        self._last_reverse_at_y = 0.0
        self._pan_integral = 0.0
        self._settled = False
        self._had_target = False

    def command_for_target(
        self,
        target: Detection | None,
        frame_w: int,
        frame_h: int,
    ) -> dict | None:
        if not self.config.enabled:
            return None

        now = time.time()
        if now - self._last_command_at < self.config.command_interval_seconds:
            return None

        self._last_command_at = now

        if target is None:
            self._aim_x = None
            self._aim_y = None
            self._err_x = None
            self._err_y = None
            self._pan_integral = 0.0
            self._settled = False
            self._had_target = False
            if self._moving:
                self._moving = False
                self._last_pan = 0
                self._last_tilt = 0
                return {"mode": "stop"}
            return None

        if not self._had_target:
            # Sync commanded position to estimated on reacquisition so the first
            # absolute command doesn't jump from a stale accumulated value.
            self._commanded_pan = self._estimated_pan
            self._commanded_tilt = self._estimated_tilt
            self._had_target = True

        cx, cy = self._smoothed_aim(target.aim_point)
        if not target.predicted:
            cx += target.vx * self.config.target_velocity_lead_x_seconds
            cy += target.vy * self.config.target_velocity_lead_y_seconds
        if target.aim_source == "bbox":
            _, _, _, h = target.bbox
            cy -= h * self.config.target_head_offset_ratio

        raw_err_x = (cx - frame_w / 2.0) / max(frame_w / 2.0, 1)
        raw_err_y = -(cy - frame_h / 2.0) / max(frame_h / 2.0, 1)
        raw_pan_degrees, raw_tilt_degrees = self._pixel_to_angle(cx, cy, frame_w, frame_h)
        angular_distance = math.hypot(raw_pan_degrees, raw_tilt_degrees)
        command_scale, settle_scale = self._target_size_scales(target, frame_h)
        settle_enter = self.config.settle_enter_degrees * settle_scale
        settle_exit = max(settle_enter, self.config.settle_exit_degrees * settle_scale)
        err_x, err_y = self._smoothed_error(raw_err_x, raw_err_y)

        if abs(err_x) < self.config.deadzone_x:
            err_x = 0.0
        if abs(err_y) < self.config.deadzone_y:
            err_y = 0.0

        edge_command = self._edge_reacquire_command(err_x, target.predicted, now)
        if edge_command is not None:
            self._moving = True
            self._settled = False
            return edge_command

        # Apply settle hysteresis for both absolute and relative modes.
        # Without this, absolute mode micro-corrects endlessly near the target.
        settle_command = self._settle_command(
            angular_distance=angular_distance,
            settle_enter=settle_enter,
            settle_exit=settle_exit,
            predicted=target.predicted,
        )
        if settle_command == "hold":
            return None
        if isinstance(settle_command, dict):
            return settle_command

        if self.config.pantilt_mode == "absolute":
            return self._absolute_command(
                err_x=err_x,
                err_y=err_y,
                raw_err_x=raw_err_x,
                raw_err_y=raw_err_y,
                aim_x=cx,
                aim_y=cy,
                target=target,
            )

        pan_integral = self._pan_integral_command(err_x)
        pan = self._angular_axis_command(raw_pan_degrees, self.config.angular_pan_gain, command_scale) + pan_integral
        pan = max(-self.config.max_command, min(self.config.max_command, pan))
        tilt = self._angular_axis_command(raw_tilt_degrees, self.config.angular_tilt_gain, command_scale)
        pan = self._smooth_command(pan, self._last_pan, now, "x")
        tilt = self._smooth_command(tilt, self._last_tilt, now, "y")
        self._last_pan = pan
        self._last_tilt = tilt
        self._estimated_pan = self._clamp_pan(self._estimated_pan + pan * self.config.pan_estimate_scale)
        self._estimated_tilt = self._clamp_tilt(self._estimated_tilt + tilt * self.config.tilt_estimate_scale)
        self._recent_base_turn = 0

        if pan == 0 and tilt == 0:
            if self._moving:
                self._moving = False
                self._last_pan = 0
                self._last_tilt = 0
                return {"mode": "stop"}
            return None

        self._moving = True
        return {
            "mode": "relative",
            "pan": pan,
            "tilt": tilt,
            "sx": self.config.relative_speed_x,
            "sy": self.config.relative_speed_y,
            "raw_error_x": raw_err_x,
            "raw_error_y": raw_err_y,
            "error_x": err_x,
            "error_y": err_y,
            "pan_degrees": raw_pan_degrees,
            "tilt_degrees": raw_tilt_degrees,
            "angular_distance": angular_distance,
            "settle_enter_degrees": settle_enter,
            "settle_exit_degrees": settle_exit,
            "command_scale": command_scale,
            "pan_integral": pan_integral,
            "aim_x": cx,
            "aim_y": cy,
            "target_id": target.id,
            "target_memory_id": target.memory_id,
            "aim_source": target.aim_source,
        }

    def center_command(self) -> dict:
        self._moving = False
        self._last_pan = 0
        self._last_tilt = 0
        self._estimated_pan = 0
        self._estimated_tilt = 0
        self._commanded_pan = 0.0
        self._commanded_tilt = 0.0
        self._recent_base_turn = 0
        self._aim_x = None
        self._aim_y = None
        self._err_x = None
        self._err_y = None
        self._pan_integral = 0.0
        self._settled = False
        self._last_command_at = time.time()
        return {"mode": "absolute", "pan": 0, "tilt": 0, "spd": 0, "acc": 0}

    def ego_motion(self) -> dict:
        return {
            "estimated_pan": float(self._estimated_pan),
            "estimated_tilt": float(self._estimated_tilt),
            "commanded_pan": float(self._commanded_pan),
            "commanded_tilt": float(self._commanded_tilt),
            "recent_base_turn": float(self._recent_base_turn),
            "wheel_enabled": bool(self.config.wheel_enabled),
        }

    def _absolute_command(
        self,
        err_x: float,
        err_y: float,
        raw_err_x: float,
        raw_err_y: float,
        aim_x: float,
        aim_y: float,
        target: Detection,
    ) -> dict | str | None:
        target_distance = math.hypot(err_x, err_y)
        distance_multiplier = min(
            self.config.absolute_distance_max_multiplier,
            1.0 + target_distance * self.config.absolute_distance_gain,
        )
        pan_step = self._absolute_step(err_x, self.config.absolute_pan_gain * distance_multiplier)
        tilt_step = self._absolute_step(err_y, self.config.absolute_tilt_gain * distance_multiplier)
        if target.predicted:
            half_max = abs(self.config.absolute_max_step) / 2.0
            pan_step = max(-half_max, min(half_max, pan_step))
            tilt_step = max(-half_max, min(half_max, tilt_step))
        next_pan = self._clamp_pan(self._commanded_pan + pan_step)
        next_tilt = self._clamp_tilt(self._commanded_tilt + tilt_step)
        pan_changed = abs(next_pan - self._commanded_pan) >= 0.5
        tilt_changed = abs(next_tilt - self._commanded_tilt) >= 0.5

        if not pan_changed and not tilt_changed:
            if self._moving:
                self._moving = False
                return {"mode": "stop"}
            return None

        self._commanded_pan = next_pan
        self._commanded_tilt = next_tilt
        self._estimated_pan = next_pan
        self._estimated_tilt = next_tilt
        self._last_pan = int(round(pan_step))
        self._last_tilt = int(round(tilt_step))
        self._recent_base_turn = 0
        self._moving = True
        return {
            "mode": "absolute",
            "pan": int(round(next_pan)),
            "tilt": int(round(next_tilt)),
            "spd": self.config.absolute_speed,
            "acc": self.config.absolute_accel,
            "pan_step": pan_step,
            "tilt_step": tilt_step,
            "target_distance": target_distance,
            "distance_multiplier": distance_multiplier,
            "raw_error_x": raw_err_x,
            "raw_error_y": raw_err_y,
            "error_x": err_x,
            "error_y": err_y,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "target_id": target.id,
            "target_memory_id": target.memory_id,
            "aim_source": target.aim_source,
        }

    def _absolute_step(self, error: float, gain: float) -> float:
        if error == 0.0:
            return 0.0
        step = error * gain
        min_step = abs(self.config.absolute_min_step)
        max_step = abs(self.config.absolute_max_step)
        if abs(step) < min_step:
            step = math.copysign(min_step, step)
        return max(-max_step, min(max_step, step))

    def _angular_axis_command(self, degrees: float, gain: float, command_scale: float) -> int:
        value = int(round(degrees * gain * command_scale))
        if value == 0:
            return 0
        value = int(math.copysign(max(abs(value), self.config.min_command), value))
        return max(-self.config.max_command, min(self.config.max_command, value))

    def _settle_command(
        self,
        angular_distance: float,
        settle_enter: float,
        settle_exit: float,
        predicted: bool,
    ) -> dict | None:
        if predicted:
            self._settled = False
            return None

        if self._settled:
            if angular_distance < settle_exit:
                self._pan_integral = 0.0
                if self._moving:
                    self._moving = False
                    self._last_pan = 0
                    self._last_tilt = 0
                    return {"mode": "stop"}
                return "hold"
            self._settled = False
            return None

        if angular_distance <= settle_enter:
            self._settled = True
            self._pan_integral = 0.0
            if self._moving:
                self._moving = False
                self._last_pan = 0
                self._last_tilt = 0
                return {"mode": "stop"}
            return "hold"
        return None

    def _pixel_to_angle(self, x: float, y: float, frame_w: int, frame_h: int) -> tuple[float, float]:
        half_w = max(frame_w / 2.0, 1.0)
        half_h = max(frame_h / 2.0, 1.0)
        nx = (x - half_w) / half_w
        ny = (y - half_h) / half_h
        h_fov = math.radians(max(1.0, min(179.0, self.config.camera_horizontal_fov_degrees)))
        v_fov = math.radians(max(1.0, min(179.0, self.config.camera_vertical_fov_degrees)))
        pan = math.degrees(math.atan(math.tan(h_fov / 2.0) * nx))
        tilt = -math.degrees(math.atan(math.tan(v_fov / 2.0) * ny))
        return pan, tilt

    def _target_size_scales(self, target: Detection, frame_h: int) -> tuple[float, float]:
        _, _, _, bbox_h = target.bbox
        ratio = bbox_h / max(frame_h, 1)
        if ratio >= self.config.close_target_bbox_ratio:
            return self.config.close_target_command_scale, self.config.close_target_settle_scale
        if ratio <= self.config.far_target_bbox_ratio:
            return self.config.far_target_command_scale, self.config.far_target_settle_scale
        return 1.0, 1.0

    def _pan_integral_command(self, error: float) -> int:
        decay = max(0.0, min(1.0, self.config.pan_integral_decay))
        if error == 0.0:
            self._pan_integral *= decay
        else:
            if self._pan_integral and (self._pan_integral > 0) != (error > 0):
                self._pan_integral = 0.0
            self._pan_integral = self._pan_integral * decay + error * self.config.pan_integral_gain

        limit = abs(self.config.pan_integral_limit)
        self._pan_integral = max(-limit, min(limit, self._pan_integral))
        if abs(self._pan_integral) < 1.0:
            return 0
        return int(round(self._pan_integral))

    def _smooth_command(self, desired: int, previous: int, now: float, axis: str) -> int:
        alpha = max(0.0, min(1.0, self.config.command_smoothing))
        smoothed = int(round(previous * (1.0 - alpha) + desired * alpha))
        if self._is_recent_reverse(now, axis):
            return 0
        if previous and desired and (previous > 0) != (desired > 0):
            if abs(desired) < self.config.sign_flip_deadband:
                self._mark_reverse(now, axis)
                return 0
            smoothed = int(math.copysign(min(abs(smoothed), self.config.sign_flip_deadband), desired))
            self._mark_reverse(now, axis)
            return smoothed
        delta = smoothed - previous
        if abs(delta) > self.config.max_command_step:
            smoothed = previous + int(math.copysign(self.config.max_command_step, delta))
        if desired == 0 and abs(smoothed) < self.config.min_command:
            return 0
        return smoothed

    def _smoothed_error(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        if self._err_x is None or self._err_y is None:
            self._err_x = raw_x
            self._err_y = raw_y
            return raw_x, raw_y

        alpha_x = max(0.0, min(1.0, self.config.error_smoothing_x))
        alpha_y = max(0.0, min(1.0, self.config.error_smoothing_y))
        self._err_x = self._err_x * (1.0 - alpha_x) + raw_x * alpha_x
        self._err_y = self._err_y * (1.0 - alpha_y) + raw_y * alpha_y
        return self._err_x, self._err_y

    def _smoothed_aim(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        if self._aim_x is None or self._aim_y is None:
            self._aim_x = x
            self._aim_y = y
            return x, y

        alpha_x = max(0.0, min(1.0, self.config.aim_smoothing_x))
        alpha_y = max(0.0, min(1.0, self.config.aim_smoothing_y))
        self._aim_x = self._aim_x * (1.0 - alpha_x) + x * alpha_x
        self._aim_y = self._aim_y * (1.0 - alpha_y) + y * alpha_y
        return self._aim_x, self._aim_y

    def _is_recent_reverse(self, now: float, axis: str) -> bool:
        last_reverse = self._last_reverse_at_x if axis == "x" else self._last_reverse_at_y
        return now - last_reverse < self.config.reverse_settle_seconds

    def _mark_reverse(self, now: float, axis: str) -> None:
        if axis == "x":
            self._last_reverse_at_x = now
        else:
            self._last_reverse_at_y = now

    def _edge_reacquire_command(self, err_x: float, predicted: bool, now: float) -> dict | None:
        if not self.config.edge_reacquire_enabled or not predicted:
            return None
        if now - self._last_edge_reacquire_at < self.config.edge_reacquire_cooldown:
            return None

        at_right_limit = self._estimated_pan >= self.config.estimated_pan_max - self.config.pan_limit_margin
        at_left_limit = self._estimated_pan <= self.config.estimated_pan_min + self.config.pan_limit_margin

        if err_x > self.config.deadzone_x and at_right_limit:
            self._last_edge_reacquire_at = now
            return self._compound_reacquire(turn_z=abs(self.config.edge_reacquire_base_z), reset_pan=-abs(self.config.edge_reacquire_reset_pan))

        if err_x < -self.config.deadzone_x and at_left_limit:
            self._last_edge_reacquire_at = now
            return self._compound_reacquire(turn_z=-abs(self.config.edge_reacquire_base_z), reset_pan=abs(self.config.edge_reacquire_reset_pan))

        return None

    def _compound_reacquire(self, turn_z: int, reset_pan: int) -> dict:
        reset_pan = self._clamp_pan(reset_pan)
        self._estimated_pan = reset_pan
        self._estimated_tilt = 0
        self._commanded_pan = reset_pan
        self._commanded_tilt = 0.0
        self._recent_base_turn = turn_z
        self._last_pan = 0
        self._last_tilt = 0
        return {
            "mode": "edge_reacquire",
            "pantilt": {"mode": "absolute", "pan": reset_pan, "tilt": 0, "spd": 0, "acc": 0},
            "move": {"x": 0, "z": turn_z} if self.config.wheel_enabled else None,
            "move_stop_after": self.config.edge_reacquire_base_pulse_seconds,
            "reason": "pan_limit_target_continued",
            "estimated_pan": self._estimated_pan,
            "wheel_enabled": bool(self.config.wheel_enabled),
        }

    def wheel_follow_command(self, target: Detection | None, frame_w: int, frame_h: int) -> dict | None:
        """Return a wheel move command to follow the target at a comfortable distance.

        Uses bbox height ratio as a distance proxy and _estimated_pan for steering.
        Returns None when follow is disabled or no target.
        """
        if not self.config.follow_enabled or target is None:
            return None

        _, _, _, bbox_h = target.bbox
        bbox_ratio = bbox_h / max(frame_h, 1)
        distance_error = self.config.follow_target_bbox_ratio - bbox_ratio

        if abs(distance_error) < self.config.follow_deadzone_ratio:
            x = 0
        elif distance_error > 0:
            scale = min(1.0, distance_error / self.config.follow_target_bbox_ratio)
            x = int(self.config.follow_forward_speed * scale)
        else:
            x = 0

        z = int(self._estimated_pan * self.config.follow_steer_gain)
        z = max(-500, min(500, z))

        return {"x": x, "z": z}

    def notify_external_pantilt(self, pan: float, tilt: float) -> None:
        """Sync estimated position after an external (search/scan) absolute pantilt command."""
        self._estimated_pan = self._clamp_pan(pan)
        self._estimated_tilt = self._clamp_tilt(tilt)
        self._commanded_pan = self._estimated_pan
        self._commanded_tilt = self._estimated_tilt

    def _clamp_pan(self, value: float) -> float:
        return max(self.config.estimated_pan_min, min(self.config.estimated_pan_max, value))

    def _clamp_tilt(self, value: float) -> float:
        return max(self.config.estimated_tilt_min, min(self.config.estimated_tilt_max, value))
