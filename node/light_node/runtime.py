"""Runtime camera-light control for a node-local light."""
from __future__ import annotations

import time

import cv2
import numpy as np


class AutoLightRuntime:
    def __init__(self, vision_config) -> None:
        self.brightness: int = 0
        self.auto_enabled: bool = vision_config.camera_light_auto_enabled
        self.auto_brightness: int = max(0, min(255, vision_config.camera_light_auto_brightness))
        self.ambient_level: float | None = None
        self.dark_since: float | None = None
        self.last_command_at: float = 0.0
        self.last_probe_at: float = 0.0
        self.probe_started_at: float | None = None
        self.probe_original_brightness: int | None = None
        self.probe_brightness: int | None = None

    def update_auto(self, frame: np.ndarray, robot, vision_config) -> float:
        level = self.estimate_level(frame)
        if self.ambient_level is None:
            self.ambient_level = level
        else:
            self.ambient_level = self.ambient_level * 0.85 + level * 0.15

        if not self.auto_enabled or robot is None:
            self.clear_probe()
            return self.ambient_level

        now = time.monotonic()
        if self.probe_started_at is not None:
            if now - self.probe_started_at >= 2.0:
                if level >= self.off_threshold(vision_config):
                    self.last_probe_at = now
                elif self.probe_original_brightness is not None:
                    self.brightness = self.probe_original_brightness
                    robot.camera_light(self.brightness)
                    self.last_command_at = now
                    self.last_probe_at = now
                self.clear_probe()
            return self.ambient_level

        if self.ambient_level < vision_config.camera_light_low_threshold:
            if self.dark_since is None:
                self.dark_since = now
            if now - self.dark_since < 2.0:
                return self.ambient_level
            brightness = self.auto_brightness_for(self.ambient_level, vision_config)
            if abs(brightness - self.brightness) >= 16 and now - self.last_command_at >= 1.0:
                self.brightness = brightness
                robot.camera_light(self.brightness)
                self.last_command_at = now
                self.last_probe_at = now
        elif self.brightness > 0:
            self.dark_since = None
            if now - self.last_probe_at >= 60.0 and now - self.last_command_at >= 5.0:
                self.probe_original_brightness = self.brightness
                self.probe_brightness = self.brightness // 2
                self.probe_started_at = now
                self.brightness = self.probe_brightness
                robot.camera_light(self.brightness)
                self.last_command_at = now
        else:
            self.dark_since = None
        return self.ambient_level

    def clear_probe(self) -> None:
        self.probe_started_at = None
        self.probe_original_brightness = None
        self.probe_brightness = None

    def off_threshold(self, vision_config) -> float:
        return min(255.0, float(vision_config.camera_light_low_threshold) + 25.0)

    def auto_brightness_for(self, light_level: float, vision_config) -> int:
        trigger = max(1.0, float(vision_config.camera_light_low_threshold))
        darkness = 1.0 - max(0.0, min(trigger, light_level)) / trigger
        max_brightness = max(0, min(255, int(self.auto_brightness)))
        min_brightness = min(max_brightness, 80)
        return int(round(min_brightness + (max_brightness - min_brightness) * darkness))

    def set_auto_enabled(self, enabled: bool) -> None:
        self.auto_enabled = enabled
        self.dark_since = None
        self.last_probe_at = time.monotonic()
        self.clear_probe()

    def set_enabled(self, enabled: bool, robot) -> None:
        self.auto_enabled = False
        self.dark_since = None
        self.clear_probe()
        self.brightness = 255 if enabled else 0
        if robot:
            robot.camera_light(self.brightness)

    def set_brightness(self, brightness: int, robot) -> None:
        brightness = max(0, min(255, int(brightness)))
        if self.auto_enabled:
            self.auto_brightness = brightness
        else:
            self.brightness = brightness
            if robot:
                robot.camera_light(self.brightness)

    def adjust_manual(self, delta: int, robot) -> int:
        self.auto_enabled = False
        self.brightness = max(0, min(255, self.brightness + delta))
        if robot:
            robot.camera_light(self.brightness)
        return self.brightness

    def toggle_manual(self, robot) -> int:
        self.auto_enabled = False
        self.brightness = 0 if self.brightness > 0 else 255
        if robot:
            robot.camera_light(self.brightness)
        return self.brightness

    def status(self, vision_config) -> dict:
        return {
            "camera_light_enabled": self.brightness > 0,
            "camera_light_brightness": self.auto_brightness if self.auto_enabled else self.brightness,
            "camera_light_auto_enabled": self.auto_enabled,
            "camera_light_auto_brightness": self.auto_brightness,
            "ambient_light_level": self.ambient_level,
            "camera_light_low_threshold": vision_config.camera_light_low_threshold,
            "camera_light_high_threshold": self.off_threshold(vision_config),
            "camera_light_off_threshold": self.off_threshold(vision_config),
            "camera_light_trigger_threshold": vision_config.camera_light_low_threshold,
            "camera_light_probe_active": self.probe_started_at is not None,
            "camera_light_probe_brightness": self.probe_brightness,
            "camera_light_probe_original_brightness": self.probe_original_brightness,
        }

    @staticmethod
    def estimate_level(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
