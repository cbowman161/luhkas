from __future__ import annotations

import math
import time

from .config import SearchConfig


class SearchPhase:
    IDLE = "idle"
    SWEEP = "sweep"
    SCAN = "scan"


class SearchController:
    """Drives the pan/tilt camera to search for a person when tracking is lost.

    Phases:
    - IDLE: has a live target, nothing to do
    - SWEEP: target prediction expired; pan quickly toward where the target was heading
    - SCAN: sweep found nobody; smooth sinusoidal pan/tilt pattern covering the space
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self._phase = SearchPhase.IDLE
        self._phase_start = 0.0
        self._sweep_target_pan = 0
        self._last_command_at = 0.0

        pans = [int(x) for x in self.config.scan_pan_positions.split(",") if x.strip()]
        tilts = [int(x) for x in self.config.scan_tilt_positions.split(",") if x.strip()]
        self._scan_pan_amp = max(abs(p) for p in pans) if pans else 90
        self._scan_tilt_min = min(tilts) if tilts else 0
        self._scan_tilt_max = max(tilts) if tilts else 20

    @property
    def phase(self) -> str:
        return self._phase

    def on_target_acquired(self) -> None:
        self._phase = SearchPhase.IDLE

    def on_target_lost(self, estimated_pan: float, last_vx: float = 0.0) -> None:
        if not self.config.enabled:
            self._phase = SearchPhase.IDLE
            return
        if self._phase != SearchPhase.IDLE:
            return
        self._phase = SearchPhase.SWEEP
        self._phase_start = time.time()
        if abs(last_vx) > 1.0:
            direction = 1 if last_vx > 0 else -1
        else:
            direction = -1 if estimated_pan > 0 else 1
        target = estimated_pan + direction * self.config.sweep_pan_amount
        self._sweep_target_pan = int(max(-150, min(150, target)))

    def search_command(self, current_pan: float, current_tilt: float) -> dict | None:
        if not self.config.enabled or self._phase == SearchPhase.IDLE:
            return None

        now = time.time()
        interval = (
            self.config.scan_command_interval_seconds
            if self._phase == SearchPhase.SCAN
            else self.config.command_interval_seconds
        )
        if now - self._last_command_at < interval:
            return None

        cmd = self._build_command(now)
        if cmd is not None:
            self._last_command_at = now
        return cmd

    def _build_command(self, now: float) -> dict | None:
        if self._phase == SearchPhase.SWEEP:
            if now - self._phase_start > self.config.sweep_duration_seconds:
                self._phase = SearchPhase.SCAN
                self._phase_start = now
                return self._scan_command(now)
            return {"mode": "absolute", "pan": self._sweep_target_pan, "tilt": 0, "spd": 0, "acc": 0}

        if self._phase == SearchPhase.SCAN:
            return self._scan_command(now)

        return None

    def _scan_command(self, now: float) -> dict | None:
        t = now - self._phase_start
        pan = self._scan_pan_amp * math.sin(2 * math.pi * t / self.config.scan_pan_period_seconds)
        tilt_raw = math.sin(2 * math.pi * t / self.config.scan_tilt_period_seconds)
        tilt = self._scan_tilt_min + (self._scan_tilt_max - self._scan_tilt_min) * (tilt_raw + 1) / 2
        return {"mode": "absolute", "pan": int(round(pan)), "tilt": int(round(tilt)), "spd": 0, "acc": 0}
