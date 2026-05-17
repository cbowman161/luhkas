from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class BehaviorState(Enum):
    IDLE      = auto()   # camera at rest, not tracking, not guarding
    FOLLOWING = auto()   # target acquired, pan-tilt + optional wheel follow active
    SEARCHING = auto()   # target lost, executing search pattern
    GUARDING  = auto()   # guard mode on, no tracking target, monitoring
    AVOIDING  = auto()   # collision blocked, pausing/backing off
    MANUAL    = auto()   # manual controller owns wheels / camera


@dataclass
class BehaviorConfig:
    search_timeout_seconds: float = float(os.getenv("SCOUT_SEARCH_TIMEOUT",  "30"))
    avoid_duration_seconds: float = float(os.getenv("SCOUT_AVOID_DURATION",  "3"))
    enabled:                bool  = bool(int(os.getenv("SCOUT_BEHAVIOR_ENABLED", "1")))


class BehaviorStateMachine:
    def __init__(self, config: BehaviorConfig):
        self.state = BehaviorState.IDLE
        self._config = config
        self._entered_at: float = time.monotonic()

    def update(self, *, target, tracking_enabled: bool,
               guard_enabled: bool, collision_blocked: bool,
               manual_enabled: bool = False, search_enabled: bool = True) -> BehaviorState:
        if not self._config.enabled:
            return self.state
        new = self._next(target, tracking_enabled, guard_enabled, collision_blocked, manual_enabled, search_enabled)
        if new != self.state:
            self.state = new
            self._entered_at = time.monotonic()
        return self.state

    def _next(self, target, tracking_enabled, guard_enabled, collision_blocked, manual_enabled, search_enabled) -> BehaviorState:
        age = time.monotonic() - self._entered_at

        if manual_enabled:
            return BehaviorState.MANUAL

        # Collision overrides following
        if self.state == BehaviorState.FOLLOWING and collision_blocked:
            return BehaviorState.AVOIDING

        if self.state == BehaviorState.AVOIDING:
            if not collision_blocked:
                if tracking_enabled and target:   return BehaviorState.FOLLOWING
                if tracking_enabled and search_enabled: return BehaviorState.SEARCHING
                return BehaviorState.IDLE
            if age > self._config.avoid_duration_seconds:
                return BehaviorState.SEARCHING if tracking_enabled and search_enabled else BehaviorState.IDLE
            return BehaviorState.AVOIDING

        if not tracking_enabled and not guard_enabled:
            return BehaviorState.IDLE

        if tracking_enabled and target:
            return BehaviorState.FOLLOWING

        if tracking_enabled and not target:
            if not search_enabled:
                return BehaviorState.IDLE
            if self.state == BehaviorState.SEARCHING and age > self._config.search_timeout_seconds:
                return BehaviorState.IDLE
            return BehaviorState.SEARCHING

        if guard_enabled and not tracking_enabled:
            return BehaviorState.GUARDING

        return self.state

    @property
    def state_name(self) -> str:
        return self.state.name

    def time_in_state(self) -> float:
        return time.monotonic() - self._entered_at
