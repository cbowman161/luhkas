"""UART proxy backend: read battery telemetry produced by robot_api's serial reader.

On scout the battery voltage rides on the same UART that carries motor and
IMU telemetry, owned by robot_api. Rather than duplicate the serial port, the
robot_api writes each parsed reading to a small JSON file. This backend
reads that file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .base import BatteryBackend, BatteryReading


def _default_path() -> str:
    """Per-user runtime dir (writable by the service user) under tmpfs.

    Falls back to /tmp on systems without ``XDG_RUNTIME_DIR``.
    """
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return f"{base}/luhkas-battery.json"


DEFAULT_PATH = os.environ.get("BATTERY_UART_PROXY_PATH") or _default_path()
DEFAULT_MAX_AGE = float(os.environ.get("BATTERY_UART_PROXY_MAX_AGE", "10"))


class UartProxyBackend(BatteryBackend):
    name = "uart_proxy"

    def __init__(self, path: str = DEFAULT_PATH, max_age_s: float = DEFAULT_MAX_AGE) -> None:
        self.path = Path(path)
        self.max_age_s = max_age_s

    def read(self) -> Optional[BatteryReading]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        ts = float(raw.get("timestamp") or 0.0)
        if ts and (time.time() - ts) > self.max_age_s:
            return None
        try:
            voltage = float(raw.get("voltage", 0.0))
            percent = int(raw.get("percent", 0))
        except (TypeError, ValueError):
            return None
        return BatteryReading(
            voltage=voltage,
            percent=percent,
            current_a=None,
            charging=None,
            source=self.name,
            timestamp=ts or time.time(),
        )
