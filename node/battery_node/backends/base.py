"""Battery backend interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatteryReading:
    voltage: float
    percent: int
    current_a: Optional[float] = None
    charging: Optional[bool] = None
    source: str = ""
    timestamp: float = 0.0

    def as_dict(self) -> dict:
        d = {"voltage": round(self.voltage, 3), "percent": int(self.percent)}
        if self.current_a is not None:
            d["current_a"] = round(self.current_a, 3)
        if self.charging is not None:
            d["charging"] = bool(self.charging)
        if self.source:
            d["source"] = self.source
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


class BatteryBackend:
    """Pluggable battery source.

    Subclasses implement ``read()`` returning the latest BatteryReading, or
    None if no fresh reading is available. ``name`` is used in selftests and
    /battery responses so consumers can tell where the number came from.
    """

    name: str = "base"

    def read(self) -> Optional[BatteryReading]:
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 — optional override
        pass
