"""MAX17040 fuel-gauge backend.

Used by Geekworm X1200 / X1201 / X1202 Pi UPS HATs. The MAX17040 reports
cell voltage *and* state-of-charge directly — no voltage-curve guessing.

Registers (datasheet):
  0x02  VCELL    bus voltage; upper 12 bits, LSB = 1.25 mV
  0x04  SOC      state of charge; high byte = integer percent, low byte = 1/256ths
  0xFE  COMMAND  write 0x5400 to issue power-on-reset
  0xFC  CONFIG   includes RCOMP (battery characterization byte)

Configuration via env vars:
  BATTERY_I2C_BUS         (default: 1)
  BATTERY_MAX17040_ADDR   (default: 0x36)
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import BatteryBackend, BatteryReading


_REG_VCELL = 0x02
_REG_SOC = 0x04


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "") or str(default)
    try:
        return int(raw, 0)
    except ValueError:
        return default


class Max17040Backend(BatteryBackend):
    name = "max17040"

    def __init__(self, bus: Optional[int] = None, address: Optional[int] = None) -> None:
        self.bus_id = bus if bus is not None else _env_int("BATTERY_I2C_BUS", 1)
        self.address = address if address is not None else _env_int("BATTERY_MAX17040_ADDR", 0x36)
        self._bus = None
        self._init_error: Optional[str] = None
        self._open()

    def _open(self) -> None:
        try:
            import smbus2  # type: ignore
        except Exception as exc:
            self._init_error = f"smbus2 unavailable: {exc}"
            return
        try:
            self._bus = smbus2.SMBus(self.bus_id)
        except Exception as exc:
            self._init_error = f"i2c open failed: {exc}"
            self._bus = None

    def _read_u16(self, register: int) -> int:
        if self._bus is None:
            raise RuntimeError(self._init_error or "i2c bus not open")
        data = self._bus.read_i2c_block_data(self.address, register, 2)
        return (data[0] << 8) | data[1]

    def read(self) -> Optional[BatteryReading]:
        if self._bus is None:
            return None
        try:
            vcell_raw = self._read_u16(_REG_VCELL)
            soc_raw = self._read_u16(_REG_SOC)
        except Exception:
            return None
        # VCELL: upper 12 bits, LSB = 1.25 mV → volts.
        voltage = ((vcell_raw >> 4) & 0x0FFF) * 0.00125
        # SOC: high byte = whole percent, low byte = 1/256ths.
        percent_float = (soc_raw >> 8) + ((soc_raw & 0xFF) / 256.0)
        percent = max(0, min(100, int(round(percent_float))))
        return BatteryReading(
            voltage=voltage,
            percent=percent,
            current_a=None,
            charging=None,
            source=self.name,
            timestamp=time.time(),
        )

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
