"""INA219 backend for Pi UPS HATs.

Reads bus voltage, shunt voltage and current from the INA219 on the
configured I2C bus and address. Estimates percentage from a configurable
voltage range (defaults match a 2S Li-Ion pack at ~6.0V empty / 8.4V full,
common on Pi UPS HATs).

Configuration via env vars (so the same systemd unit works across packs):
  BATTERY_I2C_BUS         (default: 1)
  BATTERY_INA219_ADDR     (default: 0x42 — common on Waveshare Pi 5 UPS)
  BATTERY_V_EMPTY         (default: 6.0)
  BATTERY_V_FULL          (default: 8.4)

Falls back to None if the smbus2 module or device is unavailable, so
battery_node still starts on dev machines.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import BatteryBackend, BatteryReading


_REG_CONFIG = 0x00
_REG_SHUNT_VOLT = 0x01
_REG_BUS_VOLT = 0x02
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int, base: int = 0) -> int:
    raw = os.environ.get(name, "") or str(default)
    try:
        return int(raw, base) if base else int(raw, 0)
    except ValueError:
        return default


class Ina219Backend(BatteryBackend):
    name = "ina219"

    def __init__(
        self,
        bus: Optional[int] = None,
        address: Optional[int] = None,
        v_empty: Optional[float] = None,
        v_full: Optional[float] = None,
    ) -> None:
        self.bus_id = bus if bus is not None else _env_int("BATTERY_I2C_BUS", 1)
        self.address = address if address is not None else _env_int("BATTERY_INA219_ADDR", 0x42)
        self.v_empty = v_empty if v_empty is not None else _env_float("BATTERY_V_EMPTY", 6.0)
        self.v_full = v_full if v_full is not None else _env_float("BATTERY_V_FULL", 8.4)
        self._bus = None
        self._init_error: Optional[str] = None
        self._configure()

    def _configure(self) -> None:
        try:
            import smbus2  # type: ignore
        except Exception as exc:
            self._init_error = f"smbus2 unavailable: {exc}"
            return
        try:
            self._bus = smbus2.SMBus(self.bus_id)
            # 32V FSR, 320mV shunt range, 12-bit, continuous bus + shunt:
            self._write_u16(_REG_CONFIG, 0x399F)
            # Calibration for 0.1 ohm shunt, 100mA per bit current LSB is fine here;
            # users who need accurate current can override via CALIBRATION env later.
            self._write_u16(_REG_CALIBRATION, 4096)
        except Exception as exc:
            self._init_error = f"INA219 init failed: {exc}"
            self._bus = None

    def _write_u16(self, register: int, value: int) -> None:
        if self._bus is None:
            return
        hi = (value >> 8) & 0xFF
        lo = value & 0xFF
        self._bus.write_i2c_block_data(self.address, register, [hi, lo])

    def _read_u16(self, register: int) -> int:
        if self._bus is None:
            raise RuntimeError(self._init_error or "i2c bus not open")
        data = self._bus.read_i2c_block_data(self.address, register, 2)
        return (data[0] << 8) | data[1]

    def _read_s16(self, register: int) -> int:
        raw = self._read_u16(register)
        return raw - 0x10000 if raw & 0x8000 else raw

    def read(self) -> Optional[BatteryReading]:
        if self._bus is None:
            return None
        try:
            bus_raw = self._read_u16(_REG_BUS_VOLT)
            voltage = (bus_raw >> 3) * 0.004  # bus voltage LSB is 4mV, lower 3 bits are flags
            current_raw = self._read_s16(_REG_CURRENT)
            current_a = current_raw * 0.0001  # with calibration 4096 / 0.1Ω shunt: 0.1mA/bit
        except Exception:
            return None
        span = max(0.001, self.v_full - self.v_empty)
        percent = max(0, min(100, int(round((voltage - self.v_empty) / span * 100))))
        return BatteryReading(
            voltage=voltage,
            percent=percent,
            current_a=current_a,
            charging=current_a > 0.02,
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
