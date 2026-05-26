"""Battery backends for battery_node."""

from .base import BatteryBackend, BatteryReading

__all__ = ["BatteryBackend", "BatteryReading"]


_AUTO_ORDER = ("uart_proxy", "max17040", "ina219")


def load_backend(name: str, **kwargs) -> BatteryBackend:
    """Return a backend instance by short name, or auto-pick when ``name`` is empty or ``auto``."""
    n = (name or "").strip().lower()
    if n in {"", "auto"}:
        return AutoBackend(**kwargs)
    if n in {"uart", "uart_proxy", "file"}:
        from .uart_proxy import UartProxyBackend
        return UartProxyBackend(**kwargs)
    if n in {"ina219"}:
        from .ina219 import Ina219Backend
        return Ina219Backend(**kwargs)
    if n in {"max17040", "geekworm", "geekworm_x120", "ups_hat"}:
        from .max17040 import Max17040Backend
        return Max17040Backend(**kwargs)
    raise ValueError(f"unknown battery backend: {n!r}")


class AutoBackend(BatteryBackend):
    """Try each backend in order; on every read, use the first that returns a reading.

    Lets a profile omit ``BATTERY_BACKEND`` entirely — scout's robot_api
    publishes UART readings to a file (the ``uart_proxy`` backend picks
    that up); a node with a UPS HAT exposes itself over I2C and gets
    discovered by ``max17040`` or ``ina219``.
    """

    name = "auto"
    available = True

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._backends = []
        for n in _AUTO_ORDER:
            try:
                self._backends.append(load_backend(n, **kwargs))
            except Exception:
                # Backend import or init failed; skip it.
                continue
        # ``name`` flips to whichever backend actually produced the last
        # reading, so /health surfaces what's really being used.
        self.active_name = "auto"

    def read(self):
        for backend in self._backends:
            try:
                reading = backend.read()
            except Exception:
                continue
            if reading is not None:
                self.active_name = backend.name
                return reading
        return None

    def close(self) -> None:
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                pass
