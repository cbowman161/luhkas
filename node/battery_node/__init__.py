"""Battery monitoring node module.

Owns the canonical battery state for a node and exposes it over HTTP on
the node's battery service port. Backend is pluggable (UART proxy for
scout, INA219 over I2C for nodes with a UPS HAT).
"""

from .commands import capabilities, handle, health, BatteryCommandConfig

__all__ = ["capabilities", "handle", "health", "BatteryCommandConfig"]
