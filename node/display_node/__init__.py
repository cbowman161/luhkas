"""Display node module: serves /ui and accepts events from other node services."""

from .commands import capabilities, handle, health, DisplayCommandConfig

__all__ = ["capabilities", "handle", "health", "DisplayCommandConfig"]
