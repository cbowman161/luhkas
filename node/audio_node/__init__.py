"""Audio node module: mic→STT→presence and presence-response→TTS→speaker."""

from .commands import capabilities, handle, health, AudioCommandConfig

__all__ = ["capabilities", "handle", "health", "AudioCommandConfig"]
