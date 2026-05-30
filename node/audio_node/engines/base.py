"""STT and TTS engine interfaces for audio_node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class STTResult:
    text: str
    confidence: Optional[float] = None
    final: bool = True


class STTEngine:
    """Streaming STT engine.

    The capture loop pushes raw 16-bit little-endian mono PCM frames at a
    fixed sample rate (default 16 kHz). Engines decide their own utterance
    boundaries and yield STTResult objects with ``final=True`` per complete
    utterance.
    """

    name: str = "base"
    sample_rate: int = 16000
    available: bool = False

    def accept(self, pcm: bytes) -> Optional[STTResult]:
        return None

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class TTSEngine:
    """Text-to-speech engine.

    ``speak`` blocks until playback finishes (so the service's request queue
    naturally backs up if speech is in progress — no separate lock needed).
    """

    name: str = "base"
    available: bool = False

    def speak(self, text: str) -> None:
        raise NotImplementedError

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        pass


class NullSTT(STTEngine):
    name = "none"
    available = False


class NullTTS(TTSEngine):
    name = "none"
    available = False

    def speak(self, text: str) -> None:
        return
