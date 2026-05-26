"""Pluggable STT/TTS engines for audio_node."""

from .base import STTEngine, TTSEngine, STTResult

__all__ = ["STTEngine", "TTSEngine", "STTResult", "load_stt", "load_tts"]


def load_stt(name: str, **kwargs) -> STTEngine:
    name = (name or "").strip().lower()
    if name in {"", "vosk"}:
        from .vosk_engine import VoskEngine
        return VoskEngine(**kwargs)
    if name in {"none", "off", "disabled"}:
        from .base import NullSTT
        return NullSTT()
    raise ValueError(f"unknown STT engine: {name!r}")


def load_tts(name: str, **kwargs) -> TTSEngine:
    name = (name or "").strip().lower()
    if name in {"", "espeak", "espeak-ng"}:
        from .espeak import EspeakEngine
        return EspeakEngine(**kwargs)
    if name in {"piper"}:
        from .piper import PiperEngine
        return PiperEngine(**kwargs)
    if name in {"none", "off", "disabled"}:
        from .base import NullTTS
        return NullTTS()
    raise ValueError(f"unknown TTS engine: {name!r}")
