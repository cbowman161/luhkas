"""Vosk streaming STT engine.

Lightweight Kaldi-based recognizer that runs fine on Pi 5 with the
``vosk-model-small-en-us`` model. Point at the model directory with
``AUDIO_VOSK_MODEL`` (defaults to ``~/.vosk-model``).

Install (handled by the bootstrap script on a fresh Pi):
  pip install vosk
  # plus a model from https://alphacephei.com/vosk/models
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .base import STTEngine, STTResult


log = logging.getLogger("audio_node.vosk")


DEFAULT_MODEL_PATH = os.environ.get("AUDIO_VOSK_MODEL") or str(Path.home() / ".vosk-model")


class VoskEngine(STTEngine):
    name = "vosk"

    def __init__(self, model_path: str | None = None, sample_rate: int | None = None) -> None:
        self.sample_rate = int(sample_rate or os.environ.get("AUDIO_STT_RATE", "16000"))
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._recognizer = None
        self._model = None
        self._init_error: Optional[str] = None
        self._configure()

    def _configure(self) -> None:
        try:
            import vosk  # type: ignore
        except Exception as exc:
            self._init_error = f"vosk unavailable: {exc}"
            return
        if not self.model_path.exists():
            self._init_error = f"vosk model not found at {self.model_path}"
            return
        try:
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(self.model_path))
            self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
            self.available = True
        except Exception as exc:
            self._init_error = f"vosk model load failed: {exc}"
            self._model = None
            self._recognizer = None

    def accept(self, pcm: bytes) -> Optional[STTResult]:
        if self._recognizer is None or not pcm:
            return None
        try:
            done = self._recognizer.AcceptWaveform(pcm)
        except Exception as exc:
            log.warning("vosk AcceptWaveform failed: %s", exc)
            return None
        if not done:
            return None
        try:
            result = json.loads(self._recognizer.Result() or "{}")
        except json.JSONDecodeError:
            return None
        text = (result.get("text") or "").strip()
        if not text:
            return None
        return STTResult(text=text, final=True)

    def reset(self) -> None:
        if self._model is not None:
            try:
                import vosk  # type: ignore
                self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
            except Exception:
                pass

    def close(self) -> None:
        self._recognizer = None
        self._model = None
