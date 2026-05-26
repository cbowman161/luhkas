"""Piper neural TTS — opt-in upgrade over espeak.

Enabled by setting ``AUDIO_TTS_ENGINE=piper`` and pointing
``AUDIO_PIPER_MODEL`` at a downloaded .onnx voice file.

Install:
  pip install piper-tts
  # plus a voice model from https://github.com/rhasspy/piper#voices
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .base import TTSEngine


class PiperEngine(TTSEngine):
    name = "piper"

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = Path(model_path or os.environ.get("AUDIO_PIPER_MODEL", ""))
        self.binary = os.environ.get("AUDIO_PIPER_BIN", "piper")
        self.device = device or os.environ.get("AUDIO_OUTPUT_DEVICE", "default")
        self.aplay = os.environ.get("AUDIO_APLAY_BIN", "aplay")
        self.available = (
            bool(shutil.which(self.binary))
            and bool(shutil.which(self.aplay))
            and self.model_path.exists()
        )

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self.available:
            return
        synth = subprocess.Popen(
            [self.binary, "--model", str(self.model_path), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            synth.stdin.write(text.encode("utf-8"))
            synth.stdin.close()
            subprocess.run(
                [self.aplay, "-q", "-r", "22050", "-f", "S16_LE", "-D", self.device],
                stdin=synth.stdout,
                check=False,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if synth.stdout is not None:
                synth.stdout.close()
            synth.wait(timeout=10)
