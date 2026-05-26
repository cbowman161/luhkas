"""espeak-ng TTS via subprocess.

Always-on default. Robotic voice, but ships everywhere and never blocks on
network or model downloads. Override with ``AUDIO_TTS_ENGINE=piper`` once
Piper voices are installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from .base import TTSEngine


class EspeakEngine(TTSEngine):
    name = "espeak"

    def __init__(
        self,
        voice: str | None = None,
        rate_wpm: int | None = None,
        device: str | None = None,
        binary: str | None = None,
    ) -> None:
        self.voice = voice or os.environ.get("AUDIO_TTS_VOICE", "en-us")
        try:
            self.rate_wpm = int(rate_wpm if rate_wpm is not None else os.environ.get("AUDIO_TTS_RATE", "175"))
        except ValueError:
            self.rate_wpm = 175
        # ``--stdout`` lets us pipe into aplay so we honor the same ALSA
        # device as everything else, instead of espeak's default pulse path.
        self.binary = binary or os.environ.get("AUDIO_TTS_BIN", "espeak-ng")
        self.device = device or os.environ.get("AUDIO_OUTPUT_DEVICE", "default")
        self.aplay = os.environ.get("AUDIO_APLAY_BIN", "aplay")
        self.available = bool(shutil.which(self.binary)) and bool(shutil.which(self.aplay))

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self.available:
            return
        synth = subprocess.Popen(
            [self.binary, "--stdout", "-v", self.voice, "-s", str(self.rate_wpm), text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            subprocess.run(
                [self.aplay, "-q", "-D", self.device],
                stdin=synth.stdout,
                check=False,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if synth.stdout is not None:
                synth.stdout.close()
            synth.wait(timeout=5)
