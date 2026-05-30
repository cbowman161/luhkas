"""espeak-ng TTS via subprocess.

Always-on default. Robotic voice, but ships everywhere and never blocks on
network or model downloads. Override with ``AUDIO_TTS_ENGINE=piper`` once
Piper voices are installed.
"""
from __future__ import annotations

import os
import signal
import shutil
import subprocess
import threading

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
        self._proc_lock = threading.Lock()
        self._synth: subprocess.Popen | None = None
        self._play: subprocess.Popen | None = None
        self.available = bool(shutil.which(self.binary)) and bool(shutil.which(self.aplay))

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self.available:
            return
        synth = subprocess.Popen(
            [self.binary, "--stdout", "-v", self.voice, "-s", str(self.rate_wpm), text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        play = None
        with self._proc_lock:
            self._synth = synth
        try:
            play = subprocess.Popen(
                [self.aplay, "-q", "-D", self.device],
                stdin=synth.stdout,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            with self._proc_lock:
                self._play = play
            play.wait()
        finally:
            if synth.stdout is not None:
                synth.stdout.close()
            try:
                synth.wait(timeout=5)
            except subprocess.TimeoutExpired:
                synth.kill()
            with self._proc_lock:
                if self._synth is synth:
                    self._synth = None
                if self._play is play:
                    self._play = None

    def interrupt(self) -> None:
        with self._proc_lock:
            procs = [self._play, self._synth]
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
