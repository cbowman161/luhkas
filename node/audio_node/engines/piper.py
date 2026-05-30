"""Piper neural TTS — opt-in upgrade over espeak.

Enabled by setting ``AUDIO_TTS_ENGINE=piper`` and pointing
``AUDIO_PIPER_MODEL`` at a downloaded .onnx voice file (plus its .onnx.json
config alongside it).

Two backends are supported transparently:

* **Python API** (preferred): loads ``PiperVoice`` once at service start
  and streams raw PCM bytes to ``aplay`` as each sentence finishes
  synthesizing. This is what we want — no per-call model reload, which used
  to cost ~1–2 s on every reply.
* **Subprocess** (fallback): spawns the ``piper`` CLI per call. Kept so the
  engine still works if the Python package isn't importable.

Sample rate is read from the voice's .onnx.json so playback matches the
model (medium voices: 22050; high: 22050; low: 16000).

Install:
  pip install piper-tts
  # plus a voice model from https://github.com/rhasspy/piper#voices
"""
from __future__ import annotations

import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .base import TTSEngine


log = logging.getLogger("audio_node.piper")


class PiperEngine(TTSEngine):
    name = "piper"

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = Path(model_path or os.environ.get("AUDIO_PIPER_MODEL", ""))
        default_binary = Path(sys.executable).parent / "piper"
        self.binary = os.environ.get("AUDIO_PIPER_BIN") or str(default_binary)
        self.device = device or os.environ.get("AUDIO_OUTPUT_DEVICE", "default")
        self.aplay = os.environ.get("AUDIO_APLAY_BIN", "aplay")
        self._proc_lock = threading.Lock()
        self._synth: subprocess.Popen | None = None
        self._play: subprocess.Popen | None = None
        self._voice = None
        self._sample_rate = self._read_sample_rate(default=22050)
        self._init_error: str | None = None

        aplay_ok = bool(shutil.which(self.aplay))
        model_ok = self.model_path.exists()
        if aplay_ok and model_ok:
            self._load_python_voice()
        elif not aplay_ok:
            self._init_error = f"aplay not found at {self.aplay!r}"
        elif not model_ok:
            self._init_error = f"piper model not found at {self.model_path}"

        self.available = aplay_ok and model_ok and (
            self._voice is not None or bool(shutil.which(self.binary))
        )
        if not self.available:
            log.warning("piper unavailable: %s", self._init_error or "missing prereqs")
        elif self._voice is not None:
            log.info("piper: python API ready (%s, %d Hz)", self.model_path.name, self._sample_rate)
        else:
            log.info("piper: subprocess fallback (%s)", self.model_path.name)

    def _read_sample_rate(self, default: int) -> int:
        cfg = self.model_path.with_suffix(self.model_path.suffix + ".json")
        try:
            data = json.loads(cfg.read_text())
            return int(data.get("audio", {}).get("sample_rate") or default)
        except Exception:
            return default

    def _load_python_voice(self) -> None:
        try:
            from piper.voice import PiperVoice
        except Exception as exc:
            self._init_error = f"piper python API unavailable: {exc}"
            return
        try:
            self._voice = PiperVoice.load(str(self.model_path))
        except Exception as exc:
            self._init_error = f"PiperVoice.load failed: {exc}"
            self._voice = None

    def _iter_audio_chunks(self, text: str):
        """Yield raw 16-bit PCM bytes from the python API.

        Handles both piper-tts API shapes:
          * older: ``voice.synthesize_stream_raw(text)`` → Iterator[bytes]
          * newer: ``voice.synthesize(text)`` → Iterator[AudioChunk]
        """
        voice = self._voice
        raw = getattr(voice, "synthesize_stream_raw", None)
        if raw is not None:
            yield from raw(text)
            return
        synth = getattr(voice, "synthesize", None)
        if synth is not None:
            for chunk in synth(text):
                audio = (
                    getattr(chunk, "audio_int16_bytes", None)
                    or getattr(chunk, "audio", None)
                    or (bytes(chunk) if isinstance(chunk, (bytes, bytearray, memoryview)) else None)
                )
                if audio:
                    yield bytes(audio)
            return
        raise RuntimeError("PiperVoice exposes no known streaming synthesis method")

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not self.available:
            return
        if self._voice is not None:
            self._speak_python(text)
        else:
            self._speak_subprocess(text)

    def _speak_python(self, text: str) -> None:
        play = subprocess.Popen(
            [
                self.aplay, "-q",
                "-r", str(self._sample_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-D", self.device,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        with self._proc_lock:
            self._play = play
        try:
            for audio_bytes in self._iter_audio_chunks(text):
                if play.poll() is not None:
                    break
                try:
                    play.stdin.write(audio_bytes)
                except BrokenPipeError:
                    break
        except Exception as exc:
            log.warning("piper python synthesis failed (%s); falling back to subprocess", exc)
            try:
                if play.stdin is not None:
                    play.stdin.close()
            except Exception:
                pass
            try:
                play.wait(timeout=2)
            except subprocess.TimeoutExpired:
                play.kill()
            with self._proc_lock:
                if self._play is play:
                    self._play = None
            self._speak_subprocess(text)
            return
        finally:
            try:
                if play.stdin is not None:
                    play.stdin.close()
            except Exception:
                pass
            try:
                play.wait(timeout=30)
            except subprocess.TimeoutExpired:
                play.kill()
            with self._proc_lock:
                if self._play is play:
                    self._play = None

    def _speak_subprocess(self, text: str) -> None:
        synth = subprocess.Popen(
            [self.binary, "--model", str(self.model_path), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        play = None
        with self._proc_lock:
            self._synth = synth
        try:
            synth.stdin.write(text.encode("utf-8"))
            synth.stdin.close()
            play = subprocess.Popen(
                [
                    self.aplay, "-q",
                    "-r", str(self._sample_rate),
                    "-f", "S16_LE",
                    "-D", self.device,
                ],
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
                synth.wait(timeout=10)
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
