"""Mic capture loop for audio_node.

Spawns ``arecord`` as a subprocess piping raw 16-bit little-endian mono
PCM, reads chunks, and feeds them into the STT engine. When the engine
emits a final STTResult, the registered callback is invoked with the
recognized text.

If the configured ``device`` is unavailable or arecord isn't installed,
the thread exits cleanly and the rest of the service keeps running.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

from .engines.base import STTEngine


log = logging.getLogger("audio_node.capture")


CaptureCallback = Callable[[str], None]


class MicCapture:
    def __init__(
        self,
        stt: STTEngine,
        on_transcript: CaptureCallback,
        device: str | None = None,
        sample_rate: int | None = None,
        chunk_bytes: int = 3200,  # 100 ms @ 16 kHz mono S16_LE
        arecord_bin: str | None = None,
        on_chunk = None,
    ) -> None:
        self.stt = stt
        self.on_chunk = on_chunk
        self.on_transcript = on_transcript
        self.device = device or os.environ.get("AUDIO_INPUT_DEVICE", "default")
        self.sample_rate = int(sample_rate or stt.sample_rate or 16000)
        self.chunk_bytes = chunk_bytes
        self.binary = arecord_bin or os.environ.get("AUDIO_ARECORD_BIN", "arecord")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._muted = threading.Event()
        self.last_error: Optional[str] = None
        self.last_transcript_at: float = 0.0
        self.last_transcript_text: str = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def start(self) -> bool:
        if not self.stt.available:
            self.last_error = "stt_engine_unavailable"
            log.warning("STT engine not available — capture disabled")
            return False
        if not shutil.which(self.binary):
            self.last_error = f"{self.binary}_not_installed"
            log.warning("%s not on PATH — capture disabled", self.binary)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-capture")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._capture_session()
            except Exception as exc:
                self.last_error = str(exc)
                log.error("capture session error: %s", exc)
                time.sleep(1.0)

    def _capture_session(self) -> None:
        cmd = [
            self.binary,
            "-q",
            "-D", self.device,
            "-c", "1",
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-t", "raw",
        ]
        log.info("starting capture: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.stdout is not None
            while not self._stop.is_set():
                pcm = proc.stdout.read(self.chunk_bytes)
                if not pcm:
                    break
                if self._muted.is_set():
                    continue
                if self.on_chunk is not None:
                    try:
                        self.on_chunk(pcm)
                    except Exception as exc:
                        log.warning('on_chunk callback failed: %s', exc)
                result = self.stt.accept(pcm)
                if result is None or not result.text:
                    continue
                self.last_transcript_at = time.time()
                self.last_transcript_text = result.text
                try:
                    self.on_transcript(result.text)
                except Exception as exc:
                    log.warning("transcript callback failed: %s", exc)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
