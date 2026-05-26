#!/usr/bin/env python3
"""audio_node HTTP service.

Owns the mic-to-speaker loop on a node:

  mic (arecord) → VAD/streaming STT → POST /presence/message → TTS → aplay

Endpoints:
  GET  /health      — engine + capture status
  POST /tts         — body {"text": "..."}; synthesize and play locally
  POST /listen      — body {"muted": bool}; pause/resume mic capture
  GET  /transcripts — last N recognized utterances (debug)

Configuration is fully env-driven so the same systemd unit works on any
node that has the RaspAudio HAT or a USB mic+speaker.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_node.capture import MicCapture
from audio_node.engines import load_stt, load_tts


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] audio_node: %(message)s")
log = logging.getLogger("audio_node")


_transcripts: deque = deque(maxlen=20)
_transcripts_lock = threading.Lock()
_tts_lock = threading.Lock()


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict | None:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        log.warning("POST %s failed: %s", url, exc)
        return None


def _notify_display(display_url: str, payload: dict) -> None:
    if not display_url:
        return
    try:
        req = Request(
            display_url.rstrip("/") + "/ui/event",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2.0):
            pass
    except Exception:
        # display is best-effort; never block the audio loop on it
        pass


def _make_transcript_handler(presence_url: str, source: str, node_id: str, tts, display_url: str) -> "callable":
    def _on_transcript(text: str) -> None:
        with _transcripts_lock:
            _transcripts.append({"text": text, "timestamp": time.time()})
        log.info("transcript: %s", text)
        _notify_display(display_url, {"type": "user_message", "text": text, "source": source})
        reply = _post_json(
            presence_url,
            {"message": text, "source": source, "node_id": node_id},
        )
        if not reply:
            return
        response = reply.get("response") or reply
        spoken = response.get("tts") or response.get("message") or ""
        if spoken:
            _notify_display(display_url, {"type": "assistant_message", "text": spoken})
        if not spoken:
            return
        with _tts_lock:
            try:
                tts.speak(spoken)
            except Exception as exc:
                log.warning("tts speak failed: %s", exc)
    return _on_transcript


class Handler(BaseHTTPRequestHandler):
    capture: MicCapture | None = None
    tts = None
    stt = None

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._json(self._health_payload())
        elif path == "/transcripts":
            with _transcripts_lock:
                items = list(_transcripts)
            self._json({"ok": True, "transcripts": items})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_json()
        if body is None:
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/tts":
            self._handle_tts(body)
        elif path == "/listen":
            self._handle_listen(body)
        else:
            self.send_error(404)

    def _handle_tts(self, body: dict) -> None:
        text = str(body.get("text") or "").strip()
        if not text:
            self.send_error(400, "missing text")
            return
        if self.tts is None or not self.tts.available:
            self._json({"ok": False, "error": "tts_unavailable", "engine": getattr(self.tts, "name", None)}, status=503)
            return
        with _tts_lock:
            try:
                self.tts.speak(text)
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
                return
        self._json({"ok": True, "engine": self.tts.name})

    def _handle_listen(self, body: dict) -> None:
        muted = bool(body.get("muted"))
        if self.capture is None:
            self._json({"ok": False, "error": "capture_unavailable"}, status=503)
            return
        if muted:
            self.capture.mute()
        else:
            self.capture.unmute()
        self._json({"ok": True, "muted": self.capture.muted})

    def _health_payload(self) -> dict:
        stt_name = getattr(self.stt, "name", "none")
        tts_name = getattr(self.tts, "name", "none")
        capture_running = bool(self.capture and self.capture.running)
        return {
            "ok": True,
            "stt": {
                "engine": stt_name,
                "available": bool(getattr(self.stt, "available", False)),
                "init_error": getattr(self.stt, "_init_error", None),
            },
            "tts": {
                "engine": tts_name,
                "available": bool(getattr(self.tts, "available", False)),
            },
            "capture": {
                "running": capture_running,
                "muted": bool(self.capture and self.capture.muted),
                "last_error": getattr(self.capture, "last_error", None),
                "last_transcript_at": getattr(self.capture, "last_transcript_at", 0.0),
                "last_transcript_text": getattr(self.capture, "last_transcript_text", ""),
            },
        }

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return None

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)


def main() -> None:
    host = os.environ.get("AUDIO_HOST", "0.0.0.0")
    port = int(os.environ.get("AUDIO_PORT", "5004"))
    stt_name = os.environ.get("AUDIO_STT_ENGINE", "vosk")
    tts_name = os.environ.get("AUDIO_TTS_ENGINE", "espeak")
    presence_url = os.environ.get(
        "AUDIO_PRESENCE_URL",
        f"http://127.0.0.1:{os.environ.get('PRESENCE_PORT', '5002')}/presence/message",
    )
    display_url = os.environ.get("AUDIO_DISPLAY_URL", "")
    source = os.environ.get("AUDIO_SOURCE", "audio_node")
    node_id = os.environ.get("LUHKAS_NODE_ID", "kiosk")

    stt = load_stt(stt_name)
    tts = load_tts(tts_name)
    log.info("stt=%s available=%s; tts=%s available=%s", stt.name, stt.available, tts.name, tts.available)
    if not stt.available:
        log.warning("STT unavailable (%s) — running output-only", getattr(stt, "_init_error", "?"))

    capture = MicCapture(
        stt=stt,
        on_transcript=_make_transcript_handler(presence_url, source, node_id, tts, display_url),
    )
    capture.start()

    Handler.capture = capture
    Handler.tts = tts
    Handler.stt = stt

    log.info("listening on http://%s:%s (presence=%s)", host, port, presence_url)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
