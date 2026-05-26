#!/usr/bin/env python3
"""display_node HTTP service.

Owns the on-screen UI for nodes that have a display. The kiosk-browser
service launches chromium in --kiosk mode pointing at /ui on this port.

Endpoints:
  GET  /              — redirects to /ui
  GET  /ui            — index HTML
  GET  /ui/assets/*   — bundled static assets (css/js)
  GET  /ui/state      — JSON of current display state (polled by the SPA)
  POST /ui/event      — other services push events here (user_message,
                        assistant_message, status, alert)
  POST /ui/mute       — convenience pass-through to audio_node
  GET  /health        — service status
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] display_node: %(message)s")
log = logging.getLogger("display_node")


_UI_DIR = Path(__file__).resolve().parent / "ui"

_state_lock = threading.Lock()
_history: deque = deque(maxlen=50)
_status: dict = {
    "node_id": os.environ.get("LUHKAS_NODE_ID", "kiosk"),
    "started_at": time.time(),
    "last_event_at": 0.0,
    "muted": False,
}


def _record_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("timestamp", time.time())
    with _state_lock:
        _history.append(event)
        _status["last_event_at"] = event["timestamp"]
        etype = event.get("type")
        if etype == "status":
            for key in ("battery", "audio", "camera"):
                if key in event:
                    _status[key] = event[key]
            if "muted" in event:
                _status["muted"] = bool(event["muted"])


def _state_snapshot() -> dict:
    with _state_lock:
        history = list(_history)
        status = dict(_status)
    user_msgs = [e for e in history if e.get("type") == "user_message"]
    asst_msgs = [e for e in history if e.get("type") == "assistant_message"]
    return {
        "ok": True,
        "status": status,
        "last_user_message": user_msgs[-1] if user_msgs else None,
        "last_assistant_message": asst_msgs[-1] if asst_msgs else None,
        "history": history,
    }


def _proxy_post(url: str, payload: dict, timeout: float = 5.0) -> dict | None:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        log.warning("proxy POST to %s failed: %s", url, exc)
        return None


class Handler(BaseHTTPRequestHandler):
    audio_url: str = ""

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/" or path == "/ui":
            self._send_file(_UI_DIR / "index.html", "text/html; charset=utf-8")
        elif path.startswith("/ui/assets/"):
            rel = path[len("/ui/assets/"):]
            target = (_UI_DIR / "assets" / rel).resolve()
            if not str(target).startswith(str(_UI_DIR.resolve())) or not target.exists():
                self.send_error(404)
                return
            mime, _ = mimetypes.guess_type(str(target))
            self._send_file(target, mime or "application/octet-stream")
        elif path == "/ui/state":
            self._json(_state_snapshot())
        elif path == "/health":
            self._json({
                "ok": True,
                "service": "display_node",
                "ui_dir": str(_UI_DIR),
                "history_size": len(_history),
            })
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        body = self._read_json()
        if body is None:
            return
        if path == "/ui/event":
            _record_event(body)
            self._json({"ok": True})
        elif path == "/ui/mute":
            if not self.audio_url:
                self._json({"ok": False, "error": "audio_unconfigured"}, status=503)
                return
            result = _proxy_post(
                self.audio_url.rstrip("/") + "/listen",
                {"muted": bool(body.get("muted"))},
            )
            self._json(result or {"ok": False, "error": "audio_unreachable"}, status=200 if result else 502)
        else:
            self.send_error(404)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

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
    host = os.environ.get("DISPLAY_HOST", "0.0.0.0")
    port = int(os.environ.get("DISPLAY_PORT", "5005"))
    Handler.audio_url = os.environ.get("AUDIO_SERVICE_URL", "http://127.0.0.1:5004")

    log.info("listening on http://%s:%s (ui=%s)", host, port, _UI_DIR)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
