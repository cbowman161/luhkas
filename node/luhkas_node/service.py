#!/usr/bin/env python3
"""LUHKAS node web/chat service.

Owns the local browser UI and text chat endpoint for a node. Module services
such as vision own their data/control APIs; this service proxies those APIs so
the UI can stay same-origin while `/ui` and `/chat` belong to `luhkas_node`.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_node.chat_log import ChatLog
from luhkas_node.chat_context import build_presence_payload
from luhkas_node.local_commands import handle as _local_command_handle
from luhkas_node.ui import ui_html
from luhkas_node.wakeword import is_wakeword_only, response as wakeword_response
from presence_state import update_state


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] luhkas_node: %(message)s")
log = logging.getLogger("luhkas_node.service")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_ID = os.environ.get("LUHKAS_NODE_ID", "kiosk")
HOST = os.environ.get("LUHKAS_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("LUHKAS_WEB_PORT", os.environ.get("DISPLAY_PORT", "5005")))
VISION_URL = os.environ.get("VISION_SERVICE_URL", f"http://127.0.0.1:{os.environ.get('SCOUT_VISION_PORT', '5000')}").rstrip("/")
PRESENCE_URL = os.environ.get(
    "AUDIO_PRESENCE_URL",
    f"http://127.0.0.1:{os.environ.get('PRESENCE_PORT', '5002')}/presence/message",
)
AUDIO_URL = os.environ.get("AUDIO_SERVICE_URL", "http://127.0.0.1:5004").rstrip("/")
DISPLAY_URL = os.environ.get("DISPLAY_SERVICE_URL", "http://127.0.0.1:5006").rstrip("/")
CHAT_LOG_MAX = int(os.environ.get("SCOUT_CHAT_LOG_MAX", "0"))
CHAT_LOG_PATH = Path(os.environ.get("SCOUT_CHAT_LOG_PATH", str(PROJECT_ROOT / "captures" / "chat_session.jsonl"))).expanduser()
if not CHAT_LOG_PATH.is_absolute():
    CHAT_LOG_PATH = PROJECT_ROOT / CHAT_LOG_PATH
chat_log = ChatLog(CHAT_LOG_PATH, CHAT_LOG_MAX)
_events = deque(maxlen=50)


def _load_modules() -> list[str]:
    try:
        from profile_loader import load_profile
        return list(load_profile(NODE_ID).get("modules", []))
    except Exception as exc:
        log.warning("could not load profile for %s: %s", NODE_ID, exc)
        return []


def _json_post(url: str, payload: dict, timeout: float = 30.0) -> dict | None:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("POST %s failed: %s", url, exc)
        return None


def _publish_display_event(event: dict) -> None:
    etype = event.get("type")
    text = str(event.get("text") or event.get("message") or "").strip()
    if etype == "user_message" and text:
        update_state({"latest_user": {"text": text, "source": event.get("source"), "timestamp": time.time()}})
    elif etype == "assistant_message" and text:
        update_state({"latest_assistant": {"text": text, "source": event.get("source"), "timestamp": time.time()}})
    if not DISPLAY_URL:
        return
    threading.Thread(
        target=_json_post,
        args=(DISPLAY_URL + "/ui/event", event),
        kwargs={"timeout": 3.0},
        daemon=True,
    ).start()


def _speak_response(response: dict, already_spoken: bool = False) -> None:
    if already_spoken or os.environ.get("LUHKAS_UI_TTS", "1") == "0":
        return
    if _audio_output_muted():
        return
    text = str(response.get("tts") or response.get("message") or "").strip()
    if text and AUDIO_URL:
        threading.Thread(
            target=_json_post,
            args=(AUDIO_URL + "/tts", {"text": text, "source": "luhkas_node", "silent": True}),
            kwargs={"timeout": 60.0},
            daemon=True,
        ).start()


def _audio_output_muted() -> bool:
    if not AUDIO_URL:
        return False
    try:
        with urlopen(AUDIO_URL + "/health", timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    tts = data.get("tts") if isinstance(data, dict) else {}
    return bool(isinstance(tts, dict) and tts.get("output_muted"))


class Handler(BaseHTTPRequestHandler):
    _VISION_GET = {
        "/capabilities",
        "/meta",
        "/chat_log",
        "/reference_poses",
        "/video_feed",
        "/snapshot",
    }
    _VISION_POST = {
        "/learn_face",
        "/collision",
        "/pantilt",
        "/move",
        "/settings",
        "/tracking",
        "/guard",
        "/clip",
        "/snapshot",
    }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/ui"):
            self._serve_ui()
        elif parsed.path == "/health":
            self._json({"ok": True, "service": "luhkas_node", "node_id": NODE_ID, "vision_url": VISION_URL})
        elif parsed.path == "/ui/state":
            self._json({"ok": True, "history": list(_events)})
        elif parsed.path in self._VISION_GET or parsed.path.startswith("/people/"):
            self._proxy("GET")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            self._chat()
        elif parsed.path == "/ui/event":
            self._ui_event()
        elif parsed.path in self._VISION_POST:
            self._proxy("POST")
        else:
            self.send_error(404)

    def _serve_ui(self) -> None:
        data = ui_html(node_label=NODE_ID, modules=_load_modules()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _chat(self) -> None:
        body = self._read_json()
        if body is None:
            return
        message = str(body.get("message", "")).strip()
        if not message:
            self.send_error(400, "message required")
            return
        chat_log.add("user", message, source="chat_input")
        _publish_display_event({"type": "user_message", "text": message, "source": "chat_input"})

        if is_wakeword_only(message):
            response = wakeword_response()
            chat_log.add("assistant", response["message"], source="wakeword")
            _publish_display_event({"type": "assistant_message", "text": response["message"], "source": "wakeword"})
            _speak_response(response)
            self._json({"ok": True, "response": response})
            return

        try:
            local_response = _local_command_handle(message)
        except Exception as exc:
            log.warning("local chat command failed: %s", exc)
            local_response = {"ok": False, "message": "I could not run that local command.", "error": str(exc)}
        if local_response is not None:
            chat_log.add(
                "assistant",
                local_response.get("tts") or local_response.get("message") or json.dumps(local_response),
                source="local_command",
            )
            _publish_display_event({
                "type": "assistant_message",
                "text": local_response.get("tts") or local_response.get("message") or json.dumps(local_response),
                "source": "local_command",
            })
            already_spoken = local_response.get("capability") == "speak"
            _speak_response(local_response, already_spoken=already_spoken)
            self._json({"ok": True, "response": local_response})
            return

        update_state({"conversation": {"thinking": True, "thinking_started_at": time.time()}})
        reply = _json_post(PRESENCE_URL, build_presence_payload(message, chat_log.snapshot(), NODE_ID), timeout=30.0)
        update_state({"conversation": {"thinking": False, "thinking_ended_at": time.time()}})
        if not reply:
            self._json({"ok": False, "error": "presence_unreachable"}, status=503)
            return
        response = reply.get("response") if isinstance(reply, dict) else None
        if response is None:
            response = reply
        chat_log.add("assistant", response.get("tts") or response.get("message") or json.dumps(response), source="presence_chat")
        _publish_display_event({
            "type": "assistant_message",
            "text": response.get("tts") or response.get("message") or json.dumps(response),
            "source": "presence_chat",
        })
        _speak_response(response)
        self._json({"ok": True, "response": response})

    def _ui_event(self) -> None:
        body = self._read_json()
        if body is None:
            return
        body.setdefault("timestamp", time.time())
        _events.append(body)
        etype = body.get("type")
        text = str(body.get("text") or body.get("message") or "").strip()
        if etype == "user_message" and text:
            chat_log.add("user", text, source=str(body.get("source") or "ui_event"))
        elif etype == "assistant_message" and text:
            chat_log.add("assistant", text, source=str(body.get("source") or "ui_event"))
        _publish_display_event(body)
        self._json({"ok": True})

    def _proxy(self, method: str) -> None:
        body = None
        headers = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
        target = VISION_URL + self.path
        try:
            req = Request(target, data=body, headers=headers, method=method)
            timeout = None if urlparse(self.path).path == "/video_feed" else 30
            with urlopen(req, timeout=timeout) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() not in {"connection", "transfer-encoding"}:
                        self.send_header(key, value)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except HTTPError as exc:
            self.send_error(exc.code, exc.reason)
        except (OSError, URLError, TimeoutError) as exc:
            self.send_error(502, str(exc))

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            return json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
        except Exception:
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
    chat_log.init_file()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("listening on http://%s:%s (vision=%s)", HOST, PORT, VISION_URL)
    server.serve_forever()


if __name__ == "__main__":
    main()
