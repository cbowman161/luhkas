#!/usr/bin/env python3
"""battery_node HTTP service.

Polls a battery backend and exposes:
  GET /health   — service + backend status, last reading
  GET /battery  — just the latest reading

Configuration via env vars:
  BATTERY_BACKEND      uart_proxy | ina219                 (default: uart_proxy)
  BATTERY_HOST         bind host                            (default: 0.0.0.0)
  BATTERY_PORT         bind port                            (default: 5003)
  BATTERY_POLL_S       seconds between backend reads        (default: 1.0)
  BATTERY_STALE_S      reading considered fresh under this  (default: 5.0)

Backend-specific env vars are documented in each backend module.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battery_node.backends import BatteryReading, load_backend


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] battery_node: %(message)s")
log = logging.getLogger("battery_node")


_state_lock = threading.Lock()
_latest: Optional[BatteryReading] = None
_last_error: Optional[str] = None


def _poller(backend, interval: float) -> None:
    global _latest, _last_error
    while True:
        try:
            reading = backend.read()
        except Exception as exc:
            with _state_lock:
                _last_error = str(exc)
            log.warning("backend read failed: %s", exc)
            time.sleep(interval)
            continue
        if reading is not None:
            with _state_lock:
                _latest = reading
                _last_error = None
        time.sleep(interval)


def _snapshot(stale_s: float) -> dict:
    with _state_lock:
        reading = _latest
        last_error = _last_error
    if reading is None:
        return {"ok": False, "stale": True, "error": last_error or "no reading yet"}
    fresh = (time.time() - (reading.timestamp or 0)) <= stale_s
    payload = reading.as_dict()
    payload["ok"] = fresh
    payload["stale"] = not fresh
    if last_error:
        payload["last_error"] = last_error
    return payload


class Handler(BaseHTTPRequestHandler):
    stale_s: float = 5.0
    backend_name: str = "uart_proxy"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            snap = _snapshot(self.stale_s)
            self._json({
                "ok": bool(snap.get("ok")),
                "backend": self.backend_name,
                "battery": snap,
            })
        elif path == "/battery":
            self._json(_snapshot(self.stale_s))
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:  # quieter access log
        log.debug(fmt, *args)

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    backend_name = os.environ.get("BATTERY_BACKEND", "auto")
    host = os.environ.get("BATTERY_HOST", "0.0.0.0")
    port = int(os.environ.get("BATTERY_PORT", "5003"))
    interval = float(os.environ.get("BATTERY_POLL_S", "1.0"))
    stale_s = float(os.environ.get("BATTERY_STALE_S", "5.0"))

    backend = load_backend(backend_name)
    Handler.stale_s = stale_s
    Handler.backend_name = backend.name

    threading.Thread(target=_poller, args=(backend, interval), daemon=True).start()

    log.info("listening on http://%s:%s (backend=%s)", host, port, backend.name)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
