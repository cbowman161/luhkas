"""Long-running asynchronous code_monkey service.

This module intentionally uses only the Python standard library so it can be
run anywhere the existing code_monkey package runs.

Key properties:
- HTTP API for LUHKAS-BRAIN or any caller to submit/poll tasks.
- Bounded worker pool; at most --workers tasks are actively building.
- Durable SQLite-backed queue using the existing tasks table.
- Startup recovery re-queues interrupted in-progress tasks.
- Graceful shutdown stops accepting work and lets current workers finish unless
  the process manager kills the service after its configured timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .build_manager import BuildManager
from .config import DATA_DIR
from .storage import Storage

DEFAULT_HOST = os.environ.get("CODE_MONKEY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("CODE_MONKEY_PORT", "8765"))
DEFAULT_WORKERS = int(os.environ.get("CODE_MONKEY_WORKERS", "2"))
DEFAULT_POLL_SECONDS = float(os.environ.get("CODE_MONKEY_POLL_SECONDS", "1.0"))

FINAL_STATES = {
    "verified",
    "build_failed",
    "test_failed",
    "failed",
    "cancelled",
}

IN_PROGRESS_STATES = {
    "planning",
    "building",
    "built",
    "testing",
    "repairing",
    "running",
    "claimed",
}

RUNNABLE_STATES = {
    "created",
    "planned",
    "queued",
}


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")


class CoderService:
    """Bounded asynchronous task runner for BuildManager jobs."""

    def __init__(self, workers: int = DEFAULT_WORKERS, poll_seconds: float = DEFAULT_POLL_SECONDS):
        self.workers = max(1, int(workers or 1))
        self.poll_seconds = max(0.1, float(poll_seconds or 1.0))
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        Storage().recover_interrupted_tasks()

    def start(self) -> None:
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"code-monkey-worker-{index + 1}",
                args=(index + 1,),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        for thread in self.threads:
            thread.join(timeout=timeout)

    def submit(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        manager = BuildManager(verbose=False)
        result = manager.submit(goal)
        task_id = result["task_id"]
        if context:
            manager.storage.blackboard_set(task_id, "caller", "context", context)
        manager.storage.enqueue_task(task_id, message="Queued for async build")
        result = manager.status(task_id)
        result["queued"] = True
        return result

    def enqueue_existing(self, task_id: str) -> Dict[str, Any]:
        storage = Storage()
        task = storage.get_task(task_id)
        if not task:
            return {"ok": False, "error": f"Unknown task_id: {task_id}"}
        if task.get("state") in FINAL_STATES:
            return {"ok": False, "task_id": task_id, "state": task.get("state"), "error": "Final tasks cannot be re-queued"}
        storage.enqueue_task(task_id, message="Queued for async build")
        return BuildManager(verbose=False).status(task_id)

    def health(self) -> Dict[str, Any]:
        storage = Storage()
        return {
            "ok": True,
            "service": "code_monkey",
            "workers": self.workers,
            "active_workers": storage.count_active_tasks(),
            "queued_tasks": storage.count_queued_tasks(),
            "unread_notifications": storage.count_unread_notifications(),
            "data_dir": str(DATA_DIR),
        }

    def _worker_loop(self, worker_number: int) -> None:
        while not self.stop_event.is_set():
            storage = Storage()
            task = storage.claim_next_task(worker_id=f"worker-{worker_number}")
            if not task:
                self.stop_event.wait(self.poll_seconds)
                continue
            task_id = task["task_id"]
            self._run_task(task_id, worker_number)

    def _run_task(self, task_id: str, worker_number: int) -> None:
        manager = BuildManager(verbose=False)
        manager.storage.blackboard_set(task_id, "service", "worker", worker_number)
        manager.storage.blackboard_set(task_id, "service", "started_at", time.time())
        try:
            manager.storage.add_event(task_id, "worker_started", "Async worker started task", {"worker": worker_number})
            result = manager.build(task_id)
            state = result.get("state")
            if state not in FINAL_STATES:
                manager.storage.add_event(task_id, "worker_completed", "Async worker finished task", {"state": state})
        except Exception as exc:
            manager.storage.update_task(task_id, "failed", "Unhandled worker failure")
            manager.storage.add_event(
                task_id,
                "worker_failed",
                "Unhandled worker failure",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
        finally:
            try:
                manager.storage.blackboard_set(task_id, "service", "finished_at", time.time())
            except Exception:
                pass


class CoderRequestHandler(BaseHTTPRequestHandler):
    server_version = "CodeMonkeyService/1.0"

    @property
    def service(self) -> CoderService:
        return self.server.service  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            if path in {"/", "/health"}:
                self._send(200, self.service.health())
                return
            if path == "/tasks":
                self._send(200, BuildManager(verbose=False).list_tasks())
                return
            if path in {"/board", "/notifications"}:
                mode = query.get("mode", [""])[0].lower().strip()
                recent_requested = mode == "recent" or query.get("recent", ["false"])[0].lower() in {"1", "true", "yes"}
                unread_only = query.get("unread", ["true"])[0].lower() not in {"0", "false", "no", "all"}
                if recent_requested:
                    unread_only = False
                mark_read = query.get("mark_read", ["true"])[0].lower() not in {"0", "false", "no"}
                if not unread_only:
                    mark_read = False
                limit = int(query.get("limit", ["50"])[0] or "50")
                storage = Storage()
                notifications = storage.list_notifications(
                    unread_only=unread_only,
                    limit=limit,
                    mark_read=mark_read,
                )
                unread_remaining = storage.count_unread_notifications()
                if notifications:
                    if unread_only:
                        message = f"New notifications: {len(notifications)} task notification(s)."
                    else:
                        message = f"Most recent notifications: {len(notifications)} task notification(s)."
                    status = "new_notifications" if unread_only else "recent_notifications"
                else:
                    status = "no_new_notifications" if unread_only else "no_notifications"
                    message = "No new notifications. Do you want the most recent?" if unread_only else "No notifications found."
                self._send(200, {
                    "ok": True,
                    "status": status,
                    "message": message,
                    "notifications": notifications,
                    "count": len(notifications),
                    "unread_remaining": unread_remaining,
                    "marked_read": bool(mark_read and unread_only and notifications),
                    "recent_url": "/board?mode=recent&limit=5",
                })
                return
            if path in {"/board/recent", "/notifications/recent"}:
                limit = int(query.get("limit", ["5"])[0] or "5")
                storage = Storage()
                notifications = storage.list_recent_notifications(limit=limit)
                self._send(200, {
                    "ok": True,
                    "status": "recent_notifications" if notifications else "no_notifications",
                    "message": f"Most recent notifications: {len(notifications)} task notification(s)." if notifications else "No notifications found.",
                    "notifications": notifications,
                    "count": len(notifications),
                    "unread_remaining": storage.count_unread_notifications(),
                    "marked_read": False,
                })
                return
            if path.startswith("/tasks/"):
                parts = path.strip("/").split("/")
                task_id = parts[1] if len(parts) >= 2 else ""
                if len(parts) == 2:
                    self._send(200, BuildManager(verbose=False).status(task_id))
                    return
                if len(parts) == 3 and parts[2] == "events":
                    self._send(200, BuildManager(verbose=False).events(task_id))
                    return
                if len(parts) == 3 and parts[2] == "session":
                    self._send(200, BuildManager(verbose=False).session(task_id))
                    return
                if len(parts) == 3 and parts[2] == "diagnostics":
                    failed = query.get("failed", ["false"])[0].lower() in {"1", "true", "yes"}
                    self._send(200, BuildManager(verbose=False).diagnostic_report(task_id, failed=failed))
                    return
            if path.startswith("/notifications/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[2] == "read":
                    ok = Storage().mark_notification_read(int(parts[1]))
                    self._send(200, {"ok": ok, "notification_id": int(parts[1])})
                    return
            self._send(404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            payload = self._read_json()
            if path == "/tasks":
                goal = str(payload.get("goal") or "").strip()
                if not goal:
                    self._send(400, {"ok": False, "error": "Missing required JSON field: goal"})
                    return
                self._send(202, self.service.submit(goal, context=payload.get("context")))
                return
            if path.startswith("/tasks/"):
                parts = path.strip("/").split("/")
                task_id = parts[1] if len(parts) >= 2 else ""
                if len(parts) == 3 and parts[2] == "enqueue":
                    self._send(202, self.service.enqueue_existing(task_id))
                    return
                if len(parts) == 3 and parts[2] == "cancel":
                    storage = Storage()
                    task = storage.get_task(task_id)
                    if not task:
                        self._send(404, {"ok": False, "error": f"Unknown task_id: {task_id}"})
                        return
                    storage.update_task(task_id, "cancelled", "Task cancelled")
                    self._send(200, {"ok": True, "task_id": task_id, "state": "cancelled"})
                    return
            if path.startswith("/notifications/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[2] == "read":
                    ok = Storage().mark_notification_read(int(parts[1]))
                    self._send(200, {"ok": ok, "notification_id": int(parts[1])})
                    return
            self._send(404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep service logs concise and systemd-friendly.
        print("[code_monkey_service] " + fmt % args, flush=True)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CoderHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], handler_class: type[BaseHTTPRequestHandler], service: CoderService):
        super().__init__(server_address, handler_class)
        self.service = service


def run_service(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, workers: int = DEFAULT_WORKERS, poll_seconds: float = DEFAULT_POLL_SECONDS) -> None:
    service = CoderService(workers=workers, poll_seconds=poll_seconds)
    service.start()
    server = CoderHTTPServer((host, int(port)), CoderRequestHandler, service)

    def _shutdown(signum: int, frame: Any) -> None:
        print(f"[code_monkey_service] signal {signum}; shutting down", flush=True)
        service.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[code_monkey_service] listening on http://{host}:{port} with workers={workers}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        service.stop()
        server.server_close()
        service.join(timeout=2.0)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m code_monkey.service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args(argv)
    run_service(host=args.host, port=args.port, workers=args.workers, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
