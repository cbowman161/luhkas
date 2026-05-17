import json
import os
import urllib.error
import urllib.parse
import urllib.request

from config import CODE_MONKEY_URL


DEFAULT_BASE_URL = CODE_MONKEY_URL


class CodeMonkeyClient:
    """Read/write boundary for the standalone Code Monkey HTTP service.

    The brain runtime must not import Code Monkey internals or write to Code
    Monkey storage directly. All interaction goes through the service API.
    """

    def __init__(self, base_url=None, timeout=10):
        self.base_url = (base_url or os.environ.get("CODE_MONKEY_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def start_task(self, goal):
        response = self._request(
            "POST",
            "/tasks",
            {"goal": goal},
        )
        task_id = response.get("task_id")
        if not task_id:
            raise RuntimeError("Code Monkey did not return a task_id.")
        return task_id

    def continue_task(self, active_task_id, followup):
        previous = self.safe_status(active_task_id)
        goal = (
            "Continue or revise the previous Code Monkey task.\n\n"
            f"Previous task id: {active_task_id}\n"
            f"Previous status summary:\n{json.dumps(previous, indent=2, default=str)[:4000]}\n\n"
            f"Follow-up request:\n{followup}"
        )
        return self.start_task(goal)

    def is_known_task(self, task_id):
        if not task_id:
            return False
        status = self.safe_status(task_id)
        return not status.get("error")

    def safe_status(self, task_id):
        try:
            return self._request("GET", f"/tasks/{urllib.parse.quote(str(task_id))}")
        except Exception as exc:
            return {
                "task_id": task_id,
                "error": str(exc),
            }

    def unread_updates(self):
        response = self._request(
            "GET",
            "/board?unread=true&mark_read=false&limit=50",
        )
        return response.get("notifications") or []

    def recent_completed_tasks(self, limit=10):
        """Return recently completed tasks regardless of read status, for review."""
        final_states = {"verified", "build_failed", "test_failed", "failed"}
        response = self._request("GET", f"/board/recent?limit={limit * 2}")
        notifications = response.get("notifications") or []
        seen = set()
        tasks = []
        for n in notifications:
            task_id = n.get("task_id")
            state = n.get("state")
            if task_id and state in final_states and task_id not in seen:
                seen.add(task_id)
                tasks.append({
                    "task_id": task_id,
                    "state": state,
                    "goal": n.get("goal", ""),
                    "message": n.get("message", ""),
                })
        return tasks[:limit]

    def list_jobs(self):
        response = self._request("GET", "/tasks")
        return response.get("tasks") or []

    def get_artifacts(self, task_id):
        """Return goal, readme, api_code, workspace, and state from a completed task."""
        status = self.safe_status(task_id)
        build_files = status.get("build_files") or {}
        files = {item["path"]: item.get("content", "") for item in build_files.get("files") or []}
        work_order = status.get("work_order") or {}
        session = status.get("session") or {}
        goal = (
            work_order.get("goal")
            or (session.get("planner") or {}).get("input_goal")
            or ""
        )
        return {
            "task_id": task_id,
            "state": status.get("state", "unknown"),
            "goal": goal,
            "capability_name": work_order.get("capability_name", ""),
            "readme": files.get("artifacts/README.md", ""),
            "api_code": files.get("src/api.py", ""),
            "test_code": files.get("tests/test_api.py", ""),
            "commands_json": files.get("commands.json", ""),
            "workspace": status.get("workspace", ""),
        }

    def health(self):
        return self._request("GET", "/health")

    def _request(self, method, path, payload=None):
        url = self.base_url + path
        data = None
        headers = {}

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Code Monkey service is unavailable at "
                f"{self.base_url}. Start it with: "
                "python3 -m code_monkey service --host 127.0.0.1 --port 8765 --workers 2. "
                f"Original error: {exc}"
            )

        if not raw.strip():
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Code Monkey returned invalid JSON: {exc}")

        if isinstance(parsed, dict) and parsed.get("ok") is False:
            raise RuntimeError(parsed.get("error") or "Code Monkey request failed.")

        return parsed
