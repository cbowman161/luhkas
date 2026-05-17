"""
LUHKAS Node — thin client that connects any edge node to the brain.

Responsibilities:
  - Register this node with the brain on startup (node_id + display caps)
  - Forward user messages to brain /runtime/message and render the response
  - Speak TTS responses when the node has no display

Nodes ask the brain for things as needed. No push, no polling threads.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from .local_commands import capabilities as _local_capabilities
    from .local_commands import handle as _local_command_handle
    from .wakeword import is_wakeword_only as _is_wakeword_only
    from .wakeword import response as _wakeword_response
except Exception:
    try:
        from local_commands import capabilities as _local_capabilities
        from local_commands import handle as _local_command_handle
        from wakeword import is_wakeword_only as _is_wakeword_only
        from wakeword import response as _wakeword_response
    except Exception:
        _local_capabilities = None
        _local_command_handle = None
        _is_wakeword_only = None
        _wakeword_response = None

def _load_config(config_path: Path | None = None) -> dict:
    path = config_path or Path(__file__).parent / "config.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _merge_env(cfg: dict) -> dict:
    """Environment variables override config.json so containers/systemd can reconfigure."""
    if v := os.environ.get("LUHKAS_NODE_ID"):
        cfg["node_id"] = v
    if v := os.environ.get("LUHKAS_BRAIN_URL"):
        cfg["brain_url"] = v
    return cfg


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no external dependencies)
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _http_post(url: str, payload: dict, timeout: float = 30.0) -> dict | None:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"brain request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _speak(text: str, engine: str = "auto") -> None:
    if not text:
        return
    if engine == "auto":
        engine = "say" if os.uname().sysname == "Darwin" else "espeak"
    try:
        if engine == "say":
            subprocess.run(["say", text], check=False, timeout=30)
        elif engine in {"espeak", "espeak-ng"}:
            subprocess.run([engine, text], check=False, timeout=30)
        else:
            print(text)
    except FileNotFoundError:
        print(text)
    except Exception:
        print(text)


# ---------------------------------------------------------------------------
# Node runtime
# ---------------------------------------------------------------------------

class NodeRuntime:
    def __init__(self, config: dict | None = None, config_path: Path | None = None):
        cfg = _load_config(config_path)
        if config:
            cfg.update(config)
        _merge_env(cfg)

        self.node_id: str = cfg.get("node_id", "unknown-node")
        self.node_name: str = cfg.get("node_name", self.node_id)
        self.brain_url: str = cfg.get("brain_url", "http://localhost:8766").rstrip("/")
        self.display: dict = cfg.get("display", {"has_display": False})
        self.tts_cfg: dict = cfg.get("tts", {"enabled": True, "engine": "auto"})
        self.has_display: bool = bool(self.display.get("has_display", False))
        self.local_first: bool = bool(cfg.get("local_first", True))

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register with the brain so it knows our display capabilities."""
        self._register()

    def _register(self) -> None:
        url = self.brain_url + "/node/register"
        try:
            capabilities = self.local_capabilities()
            result = _http_post(url, {
                "node_id": self.node_id,
                "node_name": self.node_name,
                "display": self.display,
                "capabilities": capabilities,
                "modules": capabilities.get("module_status") or {},
            }, timeout=5.0)
            if result and result.get("ok"):
                print(f"[node] registered with brain as '{self.node_id}'", flush=True)
            else:
                print(f"[node] brain registration failed: {result}", flush=True)
        except Exception as exc:
            print(f"[node] could not reach brain for registration: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Send / render
    # ------------------------------------------------------------------

    def send(self, message: str) -> dict:
        """Send a message to the brain and return the response dict."""
        url = self.brain_url + "/runtime/message"
        result = _http_post(url, {"message": message, "node_id": self.node_id})
        if result is None:
            return {"message": "Brain unreachable.", "tts": "Brain unreachable."}
        return result.get("response") or result

    def render(self, response: dict) -> None:
        """Print and/or speak a brain response depending on display capabilities."""
        if self.has_display:
            print(response.get("message") or "")
            display_content = response.get("display_content", "")
            if display_content:
                print()
                print(display_content)
        else:
            text = response.get("tts") or response.get("message") or ""
            self._say(text)

    def handle(self, message: str) -> dict:
        """Forward message to brain and render the response."""
        if _is_wakeword_only is not None and _wakeword_response is not None and _is_wakeword_only(message):
            response = _wakeword_response()
            self.render(response)
            return response
        if self.local_first and _local_command_handle is not None:
            local = _local_command_handle(message)
            if local is not None:
                self.render(local)
                return local
        response = self.send(message)
        self.render(response)
        return response

    def local_capabilities(self) -> dict:
        if _local_capabilities is None:
            return {"ok": False, "error": "local_commands_unavailable", "commands": []}
        try:
            return _local_capabilities()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "commands": []}

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _say(self, text: str) -> None:
        if not text:
            return
        if not self.tts_cfg.get("enabled", True):
            print(text)
            return
        _speak(text, self.tts_cfg.get("engine", "auto"))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        brain_health = _http_get(self.brain_url + "/health", timeout=4.0)
        return {
            "ok": True,
            "node_id": self.node_id,
            "brain_url": self.brain_url,
            "brain_reachable": bool(brain_health and brain_health.get("ok")),
            "has_display": self.has_display,
        }
