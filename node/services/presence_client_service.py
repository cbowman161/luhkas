#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luhkas_node.wakeword import is_wakeword_only as _is_wakeword_only
from luhkas_node.wakeword import response as _wakeword_response
from profile_loader import load_profile as _load_profile_resolved
try:
    from luhkas_node.local_commands import capabilities as _local_capabilities
    from luhkas_node.local_commands import selftest as _local_selftest
except Exception:
    _local_capabilities = None
    _local_selftest = None

_SELFTEST_INTERVAL = float(os.environ.get("SCOUT_SELFTEST_INTERVAL", "60"))


def _default_local_services(node_id: str) -> dict[str, str]:
    """Local services to poll in selftest, derived from the node profile.

    Accepts both shorthand (``"name": port``) and verbose
    (``"name": {"port": ...}``) service entries.
    """
    try:
        profile = _load_profile_resolved(node_id)
    except Exception:
        profile = {}
    services = profile.get("services") or {}
    urls: dict[str, str] = {}
    for name, value in services.items():
        if name == "presence":
            continue
        port = value.get("port") if isinstance(value, dict) else value
        if isinstance(port, int):
            urls[name] = f"http://127.0.0.1:{port}/health"
    if urls:
        return urls
    return {
        "vision": "http://127.0.0.1:5000/health",
        "robot_api": "http://127.0.0.1:5001/health",
    }


_LOCAL_SERVICES = _default_local_services(os.environ.get("LUHKAS_NODE_ID", "scout"))

_latest_selftest: dict = {}
_selftest_lock = threading.Lock()


DEFAULT_BRAIN_URL = os.environ.get("VAULT_CHAT_URL", "http://100.70.245.116:7000")  # Tailscale IP; see sync_manager for why mDNS is avoided
DEFAULT_SOURCE = os.environ.get("VAULT_CHAT_SOURCE", "scout_presence")



def _get_lan_ip() -> str:
    """Return the best LAN IP address for this machine."""
    try:
        # Connect to a remote address (doesn't actually send data) to find the right interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def _get_tailscale_ip() -> str:
    """Return this node's Tailscale IPv4 address when the tunnel is active."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


def _ensure_vault_pubkey(brain_url: str) -> None:
    """Fetch vault's sync SSH public key and add it to authorized_keys if absent."""
    try:
        with urlopen(f"{brain_url}/admin/pubkey", timeout=5) as r:
            data = json.loads(r.read())
        pubkey = str(data.get("pubkey", "")).strip()
        if not pubkey:
            return
        auth_keys = Path.home() / ".ssh" / "authorized_keys"
        auth_keys.parent.mkdir(mode=0o700, exist_ok=True)
        existing = auth_keys.read_text() if auth_keys.exists() else ""
        if pubkey not in existing:
            with open(auth_keys, "a") as f:
                f.write(f"\n{pubkey}\n")
            auth_keys.chmod(0o600)
            print("[scout_presence] installed vault SSH public key", flush=True)
    except Exception as exc:
        print(f"[scout_presence] could not fetch vault pubkey: {exc}", flush=True)


def _load_profile(node_id: str) -> dict:
    try:
        return _load_profile_resolved(node_id)
    except Exception:
        return {}


def _register_with_brain(config: dict, retries: int = 6) -> None:
    """Register this node with the brain, including our IP and service ports."""
    brain_url = config["brain_url"]
    node_id = config.get("node_id", "scout")
    _ensure_vault_pubkey(brain_url)
    lan_ip = _get_lan_ip()
    tailscale_ip = _get_tailscale_ip()
    prefer_tailscale = os.environ.get("LUHKAS_PREFER_TAILSCALE", "1") != "0"
    ip = tailscale_ip if prefer_tailscale and tailscale_ip else lan_ip
    profile = _load_profile(node_id)
    capabilities = _node_capabilities()
    payload = {
        "node_id": node_id,
        "node_name": f"{node_id} ({ip})",
        "ip": ip,
        "network": {
            "lan_ip": lan_ip,
            "tailscale_ip": tailscale_ip,
            "preferred": "tailscale" if ip == tailscale_ip and tailscale_ip else "lan",
        },
        "display": profile.get("display") or {"has_display": False},
        "services": profile.get("services") or {"presence": 5002},
        "capabilities": capabilities,
        "modules": capabilities.get("module_status") or {},
    }
    url = brain_url + "/node/register"
    for attempt in range(retries):
        try:
            req = Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    print(f"[scout_presence] registered with brain as {node_id} (ip={ip})", flush=True)
                    return
        except Exception as exc:
            print(f"[scout_presence] registration attempt {attempt + 1}/{retries} failed: {exc}", flush=True)
        time.sleep(5)
    print("[scout_presence] WARNING: could not register with brain after all retries", flush=True)


def _node_capabilities() -> dict:
    if _local_capabilities is None:
        return {"ok": False, "error": "local_commands_unavailable", "commands": [], "module_status": {}}
    try:
        return _local_capabilities()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "commands": [], "module_status": {}}


def _run_selftest() -> dict:
    """Check all local services and *_node modules. Returns structured report."""
    services: dict[str, dict] = {}
    for name, url in _LOCAL_SERVICES.items():
        try:
            with urlopen(url, timeout=3) as r:
                data = json.loads(r.read().decode())
            services[name] = {"ok": bool(data.get("ok", True)), "response": data}
        except Exception as exc:
            services[name] = {"ok": False, "error": str(exc)}

    modules: dict = {}
    if _local_selftest is not None:
        try:
            result = _local_selftest()
            modules = result.get("modules", {})
        except Exception as exc:
            modules = {"error": str(exc)}

    services_ok = all(v.get("ok", False) for v in services.values())
    modules_ok = all(
        v.get("ok", False) for v in modules.values()
        if isinstance(v, dict)
    ) if modules else True
    return {
        "ok": services_ok and modules_ok,
        "timestamp": time.time(),
        "services": services,
        "modules": modules,
    }


def _selftest_loop(config: dict) -> None:
    """Background thread: run self-test periodically and report to vault."""
    brain_url = config["brain_url"]
    node_id = config.get("node_id", "scout")
    time.sleep(15)
    while True:
        report = _run_selftest()
        with _selftest_lock:
            _latest_selftest.clear()
            _latest_selftest.update(report)
        try:
            payload = json.dumps({"node_id": node_id, **report}).encode()
            req = Request(
                brain_url + "/node/selftest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5):
                pass
        except Exception:
            pass
        time.sleep(_SELFTEST_INTERVAL)


def _poll_alerts(config: dict, server) -> None:
    """Background thread: poll brain for pending alerts and cache them locally."""
    brain_url = config["brain_url"]
    node_id = config.get("node_id", "scout")
    url = f"{brain_url}/alerts/pending?node_id={node_id}"
    while True:
        try:
            with urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                alerts = data.get("alerts") or []
                if alerts:
                    with server.alert_lock:
                        server.pending_alerts.extend(alerts)
                    print(f"[scout_presence] received {len(alerts)} alert(s)", flush=True)
        except Exception:
            pass
        time.sleep(3)


class PresenceProxyHandler(BaseHTTPRequestHandler):
    server_version = "RoverPresenceProxy/1.0"

    @property
    def config(self):
        return self.server.config

    def do_GET(self):
        if self.path.rstrip("/") == "/selftest":
            with _selftest_lock:
                report = dict(_latest_selftest) if _latest_selftest else {"ok": None, "note": "no test run yet"}
            self._send(200, {"ok": True, "node_id": self.config.get("node_id", "scout"), **report})
            return
        if self.path.rstrip("/") in {"", "/", "/health"}:
            brain = self._get_json(self.config["brain_url"] + "/health", timeout=4)
            self._send(200, {
                "ok": True,
                "service": "scout_presence_proxy",
                "brain_url": self.config["brain_url"],
                "source": self.config["source"],
                "brain_reachable": bool(brain and brain.get("ok")),
                "brain_health": brain,
                "uptime_seconds": time.time() - self.server.started_at,
            })
            return
        if self.path.rstrip("/") == "/session":
            self._send_upstream("GET", "/presence/session")
            return
        if self.path.rstrip("/") == "/alerts/pending":
            with self.server.alert_lock:
                alerts = list(self.server.pending_alerts)
                self.server.pending_alerts.clear()
            self._send(200, {"ok": True, "alerts": alerts})
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/presence/message":
            payload = self._read_json()
            message = str(payload.get("message") or "").strip()
            if not message:
                self._send(400, {"ok": False, "error": "missing_message"})
                return
            if _is_wakeword_only(message):
                self._send(200, {"ok": True, "response": _wakeword_response()})
                return
            payload["message"] = message
            payload.setdefault("source", self.config["source"])
            payload.setdefault("node_id", self.config.get("node_id") or self.config["source"])
            self._send_upstream("POST", "/presence/message", payload=payload)
            return
        if path == "/presence/message/stream":
            payload = self._read_json()
            message = str(payload.get("message") or "").strip()
            if not message:
                self._send(400, {"ok": False, "error": "missing_message"})
                return
            if _is_wakeword_only(message):
                # Wakeword-only: synthesize a single-event stream locally so
                # callers can use the same wire format regardless of route.
                self._emit_synthetic_stream(_wakeword_response())
                return
            payload["message"] = message
            payload.setdefault("source", self.config["source"])
            payload.setdefault("node_id", self.config.get("node_id") or self.config["source"])
            self._stream_upstream("/presence/message/stream", payload=payload)
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def log_message(self, fmt, *args):
        print("[scout_presence] " + fmt % args, flush=True)

    def _auth_headers(self) -> dict:
        """Authorization header for vault calls, if a shared secret is set.

        The secret comes from VAULT_PRESENCE_SECRET (same env var read by
        vault_service). When unset on either end, no header is sent and
        vault doesn't require one — backward compatible with deployments
        that haven't opted in to auth.
        """
        secret = str(self.config.get("presence_secret") or "").strip()
        if secret:
            return {"Authorization": f"Bearer {secret}"}
        return {}

    def _send_upstream(self, method: str, path: str, payload: dict | None = None):
        url = self.config["brain_url"] + path
        try:
            if method == "GET":
                result = self._get_json(url, timeout=8)
            else:
                result = self._post_json(url, payload or {}, timeout=60, extra_headers=self._auth_headers())
        except Exception as exc:
            self._send(502, {"ok": False, "error": str(exc), "brain_url": self.config["brain_url"]})
            return
        if result is None:
            self._send(502, {"ok": False, "error": "brain_unreachable", "brain_url": self.config["brain_url"]})
            return
        self._send(200, result)

    def _stream_upstream(self, path: str, payload: dict) -> None:
        """Proxy an NDJSON stream from the vault to our client, line-by-line."""
        url = self.config["brain_url"] + path
        headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}
        headers.update(self._auth_headers())
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            upstream = urlopen(request, timeout=120)
        except (OSError, URLError, TimeoutError) as exc:
            self._send(502, {"ok": False, "error": str(exc), "brain_url": self.config["brain_url"]})
            return
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for raw_line in upstream:
                if not raw_line:
                    continue
                try:
                    self.wfile.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    def _emit_synthetic_stream(self, response_payload: dict) -> None:
        """Emit a single-event NDJSON stream for locally-handled wakeword replies."""
        text = ""
        if isinstance(response_payload, dict):
            text = str(
                response_payload.get("tts")
                or response_payload.get("message")
                or response_payload.get("response")
                or ""
            )
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            # Wakeword reply is fully canned; emit start + done (no delta)
            # so the node speaks it as a single TTS dispatch via the
            # "deterministic" path (see audio_node/service.py).
            for event in (
                {"type": "start"},
                {"type": "done", "text": text},
            ):
                self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _read_json(self):
        length = int(self.headers.get("content-length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _get_json(url: str, timeout: float):
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _post_json(url: str, payload: dict, timeout: float, extra_headers: dict | None = None):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"vault presence request failed: {exc}") from exc


class PresenceProxyServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, config):
        super().__init__(server_address, handler_class)
        self.config = config
        self.started_at = time.time()
        self.pending_alerts: list = []
        self.alert_lock = __import__("threading").Lock()


def main():
    parser = argparse.ArgumentParser(description="Always-on scout edge proxy for the unified vault presence endpoint.")
    parser.add_argument("--host", default=os.environ.get("SCOUT_PRESENCE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SCOUT_PRESENCE_PORT", "5002")))
    parser.add_argument("--brain-url", default=DEFAULT_BRAIN_URL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--node-id", default=os.environ.get("LUHKAS_NODE_ID", "scout"))
    args = parser.parse_args()

    config = {
        "brain_url": args.brain_url.rstrip("/"),
        "source": args.source,
        "node_id": args.node_id,
        # Shared-secret auth (matches VAULT_PRESENCE_SECRET on the vault).
        # Optional: when unset on both ends, no auth header is sent and
        # vault doesn't require one — backward compatible.
        "presence_secret": os.environ.get("VAULT_PRESENCE_SECRET", "").strip(),
    }
    server = PresenceProxyServer((args.host, args.port), PresenceProxyHandler, config)
    print(f"[scout_presence] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[scout_presence] forwarding to {config['brain_url']} as {config['source']}", flush=True)

    import threading as _threading

    # Register with brain after a short delay (brain may still be booting)
    _threading.Thread(
        target=_register_with_brain, args=(config,), daemon=True
    ).start()

    # Poll brain for pending alerts every 3s
    _threading.Thread(
        target=_poll_alerts, args=(config, server), daemon=True
    ).start()

    # Run periodic node self-tests and report results to vault
    _threading.Thread(
        target=_selftest_loop, args=(config,), daemon=True
    ).start()

    server.serve_forever()


if __name__ == "__main__":
    main()
