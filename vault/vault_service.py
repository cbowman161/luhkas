import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from vault_runtime import VaultRuntime


_PUSHABLE_EVENT_TYPES = {
    "learn_succeeded", "learn_failed", "learn_needs_install",
    "install_succeeded", "install_failed",
    "world_ingest_stalled", "world_ingest_completed",
}


def _events_feed(runtime, since_id: int) -> dict:
    """Read-only poll for UI clients. Returns the unread events with
    id > since_id, filtered to types meant for user attention. Clients
    track their own last_seen_id locally so each event is shown once
    per client, without marking anything read globally (so the chat
    path's notification_alert auto-attach still fires until the user
    explicitly reads with `any updates`)."""
    try:
        unread = runtime.event_log.unread() or []
    except Exception as exc:
        return {"ok": False, "error": str(exc), "events": []}
    events = [
        {
            "id": e.get("id"),
            "event_type": e.get("event_type"),
            "message": e.get("message"),
            "data": e.get("data"),
            "created_at": e.get("created_at"),
        }
        for e in unread
        if isinstance(e.get("id"), int)
        and e["id"] > since_id
        and e.get("event_type") in _PUSHABLE_EVENT_TYPES
    ]
    return {"ok": True, "events": events}


def _world_status(runtime) -> dict:
    """Fresh-instantiate the world store per call so counts reflect writes
    made by background ingest jobs (cached table snapshots can lag)."""
    try:
        from models import get_model
        from world import WorldKnowledgeStore
        try:
            embedder = get_model("embed")
        except Exception:
            embedder = None
        store = WorldKnowledgeStore(text_embedder=embedder)
        return {"ok": True, **store.stats()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _node_sync_profile(node_id: str) -> dict:
    try:
        path = __import__("pathlib").Path(__file__).resolve().parents[1] / "node" / "profiles" / f"{node_id}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("sync") if isinstance(data, dict) and isinstance(data.get("sync"), dict) else {}
    except Exception:
        pass
    return {}


def _provision_tailscale_after_register(node_id: str, host: str, sync: dict) -> None:
    try:
        from sync_manager import provision_tailscale_for_node
        result = provision_tailscale_for_node(
            node_id=node_id,
            host=host,
            user=str(sync.get("user") or "luhkas"),
            node_dir=str(sync.get("node_dir") or "luhkas/node"),
        )
        status = "ok" if result.get("ok") else f"failed: {result.get('error')}"
        print(f"[tailscale] provision {node_id}@{host}: {status}", flush=True)
    except Exception as exc:
        print(f"[tailscale] provision {node_id}@{host}: failed: {exc}", flush=True)


def _orchestrate_if_pre_install(payload: dict, sync: dict) -> None:
    """If the node is registering in pre-install phase, kick the full
    first-time orchestration on a background thread.

    Pre-install registrations are sent by ``scripts/luhkas_firstboot.sh``
    on the freshly-flashed SD card: the node only has cloud-init's user +
    SSH + WiFi set up. The orchestrator takes it from there.
    """
    if str(payload.get("bootstrap_phase") or "") != "pre-install":
        return
    node_id = str(payload.get("node_id") or "").strip()
    network = payload.get("network") if isinstance(payload.get("network"), dict) else {}
    host = str(payload.get("ip") or network.get("lan_ip") or "").strip()
    if not node_id or not host:
        return
    try:
        from node_orchestrator import orchestrate_async
        orchestrate_async(
            node_id,
            host,
            user=str(sync.get("user") or "luhkas"),
            node_dir=str(sync.get("node_dir") or "luhkas/node"),
        )
        print(f"[orchestrator] kicked off for {node_id}@{host}", flush=True)
    except Exception as exc:
        print(f"[orchestrator] kick-off failed for {node_id}@{host}: {exc}", flush=True)


class VaultRequestHandler(BaseHTTPRequestHandler):
    server_version = "VaultRuntimeService/1.0"

    @property
    def runtime(self):
        return self.server.runtime

    def do_GET(self):
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"

            if path in {"/", "/health"}:
                self._send(200, self.runtime.health())
                return

            if path == "/updates":
                self._send(200, self.runtime.handle("updates"))
                return

            if path == "/jobs":
                self._send(200, self.runtime.handle("jobs"))
                return

            if path == "/code-monkey":
                self._send(200, self.runtime.handle("code monkey"))
                return

            if path == "/capabilities":
                self._send(200, {"ok": True, "capabilities": self.runtime.scout.capabilities()})
                return

            if path == "/session":
                self._send(200, self.runtime.scout.session())
                return

            if path == "/presence/session":
                self._send(200, self.runtime.scout.session())
                return

            if path == "/scout/state":
                self._send(200, self.runtime.scout.scout_state())
                return

            if path == "/scout/tools":
                self._send(200, self.runtime.scout.scout_tool_status())
                return

            if path == "/whoami":
                self._send(200, self.runtime.scout.whoami())
                return

            if path == "/identity":
                self._send(200, self.runtime.scout.get_identity_profile())
                return

            if path == "/debug/identity":
                self._send(200, {"ok": True, "debug": self.runtime.scout.identity_debug()})
                return

            if path == "/faces/sync":
                self._send(200, self.runtime.scout.faces_sync())
                return

            if path == "/faces/unknown":
                self._send(200, self.runtime.scout.unknown_faces())
                return

            if path == "/node/status":
                self._send(200, self.runtime.node_registry.health_summary())
                return

            if path == "/admin/sync":
                from sync_manager import last_result
                self._send(200, last_result())
                return

            if path == "/world/status":
                self._send(200, _world_status(self.runtime))
                return

            if path == "/events/feed":
                from urllib.parse import parse_qs
                qs = parse_qs(urlparse(self.path).query)
                try:
                    since_id = int((qs.get("since_id") or ["0"])[0])
                except ValueError:
                    since_id = 0
                self._send(200, _events_feed(self.runtime, since_id))
                return

            if path == "/admin/pubkey":
                from sync_manager import pubkey
                pk = pubkey()
                if pk:
                    self._send(200, {"ok": True, "pubkey": pk})
                else:
                    self._send(404, {"ok": False, "error": "no sync key found"})
                return

            if path == "/alerts/pending":
                from urllib.parse import parse_qs
                qs = parse_qs(urlparse(self.path).query)
                node_id = (qs.get("node_id") or [""])[0].strip() or "scout"
                alerts = self.runtime.node_registry.pop_alerts(node_id)
                self._send(200, {"ok": True, "node_id": node_id, "alerts": alerts})
                return

            if path.startswith("/people/") and path.endswith("/summary"):
                identity = self._identity_from_path(path, "summary")
                self._send(200, self.runtime.scout.person_summary(identity))
                return

            if path.startswith("/people/") and path.endswith("/memory"):
                identity = self._identity_from_path(path, "memory")
                self._send(200, self.runtime.scout.person_memory(identity))
                return

            self._send(404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def do_POST(self):
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"

            if path == "/presence/message":
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                # node_id takes precedence; fall back to source label for older callers
                node_id = str(
                    payload.get("node_id")
                    or payload.get("source")
                    or payload.get("client")
                    or "scout"
                ).strip()

                if not message:
                    self._send(400, {"ok": False, "error": "Missing required JSON field: message"})
                    return

                response = self.runtime.handle_presence(message, node_id=node_id, presence_context=payload)
                active_id = (response or {}).get("active_identity")
                import threading as _t
                _t.Thread(
                    target=self._update_person_count,
                    args=(node_id,), daemon=True
                ).start()
                self._send(200, {"ok": True, "response": response})
                return

            if path == "/vision/analyze":
                payload = self._read_json()
                question = str(payload.get("question") or "What do you see?")
                self._send(200, self.runtime.scout.analyze_scene(question, self.runtime.scout.scout_state()))
                return

            if path == "/identity":
                payload = self._read_json()
                self._send(200, self.runtime.scout.update_identity_profile(payload))
                return

            if path.startswith("/people/") and path.endswith("/faces"):
                payload = self._read_json()
                identity = self._identity_from_path(path, "faces")
                self._send(200, self.runtime.scout.add_face_reference(identity, payload))
                return

            if path == "/faces/unknown":
                payload = self._read_json()
                self._send(200, self.runtime.scout.add_unknown_face_observation(payload))
                return

            if path == "/faces/unknown/promote":
                payload = self._read_json()
                self._send(200, self.runtime.scout.promote_unknown_face_group(
                    group_id=str(payload.get("group_id") or ""),
                    identity=str(payload.get("identity") or ""),
                ))
                return

            if path.startswith("/people/") and path.endswith("/remember"):
                payload = self._read_json()
                identity = self._identity_from_path(path, "remember")
                self._send(200, self.runtime.scout.remember(
                    identity=identity,
                    memory_type=str(payload.get("type") or "fact"),
                    key=str(payload.get("key") or ""),
                    value=payload.get("value"),
                    source=str(payload.get("source") or "user"),
                    confidence=float(payload.get("confidence") or 1.0),
                ))
                return

            if path.startswith("/people/") and path.endswith("/preference"):
                payload = self._read_json()
                identity = self._identity_from_path(path, "preference")
                self._send(200, self.runtime.scout.remember(
                    identity=identity,
                    memory_type="preference",
                    key=str(payload.get("key") or ""),
                    value=payload.get("value"),
                    source=str(payload.get("source") or "user"),
                    confidence=1.0,
                ))
                return

            if path == "/node/register":
                payload = self._read_json()
                node_id = str(payload.get("node_id") or "").strip()
                if not node_id:
                    self._send(400, {"ok": False, "error": "Missing node_id"})
                    return
                self.runtime.node_registry.register(
                    node_id=node_id,
                    display=payload.get("display") or {},
                    node_name=str(payload.get("node_name") or node_id),
                    ip=str(payload.get("ip") or ""),
                    network=payload.get("network") or {},
                    services=payload.get("services") or {},
                    capabilities=payload.get("capabilities") or {},
                    modules=payload.get("modules") or {},
                )
                import threading as _t
                _t.Thread(
                    target=__import__("sync_manager").auto_push_if_new,
                    args=(node_id,),
                    daemon=True,
                ).start()
                sync = _node_sync_profile(node_id)
                provision_host = str(payload.get("ip") or "")
                network = payload.get("network") if isinstance(payload.get("network"), dict) else {}
                if not provision_host:
                    provision_host = str(network.get("lan_ip") or network.get("tailscale_ip") or "")
                # Pre-install registrations from a fresh SD card need full
                # first-time orchestration. Existing nodes just need the
                # Tailscale auth-key topped up.
                if str(payload.get("bootstrap_phase") or "") == "pre-install":
                    _orchestrate_if_pre_install(payload, sync)
                elif provision_host:
                    _t.Thread(
                        target=_provision_tailscale_after_register,
                        args=(node_id, provision_host, sync),
                        daemon=True,
                    ).start()
                self._send(200, {"ok": True, "node_id": node_id})
                return

            if path == "/node/selftest":
                payload = self._read_json()
                node_id = str(payload.get("node_id") or "").strip()
                if node_id:
                    self.runtime.node_registry.record_selftest(node_id, payload)
                self._send(200, {"ok": True, "node_id": node_id})
                return

            if path == "/alerts/enqueue":
                # Cross-process alert injection. The ingest watchdog (and
                # future out-of-band sources) POSTs an alert here after
                # writing the event_log row. The registry decides whether
                # to deliver immediately (a node currently has a user) or
                # defer to the pending queue until presence is detected.
                payload = self._read_json()
                alert = payload.get("alert") if isinstance(payload, dict) else None
                if not isinstance(alert, dict):
                    alert = payload if isinstance(payload, dict) else {}
                result = self.runtime.node_registry.enqueue_for_active_user(alert)
                self._send(200, result)
                return

            if path == "/admin/sync":
                payload = self._read_json()
                node_id = str(payload.get("node_id") or "").strip() or None
                import threading as _t
                result_box: list[dict] = []

                def _run():
                    from sync_manager import sync_all
                    result_box.append(sync_all(node_id=node_id))

                t = _t.Thread(target=_run, daemon=True)
                t.start()
                t.join(timeout=180)
                if result_box:
                    self._send(200, result_box[0])
                else:
                    self._send(504, {"ok": False, "error": "sync timed out"})
                return

            if path == "/guard/alert":
                payload = self._read_json()
                self.runtime.dispatch_guard_alert(payload)
                self._send(200, {"ok": True})
                return

            if path == "/ui":
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                if not message:
                    self._send(400, {"ok": False, "error": "Missing required JSON field: message"})
                    return
                node_id = str(payload.get("node_id") or "scout").strip()
                response = self.runtime.handle_presence(message, node_id=node_id, presence_context=payload)
                active_id = (response or {}).get("active_identity")
                self.runtime.node_registry.update_activity(node_id, identity=active_id)
                self._send(200, {"ok": True, "response": response})
                return

            if path != "/runtime/message":
                self._send(404, {"ok": False, "error": "Not found"})
                return

            payload = self._read_json()
            message = str(payload.get("message") or "").strip()
            node_id = str(payload.get("node_id") or "cli").strip()

            if not message:
                self._send(400, {"ok": False, "error": "Missing required JSON field: message"})
                return

            response = self.runtime.handle(message, node_id=node_id)
            self._send(200, {"ok": True, "response": response})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def _update_person_count(self, node_id: str) -> None:
        """Async: fetch /meta from node's vision service and update person_count."""
        try:
            vision_url = self.server.runtime.node_registry.node_url(node_id, "vision")
            if not vision_url:
                return
            from urllib.request import urlopen as _urlopen
            import json as _json
            with _urlopen(vision_url + "/meta", timeout=2) as r:
                meta = _json.loads(r.read())
            dets = meta.get("detections") or []
            person_count = sum(1 for d in dets if d.get("label") == "person")
            self.server.runtime.node_registry.update_activity(
                node_id, person_count=person_count
            )
        except Exception:
            pass

    def log_message(self, fmt, *args):
        client = self.address_string()
        print(f"[vault_service] {client} " + fmt % args, flush=True)

    def _read_json(self):
        length = int(self.headers.get("content-length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def _send(self, status, payload):
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _identity_from_path(self, path, suffix):
        prefix = "/people/"
        return unquote(path[len(prefix): -len(f"/{suffix}")].strip("/"))


class VaultHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, runtime):
        super().__init__(server_address, handler_class)
        self.runtime = runtime


def run_service(host="127.0.0.1", port=8766):
    runtime = VaultRuntime()
    print(f"[vault_service] model warmup: {json.dumps(runtime.model_warmup, default=str)}", flush=True)
    server = VaultHTTPServer((host, port), VaultRequestHandler, runtime)
    print(f"[vault_service] listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(prog="python3 vault_service.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    run_service(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
