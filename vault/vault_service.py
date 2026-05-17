import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from vault_runtime import VaultRuntime


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
                    services=payload.get("services") or {},
                    capabilities=payload.get("capabilities") or {},
                    modules=payload.get("modules") or {},
                )
                self._send(200, {"ok": True, "node_id": node_id})
                return

            if path == "/node/selftest":
                payload = self._read_json()
                node_id = str(payload.get("node_id") or "").strip()
                if node_id:
                    self.runtime.node_registry.record_selftest(node_id, payload)
                self._send(200, {"ok": True, "node_id": node_id})
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
        print("[vault_service] " + fmt % args, flush=True)

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
