import argparse
import hmac
import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from streaming import StreamSink, reset_stream_sink, set_stream_sink
from vault_runtime import VaultRuntime


# Shared-secret auth for the presence endpoints. Opt-in: when
# VAULT_PRESENCE_SECRET is unset, all endpoints work without auth (current
# behaviour). When set, /presence/message and /presence/message/stream
# require a matching ``Authorization: Bearer <secret>`` header. The
# presence proxy forwards the header from its own env (set by the same
# value at deploy time), so the audio_node never needs to know the
# secret — only vault and the proxy do.
_PRESENCE_SECRET = os.environ.get("VAULT_PRESENCE_SECRET", "").strip()
_AUTH_PROTECTED_PATHS = {"/presence/message", "/presence/message/stream"}


_PUSHABLE_EVENT_TYPES = {
    "learn_succeeded", "learn_failed", "learn_needs_install",
    "install_succeeded", "install_failed",
    "world_ingest_stalled", "world_ingest_completed",
}


_VAULT_FACE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#05070a">
<title>LUHKAS Presence</title>
<style>
*{box-sizing:border-box} html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#05070a;color:#eef7ff;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{display:grid;place-items:center}
.stage{position:relative;width:100vw;height:100vh;min-height:100vh;background:radial-gradient(circle at 50% 42%,#18232a 0,#091015 48%,#030506 100%);isolation:isolate}
.stage:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,rgba(78,187,181,.11),transparent 36%,rgba(255,178,91,.08) 70%,transparent);mix-blend-mode:screen}
.stage:after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.035) 0 1px,transparent 1px 4px);opacity:.18;pointer-events:none}
.presence{position:absolute;inset:0;display:grid;grid-template-rows:minmax(0,1fr) auto;align-items:center;justify-items:center;padding:5.5vh 5vw 4.5vh}
.face-wrap{position:relative;width:min(72vw,72vh);aspect-ratio:1;display:grid;place-items:center}
.aura{position:absolute;inset:3%;border-radius:50%;background:conic-gradient(from var(--spin),rgba(102,231,214,.12),rgba(255,191,120,.17),rgba(144,183,255,.16),rgba(102,231,214,.12));filter:blur(16px);animation:spin 16s linear infinite,pulse 4s ease-in-out infinite}
.orb{position:absolute;inset:10%;border-radius:50%;background:radial-gradient(circle at 50% 40%,#122932 0,#071014 68%,#020304 100%);box-shadow:0 0 0 1px rgba(180,245,255,.12),0 28px 120px rgba(46,198,190,.24),inset 0 0 70px rgba(107,219,209,.12)}
.orb:before{content:"";position:absolute;inset:9%;border-radius:50%;border:1px solid rgba(167,240,235,.18);box-shadow:inset 0 0 45px rgba(102,231,214,.11)}
.eyes{position:absolute;top:37%;left:22%;right:22%;height:18%;display:flex;align-items:center;justify-content:space-between}
.eye{width:28%;height:78%;border-radius:999px;background:#dffcff;box-shadow:0 0 20px rgba(160,246,255,.82),0 0 70px rgba(45,205,232,.42);transform:scaleY(var(--eye-open))}
.pupil{width:38%;height:55%;margin:8% auto;border-radius:50%;background:#061318;box-shadow:inset 0 0 8px rgba(255,255,255,.18);transform:translate(var(--look-x),var(--look-y))}
.mouth{position:absolute;top:59%;width:32%;height:9%;border-radius:0 0 999px 999px;border-bottom:clamp(5px,1.2vh,10px) solid #c9fbff;filter:drop-shadow(0 0 13px rgba(135,238,255,.72));transform:scaleX(var(--mouth-x)) scaleY(var(--mouth-y))}
.signal{position:absolute;inset:0;border-radius:50%;border:1px solid rgba(135,238,255,.12);animation:ripple 4.8s ease-out infinite}
.signal.two{animation-delay:1.6s}.signal.three{animation-delay:3.2s}
.name{position:absolute;top:6vh;left:6vw;font-size:clamp(24px,4vw,54px);font-weight:700;letter-spacing:0;color:#f5fbff;text-shadow:0 0 30px rgba(120,230,255,.18)}
.state{position:absolute;top:6.8vh;right:6vw;font-size:clamp(14px,1.8vw,24px);color:#a8bdc5;text-align:right}
.caption{width:min(1120px,88vw);min-height:16vh;display:grid;align-content:end;gap:1.4vh;text-align:center;text-wrap:balance}
.line{font-size:clamp(26px,5vw,76px);line-height:1.04;font-weight:650;letter-spacing:0;color:#f4fbff;text-shadow:0 0 38px rgba(96,223,229,.16)}
.subline{font-size:clamp(14px,1.7vw,24px);color:#a9c2c9;min-height:1.4em}
.awake .orb{box-shadow:0 0 0 1px rgba(180,245,255,.16),0 28px 140px rgba(46,198,190,.34),inset 0 0 84px rgba(107,219,209,.18)}
.speaking .mouth{animation:talk .28s ease-in-out infinite alternate}.speaking .aura{filter:blur(12px);opacity:1}
.alert .aura{background:conic-gradient(from var(--spin),rgba(255,114,114,.25),rgba(255,198,91,.2),rgba(255,114,114,.25))}
.sleepy{--eye-open:.42;--mouth-y:.55}
@property --spin{syntax:"<angle>";inherits:false;initial-value:0deg}
:root{--eye-open:1;--look-x:0px;--look-y:0px;--mouth-x:1;--mouth-y:1}
@keyframes spin{to{--spin:360deg}} @keyframes pulse{50%{transform:scale(1.035);opacity:.84}} @keyframes ripple{0%{transform:scale(.72);opacity:.36}100%{transform:scale(1.13);opacity:0}} @keyframes talk{from{transform:scaleX(.82) scaleY(.62)}to{transform:scaleX(1.16) scaleY(1.4)}}
</style>
</head>
<body>
<div class="stage">
  <div class="presence" id="presence">
    <div class="name">LUHKAS</div>
    <div class="state" id="state">connecting</div>
    <div class="face-wrap" aria-hidden="true">
      <div class="aura"></div><div class="signal"></div><div class="signal two"></div><div class="signal three"></div>
      <div class="orb"></div>
      <div class="eyes"><div class="eye"><div class="pupil"></div></div><div class="eye"><div class="pupil"></div></div></div>
      <div class="mouth"></div>
    </div>
    <div class="caption"><div class="line" id="line">Listening.</div><div class="subline" id="subline"></div></div>
  </div>
</div>
<script>
const qs=new URLSearchParams(location.search); const nodeId=qs.get("node_id")||"kiosk";
const presence=document.getElementById("presence"), line=document.getElementById("line"), subline=document.getElementById("subline"), state=document.getElementById("state");
let lastText="", lastChange=Date.now(), speakingUntil=0;
function setText(text, detail){ text=(text||"Listening.").trim(); if(text!==lastText){ lastText=text; lastChange=Date.now(); speakingUntil=Date.now()+Math.min(9000,1200+text.length*48); } line.textContent=text; subline.textContent=detail||""; }
function applyMood(data){
  const now=Date.now(); const health=data.node&&data.node.ok; const stale=data.node&&data.node.age_seconds>90; const alerts=(data.events||[]).length;
  presence.classList.toggle("awake", !!health && !stale); presence.classList.toggle("sleepy", stale || !health); presence.classList.toggle("alert", alerts>0); presence.classList.toggle("speaking", now<speakingUntil);
  state.textContent=(data.identity||"presence")+" / "+(health?"online":"waiting");
  document.documentElement.style.setProperty("--look-x", (Math.sin(now/1700)*7).toFixed(1)+"px");
  document.documentElement.style.setProperty("--look-y", (Math.cos(now/2300)*3).toFixed(1)+"px");
  document.documentElement.style.setProperty("--eye-open", (stale ? .62 : 1).toString());
}
async function tick(){
  try{
    const r=await fetch(`/vault/face/state?node_id=${encodeURIComponent(nodeId)}`,{cache:"no-store"});
    const data=await r.json();
    const msg=data.latest_assistant||data.latest_user||"Listening.";
    const detail=data.latest_user&&data.latest_assistant?data.latest_user:"";
    setText(msg, detail);
    applyMood(data);
  }catch(e){ state.textContent="reconnecting"; presence.classList.add("sleepy"); }
}
setInterval(()=>presence.classList.toggle("speaking",Date.now()<speakingUntil),120);
setInterval(tick,1200); tick();
</script>
</body>
</html>"""


def _event_age_seconds(event: dict) -> float | None:
    created_at = event.get("created_at") if isinstance(event, dict) else None
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
    except Exception:
        return None


def _presence_face_state(runtime, node_id: str) -> dict:
    node_id = (node_id or "kiosk").strip()
    session = {}
    try:
        session = runtime.scout.session() or {}
    except Exception:
        session = {}
    turns = session.get("turns") if isinstance(session, dict) else []
    latest_user = ""
    latest_assistant = ""
    if isinstance(turns, list):
        for turn in reversed(turns[-20:]):
            if not latest_assistant:
                latest_assistant = str(turn.get("response") or turn.get("message") or "").strip()
            if not latest_user:
                latest_user = str(turn.get("message") or "").strip()
            if latest_user and latest_assistant:
                break
    nodes = {}
    try:
        nodes = (runtime.node_registry.health_summary() or {}).get("nodes") or {}
    except Exception:
        nodes = {}
    raw_node = nodes.get(node_id) if isinstance(nodes, dict) else {}
    node = {"ok": False, "age_seconds": None, "person_count": 0}
    if isinstance(raw_node, dict):
        now = time.time()
        last_active = raw_node.get("last_active_at") or 0
        node = {
            "ok": bool((raw_node.get("selftest") or {}).get("ok", raw_node.get("last_active_at"))),
            "age_seconds": round(max(0.0, now - float(last_active or 0)), 1) if last_active else None,
            "person_count": raw_node.get("person_count") or 0,
            "last_identity_seen": raw_node.get("last_identity_seen"),
        }
    events = []
    try:
        events = [
            event for event in (_events_feed(runtime, 0).get("events") or [])
            if (_event_age_seconds(event) or 999999) <= 900
        ][-3:]
    except Exception:
        events = []
    return {
        "ok": True,
        "node_id": node_id,
        "identity": session.get("active_identity") if isinstance(session, dict) else None,
        "latest_user": latest_user,
        "latest_assistant": latest_assistant,
        "node": node,
        "events": events,
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

            if path == "/vault/face":
                self._send_html(200, _VAULT_FACE_HTML)
                return

            if path == "/vault/face/state":
                qs = parse_qs(urlparse(self.path).query)
                node_id = (qs.get("node_id") or ["kiosk"])[0]
                self._send(200, _presence_face_state(self.runtime, node_id))
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

            # Optional shared-secret auth on the presence endpoints. Off
            # when VAULT_PRESENCE_SECRET is unset (no behavioural change
            # for unauthed deployments / dev).
            if _PRESENCE_SECRET and path in _AUTH_PROTECTED_PATHS:
                auth = self.headers.get("Authorization", "") or ""
                expected = f"Bearer {_PRESENCE_SECRET}"
                if not hmac.compare_digest(auth, expected):
                    self._send(401, {"ok": False, "error": "unauthorized"})
                    return

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

            if path == "/presence/message/stream":
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                node_id = str(
                    payload.get("node_id")
                    or payload.get("source")
                    or payload.get("client")
                    or "scout"
                ).strip()
                if not message:
                    self._send(400, {"ok": False, "error": "Missing required JSON field: message"})
                    return
                self._handle_presence_stream(message, node_id, payload)
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

    def _handle_presence_stream(self, message, node_id, payload):
        """Run handle_presence with a streaming sink, emitting NDJSON events.

        Wire format (one JSON object per line, application/x-ndjson):
          {"type": "start"}                  — connection open, work begins
          {"type": "delta", "text": "..."}   — raw model token (zero or more)
          {"type": "done",  "text": "..."}   — terminal; ``text`` is the full
                                               streamed concatenation, or the
                                               full response text if no tokens
                                               streamed (deterministic route)
          {"type": "error", "error": "..."}  — exception during handling

        The node speaks whatever the LLM streams. Vault does NOT post-validate
        the streamed text — by the time a token reaches the client it's
        already queued for TTS, so a "take it back" event would be too late.
        Non-streaming requests via /presence/message still get full
        sanitizer / validator / required_terms enforcement.
        """
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event):
            try:
                self.wfile.write((json.dumps(event, default=str) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        emit({"type": "start"})

        accumulated_parts: list[str] = []

        def _on_sink_event(kind: str, text: str) -> None:
            if kind == "delta":
                accumulated_parts.append(text)
                emit({"type": "delta", "text": text})
            # Other sink kinds intentionally dropped — the node only
            # acts on delta / done.

        sink = StreamSink(_on_sink_event)
        token = set_stream_sink(sink)
        response = None
        try:
            response = self.runtime.handle_presence(
                message, node_id=node_id, presence_context=payload
            )
        except Exception as exc:
            emit({"type": "error", "error": str(exc)})
            return
        finally:
            reset_stream_sink(token)

        import threading as _t
        _t.Thread(target=self._update_person_count, args=(node_id,), daemon=True).start()

        streamed_text = "".join(accumulated_parts).strip()
        # Deterministic / non-composer routes don't stream — emit their
        # final text as a single delta so the node receives it through
        # the same wire format.
        if not streamed_text and isinstance(response, dict):
            final_text = str(
                response.get("tts")
                or response.get("message")
                or response.get("response")
                or ""
            )
            if final_text:
                emit({"type": "delta", "text": final_text})
                streamed_text = final_text
        emit({"type": "done", "text": streamed_text})

    def _send_html(self, status, html):
        body = str(html).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
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
