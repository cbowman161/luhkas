#!/usr/bin/env python3
"""display_node HTTP service.

Owns the physical screen for nodes that have a display. Chat UI and `/chat`
belong to luhkas_node; this service renders the kiosk presence face and accepts
display events from other local services.

Endpoints:
  GET  /              — full-screen animated presence face
  GET  /presence/face — full-screen animated presence face
  GET  /presence/face/state — JSON state for the face
  POST /ui/event      — local event sink (user_message, assistant_message,
                        status, alert)
  GET  /health        — service status
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presence_state import read_state, update_state


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] display_node: %(message)s")
log = logging.getLogger("display_node")


_state_lock = threading.Lock()
_history: deque = deque(maxlen=50)
_status: dict = {
    "node_id": os.environ.get("LUHKAS_NODE_ID", "kiosk"),
    "started_at": time.time(),
    "last_event_at": 0.0,
    "muted": False,
}
_FACE_MESSAGE_TTL_SECONDS = float(os.environ.get("DISPLAY_FACE_MESSAGE_TTL_SECONDS", "45"))
_VAULT_URL = os.environ.get("VAULT_SERVICE_URL", os.environ.get("VAULT_CHAT_URL", "http://luhkas-vault.local:7000")).rstrip("/")
_cpu_lock = threading.Lock()
_cpu_last_sample: tuple[float, float] | None = None
_cpu_last_percent: float | None = None
_vault_lock = threading.Lock()
_vault_last_turn_sig = ""
_vault_activity_until = 0.0
_service_cache_lock = threading.Lock()
_service_cache: dict[str, dict] = {}


_PRESENCE_FACE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#000">
<title>LUHKAS</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{width:100%;height:100%;overflow:hidden;background:#000;font-family:'JetBrains Mono','Fira Code',ui-monospace,Menlo,Consolas,monospace;color:#00d4ff}
  body{transition:color 0.8s ease}
  #scene,#glow-scene{position:fixed;inset:0;display:block;pointer-events:none}
  #scene{z-index:2}
  #glow-scene{z-index:3;filter:blur(7px) brightness(5.5) saturate(1.5);mix-blend-mode:screen;opacity:1}
  .hud{position:fixed;z-index:4;pointer-events:none;color:currentColor;text-shadow:0 0 14px currentColor;transition:color 0.8s ease}
  #brand{top:5vh;left:50%;transform:translateX(-50%);font-size:clamp(13px,1.4vw,20px);letter-spacing:0.55em;font-weight:600;opacity:0.62}
  #state{top:9.5vh;left:50%;transform:translateX(-50%);font-size:clamp(10px,1.05vw,15px);letter-spacing:0.4em;opacity:0.55}
  #cap-user{position:fixed;bottom:15vh;left:5vw;right:5vw;text-align:center;font-size:clamp(13px,1.4vw,20px);opacity:0.45;letter-spacing:0.08em;min-height:1.4em}
  #cap-assistant{position:fixed;bottom:6vh;left:5vw;right:5vw;text-align:center;font-size:clamp(22px,3vw,48px);font-weight:600;line-height:1.1;min-height:1.4em;letter-spacing:0.03em;text-shadow:0 0 22px currentColor;color:currentColor}
  #scan{position:fixed;left:0;right:0;height:160px;pointer-events:none;background:linear-gradient(to bottom,transparent,currentColor 50%,transparent);opacity:0.05;mix-blend-mode:screen;animation:scan 9s linear infinite;color:inherit;z-index:6}
  @keyframes scan{from{top:-160px}to{top:100vh}}
  body::after{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at center,transparent 45%,#000 100%);opacity:0.7;z-index:5}
</style>
</head>
<body>
<canvas id="glow-scene"></canvas>
<canvas id="scene"></canvas>
<div id="scan"></div>
<div class="hud" id="brand">L  U  H  K  A  S</div>
<div class="hud" id="state">connecting</div>
<div id="cap-user"></div>
<div id="cap-assistant"></div>
<script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
<script>
(() => {
  const STATIC_CYAN = 0x00d4ff;
  const COLORS = {
    IDLE: 0x00d4ff, GUARDING: 0x0088ff, FOLLOWING: 0xff00d4,
    SEARCHING: 0xffd000, AVOIDING: 0xff2020, MANUAL: 0xf0f0e0,
    SPEAKING: 0xffa030, LISTENING: 0x00ff88, HEARING: 0x88ff44, OFFLINE: 0x445566,
  };
  const canvas = document.getElementById('scene');
  const glowCanvas = document.getElementById('glow-scene');
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
  const glowRenderer = new THREE.WebGLRenderer({canvas: glowCanvas, antialias: true, alpha: true});
  glowRenderer.autoClear = false;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  glowRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  glowRenderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000, 1);
  glowRenderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const glowScene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 4.4);

  const wireMat = new THREE.LineBasicMaterial({color: STATIC_CYAN, transparent: true, opacity: 0.62});
  const innerMat = new THREE.LineBasicMaterial({color: STATIC_CYAN, transparent: true, opacity: 0.82});
  const coreMat = new THREE.MeshBasicMaterial({
    color: COLORS.OFFLINE,
    transparent: true,
    opacity: 1.0,
    depthWrite: false,
  });
  const coreEyeMat = new THREE.MeshBasicMaterial({
    color: COLORS.OFFLINE,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const coreGlowMat = new THREE.MeshBasicMaterial({
    color: COLORS.OFFLINE,
    transparent: true,
    opacity: 0.78,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const dotMat = new THREE.MeshBasicMaterial({color: STATIC_CYAN});
  const coreEyeForward = new THREE.Vector3(0, 0, 1);

  const sigil = new THREE.Group();
  scene.add(sigil);
  const glowSigil = new THREE.Group();
  glowScene.add(glowSigil);

  // ---- Layer 1: Outer dodecahedron shell (12 faces, slow rotation) -------
  const outerGeo = new THREE.DodecahedronGeometry(1.25, 0);
  const outer = new THREE.LineSegments(new THREE.WireframeGeometry(outerGeo), wireMat);
  sigil.add(outer);

  // ---- Layer 2: Middle icosahedron (20 faces, counter-rotating) ---------
  const midGeo = new THREE.IcosahedronGeometry(0.78, 0);
  const mid = new THREE.LineSegments(new THREE.WireframeGeometry(midGeo), innerMat);
  sigil.add(mid);

  // ---- Layer 3: Inner stellated octahedron core -------------------------
  // Back to the original sharp inner crystal, with one subdivision so the
  // vertex/edge count is richer without changing its basic silhouette.
  const octaGeo = new THREE.OctahedronGeometry(0.32, 1);
  const op = octaGeo.attributes.position;
  for (let i = 0; i < op.count; i++) {
    const x = op.getX(i), y = op.getY(i), z = op.getZ(i);
    const p = new THREE.Vector3(x, y, z);
    const directionalSpike = Math.abs(y) > Math.abs(x) && Math.abs(y) > Math.abs(z) ? 1.22 : 1.08;
    p.multiplyScalar(directionalSpike);
    op.setXYZ(i, p.x, p.y, p.z);
  }
  octaGeo.computeVertexNormals();
  const eyeFacePositions = (() => {
    const pos = octaGeo.attributes.position;
    const idx = octaGeo.index;
    let best = null;
    const vertexAt = (n) => {
      const i = idx ? idx.getX(n) : n;
      return new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i));
    };
    const triangleCount = idx ? idx.count / 3 : pos.count / 3;
    for (let f = 0; f < triangleCount; f++) {
      const faceVerts = [vertexAt(f * 3), vertexAt(f * 3 + 1), vertexAt(f * 3 + 2)];
      const edges = [
        {base: [faceVerts[0], faceVerts[1]], apex: faceVerts[2]},
        {base: [faceVerts[1], faceVerts[2]], apex: faceVerts[0]},
        {base: [faceVerts[2], faceVerts[0]], apex: faceVerts[1]},
      ].map((candidate) => ({
        ...candidate,
        length: candidate.base[0].distanceTo(candidate.base[1]),
      })).sort((p, q) => q.length - p.length);
      const candidate = edges[0];
      const base = candidate.base.slice().sort((p, q) => p.x - q.x);
      const apex = candidate.apex;
      const baseMid = new THREE.Vector3().addVectors(base[0], base[1]).multiplyScalar(0.5);
      const baseMidY = baseMid.y;
      const uprightHeight = apex.y - baseMidY;
      const baseSkew = Math.abs(base[0].y - base[1].y);
      const apexCentering = Math.abs(apex.x - baseMid.x);
      const centroid = new THREE.Vector3().addVectors(faceVerts[0], faceVerts[1]).add(faceVerts[2]).multiplyScalar(1 / 3);
      const normal = new THREE.Vector3()
        .subVectors(faceVerts[1], faceVerts[0])
        .cross(new THREE.Vector3().subVectors(faceVerts[2], faceVerts[0]))
        .normalize();
      if (normal.z < 0) normal.multiplyScalar(-1);
      const score = (
        centroid.z
        + normal.z * 0.18
        + uprightHeight * 1.45
        + candidate.length * 0.55
        - baseSkew * 2.2
        - apexCentering * 0.75
        - Math.abs(centroid.x) * 0.12
      );
      if (!best || score > best.score) best = {apex, base, normal, score};
    }
    const lift = best.normal.clone().multiplyScalar(0.004);
    coreEyeForward.copy(best.normal).normalize();
    const ordered = [best.apex, best.base[0], best.base[1]];
    const orderedNormal = new THREE.Vector3()
      .subVectors(ordered[1], ordered[0])
      .cross(new THREE.Vector3().subVectors(ordered[2], ordered[0]))
      .normalize();
    if (orderedNormal.z < 0) {
      [ordered[1], ordered[2]] = [ordered[2], ordered[1]];
    }
    return ordered.flatMap((v) => {
      const p = v.clone().add(lift);
      return [p.x, p.y, p.z];
    });
  })();
  const coreWireGeo = new THREE.WireframeGeometry(octaGeo);
  const core = new THREE.Group();
  const glowCore = new THREE.Group();
  const coreEdges = [];
  const edgePos = coreWireGeo.attributes.position;
  for (let i = 0; i < edgePos.count; i += 2) {
    coreEdges.push([
      new THREE.Vector3(edgePos.getX(i), edgePos.getY(i), edgePos.getZ(i)),
      new THREE.Vector3(edgePos.getX(i + 1), edgePos.getY(i + 1), edgePos.getZ(i + 1)),
    ]);
  }
  const visibleEdgeRadius = 0.006;
  const edgeRadius = 0.012;
  for (const [a, b] of coreEdges) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length();
    if (len <= 0.0001) continue;
    const visibleTube = new THREE.Mesh(new THREE.CylinderGeometry(visibleEdgeRadius, visibleEdgeRadius, len, 8, 1, true), coreMat);
    visibleTube.position.copy(a).add(b).multiplyScalar(0.5);
    visibleTube.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    core.add(visibleTube);
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(edgeRadius, edgeRadius, len, 8, 1, true), coreGlowMat);
    tube.position.copy(a).add(b).multiplyScalar(0.5);
    tube.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    glowCore.add(tube);
  }
  const eyePanelPivot = new THREE.Group();
  const eyePanelGeo = new THREE.BufferGeometry();
  eyePanelGeo.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(eyeFacePositions, 3),
  );
  eyePanelGeo.computeVertexNormals();
  const eyePanel = new THREE.Mesh(eyePanelGeo, coreEyeMat);
  eyePanelPivot.add(eyePanel);
  core.add(eyePanelPivot);
  glowSigil.add(glowCore);
  sigil.add(core);

  // ---- Vertex marker dots on the outer shell (looks like data anchors) --
  const dotGroup = new THREE.Group();
  const dotGeo = new THREE.IcosahedronGeometry(0.022, 1);
  const seen = new Set();
  const ovp = outerGeo.attributes.position;
  for (let i = 0; i < ovp.count; i++) {
    const k = ovp.getX(i).toFixed(2) + ',' + ovp.getY(i).toFixed(2) + ',' + ovp.getZ(i).toFixed(2);
    if (seen.has(k)) continue;
    seen.add(k);
    const d = new THREE.Mesh(dotGeo, dotMat);
    d.position.set(ovp.getX(i), ovp.getY(i), ovp.getZ(i));
    dotGroup.add(d);
  }
  sigil.add(dotGroup);
  // Push the sigil slightly down so the top brand/state labels don't overlap
  sigil.position.y = -0.18;
  glowSigil.position.y = -0.18;

  // ---- State + animation -------------------------------------------------
  let state = 'OFFLINE', speaking = false, listening = false, hearing = false, target = null, eyeTarget = null;
  let vaultActive = false, nodeCpuPercent = 0;
  function setColor(hex) {
    coreMat.color.setHex(hex);
    coreEyeMat.color.setHex(hex);
    coreGlowMat.color.setHex(_lightenHex(hex, 0.72));
  }

  function _lightenHex(hex, amount) {
    const c = new THREE.Color(hex);
    c.lerp(new THREE.Color(0xffffff), amount);
    return c.getHex();
  }

  async function poll() {
    try {
      const r = await fetch('/presence/face/state', {cache: 'no-store'});
      const d = await r.json();
      let s = 'IDLE';
      const ts = d.tracking_state;
      if (ts && ts.behavior_state) s = ts.behavior_state;
      speaking = !!(d.audio_state && d.audio_state.tts && d.audio_state.tts.speaking);
      listening = !!(d.audio_state && d.audio_state.capture && d.audio_state.capture.muted === false);
      // HEARING: audio_node recently produced a transcript (incl. noise-filtered)
      // = we know audio is reaching the STT pipeline. Briefly overrides color.
      const lastTrAt = (d.audio_state && d.audio_state.capture && d.audio_state.capture.last_transcript_at) || 0;
      hearing = lastTrAt > 0 && (Date.now()/1000 - lastTrAt) < 3.0;
      if (speaking) s = 'SPEAKING';
      else if (hearing) s = 'HEARING';
      else if (listening && s === 'IDLE') s = 'LISTENING';
      if (!d.ok || (!d.tracking_state && !d.audio_state)) s = 'OFFLINE';
      state = s;
      vaultActive = !!(d.vault_state && d.vault_state.cognitive_active);
      const vaultUnreachable = !!(d.vault_state && d.vault_state.unreachable);
      const vaultBusy = !!(d.vault_state && (d.vault_state.cognitive_active || d.vault_state.background_active));
      wireMat.color.setHex(vaultUnreachable ? 0xff2020 : (vaultBusy ? 0x00ff66 : STATIC_CYAN));
      nodeCpuPercent = Number((d.node_state && d.node_state.cpu_percent) || 0);
      setColor(COLORS[s] || COLORS.OFFLINE);
      document.getElementById('state').textContent = s;
      document.getElementById('cap-user').textContent = d.latest_user || '';
      document.getElementById('cap-assistant').textContent = d.latest_assistant || '';
      if (d.eye_target && typeof d.eye_target.x_norm === 'number' && typeof d.eye_target.y_norm === 'number') {
        eyeTarget = d.eye_target;
        target = d.eye_target;
      } else if (ts && ts.target && ts.target.center && ts.frame_shape) {
        eyeTarget = target = {
          x_norm: (ts.target.center[0] / ts.frame_shape[1]) - 0.5,
          y_norm: (ts.target.center[1] / ts.frame_shape[0]) - 0.5,
          label: 'tracked',
        };
      } else {
        target = null;
        eyeTarget = null;
      }
    } catch (e) {
      setColor(COLORS.OFFLINE);
      wireMat.color.setHex(0xff2020);
      document.getElementById('state').textContent = 'OFFLINE';
    }
  }
  setInterval(poll, 400); poll();

  window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    glowRenderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  });

  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const delta = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    // Per-layer rotation rate scales with state arousal
    const arousal = state === 'AVOIDING' ? 3.0 : state === 'SEARCHING' ? 1.8 : state === 'FOLLOWING' ? 1.3 : 1.0;
    const vaultBoost = vaultActive ? 3.2 : 1.0;
    const cpuBoost = 1.0 + Math.max(0, Math.min(1, (nodeCpuPercent - 45) / 55)) * 3.4;
    outer.rotation.y += 0.0028 * arousal * vaultBoost;
    outer.rotation.x += 0.0018 * arousal;
    mid.rotation.y -= 0.0052 * arousal * cpuBoost;
    mid.rotation.z += 0.0038 * arousal;
    const coreAimActive = hearing || state === 'FOLLOWING';
    if (coreAimActive) {
      const coreTrack = eyeTarget || target || {x_norm: 0, y_norm: 0};
      const desiredForward = new THREE.Vector3(
        coreTrack.x_norm * 0.9,
        -coreTrack.y_norm * 0.58,
        1,
      ).normalize();
      const coreAimQuat = new THREE.Quaternion().setFromUnitVectors(coreEyeForward, desiredForward);
      core.quaternion.slerp(coreAimQuat, 0.14);
      core.scale.lerp(new THREE.Vector3(1, 1, 1), 0.18);
      coreMat.opacity += (1 - coreMat.opacity) * 0.22;
    } else {
      core.rotation.x += 0.0140 * arousal;
      core.rotation.y += 0.0110 * arousal;
      core.rotation.z += 0.0190 * arousal;
    }
    core.scale.lerp(new THREE.Vector3(1, 1, 1), 0.18);
    coreMat.opacity += (1 - coreMat.opacity) * 0.22;
    if (eyeTarget) {
      const eyeYaw = coreAimActive ? 0 : -eyeTarget.x_norm * 0.72;
      const eyePitch = coreAimActive ? 0 : eyeTarget.y_norm * 0.48;
      eyePanelPivot.rotation.y += (eyeYaw - eyePanelPivot.rotation.y) * 0.16;
      eyePanelPivot.rotation.x += (eyePitch - eyePanelPivot.rotation.x) * 0.16;
      eyePanel.position.x += (eyeTarget.x_norm * 0.05 - eyePanel.position.x) * 0.18;
      eyePanel.position.y += (-eyeTarget.y_norm * 0.035 - eyePanel.position.y) * 0.18;
    } else {
      eyePanelPivot.rotation.y += (Math.sin(t * 0.7) * 0.12 - eyePanelPivot.rotation.y) * 0.04;
      eyePanelPivot.rotation.x += (Math.sin(t * 0.53) * 0.08 - eyePanelPivot.rotation.x) * 0.04;
      eyePanel.position.x += (0 - eyePanel.position.x) * 0.12;
      eyePanel.position.y += (0 - eyePanel.position.y) * 0.12;
    }
    dotGroup.rotation.copy(outer.rotation);
    // Tilt the whole sigil toward the target
    if (target) {
      const yaw = -target.x_norm * 0.7;
      const pitch = target.y_norm * 0.45;
      sigil.rotation.y += (yaw - sigil.rotation.y) * 0.06;
      sigil.rotation.x += (pitch - sigil.rotation.x) * 0.06;
    } else {
      sigil.rotation.y += (Math.sin(t * 0.28) * 0.18 - sigil.rotation.y) * 0.015;
      sigil.rotation.x += (Math.sin(t * 0.41) * 0.09 - sigil.rotation.x) * 0.015;
    }
    // Speaking pulse + breathing
    const breath = 1 + 0.022 * Math.sin(t * 1.2);
    const speakPulse = speaking ? 1 + 0.07 * Math.abs(Math.sin(t * 9)) : 1;
    sigil.scale.setScalar(breath * speakPulse);
    glowSigil.rotation.copy(sigil.rotation);
    glowSigil.scale.copy(sigil.scale);
    glowCore.rotation.copy(core.rotation);
    glowCore.scale.copy(core.scale);
    glowRenderer.clear();
    glowRenderer.render(glowScene, camera);
    renderer.render(scene, camera);
  }
  animate();
})();
</script>
</body></html>"""


def _record_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("timestamp", time.time())
    etype = event.get("type")
    text = str(event.get("text") or event.get("message") or "").strip()
    if etype == "user_message" and text:
        update_state({"latest_user": {"text": text, "source": event.get("source"), "timestamp": event["timestamp"]}})
    elif etype == "assistant_message" and text:
        update_state({"latest_assistant": {"text": text, "source": event.get("source"), "timestamp": event["timestamp"]}})
    elif etype == "status":
        update_state({"display": {"last_status": event, "last_status_at": event["timestamp"]}})
    with _state_lock:
        _history.append(event)
        _status["last_event_at"] = event["timestamp"]
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


def _fetch_local_json(url: str, timeout_s: float = 0.4) -> dict | None:
    """Server-side fetch of a sibling service's JSON endpoint. Returns None on
    any failure — used to enrich /presence/face/state with vision + audio
    data so the browser only hits one origin."""
    return _fetch_json(url, timeout_s)


def _fetch_json(url: str, timeout_s: float = 0.4) -> dict | None:
    try:
        with urlopen(url, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _cache_set(key: str, value: dict | None) -> None:
    if value is None:
        return
    with _service_cache_lock:
        _service_cache[key] = {"at": time.time(), "value": value}


def _cache_get(key: str) -> dict | None:
    with _service_cache_lock:
        entry = _service_cache.get(key) or {}
        value = entry.get("value")
        return value if isinstance(value, dict) else None


def _cache_age(key: str) -> float:
    """Seconds since the last successful set of ``key``. Infinity if never set."""
    with _service_cache_lock:
        entry = _service_cache.get(key) or {}
        at = entry.get("at")
        return (time.time() - float(at)) if at else float("inf")


def _service_cache_loop() -> None:
    urls = {
        "vision_meta": ("http://127.0.0.1:5000/meta", 0.25, 0.18),
        "audio_health": ("http://127.0.0.1:5004/health", 0.5, 0.15),
    }
    next_fetch = {key: 0.0 for key in urls}
    next_fetch["vault_health"] = 0.0
    next_fetch["vault_session"] = 0.0
    while True:
        now = time.time()
        for key, (url, interval, timeout_s) in urls.items():
            if now >= next_fetch[key]:
                _cache_set(key, _fetch_json(url, timeout_s=timeout_s))
                next_fetch[key] = now + interval
        if _VAULT_URL and now >= next_fetch["vault_health"]:
            _cache_set("vault_health", _fetch_json(f"{_VAULT_URL}/health", timeout_s=0.25))
            next_fetch["vault_health"] = now + 2.0
        if _VAULT_URL and now >= next_fetch["vault_session"]:
            _cache_set("vault_session", _fetch_json(f"{_VAULT_URL}/presence/session", timeout_s=0.25))
            next_fetch["vault_session"] = now + 1.2
        time.sleep(0.08)


def _node_runtime_state() -> dict:
    """Small local runtime signal for the display animation.

    Linux nodes expose cumulative CPU counters in /proc/stat. We cache the
    previous sample and report a rolling percent without pulling in psutil.
    """
    global _cpu_last_sample, _cpu_last_percent
    cpu_percent = None
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:8]
        values = [float(v) for v in fields]
        idle = values[3] + values[4]
        total = sum(values)
        with _cpu_lock:
            if _cpu_last_sample is not None:
                last_idle, last_total = _cpu_last_sample
                idle_delta = idle - last_idle
                total_delta = total - last_total
                if total_delta > 0:
                    cpu_percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
                    _cpu_last_percent = cpu_percent
            _cpu_last_sample = (idle, total)
            if cpu_percent is None:
                cpu_percent = _cpu_last_percent
    except Exception:
        cpu_percent = None
    payload = {"cpu_percent": round(cpu_percent, 1) if cpu_percent is not None else None}
    try:
        load1, load5, load15 = os.getloadavg()
        payload["loadavg"] = [round(load1, 2), round(load5, 2), round(load15, 2)]
    except Exception:
        pass
    return payload


def _vault_turn_is_cognitive(turn: dict) -> bool:
    if not isinstance(turn, dict):
        return False
    actions = turn.get("actions")
    if isinstance(actions, list) and actions:
        return True
    provenance = turn.get("answer_provenance") if isinstance(turn.get("answer_provenance"), dict) else {}
    sources = provenance.get("sources") if isinstance(provenance, dict) else []
    if isinstance(sources, list):
        source_names = {
            str(src.get("name") or "").lower()
            for src in sources
            if isinstance(src, dict)
        }
        if source_names.intersection({
            "chat_model",
            "memory_store",
            "world_knowledge",
            "web_search",
            "learned_capability",
            "code_monkey",
        }):
            return True
    route = provenance.get("route") if isinstance(provenance, dict) else {}
    if isinstance(route, dict) and route.get("from_cache") is False:
        return True
    return False


def _vault_runtime_state() -> dict:
    global _vault_last_turn_sig, _vault_activity_until
    now = time.time()
    health = _cache_get("vault_health") if _VAULT_URL else None
    session = _cache_get("vault_session") if _VAULT_URL else None
    vault_age = _cache_age("vault_health") if _VAULT_URL else float("inf")
    unreachable = bool(_VAULT_URL) and vault_age > 8.0
    code_monkey = (health or {}).get("code_monkey") if isinstance(health, dict) else {}
    background_active = bool((health or {}).get("active_task_id")) if isinstance(health, dict) else False
    if isinstance(code_monkey, dict):
        background_active = background_active or bool(code_monkey.get("active_workers") or code_monkey.get("queued_tasks"))

    latest_turn = {}
    turns = (session or {}).get("turns") if isinstance(session, dict) else None
    if isinstance(turns, list) and turns:
        latest_turn = turns[-1] if isinstance(turns[-1], dict) else {}
    sig = json.dumps({
        "message": latest_turn.get("message"),
        "response": latest_turn.get("response"),
        "actions": latest_turn.get("actions"),
    }, sort_keys=True, default=str) if latest_turn else ""
    cognitive_turn = _vault_turn_is_cognitive(latest_turn)
    with _vault_lock:
        if sig and sig != _vault_last_turn_sig:
            _vault_last_turn_sig = sig
            if cognitive_turn:
                _vault_activity_until = now + float(os.environ.get("DISPLAY_VAULT_ACTIVITY_HOLD_SECONDS", "8"))
        cognitive_active = background_active or now < _vault_activity_until
        remaining = max(0.0, _vault_activity_until - now)
    return {
        "ok": bool(health and health.get("ok")) and not unreachable,
        "url": _VAULT_URL,
        "unreachable": unreachable,
        "cognitive_active": cognitive_active,
        "cognitive_remaining_seconds": round(remaining, 1),
        "background_active": background_active,
        "latest_turn_cognitive": cognitive_turn,
    }


_EYE_TRACKING_INVERT_180 = os.environ.get("CAMERA_TRANSFORM_180", "").lower() in ("1", "true", "yes")


def _eye_target_from_vision_meta(vision_meta: dict | None) -> dict | None:
    if not isinstance(vision_meta, dict):
        return None
    tracking = vision_meta.get("tracking_state") if isinstance(vision_meta.get("tracking_state"), dict) else {}
    tracked = tracking.get("target") if isinstance(tracking.get("target"), dict) else {}
    center = tracked.get("center") if isinstance(tracked, dict) else None
    track_shape = tracking.get("frame_shape")
    if center and len(center) >= 2 and track_shape:
        try:
            xn = float(center[0]) / float(track_shape[1]) - 0.5
            yn = float(center[1]) / float(track_shape[0]) - 0.5
            if _EYE_TRACKING_INVERT_180:
                xn, yn = -xn, -yn
            return {
                "x_norm": xn,
                "y_norm": yn,
                "label": str(tracked.get("label") or "tracked"),
                "confidence": tracked.get("confidence"),
                "priority": 1,
                "source": "tracking",
            }
        except Exception:
            return None
    return None


def _presence_face_state() -> dict:
    state = _state_snapshot()
    presence = read_state(max_age_seconds=90)
    now = time.time()
    status = state["status"]
    bus_user = presence.get("latest_user") if _fresh_section(presence, "latest_user", "timestamp", _FACE_MESSAGE_TTL_SECONDS) else {}
    bus_assistant = presence.get("latest_assistant") if _fresh_section(presence, "latest_assistant", "timestamp", _FACE_MESSAGE_TTL_SECONDS) else {}
    latest_user = state["last_user_message"] or bus_user or {}
    latest_assistant = state["last_assistant_message"] or bus_assistant or {}
    last_event_at = float(status.get("last_event_at") or status.get("started_at") or time.time())
    for item in (bus_user, bus_assistant):
        if isinstance(item, dict):
            last_event_at = max(last_event_at, float(item.get("timestamp") or 0.0))
    age_seconds = round(max(0.0, time.time() - last_event_at), 1)
    if age_seconds > _FACE_MESSAGE_TTL_SECONDS:
        latest_user = {}
        latest_assistant = {}
    vision_meta = _cache_get("vision_meta") or {}
    audio_health = _cache_get("audio_health") or {}
    audio_bus = presence.get("audio") if isinstance(presence.get("audio"), dict) else {}
    audio_state = dict(audio_health)
    if audio_bus:
        tts = dict(audio_state.get("tts") or {})
        capture = dict(audio_state.get("capture") or {})
        bus_speaking = bool(audio_bus.get("speaking")) and now - float(audio_bus.get("speaking_started_at") or 0.0) < 120.0
        tts["speaking"] = bool(bus_speaking or tts.get("speaking"))
        tts["interrupt_enabled"] = False
        if audio_bus.get("last_transcript_at") and now - float(audio_bus.get("last_transcript_at") or 0.0) < _FACE_MESSAGE_TTL_SECONDS:
            capture["last_transcript_at"] = audio_bus.get("last_transcript_at")
            capture["last_transcript_text"] = audio_bus.get("last_transcript_text", capture.get("last_transcript_text", ""))
        audio_state["tts"] = tts
        audio_state["capture"] = capture
    conversation = presence.get("conversation") if isinstance(presence.get("conversation"), dict) else {}
    vault_state = _vault_runtime_state()
    thinking_started = float(conversation.get("thinking_started_at") or 0.0)
    thinking_ended = float(conversation.get("thinking_ended_at") or 0.0)
    if conversation.get("thinking") and thinking_started >= thinking_ended and now - thinking_started < 45.0:
        vault_state["cognitive_active"] = True
        vault_state["local_thinking"] = True
    return {
        "ok": True,
        "node_id": status.get("node_id"),
        "age_seconds": age_seconds,
        "latest_user": str(latest_user.get("text") or latest_user.get("message") or "").strip(),
        "latest_assistant": str(latest_assistant.get("text") or latest_assistant.get("message") or "").strip(),
        "status": status,
        "tracking_state": (vision_meta or {}).get("tracking_state"),
        "eye_target": _eye_target_from_vision_meta(vision_meta),
        "audio_state": audio_state,
        "presence_state": presence,
        "node_state": _node_runtime_state(),
        "vault_state": vault_state,
    }


def _fresh_section(state: dict, key: str, timestamp_key: str, ttl_s: float) -> bool:
    item = state.get(key) if isinstance(state, dict) else None
    if not isinstance(item, dict):
        return False
    try:
        timestamp = float(item.get(timestamp_key) or 0.0)
    except Exception:
        return False
    return timestamp > 0 and time.time() - timestamp <= ttl_s


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/" or path == "/presence/face":
            self._html(_PRESENCE_FACE_HTML)
        elif path == "/presence/face/state":
            self._json(_presence_face_state())
        elif path == "/health":
            self._json({
                "ok": True,
                "service": "display_node",
                "history_size": len(_history),
                "surface": "presence_face",
            })
        else:
            self.send_error(404)

    def _html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        body = self._read_json()
        if body is None:
            return
        if path == "/ui/event":
            _record_event(body)
            self._json({"ok": True})
        else:
            self.send_error(404)

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

    threading.Thread(target=_service_cache_loop, daemon=True, name="display-service-cache").start()
    log.info("listening on http://%s:%s (surface=presence_face)", host, port)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
