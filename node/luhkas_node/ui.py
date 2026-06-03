"""LUHKAS node UI composition shell."""
from __future__ import annotations

import logging
import os
from typing import Iterable


log = logging.getLogger("luhkas_node.ui")


# Declarative registry: which *_node packages contribute UI sidebar sections,
# and in what order they appear. Add a row here when a new *_node ships a
# `ui_sections()` callable in its `ui` submodule. Modules not listed here
# are silently skipped even if present in the node profile. Modules listed
# here but absent from the node profile are skipped at render time.
_MODULE_UI_REGISTRY: list[tuple[str, str]] = [
    ("camera_node", "camera_node.ui"),
    ("pantilt_node", "pantilt_node.ui"),
    ("rover_node", "rover_node.ui"),
    ("light_node", "light_node.ui"),
]


def _collect_sections(modules: set[str]) -> list[str]:
    sections: list[str] = []
    for module_name, import_path in _MODULE_UI_REGISTRY:
        if module_name not in modules:
            continue
        try:
            mod = __import__(import_path, fromlist=["ui_sections"])
        except ImportError as exc:
            log.warning("Skipping ui_sections for %s: import failed (%s)", module_name, exc)
            continue
        try:
            sections.extend(mod.ui_sections())
        except Exception as exc:
            log.warning("Skipping ui_sections for %s: ui_sections() raised (%s)", module_name, exc)
    return sections


def ui_html(node_label: str | None = None, modules: Iterable[str] | None = None) -> str:
    """Render the per-node web UI.

    ``node_label`` identifies which node is serving the page; appears in
    the browser tab title and the header. Falls back to ``LUHKAS_NODE_ID``
    env var, then to ``'NODE'``.

    ``modules`` is the iterable of ``*_node`` packages installed on this
    node (typically from ``node/profiles/<id>.json``'s ``modules`` list).
    Only modules in this collection AND registered in
    ``_MODULE_UI_REGISTRY`` contribute sidebar cards. Pass an empty list /
    ``None`` to render no sidebar cards.
    """
    label = (node_label or os.environ.get("LUHKAS_NODE_ID") or "node").upper()
    modules_set = set(modules or [])
    sections = "\n\n".join(_collect_sections(modules_set))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUHKAS - {label}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#eee;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;flex-shrink:0}}
h1{{font-size:.85rem;font-weight:700;letter-spacing:2px;color:#bbb}}
#dot{{font-size:1rem;color:#555;transition:color .3s}}
#dot.live{{color:#4c4;animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
#hdr-right{{margin-left:auto;font-size:.75rem;color:#666}}
main{{display:flex;flex:1;overflow:hidden}}
#feed{{flex:1;min-width:0;background:#000;display:flex;align-items:center;justify-content:center}}
#feed img{{max-width:100%;max-height:100%;object-fit:contain;display:block}}
aside{{width:290px;flex-shrink:0;overflow-y:auto;background:#161616;border-left:1px solid #252525;padding:10px;display:flex;flex-direction:column;gap:8px}}
.dpad{{display:grid;grid-template-areas:'. up . ''left ctr right''. dn . ';grid-template-columns:1fr 1fr 1fr;gap:5px;width:130px;margin:4px auto}}
.dp{{width:40px;height:40px;font-size:1rem;border-radius:5px;border:1px solid #333;background:#222;color:#888;cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none;-webkit-user-select:none;touch-action:none}}
.dp:active,.dp.held{{background:#2a2a2a;color:#eee}}
.dp-up{{grid-area:up}}.dp-dn{{grid-area:dn}}.dp-left{{grid-area:left}}.dp-right{{grid-area:right}}
.dp-ctr{{grid-area:ctr;background:#163016;border-color:#2e6a2e;color:#6c6;font-size:.7rem}}
.card{{background:#1c1c1c;border:1px solid #252525;border-radius:5px;padding:9px;display:flex;flex-direction:column;gap:7px}}
.card-title{{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#666;margin-bottom:1px}}
.row{{display:flex;align-items:center;justify-content:space-between;gap:6px}}
.lbl{{font-size:.75rem;color:#888}}
.val{{font-size:.75rem;font-weight:600;color:#ccc}}
button{{font-size:.7rem;padding:3px 9px;border-radius:3px;border:1px solid #333;cursor:pointer;background:#222;color:#999;transition:all .15s;min-width:42px}}
button.on{{background:#163016;border-color:#2e6a2e;color:#6c6}}
button.off{{background:#222;border-color:#333;color:#666}}
.srow{{display:flex;flex-direction:column;gap:2px}}
.slbls{{display:flex;justify-content:space-between;font-size:.7rem;color:#777}}
input[type=range]{{width:100%;height:4px;accent-color:#3a7;cursor:pointer;margin:2px 0}}
input[type=text]{{font-size:.72rem;padding:3px 6px;border-radius:3px;border:1px solid #333;background:#161616;color:#ccc;width:100%}}
input[type=text]:focus{{outline:none;border-color:#3a7}}
.badge{{font-size:.65rem;padding:2px 7px;border-radius:3px;font-weight:700;letter-spacing:.5px}}
.badge.clear{{background:#0e2a0e;color:#4a4;border:1px solid #1e4a1e}}
.badge.blocked{{background:#2a0e0e;color:#e44;border:1px solid #4a1e1e;animation:pulse .5s step-end infinite}}
.bhv{{font-size:.7rem;font-weight:700;letter-spacing:.5px;padding:2px 8px;border-radius:3px}}
.bhv-idle{{background:#1a1a1a;color:#555;border:1px solid #2a2a2a}}
.bhv-following{{background:#0e2a0e;color:#4c4;border:1px solid #1e4a1e}}
.bhv-searching{{background:#2a2a0e;color:#cc4;border:1px solid #4a4a1e;animation:pulse 1s ease-in-out infinite}}
.bhv-guarding{{background:#2a1a0e;color:#c84;border:1px solid #4a3a1e}}
.bhv-avoiding{{background:#2a0e0e;color:#e44;border:1px solid #4a1e1e;animation:pulse .5s step-end infinite}}
.det-item{{display:flex;justify-content:space-between;align-items:center;font-size:.7rem;padding:3px 0;border-bottom:1px solid #1e1e1e}}
.det-item:last-child{{border:none}}
.det-lbl{{color:#bbb}}.det-lbl.tgt{{color:#6cf}}
.det-conf{{color:#555}}
.none{{font-size:.7rem;color:#444;font-style:italic}}
.divider{{border:none;border-top:1px solid #222;margin:2px 0}}
#chat-panel{{width:280px;flex-shrink:0;background:#141414;border-left:1px solid #252525;border-right:1px solid #252525;display:flex;flex-direction:column;padding:10px;gap:6px}}
#chat-panel .panel-title{{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#666;flex-shrink:0}}
.chat-msgs{{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:5px}}
.chat-msg{{font-size:.73rem;line-height:1.4;padding:5px 7px;border-radius:4px;word-break:break-word;max-width:96%}}
.chat-msg.user{{background:#1a2a1a;border:1px solid #2a4a2a;color:#8e8;align-self:flex-end}}
.chat-msg.bot{{background:#1c1c2a;border:1px solid #2a2a4a;color:#bbd;align-self:flex-start}}
.chat-msg.err{{background:#2a1a1a;border:1px solid #4a2a2a;color:#d88;align-self:flex-start}}
.chat-msg.alert{{background:#2a200e;border:1px solid #4a3a1e;color:#cb6;align-self:flex-start;font-size:.7rem;font-style:italic}}
.chat-row{{display:flex;gap:4px;flex-shrink:0}}
.chat-row input[type=text]{{flex:1}}
.chat-row button{{padding:4px 10px;border-color:#2e6a2e;background:#163016;color:#6c6}}
.chat-row button:disabled{{opacity:.4;cursor:default}}
.chat-time{{display:block;font-size:.6rem;color:#444;margin-top:3px;text-align:right}}
</style>
</head>
<body>
<header>
  <span id="dot">●</span>
  <h1>LUHKAS - {label}</h1>
  <span id="hdr-right">fr -</span>
</header>
<main>
  <div id="feed"><img id="live-img" src="/video_feed" alt="live feed"></div>

  <div id="chat-panel">
    <div class="panel-title">Chat</div>
    <div id="chat-msgs" class="chat-msgs"></div>
    <div class="chat-row">
      <input type="text" id="chat-inp" placeholder="Say something..." onkeydown="if(event.key==='Enter')sendChat()">
      <button id="chat-btn" onclick="sendChat()">&#8629;</button>
    </div>
  </div>

  <aside>
{sections}
  </aside>
</main>
<script>
var STATE = {{}};
var ptInterval = null;
var ptRun = 0;
var PAN_STEP = 5;
var TILT_STEP = 5;
var lastIdentityPromptId = null;
var pollTimer = null;
var lastDetHtml = '';

function q(id){{return document.getElementById(id)}}

function post(url, body){{
  return fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
}}

function disableTrackingForManual(){{
  if(STATE['tracking_enabled']){{
    STATE['tracking_enabled'] = false;
    syncBtn('tracking_enabled', false);
    return post('/tracking', {{enabled: false}}).catch(function(){{}});
  }}
  return Promise.resolve();
}}

function startPT(panDir, tiltDir){{
  stopPT();
  var run = ++ptRun;
  function send(){{post('/pantilt',{{pan:panDir*PAN_STEP,tilt:tiltDir*TILT_STEP}});}}
  disableTrackingForManual().then(function(){{
    if(run !== ptRun) return;
    send();
    ptInterval = setInterval(send, 180);
  }});
}}

function stopPT(){{
  ptRun++;
  if(ptInterval){{clearInterval(ptInterval);ptInterval=null;}}
}}

function centerCamera(){{
  disableTrackingForManual().then(function(){{
    post('/pantilt',{{center:true}});
  }});
}}

document.addEventListener('keydown', function(e){{
  if(e.target.tagName==='INPUT') return;
  if(e.repeat) return;
  var dirs={{ArrowUp:[0,1],ArrowDown:[0,-1],ArrowLeft:[-1,0],ArrowRight:[1,0]}};
  var d=dirs[e.key];
  if(!d) return;
  e.preventDefault();
  startPT(d[0],d[1]);
}});

document.addEventListener('keyup', function(e){{
  var keys=['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
  if(keys.indexOf(e.key)>=0) stopPT();
}});

function setting(key){{
  var cur = STATE[key];
  var next = !cur;
  STATE[key] = next;
  var b = q('btn-'+key);
  if(b){{b.textContent=next?'ON':'OFF'; b.className=next?'on':'off';}}
  var body = {{}}; body[key] = next;
  post('/settings', body);
}}

function tog(stateKey, url, bodyKey){{
  var cur = STATE[stateKey];
  var next = !cur;
  STATE[stateKey] = next;
  var b = q('btn-'+stateKey);
  if(b){{b.textContent=next?'ON':'OFF'; b.className=next?'on':'off';}}
  var body = {{}}; body[bodyKey] = next;
  post(url, body);
}}

function sld(el, key, decimals, url, bodyKey){{
  var v = parseFloat(el.value);
  var vq = q('val-'+key);
  if(vq) vq.textContent = v.toFixed(decimals);
  var k = bodyKey || key;
  var body = {{}}; body[k] = v;
  post(url, body);
}}

function syncBtn(key, val){{
  STATE[key] = !!val;
  var b = q('btn-'+key);
  if(b){{
    var label = val ? 'ON' : 'OFF';
    var klass = val ? 'on' : 'off';
    if(b.textContent !== label) b.textContent = label;
    if(b.className !== klass) b.className = klass;
  }}
}}

function syncSld(key, val, decimals){{
  var el = q('sld-'+key);
  var vq = q('val-'+key);
  if(el && document.activeElement !== el){{
    var display = parseFloat(val).toFixed(decimals||2);
    if(el.value !== String(val)) el.value = val;
    if(vq && vq.textContent !== display) vq.textContent = display;
  }}
}}

function syncText(id, val){{
  var el = q(id);
  if(el && document.activeElement !== el) el.value = val || '';
}}

function setText(id, text){{
  var el = q(id);
  text = String(text);
  if(el && el.textContent !== text) el.textContent = text;
}}

function schedulePoll(delay){{
  if(pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay);
}}

function poll(){{
  var nextDelay = 1000;
  fetch('/meta').then(function(r){{return r.json()}}).then(function(d){{
    q('dot').className = 'live';
    q('hdr-right').textContent = 'fr ' + d.frame_id;

    syncBtn('tracking_enabled', d.tracking_enabled);
    syncBtn('guard_mode', d.guard_mode);
    syncBtn('follow_enabled', d.follow_enabled);
    syncBtn('search_movement_enabled', d.search_movement_enabled);
    syncBtn('wheel_enabled', d.wheel_enabled);
    syncBtn('manual_controller_enabled', d.gamepad && d.gamepad.enabled);
    syncBtn('camera_light_auto_enabled', d.camera_light_auto_enabled);
    syncBtn('camera_light_enabled', d.camera_light_enabled);
    syncBtn('edge_reacquire_enabled', d.edge_reacquire_enabled);
    syncBtn('collision_avoidance_enabled', d.collision_avoidance_enabled);
    syncBtn('face_detection_enabled', d.face_detection_enabled);
    syncBtn('face_recognition_enabled', d.face_recognition_enabled);
    syncBtn('auto_reference_capture_enabled', d.auto_reference_capture_enabled);
    syncBtn('pose_enabled', d.pose_enabled);
    syncBtn('pose_filter_persons', d.pose_filter_persons);

    var blocked = !!d.collision_blocked;
    var badge = q('collision-badge');
    if(badge){{
      badge.textContent = blocked ? 'BLOCKED' : 'CLEAR';
      badge.className = 'badge ' + (blocked ? 'blocked' : 'clear');
    }}

    if(d.behavior){{
      var st = (d.behavior.state||'IDLE').toLowerCase();
      var bhv = q('bhv-badge');
      if(bhv){{
        bhv.textContent = d.behavior.state || 'IDLE';
        bhv.className = 'bhv bhv-' + st;
      }}
      setText('bhv-time', (d.behavior.time_in_state||0).toFixed(1) + 's');
    }}
    if(d.guard){{
      var row = q('bhv-alerts-row');
      if(row) row.style.display = d.guard.enabled ? '' : 'none';
      setText('bhv-alerts', d.guard.alerts_sent || 0);
    }}
    if(d.gamepad){{
      setText('gamepad-status', d.gamepad.connected ? (d.gamepad.device || 'connected') : 'not connected');
      setText('gamepad-action', d.gamepad.last_action || '-');
    }}

    syncSld('score_threshold', d.score_threshold, 2);
    syncSld('person_score_threshold', d.person_score_threshold, 2);
    syncSld('follow_forward_speed', d.follow_forward_speed, 0);
    syncSld('follow_steer_gain', d.follow_steer_gain, 1);
    syncSld('follow_target_bbox_ratio', d.follow_target_bbox_ratio, 2);
    syncSld('close_target_bbox_ratio', d.close_target_bbox_ratio, 2);
    syncSld('follow_deadzone_ratio', d.follow_deadzone_ratio, 2);
    syncSld('max_command', d.max_command, 0);
    syncSld('min_command', d.min_command, 0);
    syncSld('max_command_step', d.max_command_step, 0);
    syncSld('command_interval_seconds', d.command_interval_seconds, 2);
    syncSld('settle_enter_degrees', d.settle_enter_degrees, 1);
    syncSld('settle_exit_degrees', d.settle_exit_degrees, 1);
    syncSld('estimated_pan_min', d.estimated_pan_min, 0);
    syncSld('estimated_pan_max', d.estimated_pan_max, 0);
    syncSld('estimated_tilt_min', d.estimated_tilt_min, 0);
    syncSld('estimated_tilt_max', d.estimated_tilt_max, 0);
    syncSld('camera_light_brightness', d.camera_light_brightness, 0);
    syncSld('camera_light_trigger_threshold', d.camera_light_trigger_threshold, 0);
    syncSld('pan_estimate_scale', d.pan_estimate_scale, 1);
    syncSld('tilt_estimate_scale', d.tilt_estimate_scale, 1);
    syncSld('pan_limit_margin', d.pan_limit_margin, 0);
    syncSld('collision_height_threshold', d.collision_height_threshold, 2);
    syncSld('collision_center_zone_fraction', d.collision_center_zone_fraction, 2);
    syncSld('auto_reference_min_confidence', d.auto_reference_min_confidence, 2);
    syncSld('pose_interval_frames', d.pose_interval_frames, 0);
    syncSld('pose_score_threshold', d.pose_score_threshold, 2);
    syncSld('jpeg_quality', d.jpeg_quality, 0);
    setText('ambient-light-level', d.ambient_light_level != null ? Math.round(d.ambient_light_level) : '-');

    syncText('inp-identity', d.target_identity);

    setText('tgt-state', d.target_state || 'none');
    setText('tgt-identity', (d.target && d.target.identity) || '-');
    setText('tgt-id', d.target_id != null ? d.target_id : '-');
    var iq = d.identity_prompt_queue || {{}};
    setText('face-queue', (iq.unknown_face_count || 0) + '/' + (iq.visible_face_count || 0));
    setText('face-asking', iq.active_index ? (iq.active_index + ' of ' + iq.unknown_face_count) : '-');
    if(d.identity_prompt && d.identity_prompt.id && d.identity_prompt.id !== lastIdentityPromptId){{
      lastIdentityPromptId = d.identity_prompt.id;
      var promptText = (d.identity_prompt.prompt || '').trim();
      if(promptText) appendMsg('bot', promptText);
    }}

    var dets = d.detections || [];
    setText('det-count', '(' + dets.length + ')');
    var detList = q('det-list');
    if(detList){{
      var detHtml = dets.length ? dets.map(function(det){{
        var isT = det.id === d.target_id;
        var lbl = (isT ? '▶ ' : '') + det.label + (det.identity ? ' · ' + det.identity : '');
        var conf = Math.round(det.confidence * 100) + '%';
        return '<div class="det-item"><span class="det-lbl' + (isT ? ' tgt' : '') + '">' + lbl +
               '</span><span class="det-conf">' + conf + '</span></div>';
      }}).join('') : '<span class="none">none</span>';
      if(detHtml !== lastDetHtml){{
        lastDetHtml = detHtml;
        detList.innerHTML = detHtml;
      }}
    }}

    schedulePoll(nextDelay);
  }}).catch(function(){{
    q('dot').className = '';
    q('hdr-right').textContent = 'disconnected';
    schedulePoll(1500);
  }});
}}

schedulePoll(0);

function appendMsg(type, text, elapsed){{
  var msgs = q('chat-msgs');
  var div = document.createElement('div');
  div.className = 'chat-msg ' + type;
  div.textContent = text;
  if(elapsed){{
    var t = document.createElement('span');
    t.className = 'chat-time';
    t.textContent = elapsed;
    div.appendChild(t);
  }}
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}}

function sendChat(){{
  var inp = q('chat-inp');
  var msg = inp.value.trim();
  if(!msg) return;
  inp.value = '';
  appendMsg('user', msg);
  var btn = q('chat-btn');
  btn.disabled = true;
  btn.textContent = '...';
  var t0 = Date.now();
  fetch('/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{message:msg}})}})
    .then(function(r){{return r.json()}})
    .then(function(d){{
      var elapsed = ((Date.now() - t0) / 1000).toFixed(1) + 's';
      btn.disabled = false;
      btn.textContent = '↵';
      if(d.ok){{
        var r = d.response || {{}};
        appendMsg('bot', r.tts || r.message || '(no response)', elapsed);
        // Pending push alerts that vault attached to this response
        // (background-job results, ingest stalls, presence-triggered
        // pushes). Vault drains its per-node queue when the user types
        // here, so each alert appears exactly once.
        (r.pending_alerts || []).forEach(function(a){{
          var label = a.event_type || a.type || 'alert';
          var text = (a.message || '').toString();
          appendMsg('alert', '[' + label + '] ' + text);
        }});
      }} else {{
        var er = d.response || {{}};
        appendMsg('err', er.message || d.error || 'error');
      }}
    }})
    .catch(function(){{
      btn.disabled = false;
      btn.textContent = '↵';
      appendMsg('err', 'request failed');
    }});
}}
</script>
</body>
</html>"""
