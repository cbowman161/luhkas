'use strict';

const POLL_MS = 500;

const els = {
  nodeId: document.getElementById('nodeId'),
  battery: document.getElementById('batteryBadge'),
  audio: document.getElementById('audioBadge'),
  camera: document.getElementById('cameraBadge'),
  clock: document.getElementById('clock'),
  userText: document.getElementById('userText'),
  userMeta: document.getElementById('userMeta'),
  assistantText: document.getElementById('assistantText'),
  assistantMeta: document.getElementById('assistantMeta'),
  history: document.getElementById('history'),
  muteBtn: document.getElementById('muteBtn'),
};

let lastUserTs = 0;
let lastAsstTs = 0;
let muted = false;

function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function setBadge(el, text, kind) {
  el.textContent = text;
  el.classList.remove('ok', 'warn', 'danger');
  if (kind) el.classList.add(kind);
}

function batteryKind(percent) {
  if (percent == null) return null;
  if (percent < 15) return 'danger';
  if (percent < 30) return 'warn';
  return 'ok';
}

function renderHistory(history) {
  els.history.innerHTML = '';
  for (const item of history.slice(-30)) {
    if (item.type !== 'user_message' && item.type !== 'assistant_message') continue;
    const row = document.createElement('div');
    row.className = 'entry';
    const role = document.createElement('div');
    role.className = 'role';
    role.textContent = item.type === 'user_message' ? 'You' : 'LUHKAS';
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = item.text || '';
    row.appendChild(role);
    row.appendChild(body);
    els.history.appendChild(row);
  }
  els.history.scrollTop = els.history.scrollHeight;
}

async function poll() {
  try {
    const r = await fetch('/ui/state', { cache: 'no-store' });
    if (!r.ok) return;
    const state = await r.json();
    const status = state.status || {};

    els.nodeId.textContent = status.node_id || '—';

    if (status.battery && typeof status.battery.percent === 'number') {
      setBadge(els.battery, `${status.battery.percent}%`, batteryKind(status.battery.percent));
    } else {
      setBadge(els.battery, '— %', null);
    }

    if (status.audio) {
      const a = status.audio;
      setBadge(els.audio, a.muted ? 'mic muted' : (a.listening ? 'listening' : 'idle'),
        a.muted ? 'warn' : (a.listening ? 'ok' : null));
    }

    if (status.camera) {
      setBadge(els.camera, status.camera.active ? 'cam on' : 'cam off',
        status.camera.active ? 'ok' : null);
    }

    muted = !!status.muted;
    els.muteBtn.textContent = muted ? 'Unmute mic' : 'Mute mic';
    els.muteBtn.classList.toggle('muted', muted);

    const u = state.last_user_message;
    if (u && u.timestamp !== lastUserTs) {
      els.userText.textContent = u.text || '';
      els.userMeta.textContent = fmtTime(u.timestamp);
      lastUserTs = u.timestamp;
    }
    const a = state.last_assistant_message;
    if (a && a.timestamp !== lastAsstTs) {
      els.assistantText.textContent = a.text || '';
      els.assistantMeta.textContent = fmtTime(a.timestamp);
      lastAsstTs = a.timestamp;
    }

    renderHistory(state.history || []);
  } catch (err) {
    // swallow; next tick retries
  }
}

function tickClock() {
  const d = new Date();
  els.clock.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

els.muteBtn.addEventListener('click', async () => {
  const next = !muted;
  try {
    await fetch('/ui/mute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ muted: next }),
    });
  } catch (_) { /* ignore */ }
});

setInterval(poll, POLL_MS);
setInterval(tickClock, 5000);
tickClock();
poll();
