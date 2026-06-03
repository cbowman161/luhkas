"""
NodeRegistry — tracks interaction nodes, display capabilities, network addresses,
activity, and pending alert queues.

Static config: data/nodes.json
Live registrations: data/nodes_registered.json (persisted across restarts)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from config import DATA_DIR


NODES_FILE = DATA_DIR / "nodes.json"
REGISTERED_FILE = DATA_DIR / "nodes_registered.json"
PENDING_ALERTS_FILE = DATA_DIR / "nodes_pending_alerts.json"
VAULT_NODE_ID = "vault"
# Tight "currently active" window — alerts fire only to nodes where a user
# is interacting RIGHT NOW (face-detected with adopted identity, or a
# /presence/message arrived within this many seconds). The wider 5-min
# `find_alert_targets` window is preserved for legacy callers but the
# enqueue-for-active-user routing uses this tighter window.
CURRENTLY_ACTIVE_WINDOW_S = 30
_SYNTHETIC_NODE_ID_PREFIXES = ("batch", "final", "postfix", "temp", "test")

_DEFAULTS: dict = {
    "vault": {
        "name": "Vault",
        "has_display": True,
        "display": {
            "type": "vault_runtime",
            "can_show_code": True,
            "can_open_browser": True,
            "can_show_images": True,
        },
    },
    "cli": {
        "name": "CLI Terminal",
        "has_display": True,
        "display": {
            "type": "terminal",
            "can_show_code": True,
            "can_open_browser": True,
            "can_show_images": False,
        },
    },
    "scout": {
        "name": "Scout Edge Node",
        "has_display": False,
    },
}


class NodeRegistry:
    def __init__(self):
        self._nodes: dict = dict(_DEFAULTS)
        self._registered: dict = {}  # node_id -> {ip, services, display, node_name, ...}
        self._activity: dict = {}    # node_id -> {last_active_at, last_identity_seen, person_count}
        self._alerts: dict = {}      # node_id -> list[dict]  (per-node push queue)
        # Alerts with no currently-active target wait here until ANY node
        # gets a user-present signal (face-adopt or fresh /presence/message),
        # at which point they're flushed to that node's queue. Persisted
        # to disk so vault-restart and cross-process writes (watchdog via
        # HTTP) share a consistent view.
        self._pending_alerts: list[dict] = []
        self._alerts_lock = threading.Lock()
        self._load()
        self._load_registered()
        self._sanitize_registered()
        self._load_pending_alerts()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self):
        if NODES_FILE.exists():
            try:
                data = json.loads(NODES_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for node_id, cfg in data.items():
                        if node_id in self._nodes and isinstance(cfg, dict):
                            self._nodes[node_id] = {**self._nodes[node_id], **cfg}
                        else:
                            self._nodes[node_id] = cfg
            except Exception:
                pass

    def _load_registered(self):
        if REGISTERED_FILE.exists():
            try:
                data = json.loads(REGISTERED_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._registered = data
            except Exception:
                pass
        self._registered.setdefault(VAULT_NODE_ID, self._vault_registration())

    def _vault_registration(self) -> dict:
        return {
            "display": dict(_DEFAULTS[VAULT_NODE_ID]["display"]),
            "node_name": "Vault",
            "ip": "",
            "services": {"runtime": 7000},
            "capabilities": {},
            "modules": {},
            "registered_at": None,
            "intrinsic": True,
        }

    def _is_synthetic_node_id(self, node_id: str) -> bool:
        lowered = str(node_id or "").strip().lower()
        return lowered.startswith(_SYNTHETIC_NODE_ID_PREFIXES)

    def _sanitize_registered(self):
        cleaned = {}
        changed = False
        for node_id, cfg in self._registered.items():
            clean_id = str(node_id or "").strip()
            if not clean_id or self._is_synthetic_node_id(clean_id):
                changed = True
                continue
            if not isinstance(cfg, dict):
                changed = True
                continue
            # Prune entries that look like ad-hoc /ui callers rather than real
            # registered nodes. A real /node/register call always supplies at
            # least one of ip / services / modules. Intrinsic entries (e.g.
            # vault) are preserved regardless.
            if not cfg.get("intrinsic"):
                has_substance = bool(
                    cfg.get("ip")
                    or cfg.get("services")
                    or cfg.get("modules")
                    or cfg.get("capabilities")
                )
                if not has_substance:
                    changed = True
                    continue
            cleaned[clean_id] = cfg
        if VAULT_NODE_ID not in cleaned:
            cleaned[VAULT_NODE_ID] = self._vault_registration()
            changed = True
        if cleaned.get(VAULT_NODE_ID, {}).get("intrinsic") is not True:
            cleaned[VAULT_NODE_ID] = {**self._vault_registration(), **cleaned.get(VAULT_NODE_ID, {}), "intrinsic": True}
            changed = True
        if changed or cleaned != self._registered:
            self._registered = cleaned
            self._save_registered()

    def _save_registered(self):
        try:
            REGISTERED_FILE.write_text(
                json.dumps(self._registered, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_pending_alerts(self) -> None:
        if not PENDING_ALERTS_FILE.exists():
            return
        try:
            data = json.loads(PENDING_ALERTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._pending_alerts = [a for a in data if isinstance(a, dict)]
        except Exception:
            self._pending_alerts = []

    def _save_pending_alerts(self) -> None:
        try:
            PENDING_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PENDING_ALERTS_FILE.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._pending_alerts, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(PENDING_ALERTS_FILE)
        except Exception:
            pass

    # ── registration ────────────────────────────────────────────────────────

    def register(self, node_id: str, display: dict, node_name: str = "",
                 ip: str = "", services: dict | None = None,
                 capabilities: dict | None = None, modules: dict | None = None,
                 network: dict | None = None) -> None:
        """Record a live node's display capabilities and network address."""
        node_id = str(node_id or "").strip()
        if not node_id or self._is_synthetic_node_id(node_id):
            return
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        modules = modules if isinstance(modules, dict) else capabilities.get("module_status")
        if not isinstance(modules, dict):
            modules = {}
        network = network if isinstance(network, dict) else {}
        tailscale_ip = str(network.get("tailscale_ip") or "").strip()
        lan_ip = str(network.get("lan_ip") or "").strip()
        preferred_network = str(network.get("preferred") or "").strip()
        self._registered[node_id] = {
            "display": display,
            "node_name": node_name or node_id,
            "ip": ip,
            "network": {
                "lan_ip": lan_ip,
                "tailscale_ip": tailscale_ip,
                "preferred": preferred_network or ("tailscale" if tailscale_ip and ip == tailscale_ip else "lan"),
            },
            "services": services or {},
            "capabilities": capabilities,
            "modules": modules,
            "registered_at": time.time(),
        }
        existing = self._nodes.get(node_id, {})
        self._nodes[node_id] = {
            **existing,
            "has_display": bool(display.get("has_display")),
            "display": display,
            "capabilities": capabilities,
            "modules": modules,
        }
        self._save_registered()

    def update_capabilities(self, node_id: str, capabilities: dict | None = None,
                            modules: dict | None = None) -> None:
        """Update capabilities/modules for an already-registered node.

        Does NOT auto-create a registry entry: only nodes that have explicitly
        registered via /node/register can have their capabilities updated here.
        This prevents arbitrary ad-hoc node_ids (e.g. one-shot CLI/test calls
        through /ui) from polluting the registry.
        """
        node_id = str(node_id or "").strip()
        if not node_id or self._is_synthetic_node_id(node_id):
            return
        reg = self._registered.get(node_id)
        if not isinstance(reg, dict):
            return
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        modules = modules if isinstance(modules, dict) else capabilities.get("module_status")
        if not isinstance(modules, dict):
            modules = {}
        if capabilities:
            reg["capabilities"] = capabilities
        if modules:
            reg["modules"] = modules
        existing = self._nodes.get(node_id, {})
        self._nodes[node_id] = {
            **existing,
            "capabilities": reg.get("capabilities", {}),
            "modules": reg.get("modules", {}),
        }
        self._save_registered()

    # ── activity tracking ────────────────────────────────────────────────────

    def update_activity(self, node_id: str, identity: str | None = None,
                        person_count: int = 0) -> None:
        """Called on every incoming message/presence event to track node state."""
        act = self._activity.setdefault(node_id, {})
        act["last_active_at"] = time.time()
        act["person_count"] = person_count
        if identity:
            act["last_identity_seen"] = identity

    def update_health_results(self, results: dict[str, dict]) -> None:
        """Store the latest vault-initiated health check results."""
        for node_id, result in results.items():
            act = self._activity.setdefault(node_id, {})
            act["last_health_check"] = {"ok": result.get("ok"), "checked_at": time.time(), **result}

    def record_selftest(self, node_id: str, report: dict) -> None:
        """Store the latest self-test report pushed from a node."""
        act = self._activity.setdefault(node_id, {})
        act["last_selftest"] = {"reported_at": time.time(), **report}

    def health_summary(self) -> dict:
        """Return a summary of node health: vault-ping results and self-test reports."""
        summary = {}
        for node_id, act in self._activity.items():
            summary[node_id] = {
                "last_active_at": act.get("last_active_at"),
                "last_identity_seen": act.get("last_identity_seen"),
                "person_count": act.get("person_count", 0),
                "health_check": act.get("last_health_check"),
                "selftest": act.get("last_selftest"),
            }
        return {"ok": True, "nodes": summary}

    # ── network helpers ───────────────────────────────────────────────────────

    def node_url(self, node_id: str, service: str = "presence") -> str | None:
        """Return the HTTP base URL for a given service on a registered node."""
        reg = self._registered.get(node_id, {})
        network = reg.get("network") if isinstance(reg.get("network"), dict) else {}
        ip = network.get("tailscale_ip") or reg.get("ip", "")
        service_cfg = reg.get("services", {}).get(service)
        port = service_cfg.get("port") if isinstance(service_cfg, dict) else service_cfg
        if ip and port:
            return f"http://{ip}:{port}"
        return None

    # ── alert routing ─────────────────────────────────────────────────────────

    def find_alert_targets(self, primary_user: str | None = None) -> list[str]:
        """Return node_ids that should receive a critical alert, in priority order.

        Priority:
          1. Node where primary_user was most recently identified (within 5 min)
          2. All nodes with people currently detected (highest count first)
          3. Most recently active node (fallback)
        """
        now = time.time()
        recent = now - 300  # 5 minutes

        if primary_user:
            pu = primary_user.lower()
            candidates = [
                (nid, act) for nid, act in self._activity.items()
                if (act.get("last_identity_seen") or "").lower() == pu
                and act.get("last_active_at", 0) > recent
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1].get("last_active_at", 0), reverse=True)
                return [candidates[0][0]]

        nodes_with_people = [
            (nid, act) for nid, act in self._activity.items()
            if act.get("person_count", 0) > 0 and act.get("last_active_at", 0) > recent
        ]
        if nodes_with_people:
            nodes_with_people.sort(key=lambda x: x[1].get("person_count", 0), reverse=True)
            return [n[0] for n in nodes_with_people]

        if self._activity:
            most_recent = max(
                self._activity.items(), key=lambda x: x[1].get("last_active_at", 0)
            )
            return [most_recent[0]]

        return ["cli"]

    def currently_active_node_ids(
        self, window_s: int = CURRENTLY_ACTIVE_WINDOW_S
    ) -> list[str]:
        """Return node_ids where a user is CURRENTLY interacting — narrower
        than `find_alert_targets`'s 5-min window.

        A node counts as currently active if EITHER:
          - its last_active_at is within ``window_s`` seconds (user is
            actively typing / sending presence messages), OR
          - it has an adopted identity that was seen within ``window_s``
            (face/voice recognition just confirmed a user is present).

        Used by ``enqueue_for_active_user`` to decide whether to deliver
        an alert immediately vs defer it until presence is detected."""
        now = time.time()
        cutoff = now - max(1, window_s)
        out: list[str] = []
        for node_id, act in self._activity.items():
            last = act.get("last_active_at") or 0
            if last >= cutoff:
                # Identity-adoption also bumps last_active_at via
                # update_activity, so this branch covers both signals.
                out.append(node_id)
        return out

    def has_audio_module(self, node_id: str) -> bool:
        """True iff the node has declared an audio output module (so vault
        can route TTS-bound responses to it). Checked against the
        registered `modules` dict — the same surface Scout uses to declare
        camera_node / rover_node / etc."""
        reg = self._registered.get(node_id) or {}
        modules = reg.get("modules") or {}
        for key in ("audio_node", "audio_out", "tts_node", "speaker_node"):
            if modules.get(key):
                return True
        # Fall back to a display.has_audio_out flag if a future node
        # declares audio via display caps rather than a module.
        display = reg.get("display") or {}
        return bool(display.get("has_audio_out"))

    def enqueue_for_active_user(self, alert: dict) -> dict:
        """Deliver immediately to currently-active nodes, OR defer to the
        pending queue until one appears.

        Returns a small dict describing what happened so the caller (and
        observability tooling) can see whether the alert fired right away
        or is sitting in the pending queue."""
        if not isinstance(alert, dict):
            return {"ok": False, "error": "alert must be a dict"}
        # Stamp a queued_at marker so deferred alerts can be displayed
        # with "stalled N hours ago" wording on eventual delivery.
        alert = dict(alert)
        alert.setdefault("queued_at", time.time())
        with self._alerts_lock:
            active = self.currently_active_node_ids()
            if active:
                for node_id in active:
                    self._alerts.setdefault(node_id, []).append(dict(alert))
                return {"ok": True, "delivered_to": list(active), "deferred": False}
            self._pending_alerts.append(alert)
            self._save_pending_alerts()
            return {"ok": True, "delivered_to": [], "deferred": True,
                    "pending_count": len(self._pending_alerts)}

    def flush_pending_to(self, node_id: str) -> int:
        """Move every queued-while-idle alert into this node's per-node
        queue. Called when a user-present signal arrives at the node
        (face adopted or fresh /presence/message)."""
        node_id = (node_id or "").strip()
        if not node_id:
            return 0
        with self._alerts_lock:
            if not self._pending_alerts:
                return 0
            drained = list(self._pending_alerts)
            self._pending_alerts.clear()
            self._save_pending_alerts()
            self._alerts.setdefault(node_id, []).extend(drained)
        return len(drained)

    def pending_alerts_snapshot(self) -> list[dict]:
        """Read-only view of currently-deferred alerts. Useful for
        debugging / 'any updates' style commands."""
        with self._alerts_lock:
            return [dict(a) for a in self._pending_alerts]

    def queue_alert(self, node_id: str, alert: dict) -> None:
        """Queue an alert for delivery to a node on its next poll."""
        with self._alerts_lock:
            self._alerts.setdefault(node_id, []).append(alert)

    def pop_alerts(self, node_id: str) -> list[dict]:
        """Return and clear pending alerts for a node."""
        with self._alerts_lock:
            return self._alerts.pop(node_id, [])

    # ── display capability queries ───────────────────────────────────────────

    def registered_nodes(self) -> dict:
        return dict(self._registered)

    def get(self, node_id: str) -> dict:
        return self._nodes.get(node_id) or {"name": node_id, "has_display": False}

    def has_display(self, node_id: str) -> bool:
        return bool(self.get(node_id).get("has_display", False))

    def has_audio(self, node_id: str) -> bool:
        """Mirror of has_display for audio output. True iff the node has
        an audio-output module declared in its registration. Vault uses
        this to decide whether `response['tts']` is meaningful for the
        node — `response['message']` is always preserved regardless."""
        return self.has_audio_module(node_id)

    def display_caps(self, node_id: str) -> dict:
        return self.get(node_id).get("display") or {}

    def can_show_code(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_show_code", False))

    def can_open_browser(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_open_browser", False))

    def can_show_images(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_show_images", False))
