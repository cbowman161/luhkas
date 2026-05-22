"""
NodeRegistry — tracks interaction nodes, display capabilities, network addresses,
activity, and pending alert queues.

Static config: data/nodes.json
Live registrations: data/nodes_registered.json (persisted across restarts)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import DATA_DIR


NODES_FILE = DATA_DIR / "nodes.json"
REGISTERED_FILE = DATA_DIR / "nodes_registered.json"
VAULT_NODE_ID = "vault"
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
        self._alerts: dict = {}      # node_id -> list[dict]  (pending push alerts)
        self._load()
        self._load_registered()
        self._sanitize_registered()

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

    # ── registration ────────────────────────────────────────────────────────

    def register(self, node_id: str, display: dict, node_name: str = "",
                 ip: str = "", services: dict | None = None,
                 capabilities: dict | None = None, modules: dict | None = None) -> None:
        """Record a live node's display capabilities and network address."""
        node_id = str(node_id or "").strip()
        if not node_id or self._is_synthetic_node_id(node_id):
            return
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        modules = modules if isinstance(modules, dict) else capabilities.get("module_status")
        if not isinstance(modules, dict):
            modules = {}
        self._registered[node_id] = {
            "display": display,
            "node_name": node_name or node_id,
            "ip": ip,
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
        ip = reg.get("ip", "")
        port = reg.get("services", {}).get(service)
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

    def queue_alert(self, node_id: str, alert: dict) -> None:
        """Queue an alert for delivery to a node on its next poll."""
        self._alerts.setdefault(node_id, []).append(alert)

    def pop_alerts(self, node_id: str) -> list[dict]:
        """Return and clear pending alerts for a node."""
        return self._alerts.pop(node_id, [])

    # ── display capability queries ───────────────────────────────────────────

    def registered_nodes(self) -> dict:
        return dict(self._registered)

    def get(self, node_id: str) -> dict:
        return self._nodes.get(node_id) or {"name": node_id, "has_display": False}

    def has_display(self, node_id: str) -> bool:
        return bool(self.get(node_id).get("has_display", False))

    def display_caps(self, node_id: str) -> dict:
        return self.get(node_id).get("display") or {}

    def can_show_code(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_show_code", False))

    def can_open_browser(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_open_browser", False))

    def can_show_images(self, node_id: str) -> bool:
        return bool(self.display_caps(node_id).get("can_show_images", False))
