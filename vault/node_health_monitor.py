"""Background vault-side health monitor for all registered edge nodes.

Periodically pings each registered node's presence /health endpoint.
On status change or periodic cadence, queues alerts via NodeRegistry
and writes to EventLog for the notifications table.

Alert logic:
- Node newly fails → queue warning alert to all active nodes + EventLog
- Node recovers → queue info alert + EventLog
- All nodes OK every HEALTHY_CADENCE checks → queue "all stable" info alert
"""
from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen


_STARTUP_DELAY_SECONDS = 45.0
_HEALTHY_NOTIFY_EVERY = 10


class NodeHealthMonitor:
    def __init__(
        self,
        node_registry,
        event_log,
        interval_seconds: float = 60.0,
    ) -> None:
        self._registry = node_registry
        self._event_log = event_log
        self._interval = interval_seconds
        self._last_status: dict[str, bool] = {}
        self._check_count = 0

    def start(self) -> None:
        threading.Thread(
            target=self._loop,
            daemon=True,
            name="node-health-monitor",
        ).start()

    def _loop(self) -> None:
        time.sleep(_STARTUP_DELAY_SECONDS)
        while True:
            try:
                self._check_all()
            except Exception:
                pass
            time.sleep(self._interval)

    def _check_all(self) -> None:
        self._check_count += 1
        registered = self._registry.registered_nodes()
        if not registered:
            return

        results: dict[str, dict] = {}
        for node_id, reg in registered.items():
            if reg.get("intrinsic"):
                continue
            ip = reg.get("ip", "")
            port = (reg.get("services") or {}).get("presence")
            if not ip or not port:
                results[node_id] = {"ok": False, "error": "no_address_registered"}
                continue
            url = f"http://{ip}:{port}/health"
            try:
                with urlopen(url, timeout=5) as r:
                    data = json.loads(r.read().decode())
                results[node_id] = {"ok": bool(data.get("ok")), "response": data}
            except Exception as exc:
                results[node_id] = {"ok": False, "error": str(exc)}

        self._registry.update_health_results(results)
        self._dispatch_alerts(results)

    def _dispatch_alerts(self, results: dict[str, dict]) -> None:
        all_ok = all(v["ok"] for v in results.values())
        active_nodes = [nid for nid, v in results.items() if v["ok"]]
        failing = {nid: v for nid, v in results.items() if not v["ok"]}

        newly_failing = {
            nid for nid in results
            if not results[nid]["ok"] and self._last_status.get(nid, True)
        }
        newly_recovered = {
            nid for nid in results
            if results[nid]["ok"] and self._last_status.get(nid) is False
        }

        for node_id in newly_failing:
            error = results[node_id].get("error", "unknown")
            msg = f"Node '{node_id}' is not responding: {error}"
            self._event_log.notify("node_health", "warning", msg, results[node_id])
            alert = {
                "type": "node_health",
                "severity": "warning",
                "message": msg,
                "node_id": node_id,
                "error": error,
            }
            for active in active_nodes:
                self._registry.queue_alert(active, alert)

        for node_id in newly_recovered:
            msg = f"Node '{node_id}' has recovered and is now active."
            self._event_log.notify("node_health", "info", msg, results[node_id])
            alert = {
                "type": "node_health",
                "severity": "info",
                "message": msg,
                "node_id": node_id,
            }
            for active in active_nodes:
                self._registry.queue_alert(active, alert)

        emit_healthy = all_ok and (
            newly_recovered
            or self._check_count % _HEALTHY_NOTIFY_EVERY == 1
        )
        if emit_healthy:
            msg = "All nodes and vault systems active and stable."
            self._event_log.notify("node_health", "info", msg, {"nodes": active_nodes})
            alert = {
                "type": "node_health",
                "severity": "info",
                "message": msg,
                "nodes": active_nodes,
            }
            for node_id in active_nodes:
                self._registry.queue_alert(node_id, alert)

        self._last_status = {nid: v["ok"] for nid, v in results.items()}
