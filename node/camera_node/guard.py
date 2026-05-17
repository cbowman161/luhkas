"""Camera guard-mode alert dispatch."""
from __future__ import annotations

import base64
import datetime

from scout.config import GuardConfig


def dispatch_guard_alert(identity: str | None, confidence: float, jpeg: bytes | None, cfg_guard: GuardConfig) -> None:
    import requests as _requests

    payload = {
        "type": "guard",
        "node_id": "scout",
        "identity": identity or "unknown",
        "confidence": round(confidence, 3),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if cfg_guard.snapshot_on_alert and jpeg:
        payload["snapshot"] = base64.b64encode(jpeg).decode()
    try:
        _requests.post(cfg_guard.alert_url, json=payload, timeout=3)
    except Exception:
        pass
