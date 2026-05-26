"""Deterministic commands for display_node.

Most display interaction is via touch on the SPA; the natural-language
handler is mainly a status/health probe for selftest.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import urlopen


@dataclass
class DisplayCommandConfig:
    service_url: str = os.environ.get("DISPLAY_SERVICE_URL", "http://127.0.0.1:5005")
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "kiosk")
    scope: str = "node_local"
    dispatch_type: str = "local_display"

    @property
    def base_url(self) -> str:
        return self.service_url.rstrip("/")


def capabilities() -> list[dict]:
    cfg = DisplayCommandConfig()
    return [
        {
            "name": "display_status",
            "description": "Report on-screen UI server status.",
            "scope": cfg.scope,
            "dispatch_type": cfg.dispatch_type,
            "owner_node": cfg.node_id,
            "target_node": cfg.node_id,
            "examples": [],
        }
    ]


def handle(user_input: str, config: DisplayCommandConfig | None = None) -> dict | None:
    # No natural-language dispatch — display interaction is via touch UI.
    return None


def health(config: DisplayCommandConfig | None = None) -> dict:
    cfg = config or DisplayCommandConfig()
    try:
        with urlopen(f"{cfg.base_url}/health", timeout=2.0) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return {"ok": bool(payload.get("ok")), **payload}
    except Exception:
        return {"ok": False, "error": "display_service_unreachable"}
