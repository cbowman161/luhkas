"""Portable deterministic commands for node-local audio control."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class AudioCommandConfig:
    service_url: str = os.environ.get("AUDIO_SERVICE_URL", "http://127.0.0.1:5004")
    node_id: str = os.environ.get("LUHKAS_NODE_ID", "kiosk")
    scope: str = "node_local"
    dispatch_type: str = "local_audio"

    @property
    def base_url(self) -> str:
        return self.service_url.rstrip("/")


_OUTPUT_MUTE_PATTERNS = [
    r"\b(mute|silence) (the )?(speaker|speakers|output|tts|voice|audio)\b",
    r"\b(audio|voice|speaker|speakers|tts) (mute|off)\b",
]
_OUTPUT_UNMUTE_PATTERNS = [
    r"\bunmute (the )?(speaker|speakers|output|tts|voice|audio)\b",
    r"\b(audio|voice|speaker|speakers|tts) (unmute|on)\b",
]
_MUTE_PATTERNS = [r"\b(mute|silence) (the )?(mic|microphone|input)\b", r"\bstop listening\b"]
_UNMUTE_PATTERNS = [r"\bunmute (the )?(mic|microphone|input)\b", r"\bstart listening\b"]
_SAY_PREFIX = re.compile(r"^(say|repeat|tell me)\s+", re.IGNORECASE)


def capabilities() -> list[dict]:
    cfg = AudioCommandConfig()
    examples = [
        "mute audio",
        "unmute audio",
        "say hello world",
    ]
    if cfg.node_id != "kiosk":
        examples.extend([
            "mute the mic",
            "stop listening",
            "unmute the microphone",
        ])
    return [
        {
            "name": "control_audio",
            "description": "Mute/unmute audio output, speak a phrase, or query audio status.",
            "scope": cfg.scope,
            "dispatch_type": cfg.dispatch_type,
            "owner_node": cfg.node_id,
            "target_node": cfg.node_id,
            "examples": examples,
        }
    ]


def handle(user_input: str, config: AudioCommandConfig | None = None) -> dict | None:
    cfg = config or AudioCommandConfig()
    text = (user_input or "").strip()
    if not text:
        return None
    low = text.lower()
    if any(re.search(p, low) for p in _OUTPUT_MUTE_PATTERNS):
        return _wrap("mute_output", _post(cfg, "/mute", {"muted": True}), "Audio muted. I'll switch to text.", "I could not mute audio.")
    if any(re.search(p, low) for p in _OUTPUT_UNMUTE_PATTERNS):
        return _wrap("unmute_output", _post(cfg, "/mute", {"muted": False}), "Audio unmuted.", "I could not unmute audio.")
    if any(re.search(p, low) for p in _MUTE_PATTERNS):
        if cfg.node_id == "kiosk":
            return _wrap("mic_mute_blocked", {"ok": True, "muted": False}, "The kiosk microphone stays live. Use mute audio to switch responses to text.", "")
        return _wrap("mute_mic", _post(cfg, "/listen", {"muted": True}), "Microphone muted.", "I could not mute the microphone.")
    if any(re.search(p, low) for p in _UNMUTE_PATTERNS):
        return _wrap("unmute_mic", _post(cfg, "/listen", {"muted": False}), "Microphone is listening.", "I could not unmute the microphone.")
    if _SAY_PREFIX.match(text):
        spoken = _SAY_PREFIX.sub("", text).strip().strip(".\"'")
        if not spoken:
            return None
        result = _post(cfg, "/tts", {"text": spoken})
        return _wrap("speak", result, spoken, "I could not speak that.")
    return None


def health(config: AudioCommandConfig | None = None) -> dict:
    cfg = config or AudioCommandConfig()
    info = _get(cfg, "/health")
    if info is None:
        return {"ok": False, "error": "audio_service_unreachable"}
    return {"ok": bool(info.get("ok")), **info}


def _wrap(capability: str, result: dict | None, success_msg: str, failure_msg: str) -> dict:
    ok = bool(result and result.get("ok"))
    message = success_msg if ok else failure_msg
    return {
        "ok": ok,
        "capability": capability,
        "message": message,
        "tts": message,
        "data": result or {},
    }


def _post(cfg: AudioCommandConfig, path: str, body: dict) -> dict | None:
    try:
        req = Request(
            f"{cfg.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _get(cfg: AudioCommandConfig, path: str) -> dict | None:
    try:
        with urlopen(f"{cfg.base_url}{path}", timeout=2.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
