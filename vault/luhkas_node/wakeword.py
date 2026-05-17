"""Shared LUHKAS wakeword handling for node runtimes."""
from __future__ import annotations

import re


WAKEWORD_RESPONSE = "Yes? What can I do for you?"

_OPENERS = {"hey", "hi", "hello", "yo", "ok", "okay"}
_WAKEWORD_VARIANTS = {
    "luhkas",
    "luhkus",
    "lukas",
    "lucas",
    "loukas",
    "loucas",
    "leukas",
    "lukus",
}


def is_wakeword_only(message: str) -> bool:
    text = re.sub(r"[^a-zA-Z\s]", " ", str(message or "").casefold())
    words = [word for word in text.split() if word not in _OPENERS]
    if not words:
        return False
    return all(sounds_like_luhkas(word) for word in words)


def sounds_like_luhkas(word: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", str(word or "").casefold())
    return normalized in _WAKEWORD_VARIANTS


def response() -> dict:
    return {
        "ok": True,
        "mode": "wakeword",
        "message": WAKEWORD_RESPONSE,
        "tts": WAKEWORD_RESPONSE,
    }
