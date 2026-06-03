from __future__ import annotations

import json

import requests

from config import (
    VAULT_ANALYST_MODEL,
    VAULT_CHAT_MODEL,
    VAULT_CODER_MODEL,
    VAULT_EMBED_MODEL,
    VAULT_FAST_CODER_MODEL,
    VAULT_BACKGROUND_KEEP_ALIVE,
    VAULT_IMMEDIATE_KEEP_ALIVE,
    VAULT_PLANNER_MODEL,
    VAULT_REASONER_MODEL,
    VAULT_ROUTER_MODEL,
    VAULT_TEACHER_MODEL,
    VAULT_VISION_MODEL,
    VAULT_WARM_MODEL_ROLES,
    OLLAMA_URL,
)


OLLAMA_GENERATE_URL = f"{OLLAMA_URL.rstrip('/')}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_URL.rstrip('/')}/api/embed"


# Last-seen monotonic-style epoch timestamp of any Ollama call dispatched
# from this process. The wiki ingest supervisor reads this (via /health)
# to decide when to pause/resume — VRAM contention only matters when an
# actual GPU model is running, so this is a strictly tighter signal than
# "user activity" (deterministic routes that never invoke Ollama don't
# need to pause ingest).
#
# Bumped at the *start* of every generate/generate_stream/embed call so
# the supervisor sees the busy state before the response lands. The bump
# is cheap (one time.time() + module-global write); no lock needed because
# concurrent writers all want the same approximate value.
import time as _time

_last_ollama_activity_at: float = 0.0


def _bump_ollama_activity() -> None:
    """Record that an Ollama call is about to start (or just started)."""
    global _last_ollama_activity_at
    _last_ollama_activity_at = _time.time()


def get_last_ollama_activity_at() -> float:
    """Epoch timestamp of the most recent Ollama dispatch, or 0.0."""
    return _last_ollama_activity_at

MODEL_ROLES = {
    "router": VAULT_ROUTER_MODEL,
    "chat": VAULT_CHAT_MODEL,
    "reasoner": VAULT_REASONER_MODEL,
    "planner": VAULT_PLANNER_MODEL,
    "analyst": VAULT_ANALYST_MODEL,
    "coder": VAULT_CODER_MODEL,
    "fast_coder": VAULT_FAST_CODER_MODEL,
    "vision": VAULT_VISION_MODEL,
    "embed": VAULT_EMBED_MODEL,
    "teacher": VAULT_TEACHER_MODEL,
}

# Vision lives here (not in BACKGROUND_ROLES) because it's in the default
# VAULT_WARM_MODEL_ROLES list. Keeping it on the short BACKGROUND keep-alive
# meant it was warmed once at boot and then evicted after 5 min idle, so the
# next vision question paid a 5-7s cold load — defeating the warm config.
# "teacher" is IMMEDIATE so it stays loaded for the duration of a
# classroom session; the controller evicts other IMMEDIATE roles before
# loading it so the 24 GB card has the headroom.
IMMEDIATE_ROLES = {"router", "chat", "embed", "vision", "teacher"}
BACKGROUND_ROLES = {"reasoner", "planner", "analyst", "coder", "fast_coder"}

ROLE_OPTIONS = {
    "router": {
        "temperature": 0.0,
        "top_p": 0.8,
        "repeat_penalty": 1.05,
        "num_ctx": 2048,
        "num_predict": 80,
    },
    "chat": {
        "temperature": 0.35,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
        # 8192 fits comfortably in VRAM once Ollama is running with
        # OLLAMA_KV_CACHE_TYPE=q8_0 (halved KV memory) and FA2 enabled.
        # Larger window = better multi-turn recall and richer fact context.
        "num_ctx": 8192,
        "num_predict": 240,
    },
    "reasoner": {
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
        "num_predict": 2048,
    },
    "planner": {
        "temperature": 0.15,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
        "num_predict": 2048,
    },
    "analyst": {
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
        "num_predict": 700,
    },
    "coder": {
        "temperature": 0.08,
        "top_p": 0.9,
        "repeat_penalty": 1.08,
        "num_predict": 1600,
    },
    "fast_coder": {
        "temperature": 0.08,
        "top_p": 0.9,
        "repeat_penalty": 1.08,
        "num_predict": 900,
    },
    "teacher": {
        # Slightly warmer than analyst so explanations breathe a little
        # without drifting off-topic. The classroom prompt enforces
        # structure (on-topic discipline + optional JSON tail), so a
        # touch of temperature is fine.
        # num_predict is generous (1800) because qwen3:30b is a
        # thinking model — even with think=True, complex per-turn
        # prompts can spend hundreds of tokens reasoning before the
        # visible response begins. Underprovision and you get a turn
        # that never closes its </think> tag.
        "temperature": 0.25,
        "top_p": 0.9,
        "repeat_penalty": 1.10,
        "num_ctx": 8192,
        "num_predict": 4000,
    },
}


class BaseModel:
    def __init__(self, model_name: str, role: str):
        self.model_name = model_name
        self.role = role

    def generate(
        self,
        prompt: str,
        *,
        response_format=None,
        options: dict | None = None,
        timeout=120,
        allow_empty=False,
        think: bool | None = None,
        images: list[str] | None = None,
    ):
        """Single-shot generate.

        ``images`` accepts a list of base64-encoded image strings for
        multimodal models (e.g. qwen2.5vl). Pass-through to Ollama's
        ``images`` field. Text-only models will ignore or error on this.
        """
        model_options = dict(ROLE_OPTIONS.get(self.role, ROLE_OPTIONS["chat"]))
        if options:
            model_options.update(options)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": model_options,
            "keep_alive": self.keep_alive,
        }
        if think is not None:
            payload["think"] = think
        if response_format is not None:
            payload["format"] = response_format
        if images:
            payload["images"] = images
        _bump_ollama_activity()
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        data = response.json()
        text = (data.get("response") or "").strip()
        if not text and not allow_empty:
            done_reason = data.get("done_reason")
            thinking = str(data.get("thinking") or "")
            thinking_preview = thinking[:240].replace("\n", " ")
            detail = f"done_reason={done_reason}"
            if thinking_preview:
                detail += f", thinking_preview={thinking_preview!r}"
            raise RuntimeError(
                f"Ollama returned an empty response for {self.model_name} ({detail})"
            )
        return text

    def generate_stream(
        self,
        prompt: str,
        *,
        options: dict | None = None,
        timeout=120,
        think: bool | None = None,
    ):
        """Yield text deltas as Ollama produces them.

        Uses Ollama's streaming API (``stream: True``). Each yielded value is
        the latest ``response`` token chunk. Iteration ends on ``done: True``
        or a closed connection. Raises on non-200; otherwise empty deltas are
        skipped silently.
        """
        model_options = dict(ROLE_OPTIONS.get(self.role, ROLE_OPTIONS["chat"]))
        if options:
            model_options.update(options)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": model_options,
            "keep_alive": self.keep_alive,
        }
        if think is not None:
            payload["think"] = think
        _bump_ollama_activity()
        with requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout, stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")
            for line in response.iter_lines(decode_unicode=False):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                chunk = data.get("response") or ""
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    @property
    def keep_alive(self):
        if self.role in IMMEDIATE_ROLES:
            return VAULT_IMMEDIATE_KEEP_ALIVE
        if self.role in BACKGROUND_ROLES:
            return VAULT_BACKGROUND_KEEP_ALIVE
        return VAULT_BACKGROUND_KEEP_ALIVE


class EmbeddingModel:
    # The "embed" role is in IMMEDIATE_ROLES (warm), but without sending
    # keep_alive in each request Ollama falls back to its 5-minute default
    # and evicts bge-m3 between memory queries — slow cold-load on every
    # gap. Pin it to the immediate keep-alive value just like BaseModel.
    role = "embed"

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def keep_alive(self):
        return VAULT_IMMEDIATE_KEEP_ALIVE

    def embed(self, text: str | list[str], timeout=120):
        _bump_ollama_activity()
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": self.model_name,
                "input": text,
                "keep_alive": self.keep_alive,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama embed error: {response.text}")
        return response.json().get("embeddings", [])


def get_model(role: str):
    if role not in MODEL_ROLES:
        raise ValueError(f"Unknown model role: {role}")
    if role == "embed":
        return EmbeddingModel(MODEL_ROLES[role])
    return BaseModel(MODEL_ROLES[role], role)


def model_manifest():
    return dict(MODEL_ROLES)


def evict_model(role_or_name: str) -> bool:
    """Tell Ollama to immediately unload a model from VRAM. Accepts either
    a role name ("router", "chat", ...) or a raw Ollama model name.
    Returns True if Ollama accepted the eviction request.

    Used by the classroom controller to free VRAM before loading the
    much larger teacher model — sending keep_alive=0 to a loaded model
    causes Ollama to unload it immediately rather than waiting for the
    keep-alive timer.
    """
    model_name = MODEL_ROLES.get(role_or_name, role_or_name)
    if not model_name:
        return False
    try:
        _bump_ollama_activity()
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def warm_model_role(role: str) -> bool:
    """Pre-load a single model into VRAM by running a tiny generate.
    Returns True on success. Used to restore the default warm set after
    a classroom session ends."""
    try:
        model = get_model(role)
        if not isinstance(model, BaseModel):
            return False
        model.generate(
            "Reply with OK.",
            options={"num_predict": 2, "temperature": 0},
            timeout=120,
            allow_empty=True,
        )
        return True
    except Exception:
        return False


def warm_models(roles: list[str] | None = None):
    roles = roles or [
        role.strip()
        for role in VAULT_WARM_MODEL_ROLES.split(",")
        if role.strip()
    ]
    results = []
    for role in roles:
        if role == "embed":
            continue
        try:
            model = get_model(role)
            if not isinstance(model, BaseModel):
                continue
            model.generate(
                "Reply with OK.",
                options={"num_predict": 2, "temperature": 0},
                timeout=120,
                allow_empty=True,
            )
            results.append({
                "role": role,
                "model": model.model_name,
                "ok": True,
                "keep_alive": model.keep_alive,
            })
        except Exception as exc:
            results.append({
                "role": role,
                "model": MODEL_ROLES.get(role),
                "ok": False,
                "error": str(exc),
            })
    return results
