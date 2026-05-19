from __future__ import annotations

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
    VAULT_VISION_MODEL,
    VAULT_WARM_MODEL_ROLES,
    OLLAMA_URL,
)


OLLAMA_GENERATE_URL = f"{OLLAMA_URL.rstrip('/')}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_URL.rstrip('/')}/api/embed"

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
}

IMMEDIATE_ROLES = {"router", "chat", "embed"}
BACKGROUND_ROLES = {"reasoner", "planner", "analyst", "coder", "fast_coder", "vision"}

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
        "num_ctx": 4096,
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
    ):
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

    @property
    def keep_alive(self):
        if self.role in IMMEDIATE_ROLES:
            return VAULT_IMMEDIATE_KEEP_ALIVE
        if self.role in BACKGROUND_ROLES:
            return VAULT_BACKGROUND_KEEP_ALIVE
        return VAULT_BACKGROUND_KEEP_ALIVE


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, text: str | list[str], timeout=120):
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": self.model_name, "input": text},
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
