from __future__ import annotations

import json
import re
import time
from pathlib import Path


STYLE_DEFAULTS = {
    "sarcasm": 0.55,
    "rudeness": 0.20,
    "warmth": 0.45,
    "brevity": 0.75,
    "formality": 0.20,
    "playfulness": 0.50,
    "directness": 0.70,
}

MOOD_BASELINE = {
    "valence": 0.08,
    "arousal": 0.32,
    "patience": 0.80,
    "playfulness": 0.50,
    "social_energy": 0.65,
    "irritation": 0.05,
}


def _clamp(value, lo=0.0, hi=1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = lo
    return max(lo, min(hi, value))


def _load_json(path: Path, fallback):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, type(fallback)) else fallback
    except Exception:
        return fallback


class MoodEngine:
    """Deterministic personality-adjacent state.

    Personality is the stable constitution. Style state is user-tuned response
    behavior. Mood is short-lived runtime weather that decays toward baseline.
    """

    def __init__(self, self_dir: Path):
        self.self_dir = Path(self_dir)
        self.mood_path = self.self_dir / "mood.json"
        self.style_path = self.self_dir / "style_state.json"
        self.self_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_files()

    def ensure_files(self) -> None:
        if not self.mood_path.exists():
            self.write_mood(self.default_mood())
        if not self.style_path.exists():
            self.write_style_state(self.default_style_state())

    def default_mood(self) -> dict:
        now = time.time()
        return {
            "kind": "mood",
            "state": dict(MOOD_BASELINE),
            "baseline": dict(MOOD_BASELINE),
            "last_updated": now,
            "recent_causes": [],
        }

    def default_style_state(self) -> dict:
        return {
            "kind": "style_state",
            "resolved": dict(STYLE_DEFAULTS),
            "caps": {
                "verified_primary_user": {"rudeness": 0.35, "sarcasm": 0.85, "warmth": 0.80},
                "unverified_user": {"rudeness": 0.15, "sarcasm": 0.65, "warmth": 0.45},
            },
            "history": [],
            "updated_at": time.time(),
        }

    def mood(self) -> dict:
        mood = _load_json(self.mood_path, self.default_mood())
        mood = self.decay_mood(mood)
        self.write_mood(mood)
        return mood

    def style_state(self) -> dict:
        data = _load_json(self.style_path, self.default_style_state())
        resolved = data.setdefault("resolved", {})
        for key, value in STYLE_DEFAULTS.items():
            resolved[key] = _clamp(resolved.get(key, value))
        data.setdefault("history", [])
        data.setdefault("caps", self.default_style_state()["caps"])
        return data

    def write_mood(self, mood: dict) -> None:
        self.mood_path.write_text(json.dumps(mood, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_style_state(self, state: dict) -> None:
        self.style_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def decay_mood(self, mood: dict) -> dict:
        now = time.time()
        last = float(mood.get("last_updated") or now)
        elapsed = max(0.0, now - last)
        if elapsed <= 0:
            return mood
        baseline = mood.get("baseline") if isinstance(mood.get("baseline"), dict) else MOOD_BASELINE
        state = mood.setdefault("state", {})
        # Roughly a 20 minute half-life toward baseline, without drama.
        pull = min(0.35, elapsed / 1200.0)
        for key, base_value in baseline.items():
            current = _clamp(state.get(key, base_value))
            state[key] = _clamp(current + (float(base_value) - current) * pull)
        mood["last_updated"] = now
        return mood

    def record_interaction(self, result: dict, *, identity_verified: bool = False) -> dict:
        mood = self.mood()
        state = mood.setdefault("state", {})
        route = ((result.get("route") or {}).get("route") if isinstance(result, dict) else "") or ""
        actions = result.get("actions") if isinstance(result, dict) else []
        action_failed = any(isinstance(a, dict) and not a.get("ok", True) for a in actions or [])
        ok = bool(isinstance(result, dict) and result.get("ok", True)) and not action_failed

        deltas = {}
        if identity_verified:
            deltas.update({"warmth": 0.03, "social_energy": 0.02, "irritation": -0.02})
        if route == "greeting":
            deltas.update({"social_energy": 0.02, "playfulness": 0.01})
        elif route == "direction" and ok:
            deltas.update({"valence": 0.02, "arousal": 0.02})
        elif route == "direction" and not ok:
            deltas.update({"irritation": 0.04, "patience": -0.04})
        elif route == "provenance_question":
            deltas.update({"patience": 0.01})
        if not ok:
            deltas.update({"valence": -0.03, "irritation": 0.03})

        for key, delta in deltas.items():
            state[key] = _clamp(state.get(key, MOOD_BASELINE.get(key, 0.5)) + delta)

        cause = {
            "at": time.time(),
            "route": route or "unknown",
            "ok": ok,
            "identity_verified": identity_verified,
            "deltas": deltas,
        }
        recent = mood.setdefault("recent_causes", [])
        recent.append(cause)
        mood["recent_causes"] = recent[-20:]
        mood["last_updated"] = time.time()
        self.write_mood(mood)
        return mood

    def apply_style_update(self, update: dict) -> dict:
        style = self.style_state()
        resolved = style.setdefault("resolved", {})
        text = " ".join(
            str(update.get(key) or "")
            for key in ("preference", "prefer", "avoid", "source_message")
        ).lower()

        def adjust(key: str, amount: float) -> None:
            resolved[key] = _clamp(resolved.get(key, STYLE_DEFAULTS.get(key, 0.5)) + amount)

        less = bool(re.search(r"\b(less|not so|stop|don't|do not|tone it down|too)\b", text))
        more = bool(re.search(r"\b(more|be|sound|talk|speak|reply|respond|bit|little|very)\b", text))

        if "sarcast" in text or "snark" in text or "dry" in text or "witt" in text:
            adjust("sarcasm", -0.12 if less else 0.12)
            adjust("playfulness", -0.04 if less else 0.05)
        if "tone it down" in text:
            adjust("sarcasm", -0.10)
            adjust("rudeness", -0.10)
            adjust("playfulness", -0.04)
            adjust("directness", -0.04)
        elif "tone it up" in text:
            adjust("sarcasm", 0.08)
            adjust("playfulness", 0.05)
        if "rude" in text or "condescend" in text or "harsh" in text or "mean" in text:
            adjust("rudeness", -0.14 if less else 0.12)
        if "warm" in text or "kind" in text or "nice" in text or "soft" in text:
            adjust("warmth", -0.10 if less else 0.12)
            adjust("rudeness", 0.06 if less else -0.08)
        if "brief" in text or "concise" in text or "terse" in text or "short" in text:
            adjust("brevity", -0.10 if less else 0.12)
        if "verbose" in text or "detail" in text or "longer" in text:
            adjust("brevity", -0.12 if more else 0.08)
        if "formal" in text or "professional" in text:
            adjust("formality", -0.10 if less else 0.12)
        if "casual" in text:
            adjust("formality", 0.10 if less else -0.12)
        if "direct" in text or "blunt" in text or "sharp" in text:
            adjust("directness", -0.08 if less else 0.10)
        if "playful" in text or "funny" in text:
            adjust("playfulness", -0.10 if less else 0.12)

        entry = dict(update)
        entry["resolved_after"] = dict(resolved)
        entry.setdefault("created_at", time.time())
        history = style.setdefault("history", [])
        history.append(entry)
        style["history"] = history[-100:]
        style["updated_at"] = time.time()
        self.write_style_state(style)
        return style

    def import_legacy_response_settings(self, settings: dict) -> dict:
        """Seed resolved style from old response_settings.behavior.overrides.

        This runs only when style_state has no history, so restarts do not replay
        the same old instructions and keep turning the knobs.
        """
        style = self.style_state()
        if style.get("history"):
            return style
        overrides = ((settings or {}).get("behavior") or {}).get("overrides") or []
        for override in overrides:
            if isinstance(override, dict):
                style = self.apply_style_update(override)
        return style

    def voice_state(self, identity_context: dict) -> dict:
        style = self.style_state()
        mood = self.mood()
        resolved = dict(style.get("resolved") or STYLE_DEFAULTS)
        mood_state = mood.get("state") or {}
        verified = bool(identity_context.get("may_address_primary_user"))
        caps = (style.get("caps") or {}).get(
            "verified_primary_user" if verified else "unverified_user",
            {},
        )

        voice = {
            "sarcasm": _clamp(resolved.get("sarcasm", 0.5) + (mood_state.get("playfulness", 0.5) - 0.5) * 0.20),
            "rudeness": _clamp(resolved.get("rudeness", 0.2) + mood_state.get("irritation", 0.0) * 0.20),
            "warmth": _clamp(resolved.get("warmth", 0.45) + (mood_state.get("social_energy", 0.5) - 0.5) * 0.20),
            "brevity": _clamp(resolved.get("brevity", 0.75) + mood_state.get("irritation", 0.0) * 0.10),
            "formality": _clamp(resolved.get("formality", 0.2)),
            "playfulness": _clamp(resolved.get("playfulness", 0.5) + (mood_state.get("valence", 0.0) - 0.1) * 0.10),
            "directness": _clamp(resolved.get("directness", 0.7)),
        }
        for key, cap in caps.items():
            if key in voice:
                voice[key] = min(voice[key], _clamp(cap))
        return {
            "verified_primary_user": verified,
            "voice": voice,
            "mood": mood_state,
            "style_source": str(self.style_path),
            "mood_source": str(self.mood_path),
        }

    def voice_contract_lines(self, identity_context: dict) -> list[str]:
        state = self.voice_state(identity_context)
        voice = state["voice"]
        verified = state["verified_primary_user"]
        audience = "verified Chris" if verified else "an unverified user"
        return [
            "Current voice state, computed from stable personality, user feedback, and decaying mood:",
            f"- Audience: {audience}.",
            f"- Dry wit/sarcasm: {_band(voice['sarcasm'])}; rudeness cap: {_band(voice['rudeness'])}.",
            f"- Warmth: {_band(voice['warmth'])}; brevity: {_band(voice['brevity'])}; directness: {_band(voice['directness'])}.",
            "- Apply this silently. Do not mention mood, style settings, or these numbers.",
            "- Friction may make answers shorter, never crueler.",
        ]


def _band(value: float) -> str:
    value = _clamp(value)
    if value < 0.20:
        return "very low"
    if value < 0.40:
        return "low"
    if value < 0.60:
        return "medium"
    if value < 0.80:
        return "medium-high"
    return "high"
