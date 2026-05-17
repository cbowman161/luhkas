from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .config import PersonMemoryConfig


class PersonMemoryStore:
    def __init__(self, config: PersonMemoryConfig | None = None, brain_client=None) -> None:
        self.config = config or PersonMemoryConfig()
        self.brain_client = brain_client
        self.people_dir = Path(self.config.people_dir).expanduser()
        if self.config.enabled:
            self.people_dir.mkdir(parents=True, exist_ok=True)

    def summary_for(self, identity: str | None) -> dict | None:
        identity = _safe_identity(identity or "")
        if not self.config.enabled or not identity:
            return None

        if self.brain_client is not None and self.brain_client.config.prefer_brain_person_memory:
            summary = self.brain_client.person_summary(identity)
            if summary is not None:
                return summary

        profile = self._load_profile(identity)
        preferences = profile.get("preferences", {})
        facts = profile.get("facts", {})
        summary_keys = self._summary_keys()
        summary_preferences = {
            key: preferences[key]
            for key in summary_keys
            if key in preferences
        }
        summary_facts = {
            key: facts[key]
            for key in summary_keys
            if key in facts
        }
        return {
            "identity": identity,
            "known": True,
            "display_name": profile.get("display_name") or facts.get("display_name") or identity,
            "preferences": summary_preferences,
            "facts": summary_facts,
            "updated_at": profile.get("updated_at"),
        }

    def get_profile(self, identity: str) -> dict:
        identity = _safe_identity(identity)
        if not identity:
            return {"ok": False, "error": "missing_identity"}
        if self.brain_client is not None and self.brain_client.config.prefer_brain_person_memory:
            result = self.brain_client.get_person_memory(identity)
            if result is not None:
                return result
        return {"ok": True, "profile": self._load_profile(identity), "memories": self._load_memories(identity)}

    def remember(self, identity: str, memory_type: str, key: str, value: Any, source: str = "user", confidence: float = 1.0) -> dict:
        identity = _safe_identity(identity)
        key = _safe_key(key)
        if not self.config.enabled:
            return {"ok": False, "error": "person_memory_disabled"}
        if not identity or not key:
            return {"ok": False, "error": "missing_identity_or_key"}

        memory_type = memory_type if memory_type in {"fact", "preference", "interaction", "note"} else "fact"
        payload = {
            "type": memory_type,
            "key": key,
            "value": value,
            "source": source,
            "confidence": float(confidence),
        }
        if self.brain_client is not None and self.brain_client.config.prefer_brain_person_memory:
            result = self.brain_client.remember(identity, payload)
            if result is not None:
                return result

        profile = self._load_profile(identity)
        now = time.time()
        if memory_type == "preference":
            profile.setdefault("preferences", {})[key] = value
        elif memory_type == "fact":
            profile.setdefault("facts", {})[key] = value
            if key == "display_name":
                profile["display_name"] = str(value)

        profile["identity"] = identity
        profile["updated_at"] = now
        self._write_profile(identity, profile)
        event = {
            "type": memory_type,
            "key": key,
            "value": value,
            "source": source,
            "confidence": float(confidence),
            "created_at": now,
        }
        self._append_memory(identity, event)
        return {"ok": True, "identity": identity, "summary": self.summary_for(identity), "event": event}

    def set_preference(self, identity: str, key: str, value: Any, source: str = "user") -> dict:
        identity = _safe_identity(identity)
        key = _safe_key(key)
        if self.brain_client is not None and self.brain_client.config.prefer_brain_person_memory:
            result = self.brain_client.set_preference(identity, {"key": key, "value": value, "source": source})
            if result is not None:
                return result
        return self.remember(identity, "preference", key, value, source=source, confidence=1.0)

    def _summary_keys(self) -> set[str]:
        return {
            key.strip()
            for key in self.config.summary_preference_keys.split(",")
            if key.strip()
        }

    def _person_dir(self, identity: str) -> Path:
        return self.people_dir / identity

    def _profile_path(self, identity: str) -> Path:
        return self._person_dir(identity) / "profile.json"

    def _memories_path(self, identity: str) -> Path:
        return self._person_dir(identity) / "memories.jsonl"

    def _load_profile(self, identity: str) -> dict:
        path = self._profile_path(identity)
        if not path.exists():
            return {
                "identity": identity,
                "display_name": identity,
                "preferences": {},
                "facts": {},
                "created_at": time.time(),
                "updated_at": None,
            }
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("identity", identity)
        data.setdefault("display_name", identity)
        data.setdefault("preferences", {})
        data.setdefault("facts", {})
        return data

    def _write_profile(self, identity: str, profile: dict) -> None:
        person_dir = self._person_dir(identity)
        person_dir.mkdir(parents=True, exist_ok=True)
        with open(self._profile_path(identity), "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _append_memory(self, identity: str, event: dict) -> None:
        person_dir = self._person_dir(identity)
        person_dir.mkdir(parents=True, exist_ok=True)
        with open(self._memories_path(identity), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    def _load_memories(self, identity: str, limit: int = 50) -> list[dict]:
        path = self._memories_path(identity)
        if not path.exists():
            return []
        memories = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    memories.append(event)
        return memories[-limit:]


def _safe_identity(identity: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity.strip())
    return cleaned.strip("._-")[:64]


def _safe_key(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip())
    return cleaned.strip("._-")[:80]
