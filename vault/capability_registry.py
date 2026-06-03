"""Capability registry — system + learned capabilities loaded from JSON.

Hot-path optimizations (added during the orchestration audit):

* ``list()`` returns a CACHED concatenation rather than rebuilding the
  list on every call. The planner LLM round-trip calls ``list()``
  multiple times per turn (directly and via ``describe_for_prompt``).

* ``get(name)`` is O(1) via a name index.

* ``lookup_by_alias(normalized_text)`` is O(1) via a precomputed alias
  map. vault_runtime's ``_handle_named_capability_command`` used to
  iterate every capability and re-normalize all its alias strings on
  every presence turn.

* ``describe_for_prompt`` tracks source per-capability via a set of
  ``id()`` markers populated at load — was O(N²) with ``cap in
  self.learned_capabilities`` per iteration.

* ``add()`` no longer re-reads from disk after writing. The in-memory
  list is the source of truth for the rest of the request that
  triggered the write; the prior reload was throwing it away.
"""
import json
import os
import re

from config import LEARNED_CAPABILITIES_PATH, SYSTEM_CAPABILITIES_PATH


def _normalize_alias(text: str) -> str:
    """Same normalization vault_runtime._command_text applies. Lifted
    here so the registry can pre-compute the alias index at load time
    without importing from vault_runtime (which would be a cycle)."""
    return re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()


def _aliases_for(item: dict) -> set[str]:
    aliases: set[str] = set()
    for key in ("name", "display_name"):
        value = item.get(key)
        if value:
            aliases.add(_normalize_alias(str(value).replace("_", " ")))
            aliases.add(_normalize_alias(value))
    for example in item.get("examples") or []:
        if example:
            aliases.add(_normalize_alias(str(example).replace("_", " ")))
    return {alias for alias in aliases if alias}


class CapabilityRegistry:
    def __init__(
        self,
        system_path=SYSTEM_CAPABILITIES_PATH,
        learned_path=LEARNED_CAPABILITIES_PATH,
    ):
        self.system_path = system_path
        self.learned_path = learned_path
        self.system_capabilities: list[dict] = []
        self.learned_capabilities: list[dict] = []
        # Caches rebuilt on every load() / add()
        self._all_cache: list[dict] = []
        self._by_name: dict[str, dict] = {}
        self._by_alias: dict[str, dict] = {}
        self._learned_ids: set[int] = set()
        self.load()

    def load(self):
        self.system_capabilities = self._load_json(self.system_path)
        self.learned_capabilities = self._load_json(self.learned_path)
        self._rebuild_indices()

    def _load_json(self, path):
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")

        return data

    def _rebuild_indices(self) -> None:
        """Rebuild list/name/alias caches. Cheap — called only on load
        and add, not on the hot per-turn path."""
        self._all_cache = self.system_capabilities + self.learned_capabilities
        self._by_name = {}
        self._by_alias = {}
        for cap in self._all_cache:
            name = cap.get("name")
            if name:
                self._by_name[name] = cap
            for alias in _aliases_for(cap):
                # First writer wins so system capabilities take
                # precedence over a learned one with a colliding alias.
                self._by_alias.setdefault(alias, cap)
        self._learned_ids = {id(c) for c in self.learned_capabilities}

    def _save_learned(self):
        os.makedirs(os.path.dirname(self.learned_path), exist_ok=True)

        with open(self.learned_path, "w", encoding="utf-8") as f:
            json.dump(self.learned_capabilities, f, indent=2)

    def list(self) -> list[dict]:
        # Return the same list object each time — callers iterate and
        # don't mutate, so the copy that used to be returned per call
        # was just wasted GC pressure.
        return self._all_cache

    def get(self, name):
        return self._by_name.get(name)

    def lookup_by_alias(self, normalized_text: str) -> dict | None:
        """O(1) alias match. Caller passes a string already normalized
        by the same rules used in ``_normalize_alias`` (no punctuation,
        lowercased, whitespace-collapsed). Returns the capability dict
        or None."""
        if not normalized_text:
            return None
        return self._by_alias.get(normalized_text)

    def add(self, capability):
        if not capability.get("name"):
            raise ValueError("Capability requires name")

        existing = None

        for cap in self.learned_capabilities:
            if cap.get("name") == capability["name"]:
                existing = cap
                break

        if existing:
            existing.update(capability)
        else:
            self.learned_capabilities.append(capability)

        self._save_learned()
        # Previously this called self.load() which re-parsed both JSON
        # files from disk just to throw away the in-memory state we
        # just updated. Just rebuild the indices in place.
        self._rebuild_indices()

    def describe_for_prompt(self):
        lines = []
        for cap in self._all_cache:
            name = cap.get("name")
            desc = cap.get("description")
            examples = ", ".join(cap.get("examples", [])[:3])
            source = "learned" if id(cap) in self._learned_ids else "system"

            lines.append(
                f"- {name} [{source}]: {desc} (examples: {examples})"
            )

        return "\n".join(lines)
