"""Extract person names and node names from user messages as routing hints.

Used by chat_context.py to enrich vault presence payloads, and by
vault-side routing to narrow LLM inference when no deterministic match fires.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


_RUNTIME_ROOT = Path(__file__).resolve().parents[1]

_BUILTIN_NODE_ALIASES: set[str] = {
    "scout", "the scout", "robot", "the robot", "rover", "the rover",
    "vault", "the vault", "brain", "the brain", "server", "home server",
    "wall mount", "the wall mount", "wall-mount", "wallmount", "wall node",
}


def extract_keywords(message: str, config_root: Path | None = None) -> dict:
    """Return people names and node names mentioned in *message*.

    Returns:
        {
            "people": [...],        # known person names found in message
            "nodes": [...],         # node/device aliases found in message
            "has_person_reference": bool,
            "has_node_reference": bool,
        }
    """
    root = Path(config_root) if config_root else _RUNTIME_ROOT
    text = str(message or "").casefold()

    people = _known_people(root)
    matched_people = [name for name in people if _phrase_in_text(name, text)]

    node_aliases = set(_BUILTIN_NODE_ALIASES)
    for extra in os.environ.get("LUHKAS_NODE_ALIASES", "").split(","):
        alias = extra.strip().casefold()
        if alias:
            node_aliases.add(alias)
    matched_nodes = sorted({alias for alias in node_aliases if _phrase_in_text(alias, text)})

    return {
        "people": matched_people,
        "nodes": matched_nodes,
        "has_person_reference": bool(matched_people),
        "has_node_reference": bool(matched_nodes),
    }


def _known_people(root: Path) -> list[str]:
    people: set[str] = set()
    for subdir in ("config/vault_faces", "config/faces", "config/people"):
        d = root / subdir
        if not d.exists():
            continue
        for entry in d.iterdir():
            name = entry.name
            if entry.is_dir() and not name.startswith(".") and name:
                people.add(name.casefold())
    return sorted(people)


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized = re.sub(r"[^\w\s-]", " ", phrase)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text))
