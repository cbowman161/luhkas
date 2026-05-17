"""Lightweight request-owner hints for LUHKAS node chat routing."""
from __future__ import annotations

import os
import re


_VAULT_ALIASES = {
    "vault",
    "the vault",
    "brain",
    "the brain",
    "server",
    "home server",
}

_DEFAULT_NODE_ALIASES = {
    "scout": {"scout", "the scout", "robot", "the robot", "rover", "the rover"},
    "wall_mount": {
        "wall mount",
        "wall-mounted",
        "wall mounted",
        "wallmount",
        "wall node",
        "the wall mount",
    },
}


def classify_request_target(message: str, current_node_id: str | None = None) -> dict:
    """Return a conservative owner hint for a request.

    This does not classify the command itself. It only answers whether an
    explicitly named owner should prevent local-first handling on this node.
    """
    node_id = _normalize_node_id(current_node_id or os.environ.get("LUHKAS_NODE_ID") or "scout")
    text = _normalized_text(message)
    if not text:
        return _intent("unspecified", node_id, False, 0.0, "empty message")

    if _contains_alias(text, _VAULT_ALIASES):
        return _intent("vault", "vault", True, 0.95, "explicit vault/brain target")

    aliases = _known_node_aliases(node_id)
    for target_node, target_aliases in aliases.items():
        if _contains_alias(text, target_aliases):
            if target_node == node_id:
                return _intent("current_node", node_id, True, 0.95, "explicit current-node target")
            return _intent("node", target_node, True, 0.9, "explicit other-node target")

    return _intent("unspecified", node_id, False, 0.45, "no explicit target")


def should_attempt_local(message: str, current_node_id: str | None = None) -> bool:
    intent = classify_request_target(message, current_node_id)
    owner = intent.get("request_owner")
    target = intent.get("target_node")
    current = _normalize_node_id(current_node_id or os.environ.get("LUHKAS_NODE_ID") or "scout")
    return owner in {"unspecified", "current_node"} or target == current


def _known_node_aliases(current_node_id: str) -> dict[str, set[str]]:
    aliases = {node: set(values) for node, values in _DEFAULT_NODE_ALIASES.items()}
    aliases.setdefault(current_node_id, set()).update({current_node_id, current_node_id.replace("_", " ")})
    for alias in os.environ.get("LUHKAS_NODE_ALIASES", "").split(","):
        alias = alias.strip()
        if alias:
            aliases[current_node_id].add(alias)
    return aliases


def _intent(owner: str, target_node: str, explicit: bool, confidence: float, reason: str) -> dict:
    return {
        "request_owner": owner,
        "target_node": target_node,
        "explicit_target": explicit,
        "confidence": confidence,
        "reason": reason,
    }


def _normalize_node_id(value: str) -> str:
    value = re.sub(r"[^\w]+", "_", str(value).strip().casefold()).strip("_")
    return value or "unknown"


def _normalized_text(value: str) -> str:
    text = re.sub(r"[^\w\s-]", " ", str(value).casefold())
    return re.sub(r"\s+", " ", text).strip()


def _contains_alias(text: str, aliases: set[str]) -> bool:
    padded = f" {text} "
    for alias in aliases:
        normalized = _normalized_text(alias)
        if normalized and f" {normalized} " in padded:
            return True
    return False
