"""User-facing review of consolidated behavior conflicts.

The consolidation algorithm in :py:mod:`behavior_consolidator` produces
``ConflictPair`` objects but doesn't act on them — by design, because
silent resolution of user-stated preferences is untrustworthy. This
module is the UX glue: it turns verified conflicts into pending
decisions ("you said A, but also B — which would you like to keep?")
and acts on the user's answer.

Two entry points:

  - :py:meth:`try_handle_review_intent` — matches verbal triggers
    (`"review my preferences"`, `"consolidate preferences"`, etc.),
    runs the consolidator, surfaces the first verified conflict (if
    any) as a pending decision, and queues the rest for follow-up
    turns.

  - :py:meth:`resolve_conflict` — called by the runtime when an active
    pending of type ``behavior_conflict_resolution`` is being resolved.
    Parses the user's choice (first/second/both/neither), applies it
    (delete loser, keep winner, etc.), and either advances to the next
    queued conflict or closes the pending.

Pending-state plumbing is dependency-injected via callbacks so the
controller stays free of vault_runtime imports and is testable in
isolation.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable, Optional

from behavior_consolidator import BehaviorConsolidator, ConflictPair
from onboarding import ProfileStore, TTSChannel
from storage.behavior_store import BehaviorMemoryStore


log = logging.getLogger(__name__)


PENDING_TYPE = "behavior_conflict_resolution"


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

_REVIEW_PATTERNS = [
    re.compile(r"\breview\s+(?:my\s+|the\s+)?(?:behavior\s+notes|preferences|conflicting\s+preferences)\b", re.I),
    re.compile(r"\bconsolidate\s+(?:my\s+|the\s+)?preferences\b", re.I),
    re.compile(r"\baudit\s+(?:my\s+|the\s+)?preferences\b", re.I),
    re.compile(r"\bcheck\s+(?:for\s+)?(?:conflicting\s+)?preferences\b", re.I),
    re.compile(r"\bfind\s+conflicts\s+in\s+(?:my\s+|the\s+)?preferences\b", re.I),
]

_FIRST = re.compile(r"\b(?:first|the\s+first|first\s+one|keep\s+(?:the\s+)?first|a|keep\s+a)\b", re.I)
_SECOND = re.compile(r"\b(?:second|the\s+second|second\s+one|keep\s+(?:the\s+)?second|b|keep\s+b)\b", re.I)
_BOTH = re.compile(r"\b(?:both|keep\s+both|leave\s+(?:them\s+)?(?:as\s+is|alone)|don'?t\s+change)\b", re.I)
_NEITHER = re.compile(r"\b(?:neither|delete\s+both|remove\s+both)\b", re.I)
_CANCEL = re.compile(r"\b(?:cancel|never\s+mind|stop|skip)\b", re.I)


def classify_review_intent(user_input: str) -> bool:
    """True iff ``user_input`` matches a 'kick off the review' phrase."""
    if not user_input:
        return False
    return any(p.search(user_input) for p in _REVIEW_PATTERNS)


def classify_resolution(user_input: str) -> Optional[str]:
    """Parse the user's answer to a surfaced conflict. Returns one of
    ``"first" | "second" | "both" | "neither" | "cancel"`` or ``None``
    if ambiguous (caller should re-ask).

    Cancel is matched first so "never mind, the first one" doesn't
    accidentally pick first. Both/neither before first/second so
    "both of them are fine" doesn't trigger A."""
    if not user_input:
        return None
    if _CANCEL.search(user_input):
        return "cancel"
    # ``_NEITHER`` checked before ``_BOTH`` because phrases like
    # "delete both" / "remove both" contain the word "both" — without
    # this order, _BOTH would shadow them and return the wrong intent.
    if _NEITHER.search(user_input):
        return "neither"
    if _BOTH.search(user_input):
        return "both"
    if _FIRST.search(user_input):
        return "first"
    if _SECOND.search(user_input):
        return "second"
    return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class BehaviorReviewController:
    """Run the review-and-resolve flow for verified behavior conflicts.

    The runtime injects: a consolidator, the store (for deletes), the
    TTS channel (for speaking surfacing questions), the profile store
    (for resolving the current speaker), and three callbacks for
    pending-state plumbing. Per-node conflict queues live in pending
    state too; the runtime is the single owner of that state."""

    def __init__(
        self,
        *,
        consolidator: BehaviorConsolidator,
        store: BehaviorMemoryStore,
        profile_store: ProfileStore,
        tts: TTSChannel,
        pending_set: Callable[[str, dict], None],
        pending_get: Callable[[str], Optional[dict]],
        pending_clear: Callable[[str], None],
    ) -> None:
        self.consolidator = consolidator
        self.store = store
        self.profile_store = profile_store
        self.tts = tts
        self._pending_set = pending_set
        self._pending_get = pending_get
        self._pending_clear = pending_clear

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def try_handle_review_intent(self, user_input: str, node_id: str) -> Optional[dict]:
        """If the user kicked off a review, run consolidation and
        surface the first verified conflict (if any). Returns the
        response dict; returns None when the input wasn't a review
        intent so the runtime falls through to normal dispatch."""
        if not classify_review_intent(user_input):
            return None

        identity = self._caller_identity()
        try:
            conflicts = self.consolidator.consolidate(identity=identity)
        except Exception as exc:
            log.warning("consolidator.consolidate failed: %s", exc)
            conflicts = []

        if not conflicts:
            return self._reply(
                node_id,
                "I didn't find any conflicting preferences — nothing to resolve.",
                event="no_conflicts",
            )

        # Queue all conflicts; surface the first.
        queue = [_conflict_to_pending_item(c) for c in conflicts]
        first = queue[0]
        self._pending_set(node_id, {
            "type": PENDING_TYPE,
            "queue": queue,
            "current_index": 0,
        })
        return self._surface(node_id, first, queued=len(queue), index=0)

    def resolve_conflict(self, user_input: str, node_id: str) -> Optional[dict]:
        """Runtime calls this when the active pending is of type
        ``behavior_conflict_resolution`` and the user just spoke.
        Returns None if no such pending is active so the caller can
        fall through."""
        pending = self._pending_get(node_id)
        if not pending or pending.get("type") != PENDING_TYPE:
            return None

        choice = classify_resolution(user_input)
        if choice is None:
            return self._reply(
                node_id,
                "Say 'first', 'second', 'both', or 'neither' — or 'cancel' to skip.",
                event="ambiguous",
            )

        queue: list = list(pending.get("queue") or [])
        idx = int(pending.get("current_index") or 0)
        if idx >= len(queue):
            self._pending_clear(node_id)
            return self._reply(node_id, "Done.", event="completed")

        item = queue[idx]
        note_a_id = item.get("note_a_id")
        note_b_id = item.get("note_b_id")
        a_content = item.get("note_a_content", "")
        b_content = item.get("note_b_content", "")

        if choice == "cancel":
            self._pending_clear(node_id)
            return self._reply(node_id, "Okay — leaving everything as is.", event="canceled")

        # Apply the resolution.
        deleted: list[str] = []
        if choice == "first":
            self._safe_delete(note_b_id, deleted)
            applied_msg = f"Got it — keeping the first one. Removed: \"{b_content}\"."
        elif choice == "second":
            self._safe_delete(note_a_id, deleted)
            applied_msg = f"Got it — keeping the second one. Removed: \"{a_content}\"."
        elif choice == "both":
            applied_msg = "Got it — leaving both."
        elif choice == "neither":
            self._safe_delete(note_a_id, deleted)
            self._safe_delete(note_b_id, deleted)
            applied_msg = "Got it — removed both."
        else:  # pragma: no cover — exhaustive above
            applied_msg = "Okay."

        # Advance the queue. If more remain, surface the next; else close.
        next_idx = idx + 1
        if next_idx >= len(queue):
            self._pending_clear(node_id)
            self.tts.say(node_id, applied_msg)
            return {
                "mode": "direct",
                "message": applied_msg,
                "behavior_review": {
                    "event": "resolved_last",
                    "choice": choice,
                    "deleted": deleted,
                },
            }

        next_item = queue[next_idx]
        self._pending_set(node_id, {
            **pending,
            "current_index": next_idx,
        })
        self.tts.say(node_id, applied_msg)
        return self._surface(
            node_id, next_item,
            queued=len(queue), index=next_idx,
            prefix=applied_msg + " Next: ",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _caller_identity(self) -> str | None:
        """Identity to consolidate for. Defaults to the current primary
        user (so 'review my preferences' from the primary user reviews
        their notes). Falls back to None → global namespace."""
        try:
            return self.profile_store.primary_user()
        except Exception:
            return None

    def _safe_delete(self, note_id: str | None, sink: list[str]) -> None:
        if not note_id:
            return
        try:
            ok = self.store.delete_by_id(note_id)
        except Exception as exc:
            log.warning("delete_by_id(%s) failed: %s", note_id, exc)
            return
        if ok:
            sink.append(note_id)

    def _surface(
        self,
        node_id: str,
        item: dict,
        *,
        queued: int,
        index: int,
        prefix: str = "",
    ) -> dict:
        a = item.get("note_a_content", "")
        b = item.get("note_b_content", "")
        suffix = ""
        if queued > 1:
            suffix = f" (Conflict {index + 1} of {queued}.)"
        message = (
            f"{prefix}You said: \"{a}\". And also: \"{b}\". "
            f"Which should I keep — first, second, both, or neither?{suffix}"
        )
        self.tts.say(node_id, message)
        return {
            "mode": "direct",
            "message": message,
            "behavior_review": {
                "event": "surfacing",
                "current_index": index,
                "queue_size": queued,
                "note_a_id": item.get("note_a_id"),
                "note_b_id": item.get("note_b_id"),
            },
        }

    def _reply(self, node_id: str, text: str, *, event: str) -> dict:
        self.tts.say(node_id, text)
        return {
            "mode": "direct",
            "message": text,
            "behavior_review": {"event": event},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conflict_to_pending_item(pair: ConflictPair) -> dict:
    """Flatten a ConflictPair into a JSON-safe dict for the pending
    queue. Drops vectors and other non-essential fields; keeps only
    what surfacing + resolution needs (ids + content)."""
    return {
        "note_a_id": (pair.note_a or {}).get("id"),
        "note_b_id": (pair.note_b or {}).get("id"),
        "note_a_content": (pair.note_a or {}).get("content", ""),
        "note_b_content": (pair.note_b or {}).get("content", ""),
        "distance": pair.distance,
        "reason": pair.reason,
    }
