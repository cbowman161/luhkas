"""Behavior-note consolidation — the Consolidate stage.

Iterates the behavior store looking for **conflicts**: pairs of notes
whose embeddings are close enough to be about the same subject but
not so close they're duplicates. The hard call ("is this an actual
contradiction or just a coincidence of similar wording?") goes to an
LLM verifier — the substrate gives us a cheap pre-filter, the model
gives us the judgment.

Surfacing the verified conflicts to the user (so they can resolve)
is the caller's job; this module is pure data. The runtime calls
``consolidate()`` periodically (much slower cadence than the
onboarding ticker — minutes/hours, not seconds) and routes the
result into a pending-decision flow.

Deliberately scoped narrow:
  - **No decay.** Hard-deleting notes that haven't been re-applied is
    destructive — the user might just not have hit a query that
    surfaced them. Confidence-based fade-out is a future addition.
  - **No automatic resolution.** When we find a real contradiction,
    BOTH notes stay in the store. The user picks which to keep.
  - **Per-namespace.** Each (identity) is consolidated independently;
    global notes are their own namespace. Cross-identity contradictions
    aren't meaningful — different users can hold different preferences.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from storage.behavior_store import BehaviorMemoryStore, GLOBAL_IDENTITY


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ConflictPair:
    """A candidate or verified conflict between two notes.

    ``verified`` is True only after the LLM verifier confirms an actual
    contradiction. Before verification it's just a pre-filter hit —
    close embeddings that *could* be about the same subject."""

    note_a: dict[str, Any]
    note_b: dict[str, Any]
    distance: float
    verified: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class BehaviorConsolidator:
    """Find + verify conflicting behavior notes.

    Constructed with the store and optionally an LLM verifier
    (``model`` exposing ``.generate(prompt, ...)``). Without a model,
    :py:meth:`find_pairs` still works — useful for tests and for
    debugging the pre-filter cleanly. :py:meth:`verify_pair` and
    :py:meth:`consolidate` require the model."""

    def __init__(self, store: BehaviorMemoryStore, *, model=None):
        self.store = store
        self.model = model

    # ------------------------------------------------------------------
    # Pre-filter — pure substrate, no LLM
    # ------------------------------------------------------------------

    def find_pairs(
        self,
        identity: str | None = None,
        *,
        distance_min: float = 0.25,
        distance_max: float = 0.65,
    ) -> list[ConflictPair]:
        """Iterate notes in the (identity) namespace, asking the store
        for conflict candidates per note. Returns deduplicated pairs
        (each unordered pair is yielded once)."""
        ident = identity or GLOBAL_IDENTITY
        try:
            notes = self.store.list_for_identity(ident, limit=1000)
        except Exception as exc:
            log.warning("find_pairs list_for_identity(%s) failed: %s", ident, exc)
            return []
        # list_for_identity returns the named identity's notes PLUS
        # global notes when identity != GLOBAL_IDENTITY. For
        # consolidation purposes we want to look only at one namespace
        # at a time, so filter to exact identity match here.
        notes = [n for n in notes if n.get("identity") == ident]
        if len(notes) < 2:
            return []

        seen: set[tuple[str, str]] = set()
        pairs: list[ConflictPair] = []
        for note in notes:
            content = note.get("content")
            if not content:
                continue
            try:
                candidates = self.store.find_conflict_candidates(
                    content,
                    identity=ident,
                    distance_min=distance_min,
                    distance_max=distance_max,
                )
            except Exception as exc:
                log.warning("find_conflict_candidates failed: %s", exc)
                continue
            for cand in candidates:
                if cand.get("id") == note.get("id"):
                    continue  # self-match (shouldn't fire — distance > min)
                key = _pair_key(note.get("id"), cand.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(ConflictPair(
                    note_a=note,
                    note_b=cand,
                    distance=float(cand.get("distance") or 0.0),
                ))
        return pairs

    # ------------------------------------------------------------------
    # Verification — LLM judgment
    # ------------------------------------------------------------------

    def verify_pair(self, pair: ConflictPair) -> ConflictPair:
        """Ask the LLM whether two notes actually contradict each
        other. Returns the same pair with ``verified`` set and
        ``reason`` filled. Defaults to ``verified=False`` on any
        failure — surfacing a false contradiction to the user is
        worse than missing one."""
        if self.model is None:
            return pair
        try:
            raw = self.model.generate(
                _verifier_prompt(pair.note_a, pair.note_b),
                think=False,
                timeout=10,
                options={"num_predict": 100, "temperature": 0.1},
            )
        except Exception as exc:
            log.warning("verifier model.generate failed: %s", exc)
            return pair
        verdict = _parse_verdict(raw or "")
        return ConflictPair(
            note_a=pair.note_a,
            note_b=pair.note_b,
            distance=pair.distance,
            verified=bool(verdict.get("contradicts")),
            reason=str(verdict.get("reason") or "").strip(),
        )

    # ------------------------------------------------------------------
    # Top-level entry
    # ------------------------------------------------------------------

    def consolidate(
        self,
        identity: str | None = None,
        *,
        verify: bool = True,
        max_pairs: int = 20,
    ) -> list[ConflictPair]:
        """Find candidate pairs and (when ``verify`` and a model are
        available) confirm each via LLM. Returns the list of pairs
        where ``verified=True``.

        ``max_pairs`` caps the LLM calls per pass — even a small store
        can produce O(n²) candidates before filtering. Capping keeps
        each consolidation pass bounded; a future pass will catch any
        deferred pairs on the next run."""
        pairs = self.find_pairs(identity)
        if not pairs:
            return []
        if not verify or self.model is None:
            return pairs
        confirmed: list[ConflictPair] = []
        for pair in pairs[:max_pairs]:
            verified = self.verify_pair(pair)
            if verified.verified:
                confirmed.append(verified)
        return confirmed


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------


def _verifier_prompt(note_a: dict, note_b: dict) -> str:
    """The verifier asks the smallest question that gets a clean
    yes/no: do these two user statements give the system contradictory
    instructions on how to behave?

    Not "are they semantically related" (the embedding pre-filter
    already established that) and not "which is right" (that's the
    user's call). Just the contradiction check.
    """
    return f"""You are checking whether two user-given behavior preferences contradict each other.

Statement A: "{(note_a.get('content') or '').strip()}"
Statement B: "{(note_b.get('content') or '').strip()}"

Question: If a system tried to honor BOTH statements, would it have to violate one of them?

Respond with a JSON object only — no other text. Example shape:
{{"contradicts": true, "reason": "A asks for brief responses, B asks for detailed responses; these can't both be honored."}}
or
{{"contradicts": false, "reason": "Both can be honored simultaneously."}}
"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _parse_verdict(raw: str) -> dict:
    """Best-effort JSON extraction. The verifier prompt asks for clean
    JSON but small models often add prose. We accept the first ``{...}``
    block. On any failure, default to ``contradicts=False`` —
    surfacing a false contradiction is worse than missing one."""
    text = (raw or "").strip()
    if not text:
        return {"contradicts": False, "reason": "empty model response"}
    m = _JSON_BLOCK.search(text)
    if not m:
        return {"contradicts": False, "reason": "no JSON found"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"contradicts": False, "reason": "malformed JSON"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_key(a_id: str | None, b_id: str | None) -> tuple[str, str]:
    """Canonical (unordered) key for a pair of note ids so iteration
    doesn't yield (a,b) and (b,a) as separate pairs."""
    if a_id is None or b_id is None:
        return ("", "")
    return tuple(sorted([str(a_id), str(b_id)]))
