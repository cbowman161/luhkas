"""Retrieve + format behavior notes for injection into model prompts.

The Retrieve + Apply stages of the feedback-learning loop:

  - **Retrieve**: query the behavior store, filter to relevant scope
    (current identity + active route + current domain), pick top-N.
  - **Apply**: format as a compact system-prompt preamble that any
    model-call site can prepend. Bump ``apply_count`` on success so
    the consolidation pass later knows which notes are pulling weight.

Designed to be cheap on miss (empty result is a no-op, no string
formatting). When notes hit, the injected text is small on purpose —
the model is supposed to *internalize* the preferences, not have a
wall of rules pasted in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from storage.behavior_store import BehaviorMemoryStore


log = logging.getLogger(__name__)


# The preamble header. Kept short and natural — "User preferences:"
# reads cleaner inside a chat-completion prompt than "SYSTEM_DIRECTIVE".
_PREAMBLE_HEADER = "User preferences:"


@dataclass
class BehaviorContext:
    """Result of a context build. ``text`` is ready to inject into a
    system prompt; ``note_ids`` is the list of notes that contributed,
    so the caller can bump ``apply_count`` after the response lands.

    Empty context is falsy — ``if context: ...`` does the right thing
    on the no-retrieval case."""

    text: str = ""
    note_ids: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text and self.note_ids)


class BehaviorContextBuilder:
    """Build a prompt preamble from the behavior store.

    Wraps :py:class:`BehaviorMemoryStore`. Call sites in chat paths
    pass the current ``(identity, active_route, current_domain)`` and
    the user's message as the query; ``build()`` returns a
    :class:`BehaviorContext` they can hand to the response composer
    (or any model-call site)."""

    def __init__(self, store: BehaviorMemoryStore, *, top_k: int = 5):
        self.store = store
        self.top_k = max(1, int(top_k))

    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        *,
        identity: str | None = None,
        active_route: str | None = None,
        current_domain: str | None = None,
        category: str | Iterable[str] | None = None,
    ) -> BehaviorContext:
        """Retrieve relevant notes and format them as a preamble.
        Returns an empty :class:`BehaviorContext` when no notes match
        — caller can pass it through; falsy contexts inject nothing."""
        if not query or not query.strip():
            return BehaviorContext()
        try:
            notes = self.store.retrieve(
                query=query,
                identity=identity,
                active_route=active_route,
                domain=current_domain,
                top_k=self.top_k,
                category=category,
            )
        except Exception as exc:
            log.warning("BehaviorContextBuilder.retrieve failed: %s", exc)
            return BehaviorContext()
        if not notes:
            return BehaviorContext()
        return BehaviorContext(
            text=_format_preamble(notes),
            note_ids=[n["id"] for n in notes if n.get("id")],
        )

    def apply(self, context: BehaviorContext) -> None:
        """Bump ``apply_count`` on every note that contributed to a
        response. Call from the response-composer's success path so we
        only count notes that *actually* influenced a model call —
        retrievals that landed in the prompt but were then discarded
        (validator rejected, fallback fired) shouldn't inflate the
        promotion signal."""
        for note_id in context.note_ids:
            try:
                self.store.update_apply_count(note_id)
            except Exception as exc:
                log.warning("update_apply_count(%s) failed: %s", note_id, exc)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_preamble(notes: list[dict]) -> str:
    """Render notes as a compact bullet list under a single header.

    Order: by category (constraints first — they're hard rules the
    model must respect), then by descending apply_count (notes that
    have been useful before are pulling their weight), then by recency.

    The bullet text is the user's exact words. Natural language in,
    natural language out — the model interprets, we don't pre-digest.
    """
    # Group by category, then sort within groups.
    by_category: dict[str, list[dict]] = {}
    for n in notes:
        by_category.setdefault(n.get("category", "preference"), []).append(n)

    # Ordering: constraint first (hard rules), preference, capability_hint,
    # correction last (per-turn corrections rarely re-apply to future turns
    # but they're useful for "you said X — actually it was Y" recall).
    category_order = ("constraint", "preference", "capability_hint", "correction")
    bullets: list[str] = []
    for cat in category_order:
        items = by_category.get(cat, [])
        items.sort(
            key=lambda n: (
                -int(n.get("apply_count") or 0),
                -(float(n.get("updated_at") or 0.0)),
            )
        )
        for n in items:
            content = (n.get("content") or "").strip()
            if not content:
                continue
            bullets.append(f"- {content}")

    if not bullets:
        return ""
    return f"{_PREAMBLE_HEADER}\n" + "\n".join(bullets)
