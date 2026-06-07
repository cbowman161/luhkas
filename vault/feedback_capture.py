"""Capture-side detector for the feedback-learning loop.

Watches every user turn for phrases that signal the user is teaching
the system how to behave — "be more concise", "always confirm before
deleting", "I prefer pytest" — and writes them to BehaviorMemoryStore.
The Apply layer (separate, future) reads from that store at model-call
prompt-construction time.

Design: **L0 only** for now. Phrase/regex patterns that are inherently
directive ("be more X", "I prefer X", "never X"). High precision —
better to miss soft feedback than false-positive on factual statements
("never gonna give you up" must NOT capture). L1 (embedding-similarity
against canonical examples) and L2 (small LLM fallback for ambiguous
cases) are deferred until real-usage telemetry shows what L0 misses.

The "feel" of teachability comes from acknowledgment: when a capture
fires, the system briefly confirms ("Got it — I'll keep things more
concise") so the user can see the lesson landed. Short on purpose —
the user often just asked us to be more concise.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from storage.behavior_store import BehaviorMemoryStore


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------
#
# Each entry is (regex, category, ack_template).
#
# Patterns are matched in order; first hit wins. Order matters when
# phrasings overlap — e.g., "actually" should catch correction before
# any preference pattern that mentions "actually."
#
# The regex captures a leading verb phrase or directive payload in
# group 1 (or 2 for the "be (more|less)" shape) when present, so the
# stored content can be re-shaped from the user's raw words. The raw
# user input is *also* stored as the content — natural language is the
# storage form on purpose. The captured group is just for the ack.

# Adjectives that appear in "be (more|less) X" — used to keep that
# pattern from matching things like "be more careful with this query"
# (a turn-specific instruction, not a behavior preference).
_BEHAVIOR_ADJECTIVES = (
    "concise|brief|terse|short|tight|punchy|direct|"
    "verbose|wordy|chatty|long-winded|long winded|"
    "conversational|warm|friendly|casual|"
    "formal|professional|polite|"
    "blunt|harsh|snarky|sarcastic|witty|funny|playful|"
    "patient|gentle|stern|"
    "specific|detailed|thorough|"
    "careful|cautious"
)


_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ── Correction (highest priority — "actually" can prefix other shapes) ──
    (
        re.compile(r"^actually[,\s]+(.+)$", re.I),
        "correction",
        "Noted — I'll factor that in.",
    ),
    (
        re.compile(r"\b(?:that(?:'s| is| was)|you got that)\s+wrong\b", re.I),
        "correction",
        "Got it — I'll correct that.",
    ),
    (
        re.compile(r"\bI meant\s+([^.!?]+)", re.I),
        "correction",
        "Got it.",
    ),

    # ── Preference: "be (more|less) X" ─────────────────────────────────────
    (
        re.compile(rf"\bbe\s+more\s+({_BEHAVIOR_ADJECTIVES})\b", re.I),
        "preference",
        "Got it — I'll be more {match}.",
    ),
    (
        re.compile(rf"\bbe\s+less\s+({_BEHAVIOR_ADJECTIVES})\b", re.I),
        "preference",
        "Got it — I'll dial back on the {match}.",
    ),
    (
        re.compile(rf"\bdon'?t\s+be\s+(?:so|too|that)?\s*({_BEHAVIOR_ADJECTIVES})\b", re.I),
        "preference",
        "Got it — I'll dial back on the {match}.",
    ),
    (
        re.compile(rf"\bstop\s+being\s+(?:so\s+)?({_BEHAVIOR_ADJECTIVES})\b", re.I),
        "preference",
        "Got it — I'll dial that back.",
    ),

    # ── Preference: "keep (it|things|responses) (brief|...)" ───────────────
    (
        re.compile(
            rf"\bkeep\s+(?:it|things|responses|answers|replies|stuff)\s+({_BEHAVIOR_ADJECTIVES})\b",
            re.I,
        ),
        "preference",
        "Got it — I'll keep things {match}.",
    ),

    # ── Preference: explicit "I prefer / I want you to" ────────────────────
    (
        re.compile(r"\bI(?:'d| would)?\s+prefer\s+(?:that\s+you|you|it\s+if\s+you)\s+(.+?)(?:[.!?]|$)", re.I),
        "preference",
        "Got it — preference noted.",
    ),
    (
        re.compile(r"\bI\s+want\s+you\s+to\s+(.+?)(?:[.!?]|$)", re.I),
        "preference",
        "Got it — I'll do that going forward.",
    ),

    # ── Constraint: "always X" / "never X" / "from now on X" ───────────────
    # Anchored at start-of-clause to reduce false positives like
    # "I always go to the store on Sundays."
    (
        re.compile(r"(?:^|[.!?]\s+|\b(?:and|but)\s+)always\s+(.+?)(?:[.!?]|$)", re.I),
        "constraint",
        "Got it — always.",
    ),
    (
        re.compile(r"(?:^|[.!?]\s+|\b(?:and|but)\s+)never\s+(.+?)(?:[.!?]|$)", re.I),
        "constraint",
        "Got it — never.",
    ),
    (
        re.compile(r"\bfrom\s+now\s+on[,\s]+(.+?)(?:[.!?]|$)", re.I),
        "preference",
        "Got it — going forward.",
    ),
    (
        re.compile(r"\bgoing\s+forward[,\s]+(.+?)(?:[.!?]|$)", re.I),
        "preference",
        "Got it — going forward.",
    ),

    # ── Constraint: "ask before X" / "confirm before X" ────────────────────
    (
        re.compile(r"\b(?:ask|confirm)\s+(?:me\s+)?before\s+(.+?)(?:[.!?]|$)", re.I),
        "constraint",
        "Got it — I'll confirm first.",
    ),
    (
        re.compile(r"\bdon'?t\s+(.+?)\s+without\s+(?:asking|confirming|checking)\b", re.I),
        "constraint",
        "Got it — I'll confirm first.",
    ),
]


# Scope-hint patterns. Run BEFORE pattern matching above to decide
# whether to override the default scope="global". First hit wins;
# overlapping hints (route + domain) shouldn't appear in practice.
_ROUTE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:in|during|while in)\s+classroom\b", re.I), "classroom"),
    (re.compile(r"\b(?:in|during)\s+code\s+reviews?\b", re.I), "review"),
    (re.compile(r"\bwhen\s+teaching\b", re.I), "classroom"),
]

_DOMAIN_HINTS: list[tuple[re.Pattern, str | None]] = [
    # The second value is the domain key. ``None`` means "infer from
    # context the caller passes in (current_domain)" — for v1 we just
    # mark the scope=domain and use the explicit current_domain.
    (re.compile(r"\b(?:in|for)\s+this\s+(?:repo|repository|codebase|project)\b", re.I), None),
]


# ---------------------------------------------------------------------------
# Capture controller
# ---------------------------------------------------------------------------


class FeedbackCapture:
    """Watches turns for feedback signals; writes hits to the
    behavior store; returns an acknowledgment response when a capture
    fires."""

    def __init__(self, store: BehaviorMemoryStore):
        self.store = store

    def maybe_capture(
        self,
        user_input: str,
        *,
        identity: str | None = None,
        active_route: str | None = None,
        current_domain: str | None = None,
        last_response: str | None = None,
    ) -> Optional[dict]:
        """Inspect ``user_input``; if it's feedback, write to the store
        and return a response dict with the acknowledgment. Returns None
        if no feedback signal detected — runtime continues normal
        dispatch.

        ``last_response`` becomes the source_context if the capture is
        a correction (so consolidation can later see what the user was
        reacting to)."""
        if not user_input:
            return None
        text = user_input.strip()
        if not text:
            return None

        # Scope hints prefix the directive in natural usage ("in
        # classroom, always be patient" / "in this repo always use
        # pytest"). Extract the hint first and run pattern matching
        # against the residual — cleaner than encoding every possible
        # scope prefix into the anchor of every constraint pattern.
        scope, route_at_capture, domain, residual = self._extract_scope(
            text, current_domain=current_domain,
        )

        match_result = self._match(residual)
        if match_result is None:
            return None
        pattern, category, ack_template, match = match_result

        # Render the ack. Pattern groups can be substituted via {match}
        # for ack templates that want to echo the captured term.
        first_group = ""
        try:
            if match.lastindex:
                first_group = (match.group(1) or "").strip()
        except (IndexError, AttributeError):
            first_group = ""
        ack = ack_template.format(match=first_group) if "{match}" in ack_template else ack_template

        # Source context = the response the user was reacting to, for
        # corrections. For preferences/constraints it's the user's own
        # words (useful when consolidating "do these two notes contradict?").
        source_context = (last_response or "")[:300] if category == "correction" else text[:300]

        write_result = self.store.add(
            content=text,
            identity=identity,
            scope=scope,
            route_at_capture=route_at_capture or "",
            domain=domain or "",
            category=category,
            source="explicit",
            source_context=source_context,
            confidence=1.0,
        )

        if not write_result.get("ok"):
            log.warning("feedback_capture store.add failed: %s", write_result)
            return None

        return {
            "mode": "direct",
            "message": ack,
            "feedback": {
                "event": "captured",
                "category": category,
                "scope": scope,
                "duplicate": bool(write_result.get("duplicate")),
                "note_id": write_result.get("record", {}).get("id"),
            },
        }

    # ------------------------------------------------------------------

    def _match(self, text: str):
        """First-hit-wins pattern matching. Returns (pattern, category,
        ack_template, match) or None."""
        for pattern, category, ack in _PATTERNS:
            m = pattern.search(text)
            if m:
                return pattern, category, ack, m
        return None

    def _extract_scope(
        self,
        text: str,
        *,
        current_domain: str | None,
    ) -> tuple[str, str | None, str | None, str]:
        """Find a scope-hint prefix in ``text`` and strip it; return
        (scope, route_at_capture, domain, residual_text).

        Default: scope='global', residual=text (unchanged). Explicit
        hints override:
          - route hint ("in classroom", "during code reviews") → scope='route'
          - domain hint ("in this repo", "for this codebase") → scope='domain'

        If a domain hint matches but the caller didn't pass in
        ``current_domain``, scope stays 'global' — we don't have a
        domain key to stamp on the note, so per-repo isn't enforceable
        — but the hint phrase is still stripped from the residual so
        pattern matching works on the directive.
        """
        for pat, route_key in _ROUTE_HINTS:
            m = pat.search(text)
            if m:
                residual = _strip_match_and_trailing_comma(text, m)
                # Prefer the hint's explicit route key over the
                # active_route — the user is telling us what scope they
                # mean, not necessarily where they are right now.
                return "route", route_key, None, residual
        for pat, _ in _DOMAIN_HINTS:
            m = pat.search(text)
            if m:
                residual = _strip_match_and_trailing_comma(text, m)
                if current_domain:
                    return "domain", None, current_domain, residual
                # Domain hint without a known current_domain — best we
                # can do is global. Future: log this and ask the user.
                # Residual is still stripped so pattern matching can
                # find the directive.
                return "global", None, None, residual
        return "global", None, None, text


def _strip_match_and_trailing_comma(text: str, match: re.Match) -> str:
    """Remove a matched scope-hint span from ``text`` and clean up any
    immediately-following comma + whitespace, so the residual reads
    like a standalone directive. ``"in classroom, always be patient"``
    → ``"always be patient"``."""
    start, end = match.span()
    after = text[end:]
    # Eat one optional comma and any whitespace right after the hint.
    after = re.sub(r"^\s*,?\s*", "", after)
    before = text[:start].rstrip()
    if before and after:
        residual = f"{before} {after}"
    else:
        residual = before or after
    return residual.strip()
