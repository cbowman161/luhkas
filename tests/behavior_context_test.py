#!/usr/bin/env python3
"""Tests for vault/behavior_context.py — retrieve+format+apply for the
behavior-feedback loop.

Includes an end-to-end test that goes capture → store → build →
compose-inject to prove the pipeline is live as a unit (no model call
— uses a fake model that echoes the prompt back so we can inspect
what the composer actually got)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

try:
    import lancedb  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest("lancedb not installed") from exc

from behavior_context import (  # noqa: E402
    BehaviorContext,
    BehaviorContextBuilder,
    _format_preamble,
)
from feedback_capture import FeedbackCapture  # noqa: E402
from response_composer import ResponseComposer  # noqa: E402
from storage.behavior_store import BehaviorMemoryStore, EMBED_DIM  # noqa: E402


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for i, ch in enumerate((text or "x").encode("utf-8")):
            vec[i % EMBED_DIM] += float(ch)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def _make_store() -> tuple[BehaviorMemoryStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = BehaviorMemoryStore(
        embedder=_FakeEmbedder(),
        path=Path(tmp.name) / "memory.lance",
    )
    return store, tmp


# ---------------------------------------------------------------------------
# BehaviorContext dataclass
# ---------------------------------------------------------------------------


class ContextDataclass(unittest.TestCase):
    def test_empty_context_is_falsy(self):
        self.assertFalse(bool(BehaviorContext()))
        # text without note_ids is also falsy.
        self.assertFalse(bool(BehaviorContext(text="anything", note_ids=[])))

    def test_populated_context_is_truthy(self):
        self.assertTrue(bool(BehaviorContext(text="x", note_ids=["a"])))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class BuilderRetrieve(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()
        self.builder = BehaviorContextBuilder(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_store_returns_empty_context(self):
        ctx = self.builder.build("anything")
        self.assertFalse(bool(ctx))

    def test_empty_query_returns_empty_context(self):
        self.store.add("be more concise")
        self.assertFalse(bool(self.builder.build("")))
        self.assertFalse(bool(self.builder.build("   ")))

    def test_global_notes_returned_to_any_identity(self):
        self.store.add("be more concise")  # global
        ctx = self.builder.build("be more concise", identity="alex")
        self.assertTrue(bool(ctx))
        self.assertIn("be more concise", ctx.text)
        self.assertEqual(len(ctx.note_ids), 1)

    def test_identity_scoped_notes(self):
        self.store.add("be polite", identity="alex")
        self.store.add("be polite", identity="bob")
        ctx = self.builder.build("be polite", identity="alex")
        # Alex sees only the alex note (no global since neither was global).
        self.assertEqual(len(ctx.note_ids), 1)
        # The text contains the bullet.
        self.assertIn("be polite", ctx.text)

    def test_route_scoped_note_visible_in_matching_route(self):
        self.store.add(
            "be patient",
            scope="route",
            route_at_capture="classroom",
        )
        # Active route matches → visible.
        ctx_match = self.builder.build("be patient", active_route="classroom")
        self.assertTrue(bool(ctx_match))
        # No route given → hidden.
        ctx_none = self.builder.build("be patient")
        self.assertFalse(bool(ctx_none))


class BuilderApply(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()
        self.builder = BehaviorContextBuilder(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_apply_bumps_apply_count(self):
        self.store.add("be more concise")
        ctx = self.builder.build("be more concise")
        self.assertEqual(len(ctx.note_ids), 1)
        self.builder.apply(ctx)
        listed = self.store.list_for_identity(None)
        self.assertEqual(listed[0]["apply_count"], 1)
        # Re-apply accumulates.
        self.builder.apply(ctx)
        listed = self.store.list_for_identity(None)
        self.assertEqual(listed[0]["apply_count"], 2)

    def test_apply_empty_context_is_no_op(self):
        # Doesn't raise, doesn't bump anything.
        self.builder.apply(BehaviorContext())


# ---------------------------------------------------------------------------
# Preamble formatting
# ---------------------------------------------------------------------------


class Preamble(unittest.TestCase):
    def test_constraints_come_first(self):
        notes = [
            {"category": "preference", "content": "be concise", "id": "a"},
            {"category": "constraint", "content": "always confirm", "id": "b"},
            {"category": "preference", "content": "warm tone", "id": "c"},
        ]
        text = _format_preamble(notes)
        a_idx = text.index("be concise")
        b_idx = text.index("always confirm")
        # Constraint is rendered before any preference.
        self.assertLess(b_idx, a_idx)

    def test_high_apply_count_first_within_category(self):
        notes = [
            {"category": "preference", "content": "fresh", "id": "a",
             "apply_count": 0, "updated_at": 1.0},
            {"category": "preference", "content": "proven", "id": "b",
             "apply_count": 10, "updated_at": 0.0},
        ]
        text = _format_preamble(notes)
        self.assertLess(text.index("proven"), text.index("fresh"))

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(_format_preamble([]), "")
        # All-blank content also drops out.
        self.assertEqual(
            _format_preamble([{"category": "preference", "content": ""}]),
            "",
        )

    def test_preamble_has_header_and_bullets(self):
        notes = [{"category": "preference", "content": "be concise", "id": "a"}]
        text = _format_preamble(notes)
        self.assertIn("User preferences:", text)
        self.assertIn("- be concise", text)


# ---------------------------------------------------------------------------
# ResponseComposer injection
# ---------------------------------------------------------------------------


class _EchoModel:
    """Returns the prompt verbatim so tests can inspect injection."""
    def __init__(self):
        self.last_prompt: str | None = None

    def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return "model said something"


class ComposerInjection(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()
        self.builder = BehaviorContextBuilder(self.store)
        self.model = _EchoModel()
        self.composer = ResponseComposer(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_compose_without_behavior_context_leaves_prompt_clean(self):
        result = self.composer.compose(
            response_type="general",
            user_message="what's up",
            facts={},
            fallback="oops",
        )
        self.assertEqual(result, "model said something")
        self.assertNotIn("User preferences:", self.model.last_prompt)

    def test_compose_with_behavior_context_injects_preamble(self):
        self.store.add("be more concise")
        ctx = self.builder.build("be more concise")
        self.assertTrue(bool(ctx))
        result = self.composer.compose(
            response_type="general",
            user_message="what's up",
            facts={},
            fallback="oops",
            behavior_context=ctx,
        )
        self.assertEqual(result, "model said something")
        self.assertIn("User preferences:", self.model.last_prompt)
        self.assertIn("- be more concise", self.model.last_prompt)

    def test_apply_count_bumped_on_successful_compose(self):
        self.store.add("be more concise")
        ctx = self.builder.build("be more concise")
        self.composer.compose(
            response_type="general",
            user_message="hi",
            facts={},
            fallback="oops",
            behavior_context=ctx,
            behavior_apply=self.builder.apply,
        )
        listed = self.store.list_for_identity(None)
        self.assertEqual(listed[0]["apply_count"], 1)

    def test_apply_count_not_bumped_on_fallback(self):
        # Validator forces fallback — apply_count must stay 0.
        self.store.add("be more concise")
        ctx = self.builder.build("be more concise")
        self.composer.compose(
            response_type="general",
            user_message="hi",
            facts={},
            fallback="oops",
            behavior_context=ctx,
            behavior_apply=self.builder.apply,
            validator=lambda text: "rejected",
        )
        listed = self.store.list_for_identity(None)
        self.assertEqual(listed[0]["apply_count"], 0)


# ---------------------------------------------------------------------------
# End-to-end: capture → store → build → compose-inject
# ---------------------------------------------------------------------------


class EndToEnd(unittest.TestCase):
    """Black-box test of the full pipeline as wired in VaultRuntime.
    Walks the user through: (1) tells the system to be more concise,
    (2) asks a follow-up question, (3) verifies the preference landed
    in the next model prompt."""

    def test_capture_then_inject(self):
        store, tmp = _make_store()
        try:
            capture = FeedbackCapture(store)
            builder = BehaviorContextBuilder(store)
            model = _EchoModel()
            composer = ResponseComposer(model)

            # Turn 1: user tells system to be more concise.
            cap_resp = capture.maybe_capture("be more concise")
            self.assertIsNotNone(cap_resp)
            self.assertEqual(cap_resp["feedback"]["category"], "preference")

            # Turn 2: user asks a question. Builder + composer plug
            # the captured preference into the prompt.
            ctx = builder.build("what time is it")
            self.assertTrue(bool(ctx))
            composer.compose(
                response_type="general",
                user_message="what time is it",
                facts={"time": "noon"},
                fallback="not sure",
                behavior_context=ctx,
                behavior_apply=builder.apply,
            )
            # The preference made it into the model prompt.
            self.assertIn("User preferences:", model.last_prompt)
            self.assertIn("be more concise", model.last_prompt)

            # And the note's apply_count reflects the use.
            listed = store.list_for_identity(None)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["apply_count"], 1)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
