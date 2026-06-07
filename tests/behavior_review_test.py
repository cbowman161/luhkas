#!/usr/bin/env python3
"""Tests for vault/behavior_review.py — verbal review intent +
conflict-resolution flow via the pending-decision UX."""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

try:
    import lancedb  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest("lancedb not installed") from exc

from behavior_consolidator import BehaviorConsolidator, ConflictPair  # noqa: E402
from behavior_review import (  # noqa: E402
    BehaviorReviewController,
    PENDING_TYPE,
    classify_review_intent,
    classify_resolution,
)
from storage.behavior_store import BehaviorMemoryStore, EMBED_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Embedder:
    """Controlled embedder with set_basis / set_mix — reused from the
    consolidator tests' pattern. Pins specific texts to specific
    vectors so distance is predictable."""

    def __init__(self):
        self.m: dict[str, list[float]] = {}

    def set_basis(self, text: str, axis: int):
        v = [0.0] * EMBED_DIM
        v[axis % EMBED_DIM] = 1.0
        self.m[text] = v

    def set_mix(self, text: str, *components):
        v = [0.0] * EMBED_DIM
        for axis, weight in components:
            v[axis % EMBED_DIM] += float(weight)
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        self.m[text] = [x / norm for x in v]

    def embed(self, text: str) -> list[float]:
        if text in self.m:
            return list(self.m[text])
        v = [0.0] * EMBED_DIM
        v[(hash(text) + 5000) % EMBED_DIM] = 1.0
        return v


class _ScriptedModel:
    """Returns predetermined verdicts in order."""
    def __init__(self, responses: list[str]):
        self._r = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        if not self._r:
            raise RuntimeError("scripted model out of responses")
        return self._r.pop(0)


@dataclass
class _FakeProfile:
    primary: str | None = "alex"
    enrolled: set[str] = field(default_factory=lambda: {"alex"})

    def set_display_name(self, identity, name): pass
    def set_preference(self, identity, key, value): pass
    def designate_primary_user(self, identity): self.primary = identity
    def unset_primary_user(self): self.primary = None
    def primary_user(self): return self.primary
    def has_profile(self, identity): return identity in self.enrolled


@dataclass
class _FakeTTS:
    spoken: list[tuple[str, str]] = field(default_factory=list)
    def say(self, node_id, text):
        self.spoken.append((node_id, text))


class _PendingStore:
    """Single-slot per-node pending store. Mimics the runtime's
    _node_pendings without bringing in the full runtime."""

    def __init__(self):
        self.state: dict[str, dict] = {}

    def set(self, node_id, pending):
        self.state[node_id] = pending

    def get(self, node_id):
        return self.state.get(node_id)

    def clear(self, node_id):
        self.state.pop(node_id, None)


def _setup_with_conflicts(*, verifier_responses: list[str] | None = None):
    """Build a store with two conflict-band notes pinned, plus a
    review controller wired against it. Returns (controller,
    components) so tests can poke each piece."""
    emb = _Embedder()
    # cos = 0.78 → squared L2 = 0.44 → inside conflict band 0.25..0.65
    emb.set_basis("concise", axis=0)
    emb.set_mix("verbose", (0, 0.78), (1, 0.626))
    tmp = tempfile.TemporaryDirectory()
    store = BehaviorMemoryStore(embedder=emb, path=Path(tmp.name) / "memory.lance")
    a = store.add("concise", identity="alex")["record"]
    b = store.add("verbose", identity="alex")["record"]
    model = _ScriptedModel(verifier_responses or [
        '{"contradicts": true, "reason": "brief vs verbose"}',
    ])
    consolidator = BehaviorConsolidator(store, model=model)
    profile = _FakeProfile(primary="alex")
    tts = _FakeTTS()
    pending = _PendingStore()
    controller = BehaviorReviewController(
        consolidator=consolidator,
        store=store,
        profile_store=profile,
        tts=tts,
        pending_set=pending.set,
        pending_get=pending.get,
        pending_clear=pending.clear,
    )
    return controller, dict(
        store=store, tts=tts, pending=pending, profile=profile,
        model=model, note_a=a, note_b=b, tmp=tmp,
    )


# ---------------------------------------------------------------------------
# Intent classifiers
# ---------------------------------------------------------------------------


class ReviewIntent(unittest.TestCase):
    def test_recognized_phrases(self):
        for phrase in [
            "review my preferences",
            "Review the preferences",
            "consolidate preferences",
            "audit preferences",
            "check for conflicting preferences",
            "find conflicts in my preferences",
            "review behavior notes",
        ]:
            self.assertTrue(classify_review_intent(phrase), phrase)

    def test_unrelated_input_not_recognized(self):
        for phrase in ["", "hi", "what's up", "review the code", "be more concise"]:
            self.assertFalse(classify_review_intent(phrase), phrase)


class ResolutionIntent(unittest.TestCase):
    def test_first(self):
        for phrase in ["first", "the first", "first one", "keep the first", "A", "keep A"]:
            self.assertEqual(classify_resolution(phrase), "first", phrase)

    def test_second(self):
        for phrase in ["second", "the second", "keep B", "B", "second one"]:
            self.assertEqual(classify_resolution(phrase), "second", phrase)

    def test_both(self):
        for phrase in ["both", "keep both", "leave them alone", "don't change"]:
            self.assertEqual(classify_resolution(phrase), "both", phrase)

    def test_neither(self):
        for phrase in ["neither", "delete both", "remove both"]:
            self.assertEqual(classify_resolution(phrase), "neither", phrase)

    def test_cancel(self):
        for phrase in ["cancel", "never mind", "stop", "skip"]:
            self.assertEqual(classify_resolution(phrase), "cancel", phrase)

    def test_cancel_takes_priority_over_first(self):
        # "never mind, the first one" — cancel should win.
        self.assertEqual(classify_resolution("never mind, the first one"), "cancel")

    def test_ambiguous_returns_none(self):
        for phrase in ["", "uhh", "what", "this one"]:
            self.assertIsNone(classify_resolution(phrase), phrase)


# ---------------------------------------------------------------------------
# Review flow
# ---------------------------------------------------------------------------


class ReviewFlow(unittest.TestCase):
    def test_non_review_input_returns_none(self):
        controller, comp = _setup_with_conflicts()
        try:
            self.assertIsNone(controller.try_handle_review_intent("hi", "n"))
            # No pending state set.
            self.assertIsNone(comp["pending"].get("n"))
        finally:
            comp["tmp"].cleanup()

    def test_review_with_no_conflicts_replies_cleanly(self):
        controller, comp = _setup_with_conflicts(
            verifier_responses=['{"contradicts": false, "reason": "compatible"}'],
        )
        try:
            resp = controller.try_handle_review_intent("review my preferences", "n")
            self.assertEqual(resp["behavior_review"]["event"], "no_conflicts")
            self.assertIn("didn't find", resp["message"].lower())
            # No pending decision queued.
            self.assertIsNone(comp["pending"].get("n"))
        finally:
            comp["tmp"].cleanup()

    def test_review_with_conflict_surfaces_first(self):
        controller, comp = _setup_with_conflicts()
        try:
            resp = controller.try_handle_review_intent("review my preferences", "n")
            self.assertEqual(resp["behavior_review"]["event"], "surfacing")
            # Pending queued for n.
            pending = comp["pending"].get("n")
            self.assertEqual(pending["type"], PENDING_TYPE)
            self.assertEqual(pending["current_index"], 0)
            self.assertEqual(len(pending["queue"]), 1)
            # Both note contents in the surfacing message.
            self.assertIn("concise", resp["message"])
            self.assertIn("verbose", resp["message"])
        finally:
            comp["tmp"].cleanup()


class ResolveFlow(unittest.TestCase):
    def test_resolve_without_active_pending_returns_none(self):
        controller, comp = _setup_with_conflicts()
        try:
            # No review kicked off → no pending → resolve no-ops.
            self.assertIsNone(controller.resolve_conflict("first", "n"))
        finally:
            comp["tmp"].cleanup()

    def test_resolve_first_keeps_first_note_in_surfacing_message(self):
        # The "first" and "second" of the surfacing message depend on
        # list_for_identity iteration order (by updated_at desc). The
        # test verifies the *correct* note got kept, by reading the
        # pending queue's note_a_id (= the "first" the message named).
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            pending = comp["pending"].get("n")
            kept_id = pending["queue"][0]["note_a_id"]
            deleted_id = pending["queue"][0]["note_b_id"]
            controller.resolve_conflict("first", "n")
            remaining_ids = {n["id"] for n in comp["store"].list_for_identity("alex")}
            self.assertIn(kept_id, remaining_ids)
            self.assertNotIn(deleted_id, remaining_ids)
            self.assertIsNone(comp["pending"].get("n"))
        finally:
            comp["tmp"].cleanup()

    def test_resolve_second_keeps_second_note_in_surfacing_message(self):
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            pending = comp["pending"].get("n")
            deleted_id = pending["queue"][0]["note_a_id"]
            kept_id = pending["queue"][0]["note_b_id"]
            controller.resolve_conflict("second", "n")
            remaining_ids = {n["id"] for n in comp["store"].list_for_identity("alex")}
            self.assertIn(kept_id, remaining_ids)
            self.assertNotIn(deleted_id, remaining_ids)
        finally:
            comp["tmp"].cleanup()

    def test_resolve_both_keeps_everything(self):
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            resp = controller.resolve_conflict("both", "n")
            self.assertEqual(resp["behavior_review"]["event"], "resolved_last")
            # Neither note deleted.
            remaining = comp["store"].list_for_identity("alex")
            self.assertEqual(len(remaining), 2)
        finally:
            comp["tmp"].cleanup()

    def test_resolve_neither_deletes_both(self):
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            controller.resolve_conflict("neither", "n")
            remaining = comp["store"].list_for_identity("alex")
            self.assertEqual(remaining, [])
        finally:
            comp["tmp"].cleanup()

    def test_resolve_cancel_leaves_everything_clears_pending(self):
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            resp = controller.resolve_conflict("cancel", "n")
            self.assertEqual(resp["behavior_review"]["event"], "canceled")
            # Pending cleared, both notes still present.
            self.assertIsNone(comp["pending"].get("n"))
            remaining = comp["store"].list_for_identity("alex")
            self.assertEqual(len(remaining), 2)
        finally:
            comp["tmp"].cleanup()

    def test_resolve_ambiguous_re_asks(self):
        controller, comp = _setup_with_conflicts()
        try:
            controller.try_handle_review_intent("review my preferences", "n")
            resp = controller.resolve_conflict("uhh", "n")
            self.assertEqual(resp["behavior_review"]["event"], "ambiguous")
            # Pending still active.
            self.assertIsNotNone(comp["pending"].get("n"))
        finally:
            comp["tmp"].cleanup()


class MultiConflictQueue(unittest.TestCase):
    def test_queue_advances_through_multiple_conflicts(self):
        # Pin 3 notes so each pair lands in the 0.25..0.65 conflict
        # band (squared L2). cos=0.85 between unit vectors of form
        # (0.85, 0.527·axis-N) gives pairwise cos=0.7225 → squared L2
        # =0.555 (inside band). (0.95, 0.31·axis-N) was too close to
        # axis 0 → cos=0.904 → squared L2=0.19 → duplicate-guarded.
        emb = _Embedder()
        emb.set_mix("a", (0, 0.85), (1, 0.527))
        emb.set_mix("b", (0, 0.85), (2, 0.527))
        emb.set_mix("c", (0, 0.85), (3, 0.527))
        tmp = tempfile.TemporaryDirectory()
        try:
            store = BehaviorMemoryStore(
                embedder=emb, path=Path(tmp.name) / "memory.lance",
            )
            store.add("a", identity="alex")
            store.add("b", identity="alex")
            store.add("c", identity="alex")
            # All pairs verified as contradicting.
            model = _ScriptedModel(
                ['{"contradicts": true, "reason": "..."}'] * 10
            )
            consolidator = BehaviorConsolidator(store, model=model)
            profile = _FakeProfile(primary="alex")
            tts = _FakeTTS()
            pending = _PendingStore()
            controller = BehaviorReviewController(
                consolidator=consolidator,
                store=store,
                profile_store=profile,
                tts=tts,
                pending_set=pending.set,
                pending_get=pending.get,
                pending_clear=pending.clear,
            )

            resp1 = controller.try_handle_review_intent("review my preferences", "n")
            self.assertEqual(resp1["behavior_review"]["event"], "surfacing")
            queued = resp1["behavior_review"]["queue_size"]
            self.assertGreaterEqual(queued, 2)

            # Answer the first; should advance to next in queue.
            resp2 = controller.resolve_conflict("both", "n")
            if queued > 1:
                self.assertEqual(resp2["behavior_review"]["event"], "surfacing")
                self.assertEqual(resp2["behavior_review"]["current_index"], 1)
            else:  # pragma: no cover
                self.assertEqual(resp2["behavior_review"]["event"], "resolved_last")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
