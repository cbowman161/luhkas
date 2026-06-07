#!/usr/bin/env python3
"""Tests for vault/feedback_capture.py."""
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
    raise unittest.SkipTest("lancedb not installed; feedback_capture tests skipped") from exc

from feedback_capture import FeedbackCapture  # noqa: E402
from storage.behavior_store import BehaviorMemoryStore, EMBED_DIM  # noqa: E402


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for i, ch in enumerate((text or "x").encode("utf-8")):
            vec[i % EMBED_DIM] += float(ch)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def _make_capture():
    tmp = tempfile.TemporaryDirectory()
    store = BehaviorMemoryStore(
        embedder=_FakeEmbedder(),
        path=Path(tmp.name) / "memory.lance",
    )
    return FeedbackCapture(store), store, tmp


class PreferenceCapture(unittest.TestCase):
    def test_be_more_concise(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("be more concise")
            self.assertIsNotNone(resp)
            self.assertEqual(resp["feedback"]["category"], "preference")
            self.assertEqual(resp["feedback"]["scope"], "global")
            self.assertIn("concise", resp["message"].lower())
            self.assertEqual(store.count(), 1)
        finally:
            tmp.cleanup()

    def test_be_less_verbose(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("Be less verbose please.")
            self.assertEqual(resp["feedback"]["category"], "preference")
            self.assertIn("verbose", resp["message"].lower())
        finally:
            tmp.cleanup()

    def test_dont_be_so_chatty(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("don't be so chatty")
            self.assertEqual(resp["feedback"]["category"], "preference")
        finally:
            tmp.cleanup()

    def test_keep_responses_brief(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("keep responses brief")
            self.assertEqual(resp["feedback"]["category"], "preference")
            self.assertIn("brief", resp["message"].lower())
        finally:
            tmp.cleanup()

    def test_i_prefer_you_to(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("I'd prefer you to summarize before drilling in.")
            self.assertEqual(resp["feedback"]["category"], "preference")
        finally:
            tmp.cleanup()

    def test_i_want_you_to(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("I want you to confirm the file list first.")
            self.assertEqual(resp["feedback"]["category"], "preference")
        finally:
            tmp.cleanup()


class ConstraintCapture(unittest.TestCase):
    def test_always_at_start(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("always confirm before deleting files")
            self.assertEqual(resp["feedback"]["category"], "constraint")
        finally:
            tmp.cleanup()

    def test_never_at_start(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("never run rm -rf without asking")
            self.assertEqual(resp["feedback"]["category"], "constraint")
        finally:
            tmp.cleanup()

    def test_ask_before(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("ask before installing packages")
            self.assertEqual(resp["feedback"]["category"], "constraint")
        finally:
            tmp.cleanup()

    def test_dont_X_without_asking(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("don't push to remote without asking")
            self.assertEqual(resp["feedback"]["category"], "constraint")
        finally:
            tmp.cleanup()


class CorrectionCapture(unittest.TestCase):
    def test_actually_prefix(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture(
                "actually, the function returns a list not a tuple",
                last_response="It returns a tuple.",
            )
            self.assertEqual(resp["feedback"]["category"], "correction")
            # Source context should hold the previous response.
            rec = store.list_for_identity(None)[0]
            self.assertEqual(rec["source_context"], "It returns a tuple.")
        finally:
            tmp.cleanup()

    def test_that_was_wrong(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("that was wrong")
            self.assertEqual(resp["feedback"]["category"], "correction")
        finally:
            tmp.cleanup()

    def test_i_meant(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("I meant the second file, not the first")
            self.assertEqual(resp["feedback"]["category"], "correction")
        finally:
            tmp.cleanup()


class ScopeInference(unittest.TestCase):
    def test_route_hint_classroom(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("in classroom, always be patient")
            self.assertEqual(resp["feedback"]["scope"], "route")
            rec = store.list_for_identity(None)[0]
            self.assertEqual(rec["route_at_capture"], "classroom")
        finally:
            tmp.cleanup()

    def test_route_hint_code_review(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture(
                "during code reviews be more direct"
            )
            self.assertEqual(resp["feedback"]["scope"], "route")
            rec = store.list_for_identity(None)[0]
            self.assertEqual(rec["route_at_capture"], "review")
        finally:
            tmp.cleanup()

    def test_domain_hint_with_known_domain(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture(
                "in this repo always use pytest",
                current_domain="luhkas",
            )
            self.assertEqual(resp["feedback"]["scope"], "domain")
            rec = store.list_for_identity(None)[0]
            self.assertEqual(rec["domain"], "luhkas")
        finally:
            tmp.cleanup()

    def test_domain_hint_without_known_domain_falls_back_to_global(self):
        cap, store, tmp = _make_capture()
        try:
            # Domain hint present but no current_domain passed →
            # can't stamp a domain key, so scope stays global.
            resp = cap.maybe_capture("in this repo always use pytest")
            self.assertEqual(resp["feedback"]["scope"], "global")
        finally:
            tmp.cleanup()

    def test_no_hint_defaults_global(self):
        cap, store, tmp = _make_capture()
        try:
            resp = cap.maybe_capture("be more brief")
            self.assertEqual(resp["feedback"]["scope"], "global")
        finally:
            tmp.cleanup()


class IdentityPropagation(unittest.TestCase):
    def test_identity_stamped_on_note(self):
        cap, store, tmp = _make_capture()
        try:
            cap.maybe_capture("be more brief", identity="alex")
            rec = store.list_for_identity("alex")[0]
            self.assertEqual(rec["identity"], "alex")
        finally:
            tmp.cleanup()


class FalsePositiveResistance(unittest.TestCase):
    """The capture layer must not eat normal turns. False positives
    are worse than misses — they make the system feel possessive of
    every utterance."""

    def test_question_not_captured(self):
        cap, _, tmp = _make_capture()
        try:
            self.assertIsNone(cap.maybe_capture("what's the weather?"))
            self.assertIsNone(cap.maybe_capture("how does this work"))
            self.assertIsNone(cap.maybe_capture("can you explain X"))
        finally:
            tmp.cleanup()

    def test_factual_statement_not_captured(self):
        cap, _, tmp = _make_capture()
        try:
            self.assertIsNone(cap.maybe_capture("the meeting is at 3pm"))
            self.assertIsNone(cap.maybe_capture("my favorite color is blue"))
            self.assertIsNone(cap.maybe_capture("I work at Acme"))
        finally:
            tmp.cleanup()

    def test_always_mid_sentence_factual(self):
        # "I always go to the store on Sundays" — 'always' isn't at a
        # clause boundary the way "always confirm before X" is.
        cap, _, tmp = _make_capture()
        try:
            # Our anchored regex requires start-of-input or [.!?]/and/but
            # before 'always'. A bare mid-sentence 'always' embedded in
            # subject+verb shouldn't match.
            self.assertIsNone(
                cap.maybe_capture("when I'm hungry I always grab a snack"),
            )
        finally:
            tmp.cleanup()

    def test_never_song_lyric_not_captured(self):
        # "never gonna give you up" — 'never' at start of clause; this
        # IS a false positive we accept. The capture writes a constraint
        # note that future consolidation would have to deal with. We
        # document this rather than fix at L0; the user can revoke.
        # But "I'm never giving up" embedded mid-sentence shouldn't.
        cap, _, tmp = _make_capture()
        try:
            self.assertIsNone(cap.maybe_capture("I'm never giving up on this project"))
        finally:
            tmp.cleanup()

    def test_be_X_without_behavior_adjective_not_captured(self):
        # "Be more careful with this query" — "careful" IS in the
        # adjective list, so it WOULD match. Add an out-of-vocabulary
        # test to confirm the adjective list gates capture.
        cap, _, tmp = _make_capture()
        try:
            self.assertIsNone(cap.maybe_capture("be more aggressive about it"))
            # "interesting" is not in the behavior adjective list.
            self.assertIsNone(cap.maybe_capture("be more interesting"))
        finally:
            tmp.cleanup()


class DuplicateBehavior(unittest.TestCase):
    def test_re_capturing_same_feedback_is_marked_duplicate(self):
        cap, store, tmp = _make_capture()
        try:
            r1 = cap.maybe_capture("be more concise")
            self.assertFalse(r1["feedback"]["duplicate"])
            r2 = cap.maybe_capture("be more concise")
            self.assertTrue(r2["feedback"]["duplicate"])
            # Only one row in the store.
            self.assertEqual(store.count(), 1)
        finally:
            tmp.cleanup()


class EmptyInput(unittest.TestCase):
    def test_empty_input_returns_none(self):
        cap, _, tmp = _make_capture()
        try:
            self.assertIsNone(cap.maybe_capture(""))
            self.assertIsNone(cap.maybe_capture("   "))
            self.assertIsNone(cap.maybe_capture(None))
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
