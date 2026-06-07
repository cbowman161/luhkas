#!/usr/bin/env python3
"""Tests for vault/storage/behavior_store.py.

Uses a deterministic fake embedder so distances are predictable and
tests don't need bge-m3 loaded. Each test gets its own tempdir-backed
LanceDB so they're isolated.
"""
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
    raise unittest.SkipTest("lancedb not installed; behavior store tests skipped") from exc

from storage.behavior_store import (  # noqa: E402
    BehaviorMemoryStore,
    GLOBAL_IDENTITY,
    EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Hashes the input string deterministically into the embedding
    space so identical text → identical vector (distance 0) and similar
    text → close vectors. Not semantically meaningful — designed to
    exercise the store's filter/rank plumbing, not the embedder."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        # Distribute the bytes of ``text`` across the dim, normalized.
        # Two strings that share a prefix will produce close vectors.
        vec = [0.0] * self.dim
        if not text:
            vec[0] = 1.0
            return vec
        for i, ch in enumerate(text.encode("utf-8")):
            slot = i % self.dim
            vec[slot] += float(ch)
        # L2-normalize so cosine distance behaves sensibly.
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def _make_store(tmp_path: Path) -> BehaviorMemoryStore:
    return BehaviorMemoryStore(embedder=FakeEmbedder(), path=tmp_path)


# ---------------------------------------------------------------------------
# Add / schema
# ---------------------------------------------------------------------------


class AddBasic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_writes_a_row(self):
        result = self.store.add("be more concise")
        self.assertTrue(result["ok"])
        self.assertFalse(result["duplicate"])
        self.assertEqual(self.store.count(), 1)
        rec = result["record"]
        self.assertEqual(rec["content"], "be more concise")
        self.assertEqual(rec["identity"], GLOBAL_IDENTITY)
        self.assertEqual(rec["scope"], "global")
        self.assertEqual(rec["category"], "preference")
        self.assertEqual(rec["source"], "explicit")
        self.assertEqual(rec["apply_count"], 0)
        self.assertGreater(rec["created_at"], 0)

    def test_empty_content_rejected(self):
        self.assertEqual(self.store.add("")["error"], "empty_content")
        self.assertEqual(self.store.add("   ")["error"], "empty_content")
        self.assertEqual(self.store.count(), 0)

    def test_invalid_enum_values_fall_back_to_defaults(self):
        # Bad values shouldn't crash a turn — they fall back to defaults
        # so the note still lands.
        result = self.store.add(
            "do the thing",
            scope="weird-value",
            category="nonsense",
            source="unknown",
        )
        self.assertTrue(result["ok"])
        rec = result["record"]
        self.assertEqual(rec["scope"], "global")
        self.assertEqual(rec["category"], "preference")
        self.assertEqual(rec["source"], "explicit")

    def test_identity_normalization(self):
        # None / empty / whitespace all collapse to GLOBAL_IDENTITY.
        for ident in (None, "", "   "):
            r = self.store.add(f"note for {ident!r}", identity=ident)
            self.assertEqual(r["record"]["identity"], GLOBAL_IDENTITY)
        # Real identities are lowercased + stripped.
        r = self.store.add("note for alex", identity="  Alex  ")
        self.assertEqual(r["record"]["identity"], "alex")


class DuplicateGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_content_returns_duplicate(self):
        r1 = self.store.add("be more concise")
        r2 = self.store.add("be more concise")
        self.assertTrue(r2["ok"])
        self.assertTrue(r2["duplicate"])
        self.assertEqual(self.store.count(), 1)
        # Both calls see the same id back.
        self.assertEqual(r1["record"]["id"], r2["record"]["id"])

    def test_duplicate_in_different_identity_writes_new_row(self):
        # 'alex' and 'bob' have separate (identity, scope) namespaces,
        # so the same content lands twice.
        self.store.add("be more concise", identity="alex")
        self.store.add("be more concise", identity="bob")
        self.assertEqual(self.store.count(), 2)

    def test_duplicate_in_different_scope_writes_new_row(self):
        self.store.add("use pytest", scope="global")
        self.store.add("use pytest", scope="domain", domain="this_repo")
        self.assertEqual(self.store.count(), 2)


# ---------------------------------------------------------------------------
# Retrieve / filtering
# ---------------------------------------------------------------------------


class RetrieveScope(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_global_notes_always_returned(self):
        self.store.add("be more concise", scope="global")
        results = self.store.retrieve("be more concise")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["scope"], "global")

    def test_route_scope_only_returned_for_matching_route(self):
        self.store.add(
            "in classroom be brief",
            scope="route",
            route_at_capture="classroom",
        )
        # No active_route → route-scoped note hidden.
        self.assertEqual(len(self.store.retrieve("brief")), 0)
        # Matching active_route → visible.
        results = self.store.retrieve("brief", active_route="classroom")
        self.assertEqual(len(results), 1)
        # Mismatched active_route → hidden.
        self.assertEqual(
            len(self.store.retrieve("brief", active_route="general")),
            0,
        )

    def test_domain_scope_only_returned_for_matching_domain(self):
        self.store.add("use pytest", scope="domain", domain="this_repo")
        self.assertEqual(len(self.store.retrieve("pytest")), 0)
        results = self.store.retrieve("pytest", domain="this_repo")
        self.assertEqual(len(results), 1)
        self.assertEqual(
            len(self.store.retrieve("pytest", domain="other_repo")),
            0,
        )

    def test_session_scope_never_returned(self):
        # Session-scoped notes are caller-managed; the store never
        # surfaces them through retrieve().
        self.store.add("just for now", scope="session")
        self.assertEqual(len(self.store.retrieve("just for now")), 0)
        self.assertEqual(
            len(self.store.retrieve("just for now", active_route="classroom")),
            0,
        )

    def test_global_plus_route_mix(self):
        self.store.add("always be polite")  # global
        self.store.add(
            "in code reviews be terse",
            scope="route",
            route_at_capture="review",
        )
        self.store.add(
            "in classroom be patient",
            scope="route",
            route_at_capture="classroom",
        )
        # Active route 'review' → see global + review note, NOT classroom.
        results = self.store.retrieve("be", active_route="review", top_k=10)
        contents = {r["content"] for r in results}
        self.assertEqual(
            contents,
            {"always be polite", "in code reviews be terse"},
        )


class RetrieveIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_isolation(self):
        # Same content under three identities; querying each gets a
        # mix of that identity's note + global notes (none here).
        self.store.add("X", identity="alex")
        self.store.add("X", identity="bob")
        # 'alex' query: alex's note only.
        results = self.store.retrieve("X", identity="alex")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["identity"], "alex")

    def test_global_notes_visible_to_specific_identity(self):
        self.store.add("be kind")  # global
        self.store.add("alex specific", identity="alex")
        # 'alex' query: alex notes + global notes.
        results = self.store.retrieve("be kind alex specific", identity="alex", top_k=10)
        idents = {r["identity"] for r in results}
        self.assertEqual(idents, {"alex", GLOBAL_IDENTITY})

    def test_global_identity_query_only_returns_global(self):
        self.store.add("be kind")
        self.store.add("alex specific", identity="alex")
        results = self.store.retrieve("be kind alex specific", identity=None, top_k=10)
        idents = {r["identity"] for r in results}
        self.assertEqual(idents, {GLOBAL_IDENTITY})


class RetrieveCategory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_filter_by_single_category(self):
        self.store.add("be brief", category="preference")
        self.store.add("never delete without confirming", category="constraint")
        prefs = self.store.retrieve(
            "be brief never delete", category="preference", top_k=10,
        )
        self.assertEqual([r["category"] for r in prefs], ["preference"])
        constraints = self.store.retrieve(
            "be brief never delete", category="constraint", top_k=10,
        )
        self.assertEqual([r["category"] for r in constraints], ["constraint"])

    def test_filter_by_category_list(self):
        self.store.add("be brief", category="preference")
        self.store.add("never delete", category="constraint")
        self.store.add("you said X wrong", category="correction")
        results = self.store.retrieve(
            "be brief never delete you said X",
            category=["preference", "constraint"],
            top_k=10,
        )
        cats = {r["category"] for r in results}
        self.assertEqual(cats, {"preference", "constraint"})


# ---------------------------------------------------------------------------
# Conflict candidates
# ---------------------------------------------------------------------------


class ConflictCandidates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_content_is_NOT_a_conflict_candidate(self):
        # Duplicates (distance < distance_min) get excluded — they're
        # the same statement, not contradictions.
        self.store.add("be more concise", identity="alex")
        candidates = self.store.find_conflict_candidates(
            "be more concise", identity="alex",
        )
        self.assertEqual(candidates, [])

    def test_unrelated_content_is_NOT_a_conflict_candidate(self):
        # Far-apart content (distance > distance_max) also excluded —
        # it's not about the same subject.
        self.store.add("be more concise", identity="alex")
        # 'zzzzzzz' will hash to a very different vector under FakeEmbedder.
        candidates = self.store.find_conflict_candidates(
            "zzzzzzzzzzzzzz", identity="alex",
        )
        self.assertEqual(candidates, [])

    def test_candidates_scoped_to_identity(self):
        self.store.add("be more concise", identity="alex")
        # Bob's namespace is empty → no candidates.
        self.assertEqual(
            self.store.find_conflict_candidates("be brief", identity="bob"),
            [],
        )


# ---------------------------------------------------------------------------
# Telemetry + lifecycle
# ---------------------------------------------------------------------------


class TelemetryAndLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(Path(self.tmp.name) / "memory.lance")

    def tearDown(self):
        self.tmp.cleanup()

    def test_update_apply_count(self):
        rec = self.store.add("be brief")["record"]
        self.assertEqual(rec["apply_count"], 0)
        ok = self.store.update_apply_count(rec["id"])
        self.assertTrue(ok)
        listed = self.store.list_for_identity(None)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["apply_count"], 1)
        # Multiple bumps accumulate.
        self.store.update_apply_count(rec["id"], delta=3)
        listed = self.store.list_for_identity(None)
        self.assertEqual(listed[0]["apply_count"], 4)

    def test_update_apply_count_unknown_id(self):
        self.assertFalse(self.store.update_apply_count("does-not-exist"))
        self.assertFalse(self.store.update_apply_count(""))

    def test_delete_by_id(self):
        rec = self.store.add("be brief")["record"]
        self.assertEqual(self.store.count(), 1)
        self.assertTrue(self.store.delete_by_id(rec["id"]))
        self.assertEqual(self.store.count(), 0)

    def test_duplicate_add_touches_updated_at(self):
        # Reinforcement signal: re-stating the same preference should
        # bump updated_at so consolidation sees recency.
        import time as _t
        rec1 = self.store.add("be brief")["record"]
        original_updated = rec1["updated_at"]
        _t.sleep(0.01)
        rec2 = self.store.add("be brief")["record"]
        self.assertTrue(rec2.get("id"))
        # Re-list and confirm updated_at moved forward.
        listed = self.store.list_for_identity(None)
        self.assertGreater(listed[0]["updated_at"], original_updated)

    def test_list_for_identity_includes_global(self):
        self.store.add("global note 1")
        self.store.add("alex note", identity="alex")
        self.store.add("bob note", identity="bob")
        listed = self.store.list_for_identity("alex")
        contents = {r["content"] for r in listed}
        self.assertEqual(contents, {"global note 1", "alex note"})


if __name__ == "__main__":
    unittest.main()
