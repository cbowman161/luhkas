#!/usr/bin/env python3
"""Tests for vault/behavior_consolidator.py."""
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

from behavior_consolidator import (  # noqa: E402
    BehaviorConsolidator,
    ConflictPair,
    _parse_verdict,
    _pair_key,
    _verifier_prompt,
)
from storage.behavior_store import BehaviorMemoryStore, EMBED_DIM  # noqa: E402


class _ControlledEmbedder:
    """Lets tests pin specific texts to specific vectors so the
    distance between them is predictable. The byte-hash fake doesn't
    correlate distance with semantic similarity the way bge-m3 does —
    e.g., 'be more concise' vs 'be more verbose' lands as a duplicate
    (d=0.04) while 'be more concise' vs 'zzz...' lands in the conflict
    band (d=0.51). For consolidator tests we need control over which
    pairs land in the band, so we register vectors explicitly."""

    def __init__(self):
        self._mapping: dict[str, list[float]] = {}
        # Used as a salt for any unregistered strings — they get
        # unique vectors far from the registered ones.
        self._unknown_seed = 1000

    def set_basis(self, text: str, axis: int, weight: float = 1.0):
        """Place ``text`` at a single-axis unit vector. axis=0 →
        [1, 0, 0, ...]. Two texts on the same axis with weight 1 are
        duplicates (distance 0)."""
        v = [0.0] * EMBED_DIM
        v[axis % EMBED_DIM] = float(weight)
        self._normalize_and_store(text, v)

    def set_mix(self, text: str, *components: tuple[int, float]):
        """Place ``text`` as a linear combo of basis axes:
        ``set_mix("foo", (0, 0.8), (1, 0.6))`` → vector mostly along
        axis 0 with some axis-1 component."""
        v = [0.0] * EMBED_DIM
        for axis, weight in components:
            v[axis % EMBED_DIM] += float(weight)
        self._normalize_and_store(text, v)

    def _normalize_and_store(self, text: str, v: list[float]):
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        self._mapping[text] = [x / norm for x in v]

    def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        # Unregistered strings get unique, far-apart vectors so they
        # don't accidentally land near pinned ones.
        v = [0.0] * EMBED_DIM
        slot = (hash(text) + self._unknown_seed) % EMBED_DIM
        v[slot] = 1.0
        return v


class _ScriptedModel:
    """Returns predetermined responses in order. Records the prompts
    it was called with so tests can verify the verifier prompt shape.

    Each call consumes one entry from ``responses``; if the script is
    exhausted the model raises (so tests catch under-stubbed cases)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("scripted model out of responses")
        return self._responses.pop(0)


def _make_store(embedder=None):
    tmp = tempfile.TemporaryDirectory()
    store = BehaviorMemoryStore(
        embedder=embedder or _ControlledEmbedder(),
        path=Path(tmp.name) / "memory.lance",
    )
    return store, tmp


def _conflict_band_vectors():
    """Return an embedder configured so:
      - 'concise' and 'verbose' land at squared-L2 ≈ 0.44 (in the
        0.25..0.65 conflict band)
      - 'unrelated' lands far from both (squared-L2 ≈ 2.0)

    LanceDB returns squared L2 (not L2) as ``_distance``. For two
    unit vectors with cosine similarity c, squared L2 = 2 - 2c.
    Targeting d² = 0.44 → c = 0.78. Achieve cos=0.78 between [1,0]
    and [0.78, sqrt(1-0.78²)≈0.626].
    """
    emb = _ControlledEmbedder()
    emb.set_basis("concise", axis=0)
    emb.set_mix("verbose", (0, 0.78), (1, 0.626))
    emb.set_basis("unrelated", axis=50)
    return emb


# ---------------------------------------------------------------------------
# find_pairs — pre-filter, no LLM needed
# ---------------------------------------------------------------------------


class FindPairs(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()
        self.consolidator = BehaviorConsolidator(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_store_returns_no_pairs(self):
        self.assertEqual(self.consolidator.find_pairs(), [])

    def test_single_note_returns_no_pairs(self):
        self.store.add("be more concise")
        self.assertEqual(self.consolidator.find_pairs(), [])

    def test_distant_notes_NOT_paired(self):
        # Pin two notes orthogonal in embedding space → distance ≈ 1.41
        # → far outside the 0.25..0.65 conflict band → no pair.
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("unrelated")
            consolidator = BehaviorConsolidator(store)
            self.assertEqual(consolidator.find_pairs(), [])
        finally:
            tmp.cleanup()

    def test_close_notes_in_conflict_band_yield_pair(self):
        # Pin two notes at cos=0.9 → L2 ≈ 0.45 → inside the band.
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("verbose")
            consolidator = BehaviorConsolidator(store)
            pairs = consolidator.find_pairs()
            self.assertGreaterEqual(len(pairs), 1)
            ids = {p.note_a["id"] for p in pairs} | {p.note_b["id"] for p in pairs}
            self.assertEqual(len(ids), 2)
        finally:
            tmp.cleanup()

    def test_pair_is_deduplicated_across_directions(self):
        # Pre-filter scans each note as a query; without dedup the
        # same pair would appear twice (a→b and b→a).
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("verbose")
            consolidator = BehaviorConsolidator(store)
            pairs = consolidator.find_pairs()
            keys = {tuple(sorted([p.note_a["id"], p.note_b["id"]])) for p in pairs}
            self.assertEqual(len(pairs), len(keys))
        finally:
            tmp.cleanup()

    def test_identity_isolation(self):
        # alex's notes and bob's notes are separate namespaces; no
        # cross-identity pairs are produced even when vectors are close.
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise", identity="alex")
            store.add("verbose", identity="bob")
            consolidator = BehaviorConsolidator(store)
            self.assertEqual(consolidator.find_pairs(identity="alex"), [])
            self.assertEqual(consolidator.find_pairs(identity="bob"), [])
        finally:
            tmp.cleanup()

    def test_global_namespace_isolated_from_identity(self):
        # Even though list_for_identity('alex') includes global notes,
        # find_pairs filters to exact identity match so global-vs-alex
        # cross-namespace pairs aren't produced.
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")  # global
            store.add("verbose", identity="alex")
            consolidator = BehaviorConsolidator(store)
            self.assertEqual(consolidator.find_pairs(identity="alex"), [])
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# verify_pair — LLM-backed
# ---------------------------------------------------------------------------


class VerifyPair(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()

    def tearDown(self):
        self.tmp.cleanup()

    def _pair(self) -> ConflictPair:
        a = self.store.add("be more concise")["record"]
        b = self.store.add(
            "be more verbose",
            identity="alex",  # different namespace ok for this test
        )["record"]
        return ConflictPair(note_a=a, note_b=b, distance=0.4)

    def test_verify_without_model_is_no_op(self):
        consolidator = BehaviorConsolidator(self.store, model=None)
        pair = self._pair()
        verified = consolidator.verify_pair(pair)
        self.assertFalse(verified.verified)
        self.assertEqual(verified.reason, "")

    def test_verify_true_when_model_says_contradicts(self):
        model = _ScriptedModel(['{"contradicts": true, "reason": "A is brief, B is verbose"}'])
        consolidator = BehaviorConsolidator(self.store, model=model)
        verified = consolidator.verify_pair(self._pair())
        self.assertTrue(verified.verified)
        self.assertIn("brief", verified.reason)
        # Prompt was assembled with both contents.
        self.assertIn("be more concise", model.prompts[0])
        self.assertIn("be more verbose", model.prompts[0])

    def test_verify_false_when_model_says_no_contradiction(self):
        model = _ScriptedModel(['{"contradicts": false, "reason": "Compatible"}'])
        consolidator = BehaviorConsolidator(self.store, model=model)
        verified = consolidator.verify_pair(self._pair())
        self.assertFalse(verified.verified)
        self.assertIn("Compatible", verified.reason)

    def test_verify_tolerates_prose_around_json(self):
        # Small models sometimes wrap JSON in prose. The parser should
        # still extract the verdict from the first {...} block.
        model = _ScriptedModel([
            'Here is my analysis:\n{"contradicts": true, "reason": "yep"}\nThat is all.'
        ])
        consolidator = BehaviorConsolidator(self.store, model=model)
        verified = consolidator.verify_pair(self._pair())
        self.assertTrue(verified.verified)

    def test_verify_defaults_false_on_malformed_response(self):
        # Empty / garbage / non-JSON all default to "no contradiction"
        # — surfacing a false positive is worse than missing one.
        for bad in ["", "no", "the answer is yes", "{not json"]:
            with self.subTest(bad=bad):
                model = _ScriptedModel([bad])
                consolidator = BehaviorConsolidator(self.store, model=model)
                verified = consolidator.verify_pair(self._pair())
                self.assertFalse(verified.verified)

    def test_verify_swallows_model_exception(self):
        class _Boom:
            def generate(self, prompt, **kw):
                raise RuntimeError("model down")
        consolidator = BehaviorConsolidator(self.store, model=_Boom())
        verified = consolidator.verify_pair(self._pair())
        self.assertFalse(verified.verified)


# ---------------------------------------------------------------------------
# consolidate — top-level entry
# ---------------------------------------------------------------------------


class Consolidate(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()

    def tearDown(self):
        self.tmp.cleanup()

    def test_consolidate_returns_only_verified_contradictions(self):
        # Two notes pinned in the conflict band; verifier says they
        # contradict. consolidate() should return exactly that pair.
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("verbose")
            responses = ['{"contradicts": true, "reason": "brief vs verbose"}']
            consolidator = BehaviorConsolidator(store, model=_ScriptedModel(responses))
            confirmed = consolidator.consolidate()
            self.assertEqual(len(confirmed), 1)
            self.assertTrue(confirmed[0].verified)
            self.assertIn("brief", confirmed[0].reason)
        finally:
            tmp.cleanup()

    def test_consolidate_filters_out_false_verdicts(self):
        # Same setup but verifier says "not actually contradictory" —
        # consolidate() returns empty list (only verified-true pairs).
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("verbose")
            model = _ScriptedModel(['{"contradicts": false, "reason": "compatible"}'])
            consolidator = BehaviorConsolidator(store, model=model)
            confirmed = consolidator.consolidate()
            self.assertEqual(confirmed, [])
        finally:
            tmp.cleanup()

    def test_consolidate_without_verify_returns_all_pairs_unverified(self):
        store, tmp = _make_store(_conflict_band_vectors())
        try:
            store.add("concise")
            store.add("verbose")
            consolidator = BehaviorConsolidator(store, model=_ScriptedModel([]))
            # verify=False bypasses the LLM entirely.
            pairs = consolidator.consolidate(verify=False)
            self.assertGreaterEqual(len(pairs), 1)
            for pair in pairs:
                self.assertFalse(pair.verified)
        finally:
            tmp.cleanup()

    def test_consolidate_caps_at_max_pairs(self):
        # Stuff the store with enough close-vector notes to exceed
        # max_pairs. We pin 6 notes in a tight cluster around axis 0.
        emb = _ControlledEmbedder()
        for i in range(6):
            # Each note slightly off the axis-0 basis — pairwise
            # distances all in the conflict band.
            emb.set_mix(f"variant_{i}", (0, 0.95), (i + 1, 0.31))
        store, tmp = _make_store(emb)
        try:
            for i in range(6):
                store.add(f"variant_{i}")
            # Pad the script so we don't run out.
            model = _ScriptedModel(['{"contradicts": false, "reason": "x"}'] * 50)
            consolidator = BehaviorConsolidator(store, model=model)
            consolidator.consolidate(max_pairs=3)
            # Model called at most 3 times even with more pairs available.
            self.assertLessEqual(len(model.prompts), 3)
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Helpers(unittest.TestCase):
    def test_pair_key_is_unordered(self):
        self.assertEqual(_pair_key("a", "b"), _pair_key("b", "a"))

    def test_pair_key_handles_missing_ids(self):
        # Doesn't raise; just returns empty.
        self.assertEqual(_pair_key(None, "b"), ("", ""))
        self.assertEqual(_pair_key("a", None), ("", ""))

    def test_verifier_prompt_includes_both_statements(self):
        prompt = _verifier_prompt(
            {"content": "be brief"},
            {"content": "be verbose"},
        )
        self.assertIn("be brief", prompt)
        self.assertIn("be verbose", prompt)
        self.assertIn("contradicts", prompt.lower())

    def test_parse_verdict_extracts_first_json_block(self):
        self.assertEqual(
            _parse_verdict('{"contradicts": true, "reason": "x"}'),
            {"contradicts": True, "reason": "x"},
        )

    def test_parse_verdict_defaults_false_on_bad_input(self):
        self.assertFalse(_parse_verdict("")["contradicts"])
        self.assertFalse(_parse_verdict("nope")["contradicts"])
        self.assertFalse(_parse_verdict("{malformed")["contradicts"])


if __name__ == "__main__":
    unittest.main()
