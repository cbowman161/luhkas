#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))


class DeterministicEmbedder:
    """Hash-derived pseudo-vectors so two equal strings get equal vectors and
    distinct strings get distinct vectors. Avoids needing Ollama for tests."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if isinstance(text, list):
            return [self._one(t) for t in text]
        return self._one(text)

    def _one(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        vec = []
        i = 0
        while len(vec) < self.dim:
            vec.append(((h[i % len(h)] - 128) / 128.0))
            i += 1
        return vec


class WorldStoreRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        import os
        os.environ["WORLD_VAULT_ROOT"] = self.tmp.name
        os.environ["WORLD_DB_PATH"] = str(Path(self.tmp.name) / "world.lance")
        # Re-import config + store with fresh env
        for mod in [
            "world", "world.config", "world.world_store",
        ]:
            sys.modules.pop(mod, None)
        from world import WorldKnowledgeStore
        from world.config import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
        self.text_dim = TEXT_EMBED_DIM
        self.image_dim = IMAGE_EMBED_DIM
        self.store = WorldKnowledgeStore(
            text_embedder=DeterministicEmbedder(TEXT_EMBED_DIM),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_wiki_article_and_chunks_round_trip(self):
        embedder = DeterministicEmbedder(self.text_dim)
        self.store.upsert_wiki_article(
            article_id="enwiki:Mongolia",
            title="Mongolia",
            slug="Mongolia",
            revision="rev-1",
            lang="en",
        )
        chunks = [
            {
                "article_id": "enwiki:Mongolia",
                "title": "Mongolia",
                "section_path": "Lead",
                "chunk_idx": 0,
                "content": "Mongolia is a landlocked country in East Asia.",
                "content_hash": "h0",
                "vector": embedder.embed("Mongolia is a landlocked country in East Asia."),
            },
            {
                "article_id": "enwiki:Mongolia",
                "title": "Mongolia",
                "section_path": "Capital",
                "chunk_idx": 1,
                "content": "The capital of Mongolia is Ulaanbaatar.",
                "content_hash": "h1",
                "vector": embedder.embed("The capital of Mongolia is Ulaanbaatar."),
            },
        ]
        result = self.store.add_wiki_chunks(chunks)
        self.assertEqual(result["added"], 2)

        hits = self.store.search_wiki("The capital of Mongolia is Ulaanbaatar.", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        # Deterministic embedder => exact-match query returns that chunk first.
        self.assertEqual(hits[0]["content"], "The capital of Mongolia is Ulaanbaatar.")
        self.assertEqual(hits[0]["article_id"], "enwiki:Mongolia")
        self.assertNotIn("vector", hits[0])

    def test_wiki_replace_clears_old_chunks(self):
        embedder = DeterministicEmbedder(self.text_dim)
        self.store.upsert_wiki_article(article_id="a1", title="A")
        self.store.add_wiki_chunks([
            {"article_id": "a1", "title": "A", "section_path": "", "chunk_idx": 0,
             "content": "old text", "content_hash": "old", "vector": embedder.embed("old text")},
        ])
        self.store.replace_wiki_chunks_for_article("a1", [
            {"article_id": "a1", "title": "A", "section_path": "", "chunk_idx": 0,
             "content": "new text", "content_hash": "new", "vector": embedder.embed("new text")},
        ])
        hits = self.store.search_wiki("old text", top_k=3)
        contents = [h["content"] for h in hits]
        self.assertNotIn("old text", contents)
        self.assertIn("new text", contents)

    def test_media_asset_dedup_by_sha(self):
        first = self.store.add_media_asset(
            path="/tmp/cat.jpg", kind="image", sha256="abc123",
            mime="image/jpeg", width=640, height=480,
        )
        second = self.store.add_media_asset(
            path="/tmp/cat-duplicate.jpg", kind="image", sha256="abc123",
            mime="image/jpeg", width=640, height=480,
        )
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["asset_id"], second["asset_id"])

    def test_vector_search_reuses_precomputed_embedding(self):
        embedder = DeterministicEmbedder(self.text_dim)
        self.store.upsert_wiki_article(article_id="enwiki:Mongolia", title="Mongolia")
        self.store.add_wiki_chunks([
            {
                "article_id": "enwiki:Mongolia",
                "title": "Mongolia",
                "section_path": "Capital",
                "chunk_idx": 0,
                "content": "The capital of Mongolia is Ulaanbaatar.",
                "content_hash": "h1",
                "vector": embedder.embed("The capital of Mongolia is Ulaanbaatar."),
            },
        ])
        query_vec = embedder.embed("The capital of Mongolia is Ulaanbaatar.")
        calls_before = len(self.store.text_embedder.calls)

        hits = self.store.search_wiki_vector(query_vec, top_k=1)

        self.assertEqual(hits[0]["article_id"], "enwiki:Mongolia")
        self.assertEqual(len(self.store.text_embedder.calls), calls_before)

    def test_media_text_vector_search_reuses_precomputed_embedding(self):
        embedder = DeterministicEmbedder(self.text_dim)
        asset = self.store.add_media_asset(
            path="/tmp/talk.mp3", kind="audio", sha256="aud-vector",
            mime="audio/mpeg", duration_s=60.0,
        )
        self.store.add_media_text_chunks([
            {
                "asset_id": asset["asset_id"],
                "modality": "transcript",
                "start_s": 0.0,
                "end_s": 30.0,
                "content": "Hello world, this is a test recording.",
                "vector": embedder.embed("Hello world, this is a test recording."),
            },
        ])
        query_vec = embedder.embed("Hello world, this is a test recording.")
        calls_before = len(self.store.text_embedder.calls)

        hits = self.store.search_media_text_vector(query_vec, top_k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["asset_id"], asset["asset_id"])
        self.assertEqual(len(self.store.text_embedder.calls), calls_before)

    def test_media_text_chunk_search(self):
        embedder = DeterministicEmbedder(self.text_dim)
        asset = self.store.add_media_asset(
            path="/tmp/talk.mp3", kind="audio", sha256="aud1",
            mime="audio/mpeg", duration_s=60.0,
        )
        self.store.add_media_text_chunks([
            {
                "asset_id": asset["asset_id"],
                "modality": "transcript",
                "start_s": 0.0,
                "end_s": 30.0,
                "content": "Hello world, this is a test recording.",
                "vector": embedder.embed("Hello world, this is a test recording."),
            },
        ])
        hits = self.store.search_media_text("Hello world, this is a test recording.", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["modality"], "transcript")
        self.assertEqual(hits[0]["asset_id"], asset["asset_id"])

    def test_image_vector_search(self):
        asset = self.store.add_media_asset(
            path="/tmp/cat.jpg", kind="image", sha256="img-only",
            mime="image/jpeg",
        )
        embedder = DeterministicEmbedder(self.image_dim)
        vec_a = embedder.embed("cat photo")
        vec_b = embedder.embed("dog photo")
        self.store.add_media_image_vecs([
            {"asset_id": asset["asset_id"], "frame_idx": 0, "t_s": 0.0, "vector": vec_a},
            {"asset_id": asset["asset_id"], "frame_idx": 1, "t_s": 0.0, "vector": vec_b},
        ])
        hits = self.store.search_media_image(vec_a, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["frame_idx"], 0)
        self.assertNotIn("vector", hits[0])

    def test_stats_reports_counts_and_disk(self):
        stats = self.store.stats()
        self.assertIn("counts", stats)
        self.assertIn("disk", stats)
        for table in (
            "wiki_articles", "wiki_chunks", "media_assets",
            "media_text_chunks", "media_image_vecs",
        ):
            self.assertIn(table, stats["counts"])
        self.assertTrue(stats["disk"]["ok"])
        self.assertIn(stats["disk"]["level"], {"ok", "warn", "critical"})


if __name__ == "__main__":
    unittest.main()
