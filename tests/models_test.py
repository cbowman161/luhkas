from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vault"))
import models


class _FakeResponse:
    status_code = 200
    text = "OK"

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def json(self):
        return {"embeddings": self._embeddings}


class EmbeddingModelCacheTest(unittest.TestCase):
    def setUp(self):
        models.clear_embedding_cache()

    def tearDown(self):
        models.clear_embedding_cache()

    def test_same_string_reuses_embedding_across_instances(self):
        post = mock.Mock(return_value=_FakeResponse([[1.0, 2.0, 3.0]]))
        with mock.patch.object(models.requests, "post", post, create=True):
            first = models.EmbeddingModel("bge-m3").embed("same query")
            second = models.EmbeddingModel("bge-m3").embed("same query")

        self.assertEqual(first, [[1.0, 2.0, 3.0]])
        self.assertEqual(second, [[1.0, 2.0, 3.0]])
        self.assertEqual(post.call_count, 1)

    def test_batch_embedding_is_not_cached(self):
        post = mock.Mock(return_value=_FakeResponse([[1.0], [2.0]]))
        with mock.patch.object(models.requests, "post", post, create=True):
            models.EmbeddingModel("bge-m3").embed(["a", "b"])
            models.EmbeddingModel("bge-m3").embed(["a", "b"])

        self.assertEqual(post.call_count, 2)

    def test_cached_embeddings_are_returned_as_copies(self):
        post = mock.Mock(return_value=_FakeResponse([[1.0, 2.0, 3.0]]))
        with mock.patch.object(models.requests, "post", post, create=True):
            first = models.EmbeddingModel("bge-m3").embed("mutable query")
            first[0][0] = 99.0
            second = models.EmbeddingModel("bge-m3").embed("mutable query")

        self.assertEqual(second, [[1.0, 2.0, 3.0]])
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
