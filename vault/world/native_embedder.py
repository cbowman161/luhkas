"""Native bge-m3 embedder on the local GPU.

Used for bulk ingest (Wikipedia, media corpora) where the per-call overhead
of going through Ollama dominates wall-time. Sentence-transformers batches
hundreds of inputs per forward pass on the 3090, giving 10-30× the Ollama
throughput for the same model.

The chat/runtime path keeps using Ollama (already loaded, keep-alive 24h);
this class is instantiated only by ingest tools."""
from __future__ import annotations

import os
from typing import Iterable


DEFAULT_MODEL = os.environ.get("WORLD_NATIVE_EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_BATCH = int(os.environ.get("WORLD_NATIVE_EMBED_BATCH", "64"))
DEFAULT_DEVICE = os.environ.get("WORLD_NATIVE_EMBED_DEVICE", "cuda")
DEFAULT_DTYPE = os.environ.get("WORLD_NATIVE_EMBED_DTYPE", "float16")  # fp16 halves memory


class NativeEmbedder:
    """Sentence-transformers wrapper that mirrors the `.embed(text)` API of
    the existing Ollama EmbeddingModel so callers can swap them in place."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH,
        dtype: str = DEFAULT_DTYPE,
    ) -> None:
        # Lazy imports so importing this module doesn't pull torch into
        # processes that don't need the GPU embedder.
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda" and dtype != "float32":
            self.model = self.model.half() if dtype == "float16" else self.model.to(torch_dtype)
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name

    def embed(self, text):
        """Match Ollama EmbeddingModel.embed shape: str → list[float],
        list[str] → list[list[float]]."""
        single = isinstance(text, str)
        items: list[str] = [text] if single else list(text)
        if not items:
            return [] if not single else []
        vectors = self.model.encode(
            items,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # bge-m3 best practice for cosine
        )
        result = [v.astype("float32").tolist() for v in vectors]
        return result[0] if single else result
