"""Identity-free reference corpus backed by LanceDB.

Holds:
- Wikipedia article metadata + chunked text with bge-m3 embeddings
- Media asset metadata (image/audio/video)
- Per-modality media text chunks (captions, OCR, transcripts) with bge-m3
- Per-image SigLIP visual embeddings

Phase 1 surface: schema + add/search/upsert/stats. Ingestion lives in
sibling modules added in Phase 2 (wiki) and Phase 3 (media)."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import lancedb
import pyarrow as pa

from world.config import (
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    WORLD_DB_PATH,
    WORLD_DISK_CRITICAL_PCT,
    WORLD_DISK_WARN_PCT,
    WORLD_VAULT_ROOT,
    ensure_dirs,
)


WIKI_ARTICLES_TABLE = "wiki_articles"
WIKI_CHUNKS_TABLE = "wiki_chunks"
MEDIA_ASSETS_TABLE = "media_assets"
MEDIA_TEXT_CHUNKS_TABLE = "media_text_chunks"
MEDIA_IMAGE_VECS_TABLE = "media_image_vecs"


def _wiki_articles_schema() -> pa.Schema:
    return pa.schema([
        pa.field("article_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("slug", pa.string()),
        pa.field("revision", pa.string()),
        pa.field("lang", pa.string()),
        pa.field("ingested_at", pa.float64()),
    ])


def _wiki_chunks_schema() -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("section_path", pa.string()),
        pa.field("chunk_idx", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), TEXT_EMBED_DIM)),
    ])


def _media_assets_schema() -> pa.Schema:
    return pa.schema([
        pa.field("asset_id", pa.string()),
        pa.field("path", pa.string()),
        pa.field("kind", pa.string()),         # image | audio | video
        pa.field("mime", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("captured_at", pa.float64()),  # 0.0 = unknown
        pa.field("ingested_at", pa.float64()),
        pa.field("caption", pa.string()),
        pa.field("ocr_text", pa.string()),
        pa.field("transcript", pa.string()),
        pa.field("duration_s", pa.float64()),   # 0.0 for stills
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
    ])


def _media_text_chunks_schema() -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("modality", pa.string()),     # caption | ocr | transcript
        pa.field("start_s", pa.float64()),      # 0.0 if non-temporal
        pa.field("end_s", pa.float64()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), TEXT_EMBED_DIM)),
    ])


def _media_image_vecs_schema() -> pa.Schema:
    return pa.schema([
        pa.field("asset_id", pa.string()),
        pa.field("frame_idx", pa.int32()),     # 0 for stills
        pa.field("t_s", pa.float64()),          # 0.0 for stills
        pa.field("vector", pa.list_(pa.float32(), IMAGE_EMBED_DIM)),
    ])


_TABLE_SCHEMAS: dict[str, pa.Schema] = {
    WIKI_ARTICLES_TABLE: _wiki_articles_schema(),
    WIKI_CHUNKS_TABLE: _wiki_chunks_schema(),
    MEDIA_ASSETS_TABLE: _media_assets_schema(),
    MEDIA_TEXT_CHUNKS_TABLE: _media_text_chunks_schema(),
    MEDIA_IMAGE_VECS_TABLE: _media_image_vecs_schema(),
}


def _sql_escape(value: str) -> str:
    return (value or "").replace("'", "''")


class WorldKnowledgeStore:
    def __init__(
        self,
        text_embedder=None,
        image_embedder=None,
        path: str | Path | None = None,
    ):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.path = Path(path or WORLD_DB_PATH)
        ensure_dirs()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.path))
        self._lock = threading.Lock()
        self._tables: dict[str, Any] = {}
        for name, schema in _TABLE_SCHEMAS.items():
            self._tables[name] = self._ensure_table(name, schema)

    def _ensure_table(self, name: str, schema: pa.Schema):
        try:
            return self.db.open_table(name)
        except (FileNotFoundError, ValueError):
            return self.db.create_table(name, schema=schema)

    def _embed_text(self, text: str) -> list[float]:
        if not self.text_embedder:
            raise RuntimeError("WorldKnowledgeStore.text_embedder is not configured")
        result = self.text_embedder.embed(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result[0]
        return result  # type: ignore[return-value]

    # --- wiki -------------------------------------------------------------

    def upsert_wiki_article(
        self,
        article_id: str,
        title: str,
        *,
        slug: str = "",
        revision: str = "",
        lang: str = "en",
    ) -> dict[str, Any]:
        record = {
            "article_id": article_id,
            "title": title,
            "slug": slug,
            "revision": revision,
            "lang": lang,
            "ingested_at": time.time(),
        }
        with self._lock:
            self._tables[WIKI_ARTICLES_TABLE].delete(
                f"article_id = '{_sql_escape(article_id)}'"
            )
            self._tables[WIKI_ARTICLES_TABLE].add([record])
        return {"ok": True, "record": record}

    def add_new_wiki_articles(
        self, articles: Iterable[dict[str, Any]], *, lang: str = "en",
    ) -> dict[str, Any]:
        """Batch-insert article markers known to be new (no existing
        article_id collision). Skips the per-row delete that
        upsert_wiki_article performs — 100× faster at ingest scale
        because each delete creates a new Lance dataset fragment."""
        rows = []
        now = time.time()
        for a in articles:
            rows.append({
                "article_id": a["article_id"],
                "title": a.get("title", ""),
                "slug": a.get("slug", ""),
                "revision": a.get("content_hash") or a.get("revision", ""),
                "lang": lang,
                "ingested_at": now,
            })
        if not rows:
            return {"ok": True, "added": 0}
        with self._lock:
            self._tables[WIKI_ARTICLES_TABLE].add(rows)
        return {"ok": True, "added": len(rows)}

    def delete_wiki_articles(self, article_ids: list[str]) -> int:
        """Batch-delete article markers by id (used on refresh when
        revisions change)."""
        if not article_ids:
            return 0
        # SQL IN-list, escaped.
        quoted = ", ".join(f"'{_sql_escape(a)}'" for a in article_ids)
        with self._lock:
            self._tables[WIKI_ARTICLES_TABLE].delete(
                f"article_id IN ({quoted})"
            )
        return len(article_ids)

    def add_wiki_chunks(self, chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Insert pre-embedded wiki chunks. Each chunk dict must include
        `article_id, title, section_path, chunk_idx, content, content_hash, vector`.
        `chunk_id` is generated if missing."""
        rows = []
        for c in chunks:
            row = {
                "chunk_id": c.get("chunk_id") or str(uuid.uuid4()),
                "article_id": c["article_id"],
                "title": c.get("title", ""),
                "section_path": c.get("section_path", ""),
                "chunk_idx": int(c.get("chunk_idx", 0)),
                "content": c["content"],
                "content_hash": c.get("content_hash", ""),
                "vector": c["vector"],
            }
            rows.append(row)
        if not rows:
            return {"ok": True, "added": 0}
        with self._lock:
            self._tables[WIKI_CHUNKS_TABLE].add(rows)
        return {"ok": True, "added": len(rows)}

    def replace_wiki_chunks_for_article(
        self, article_id: str, chunks: Iterable[dict[str, Any]]
    ) -> dict[str, Any]:
        """Atomic replace — used on refresh when an article's revision changed."""
        with self._lock:
            self._tables[WIKI_CHUNKS_TABLE].delete(
                f"article_id = '{_sql_escape(article_id)}'"
            )
        return self.add_wiki_chunks(chunks)

    def search_wiki(
        self,
        query: str,
        top_k: int = 5,
        distance_max: float | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        vec = self._embed_text(query)
        with self._lock:
            res = (
                self._tables[WIKI_CHUNKS_TABLE]
                .search(vec)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        return _project_hits(res, distance_max, drop=("vector",))

    # --- media ------------------------------------------------------------

    def get_media_asset_by_sha(self, sha256: str) -> dict[str, Any] | None:
        if not sha256:
            return None
        with self._lock:
            rows = (
                self._tables[MEDIA_ASSETS_TABLE]
                .search()
                .where(f"sha256 = '{_sql_escape(sha256)}'")
                .limit(1)
                .to_list()
            )
        return rows[0] if rows else None

    def add_media_asset(
        self,
        *,
        path: str,
        kind: str,
        sha256: str,
        mime: str = "",
        captured_at: float = 0.0,
        caption: str = "",
        ocr_text: str = "",
        transcript: str = "",
        duration_s: float = 0.0,
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        existing = self.get_media_asset_by_sha(sha256) if sha256 else None
        if existing:
            return {"ok": True, "duplicate": True, "asset_id": existing["asset_id"]}
        record = {
            "asset_id": str(uuid.uuid4()),
            "path": path,
            "kind": kind,
            "mime": mime,
            "sha256": sha256,
            "captured_at": float(captured_at or 0.0),
            "ingested_at": time.time(),
            "caption": caption,
            "ocr_text": ocr_text,
            "transcript": transcript,
            "duration_s": float(duration_s or 0.0),
            "width": int(width or 0),
            "height": int(height or 0),
        }
        with self._lock:
            self._tables[MEDIA_ASSETS_TABLE].add([record])
        return {"ok": True, "duplicate": False, "asset_id": record["asset_id"], "record": record}

    def add_media_text_chunks(self, chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for c in chunks:
            rows.append({
                "chunk_id": c.get("chunk_id") or str(uuid.uuid4()),
                "asset_id": c["asset_id"],
                "modality": c.get("modality", "caption"),
                "start_s": float(c.get("start_s") or 0.0),
                "end_s": float(c.get("end_s") or 0.0),
                "content": c["content"],
                "vector": c["vector"],
            })
        if not rows:
            return {"ok": True, "added": 0}
        with self._lock:
            self._tables[MEDIA_TEXT_CHUNKS_TABLE].add(rows)
        return {"ok": True, "added": len(rows)}

    def add_media_image_vecs(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        prepared = []
        for r in rows:
            prepared.append({
                "asset_id": r["asset_id"],
                "frame_idx": int(r.get("frame_idx", 0)),
                "t_s": float(r.get("t_s") or 0.0),
                "vector": r["vector"],
            })
        if not prepared:
            return {"ok": True, "added": 0}
        with self._lock:
            self._tables[MEDIA_IMAGE_VECS_TABLE].add(prepared)
        return {"ok": True, "added": len(prepared)}

    def search_media_text(
        self,
        query: str,
        top_k: int = 5,
        distance_max: float | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        vec = self._embed_text(query)
        with self._lock:
            res = (
                self._tables[MEDIA_TEXT_CHUNKS_TABLE]
                .search(vec)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        return _project_hits(res, distance_max, drop=("vector",))

    def search_media_image(
        self,
        image_vector: list[float],
        top_k: int = 5,
        distance_max: float | None = None,
    ) -> list[dict[str, Any]]:
        if not image_vector:
            return []
        with self._lock:
            res = (
                self._tables[MEDIA_IMAGE_VECS_TABLE]
                .search(image_vector)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        return _project_hits(res, distance_max, drop=("vector",))

    # --- indexing ---------------------------------------------------------

    def wiki_index_status(self) -> dict[str, Any]:
        """Whether the wiki_chunks vector column has an ANN index built."""
        try:
            indices = list(self._tables[WIKI_CHUNKS_TABLE].list_indices())
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        info = [
            {"name": getattr(i, "name", str(i)),
             "columns": list(getattr(i, "columns", []) or []),
             "index_type": getattr(i, "index_type", "?")}
            for i in indices
        ]
        return {"ok": True, "indices": info, "has_vector_index": any(
            "vector" in (entry.get("columns") or []) for entry in info
        )}

    def build_wiki_index(
        self,
        *,
        force: bool = False,
        num_partitions: int | None = None,
        num_sub_vectors: int = 64,
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Build an IVF_PQ index on wiki_chunks.vector.

        Brute-force scan over millions of 1024-dim vectors takes seconds per
        query — unusable in the chat path. IVF_PQ partitions the space and
        product-quantizes each subspace; lookup becomes sub-50ms with a
        tiny recall hit. Build time scales roughly with row count; expect
        ~30-60 min on 50M vectors.

        ``num_partitions`` defaults to ~sqrt(num_rows), the standard rule.
        ``num_sub_vectors`` of 64 with 1024-dim vectors gives 16-d subvecs
        — a good speed/recall tradeoff."""
        table = self._tables[WIKI_CHUNKS_TABLE]
        n = table.count_rows()
        status = self.wiki_index_status()
        if status.get("has_vector_index") and not force:
            return {"ok": True, "skipped": True, "reason": "index already exists",
                    "rows": n, "status": status}
        if n < 256:
            return {"ok": False, "error": "not enough rows to build IVF index",
                    "rows": n}
        if num_partitions is None:
            # sqrt-ish, clamped to a sane range.
            import math
            num_partitions = max(16, min(8192, int(math.sqrt(n))))
        import time as _t
        t0 = _t.time()
        table.create_index(
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
            replace=True,
        )
        elapsed = _t.time() - t0
        return {
            "ok": True,
            "skipped": False,
            "rows": n,
            "num_partitions": num_partitions,
            "num_sub_vectors": num_sub_vectors,
            "metric": metric,
            "elapsed_s": round(elapsed, 2),
            "status": self.wiki_index_status(),
        }

    # --- ops --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts = {name: tbl.count_rows() for name, tbl in self._tables.items()}
        disk = self._disk_stats()
        return {"counts": counts, "disk": disk, "path": str(self.path)}

    def _disk_stats(self) -> dict[str, Any]:
        target = WORLD_VAULT_ROOT if WORLD_VAULT_ROOT.exists() else self.path.parent
        try:
            usage = shutil.disk_usage(target)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "target": str(target)}
        used_pct = (usage.used / usage.total * 100) if usage.total else 0.0
        if used_pct >= WORLD_DISK_CRITICAL_PCT:
            level = "critical"
        elif used_pct >= WORLD_DISK_WARN_PCT:
            level = "warn"
        else:
            level = "ok"
        return {
            "ok": True,
            "target": str(target),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_pct": round(used_pct, 2),
            "warn_pct": WORLD_DISK_WARN_PCT,
            "critical_pct": WORLD_DISK_CRITICAL_PCT,
            "level": level,
        }


def _project_hits(
    rows: list[dict[str, Any]],
    distance_max: float | None,
    drop: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        dist = row.get("_distance")
        if distance_max is not None and dist is not None and dist > distance_max:
            continue
        projected = {k: v for k, v in row.items() if k not in drop}
        projected["distance"] = dist
        out.append(projected)
    return out
