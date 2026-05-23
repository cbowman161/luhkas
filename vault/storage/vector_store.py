"""Identity-scoped semantic memory backed by LanceDB + bge-m3 embeddings.

Writes facts the speaker has stated about themselves under their identity
namespace (face-recognized identity, or "unknown" when no face is identified).
Reads via cosine top-k search.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


EMBED_DIM = 1024  # bge-m3
TABLE_NAME = "user_facts"
DEFAULT_PATH = Path(os.environ.get("VAULT_MEMORY_DB", "/home/vault/vault_data/memory.lance"))


def _norm_identity(identity: str | None) -> str:
    if not identity:
        return "unknown"
    return identity.strip().lower() or "unknown"


class MemoryStore:
    def __init__(self, embedder=None, path: str | Path | None = None):
        self.embedder = embedder
        self.path = Path(path or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.path))
        self._lock = threading.Lock()
        self._table = self._ensure_table()

    def _ensure_table(self):
        if TABLE_NAME in self.db.table_names():
            return self.db.open_table(TABLE_NAME)
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("identity", pa.string()),
            pa.field("unidentified_face_ref", pa.string()),
            pa.field("content", pa.string()),
            pa.field("category", pa.string()),
            pa.field("source_message", pa.string()),
            pa.field("created_at", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        ])
        return self.db.create_table(TABLE_NAME, schema=schema)

    def _embed(self, text: str) -> list[float]:
        if not self.embedder:
            raise RuntimeError("MemoryStore.embedder is not configured")
        result = self.embedder.embed(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result[0]
        return result  # type: ignore[return-value]

    def add(
        self,
        content: str,
        identity: str | None = None,
        unidentified_face_ref: str | None = None,
        category: str = "fact",
        source_message: str = "",
        duplicate_distance: float = 0.25,
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            return {"ok": False, "error": "empty_content"}
        ident = _norm_identity(identity)
        vec = self._embed(content)
        # Duplicate guard: if a very-close match already exists in this
        # identity's namespace, return it instead of inserting a second copy.
        with self._lock:
            existing = (
                self._table.search(vec)
                .where(f"identity = '{ident}'", prefilter=True)
                .limit(1)
                .to_list()
            )
        if existing:
            dist = existing[0].get("_distance")
            if dist is not None and dist <= duplicate_distance:
                row = existing[0]
                return {
                    "ok": True,
                    "duplicate": True,
                    "distance": dist,
                    "record": {k: v for k, v in row.items() if k != "vector"},
                }
        record = {
            "id": str(uuid.uuid4()),
            "identity": ident,
            "unidentified_face_ref": (unidentified_face_ref or "").strip(),
            "content": content,
            "category": category or "fact",
            "source_message": (source_message or "").strip(),
            "created_at": time.time(),
            "vector": vec,
        }
        with self._lock:
            self._table.add([record])
        return {"ok": True, "duplicate": False, "record": {k: v for k, v in record.items() if k != "vector"}}

    def search(
        self,
        query: str,
        identity: str | None = None,
        top_k: int = 5,
        distance_max: float | None = 1.5,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        vec = self._embed(query)
        ident = _norm_identity(identity)
        with self._lock:
            res = (
                self._table.search(vec)
                .where(f"identity = '{ident}'", prefilter=True)
                .limit(top_k)
                .to_list()
            )
        out = []
        for row in res:
            dist = row.get("_distance")
            if distance_max is not None and dist is not None and dist > distance_max:
                continue
            out.append({
                "id": row.get("id"),
                "identity": row.get("identity"),
                "unidentified_face_ref": row.get("unidentified_face_ref") or None,
                "content": row.get("content"),
                "category": row.get("category"),
                "source_message": row.get("source_message"),
                "created_at": row.get("created_at"),
                "distance": dist,
            })
        return out

    def find_conflict_candidates(
        self,
        content: str,
        identity: str | None = None,
        distance_min: float = 0.25,
        distance_max: float = 0.65,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return facts in the speaker's namespace that are close enough to be
        about the same subject (could contradict) but not so close that they
        are already-known duplicates. Caller verifies actual contradiction
        with an LLM classifier."""
        content = (content or "").strip()
        if not content:
            return []
        vec = self._embed(content)
        ident = _norm_identity(identity)
        with self._lock:
            res = (
                self._table.search(vec)
                .where(f"identity = '{ident}'", prefilter=True)
                .limit(top_k + 5)
                .to_list()
            )
        out = []
        for row in res:
            dist = row.get("_distance")
            if dist is None:
                continue
            if dist <= distance_min:
                continue
            if dist > distance_max:
                continue
            out.append({
                "id": row.get("id"),
                "identity": row.get("identity"),
                "content": row.get("content"),
                "category": row.get("category"),
                "source_message": row.get("source_message"),
                "created_at": row.get("created_at"),
                "distance": dist,
            })
            if len(out) >= top_k:
                break
        return out

    def delete_by_id(self, fact_id: str) -> bool:
        if not fact_id:
            return False
        with self._lock:
            try:
                self._table.delete(f"id = '{fact_id}'")
                return True
            except Exception:
                return False

    def replace(
        self,
        old_id: str,
        content: str,
        identity: str | None = None,
        unidentified_face_ref: str | None = None,
        category: str = "fact",
        source_message: str = "",
    ) -> dict[str, Any]:
        """Delete the old record by id and insert the new content as a fresh
        record (no duplicate check — caller already classified the relation)."""
        self.delete_by_id(old_id)
        content = (content or "").strip()
        if not content:
            return {"ok": False, "error": "empty_content"}
        record = {
            "id": str(uuid.uuid4()),
            "identity": _norm_identity(identity),
            "unidentified_face_ref": (unidentified_face_ref or "").strip(),
            "content": content,
            "category": category or "fact",
            "source_message": (source_message or "").strip(),
            "created_at": time.time(),
            "vector": self._embed(content),
        }
        with self._lock:
            self._table.add([record])
        return {"ok": True, "replaced_id": old_id, "record": {k: v for k, v in record.items() if k != "vector"}}

    def list_for_identity(self, identity: str | None, limit: int = 200) -> list[dict[str, Any]]:
        ident = _norm_identity(identity)
        with self._lock:
            rows = (
                self._table.search()
                .where(f"identity = '{ident}'")
                .limit(limit)
                .to_list()
            )
        rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return [
            {k: v for k, v in r.items() if k != "vector"}
            for r in rows
        ]

    def count(self) -> int:
        with self._lock:
            return self._table.count_rows()


# Back-compat shim: keep the old VectorStore name pointing at MemoryStore for
# any historical imports (blackboard.py instantiates it with no args, so we
# leave that path working with a stub-free fallback that just no-ops if no
# embedder is wired).
class VectorStore:
    def __init__(self, embedder=None, path: str | Path | None = None):
        try:
            self._inner = MemoryStore(embedder=embedder, path=path)
        except Exception:
            self._inner = None

    def add(self, content, metadata=None):
        if self._inner is None:
            return None
        identity = (metadata or {}).get("identity")
        return self._inner.add(content, identity=identity)

    def search(self, query, top_k=5):
        if self._inner is None:
            return []
        return self._inner.search(query, top_k=top_k)
