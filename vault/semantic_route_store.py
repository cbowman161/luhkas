from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


EMBED_DIM = 1024
DEFAULT_PATH = Path(os.environ.get("VAULT_SEMANTIC_ROUTE_DB", "/home/vault/vault_data/semantic_routes.lance"))
LEARNED_CAPS_TABLE = "learned_capability_candidates"
DETERMINISTIC_ROUTES_TABLE = "deterministic_route_candidates"


def _embed_one(embedder, text: str) -> list[float]:
    if embedder is None:
        raise RuntimeError("embedder is not configured")
    result = embedder.embed(text)
    if isinstance(result, list) and result and isinstance(result[0], list):
        return result[0]
    return result


def _project(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "vector"}
    if "_distance" in row:
        out["distance"] = row.get("_distance")
    return out


class SemanticRouteStore:
    """Rebuildable LanceDB sidecar for fuzzy learned command/route candidates.

    JSON remains the source of truth. This store only proposes candidates;
    callers decide whether a distance is tight enough to act on.
    """

    def __init__(self, embedder=None, path: str | Path | None = None):
        self.embedder = embedder
        self.path = Path(path or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.path))
        self._lock = threading.Lock()
        self._tables = {
            LEARNED_CAPS_TABLE: self._ensure_table(LEARNED_CAPS_TABLE, self._learned_schema()),
            DETERMINISTIC_ROUTES_TABLE: self._ensure_table(DETERMINISTIC_ROUTES_TABLE, self._route_schema()),
        }

    @staticmethod
    def _learned_schema() -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("normalized_input", pa.string()),
            pa.field("input", pa.string()),
            pa.field("description", pa.string()),
            pa.field("intent", pa.string()),
            pa.field("topic", pa.string()),
            pa.field("aspect", pa.string()),
            pa.field("updated_at", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        ])

    @staticmethod
    def _route_schema() -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("normalized_input", pa.string()),
            pa.field("input", pa.string()),
            pa.field("route", pa.string()),
            pa.field("self_route", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("updated_at", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        ])

    def _ensure_table(self, name: str, schema: pa.Schema):
        try:
            return self.db.open_table(name)
        except (FileNotFoundError, ValueError):
            return self.db.create_table(name, schema=schema)

    def upsert_learned_capability(self, key: str, cap: dict[str, Any]) -> None:
        key = (key or cap.get("normalized_input") or "").strip()
        if not key:
            return
        text = self._learned_text(key, cap)
        record = {
            "id": key,
            "kind": "learned_capability",
            "normalized_input": key,
            "input": str(cap.get("input") or key),
            "description": str(cap.get("description") or ""),
            "intent": str(cap.get("intent") or cap.get("name") or ""),
            "topic": str((cap.get("inferred") or {}).get("topic") or ""),
            "aspect": str((cap.get("inferred") or {}).get("aspect") or ""),
            "updated_at": float(cap.get("updated_at") or cap.get("created_at") or time.time()),
            "vector": _embed_one(self.embedder, text),
        }
        table = self._tables[LEARNED_CAPS_TABLE]
        with self._lock:
            table.delete(f"id = '{self._sql_escape(key)}'")
            table.add([record])

    def upsert_deterministic_route(self, key: str, entry: dict[str, Any]) -> None:
        key = (key or entry.get("normalized_input") or "").strip()
        if not key:
            return
        route = entry.get("route") if isinstance(entry.get("route"), dict) else {}
        self_route = route.get("self_route") if isinstance(route.get("self_route"), dict) else {}
        text = self._route_text(key, entry, route, self_route)
        record = {
            "id": key,
            "kind": "deterministic_route",
            "normalized_input": key,
            "input": str(entry.get("input") or key),
            "route": str(route.get("route") or entry.get("decided_route") or ""),
            "self_route": str(self_route.get("route") or entry.get("self_route") or ""),
            "reason": str(route.get("reason") or entry.get("reason") or ""),
            "updated_at": float(entry.get("confirmed_at") or time.time()),
            "vector": _embed_one(self.embedder, text),
        }
        table = self._tables[DETERMINISTIC_ROUTES_TABLE]
        with self._lock:
            table.delete(f"id = '{self._sql_escape(key)}'")
            table.add([record])

    def search_learned_capabilities(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        table = self._tables[LEARNED_CAPS_TABLE]
        if int(table.count_rows()) <= 0:
            return []
        vec = _embed_one(self.embedder, query)
        with self._lock:
            rows = (
                table
                .search(vec)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        return [_project(r) for r in rows]

    def search_deterministic_routes(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        table = self._tables[DETERMINISTIC_ROUTES_TABLE]
        if int(table.count_rows()) <= 0:
            return []
        vec = _embed_one(self.embedder, query)
        with self._lock:
            rows = (
                table
                .search(vec)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        return [_project(r) for r in rows]

    def learned_count(self) -> int:
        return int(self._tables[LEARNED_CAPS_TABLE].count_rows())

    def route_count(self) -> int:
        return int(self._tables[DETERMINISTIC_ROUTES_TABLE].count_rows())

    @staticmethod
    def _learned_text(key: str, cap: dict[str, Any]) -> str:
        inferred = cap.get("inferred") or {}
        examples = cap.get("examples") if isinstance(cap.get("examples"), list) else []
        example_text = " ".join(str(e.get("input") or "") for e in examples[-5:] if isinstance(e, dict))
        return " ".join(part for part in (
            str(cap.get("input") or key),
            str(cap.get("description") or ""),
            str(cap.get("intent") or cap.get("name") or ""),
            str(inferred.get("topic") or ""),
            str(inferred.get("aspect") or ""),
            example_text,
        ) if part).strip()

    @staticmethod
    def _route_text(key: str, entry: dict[str, Any], route: dict[str, Any], self_route: dict[str, Any]) -> str:
        examples = entry.get("examples") if isinstance(entry.get("examples"), list) else []
        example_text = " ".join(str(e.get("input") or "") for e in examples[-5:] if isinstance(e, dict))
        return " ".join(part for part in (
            str(entry.get("input") or key),
            str(route.get("route") or entry.get("decided_route") or ""),
            str(self_route.get("route") or entry.get("self_route") or ""),
            str(route.get("reason") or entry.get("reason") or ""),
            example_text,
        ) if part).strip()

    @staticmethod
    def _sql_escape(value: str) -> str:
        return (value or "").replace("'", "''")
