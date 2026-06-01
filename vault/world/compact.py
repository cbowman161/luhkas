#!/usr/bin/env python3
"""Compact LanceDB tables + maintain ANN indexes.

Two maintenance passes folded into one script (called daily via
world-compact.timer):

1. **Compaction.** Each ingest flush appends a new fragment file. After
   ~10k fragments per table, write throughput collapses (observed:
   21 articles/sec → 0.2 articles/sec at ~40k fragments). optimize()
   merges fragments and reclaims disk used by old versions.

2. **Vector index build/maintain.** wiki_chunks holds ~millions of
   1024-dim bge-m3 vectors. Without an ANN index, every RAG query
   brute-force scans the table (~2.4 s on 2.5M rows). IVF-PQ index
   drops that to milliseconds. This script creates the index if
   missing; subsequent optimize() calls incrementally update it as
   new chunks are ingested.

Both ops are safe to run while world-ingest-supervisor is active —
optimize() and create_index() use lance's transaction model and the
supervisor's child sees the new state on its next flush.

Env:
  WORLD_DB_PATH               default /home/vault/world_data/world.lance
  WORLD_COMPACT_TABLES        comma list, default "wiki_articles,wiki_chunks"
  WORLD_COMPACT_CLEANUP_DAYS  retention for old versions before reclaim;
                              default 0 (immediate reclaim — we never want
                              to roll back to a historical Wikipedia
                              snapshot)
  WORLD_INDEX_ENABLE          "0" to skip index maintenance (default 1)
  WORLD_INDEX_NUM_PARTITIONS  IVF partitions, default 512
                              (rule-of-thumb 4·sqrt(N)/16 for 2.5M rows)
  WORLD_INDEX_NUM_SUBVECTORS  PQ sub-vectors, default 128
                              (1024 dim / 8 dims-per-sub = 32× compression,
                              keeps index footprint ~500 MB vs 12 GB raw)
  WORLD_INDEX_METRIC          distance metric, default "cosine"
                              (matches world_store.search_wiki())
  WORLD_COMPACT_DRY_RUN       if "1", print fragment counts and exit
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] world_compact: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("world_compact")


DB_PATH = os.environ.get("WORLD_DB_PATH", "/home/vault/world_data/world.lance")
TABLES = [t.strip() for t in os.environ.get("WORLD_COMPACT_TABLES", "wiki_articles,wiki_chunks").split(",") if t.strip()]
CLEANUP_DAYS = float(os.environ.get("WORLD_COMPACT_CLEANUP_DAYS", "0"))
DRY_RUN = os.environ.get("WORLD_COMPACT_DRY_RUN", "0").lower() in ("1", "true", "yes")

# Vector index maintenance — IVF_PQ, configured for RAM efficiency.
INDEX_ENABLE = os.environ.get("WORLD_INDEX_ENABLE", "1").lower() not in ("0", "false", "no", "")
INDEX_NUM_PARTITIONS = int(os.environ.get("WORLD_INDEX_NUM_PARTITIONS", "512"))
INDEX_NUM_SUBVECTORS = int(os.environ.get("WORLD_INDEX_NUM_SUBVECTORS", "128"))
INDEX_METRIC = os.environ.get("WORLD_INDEX_METRIC", "cosine")
# Tables that have a `vector` column the ANN index applies to. Skip
# tables without one (wiki_articles holds metadata only).
INDEX_VECTOR_COLUMN = "vector"


def _count_fragments(table_path: Path) -> int:
    """Cheap fragment count from the filesystem — avoids opening the
    table (which is slow with many fragments)."""
    data_dir = table_path / "data"
    if not data_dir.is_dir():
        return 0
    try:
        return sum(1 for _ in data_dir.iterdir())
    except OSError:
        return -1


def _table_has_vector_column(table) -> bool:
    """True if the table has a column we'd ANN-index."""
    try:
        names = [f.name for f in table.schema]
        return INDEX_VECTOR_COLUMN in names
    except Exception:
        return False


def _existing_vector_index(table) -> dict | None:
    """Return the IVF_PQ index descriptor for the vector column, or None
    if none exists. lancedb.list_indices returns IndexConfig objects
    with at least .name, .columns, .index_type."""
    try:
        for idx in table.list_indices():
            cols = getattr(idx, "columns", None) or []
            if INDEX_VECTOR_COLUMN in cols:
                return {
                    "name": getattr(idx, "name", "?"),
                    "type": getattr(idx, "index_type", "?"),
                    "columns": list(cols),
                }
    except Exception as exc:
        log.warning("list_indices failed: %s", exc)
    return None


def _maintain_index(table, table_name: str) -> None:
    """Build the vector index if missing. Existing index is left to
    optimize() (called before this) to incrementally update."""
    if not INDEX_ENABLE:
        log.info("%s: index maintenance disabled (WORLD_INDEX_ENABLE=0)", table_name)
        return
    if not _table_has_vector_column(table):
        log.info("%s: no vector column — skipping index", table_name)
        return
    existing = _existing_vector_index(table)
    if existing is not None:
        log.info("%s: vector index already present (name=%s type=%s)",
                 table_name, existing["name"], existing["type"])
        return
    log.info(
        "%s: building IVF_PQ (num_partitions=%d num_sub_vectors=%d metric=%s)...",
        table_name, INDEX_NUM_PARTITIONS, INDEX_NUM_SUBVECTORS, INDEX_METRIC,
    )
    t0 = time.time()
    try:
        table.create_index(
            metric=INDEX_METRIC,
            num_partitions=INDEX_NUM_PARTITIONS,
            num_sub_vectors=INDEX_NUM_SUBVECTORS,
            vector_column_name=INDEX_VECTOR_COLUMN,
            index_type="IVF_PQ",
        )
        log.info("%s: index built in %.1fs", table_name, time.time() - t0)
    except Exception as exc:
        log.error("%s: create_index failed: %s", table_name, exc)


def main() -> int:
    import lancedb

    db_path = Path(DB_PATH)
    if not db_path.is_dir():
        log.error("DB not found: %s", db_path)
        return 2

    start = time.time()
    db = lancedb.connect(str(db_path))

    for name in TABLES:
        table_path = db_path / f"{name}.lance"
        pre_frags = _count_fragments(table_path)
        if DRY_RUN:
            log.info("%s: %d fragments (dry-run)", name, pre_frags)
            continue
        if pre_frags <= 0:
            log.info("%s: skipping (no data dir or empty)", name)
            continue
        log.info("%s: pre-optimize fragments=%d", name, pre_frags)
        try:
            t = db.open_table(name)
        except Exception as exc:
            log.warning("%s: open failed (%s); skipping", name, exc)
            continue
        t0 = time.time()
        try:
            stats = t.optimize(cleanup_older_than=datetime.timedelta(days=CLEANUP_DAYS))
        except Exception as exc:
            log.error("%s: optimize failed: %s", name, exc)
            continue
        post_frags = _count_fragments(table_path)
        log.info(
            "%s: optimize done in %.1fs — fragments %d → %d (Δ%+d) — stats=%s",
            name, time.time() - t0, pre_frags, post_frags, post_frags - pre_frags, stats,
        )
        # ANN index — build if missing. optimize() above already does
        # incremental updates for an existing index, so this is a no-op
        # on subsequent runs.
        _maintain_index(t, name)

    log.info("total elapsed: %.1fs", time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
