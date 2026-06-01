#!/usr/bin/env python3
"""Compact LanceDB tables to recover write throughput.

Run periodically (via world-compact.timer) so fragmentation never gets
catastrophic. Without this, each ingest flush creates a new fragment
file; after ~10k+ fragments LanceDB writes start coordinating metadata
across thousands of files and throughput collapses (observed: 21
articles/sec → 0.2 articles/sec at ~40k fragments).

Safe to run while world-ingest-supervisor is active — optimize() is
designed for concurrent reads/writes, and lance's transaction model
handles fragment merges atomically. The supervisor's child will see a
shorter fragment list on its next flush.

Env:
  WORLD_DB_PATH               default /home/vault/world_data/world.lance
  WORLD_COMPACT_TABLES        comma list, default "wiki_articles,wiki_chunks"
  WORLD_COMPACT_CLEANUP_DAYS  retention for old versions before reclaim;
                              default 0 (immediate reclaim — we never want
                              to roll back to a historical Wikipedia
                              snapshot)
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

    log.info("total elapsed: %.1fs", time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
