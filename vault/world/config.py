"""World vault paths and tunables.

Defaults assume the standard luhkas-vault layout: a single 2TB NVMe with
plenty of free space. Override via env vars when relocating to a dedicated
drive."""
from __future__ import annotations

import os
from pathlib import Path


WORLD_VAULT_ROOT = Path(os.environ.get(
    "WORLD_VAULT_ROOT", "/home/vault/world_data"
))
WORLD_DB_PATH = Path(os.environ.get(
    "WORLD_DB_PATH", str(WORLD_VAULT_ROOT / "world.lance")
))
WORLD_ORIGINALS_DIR = Path(os.environ.get(
    "WORLD_ORIGINALS_DIR", str(WORLD_VAULT_ROOT / "originals")
))
WORLD_INBOX_DIR = Path(os.environ.get(
    "WORLD_INBOX_DIR", str(WORLD_VAULT_ROOT / "inbox")
))
WORLD_ZIM_DIR = Path(os.environ.get(
    "WORLD_ZIM_DIR", str(WORLD_VAULT_ROOT / "zim")
))

# Disk pressure thresholds for the volume backing WORLD_VAULT_ROOT.
# `/world/status` returns warn/critical so the brain can surface alerts before
# refresh churn fills the drive.
WORLD_DISK_WARN_PCT = float(os.environ.get("WORLD_DISK_WARN_PCT", "75"))
WORLD_DISK_CRITICAL_PCT = float(os.environ.get("WORLD_DISK_CRITICAL_PCT", "85"))

# Vector recall gates. Auto-lookup inside answer_with_context only injects a
# world-knowledge context block when the top hit's cosine distance is at or
# below this value. Tuned conservatively in Phase 1; revisit after first
# real-world batteries.
WORLD_WIKI_DISTANCE_MAX = float(os.environ.get("WORLD_WIKI_DISTANCE_MAX", "0.55"))
WORLD_MEDIA_TEXT_DISTANCE_MAX = float(os.environ.get(
    "WORLD_MEDIA_TEXT_DISTANCE_MAX", "0.55"
))
WORLD_IMAGE_DISTANCE_MAX = float(os.environ.get(
    "WORLD_IMAGE_DISTANCE_MAX", "0.45"
))

# Embedding dimensions (kept here so the store schema and any future
# embedder swaps stay aligned).
TEXT_EMBED_DIM = int(os.environ.get("WORLD_TEXT_EMBED_DIM", "1024"))  # bge-m3
IMAGE_EMBED_DIM = int(os.environ.get("WORLD_IMAGE_EMBED_DIM", "768"))  # SigLIP-2 base


def ensure_dirs() -> None:
    """Create the world-vault directory tree if it doesn't exist."""
    for path in (
        WORLD_VAULT_ROOT,
        WORLD_ORIGINALS_DIR,
        WORLD_INBOX_DIR,
        WORLD_ZIM_DIR,
        WORLD_DB_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
