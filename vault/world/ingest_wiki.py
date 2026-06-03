"""Ingest a Kiwix Wikipedia ZIM into WorldKnowledgeStore.

Flow per article:
  HTML  →  section-aware text  →  ~400-word sliding-window chunks
        →  bge-m3 embed (batched)  →  upsert into wiki_chunks

Resumable + revision-aware: an article's content is hashed; if the hash matches
the stored hash for that article_id, ingestion skips it. If different, the
article's chunks are atomically replaced.

CLI:
    python3 -m world.ingest_wiki /path/to/file.zim [--limit N] [--lang en]
                                  [--no-resume] [--batch 32] [--start N]
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import sys
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Iterator

import unicodedata

from bs4 import BeautifulSoup
from libzim.reader import Archive

from world.world_store import WorldKnowledgeStore, WIKI_ARTICLES_TABLE


def _normalize_one(vec):
    """Some Ollama embed responses come back as [[...]] for single inputs."""
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        return vec[0]
    return vec


def _has_nan(vec) -> bool:
    return bool(vec) and any(x != x for x in vec)


def robust_embed(embedder, text: str) -> list[float] | None:
    """Embed one chunk, working around bge-m3+Ollama's occasional NaN bug.

    bge-m3 served via Ollama deterministically returns NaN for some specific
    inputs even when batch position varies. The reliable workaround we
    found is to embed the text concatenated with itself — the model returns
    a valid vector and we keep only the first copy's contribution (cosine
    of `x+x` to a query is identical to cosine of `x` to a query, so this is
    semantically lossless for retrieval)."""
    if not text:
        return None
    for candidate in (text, unicodedata.normalize("NFKC", text)):
        try:
            v = _normalize_one(embedder.embed(candidate))
            if v and not _has_nan(v):
                return v
        except Exception:
            pass
    try:
        v = _normalize_one(embedder.embed(text + " " + text))
        if v and not _has_nan(v):
            return v
    except Exception:
        pass
    return None


def robust_embed_batch(
    embedder, texts: list[str], *, concurrency: int = 1
) -> list[list[float] | None]:
    """Try a single batch call; fall back per-item on failure (any text in
    the batch can poison the whole response). When ``concurrency > 1`` the
    per-item fallback runs in parallel via a thread pool — Ollama happily
    serves concurrent embed requests and this cuts wall-time roughly
    proportional to the worker count on a single GPU."""
    if not texts:
        return []
    try:
        raw = embedder.embed(texts)
        if isinstance(raw, list) and len(raw) == len(texts) and all(
            isinstance(r, list) for r in raw
        ):
            if not any(_has_nan(r) for r in raw):
                return raw
    except Exception:
        pass
    if concurrency <= 1:
        return [robust_embed(embedder, t) for t in texts]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda t: robust_embed(embedder, t), texts))


def parallel_embed(
    embedder, texts: list[str], *, concurrency: int = 4
) -> list[list[float] | None]:
    """Fan out one-call-per-text across a thread pool. Use this when the
    batch path is unreliable (bge-m3+Ollama returns NaN on poisoned
    batches), trading the batch optimization for predictable throughput
    via concurrency."""
    if not texts:
        return []
    if concurrency <= 1:
        return [robust_embed(embedder, t) for t in texts]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda t: robust_embed(embedder, t), texts))


# --- HTML → sections --------------------------------------------------------

_DROP_SELECTORS = (
    "table.infobox",
    "table.navbox",
    "table.metadata",
    "table.sidebar",
    "div.reflist",
    "ol.references",
    "sup.reference",
    "div.thumb",
    "div.hatnote",
    "div.navbox",
    "div.toc",
    "div#toc",
    "span.mw-editsection",
    "div.mw-references-wrap",
    "div.printfooter",
    "div.catlinks",
    "div.mw-indicators",
)


@dataclass
class Section:
    path: str
    text: str


_BS4_PARSER = os.environ.get("WORLD_BS4_PARSER", "lxml")

# Sections whose content is bibliographic/navigational noise — useless
# for chat retrieval and ugly to speak. Matched case-insensitively against
# the section heading text. Anything ending in one of these words is
# treated as junk too (e.g. "Selected discography" → matches "discography").
_JUNK_SECTION_KEYWORDS = (
    "references", "external links", "see also", "bibliography",
    "further reading", "notes", "citations", "sources", "footnotes",
    "works cited", "publications", "selected works",
    "external link", "discography", "filmography", "track listing",
    "track list", "tracklist", "tracks", "personnel", "credits",
    "awards", "accolades", "honors", "honours",
    "chart performance", "chart positions", "charts",
    "release history", "release dates",
)

# Inline residue patterns left over after BS4 tag-stripping.
_CITATION_BRACKET_RE = re.compile(r"\[(?:\d+|edit|citation needed|clarification needed|when\?|who\?|where\?|why\?|update|note \d+|note [a-z]+)\]", re.IGNORECASE)
_PARENS_NOTE_RE = re.compile(r"\(\s*(?:listen|help[.·]info|pronounced[^)]*|/[^)]+/)\s*\)", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(])\s+")
_REPEATED_PUNCT_RE = re.compile(r"([.,;:!?])\1{2,}")
_MULTI_WS_RE = re.compile(r"\s+")  # also collapses single newlines/tabs


def _is_junk_section(heading: str) -> bool:
    """True if this section's heading marks it as nav/biblio noise."""
    if not heading:
        return False
    h = heading.strip().lower()
    if not h:
        return False
    if h in _JUNK_SECTION_KEYWORDS:
        return True
    # "Selected discography" / "Album discography" / "External links and references"
    for kw in _JUNK_SECTION_KEYWORDS:
        if h.endswith(kw) or h.startswith(kw):
            return True
    return False


def clean_speakable(text: str) -> str:
    """Normalize text so TTS reads it cleanly and embeddings don't waste
    capacity on whitespace/punctuation artifacts.

    Applied per-section before chunking so the same cleanup also helps
    retrieval quality (fewer near-duplicates that differ only in stray
    whitespace, fewer chunks dominated by citation cruft)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _CITATION_BRACKET_RE.sub("", t)
    t = _PARENS_NOTE_RE.sub("", t)
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN_RE.sub(r"\1", t)
    t = _REPEATED_PUNCT_RE.sub(r"\1", t)
    t = _MULTI_WS_RE.sub(" ", t)
    # Empty parens left behind by removed inline notes.
    t = re.sub(r"\(\s*\)", "", t)
    t = _MULTI_WS_RE.sub(" ", t)
    return t.strip()


def parse_article_html(html: str, *, min_section_words: int = 20) -> list[Section]:
    """Walk article body, strip cruft, group paragraphs by heading.

    Returns a list of (section_path, text) tuples ordered by appearance.
    Sections matching junk headings (References, External links, etc.)
    are dropped wholesale, as are sections under ``min_section_words``
    which tend to be degenerate stubs."""
    try:
        soup = BeautifulSoup(html, _BS4_PARSER)
    except Exception:
        # Fallback to stdlib parser if lxml isn't installed.
        soup = BeautifulSoup(html, "html.parser")
    content = (
        soup.find("div", id="mw-content-text")
        or soup.find("main")
        or soup.body
        or soup
    )
    for sel in _DROP_SELECTORS:
        for n in content.select(sel):
            n.decompose()

    sections: list[Section] = []
    current_path = "Lead"
    current_buf: list[str] = []
    current_is_junk = False

    def flush() -> None:
        if current_is_junk or not current_buf:
            return
        joined = " ".join(part for part in current_buf if part)
        cleaned = clean_speakable(joined)
        if not cleaned:
            return
        # Lead is always kept so even a one-sentence stub article
        # ("X is a district of Y") stays retrievable. Only non-Lead
        # sections are subject to the word-count floor, to drop
        # degenerate per-section stubs.
        if current_path != "Lead" and len(cleaned.split()) < min_section_words:
            return
        sections.append(Section(path=current_path, text=cleaned))

    for el in content.find_all(["h2", "h3", "h4", "p", "li"], recursive=True):
        if el.name in ("h2", "h3", "h4"):
            flush()
            current_buf = []
            current_path = el.get_text(" ", strip=True) or current_path
            current_is_junk = _is_junk_section(current_path)
            continue
        if current_is_junk:
            continue
        t = el.get_text(" ", strip=True)
        if t:
            current_buf.append(t)
    flush()
    return sections


# --- sections → chunks ------------------------------------------------------

@dataclass
class Chunk:
    section_path: str
    chunk_idx: int
    content: str
    content_hash: str


def _speakable_title(title: str) -> str:
    """Strip non-speakable leading punctuation from titles. Wikipedia
    stylizes a lot of articles with leading symbols (``"`` ``'`` ``!``
    ``&`` ``$``) that read as garbage when a TTS engine vocalises them
    as the first thing in a sentence. We keep them only when they're
    integral to the meaning (e.g. "5G" stays "5G")."""
    if not title:
        return title
    t = unicodedata.normalize("NFKC", title).strip()
    # Trim leading runs of pure punctuation.
    while t and t[0] in "\"'`’‘“”!?¿¡&$#~^*+=:;,.|/\\<>(){}[]—–-_":
        t = t[1:].lstrip()
    return t or title  # fallback to original if we stripped everything


def chunk_sections(
    sections: Iterable[Section],
    title: str,
    *,
    max_words: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Sliding-window chunker. Each chunk's stored content is prefixed with
    "<title>. <section>. " so the embedding carries article + section
    context, which is critical for short queries (e.g. "Ulaanbaatar"),
    and a TTS engine reads the prefix as natural sentences instead of
    "em dash colon"."""
    chunks: list[Chunk] = []
    idx = 0
    speakable_title = _speakable_title(title)
    for section in sections:
        words = section.text.split()
        if not words:
            continue
        step = max(1, max_words - overlap)
        speakable_section = clean_speakable(section.path)
        prefix_parts = [speakable_title]
        if speakable_section and speakable_section.lower() != "lead":
            prefix_parts.append(speakable_section)
        prefix = ". ".join(prefix_parts).rstrip(".") + ". "
        for start in range(0, len(words), step):
            window = words[start:start + max_words]
            if not window:
                break
            body = " ".join(window)
            content = f"{prefix}{body}"
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunks.append(Chunk(
                section_path=section.path,
                chunk_idx=idx,
                content=content,
                content_hash=h,
            ))
            idx += 1
            if start + max_words >= len(words):
                break
    return chunks


# --- ZIM iteration ----------------------------------------------------------

@dataclass
class WikiArticle:
    entry_index: int          # ZIM entry id — used for resume cursor
    article_id: str
    title: str
    slug: str
    html: str
    content_hash: str  # hash of the source HTML (cheap signal for revision)


def iter_zim_articles(zim_path: str, *, lang: str = "en", start: int = 0) -> Iterator[WikiArticle]:
    archive = Archive(zim_path)
    total = archive.entry_count
    for i in range(start, total):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue
        if entry.is_redirect:
            continue
        item = entry.get_item()
        mimetype = item.mimetype or ""
        if not mimetype.startswith("text/html"):
            continue
        slug = entry.path or ""
        if not slug or slug.startswith("-/") or slug.startswith("_/"):
            continue
        try:
            raw = bytes(item.content)
        except Exception:
            continue
        if not raw:
            continue
        # Soft-redirect stubs: Wikipedia ZIMs include tiny pages whose body
        # is just a <meta http-equiv="refresh"> pointing at the canonical
        # article. libzim's `is_redirect` doesn't catch these. Anything
        # under ~600 bytes with a meta-refresh is one.
        if len(raw) < 800 and b"http-equiv=\"refresh\"" in raw:
            continue
        html = raw.decode("utf-8", errors="replace")
        yield WikiArticle(
            entry_index=i,
            article_id=f"{lang}wiki:{slug}",
            title=entry.title or slug,
            slug=slug,
            html=html,
            content_hash=hashlib.sha256(raw).hexdigest(),
        )


# --- store helpers ----------------------------------------------------------

def load_existing_article_hashes(store: WorldKnowledgeStore) -> dict[str, str]:
    """Single-shot dump of (article_id -> content_hash via revision field).
    The wiki_articles table is small enough (<<1M rows) that this fits in RAM
    and saves a per-article roundtrip."""
    table = store._tables[WIKI_ARTICLES_TABLE]
    out: dict[str, str] = {}
    try:
        rows = table.to_pandas()[["article_id", "revision"]].to_dict("records")
        for r in rows:
            out[r["article_id"]] = r.get("revision") or ""
    except Exception:
        # Fallback to lance scan if pandas conversion fails
        for row in table.search().limit(0).to_list():
            out[row["article_id"]] = row.get("revision") or ""
    return out


# --- main ingest ------------------------------------------------------------

def _atomic_write_state(path: str | None, state: dict) -> None:
    if not path:
        return
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def ingest_zim(
    zim_path: str,
    store: WorldKnowledgeStore,
    embedder,
    *,
    lang: str = "en",
    limit: int | None = None,
    resume: bool = True,
    batch_size: int = 32,
    progress: bool = True,
    start: int = 0,
    max_words: int = 400,
    overlap: int = 50,
    concurrency: int = 1,
    prefetch: bool = True,
    prefetch_queue_size: int = 24,
    state_path: str | None = None,
    stop_event: threading.Event | None = None,
) -> dict:
    """Resume-safe ZIM ingest.

    Crash-safety: chunks are written to ``wiki_chunks`` before the
    article-level marker is written to ``wiki_articles``. This means a
    kill at any point either leaves the article entirely unprocessed (next
    run re-ingests it from scratch) or fully processed. There is no state
    where ``wiki_articles`` claims a content_hash but the corresponding
    chunks are missing.

    ``state_path`` (optional): JSON file updated atomically after every
    flush with the resume cursor (last fully-committed ZIM entry index)
    and running counters. Use it for monitoring and to pass ``--start
    <next_zim_index>`` on a manual restart."""
    existing_hashes = load_existing_article_hashes(store) if resume else {}
    stats = {
        "articles_seen": 0,
        "articles_skipped_unchanged": 0,
        "articles_replaced": 0,
        "articles_new": 0,
        "articles_empty": 0,
        "chunks_written": 0,
        "started_at": time.time(),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zim_path": zim_path,
        "start_index": start,
        "last_committed_index": max(start - 1, -1),
        "completed": False,
    }
    # Write a starter state immediately so status queries before the first
    # flush still find the resume cursor and elapsed time.
    _atomic_write_state(state_path, stats)
    last_report = time.time()
    # Each pending entry: {"article": {...}, "chunks": [...]}
    # The article marker is only persisted after its chunks have been
    # committed to wiki_chunks, so crash-recovery is clean.
    pending: list[dict] = []

    def flush_pending() -> None:
        if not pending:
            return
        chunk_rows: list[dict] = []
        for p in pending:
            chunk_rows.extend(p["chunks"])
        texts = [c["content"] for c in chunk_rows]
        if concurrency > 1:
            vectors = parallel_embed(embedder, texts, concurrency=concurrency)
        else:
            vectors = robust_embed_batch(embedder, texts)
        keepers = []
        per_article_kept: dict[str, int] = {}
        for c, vec in zip(chunk_rows, vectors):
            if vec is None:
                stats.setdefault("chunks_unembeddable", 0)
                stats["chunks_unembeddable"] += 1
                continue
            c["vector"] = vec
            keepers.append(c)
            per_article_kept[c["article_id"]] = per_article_kept.get(c["article_id"], 0) + 1
        if keepers:
            # Chunks first, then article markers.
            store.add_wiki_chunks(keepers)
            stats["chunks_written"] += len(keepers)
        # Batch the article-marker writes. The pre-refactor code called
        # upsert_wiki_article per article — each one did a Lance
        # delete+add and created a new dataset fragment, costing ~1.5s
        # per article at scale (95% of total ingest wall time, profiled).
        # Now: one batch IN-list delete for replaces, one batch add for
        # everything.
        survivors = [
            p for p in pending
            if not (per_article_kept.get(p["article"]["article_id"], 0) == 0 and p["chunks"])
        ]
        replace_ids = [
            p["article"]["article_id"] for p in survivors if p["is_replace"]
        ]
        if replace_ids:
            store.delete_wiki_articles(replace_ids)
        if survivors:
            store.add_new_wiki_articles(
                (p["article"] for p in survivors), lang=lang,
            )
            for p in survivors:
                existing_hashes[p["article"]["article_id"]] = p["article"]["content_hash"]
        stats["last_committed_index"] = max(
            stats["last_committed_index"],
            max((p["entry_index"] for p in pending), default=stats["last_committed_index"]),
        )
        pending.clear()
        _atomic_write_state(state_path, stats)

    # Producer/consumer split: the parse stage (lxml is CPU-bound but
    # releases the GIL during its C work) runs in a background thread so
    # CPU parse can overlap with the GPU embed in the main thread. The
    # earlier ProcessPoolExecutor attempt was 8x slower because of pipe
    # IPC; threads avoid that. Production ratio is ~25% parse / 70%
    # embed, so this hides the parse cost behind the embed and should
    # buy ~+25% throughput.
    #
    # When `prefetch=False`, parsing happens inline in the main thread
    # (original behavior) — useful as a debug fallback.
    SENTINEL = object()
    produced: "queue.Queue[object]" = queue.Queue(maxsize=max(2, prefetch_queue_size))
    producer_stop = threading.Event()

    def _produce() -> None:
        try:
            for art in iter_zim_articles(zim_path, lang=lang, start=start):
                if producer_stop.is_set():
                    break
                existing = existing_hashes.get(art.article_id)
                if existing == art.content_hash:
                    produced.put(("skip", art, None, None))
                    continue
                sections = parse_article_html(art.html)
                chunks = chunk_sections(
                    sections, art.title, max_words=max_words, overlap=overlap,
                ) if sections else []
                produced.put(("article", art, bool(existing), chunks))
        except Exception as exc:
            produced.put(("error", str(exc), None, None))
        finally:
            produced.put(SENTINEL)

    producer_thread: threading.Thread | None = None
    if prefetch:
        producer_thread = threading.Thread(target=_produce, name="zim-prefetch", daemon=True)
        producer_thread.start()
        iter_inline = None
    else:
        iter_inline = iter_zim_articles(zim_path, lang=lang, start=start)

    def _next_item():
        """Returns (kind, art, is_replace, chunks) or None on exhaustion."""
        if prefetch:
            item = produced.get()
            if item is SENTINEL:
                return None
            return item
        # Inline path mirrors the producer logic.
        try:
            art = next(iter_inline)
        except StopIteration:
            return None
        existing = existing_hashes.get(art.article_id)
        if existing == art.content_hash:
            return ("skip", art, None, None)
        sections = parse_article_html(art.html)
        chunks = chunk_sections(
            sections, art.title, max_words=max_words, overlap=overlap,
        ) if sections else []
        return ("article", art, bool(existing), chunks)

    try:
        while True:
            # Article-boundary stop check. The supervisor sends SIGTERM
            # to a busy ingest worker; if the signal arrives mid-flush,
            # Python finishes the bytecode step it was on (which means
            # LanceDB writes complete atomically) and runs the handler
            # right after. The handler sets stop_event; the next loop
            # iteration here observes it and breaks BEFORE pulling
            # another article — guaranteeing the `finally` block flushes
            # the pending batch atomically, no orphaned chunks.
            if stop_event is not None and stop_event.is_set():
                print("[ingest_wiki] stop_event set — exiting main loop", flush=True)
                break
            item = _next_item()
            if item is None:
                break
            kind, payload, is_replace, chunks = item
            if kind == "error":
                print(f"[ingest_wiki] producer error: {payload}", flush=True)
                continue
            art = payload
            stats["articles_seen"] += 1
            if kind == "skip":
                stats["articles_skipped_unchanged"] += 1
                stats["last_committed_index"] = max(
                    stats["last_committed_index"], art.entry_index
                )
            elif not chunks:
                stats["articles_empty"] += 1
                stats["last_committed_index"] = max(
                    stats["last_committed_index"], art.entry_index
                )
            else:
                if is_replace:
                    store._tables["wiki_chunks"].delete(
                        f"article_id = '{art.article_id.replace(chr(39), chr(39)+chr(39))}'"
                    )
                    stats["articles_replaced"] += 1
                else:
                    stats["articles_new"] += 1
                pending.append({
                    "entry_index": art.entry_index,
                    "is_replace": is_replace,
                    "article": {
                        "article_id": art.article_id,
                        "title": art.title,
                        "slug": art.slug,
                        "content_hash": art.content_hash,
                    },
                    "chunks": [
                        {
                            "article_id": art.article_id,
                            "title": art.title,
                            "section_path": c.section_path,
                            "chunk_idx": c.chunk_idx,
                            "content": c.content,
                            "content_hash": c.content_hash,
                        }
                        for c in chunks
                    ],
                })
                total_pending_chunks = sum(len(p["chunks"]) for p in pending)
                if total_pending_chunks >= batch_size:
                    flush_pending()

            if progress and time.time() - last_report > 5.0:
                elapsed = max(time.time() - stats["started_at"], 1.0)
                rate = stats["articles_seen"] / elapsed
                ingest_rate = (
                    (stats["articles_new"] + stats["articles_replaced"]) / elapsed
                )
                qsize = produced.qsize() if prefetch else 0
                print(
                    f"[ingest_wiki] seen={stats['articles_seen']} "
                    f"new={stats['articles_new']} "
                    f"replaced={stats['articles_replaced']} "
                    f"skipped={stats['articles_skipped_unchanged']} "
                    f"empty={stats['articles_empty']} "
                    f"chunks={stats['chunks_written']} "
                    f"scan={rate:.1f}/s ingest={ingest_rate:.1f}/s "
                    f"queue={qsize}/{prefetch_queue_size} "
                    f"cursor={stats['last_committed_index']} "
                    f"elapsed={int(elapsed)}s",
                    flush=True,
                )
                last_report = time.time()
                _atomic_write_state(state_path, stats)

            if limit and stats["articles_seen"] >= limit:
                break
    finally:
        if producer_thread is not None:
            producer_stop.set()
            # Drain quickly so the producer thread isn't blocked on
            # `produced.put()` and can observe the stop flag.
            try:
                while True:
                    produced.get_nowait()
            except queue.Empty:
                pass
            producer_thread.join(timeout=10)

    flush_pending()
    stats["elapsed_s"] = round(time.time() - stats["started_at"], 2)
    # ZIM is only "completed" when we walked off the end of the iterator
    # — NOT when the supervisor's SIGTERM caused us to exit at an
    # article boundary. Marking completed=True on signal exit makes the
    # supervisor go dormant and the corpus never finishes ingesting.
    stats["completed"] = not (stop_event is not None and stop_event.is_set())
    _atomic_write_state(state_path, stats)
    # Build/refresh the ANN index on completion if the corpus is large
    # enough to benefit. Cheap if the index already exists (LanceDB
    # rebuilds with `replace=True`); skipped entirely for tiny corpora.
    try:
        rows = store._tables["wiki_chunks"].count_rows()
        if rows >= 50_000 and not store.wiki_index_status().get("has_vector_index"):
            idx_t0 = time.time()
            print(f"[ingest_wiki] building IVF_PQ index on {rows:,} chunks…", flush=True)
            idx_result = store.build_wiki_index()
            stats["index_built"] = idx_result
            stats["index_build_elapsed_s"] = round(time.time() - idx_t0, 2)
            _atomic_write_state(state_path, stats)
            print(
                f"[ingest_wiki] index built in {stats['index_build_elapsed_s']}s",
                flush=True,
            )
    except Exception as exc:
        stats["index_build_error"] = str(exc)
        print(f"[ingest_wiki] index build failed: {exc}", flush=True)
        _atomic_write_state(state_path, stats)
    return stats


# --- CLI --------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m world.ingest_wiki")
    p.add_argument("zim_path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lang", default="en")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-words", type=int, default=400)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel embed workers. Ollama serves concurrent "
                        "embed requests; 4-8 typically saturates the GPU "
                        "with bge-m3. Ignored with --embedder native.")
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                   help="Disable the threaded ZIM iterator/parser prefetch. "
                        "By default, parse runs in a background thread so "
                        "lxml CPU work overlaps with GPU embed.")
    p.add_argument("--prefetch-queue", type=int, default=24,
                   help="Max parsed articles buffered between the parser "
                        "thread and the main embed loop.")
    p.add_argument("--embedder", choices=["ollama", "native"], default="native",
                   help="ollama: route embeds through the chat-runtime "
                        "Ollama instance. native: sentence-transformers "
                        "bge-m3 directly on the GPU (~20x throughput).")
    p.add_argument("--state-file", default=None,
                   help="JSON file updated after every flush with the "
                        "resume cursor + counters. Read it for live "
                        "progress; pass with --resume-from-state to "
                        "restart from where the last run left off.")
    p.add_argument("--resume-from-state", action="store_true",
                   help="If --state-file exists and has last_committed_index, "
                        "start from index+1 instead of --start.")
    return p


def _acquire_singleton_lock(state_path: str | None) -> int:
    """Acquire an exclusive flock so only one ingest_wiki can ever run
    at a time. Returns the lockfile FD on success — keep it open for
    the lifetime of the process; the OS releases the lock when the FD
    closes (clean exit, signal exit, crash, SIGKILL — all fine).

    Raises ``RuntimeError`` when another instance holds the lock. The
    caller should print the message and exit non-zero so the supervisor
    (or whoever spawned this) doesn't think the start succeeded.

    Why a file lock and not a pidfile: pidfiles go stale on crash and
    require manual cleanup; flock is automatic. Why not check the
    supervisor's tracked child PID: chat-triggered starts and manual
    CLI runs spawn outside the supervisor's tracking, and we just
    burned ourselves twice on exactly that race.
    """
    if state_path:
        lock_path = Path(state_path).with_suffix(".lock")
    else:
        lock_path = Path("/tmp/ingest_wiki.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            holder = os.read(fd, 256).decode("utf-8", "replace").strip()
        except Exception:
            holder = ""
        os.close(fd)
        raise RuntimeError(
            f"another ingest_wiki is already running (lock {lock_path}; "
            f"holder={holder or 'unknown'}). Refusing to start a second copy."
        )
    # Stamp diagnostic info — purely informational; the flock is the
    # actual mutex.
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} started_at={time.time():.0f}\n".encode("utf-8"))
    except Exception:
        pass
    return fd


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not os.path.exists(args.zim_path):
        print(f"error: zim not found: {args.zim_path}", file=sys.stderr)
        return 2

    # Singleton lock — see _acquire_singleton_lock. Held for the whole
    # process lifetime; the local name keeps the fd referenced so the
    # GC can't accidentally close it.
    try:
        _ingest_lock_fd = _acquire_singleton_lock(args.state_file)  # noqa: F841
    except RuntimeError as exc:
        print(f"[ingest_wiki] {exc}", file=sys.stderr, flush=True)
        return 3

    # Graceful-stop handler so SIGTERM (sent by the supervisor on
    # busy-pause) is observed at an article boundary rather than
    # interrupting a flush mid-write — the latter can orphan chunks
    # whose article markers never land. The handler does the minimum
    # possible (sets the flag); the main loop reads it between articles
    # and the ``finally`` in ingest_zim then runs a final atomic flush.
    import signal
    stop_event = threading.Event()

    def _on_stop(signum, _frame):
        if not stop_event.is_set():
            print(f"[ingest_wiki] received signal {signum} — will stop "
                  "at next article boundary", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    # Resolve the resume cursor FIRST — before overwriting the state
    # file with the starter stub. Previously the stub wrote
    # last_committed_index=-1 on every start and then the resume code
    # below would re-read that fresh -1, treating every run as a
    # restart from index 0. That's why a ~700k-cursor scan would
    # re-scan the same 700k articles on every supervisor relaunch.
    start = args.start
    resume_cursor = -1
    if args.resume_from_state and args.state_file and os.path.exists(args.state_file):
        try:
            with open(args.state_file) as fh:
                prior = json.load(fh)
            resume_cursor = int(prior.get("last_committed_index", -1))
            if resume_cursor >= 0:
                start = resume_cursor + 1
                print(
                    f"[ingest_wiki] resuming from state cursor={resume_cursor} -> "
                    f"start={start}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ingest_wiki] state file unreadable, ignoring: {exc}", flush=True)

    # Pre-load state stub so chat status queries during model-load (~15s
    # for native bge-m3) report progress correctly even before the first
    # batch flush. Stub gets overwritten on the first real flush.
    # IMPORTANT: preserve last_committed_index from the resume read so
    # any concurrent reader (or a crash before the first flush) doesn't
    # see -1 and conclude no progress was made.
    if args.state_file:
        _atomic_write_state(args.state_file, {
            "phase": "starting",
            "zim_path": args.zim_path,
            "started_at": time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "articles_seen": 0,
            "chunks_written": 0,
            "start_index": start,
            "last_committed_index": resume_cursor,
            "completed": False,
        })

    # Local imports so the module is importable without the rest of the
    # vault runtime configured (e.g. inside tests with stub embedders).
    if args.embedder == "native":
        from world.native_embedder import NativeEmbedder
        embedder = NativeEmbedder()
        print(f"[ingest_wiki] native embedder ready: {embedder.model_name} on {embedder.device}", flush=True)
    else:
        from models import get_model
        embedder = get_model("embed")
        print("[ingest_wiki] ollama embedder", flush=True)
    store = WorldKnowledgeStore(text_embedder=embedder)
    stats = ingest_zim(
        args.zim_path,
        store,
        embedder,
        lang=args.lang,
        limit=args.limit,
        resume=args.resume,
        batch_size=args.batch,
        start=start,
        max_words=args.max_words,
        overlap=args.overlap,
        concurrency=args.concurrency,
        prefetch=args.prefetch,
        prefetch_queue_size=args.prefetch_queue,
        state_path=args.state_file,
        stop_event=stop_event,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
