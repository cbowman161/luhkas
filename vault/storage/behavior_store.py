"""Behavior-feedback notes — natural-language statements about how the
system should respond, retrieved at model-prompt construction time to
condition responses.

Distinct from MemoryStore (which holds user-stated *facts about
themselves*) because the schema, query pattern, and consolidation
needs are genuinely different:

  - Schema carries ``scope`` (global / route / domain / session),
    ``route_at_capture``, ``category``, ``source`` (explicit/implicit),
    ``apply_count`` — none of which user_facts needs.
  - Searched on **every model call** (to inject relevant notes into the
    prompt), not on user-question events. Mixing the two tables would
    pollute both query directions.
  - ``identity`` here can be the literal string ``"global"`` to mean
    "this note applies regardless of which user is speaking"
    (e.g., "always confirm before deleting"). user_facts has no
    equivalent.

Shares the LanceDB substrate, embedder, and the ``find_conflict_candidates``
shape with MemoryStore — same .lance directory, same bge-m3 1024-dim
embedding model, just a parallel table.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import lancedb
import pyarrow as pa


EMBED_DIM = 1024  # bge-m3, same as vector_store.MemoryStore
TABLE_NAME = "behavior_notes"
DEFAULT_PATH = Path(os.environ.get("VAULT_MEMORY_DB", "/home/vault/vault_data/memory.lance"))

# Sentinel for "applies to every user" — stored literally in the
# identity column so the LanceDB where-clause is a simple disjunction
# rather than a NULL check.
GLOBAL_IDENTITY = "global"

# Valid scope values. ``global``: applies always. ``route``: applies
# only when the active route at retrieval time matches route_at_capture.
# ``domain``: applies only when the current domain matches the note's
# domain field. ``session``: applies only within the conversation it
# was captured in (filtered by caller; we don't model session ids in
# the store).
VALID_SCOPES = frozenset({"global", "route", "domain", "session"})

# Valid category values. Used by the retrieval call site to filter
# (e.g., "only constraint-class notes for risky-action gating").
VALID_CATEGORIES = frozenset({
    "preference",       # "be more concise"
    "constraint",       # "always confirm before deleting"
    "correction",       # "you got X wrong — it was actually Y"
    "capability_hint",  # "I prefer pytest over unittest in this repo"
})

VALID_SOURCES = frozenset({"explicit", "implicit"})


def _norm_identity(identity: str | None) -> str:
    """Normalize identity: empty / None / whitespace → 'global'. The
    sentinel name is reserved; callers shouldn't pass it as a literal
    user identity (would silently merge with global notes)."""
    if not identity:
        return GLOBAL_IDENTITY
    norm = str(identity).strip().lower()
    return norm or GLOBAL_IDENTITY


def _validate(value: str, allowed: Iterable[str], field_name: str, default: str) -> str:
    """Reject obviously-wrong values at write time so bad data doesn't
    silently land in the store. Falls back to ``default`` rather than
    raising — capture paths must not crash the user's turn."""
    if value in allowed:
        return value
    return default


class BehaviorMemoryStore:
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
            # 'global' or a user identity (face-recognized name).
            pa.field("identity", pa.string()),
            # global / route / domain / session.
            pa.field("scope", pa.string()),
            # When scope='route', the active route at capture time.
            # Empty string when scope is global or domain.
            pa.field("route_at_capture", pa.string()),
            # When scope='domain', the domain key (e.g., a repo slug,
            # a knowledge area). Empty string otherwise.
            pa.field("domain", pa.string()),
            # preference / constraint / correction / capability_hint.
            pa.field("category", pa.string()),
            # The user's exact words, lightly cleaned. Natural language
            # is the storage form on purpose — rules calcify, examples
            # compose.
            pa.field("content", pa.string()),
            # explicit (user typed "be more concise") or implicit
            # (system inferred from a correction / a reversed
            # confirmation / a repeated complaint).
            pa.field("source", pa.string()),
            # The triggering context: last response, last action, etc.
            # Free-form; retrieval doesn't query against this, but the
            # consolidation pass uses it to disambiguate similar notes.
            pa.field("source_context", pa.string()),
            pa.field("created_at", pa.float64()),
            pa.field("updated_at", pa.float64()),
            # 1.0 for explicit captures; lower for implicit signals.
            # Re-applies (via update_apply_count) don't change this —
            # it's about capture confidence, not utility.
            pa.field("confidence", pa.float32()),
            # Telemetry: how many times this note has been retrieved
            # *and used* (the caller decides what "used" means; the
            # store just bumps a counter when told). Drives promotion
            # to system-prompt edits in the future Apply layer.
            pa.field("apply_count", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        ])
        return self.db.create_table(TABLE_NAME, schema=schema)

    # ------------------------------------------------------------------
    # Embedding helper (shared shape with MemoryStore so we can swap)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        if not self.embedder:
            raise RuntimeError("BehaviorMemoryStore.embedder is not configured")
        result = self.embedder.embed(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result[0]
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        identity: str | None = None,
        *,
        scope: str = "global",
        route_at_capture: str | None = None,
        domain: str | None = None,
        category: str = "preference",
        source: str = "explicit",
        source_context: str = "",
        confidence: float = 1.0,
        duplicate_distance: float = 0.25,
    ) -> dict[str, Any]:
        """Append a new behavior note. Duplicate-guarded: if a near-
        identical note (cosine distance ≤ ``duplicate_distance``) already
        exists for the same (identity, scope), the existing row is
        returned and its ``updated_at`` is bumped — no second copy is
        written. This is what saves the user from re-correcting the
        same thing twice."""
        content = (content or "").strip()
        if not content:
            return {"ok": False, "error": "empty_content"}
        ident = _norm_identity(identity)
        scope = _validate(scope, VALID_SCOPES, "scope", "global")
        category = _validate(category, VALID_CATEGORIES, "category", "preference")
        source = _validate(source, VALID_SOURCES, "source", "explicit")
        route_at_capture = (route_at_capture or "").strip()
        domain = (domain or "").strip()

        vec = self._embed(content)
        with self._lock:
            existing = (
                self._table.search(vec)
                .where(f"identity = '{_sql_escape(ident)}' AND scope = '{_sql_escape(scope)}'", prefilter=True)
                .limit(1)
                .to_list()
            )
        if existing:
            dist = existing[0].get("_distance")
            if dist is not None and dist <= duplicate_distance:
                row = existing[0]
                # Bump updated_at so consolidation sees recent reinforcement.
                self._touch(row.get("id"))
                return {
                    "ok": True,
                    "duplicate": True,
                    "distance": dist,
                    "record": _strip_vector(row),
                }
        now = time.time()
        record = {
            "id": str(uuid.uuid4()),
            "identity": ident,
            "scope": scope,
            "route_at_capture": route_at_capture,
            "domain": domain,
            "category": category,
            "content": content,
            "source": source,
            "source_context": (source_context or "").strip(),
            "created_at": now,
            "updated_at": now,
            "confidence": float(confidence),
            "apply_count": 0,
            "vector": vec,
        }
        with self._lock:
            self._table.add([record])
        return {"ok": True, "duplicate": False, "record": _strip_vector(record)}

    def retrieve(
        self,
        query: str,
        identity: str | None = None,
        *,
        active_route: str | None = None,
        domain: str | None = None,
        top_k: int = 5,
        distance_max: float | None = 1.5,
        category: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return notes relevant to ``query`` for the current context.

        Scope filter: ``scope='global'`` always passes; ``scope='route'``
        passes only when ``route_at_capture == active_route``; ``scope='domain'``
        passes only when ``domain`` matches; ``scope='session'`` is never
        returned here — session-scoped notes are the caller's
        responsibility (they're transient and the caller knows which
        session is current).

        Identity filter: returns notes for the named identity AND global
        notes. Passing identity=None returns global notes only.

        Ranked by cosine similarity (LanceDB default); recency tie-
        breaking happens at the SQL level via sort. Caller can filter
        further by category if they only want, say, constraints during
        risky-action gating."""
        query = (query or "").strip()
        if not query:
            return []
        ident = _norm_identity(identity)
        vec = self._embed(query)

        clauses: list[str] = []
        # Identity: this user's notes OR global notes. Global notes for
        # GLOBAL_IDENTITY callers are the same row, so we dedupe naturally.
        if ident == GLOBAL_IDENTITY:
            clauses.append(f"identity = '{_sql_escape(GLOBAL_IDENTITY)}'")
        else:
            clauses.append(
                f"(identity = '{_sql_escape(ident)}' OR identity = '{_sql_escape(GLOBAL_IDENTITY)}')"
            )
        # Scope: global always; route iff matches; domain iff matches;
        # session never (caller handles).
        scope_subs = ["scope = 'global'"]
        if active_route:
            scope_subs.append(
                f"(scope = 'route' AND route_at_capture = '{_sql_escape(active_route)}')"
            )
        if domain:
            scope_subs.append(
                f"(scope = 'domain' AND domain = '{_sql_escape(domain)}')"
            )
        clauses.append("(" + " OR ".join(scope_subs) + ")")
        # Category filter (optional).
        if category is not None:
            cats = [category] if isinstance(category, str) else list(category)
            if cats:
                quoted = ", ".join(f"'{_sql_escape(c)}'" for c in cats)
                clauses.append(f"category IN ({quoted})")

        where = " AND ".join(clauses)
        with self._lock:
            res = (
                self._table.search(vec)
                .where(where, prefilter=True)
                .limit(top_k * 2)  # overfetch so distance_max can cut
                .to_list()
            )

        out = []
        for row in res:
            dist = row.get("_distance")
            if distance_max is not None and dist is not None and dist > distance_max:
                continue
            out.append({**_strip_vector(row), "distance": dist})
            if len(out) >= top_k:
                break
        return out

    def find_conflict_candidates(
        self,
        content: str,
        identity: str | None = None,
        *,
        distance_min: float = 0.25,
        distance_max: float = 0.65,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return notes in the same (identity, scope) namespace whose
        content is in the 'close but not identical' band — likely about
        the same subject, possibly contradicting. Designed to be
        followed by an LLM classifier that confirms actual contradiction,
        then a user prompt to resolve.

        Mirrors MemoryStore.find_conflict_candidates exactly; ported
        rather than imported so behavior_store is self-contained."""
        content = (content or "").strip()
        if not content:
            return []
        vec = self._embed(content)
        ident = _norm_identity(identity)
        with self._lock:
            res = (
                self._table.search(vec)
                .where(f"identity = '{_sql_escape(ident)}'", prefilter=True)
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
            out.append({**_strip_vector(row), "distance": dist})
            if len(out) >= top_k:
                break
        return out

    def update_apply_count(self, note_id: str, delta: int = 1) -> bool:
        """Bump apply_count for a note. Called by the Apply layer when
        a retrieved note actually influenced a response. Drives the
        future promotion-to-system-prompt heuristic."""
        if not note_id:
            return False
        with self._lock:
            try:
                rows = self._table.search().where(f"id = '{_sql_escape(note_id)}'").limit(1).to_list()
            except Exception:
                return False
        if not rows:
            return False
        row = rows[0]
        # LanceDB update via delete+add (no in-place mutate API at this
        # version). Preserves the vector so we don't re-embed.
        with self._lock:
            try:
                self._table.delete(f"id = '{_sql_escape(note_id)}'")
            except Exception:
                return False
            row["apply_count"] = int(row.get("apply_count") or 0) + int(delta)
            row["updated_at"] = time.time()
            row.pop("_distance", None)
            self._table.add([row])
        return True

    def delete_by_id(self, note_id: str) -> bool:
        if not note_id:
            return False
        with self._lock:
            try:
                self._table.delete(f"id = '{_sql_escape(note_id)}'")
                return True
            except Exception:
                return False

    def list_for_identity(
        self, identity: str | None, *, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """All notes for an identity (and global notes). Used by
        'what have you learned about me?' surfaces — inspectability is
        a first-class requirement for teachable systems."""
        ident = _norm_identity(identity)
        if ident == GLOBAL_IDENTITY:
            where = f"identity = '{_sql_escape(GLOBAL_IDENTITY)}'"
        else:
            where = (
                f"(identity = '{_sql_escape(ident)}' OR "
                f"identity = '{_sql_escape(GLOBAL_IDENTITY)}')"
            )
        with self._lock:
            rows = (
                self._table.search()
                .where(where)
                .limit(limit)
                .to_list()
            )
        rows.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
        return [_strip_vector(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._table.count_rows()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _touch(self, note_id: str | None) -> None:
        """Bump updated_at on an existing row without changing anything
        else. Used when a duplicate-add hits an existing note —
        consolidation needs to see that the user reinforced this."""
        if not note_id:
            return
        with self._lock:
            try:
                rows = self._table.search().where(f"id = '{_sql_escape(note_id)}'").limit(1).to_list()
            except Exception:
                return
        if not rows:
            return
        row = rows[0]
        with self._lock:
            try:
                self._table.delete(f"id = '{_sql_escape(note_id)}'")
            except Exception:
                return
            row["updated_at"] = time.time()
            row.pop("_distance", None)
            self._table.add([row])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_vector(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "vector" and k != "_distance"}


def _sql_escape(value: str) -> str:
    """Minimal escaping for the where-clauses we build. Identities,
    scopes, and categories are all controlled vocabularies or
    user-confirmed names; this is defense-in-depth, not the primary
    sanitizer (lancedb's parser will reject malformed input outright)."""
    return str(value).replace("'", "''")
