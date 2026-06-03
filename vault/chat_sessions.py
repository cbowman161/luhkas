"""Chat session tracking (Phase 1A — shadow observation).

Groups related turns into a ``ChatSession`` so the system has a single
record that holds the user's *original* input plus every clarification
or confirmation turn that resolved it. This is the foundation for:

* Closing the learning loop (the original phrase is never lost, even if
  the per-pending ``_expires_at`` TTL trips mid-conversation).
* "Back to what we were talking about" resumption (Phase 1D).
* Topic switching with parked sessions (Phase 1C).
* A diagnostic JSONL of every interaction the system actually had,
  ready to feed into the learning aggregator (Layer 3).

**Phase 1A behavior is observational only.** Sessions are written to
disk and held in memory; they do not change any dispatch decision.
Toggle off with ``VAULT_CHAT_SESSIONS_ENABLE=0`` if anything misbehaves.

Storage format: ``vault/data/chat_sessions/{node_id}.jsonl``, one JSON
object per line. Each line is the full session state at the moment of
writing; recovery reads all lines and the last one per session_id wins.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


log = logging.getLogger("vault.chat_sessions")


SessionState = Literal["open", "awaiting", "parked", "closed"]


# Tunables (env-overridable).
PARKED_PER_NODE_MAX = int(os.environ.get("VAULT_CHAT_SESSIONS_PARKED_MAX", "10"))
PARKED_TTL_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_PARKED_TTL", "86400"))  # 24h
INACTIVITY_CLOSE_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_IDLE_CLOSE", "600"))  # 10 min
# Classroom sessions auto-park (not close) after this much idle so the
# user can pick the lesson back up with "continue the lesson". Shorter
# than the default close timeout because a classroom session is heavy
# (mode prompt, retrieval, etc.) and a 5-minute silence usually means
# the user has walked away.
CLASSROOM_IDLE_PARK_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_CLASSROOM_IDLE_PARK", "300"))
# Throttle sweep_idle so the hot presence path doesn't walk every node
# on every turn. 30s gives a freshly-idled session at most that long to
# wait beyond the configured timeout before being closed/parked.
SWEEP_IDLE_MIN_INTERVAL_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_SWEEP_INTERVAL", "30"))
ENABLED = os.environ.get("VAULT_CHAT_SESSIONS_ENABLE", "1").lower() not in ("0", "false", "no", "")


@dataclass
class ChatSession:
    """One multi-turn interaction. See module docstring for semantics."""

    id: str
    node_id: str
    identity: str | None
    created_at: float
    updated_at: float
    state: SessionState
    original_message: str
    original_route: str | None = None
    closed_at: float | None = None
    # Indices into the node's NodeSession.turns deque. NOT canonical
    # storage — for joining sessions back to their turn payloads.
    turn_indices: list[int] = field(default_factory=list)
    # Active pending prompt (mirrors vault_runtime's pending_state for
    # this session). None when the session isn't waiting on the user.
    awaiting: dict | None = None
    # Set when state -> "closed". {"action": str, "result": dict,
    # "learned": list[dict]} — sketches what the session accomplished.
    outcome: dict | None = None
    # Heuristic topic key for "back to that" matching. Phase 1A leaves
    # this null; future work may populate from route classification.
    topic: str | None = None
    topic_summary: str | None = None
    # Mode for the conversation. "default" is the normal planner-driven
    # path; "classroom" routes every turn through ClassroomController so
    # the assistant stays on-subject and tracks lesson progress. Mode
    # changes are written through set_mode() so the JSONL stays a faithful
    # log of session evolution.
    mode: str = "default"
    # Mode-specific scratch dict. For classroom: {lesson_id, module_idx,
    # step_idx, last_check}. Free-form so future modes can use it without
    # schema churn.
    mode_state: dict | None = None
    # Promotion-engine capture: each {phrase, route, ts} the dispatcher
    # took during this session. At close time, if outcome is otherwise
    # unset, these get folded into outcome.learned as silent_route
    # entries — the raw signal the learning_promoter (future) will use
    # to count silent successes and promote to deterministic_router
    # after N corroborating uses.
    routes_taken: list[dict] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, data: dict) -> "ChatSession":
        # Tolerate forward-compat additions: drop unknown keys instead
        # of crashing.
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


class ChatSessionManager:
    """Per-node session bookkeeping + JSONL persistence.

    Thread-safe: ThreadingHTTPServer means multiple request handler
    threads can hit this concurrently. Uses one lock per node to keep
    contention low.
    """

    def __init__(self, data_dir: Path, enabled: bool = ENABLED) -> None:
        self.data_dir = Path(data_dir)
        self.enabled = enabled
        if self.enabled:
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                log.warning("could not create %s; disabling: %s", self.data_dir, exc)
                self.enabled = False
        # node_id -> {"active": ChatSession|None, "parked": list[ChatSession],
        #             "recently_closed": deque[ChatSession], "lock": Lock}
        # recently_closed lets the correction handler attach
        # "user_corrected" markers to sessions that have already resolved
        # (e.g., user said "yes" to confirm, system did the thing, NOW
        # user says "no that was wrong" — without recently_closed we'd
        # have nothing in memory to flag).
        self._nodes: dict[str, dict] = {}
        self._global_lock = threading.Lock()
        self._recovered: set[str] = set()
        # Throttle for sweep_idle so it doesn't run on every presence
        # turn (it walks every registered node). 30 s between full
        # sweeps is fine: idle thresholds are minutes and a freshly-
        # idled session waits at most 30 s extra before getting closed.
        self._last_sweep_at: float = 0.0
        # External hooks fired when a session transitions to "parked"
        # (either explicitly via park() or auto via sweep_idle). Used by
        # the classroom controller to restore evicted models when an
        # idle-park happens without a hands-on end_lesson call.
        self._park_callbacks: list = []

    # ---- Lock helpers --------------------------------------------------

    def _node_state(self, node_id: str) -> dict:
        with self._global_lock:
            state = self._nodes.get(node_id)
            if state is None:
                state = {
                    "active": None,
                    "parked": [],
                    "recently_closed": deque(maxlen=5),
                    "lock": threading.Lock(),
                }
                self._nodes[node_id] = state
        # Lazily recover from disk on first access.
        if node_id not in self._recovered:
            self._recover_node(node_id)
        return state

    def _recover_node(self, node_id: str) -> None:
        with self._global_lock:
            if node_id in self._recovered:
                return
            self._recovered.add(node_id)
        if not self.enabled:
            return
        path = self.data_dir / f"{node_id}.jsonl"
        if not path.exists():
            return
        latest: dict[str, ChatSession] = {}
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sess = ChatSession.from_jsonable(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    latest[sess.id] = sess
        except Exception as exc:
            log.warning("recover %s failed: %s", path, exc)
            return
        now = time.time()
        cutoff = now - PARKED_TTL_SECONDS
        active: ChatSession | None = None
        parked: list[ChatSession] = []
        for sess in latest.values():
            if sess.state == "closed":
                continue
            if sess.updated_at < cutoff:
                continue
            if sess.state == "parked":
                parked.append(sess)
            else:
                # "open" / "awaiting" from a previous process — park it.
                # We don't know what the user said next.
                sess.state = "parked"
                parked.append(sess)
        parked.sort(key=lambda s: s.updated_at, reverse=True)
        parked = parked[:PARKED_PER_NODE_MAX]
        state = self._nodes[node_id]
        state["active"] = active
        state["parked"] = parked

    # ---- Park callbacks ------------------------------------------------

    def register_park_callback(self, fn) -> None:
        """Register a ``fn(node_id, session)`` callable that fires when a
        session transitions to parked. Errors in callbacks are logged and
        swallowed so a bad listener can't break session bookkeeping."""
        self._park_callbacks.append(fn)

    def _fire_parked(self, node_id: str, session: "ChatSession") -> None:
        for fn in self._park_callbacks:
            try:
                fn(node_id, session)
            except Exception as exc:
                log.warning("park callback failed: %s", exc)

    # ---- Persistence ---------------------------------------------------

    def _persist(self, session: ChatSession) -> None:
        if not self.enabled:
            return
        path = self.data_dir / f"{session.node_id}.jsonl"
        try:
            with path.open("a") as f:
                f.write(json.dumps(session.to_jsonable(), default=str) + "\n")
        except Exception as exc:
            log.warning("persist %s failed: %s", path, exc)

    # ---- Public API ----------------------------------------------------

    def open_session(
        self,
        node_id: str,
        identity: str | None,
        original_message: str,
        original_route: str | None = None,
    ) -> ChatSession | None:
        """Start a new session. Closes/parks any existing active session
        first. Returns the new session, or None if the manager is disabled.
        """
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        now = time.time()
        with state["lock"]:
            prev = state["active"]
            if prev is not None:
                if prev.state == "awaiting":
                    # User moved on without answering — park (preserves
                    # the awaiting payload for diagnostic / future
                    # resume work).
                    prev.state = "parked"
                    state["parked"].insert(0, prev)
                    state["parked"] = state["parked"][:PARKED_PER_NODE_MAX]
                    self._fire_parked(node_id, prev)
                else:
                    prev.state = "closed"
                    prev.closed_at = now
                    if prev.outcome is None:
                        # Auto-synthesize outcome from routes_taken.
                        # Every silent-routed turn becomes a candidate
                        # for promotion to deterministic; the engine
                        # threshold-counts these to decide.
                        learned = [
                            {"kind": "silent_route",
                             "phrase": r.get("phrase"),
                             "route": r.get("route"),
                             "ts": r.get("ts")}
                            for r in prev.routes_taken
                            if r.get("phrase") and r.get("route")
                        ]
                        action = "silent_routed" if learned else "open_ended"
                        prev.outcome = {"action": action, "result": {}, "learned": learned}
                    # Keep a handle to the just-closed session so
                    # flag_last_wrong (delayed corrections) can find it.
                    state["recently_closed"].append(prev)
                prev.updated_at = now
                self._persist(prev)
            session = ChatSession(
                id=str(uuid.uuid4()),
                node_id=node_id,
                identity=identity,
                created_at=now,
                updated_at=now,
                state="open",
                original_message=original_message,
                original_route=original_route,
            )
            state["active"] = session
            self._persist(session)
            return session

    def get_active(self, node_id: str) -> ChatSession | None:
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        with state["lock"]:
            return state["active"]

    def get_parked(self, node_id: str, limit: int = 10) -> list[ChatSession]:
        if not self.enabled:
            return []
        state = self._node_state(node_id)
        with state["lock"]:
            return list(state["parked"][:limit])

    def set_awaiting(self, node_id: str, awaiting: dict | None) -> None:
        """Record the pending question that ``vault_runtime._set_pending``
        just installed. Shadow-mirror; does not replace pending_state."""
        if not self.enabled:
            return
        state = self._node_state(node_id)
        with state["lock"]:
            session = state["active"]
            if session is None:
                return
            session.awaiting = dict(awaiting) if isinstance(awaiting, dict) else None
            session.state = "awaiting" if session.awaiting else "open"
            session.updated_at = time.time()
            self._persist(session)

    def set_mode(self, node_id: str, mode: str, mode_state: dict | None = None) -> ChatSession | None:
        """Stamp the active session with a mode and optional mode_state.
        Returns the updated session, or None if there's no active session."""
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        with state["lock"]:
            session = state["active"]
            if session is None:
                return None
            session.mode = mode or "default"
            if mode_state is not None:
                session.mode_state = dict(mode_state)
            session.updated_at = time.time()
            self._persist(session)
            return session

    def update_mode_state(self, node_id: str, mode_state: dict) -> ChatSession | None:
        """Replace the active session's mode_state in place."""
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        with state["lock"]:
            session = state["active"]
            if session is None:
                return None
            session.mode_state = dict(mode_state) if mode_state else None
            session.updated_at = time.time()
            self._persist(session)
            return session

    def add_turn(self, node_id: str, turn_index: int) -> None:
        if not self.enabled:
            return
        state = self._node_state(node_id)
        with state["lock"]:
            session = state["active"]
            if session is None:
                return
            session.turn_indices.append(int(turn_index))
            session.updated_at = time.time()
            self._persist(session)

    def record_route(self, node_id: str, phrase: str, route: str) -> None:
        """Capture that the LLM router took ``route`` for ``phrase`` on
        the active session. These accumulate during a session and, at
        close time, become silent_route entries in outcome.learned.

        The future learning_promoter reads these to count silent
        successes per (normalized_phrase → route) and auto-promote to
        deterministic_router after a threshold. Corrections on the same
        session decrement / blacklist.
        """
        if not self.enabled:
            return
        if not phrase or not route:
            return
        state = self._node_state(node_id)
        with state["lock"]:
            sess = state["active"]
            if sess is None:
                return
            sess.routes_taken.append({
                "phrase": str(phrase)[:300],
                "route": str(route),
                "ts": time.time(),
            })
            sess.updated_at = time.time()
            self._persist(sess)

    def flag_last_wrong(self, node_id: str, correction_text: str) -> ChatSession | None:
        """Mark the most recent active-or-recently-closed session as
        having produced a response the user just corrected. Writes a
        ``user_corrected`` marker into the session's outcome so the
        learning_promoter (future Layer 3) can decay confidence on
        whatever that session "learned" — the user rejected its result.

        Checks ``active`` first, then walks ``recently_closed`` (newest
        first). The closed-session lookup is essential for the
        "confirmed-then-corrected" pattern: user says "yes" to confirm
        (which CLOSES the session), then "no that was wrong" arrives
        in the next turn against an already-resolved session.

        Returns the flagged session, or None if nothing to flag.
        """
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        now = time.time()
        with state["lock"]:
            target = state["active"]
            if target is None:
                # Phase B fix: walk recently_closed (newest first) so
                # delayed corrections after a resolution still attach.
                if state["recently_closed"]:
                    target = state["recently_closed"][-1]
            if target is None:
                return None
            existing = target.outcome or {"action": "open_ended", "result": {}, "learned": []}
            corrections = existing.get("corrections") or []
            corrections.append({"at": now, "text": str(correction_text or "").strip()[:300]})
            existing["corrections"] = corrections
            # If the outcome was a confirmation of something, demote it
            # so the aggregator sees the contradiction (confirmed-then-
            # corrected = low confidence).
            if existing.get("action", "").endswith("_confirmed"):
                existing["action"] = existing["action"].replace("_confirmed", "_confirmed_then_corrected")
            target.outcome = existing
            target.updated_at = now
            self._persist(target)
            return target

    def close(self, node_id: str, outcome: dict | None = None) -> ChatSession | None:
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        now = time.time()
        with state["lock"]:
            session = state["active"]
            if session is None:
                return None
            session.state = "closed"
            session.closed_at = now
            session.updated_at = now
            session.awaiting = None
            if outcome is not None:
                # If caller supplied a typed outcome but routes were
                # also captured during this session, fold the
                # silent_route records in so they're not lost.
                session.outcome = outcome
                if session.routes_taken:
                    learned = session.outcome.get("learned") or []
                    for r in session.routes_taken:
                        if r.get("phrase") and r.get("route"):
                            learned.append({
                                "kind": "silent_route",
                                "phrase": r["phrase"],
                                "route": r["route"],
                                "ts": r.get("ts"),
                            })
                    session.outcome["learned"] = learned
            # Push to recently_closed so delayed corrections can find it.
            state["recently_closed"].append(session)
            self._persist(session)
            state["active"] = None
            return session

    def park(self, node_id: str) -> ChatSession | None:
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        now = time.time()
        with state["lock"]:
            session = state["active"]
            if session is None:
                return None
            session.state = "parked"
            session.updated_at = now
            state["parked"].insert(0, session)
            state["parked"] = state["parked"][:PARKED_PER_NODE_MAX]
            state["active"] = None
            self._persist(session)
        self._fire_parked(node_id, session)
        return session

    def sweep_idle(self) -> int:
        """Close (or, for classroom mode, park) active sessions that have
        been idle past the inactivity threshold. Call periodically (or on
        each request) to prevent sessions from lingering forever.

        Classroom sessions use a shorter timeout
        (CLASSROOM_IDLE_PARK_SECONDS, default 5 min) and are parked
        instead of closed so the user can pick the lesson back up later
        with "continue the lesson". Returns the count of sessions
        transitioned out of active.
        """
        if not self.enabled:
            return 0
        now = time.time()
        # Cheap throttle — caller is the hot per-turn path.
        if now - self._last_sweep_at < SWEEP_IDLE_MIN_INTERVAL_SECONDS:
            return 0
        self._last_sweep_at = now
        default_cutoff = now - INACTIVITY_CLOSE_SECONDS
        classroom_cutoff = now - CLASSROOM_IDLE_PARK_SECONDS
        transitioned = 0
        with self._global_lock:
            node_ids = list(self._nodes.keys())
        for nid in node_ids:
            state = self._nodes[nid]
            with state["lock"]:
                session = state["active"]
                if session is None:
                    continue
                if session.mode == "classroom":
                    if session.updated_at < classroom_cutoff:
                        session.state = "parked"
                        session.updated_at = now
                        state["parked"].insert(0, session)
                        state["parked"] = state["parked"][:PARKED_PER_NODE_MAX]
                        state["active"] = None
                        self._persist(session)
                        # Fire outside the lock by stashing for a post-loop
                        # invocation — but for now firing under the lock
                        # is fine because park callbacks are not expected
                        # to call back into ChatSessionManager.
                        self._fire_parked(nid, session)
                        transitioned += 1
                    continue
                if session.updated_at < default_cutoff:
                    session.state = "closed"
                    session.closed_at = now
                    session.updated_at = now
                    if session.outcome is None:
                        session.outcome = {"action": "idle_closed", "result": {}, "learned": []}
                    self._persist(session)
                    state["active"] = None
                    transitioned += 1
        return transitioned

    def aggregate_silent_routes(
        self,
        node_id: str,
        *,
        lookback_days: float = 14.0,
        min_hits: int = 1,
    ) -> list[dict]:
        """Walk the JSONL and aggregate (normalized_phrase, route) silent_route
        records into promotion-candidate rows.

        This is the read-only complement of the (future) learning_promoter.
        Surfaces the live distribution of "what phrase the LLM router
        picked for what route" so you can eyeball which pairs are
        approaching threshold and whether any have been corrected.

        Each row:
          {
            "phrase":        "what is the capital of france",
            "phrase_norm":   "what is the capital of france",
            "route":         "general_question",
            "hits":          3,
            "first_seen":    <ts>,
            "last_seen":     <ts>,
            "corrections":   1,   # sessions that contained this pair AND were corrected
            "sample_session_ids": ["abc12345", ...],
          }

        Sorted by (hits - corrections) desc — i.e., "most-confirmed
        candidates" first. min_hits filters out long-tail noise.
        """
        if not self.enabled:
            return []
        path = self.data_dir / f"{node_id}.jsonl"
        if not path.exists():
            return []
        cutoff = time.time() - (lookback_days * 86400)
        # Last-write-wins per session_id, then aggregate.
        sessions: dict[str, dict] = {}
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sess = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = sess.get("id")
                    if not sid:
                        continue
                    if sess.get("updated_at", 0) < cutoff:
                        continue
                    sessions[sid] = sess
        except Exception as exc:
            log.warning("aggregate read %s failed: %s", path, exc)
            return []
        # Bucket by (normalized_phrase, route).
        candidates: dict[tuple, dict] = {}
        for sid, sess in sessions.items():
            out = sess.get("outcome") or {}
            corrections = out.get("corrections") or []
            corrected = bool(corrections)
            for L in (out.get("learned") or []):
                if L.get("kind") != "silent_route":
                    continue
                phrase = L.get("phrase") or ""
                route = L.get("route") or ""
                if not phrase or not route:
                    continue
                norm = " ".join(phrase.lower().split())  # cheap normalization
                key = (norm, route)
                bucket = candidates.get(key)
                if bucket is None:
                    bucket = {
                        "phrase": phrase,
                        "phrase_norm": norm,
                        "route": route,
                        "hits": 0,
                        "first_seen": L.get("ts") or sess.get("created_at"),
                        "last_seen": L.get("ts") or sess.get("updated_at"),
                        "corrections": 0,
                        "sample_session_ids": [],
                    }
                    candidates[key] = bucket
                bucket["hits"] += 1
                ts = L.get("ts") or sess.get("updated_at") or 0
                if ts and (bucket["first_seen"] is None or ts < bucket["first_seen"]):
                    bucket["first_seen"] = ts
                if ts and (bucket["last_seen"] is None or ts > bucket["last_seen"]):
                    bucket["last_seen"] = ts
                if corrected:
                    bucket["corrections"] += 1
                if len(bucket["sample_session_ids"]) < 3:
                    bucket["sample_session_ids"].append(sid[:8])
        rows = [b for b in candidates.values() if b["hits"] >= min_hits]
        rows.sort(key=lambda b: (b["hits"] - b["corrections"], b["hits"]), reverse=True)
        return rows

    def snapshot(self, node_id: str) -> dict:
        """Diagnostic — return current state for /chat/sessions endpoint
        or debug logging."""
        if not self.enabled:
            return {"enabled": False}
        state = self._node_state(node_id)
        with state["lock"]:
            return {
                "enabled": True,
                "node_id": node_id,
                "active": state["active"].to_jsonable() if state["active"] else None,
                "parked": [s.to_jsonable() for s in state["parked"]],
            }
