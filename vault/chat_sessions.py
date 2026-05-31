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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


log = logging.getLogger("vault.chat_sessions")


SessionState = Literal["open", "awaiting", "parked", "closed"]


# Tunables (env-overridable).
PARKED_PER_NODE_MAX = int(os.environ.get("VAULT_CHAT_SESSIONS_PARKED_MAX", "10"))
PARKED_TTL_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_PARKED_TTL", "86400"))  # 24h
INACTIVITY_CLOSE_SECONDS = float(os.environ.get("VAULT_CHAT_SESSIONS_IDLE_CLOSE", "600"))  # 10 min
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
    # this null; Phase 1C will populate from route classification.
    topic: str | None = None
    topic_summary: str | None = None

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
        # node_id -> {"active": ChatSession|None, "parked": list[ChatSession], "lock": Lock}
        self._nodes: dict[str, dict] = {}
        self._global_lock = threading.Lock()
        self._recovered: set[str] = set()

    # ---- Lock helpers --------------------------------------------------

    def _node_state(self, node_id: str) -> dict:
        with self._global_lock:
            state = self._nodes.get(node_id)
            if state is None:
                state = {"active": None, "parked": [], "lock": threading.Lock()}
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
                # Idle-close vs explicit park decision (Phase 1C will
                # refine this). For 1A: if previous was awaiting, park
                # it (the user moved on without answering); otherwise
                # close it as open-ended.
                if prev.state == "awaiting":
                    prev.state = "parked"
                    state["parked"].insert(0, prev)
                    state["parked"] = state["parked"][:PARKED_PER_NODE_MAX]
                else:
                    prev.state = "closed"
                    prev.closed_at = now
                    if prev.outcome is None:
                        prev.outcome = {"action": "open_ended", "result": {}, "learned": []}
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

    def flag_last_wrong(self, node_id: str, correction_text: str) -> ChatSession | None:
        """Mark the most recent active-or-closed session as having
        produced a response the user just corrected. Writes a
        "user_corrected" outcome (or merges into an existing outcome's
        ``learned`` list) so the eventual learning aggregator can decay
        confidence on whatever that session "learned" — the user
        rejected its result.

        Returns the flagged session, or None if there's nothing to flag.
        """
        if not self.enabled:
            return None
        state = self._node_state(node_id)
        now = time.time()
        with state["lock"]:
            target = state["active"]
            if target is None:
                # Try the most recently closed session by reading the
                # tail of the JSONL (the active sessions we hold in
                # memory are gone after close — they're only on disk).
                # Don't bother for Phase B MVP; if no active session
                # exists, just no-op. The active-session path is the
                # common case.
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
                session.outcome = outcome
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
            return session

    def sweep_idle(self) -> int:
        """Close active sessions that have been idle past the inactivity
        threshold. Call periodically (or on each request) to prevent
        sessions from lingering forever. Returns count closed.
        """
        if not self.enabled:
            return 0
        now = time.time()
        cutoff = now - INACTIVITY_CLOSE_SECONDS
        closed = 0
        with self._global_lock:
            node_ids = list(self._nodes.keys())
        for nid in node_ids:
            state = self._nodes[nid]
            with state["lock"]:
                session = state["active"]
                if session is None:
                    continue
                if session.updated_at < cutoff:
                    session.state = "closed"
                    session.closed_at = now
                    session.updated_at = now
                    if session.outcome is None:
                        session.outcome = {"action": "idle_closed", "result": {}, "learned": []}
                    self._persist(session)
                    state["active"] = None
                    closed += 1
        return closed

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
