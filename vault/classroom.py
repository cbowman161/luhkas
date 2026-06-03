"""Classroom mode — pedagogical conversational loop over the world DB.

A ``Lesson`` is a skeleton of modules generated up-front from the wiki
vector store plus reasoner inference; each module's step-by-step
micro-plan is expanded just-in-time when the user enters that module,
so the model can adapt the plan to how the user has actually been
answering. Progress lives in ``ChatSession.mode_state`` so a resume
after any gap puts the user exactly where they left off.

While a session has ``mode == "classroom"``, every user turn routes
through ``ClassroomController.handle_turn`` instead of the planner.
The model gets a per-turn prompt that pins the role, the current step,
the on-topic discipline, and a small reference block of wiki chunks
relevant to the step. The model's response includes a tiny structured
tail (``{"check": "pass"|"retry", "kind": "mc"|"free"}``) when a
comprehension check has just been answered, which the controller uses
to advance the cursor.

Exit phrases ("end class", "pause classroom", etc.) are matched
deterministically inside ``handle_turn`` so the user is never trapped
in the loop even if the model goes sideways.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import CLASSROOM_DIR
from models import evict_model, get_model, warm_model_role


log = logging.getLogger("vault.classroom")


# Deterministic escape hatches. Matched against a normalized (lowercased,
# punctuation-stripped, whitespace-collapsed) form of the user's input.
# Kept small and obvious so the user always has a way out even if the
# model misbehaves.
END_PHRASES = {
    "end class", "end classroom", "end the class", "end the lesson",
    "end lesson", "stop the lesson", "stop the class", "stop class",
    "stop classroom", "exit class", "exit classroom", "exit the class",
    "exit the lesson", "leave class", "leave classroom", "quit class",
    "quit classroom", "done with class", "done with the lesson",
    "im done with class", "i am done with class", "finish class",
    "finish the lesson", "close class", "close classroom",
}
PAUSE_PHRASES = {
    "pause class", "pause classroom", "pause the lesson", "pause lesson",
    "park the lesson", "park class", "hold the lesson",
}

# Roles to evict from VRAM while a classroom session is active. The
# teacher model is large (~20 GB Q4) and needs the headroom on a 24 GB
# card to keep bge-m3 resident for per-turn retrieval. Embed is NOT in
# this list — we need it every turn. Teacher is not in this list — we
# need it loaded.
BENCH_ROLES = ("router", "chat", "vision", "coder", "fast_coder")
# Roles to re-warm after the classroom session ends. Matches the default
# VAULT_WARM_MODEL_ROLES set so post-classroom feels like pre-classroom.
RESTORE_ROLES = ("router", "chat")
# Caps on the tutor callback memory carried in mode_state. Keeps the
# per-turn prompt from ballooning over a long session.
MAX_RECENT_MISSES = 6
MAX_LEARNER_NOTES = 6


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_CLASS_PREFIX_RE = re.compile(
    r"^\s*CLASS\s*:\s*(ON_TOPIC|CLARIFYING|META|OFF_TOPIC)\b\s*\n",
    re.IGNORECASE,
)


def _strip_class_prefix(text: str) -> tuple[str, str | None]:
    """Pull a leading ``CLASS: <category>`` line off a teacher response.

    Returns ``(body, classification_or_None)``. The classification is
    the structural advance/redirect signal — putting it at the start of
    the response (rather than as a trailing JSON tail) means it survives
    the common failure mode where the model runs out of token budget
    before closing out a trailing structured block.
    """
    if not text:
        return text, None
    match = _CLASS_PREFIX_RE.match(text)
    if not match:
        return text, None
    body = text[match.end():].lstrip()
    return body, match.group(1).upper()


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a model response.

    Defensive: Ollama with ``think=True`` is supposed to route thinking
    into a separate ``thinking`` field, but qwen3 sometimes still emits
    the block inline. Also handles an unclosed leading think tag (rare
    truncation case) by dropping the prefix up to ``</think>`` when
    present.
    """
    if not text:
        return text
    # Drop balanced <think>...</think> blocks.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Handle the case where thinking leaked without an opening tag but
    # closed normally — strip up to and including </think>.
    if "</think>" in cleaned and "<think>" not in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].lstrip()
    return cleaned.strip()


def _strip_json_tail(text: str) -> tuple[str, dict | None]:
    """Pull a trailing ``{...}`` JSON object off the end of a model
    response, if present. Returns (body, tail_dict_or_None)."""
    if not text:
        return "", None
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return text, None
    # Walk back to the matching opening brace.
    depth = 0
    start = -1
    for i in range(len(stripped) - 1, -1, -1):
        ch = stripped[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return text, None
    tail = stripped[start:]
    try:
        parsed = json.loads(tail)
    except (json.JSONDecodeError, ValueError):
        return text, None
    if not isinstance(parsed, dict):
        return text, None
    body = stripped[:start].rstrip()
    return body, parsed


@dataclass
class Module:
    title: str
    objectives: list[str]
    sources: list[str] = field(default_factory=list)  # wiki chunk_ids
    steps: list[dict] | None = None  # JIT-expanded; None until visited
    status: str = "pending"  # pending | in_progress | complete


@dataclass
class Lesson:
    id: str
    subject: str
    scope: str
    created_at: float
    modules: list[Module]

    def to_jsonable(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_jsonable(cls, data: dict) -> "Lesson":
        modules = [Module(**m) for m in data.get("modules", [])]
        return cls(
            id=data["id"],
            subject=data["subject"],
            scope=data.get("scope", ""),
            created_at=float(data.get("created_at", time.time())),
            modules=modules,
        )


class LessonStore:
    """Filesystem-backed lesson persistence."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, lesson_id: str) -> Path:
        return self.root / f"{lesson_id}.json"

    def save(self, lesson: Lesson) -> None:
        try:
            self._path(lesson.id).write_text(
                json.dumps(lesson.to_jsonable(), indent=2)
            )
        except Exception as exc:
            log.warning("lesson save %s failed: %s", lesson.id, exc)

    def load(self, lesson_id: str) -> Lesson | None:
        path = self._path(lesson_id)
        if not path.exists():
            return None
        try:
            return Lesson.from_jsonable(json.loads(path.read_text()))
        except Exception as exc:
            log.warning("lesson load %s failed: %s", lesson_id, exc)
            return None


class ClassroomController:
    """Coordinates lesson planning, progress tracking, and per-turn
    classroom conversation. Construct once per VaultRuntime; thread-safe
    insofar as LessonStore is file-per-lesson and ChatSessionManager
    handles its own locking."""

    def __init__(self, chat_sessions, *, world_store=None,
                 teacher_model=None, reasoner_model=None,
                 lesson_store: LessonStore | None = None):
        self.chat_sessions = chat_sessions
        self._world_store = world_store
        self._teacher_model = teacher_model
        self._reasoner_model = reasoner_model
        self.lesson_store = lesson_store or LessonStore(CLASSROOM_DIR / "lessons")
        # Tracks whether we have evicted the default warm set in favor of
        # the teacher model. Used to ensure restore runs exactly once per
        # bench, regardless of which exit path fires (explicit end,
        # finish, or idle-park).
        self._models_benched = False
        # Hook into chat_sessions so an idle-park sweep also triggers
        # model restore. Cheap: a no-op callback when nothing is benched.
        if hasattr(chat_sessions, "register_park_callback"):
            chat_sessions.register_park_callback(self._on_session_parked)

    # ---- lazy model / store accessors ---------------------------------
    # Lazy so importing this module never triggers Ollama warmup or a
    # LanceDB connection.

    @property
    def teacher_model(self):
        if self._teacher_model is None:
            self._teacher_model = get_model("teacher")
        return self._teacher_model

    @property
    def reasoner_model(self):
        if self._reasoner_model is None:
            self._reasoner_model = get_model("reasoner")
        return self._reasoner_model

    # ---- VRAM bench / restore ----------------------------------------

    def _bench_competing_models(self) -> None:
        """Evict router / chat / vision / coder so the 32B teacher model
        has the headroom. Idempotent — repeated calls are cheap no-ops
        because Ollama just confirms the model isn't loaded."""
        if self._models_benched:
            return
        for role in BENCH_ROLES:
            try:
                evict_model(role)
            except Exception as exc:
                log.warning("evict_model(%s) failed: %s", role, exc)
        self._models_benched = True
        log.info("classroom: benched %s", ", ".join(BENCH_ROLES))

    def _restore_competing_models(self) -> None:
        """Re-warm router + chat after the classroom session ends so the
        next default-mode turn doesn't pay a cold-load. Also evicts the
        teacher to free its ~20 GB.

        Best-effort and non-blocking-on-failure: if Ollama can't reach
        a model, we log and continue rather than tripping the user's
        next turn."""
        if not self._models_benched:
            return
        try:
            evict_model("teacher")
        except Exception as exc:
            log.warning("evict_model(teacher) failed: %s", exc)
        for role in RESTORE_ROLES:
            try:
                warm_model_role(role)
            except Exception as exc:
                log.warning("warm_model_role(%s) failed: %s", role, exc)
        self._models_benched = False
        log.info("classroom: restored %s", ", ".join(RESTORE_ROLES))

    def _on_session_parked(self, node_id: str, session) -> None:
        """ChatSessionManager fires this when sweep_idle parks a session.
        We only react to classroom-mode parks — that's the path that
        leaves models benched without an explicit end_lesson call."""
        if getattr(session, "mode", "default") != "classroom":
            return
        self._restore_competing_models()

    @property
    def world_store(self):
        if self._world_store is None:
            from world.world_store import WorldKnowledgeStore
            self._world_store = WorldKnowledgeStore(text_embedder=get_model("embed"))
        return self._world_store

    # ---- entry points called by VaultRuntime --------------------------

    def start_lesson(self, raw_request: str, node_id: str,
                     identity: str | None = None) -> dict:
        """Extract subject from the user's entry phrase, build the lesson
        skeleton, stamp the active session into classroom mode, and
        return a chat-shaped response that opens the first module."""
        try:
            subject_info = self._extract_subject(raw_request)
        except Exception as exc:
            log.warning("subject extraction failed: %s", exc)
            return self._error_response("I couldn't figure out what subject to teach. "
                                        "Try: 'teach me X' or 'start a classroom on X'.")
        subject = subject_info.get("subject", "").strip()
        scope = subject_info.get("scope", "").strip()
        if not subject:
            return self._error_response("I need a subject — say something like "
                                        "'teach me Python decorators' or 'start a class on photosynthesis'.")

        # Free VRAM for the 32B teacher before any model calls. Done
        # before open_session so the first lesson-planning call (which
        # uses the reasoner) doesn't have to fight router/chat for VRAM.
        self._bench_competing_models()

        # Open a fresh session in classroom mode. open_session closes/parks
        # any prior active session, which is the right behavior here:
        # starting a new lesson cleanly replaces whatever conversation
        # was active.
        session = self.chat_sessions.open_session(
            node_id=node_id,
            identity=identity,
            original_message=raw_request,
            original_route="classroom_start",
        )

        try:
            lesson = self._plan_lesson(subject, scope)
        except Exception as exc:
            log.warning("lesson planning failed for %r: %s", subject, exc)
            # Don't leave the system half-benched if start fails before
            # any teacher work happens.
            self._restore_competing_models()
            return self._error_response(
                f"I couldn't build a lesson plan for '{subject}' just now. "
                f"(Reason: {exc})"
            )
        self.lesson_store.save(lesson)

        mode_state = {
            "lesson_id": lesson.id,
            "module_idx": 0,
            "step_idx": 0,
            "awaiting_check": False,
            "subject": subject,
            # Tutor callback memory. The teacher can emit a "miss" or
            # "note" field in its JSON tail; the controller appends them
            # here (capped) and re-injects them into subsequent prompts
            # so the teacher can naturally call back to earlier moments.
            "recent_misses": [],
            "learner_notes": [],
        }
        if session is not None:
            self.chat_sessions.set_mode(node_id, "classroom", mode_state)

        # Module-0 step expansion is deferred to the first user turn.
        # That's where _classroom_turn already calls _expand_module when
        # module.steps is None. Skipping it here saves ~5-7 s of first-
        # turn latency, and the opener works fine off module objectives.
        opening = self._compose_lesson_opening(lesson, mode_state)
        return self._direct({
            "mode": "direct",
            "message": opening,
            "deterministic": False,
            "deterministic_source": "classroom_start",
            "classroom": {
                "event": "start",
                "lesson_id": lesson.id,
                "subject": subject,
                "module": lesson.modules[0].title,
                "modules": [m.title for m in lesson.modules],
            },
        })

    def resume_lesson(self, node_id: str, identity: str | None = None) -> dict:
        """Reopen the most recent classroom session for this node. Looks
        first at the active session (if it's already classroom), then
        walks parked sessions newest-first."""
        active = self.chat_sessions.get_active(node_id)
        candidate = None
        if active is not None and active.mode == "classroom":
            candidate = active
        else:
            for parked in self.chat_sessions.get_parked(node_id, limit=20):
                if parked.mode == "classroom" and parked.mode_state:
                    candidate = parked
                    break

        if candidate is None:
            return self._error_response(
                "I don't have a paused classroom session to resume. "
                "Start a new one with 'teach me <subject>'."
            )

        mode_state = candidate.mode_state or {}
        lesson_id = mode_state.get("lesson_id")
        lesson = self.lesson_store.load(lesson_id) if lesson_id else None
        if lesson is None:
            return self._error_response(
                "I found a paused classroom session but its lesson plan is gone. "
                "Start a new one with 'teach me <subject>'."
            )

        # Bench competing models again — they may have been restored by
        # the park hook while the session was paused.
        self._bench_competing_models()

        # If the candidate is parked, re-open a fresh active session
        # carrying the same mode_state. open_session closes/parks any
        # currently-active conversation, which is what we want when the
        # user explicitly asks to come back to class.
        if active is None or active.id != candidate.id:
            self.chat_sessions.open_session(
                node_id=node_id,
                identity=identity,
                original_message="(resume classroom)",
                original_route="classroom_resume",
            )
            self.chat_sessions.set_mode(node_id, "classroom", mode_state)

        module_idx = int(mode_state.get("module_idx", 0))
        module = lesson.modules[module_idx] if 0 <= module_idx < len(lesson.modules) else None
        if module is None:
            return self._error_response(
                "The lesson has no more modules — looks like we already finished it."
            )
        module_title = module.title
        step_idx = int(mode_state.get("step_idx", 0))
        return self._direct({
            "mode": "direct",
            "message": (
                f"Picking up the classroom on **{lesson.subject}** — "
                f"module {module_idx + 1}/{len(lesson.modules)}: *{module_title}* "
                f"(step {step_idx + 1}). Ready to continue?"
            ),
            "classroom": {
                "event": "resume",
                "lesson_id": lesson.id,
                "module": module_title,
            },
        })

    def prompt_for_subject(self, node_id: str) -> dict:
        """Open-without-subject flow. Returns a response asking the user
        what they want to learn; the runtime installs a pending state
        (``classroom_subject_prompt``) and the user's next message is
        routed back to ``start_lesson`` via ``resolve_subject_prompt``.

        Deliberately does NOT bench models or open the classroom session
        here — neither happens until we actually have a subject. If the
        user cancels or never replies, we've spent zero VRAM/setup work."""
        return self._direct({
            "mode": "direct",
            "message": "What do you want to learn?",
            "classroom": {"event": "subject_prompt"},
            "pending": {"type": "classroom_subject_prompt", "node_id": node_id},
        })

    def resolve_subject_prompt(self, user_reply: str, node_id: str,
                                identity: str | None = None) -> dict:
        """Called by the runtime when a ``classroom_subject_prompt``
        pending state resolves. Treats the user's reply as the subject
        input. Returns an immediate ack and builds the lesson in a
        background thread so the user isn't staring at a frozen UI for
        20-30 seconds while qwen3:30b loads and reasons. Cancel phrases
        short-circuit with no setup."""
        text = (user_reply or "").strip()
        normalized = _normalize(text)
        cancel_words = {"cancel", "nevermind", "never mind", "stop", "abort",
                        "forget it", "no", "nope", "actually no"}
        if not text or normalized in cancel_words:
            return self._direct({
                "mode": "direct",
                "message": "OK, no classroom started.",
                "classroom": {"event": "subject_prompt_cancelled"},
            })
        return self.start_lesson_async(text, node_id=node_id, identity=identity)

    def start_lesson_async(self, subject_text: str, node_id: str,
                            identity: str | None = None) -> dict:
        """Acknowledge immediately, then build the lesson on a background
        thread. The user gets a "Got it, give me a moment..." reply right
        away; the opener arrives as a pushed alert when the plan + first
        teach beat are ready (typically 15-25 s later).

        The session is opened synchronously into classroom mode with
        ``mode_state.state == "loading"`` so any user turn that arrives
        during the build is intercepted and gets a "still building"
        reply rather than falling through to default chat."""
        subject = (subject_text or "").strip()
        if not subject:
            return self._error_response(
                "I need a subject — say something like 'Python programming' "
                "or 'the Krebs cycle'."
            )

        # Open the session immediately so the loading-state guard in
        # maybe_handle_turn can intercept any user turns that arrive
        # before the opener is ready.
        self.chat_sessions.open_session(
            node_id=node_id,
            identity=identity,
            original_message=subject,
            original_route="classroom_start_async",
        )
        loading_state = {
            "state": "loading",
            "subject": subject,
            "loading_since": time.time(),
        }
        self.chat_sessions.set_mode(node_id, "classroom", loading_state)

        # Spawn the build worker. Daemon thread — outlives this request
        # but exits with the process. Bench happens inside the worker so
        # the immediate ack stays fast.
        threading.Thread(
            target=self._build_lesson_worker,
            args=(subject, node_id, identity),
            daemon=True,
        ).start()

        ack = (
            f"Got it — give me a moment to build a lesson plan on "
            f"**{subject}**. The opener will arrive in about 20-30 seconds."
        )
        return self._direct({
            "mode": "direct",
            "message": ack,
            "classroom": {
                "event": "loading",
                "subject": subject,
            },
        })

    def _build_lesson_worker(self, subject: str, node_id: str,
                              identity: str | None) -> None:
        """Background worker: bench, plan, compose opener, stash result.

        The opener and lesson cursor land in ``mode_state`` so the next
        user turn (which `maybe_handle_turn` intercepts) can deliver
        the opener and transition into normal teaching. On failure the
        session is closed and models are restored so the user isn't
        stuck in a stale loading state."""
        try:
            self._bench_competing_models()
            lesson = self._plan_lesson(subject, scope="")
            self.lesson_store.save(lesson)
            opening = self._compose_lesson_opening(lesson, {
                "subject": subject, "lesson_id": lesson.id,
            })
            mode_state = {
                "lesson_id": lesson.id,
                "module_idx": 0,
                "step_idx": 0,
                "awaiting_check": False,
                "subject": subject,
                "recent_misses": [],
                "learner_notes": [],
                # Stashed so the next user turn delivers the welcome
                # before falling into the teach loop. Cleared on delivery.
                "pending_opener": opening,
                "modules": [m.title for m in lesson.modules],
            }
            self.chat_sessions.update_mode_state(node_id, mode_state)
        except Exception as exc:
            log.warning("async lesson build failed for %r: %s", subject, exc)
            # Stash the error in mode_state so the user's next turn
            # surfaces it (instead of asking them to send a turn into a
            # broken loading state forever).
            self.chat_sessions.update_mode_state(node_id, {
                "state": "error",
                "subject": subject,
                "error": str(exc)[:240],
            })
            # Bail out of classroom mode on the next turn — the
            # error handler in maybe_handle_turn does the close.

    def end_lesson(self, node_id: str) -> dict:
        """Explicitly end the active classroom session and clear mode."""
        active = self.chat_sessions.get_active(node_id)
        if active is None or active.mode != "classroom":
            # Defensive: if a stale bench somehow survived (shouldn't),
            # restore here too.
            self._restore_competing_models()
            return self._direct({
                "mode": "direct",
                "message": "There's no classroom session active to end.",
            })
        subject = (active.mode_state or {}).get("subject", "the lesson")
        # Close with an explicit outcome so the learning aggregator sees
        # the classroom event distinctly from a chat close.
        self.chat_sessions.close(node_id, outcome={
            "action": "classroom_ended",
            "result": {"subject": subject,
                       "mode_state": active.mode_state or {}},
            "learned": [],
        })
        self._restore_competing_models()
        return self._direct({
            "mode": "direct",
            "message": f"Class dismissed — saved your progress on {subject}. "
                       f"Say 'continue the lesson' anytime to pick back up.",
            "classroom": {"event": "end"},
        })

    def maybe_handle_turn(self, user_input: str, node_id: str) -> dict | None:
        """Entry point from VaultRuntime.handle. Returns a response dict if
        an active classroom session should consume this turn; returns None
        if the runtime should fall through to its normal flow.

        Exit/pause phrases are matched first so the user always has a
        deterministic escape hatch. Otherwise the turn is dispatched to
        the chat model with a classroom-mode prompt."""
        active = self.chat_sessions.get_active(node_id)
        if active is None or active.mode != "classroom":
            return None
        # Defensive re-bench: if anything crept back into VRAM between
        # turns (scout chat, supervisor warmup, etc.), evict it before
        # the teacher needs to run. Eviction is cheap when the model
        # isn't loaded — Ollama just acks. Without this, teacher + chat
        # + bge-m3 can blow past 24 GB and bge-m3 fails to load mid-turn.
        self._bench_competing_models()
        # Re-arm the flag even if it was cleared (e.g. by a previous
        # park-callback restore on a different node).
        self._models_benched = True

        normalized = _normalize(user_input)
        if normalized in END_PHRASES:
            return self.end_lesson(node_id)
        if normalized in PAUSE_PHRASES:
            # Park fires the registered park-callback, which calls
            # _restore_competing_models. No explicit restore needed here.
            self.chat_sessions.park(node_id)
            return self._direct({
                "mode": "direct",
                "message": "Pausing the lesson — your progress is saved. "
                           "Say 'continue the lesson' to resume.",
                "classroom": {"event": "pause"},
            })

        mode_state = dict(active.mode_state or {})

        # Background build still in progress — quick reply so the user
        # knows they aren't being ignored. They can ping any message;
        # the moment the worker finishes the next turn delivers the
        # opener via the pending_opener branch below.
        if mode_state.get("state") == "loading":
            since = mode_state.get("loading_since") or active.created_at
            waited = max(0, int(time.time() - since))
            return self._direct({
                "mode": "direct",
                "message": (
                    f"Still building your lesson plan on "
                    f"**{mode_state.get('subject', 'that subject')}** — "
                    f"about {waited}s in. Ping me again in a few seconds."
                ),
                "classroom": {"event": "loading", "waited_s": waited},
            })

        # Background build failed — surface the error, close the
        # session, and restore models. The next user message will go
        # through the default chat path.
        if mode_state.get("state") == "error":
            subject = mode_state.get("subject", "the lesson")
            err = mode_state.get("error", "unknown error")
            self.chat_sessions.close(node_id, outcome={
                "action": "classroom_aborted",
                "result": {"reason": "async_build_failed",
                           "error": err, "subject": subject},
                "learned": [],
            })
            self._restore_competing_models()
            return self._direct({
                "mode": "direct",
                "message": (
                    f"Sorry — I couldn't build a lesson plan on "
                    f"'{subject}' just now. ({err}) Try again with "
                    f"'teach me <subject>' or a different topic."
                ),
                "classroom": {"event": "error", "subject": subject},
            })

        # Background build just finished — deliver the welcome opener
        # on this turn, leave the lesson cursor at module 0 / step 0
        # so the *next* user turn starts the actual teaching. Their
        # current input gets implicitly absorbed (they probably said
        # "ready?" or similar to trigger this delivery).
        opener = mode_state.pop("pending_opener", None)
        if opener:
            modules = mode_state.pop("modules", []) or []
            self.chat_sessions.update_mode_state(node_id, mode_state)
            return self._direct({
                "mode": "direct",
                "message": opener,
                "classroom": {
                    "event": "ready",
                    "lesson_id": mode_state.get("lesson_id"),
                    "subject": mode_state.get("subject"),
                    "modules": modules,
                },
            })

        lesson_id = mode_state.get("lesson_id")
        lesson = self.lesson_store.load(lesson_id) if lesson_id else None
        if lesson is None:
            # Lesson plan vanished mid-session; recover by ending.
            self.chat_sessions.close(node_id, outcome={
                "action": "classroom_aborted",
                "result": {"reason": "missing_lesson"},
                "learned": [],
            })
            self._restore_competing_models()
            return self._error_response(
                "I lost track of your lesson plan. Start a new one with "
                "'teach me <subject>'."
            )

        return self._classroom_turn(user_input, node_id, lesson, mode_state)

    # ---- subject extraction ------------------------------------------

    def _extract_subject(self, raw_request: str) -> dict:
        """Use the reasoner to pull the subject and (optional) scope out
        of the user's entry phrase. Returns
        ``{"subject": str, "scope": str}``."""
        prompt = (
            "You extract the lesson subject from a user request that asks to start a class.\n"
            "Reply ONLY with JSON: {\"subject\": \"<short subject>\", "
            "\"scope\": \"<any scope/level/focus hints, or empty string>\"}.\n"
            "Subject should be a noun phrase: 'Python decorators', 'photosynthesis', "
            "'the French Revolution', not a verb phrase.\n"
            "Examples:\n"
            "  'teach me python decorators' -> {\"subject\": \"Python decorators\", \"scope\": \"\"}\n"
            "  'i want to learn intro chemistry for high school' -> "
            "{\"subject\": \"introductory chemistry\", \"scope\": \"high school level\"}\n"
            "  'start a classroom on the krebs cycle, focus on regulation' -> "
            "{\"subject\": \"the Krebs cycle\", \"scope\": \"focus on regulation\"}\n"
            "\n"
            f"User request: {raw_request}\n"
            "OUTPUT:"
        )
        raw = self.reasoner_model.generate(prompt, think=False, response_format="json")
        data = self._parse_json_object(raw)
        return {
            "subject": str(data.get("subject", "")).strip(),
            "scope": str(data.get("scope", "")).strip(),
        }

    # ---- lesson planning ---------------------------------------------

    def _plan_lesson(self, subject: str, scope: str) -> Lesson:
        """Build a fresh lesson skeleton: 4-8 modules, each with
        objectives and pinned wiki chunk_ids. Module steps are deferred
        to ``_expand_module``."""
        chunks = self._search_subject_chunks(subject, scope, top_k=24)
        sources_block = self._format_sources_for_prompt(chunks)
        scope_note = f"\nScope hints: {scope}" if scope else ""
        prompt = (
            "You are designing a self-paced lesson plan for a single learner.\n"
            f"Subject: {subject}{scope_note}\n\n"
            "Use the REFERENCE excerpts below where they're relevant and accurate. "
            "You may also rely on your own knowledge when a needed concept is "
            "missing from the references, but do not invent facts the learner "
            "could verify and find wrong.\n\n"
            "Produce a skeleton of 4-8 modules ordered from foundations to "
            "advanced. Each module has a short title, 2-4 concrete learning "
            "objectives, and a list of source ids from the references below "
            "that ground its content (can be empty if you're relying on inference).\n\n"
            "REFERENCE excerpts:\n"
            f"{sources_block}\n\n"
            "Reply ONLY with JSON of this exact shape:\n"
            "{\n"
            "  \"modules\": [\n"
            "    {\"title\": \"...\", \"objectives\": [\"...\", \"...\"], \"sources\": [\"<chunk_id>\", ...]}\n"
            "  ]\n"
            "}\n"
            "OUTPUT:"
        )
        # Larger num_predict for plan generation — thinking budget plus a
        # JSON array of 4-8 modules each with objectives + sources can
        # exceed the role default of 2048.
        # think=False because qwen3:30b with both think=True and
        # format=json routes the JSON output into the thinking field
        # and leaves response empty. With think=False + format=json the
        # JSON lands in response where we expect it.
        raw = self.reasoner_model.generate(
            prompt, think=False, response_format="json",
            options={"num_predict": 3500},
        )
        data = self._parse_json_object(raw)
        modules_raw = data.get("modules") or []
        if not isinstance(modules_raw, list) or not modules_raw:
            raise ValueError("planner returned no modules")
        modules = []
        valid_ids = {c.get("chunk_id") or c.get("id") for c in chunks}
        for m in modules_raw[:8]:
            title = str(m.get("title", "")).strip()
            if not title:
                continue
            objectives = [str(o).strip() for o in (m.get("objectives") or []) if str(o).strip()]
            sources = [str(s) for s in (m.get("sources") or []) if str(s) in valid_ids]
            modules.append(Module(
                title=title,
                objectives=objectives[:4],
                sources=sources[:8],
            ))
        if not modules:
            raise ValueError("planner returned no usable modules")
        return Lesson(
            id=str(uuid.uuid4()),
            subject=subject,
            scope=scope,
            created_at=time.time(),
            modules=modules,
        )

    def _expand_module(self, lesson: Lesson, module_idx: int) -> Module:
        """Generate the step-by-step micro-plan for a module: 3-6 teach
        steps followed by 1 comprehension check. Cached on the Module
        and persisted by the caller."""
        if not (0 <= module_idx < len(lesson.modules)):
            raise IndexError(f"module_idx out of range: {module_idx}")
        module = lesson.modules[module_idx]
        if module.steps:
            return module

        # Pull fresh chunks scoped to this module's objectives — more
        # focused than the lesson-level retrieval.
        query = f"{lesson.subject}: {module.title}. " + " ".join(module.objectives)
        chunks = self._search_subject_chunks(query, lesson.scope, top_k=10)
        sources_block = self._format_sources_for_prompt(chunks)
        objectives = "\n".join(f"- {o}" for o in module.objectives) or "(none specified)"
        prompt = (
            f"You are designing the step-by-step micro-plan for one lesson module.\n"
            f"Subject: {lesson.subject}\n"
            f"Module: {module.title}\n"
            f"Objectives:\n{objectives}\n\n"
            "Produce 3-6 teach steps followed by exactly 1 comprehension check. "
            "Each teach step is one focused concept or example that should fit a "
            "short conversational turn. The comprehension check should genuinely "
            "verify that the learner met the objectives — choose multiple-choice "
            "when the material is discrete/factual, or free-response when it's "
            "conceptual/explanatory.\n\n"
            "Use the REFERENCE excerpts where accurate; rely on inference where the "
            "references don't cover something.\n\n"
            "REFERENCE excerpts:\n"
            f"{sources_block}\n\n"
            "Reply ONLY with JSON of this exact shape:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\"kind\": \"teach\", \"text\": \"<what to teach this turn>\"},\n"
            "    ...\n"
            "    {\"kind\": \"check\", \"check_kind\": \"mc\"|\"free\", "
            "\"text\": \"<the question>\", "
            "\"choices\": [\"A...\", \"B...\"] | null, "
            "\"expected\": \"<what a passing answer looks like>\"}\n"
            "  ]\n"
            "}\n"
            "OUTPUT:"
        )
        raw = self.reasoner_model.generate(
            prompt, think=False, response_format="json",
            options={"num_predict": 3000},
        )
        data = self._parse_json_object(raw)
        steps_raw = data.get("steps") or []
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError("module expansion returned no steps")
        cleaned: list[dict] = []
        for s in steps_raw[:8]:
            kind = str(s.get("kind", "teach")).lower()
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            if kind == "check":
                check_kind = str(s.get("check_kind", "free")).lower()
                if check_kind not in {"mc", "free"}:
                    check_kind = "free"
                cleaned.append({
                    "kind": "check",
                    "check_kind": check_kind,
                    "text": text,
                    "choices": s.get("choices") if isinstance(s.get("choices"), list) else None,
                    "expected": str(s.get("expected", "")).strip(),
                })
            else:
                cleaned.append({"kind": "teach", "text": text})
        # Guarantee a final check step.
        if not any(c["kind"] == "check" for c in cleaned):
            cleaned.append({
                "kind": "check",
                "check_kind": "free",
                "text": f"In your own words, summarize the key idea of '{module.title}'.",
                "choices": None,
                "expected": "Learner restates the module's main idea using its objectives.",
            })
        module.steps = cleaned
        module.status = "in_progress"
        return module

    # ---- per-turn classroom chat -------------------------------------

    def _classroom_turn(self, user_input: str, node_id: str,
                         lesson: Lesson, mode_state: dict) -> dict:
        module_idx = int(mode_state.get("module_idx", 0))
        step_idx = int(mode_state.get("step_idx", 0))

        # Guard module bounds (e.g. after a check pass that advanced us
        # past the last module).
        if module_idx >= len(lesson.modules):
            return self._finish_lesson(node_id, lesson)

        module = lesson.modules[module_idx]
        if module.steps is None:
            try:
                self._expand_module(lesson, module_idx)
                self.lesson_store.save(lesson)
            except Exception as exc:
                log.warning("module expansion failed mid-turn: %s", exc)
                return self._error_response(
                    "I couldn't expand the next module right now. Try again, "
                    "or say 'pause class' and 'continue the lesson' later."
                )
        steps = module.steps or []
        if step_idx >= len(steps):
            # Shouldn't normally happen — advance to next module.
            return self._advance_after_check(node_id, lesson, mode_state, passed=True)

        current_step = steps[step_idx]
        awaiting_check = bool(mode_state.get("awaiting_check"))

        # Build the per-turn prompt. The on-topic discipline lives entirely
        # in this prompt — no hard guardrails, just instructions the chat
        # model treats as the system role.
        reference = self._format_module_reference(lesson, module)
        memory_block = self._format_memory_block(mode_state, module)
        prompt = self._compose_turn_prompt(
            lesson=lesson,
            module=module,
            module_idx=module_idx,
            step_idx=step_idx,
            step=current_step,
            awaiting_check=awaiting_check,
            reference=reference,
            memory_block=memory_block,
            user_input=user_input,
        )

        try:
            raw = self.teacher_model.generate(prompt, think=True)
        except Exception as exc:
            log.warning("classroom chat call failed: %s", exc)
            return self._error_response(f"Hit a model error mid-lesson: {exc}")

        cleaned = _strip_thinking(raw)
        cleaned, classification = _strip_class_prefix(cleaned)
        body, tail = _strip_json_tail(cleaned)

        # Decide whether to advance.
        advance = False
        if awaiting_check and isinstance(tail, dict):
            check_result = str(tail.get("check", "")).lower()
            if check_result == "pass":
                advance = True
        elif current_step.get("kind") == "teach":
            # Use the classification prefix (most reliable signal — it's
            # the first thing the model writes). OFF_TOPIC redirects and
            # CLARIFYING / META side-questions hold the cursor; ON_TOPIC
            # advances. Default to advance when the prefix is missing.
            if classification in {"OFF_TOPIC", "CLARIFYING", "META"}:
                advance = False
            else:
                advance = True

        new_state = dict(mode_state)
        # Backfill in case this session predates the memory fields
        # (e.g. resumed from an older parked session).
        new_state.setdefault("recent_misses", [])
        new_state.setdefault("learner_notes", [])
        self._absorb_tail(new_state, module, tail)

        if current_step.get("kind") == "check":
            new_state["awaiting_check"] = not advance
        else:
            new_state["awaiting_check"] = False

        if advance:
            next_step_idx = step_idx + 1
            if next_step_idx >= len(steps):
                # End of module.
                module.status = "complete"
                self.lesson_store.save(lesson)
                new_state["module_idx"] = module_idx + 1
                new_state["step_idx"] = 0
                new_state["awaiting_check"] = False
                if new_state["module_idx"] >= len(lesson.modules):
                    # Persist completion before composing finish message.
                    self.chat_sessions.update_mode_state(node_id, new_state)
                    finish = self._finish_lesson(node_id, lesson)
                    # Prepend the model's response (which likely
                    # congratulated the learner on the check) so it
                    # isn't lost.
                    if body:
                        finish["message"] = body.rstrip() + "\n\n" + finish["message"]
                    return finish
            else:
                new_state["step_idx"] = next_step_idx

        self.chat_sessions.update_mode_state(node_id, new_state)

        return self._direct({
            "mode": "direct",
            "message": body or raw,
            "classroom": {
                "event": "turn",
                "lesson_id": lesson.id,
                "module": module.title,
                "module_idx": new_state.get("module_idx", module_idx),
                "step_idx": new_state.get("step_idx", step_idx),
                "advanced": advance,
                "check": (tail or {}).get("check") if awaiting_check else None,
                "classification": classification,
            },
        })

    def _compose_turn_prompt(self, *, lesson: Lesson, module: Module,
                              module_idx: int, step_idx: int, step: dict,
                              awaiting_check: bool, reference: str,
                              memory_block: str, user_input: str) -> str:
        total_modules = len(lesson.modules)
        step_kind = step.get("kind", "teach")
        step_text = step.get("text", "")

        if step_kind == "check":
            check_kind = step.get("check_kind", "free")
            choices = step.get("choices")
            expected = step.get("expected", "")
            if awaiting_check:
                check_block = (
                    f"You just asked this comprehension check ({check_kind}):\n"
                    f"  {step_text}\n"
                    + (f"  Choices: {choices}\n" if choices else "")
                    + f"What a passing answer looks like: {expected}\n\n"
                    "Grade the learner's answer fairly: minor wording differences "
                    "are fine; the answer passes if it covers the core idea. If "
                    "it's close but incomplete, walk them through the gap, then "
                    "re-pose the question (set check=\"retry\"). If they nailed it, "
                    "give brief positive reinforcement and set check=\"pass\".\n\n"
                    "At the very end of your reply, append a JSON tail on its own line:\n"
                    f"  {{\"check\": \"pass\" or \"retry\", \"kind\": \"{check_kind}\""
                    ", \"miss\": \"<short summary of what they got wrong if retry, else omit>\""
                    ", \"note\": \"<optional one-line observation about the learner>\"}}\n"
                    "The miss field is what fuels later callbacks — keep it to "
                    "one specific sentence (e.g. 'confused glycolysis with the "
                    "Krebs cycle' or 'forgot enzyme regulation step').\n"
                )
            else:
                check_block = (
                    "It's time for the comprehension check for this module. "
                    "Ask the learner this question naturally (don't paste it verbatim "
                    "if you can phrase it better) and wait for their answer:\n"
                    f"  Question: {step_text}\n"
                    + (f"  Choices to present: {choices}\n" if choices else "")
                    + f"  Style: {check_kind}\n"
                    "Do NOT include a JSON tail this turn — the learner hasn't answered yet.\n"
                )
        else:
            check_block = (
                f"Your next teaching step:\n"
                f"  {step_text}\n\n"
                "Teach this in 2-5 short paragraphs of plain conversational language. "
                "End with one short question that invites the learner to engage — "
                "either checking they followed, or inviting their take.\n"
                "Optionally, after your reply, append a JSON tail on its own line "
                "with a single field:\n"
                "  {\"note\": \"<one short observation about this learner>\"}\n"
                "Skip the tail if you have nothing to add. The advance decision is "
                "made from the CLASS prefix on line 1, not from this tail.\n"
            )

        return (
            "You are LUHKAS Brain running a one-on-one classroom session.\n"
            f"Subject: {lesson.subject}\n"
            + (f"Scope: {lesson.scope}\n" if lesson.scope else "")
            + f"Module {module_idx + 1} of {total_modules}: {module.title}\n"
            f"Module objectives:\n"
            + "".join(f"  - {o}\n" for o in module.objectives)
            + f"Step {step_idx + 1} of {len(module.steps or [])} ({step_kind}).\n\n"
            "RESPONSE FORMAT (strict — first line is structural):\n"
            "  Line 1 MUST be exactly: CLASS: <CATEGORY>\n"
            "  where <CATEGORY> is one of ON_TOPIC, CLARIFYING, META, OFF_TOPIC.\n"
            "  Line 2 onwards: your actual reply to the learner, in plain prose.\n\n"
            "Classify the learner's turn first, then write the reply for that "
            "category:\n"
            "  - ON_TOPIC      — a substantive response/question about the subject. "
            "Respond fully, advance the lesson.\n"
            "  - CLARIFYING    — asking you to re-explain or define something covered. "
            "Answer fully (never refuse — these mean the learner is engaged), then "
            "either re-pose the current step or move on; your call.\n"
            "  - META          — about classroom mechanics (skip, repeat, slow down, "
            "where am I). Answer briefly about the mechanic, then re-pose the current "
            "step.\n"
            "  - OFF_TOPIC     — unrelated to the subject. Open with ONE short "
            f"redirect sentence (e.g. \"Let's keep our focus on {lesson.subject} "
            "for now.\"), then continue with the current teaching step (one short "
            "paragraph) and end with the engaging question. Brief is fine — the "
            "learner just diverged, don't drown them in a lecture.\n\n"
            f"{memory_block}"
            f"{check_block}"
            "\nREFERENCE (drawn from the world knowledge base; use where accurate):\n"
            f"{reference}\n\n"
            f"Learner: {user_input}\n"
            "Teacher (start with CLASS: line, then reply directly — no narration):"
        )

    def _format_memory_block(self, mode_state: dict, module: Module) -> str:
        """Render LEARNER MEMORY for the prompt. Empty string when there's
        nothing yet (don't waste tokens on an empty section)."""
        misses = mode_state.get("recent_misses") or []
        notes = mode_state.get("learner_notes") or []
        if not misses and not notes:
            return ""
        lines = ["LEARNER MEMORY (weave callbacks naturally when they fit; don't force them):"]
        if misses:
            lines.append("Recent misses:")
            for m in misses[-MAX_RECENT_MISSES:]:
                where = m.get("module", "")
                miss = m.get("miss", "")
                if where and where != module.title:
                    lines.append(f"  - [{where}] {miss}")
                else:
                    lines.append(f"  - {miss}")
        if notes:
            lines.append("Notes about this learner:")
            for n in notes[-MAX_LEARNER_NOTES:]:
                lines.append(f"  - {n}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _absorb_tail(self, mode_state: dict, module: Module,
                      tail: dict | None) -> None:
        """Pull miss/note fields out of the teacher's structured tail and
        append them to the rolling memory in mode_state. Caps list
        growth and dedupes near-duplicates (case-insensitive substring
        check). Mutates mode_state in place."""
        if not isinstance(tail, dict):
            return
        miss = str(tail.get("miss") or "").strip()
        note = str(tail.get("note") or "").strip()
        if miss:
            misses = list(mode_state.get("recent_misses") or [])
            existing = " ".join(m.get("miss", "").lower() for m in misses)
            if miss.lower() not in existing:
                misses.append({
                    "module": module.title,
                    "miss": miss[:240],
                    "ts": time.time(),
                })
                mode_state["recent_misses"] = misses[-MAX_RECENT_MISSES:]
        if note:
            notes = list(mode_state.get("learner_notes") or [])
            existing = " ".join(n.lower() for n in notes)
            if note.lower() not in existing:
                notes.append(note[:240])
                mode_state["learner_notes"] = notes[-MAX_LEARNER_NOTES:]

    def _compose_lesson_opening(self, lesson: Lesson, mode_state: dict) -> str:
        """Generate a welcome + plan outline + invitation. Does NOT
        teach the first step — that's the job of turn 1, so the user
        doesn't get a duplicate when module 0 is lazy-expanded on
        first turn."""
        module = lesson.modules[0]
        plan_outline = "\n".join(
            f"  {i + 1}. {m.title}" for i, m in enumerate(lesson.modules)
        )
        objectives_line = "; ".join(module.objectives[:2]) if module.objectives else ""
        prompt = (
            "You are LUHKAS Brain. Open a one-on-one classroom session warmly.\n"
            f"Subject: {lesson.subject}"
            + (f" ({lesson.scope})" if lesson.scope else "") + "\n\n"
            f"Planned modules:\n{plan_outline}\n\n"
            f"First module: {module.title}"
            + (f" — goals: {objectives_line}" if objectives_line else "") + "\n\n"
            "Write 3-5 sentences total: (1) a one-line warm welcome that names the "
            "subject, (2) a one-line outline pointing at the module list, (3) name "
            "the first module and what we'll explore there, (4) one short engaging "
            "question that invites the learner to begin (e.g. asking what brought "
            "them here, or what they already know). Do NOT teach the actual content "
            "of module 1 in this opener — that's the next turn's job. Do NOT include "
            "a JSON tail.\n\n"
            "Teacher (respond directly; the next line is your reply, not your reasoning):"
        )
        try:
            raw = self.teacher_model.generate(prompt, think=True)
        except Exception as exc:
            log.warning("opening message generation failed: %s", exc)
            # Fall back to a templated opener so the session still works.
            return (
                f"Starting a classroom on **{lesson.subject}**. "
                f"We'll cover {len(lesson.modules)} modules:\n{plan_outline}\n\n"
                f"Module 1: {module.title}. Ready?"
            )
        body, _ = _strip_json_tail(_strip_thinking(raw))
        return body or raw

    def _finish_lesson(self, node_id: str, lesson: Lesson) -> dict:
        self.chat_sessions.close(node_id, outcome={
            "action": "classroom_completed",
            "result": {"subject": lesson.subject, "lesson_id": lesson.id,
                       "modules": [m.title for m in lesson.modules]},
            "learned": [],
        })
        self._restore_competing_models()
        return self._direct({
            "mode": "direct",
            "message": (
                f"That's the end of the lesson on **{lesson.subject}** — "
                f"you completed all {len(lesson.modules)} modules. "
                f"Say 'teach me <subject>' anytime to start another."
            ),
            "classroom": {"event": "complete", "lesson_id": lesson.id},
        })

    def _advance_after_check(self, node_id: str, lesson: Lesson,
                              mode_state: dict, passed: bool) -> dict:
        # Defensive helper; not currently used because turn handling
        # rolls advance into the main response. Kept for future
        # explicit-grade flows.
        new_state = dict(mode_state)
        if passed:
            new_state["module_idx"] = int(new_state.get("module_idx", 0)) + 1
            new_state["step_idx"] = 0
        new_state["awaiting_check"] = False
        if new_state["module_idx"] >= len(lesson.modules):
            return self._finish_lesson(node_id, lesson)
        self.chat_sessions.update_mode_state(node_id, new_state)
        return {"mode": "direct", "message": "(advancing)", "classroom": {"event": "advance"}}

    # ---- helpers -----------------------------------------------------

    def _search_subject_chunks(self, query: str, scope: str,
                                top_k: int) -> list[dict]:
        try:
            return self.world_store.search_wiki(query, top_k=top_k) or []
        except Exception as exc:
            log.warning("world_store.search_wiki failed for %r: %s", query, exc)
            return []

    def _format_sources_for_prompt(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(no reference excerpts available — rely on your own knowledge)"
        lines = []
        for c in chunks[:12]:
            cid = c.get("chunk_id") or c.get("id") or ""
            title = c.get("title") or ""
            section = c.get("section_path") or ""
            content = (c.get("content") or "").strip().replace("\n", " ")
            if len(content) > 480:
                content = content[:480].rstrip() + "..."
            head = f"[{cid}] {title}"
            if section:
                head += f" — {section}"
            lines.append(f"{head}\n  {content}")
        return "\n".join(lines)

    def _format_module_reference(self, lesson: Lesson, module: Module) -> str:
        # Prefer the module's pinned source chunks; if none, do a fresh
        # search. The fresh-search path costs an embedding call per turn,
        # which is fine for a one-on-one loop and gives the model
        # progressively-relevant context as the conversation drifts
        # within the module.
        query = f"{lesson.subject}: {module.title}. " + " ".join(module.objectives)
        try:
            chunks = self.world_store.search_wiki(query, top_k=4) or []
        except Exception:
            chunks = []
        return self._format_sources_for_prompt(chunks)

    def _parse_json_object(self, raw: str) -> dict:
        """Pull the first balanced JSON object out of a model response.

        Handles three failure modes we actually see in the wild:
        * code-fence wrapping (```json ... ```)
        * trailing prose after a valid object (the "Extra data" error)
        * leading prose before a valid object

        Walks to the first ``{``, then uses ``raw_decode`` so trailing
        content is allowed. Recurses past objects whose contents aren't
        the right shape isn't needed — first balanced object wins.
        """
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        decoder = json.JSONDecoder()
        idx = text.find("{")
        last_err: Exception | None = None
        while idx >= 0:
            try:
                obj, _end = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError as exc:
                last_err = exc
            idx = text.find("{", idx + 1)
        raise ValueError(
            f"no JSON object found in: {raw[:200]!r}"
            + (f" (last error: {last_err})" if last_err else "")
        )

    def _error_response(self, message: str) -> dict:
        return self._direct({
            "mode": "direct",
            "message": message,
            "classroom": {"event": "error"},
        })

    def _direct(self, response: dict) -> dict:
        """Stamp a response so the runtime's _enrich step doesn't re-run
        an LLM composer over it. Classroom messages are produced by the
        teacher model and are already in their final form — composing
        them again strips the JSON tail (clobbering ``advance``) and
        truncates the body."""
        response["compose"] = False
        response["response_composed"] = True
        return response
