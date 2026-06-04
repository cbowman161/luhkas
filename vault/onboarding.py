"""vault/onboarding.py — first-time-user onboarding state machine.

Engages when the system has no primary user yet (boot-time
``awaiting_primary_user`` mode) or when an existing primary user has
initiated transfer to a not-yet-enrolled target.

Goal: actively pursue a primary-user identity by guiding a detected
person through:

    greet → name → confirm_name →
    capture_frontal → capture_left → capture_right →
    capture_up → capture_down →
    designate → [if no: park, exit]
    prefs_name → prefs_tone →
    done (persist primary_user + preferences)

Mirrors ``classroom.py``'s mode pattern: ``maybe_handle_turn()`` is the
runtime entry, the conversation state lives in the active chat_session's
``mode_state``, and the controller is dependency-injected so tests can
substitute fakes for the camera, TTS, and profile store without touching
real hardware.

Step 1 lands the state machine + tests against fake observers/TTS.
Steps 2–5 wire it into ``vault_runtime``, real ``FaceRuntime`` pose
polling, profile-store persistence, and the transfer/unset intent
handlers. The contract this module exposes (the Protocols below) is
what those steps adapt to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import logging
import time


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency Protocols
# ---------------------------------------------------------------------------
#
# Each is small on purpose. The onboarding loop only needs:
#   - to know who the camera currently sees (FaceObserver)
#   - to speak unprompted (TTSChannel)
#   - to persist the result (ProfileStore)
#   - to parse "I'm Alex" from utterances (NameExtractor — just a callable)
#
# Concrete implementations live elsewhere (camera_node FaceRuntime, the
# audio node TTS endpoint, scout/person_memory.PersonMemoryStore, and
# the existing ``_extract_introduction_name`` in scout_integration).
# Wiring happens in step 2 inside vault_runtime; the contracts here
# stay free of those imports so this module is testable in isolation.


class FaceObserver(Protocol):
    """What the onboarding loop needs to know about the camera.
    Node-scoped because the camera is a per-node device — onboarding can
    happen at one node while another is in normal interaction. Adapters
    use ``node_id`` to route to the right camera_node HTTP endpoint."""

    def current_face(self, node_id: str) -> Optional["FaceSnapshot"]:
        """Most recent confirmed face detection on ``node_id``, or None
        if no one is visible there. Should reflect live state —
        onboarding may call this repeatedly within a single turn."""
        ...

    def confirm_pose(self, node_id: str, pose: str, identity_hint: str | None) -> bool:
        """Has the visible face on ``node_id`` held the requested pose
        (frontal/left/right/up/down) recently and at recognition
        confidence high enough to capture a training sample?
        ``identity_hint`` is the in-progress onboarding identity (may be
        ``unknown_*`` if the person hasn't been enrolled yet) so the
        implementation can scope the pose check to the right
        face_group_id."""
        ...

    def capture_reference(self, node_id: str, identity: str, pose: str) -> bool:
        """Save a labelled training-sample image for ``identity`` at the
        current pose on ``node_id``. Returns True on success. Should be
        a no-op + False if no face is currently visible."""
        ...


class TTSChannel(Protocol):
    """System-initiated speech output, routed to the right audio node.
    The onboarding loop calls this during pose-capture prompts and
    acknowledgments — the spoken text is *also* returned in the response
    dict so the runtime can log it on the user-visible transcript."""

    def say(self, node_id: str, text: str) -> None:
        ...


class ProfileStore(Protocol):
    """Persistence surface. The onboarding loop only touches a few keys
    per identity, so the protocol is narrow on purpose. Identity-actions
    (transfer/unset) reuse the same surface."""

    def set_display_name(self, identity: str, display_name: str) -> None: ...

    def set_preference(self, identity: str, key: str, value: Any) -> None: ...

    def designate_primary_user(self, identity: str) -> None: ...

    def unset_primary_user(self) -> None: ...

    def primary_user(self) -> Optional[str]: ...

    def has_profile(self, identity: str) -> bool: ...


# NameExtractor is just a callable; aliased for readability.
NameExtractor = Callable[[str], Optional[str]]


@dataclass
class FaceSnapshot:
    """What ``FaceObserver.current_face()`` returns."""

    identity: str  # may be a placeholder like "unknown_0001" before enroll
    confidence: float
    pose: str  # frontal/left/right/up/down/close/far/unknown
    face_group_id: str | None = None


# ---------------------------------------------------------------------------
# State machine constants
# ---------------------------------------------------------------------------

# Steps in order. The numeric order is what `_advance()` uses, but
# control flow can also branch (e.g. designate=no skips to park).
STEP_GREET = "greet"
STEP_AWAIT_NAME = "await_name"
STEP_CONFIRM_NAME = "confirm_name"
STEP_CAPTURE_FRONTAL = "capture_frontal"
STEP_CAPTURE_LEFT = "capture_left"
STEP_CAPTURE_RIGHT = "capture_right"
STEP_CAPTURE_UP = "capture_up"
STEP_CAPTURE_DOWN = "capture_down"
STEP_AWAIT_DESIGNATE = "await_designate"
STEP_PREFS_NAME = "prefs_name"
STEP_PREFS_TONE = "prefs_tone"
STEP_DONE = "done"

# Pose-capture sequence — head orientations only. close/far are
# distance buckets, not orientations the user can produce on cue.
POSE_SEQUENCE = [
    (STEP_CAPTURE_FRONTAL, "frontal"),
    (STEP_CAPTURE_LEFT, "left"),
    (STEP_CAPTURE_RIGHT, "right"),
    (STEP_CAPTURE_UP, "up"),
    (STEP_CAPTURE_DOWN, "down"),
]
_POSE_BY_STEP = dict(POSE_SEQUENCE)

# Deterministic affirmative/negative phrasing. Matches the spirit of
# router.py's pending-decision parser but kept local to avoid cross-
# module coupling. Onboarding is a low-stakes conversation; this
# coarse matcher is fine.
_AFFIRM = {"yes", "yeah", "yep", "sure", "ok", "okay", "y", "correct", "right", "do it"}
_DENY = {"no", "nope", "nah", "n", "not now", "skip", "cancel"}

# Pose-capture pacing — pose_ticks counter increments on every tick or
# user turn at a pose-capture step. The bg ticker fires at ~500ms cadence
# (see VaultRuntime._onboarding_ticker), so these constants translate to:
#   NUDGE every ~8s ("I didn't see you turn — try again")
#   PARK after ~30s ("looks like that pose didn't land")
# An explicit user turn (maybe_handle_turn) is treated as one tick AND
# always re-speaks the current pose prompt — assumes the user spoke
# because they were confused about what was wanted.
_POSE_TICK_NUDGE = 16
_POSE_TICK_PARK = 60


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class OnboardingController:
    """Run the onboarding conversation. One instance per VaultRuntime;
    not thread-safe at the controller level, but per-node state lives in
    chat_sessions which has its own locking."""

    def __init__(
        self,
        chat_sessions,
        *,
        face_observer: FaceObserver,
        tts: TTSChannel,
        profile_store: ProfileStore,
        name_extractor: NameExtractor,
    ) -> None:
        self.chat_sessions = chat_sessions
        self.face_observer = face_observer
        self.tts = tts
        self.profile_store = profile_store
        self.name_extractor = name_extractor

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def maybe_initiate(self, node_id: str) -> Optional[dict]:
        """Called when the runtime is in ``awaiting_primary_user`` mode
        and a face has been detected. Idempotent: if an onboarding
        session is already active for this node, returns None and lets
        the existing flow continue."""
        active = self.chat_sessions.get_active(node_id)
        if active is not None and active.mode == "onboarding":
            return None
        if self.profile_store.primary_user() is not None:
            # Someone else already became primary between the runtime
            # check and now (race-tolerance, not expected in practice).
            return None
        face = self.face_observer.current_face(node_id)
        if face is None:
            return None  # nothing to greet; runtime will retry next tick

        initial_state = {
            "step": STEP_GREET,
            "identity_placeholder": face.identity,
            "face_group_id": face.face_group_id,
            "started_at": time.time(),
            "pose_ticks": 0,
        }
        self.chat_sessions.set_mode(node_id, "onboarding", initial_state)
        return self._emit_greeting(node_id, initial_state)

    def maybe_handle_turn(self, user_input: str, node_id: str) -> Optional[dict]:
        """Runtime entry for user-input turns. Returns a response dict if
        onboarding is active and should consume this turn; returns None
        to fall through to normal routing."""
        active = self.chat_sessions.get_active(node_id)
        if active is None or active.mode != "onboarding":
            return None

        state = dict(active.mode_state or {})
        step = state.get("step", STEP_GREET)
        normalized = (user_input or "").strip().lower()

        # Universal escape hatch — works at greet, name, and pose
        # capture (so the user can cancel mid-capture without waiting
        # for the timeout). Confirm-name has its own no→retry path;
        # designate has its own no→park-with-enrollment path.
        if normalized in _DENY and step in (
            STEP_GREET, STEP_AWAIT_NAME, *_POSE_BY_STEP.keys()
        ):
            return self._park(node_id, reason="declined")

        # Pose-capture steps share one handler with a force_speak knob;
        # a user turn at a pose step always re-speaks the prompt (the
        # user spoke, so they're confused — re-orient them).
        if step in _POSE_BY_STEP:
            return self._handle_pose_capture(
                user_input, normalized, node_id, state, force_speak=True,
            )

        handler = self._handlers.get(step)
        if handler is None:
            log.warning("onboarding: unknown step %r, parking", step)
            return self._park(node_id, reason="unknown_step")
        return handler(self, user_input, normalized, node_id, state)

    def tick(self, node_id: str) -> Optional[dict]:
        """Background-driver entry. Advances pose-capture without user
        input — the user is silent while holding a pose, so we need a
        way to drive the state machine forward externally. Idempotent
        and cheap when no onboarding session is active or the active
        step is non-pose (e.g., greet/await_name/prefs — those wait
        for user input regardless).

        Returns a response dict iff the tick caused a TTS-emitting
        state change (advancement, nudge, or park). Returns None on
        silent ticks (state counter bumped but no user-facing change)
        so the runtime can log only the events worth logging."""
        active = self.chat_sessions.get_active(node_id)
        if active is None or active.mode != "onboarding":
            return None
        state = dict(active.mode_state or {})
        step = state.get("step")
        if step not in _POSE_BY_STEP:
            return None
        return self._handle_pose_capture(
            "", "", node_id, state, force_speak=False,
        )

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _handle_greet(self, raw, normalized, node_id, state):
        # User responded to the greeting. Any non-"no" answer moves us
        # to name capture. We accept "yes" or just go straight to it.
        if normalized in _DENY:
            return self._park(node_id, reason="declined")
        return self._advance(node_id, state, STEP_AWAIT_NAME,
                             say="What should I call you?")

    def _handle_await_name(self, raw, normalized, node_id, state):
        name = self.name_extractor(raw)
        if not name:
            # Be forgiving — accept a bare name token if the user just
            # said their name without an introduction phrase. Requires
            # exactly one capitalized alpha token (allowing apostrophe
            # and hyphen for names like O'Brien / Mary-Jane). Lower-case
            # bare names fall through to a re-ask, which is the right
            # behavior in a voice interface where STT capitalization is
            # the only signal we have that the user said a proper noun.
            candidate = raw.strip().strip(".!?,").strip()
            tokens = candidate.split()
            if (
                len(tokens) == 1
                and tokens[0].replace("'", "").replace("-", "").isalpha()
                and tokens[0][0].isupper()
            ):
                name = tokens[0]
        if not name:
            return self._stay(node_id, state,
                              say="I didn't catch your name — say 'I'm <name>' or just your name.")
        state["proposed_name"] = name
        return self._advance(node_id, state, STEP_CONFIRM_NAME,
                             say=f"Got it — {name}. Is that right?")

    def _handle_confirm_name(self, raw, normalized, node_id, state):
        if normalized in _AFFIRM:
            # Lock in the identity. The placeholder face_group_id (e.g.
            # "unknown_0001") gets bound to this name during the first
            # pose capture in step 3; for step 1 we just record the
            # display_name on the chosen identity. ``proposed_name`` is
            # the user-facing label (proper case); ``identity`` is the
            # slug used as the storage key.
            display_name = state["proposed_name"]
            identity = _slug_identity(display_name)
            state["identity"] = identity
            state["display_name"] = display_name  # preserved across steps
            state.pop("proposed_name", None)
            self.profile_store.set_display_name(identity, display_name)
            # Move into the pose-capture sequence.
            first_step, first_pose = POSE_SEQUENCE[0]
            return self._advance(
                node_id, state, first_step,
                say=self._pose_prompt(first_pose, first=True),
            )
        if normalized in _DENY:
            state.pop("proposed_name", None)
            return self._advance(node_id, state, STEP_AWAIT_NAME,
                                 say="No problem — what should I call you?")
        # Anything ambiguous: re-ask.
        return self._stay(node_id, state,
                          say="Say 'yes' if I got your name right, or 'no' to correct it.")

    def _handle_pose_capture(self, raw, normalized, node_id, state, *,
                             force_speak: bool = True):
        """Shared handler for all capture_* steps. Polls the observer
        for the current pose; if confirmed, captures and advances to
        the next pose (or to designate when the sequence is done).

        ``force_speak``: when True (a user turn arrived), we always
        re-speak the current prompt because the user spoke and is
        probably confused. When False (background tick), we only nudge
        at ``_POSE_TICK_NUDGE`` intervals so we aren't yelling "look
        left!" twice a second."""
        step = state["step"]
        pose = _POSE_BY_STEP[step]
        identity = state["identity"]

        face = self.face_observer.current_face(node_id)
        if face is None:
            return self._pose_attempt(
                node_id, state, force_speak=force_speak,
                park_reason="lost_face",
                nudge="I can't see you right now — face the camera so I can keep going.",
            )

        if not self.face_observer.confirm_pose(node_id, pose, identity_hint=identity):
            return self._pose_attempt(
                node_id, state, force_speak=force_speak,
                park_reason="pose_timeout",
                nudge=self._pose_prompt(pose, first=False),
            )

        captured = self.face_observer.capture_reference(node_id, identity, pose)
        if not captured:
            return self._pose_attempt(
                node_id, state, force_speak=force_speak,
                park_reason="capture_failed",
                nudge="Almost — hold that pose one more time.",
            )

        # Pose captured; reset the tick counter and move on.
        state["pose_ticks"] = 0
        next_step, next_pose = self._next_pose_step(step)
        if next_step is None:
            return self._advance(node_id, state, STEP_AWAIT_DESIGNATE,
                                 say="Got it. Should I treat you as my primary user from now on?")
        return self._advance(node_id, state, next_step,
                             say=self._pose_prompt(next_pose, first=False))

    def _pose_attempt(self, node_id: str, state: dict, *,
                      force_speak: bool, park_reason: str, nudge: str):
        """Common path when a pose-capture attempt didn't land. Bumps
        the tick counter; parks at the timeout; otherwise either
        nudges (user-driven turn, or scheduled nudge interval) or
        silently updates state (background tick, no fresh nudge)."""
        ticks = state.get("pose_ticks", 0) + 1
        state["pose_ticks"] = ticks
        if ticks >= _POSE_TICK_PARK:
            return self._park(node_id, reason=park_reason)
        # Speak the nudge on user turns OR at the nudge cadence.
        if force_speak or ticks % _POSE_TICK_NUDGE == 0:
            return self._stay(node_id, state, say=nudge)
        # Silent tick: persist counter, no TTS, no response payload.
        self.chat_sessions.update_mode_state(node_id, state)
        return None

    def _handle_await_designate(self, raw, normalized, node_id, state):
        if normalized in _AFFIRM:
            state["designated"] = True
            return self._advance(node_id, state, STEP_PREFS_NAME,
                                 say=f"Great. Want me to call you {state['identity']}, or something else?")
        if normalized in _DENY:
            # Enrollment complete but no primary designation. Park
            # without persisting primary_user.
            return self._park(node_id, reason="declined_designation", finalize_enrollment=True)
        return self._stay(node_id, state,
                          say="Yes or no — should I make you my primary user?")

    def _handle_prefs_name(self, raw, normalized, node_id, state):
        candidate = raw.strip().strip(".!?,").strip()
        existing_display = state.get("display_name") or state["identity"]
        # Same-name affirmations (or saying back the existing name) skip
        # the rename and keep what was captured at confirm_name time.
        if normalized in _AFFIRM or normalized == existing_display.lower():
            display = existing_display
        else:
            display = candidate or existing_display
        state["preferred_name"] = display
        return self._advance(node_id, state, STEP_PREFS_TONE,
                             say="Last thing — do you want me to keep things brief, or more conversational?")

    def _handle_prefs_tone(self, raw, normalized, node_id, state):
        tone = _classify_tone(normalized)
        state["tone"] = tone
        return self._finalize(node_id, state)

    # Step → handler dispatch table. Pose captures all share the same
    # handler since they only differ in which pose to look for.
    _handlers: dict[str, Callable]  # populated below

    # ------------------------------------------------------------------
    # State-transition helpers
    # ------------------------------------------------------------------

    def _advance(self, node_id: str, state: dict, next_step: str, *, say: str) -> dict:
        state["step"] = next_step
        state["pose_ticks"] = 0  # reset on every step transition
        self.chat_sessions.update_mode_state(node_id, state)
        # Speak through the TTS channel here (not just embed in the
        # response). Background-ticker responses never flow back to a
        # user request handler, so the dict alone wouldn't reach the
        # audio node. For user-driven turns the duplicate POST is
        # harmless — audio_node deduplicates by request id.
        if say:
            self.tts.say(node_id, say)
        return self._respond(say, state, event="advanced", step=next_step)

    def _stay(self, node_id: str, state: dict, *, say: str) -> dict:
        self.chat_sessions.update_mode_state(node_id, state)
        if say:
            self.tts.say(node_id, say)
        return self._respond(say, state, event="repeat", step=state["step"])

    def _park(self, node_id: str, *, reason: str, finalize_enrollment: bool = False) -> dict:
        active = self.chat_sessions.get_active(node_id)
        state = dict(active.mode_state or {}) if active else {}
        outcome = {
            "action": "onboarding_parked",
            "result": {"reason": reason, "step_at_park": state.get("step")},
            "learned": [],
        }
        if finalize_enrollment and state.get("identity"):
            outcome["result"]["enrolled_identity"] = state["identity"]
        self.chat_sessions.close(node_id, outcome=outcome)
        message = _park_message(reason)
        if message:
            self.tts.say(node_id, message)
        return {
            "mode": "direct",
            "message": message or "",
            "onboarding": {"event": "parked", "reason": reason},
        }

    def _finalize(self, node_id: str, state: dict) -> dict:
        identity = state["identity"]
        if state.get("preferred_name"):
            self.profile_store.set_display_name(identity, state["preferred_name"])
        if state.get("tone"):
            self.profile_store.set_preference(identity, "tone", state["tone"])
        if state.get("designated"):
            self.profile_store.designate_primary_user(identity)
        state["step"] = STEP_DONE
        outcome = {
            "action": "onboarding_completed",
            "result": {
                "identity": identity,
                "designated_primary": bool(state.get("designated")),
                "preferred_name": state.get("preferred_name"),
                "tone": state.get("tone"),
            },
            "learned": [],
        }
        self.chat_sessions.close(node_id, outcome=outcome)
        message = f"Welcome aboard, {state.get('preferred_name') or identity}."
        self.tts.say(node_id, message)
        return {
            "mode": "direct",
            "message": message,
            "onboarding": {"event": "completed", "identity": identity,
                           "designated_primary": bool(state.get("designated"))},
        }

    def _emit_greeting(self, node_id: str, state: dict) -> dict:
        message = ("Hi — I don't have a primary user yet. "
                   "Mind if I learn who you are?")
        self.tts.say(node_id, message)
        return self._respond(message, state, event="greeting", step=STEP_GREET)

    def _respond(self, message: str, state: dict, *, event: str, step: str) -> dict:
        return {
            "mode": "direct",
            "message": message,
            "onboarding": {"event": event, "step": step},
        }

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_pose_step(current_step: str) -> tuple[str | None, str | None]:
        for idx, (step, pose) in enumerate(POSE_SEQUENCE):
            if step == current_step and idx + 1 < len(POSE_SEQUENCE):
                return POSE_SEQUENCE[idx + 1]
        return None, None

    @staticmethod
    def _pose_prompt(pose: str, first: bool) -> str:
        prompts = {
            "frontal": "Look straight at me for a moment.",
            "left": "Now turn your head to your left.",
            "right": "Now to your right.",
            "up": "Tilt your head up.",
            "down": "And tilt down.",
        }
        text = prompts.get(pose, f"Hold pose: {pose}.")
        if first and pose == "frontal":
            return "Thanks. " + text
        return text


# Late-bound handler table — defined after methods exist so name
# resolution is unambiguous and refactoring doesn't fall over.
OnboardingController._handlers = {
    STEP_GREET: OnboardingController._handle_greet,
    STEP_AWAIT_NAME: OnboardingController._handle_await_name,
    STEP_CONFIRM_NAME: OnboardingController._handle_confirm_name,
    STEP_CAPTURE_FRONTAL: OnboardingController._handle_pose_capture,
    STEP_CAPTURE_LEFT: OnboardingController._handle_pose_capture,
    STEP_CAPTURE_RIGHT: OnboardingController._handle_pose_capture,
    STEP_CAPTURE_UP: OnboardingController._handle_pose_capture,
    STEP_CAPTURE_DOWN: OnboardingController._handle_pose_capture,
    STEP_AWAIT_DESIGNATE: OnboardingController._handle_await_designate,
    STEP_PREFS_NAME: OnboardingController._handle_prefs_name,
    STEP_PREFS_TONE: OnboardingController._handle_prefs_tone,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug_identity(name: str) -> str:
    """Normalize a display name into the identity key used by the face
    recognizer and profile store. Mirrors scout.person_memory._safe_identity
    closely enough that the runtime adapter can pass either through. We
    avoid importing that helper directly so this module stays free of the
    scout package."""
    if not name:
        return "unknown"
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unknown"


def _classify_tone(text: str) -> str:
    """Classify the user's tone preference into one of two canonical
    values: ``brief`` or ``conversational``. Anything ambiguous defaults
    to ``brief`` — least surprise for first-time users."""
    text = text or ""
    if any(word in text for word in ("brief", "short", "concise", "terse", "quick", "tight")):
        return "brief"
    if any(word in text for word in ("conversational", "chatty", "long", "verbose", "warm", "more")):
        return "conversational"
    return "brief"


def _park_message(reason: str) -> str:
    """User-facing copy for park reasons. Empty string means stay
    silent (no TTS output) — used when the user declined and we don't
    want to lecture them."""
    return {
        "declined": "",
        "declined_designation": "Got it — you're enrolled, but not the primary user.",
        "lost_face": "I lost sight of you — we can pick this up later.",
        "pose_timeout": "Looks like that pose didn't land — we can try again later.",
        "capture_failed": "Something went wrong saving the image — we can try later.",
        "unknown_step": "Something got tangled — let's pick this up later.",
    }.get(reason, "")
