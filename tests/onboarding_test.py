#!/usr/bin/env python3
"""Tests for vault/onboarding.py.

These exercise the state machine in isolation. The real camera /
TTS / profile store wiring lands in steps 2-4; here we use minimal
in-memory fakes that satisfy the Protocols.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from onboarding import (  # noqa: E402
    OnboardingController,
    FaceSnapshot,
    POSE_SEQUENCE,
    STEP_GREET,
    STEP_AWAIT_NAME,
    STEP_CONFIRM_NAME,
    STEP_CAPTURE_FRONTAL,
    STEP_AWAIT_DESIGNATE,
    STEP_PREFS_NAME,
    STEP_PREFS_TONE,
    STEP_DONE,
    _POSE_TICK_NUDGE,
    _POSE_TICK_PARK,
    _slug_identity,
    _classify_tone,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeSession:
    mode: str = "default"
    mode_state: dict | None = None
    closed: bool = False
    outcome: dict | None = None


class FakeChatSessions:
    """In-memory stand-in for ChatSessionManager — only the methods
    onboarding touches."""

    def __init__(self):
        self.sessions: dict[str, _FakeSession] = {}

    def _ensure(self, node_id: str) -> _FakeSession:
        sess = self.sessions.get(node_id)
        if sess is None or sess.closed:
            sess = _FakeSession()
            self.sessions[node_id] = sess
        return sess

    def get_active(self, node_id: str) -> _FakeSession | None:
        sess = self.sessions.get(node_id)
        if sess is None or sess.closed:
            return None
        return sess

    def set_mode(self, node_id: str, mode: str, mode_state: dict | None = None):
        sess = self._ensure(node_id)
        sess.mode = mode
        sess.mode_state = dict(mode_state) if mode_state else None
        return sess

    def update_mode_state(self, node_id: str, mode_state: dict):
        sess = self._ensure(node_id)
        sess.mode_state = dict(mode_state) if mode_state else None
        return sess

    def close(self, node_id: str, outcome: dict | None = None):
        sess = self.sessions.get(node_id)
        if sess is None:
            return None
        sess.closed = True
        sess.outcome = outcome
        return sess


@dataclass
class FakeFaceObserver:
    snapshot: FaceSnapshot | None = field(default_factory=lambda: FaceSnapshot(
        identity="unknown_0001", confidence=0.5, pose="frontal",
        face_group_id="ug_1"))
    poses_confirmable: set[str] = field(default_factory=lambda: {
        "frontal", "left", "right", "up", "down"})
    captures: list[tuple[str, str, str]] = field(default_factory=list)
    capture_succeeds: bool = True

    def current_face(self, node_id: str) -> FaceSnapshot | None:
        return self.snapshot

    def confirm_pose(self, node_id: str, pose: str, identity_hint: str | None) -> bool:
        return pose in self.poses_confirmable

    def capture_reference(self, node_id: str, identity: str, pose: str) -> bool:
        if not self.capture_succeeds:
            return False
        self.captures.append((node_id, identity, pose))
        return True


@dataclass
class FakeTTS:
    spoken: list[tuple[str, str]] = field(default_factory=list)

    def say(self, node_id: str, text: str) -> None:
        self.spoken.append((node_id, text))


@dataclass
class FakeProfileStore:
    display_names: dict[str, str] = field(default_factory=dict)
    preferences: dict[tuple[str, str], object] = field(default_factory=dict)
    primary: str | None = None
    enrolled: set[str] = field(default_factory=set)

    def set_display_name(self, identity: str, display_name: str) -> None:
        self.display_names[identity] = display_name
        self.enrolled.add(identity)

    def set_preference(self, identity: str, key: str, value) -> None:
        self.preferences[(identity, key)] = value
        self.enrolled.add(identity)

    def designate_primary_user(self, identity: str) -> None:
        self.primary = identity

    def unset_primary_user(self) -> None:
        self.primary = None

    def primary_user(self) -> str | None:
        return self.primary

    def has_profile(self, identity: str) -> bool:
        return identity in self.enrolled


def _name_extractor(text: str) -> str | None:
    """Mimics the relevant bits of scout_integration._extract_introduction_name
    closely enough for the state-machine tests. The real one in step 2 has
    extra stopword filtering that doesn't matter here."""
    lowered = text.lower().split()
    tokens = text.split()
    for phrase in (("i", "am"), ("i'm",), ("im",), ("my", "name", "is"), ("call", "me")):
        for i in range(len(lowered) - len(phrase) + 1):
            if tuple(lowered[i:i + len(phrase)]) == phrase and i + len(phrase) < len(tokens):
                cand = tokens[i + len(phrase)].strip(".,!?")
                if cand and cand[0].isalpha():
                    return cand
    return None


def _make_controller(
    *,
    face: FakeFaceObserver | None = None,
    primary: str | None = None,
):
    """Wire up an OnboardingController with fresh fakes. Returns
    (controller, sessions, face, tts, profile)."""
    sessions = FakeChatSessions()
    face = face or FakeFaceObserver()
    tts = FakeTTS()
    profile = FakeProfileStore(primary=primary)
    controller = OnboardingController(
        sessions,
        face_observer=face,
        tts=tts,
        profile_store=profile,
        name_extractor=_name_extractor,
    )
    return controller, sessions, face, tts, profile


def _step(sessions: FakeChatSessions, node_id: str = "n") -> str | None:
    sess = sessions.get_active(node_id)
    return (sess.mode_state or {}).get("step") if sess else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HappyPath(unittest.TestCase):
    def test_full_flow_designates_primary_user(self):
        controller, sessions, face, tts, profile = _make_controller()

        # 1. Face detected → greeting
        resp = controller.maybe_initiate("n")
        self.assertIsNotNone(resp)
        self.assertEqual(resp["onboarding"]["event"], "greeting")
        self.assertEqual(_step(sessions), STEP_GREET)
        self.assertEqual(len(tts.spoken), 1)
        self.assertEqual(tts.spoken[0][0], "n")
        self.assertIn("primary user", tts.spoken[0][1].lower())

        # 2. User says "yes, ok" → asks for name
        resp = controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_AWAIT_NAME)
        self.assertIn("call you", resp["message"].lower())

        # 3. "I'm Alex" → confirm
        resp = controller.maybe_handle_turn("I'm Alex", "n")
        self.assertEqual(_step(sessions), STEP_CONFIRM_NAME)
        self.assertIn("Alex", resp["message"])

        # 4. "yes" → first pose
        resp = controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_CAPTURE_FRONTAL)
        self.assertIn("straight", resp["message"].lower())

        # 5. Five poses, one turn each.
        for step, pose in POSE_SEQUENCE:
            self.assertEqual(_step(sessions), step)
            # Any input — pose capture is driven by the camera fake, not by content.
            controller.maybe_handle_turn("", "n")
        self.assertEqual(_step(sessions), STEP_AWAIT_DESIGNATE)
        self.assertEqual([p for _, p in POSE_SEQUENCE],
                         [p for _, _, p in face.captures])

        # 6. "yes" to designate → prefs_name
        controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_PREFS_NAME)

        # 7. "yes" → keep same name; advance to tone
        controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_PREFS_TONE)

        # 8. "brief" → done
        resp = controller.maybe_handle_turn("brief please", "n")
        self.assertEqual(resp["onboarding"]["event"], "completed")
        self.assertTrue(resp["onboarding"]["designated_primary"])

        # Session closed with success outcome.
        sess = sessions.sessions["n"]
        self.assertTrue(sess.closed)
        self.assertEqual(sess.outcome["action"], "onboarding_completed")

        # Profile state.
        self.assertEqual(profile.primary, "alex")
        self.assertEqual(profile.display_names["alex"], "Alex")
        self.assertEqual(profile.preferences[("alex", "tone")], "brief")


class NameHandling(unittest.TestCase):
    def test_unparseable_name_re_asks(self):
        controller, sessions, _, tts, _ = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        resp = controller.maybe_handle_turn("umm hmm", "n")
        self.assertEqual(_step(sessions), STEP_AWAIT_NAME)
        self.assertIn("didn't catch", resp["message"].lower())

    def test_bare_name_accepted(self):
        controller, sessions, _, _, _ = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("Alex", "n")
        self.assertEqual(_step(sessions), STEP_CONFIRM_NAME)

    def test_name_correction_at_confirm_loops_back(self):
        controller, sessions, _, _, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("no", "n")  # not Alex
        self.assertEqual(_step(sessions), STEP_AWAIT_NAME)
        controller.maybe_handle_turn("I'm Bob", "n")
        self.assertEqual(_step(sessions), STEP_CONFIRM_NAME)
        controller.maybe_handle_turn("yes", "n")
        # Display name should be Bob's, not Alex's.
        self.assertEqual(_step(sessions), STEP_CAPTURE_FRONTAL)
        self.assertIn("bob", [k.lower() for k in profile.display_names.keys()])
        self.assertNotIn("alex", profile.display_names)

    def test_ambiguous_confirm_re_asks(self):
        controller, sessions, _, _, _ = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("maybe", "n")
        self.assertEqual(_step(sessions), STEP_CONFIRM_NAME)


class DeclineFlows(unittest.TestCase):
    def test_decline_at_greet_parks_silently(self):
        controller, sessions, _, tts, profile = _make_controller()
        controller.maybe_initiate("n")
        tts.spoken.clear()
        resp = controller.maybe_handle_turn("no", "n")
        self.assertEqual(resp["onboarding"]["event"], "parked")
        self.assertEqual(resp["onboarding"]["reason"], "declined")
        self.assertEqual(resp["message"], "")  # silent
        self.assertEqual(tts.spoken, [])  # nothing spoken
        self.assertIsNone(profile.primary)

    def test_decline_designation_finalizes_enrollment_without_primary(self):
        controller, sessions, face, _, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        for _ in POSE_SEQUENCE:
            controller.maybe_handle_turn("", "n")
        # At designate step.
        self.assertEqual(_step(sessions), STEP_AWAIT_DESIGNATE)
        resp = controller.maybe_handle_turn("no", "n")
        self.assertEqual(resp["onboarding"]["event"], "parked")
        self.assertEqual(resp["onboarding"]["reason"], "declined_designation")
        # Enrollment captured the faces but primary stays unset.
        self.assertEqual(len(face.captures), len(POSE_SEQUENCE))
        self.assertIsNone(profile.primary)
        # Display name was still set when name was confirmed.
        self.assertEqual(profile.display_names["alex"], "Alex")

    def test_ambiguous_designate_re_asks(self):
        controller, sessions, _, _, _ = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        for _ in POSE_SEQUENCE:
            controller.maybe_handle_turn("", "n")
        controller.maybe_handle_turn("hmm", "n")
        self.assertEqual(_step(sessions), STEP_AWAIT_DESIGNATE)


class PoseCaptureFailures(unittest.TestCase):
    def _advance_to_pose_capture(self):
        controller, sessions, face, tts, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_CAPTURE_FRONTAL)
        return controller, sessions, face, tts, profile

    def test_lost_face_then_recovers(self):
        controller, sessions, face, _, _ = self._advance_to_pose_capture()
        # Face disappears mid-capture.
        face.snapshot = None
        resp = controller.maybe_handle_turn("", "n")
        self.assertEqual(_step(sessions), STEP_CAPTURE_FRONTAL)
        self.assertIn("can't see you", resp["message"].lower())
        # Face returns.
        face.snapshot = FaceSnapshot(identity="unknown_0001", confidence=0.6,
                                     pose="frontal", face_group_id="ug_1")
        controller.maybe_handle_turn("", "n")
        # Now advanced past frontal.
        self.assertNotEqual(_step(sessions), STEP_CAPTURE_FRONTAL)

    def test_lost_face_persists_parks_after_timeout(self):
        controller, sessions, face, _, _ = self._advance_to_pose_capture()
        face.snapshot = None
        for _ in range(_POSE_TICK_PARK):
            controller.maybe_handle_turn("", "n")
        sess = sessions.sessions["n"]
        self.assertTrue(sess.closed)
        self.assertEqual(sess.outcome["result"]["reason"], "lost_face")

    def test_unconfirmable_pose_parks_after_timeout(self):
        controller, sessions, face, _, _ = self._advance_to_pose_capture()
        face.poses_confirmable = set()  # never confirms
        for _ in range(_POSE_TICK_PARK):
            controller.maybe_handle_turn("", "n")
        sess = sessions.sessions["n"]
        self.assertTrue(sess.closed)
        self.assertEqual(sess.outcome["result"]["reason"], "pose_timeout")

    def test_capture_failure_parks_after_timeout(self):
        controller, sessions, face, _, _ = self._advance_to_pose_capture()
        face.capture_succeeds = False
        for _ in range(_POSE_TICK_PARK):
            controller.maybe_handle_turn("", "n")
        sess = sessions.sessions["n"]
        self.assertTrue(sess.closed)
        self.assertEqual(sess.outcome["result"]["reason"], "capture_failed")


class InitiateGuards(unittest.TestCase):
    def test_idempotent_when_session_already_active(self):
        controller, sessions, _, _, _ = _make_controller()
        first = controller.maybe_initiate("n")
        self.assertIsNotNone(first)
        second = controller.maybe_initiate("n")
        self.assertIsNone(second)  # already in onboarding mode

    def test_skipped_when_primary_already_set(self):
        controller, sessions, _, _, _ = _make_controller(primary="someone_else")
        resp = controller.maybe_initiate("n")
        self.assertIsNone(resp)
        self.assertNotIn("n", sessions.sessions)

    def test_skipped_when_no_face_visible(self):
        face = FakeFaceObserver(snapshot=None)
        controller, sessions, _, _, _ = _make_controller(face=face)
        resp = controller.maybe_initiate("n")
        self.assertIsNone(resp)


class HelperFunctions(unittest.TestCase):
    def test_slug_identity_strips_and_lowercases(self):
        self.assertEqual(_slug_identity("Alex"), "alex")
        self.assertEqual(_slug_identity("Mary-Jane"), "mary_jane")
        self.assertEqual(_slug_identity("  O'Brien  "), "obrien")
        self.assertEqual(_slug_identity(""), "unknown")
        self.assertEqual(_slug_identity("!!!"), "unknown")

    def test_classify_tone_defaults_to_brief(self):
        self.assertEqual(_classify_tone("be brief"), "brief")
        self.assertEqual(_classify_tone("keep it short and tight"), "brief")
        self.assertEqual(_classify_tone("more conversational please"), "conversational")
        self.assertEqual(_classify_tone("chatty and warm"), "conversational")
        self.assertEqual(_classify_tone("whatever"), "brief")
        self.assertEqual(_classify_tone(""), "brief")


class TonePreferenceClassification(unittest.TestCase):
    def test_conversational_preference_persists(self):
        controller, sessions, _, _, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        for _ in POSE_SEQUENCE:
            controller.maybe_handle_turn("", "n")
        controller.maybe_handle_turn("yes", "n")  # designate
        controller.maybe_handle_turn("yes", "n")  # prefs_name keep
        controller.maybe_handle_turn("more conversational please", "n")
        self.assertEqual(profile.preferences[("alex", "tone")], "conversational")


class TickBehavior(unittest.TestCase):
    """The bg-ticker entry point drives pose-capture forward while the
    user is silent. These tests verify it (a) advances state when the
    pose lands, (b) doesn't speak every tick (only at nudge cadence),
    (c) returns None on silent ticks so the ticker can log meaningful
    events only, and (d) is a no-op outside pose-capture steps."""

    def _at_pose_step(self):
        controller, sessions, face, tts, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        # Now at STEP_CAPTURE_FRONTAL, prompts spoken: greeting + name + frontal
        return controller, sessions, face, tts, profile

    def test_tick_advances_when_pose_lands(self):
        controller, sessions, face, _, _ = self._at_pose_step()
        # Single tick with the pose already confirmable captures + advances.
        resp = controller.tick("n")
        self.assertIsNotNone(resp)
        self.assertEqual(resp["onboarding"]["event"], "advanced")
        self.assertEqual(_step(sessions), POSE_SEQUENCE[1][0])
        self.assertEqual(len(face.captures), 1)

    def test_tick_silent_when_pose_not_yet_landed(self):
        controller, sessions, face, tts, _ = self._at_pose_step()
        tts.spoken.clear()
        face.poses_confirmable = set()  # nothing ever confirms
        # First tick: silent — counter goes from 0 to 1, no TTS.
        resp = controller.tick("n")
        self.assertIsNone(resp)
        self.assertEqual(tts.spoken, [])
        self.assertEqual(sessions.sessions["n"].mode_state["pose_ticks"], 1)

    def test_tick_nudges_at_interval(self):
        controller, sessions, face, tts, _ = self._at_pose_step()
        tts.spoken.clear()
        face.poses_confirmable = set()
        # Tick up to (but not including) the nudge interval — no TTS yet.
        for _ in range(_POSE_TICK_NUDGE - 1):
            controller.tick("n")
        self.assertEqual(tts.spoken, [])
        # One more tick crosses the nudge boundary — system re-prompts.
        controller.tick("n")
        self.assertEqual(len(tts.spoken), 1)
        # At STEP_CAPTURE_FRONTAL the nudge re-speaks the frontal prompt.
        self.assertIn("straight", tts.spoken[0][1].lower())

    def test_tick_no_op_outside_pose_step(self):
        controller, sessions, _, tts, _ = _make_controller()
        controller.maybe_initiate("n")
        tts.spoken.clear()
        # At STEP_GREET — not a pose step. Tick should do nothing.
        resp = controller.tick("n")
        self.assertIsNone(resp)
        self.assertEqual(tts.spoken, [])
        self.assertEqual(_step(sessions), STEP_GREET)

    def test_tick_no_op_when_no_session(self):
        controller, sessions, _, tts, _ = _make_controller()
        resp = controller.tick("nobody")
        self.assertIsNone(resp)
        self.assertEqual(tts.spoken, [])

    def test_tick_parks_at_timeout(self):
        controller, sessions, face, _, _ = self._at_pose_step()
        face.poses_confirmable = set()
        for _ in range(_POSE_TICK_PARK):
            controller.tick("n")
        sess = sessions.sessions["n"]
        self.assertTrue(sess.closed)
        self.assertEqual(sess.outcome["result"]["reason"], "pose_timeout")


class UserDenyDuringPose(unittest.TestCase):
    """A user 'no'/'cancel' during pose-capture should park immediately —
    the user is opting out and shouldn't have to wait for the timeout."""

    def test_deny_at_pose_step_parks(self):
        controller, sessions, _, _, _ = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alex", "n")
        controller.maybe_handle_turn("yes", "n")
        self.assertEqual(_step(sessions), STEP_CAPTURE_FRONTAL)
        resp = controller.maybe_handle_turn("cancel", "n")
        self.assertEqual(resp["onboarding"]["event"], "parked")
        self.assertEqual(resp["onboarding"]["reason"], "declined")
        self.assertTrue(sessions.sessions["n"].closed)


class CustomPreferredName(unittest.TestCase):
    def test_user_picks_a_different_preferred_name(self):
        controller, sessions, _, _, profile = _make_controller()
        controller.maybe_initiate("n")
        controller.maybe_handle_turn("yes", "n")
        controller.maybe_handle_turn("I'm Alexander", "n")
        controller.maybe_handle_turn("yes", "n")
        for _ in POSE_SEQUENCE:
            controller.maybe_handle_turn("", "n")
        controller.maybe_handle_turn("yes", "n")  # designate
        controller.maybe_handle_turn("Alex", "n")  # nickname
        controller.maybe_handle_turn("brief", "n")
        # The original display name from confirm-name was "Alexander";
        # finalize overwrote it with the preferred-name choice.
        self.assertEqual(profile.display_names["alexander"], "Alex")
        self.assertEqual(profile.primary, "alexander")


if __name__ == "__main__":
    unittest.main()
