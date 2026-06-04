#!/usr/bin/env python3
"""Tests for vault/identity_actions.py — privileged transfer / unset
verbal intents and the face-verification gate."""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from identity_actions import (  # noqa: E402
    IdentityActionsController,
    classify_intent,
)
from onboarding import FaceSnapshot  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes — narrower than the onboarding ones since identity_actions only
# touches a few methods on each adapter.
# ---------------------------------------------------------------------------


@dataclass
class FakeFace:
    snapshot: FaceSnapshot | None = None

    def current_face(self, node_id: str) -> FaceSnapshot | None:
        return self.snapshot

    def confirm_pose(self, *a, **kw):
        raise AssertionError("identity_actions should not touch confirm_pose")

    def capture_reference(self, *a, **kw):
        raise AssertionError("identity_actions should not touch capture_reference")


@dataclass
class FakeProfile:
    primary: str | None = None
    enrolled: set[str] = field(default_factory=set)
    designate_calls: list[str] = field(default_factory=list)
    unset_calls: int = 0

    def set_display_name(self, identity, name): pass
    def set_preference(self, identity, key, value): pass

    def designate_primary_user(self, identity):
        self.primary = identity
        self.designate_calls.append(identity)

    def unset_primary_user(self):
        self.primary = None
        self.unset_calls += 1

    def primary_user(self):
        return self.primary

    def has_profile(self, identity):
        return identity in self.enrolled


@dataclass
class FakeTTS:
    spoken: list[tuple[str, str]] = field(default_factory=list)
    def say(self, node_id, text):
        self.spoken.append((node_id, text))


def _make(*, primary="alex", face_conf=0.9, face_identity="alex", enrolled=None):
    face = FakeFace(snapshot=FaceSnapshot(
        identity=face_identity, confidence=face_conf, pose="frontal",
        face_group_id="ug_1",
    ))
    enrolled = set(enrolled) if enrolled is not None else {"alex"}
    profile = FakeProfile(primary=primary, enrolled=enrolled)
    tts = FakeTTS()
    on_unset_calls = []
    controller = IdentityActionsController(
        face_observer=face,
        profile_store=profile,
        tts=tts,
        min_confidence=0.7,
        on_unset=lambda: on_unset_calls.append(True),
    )
    return controller, face, profile, tts, on_unset_calls


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------


class IntentClassifier(unittest.TestCase):
    def test_recognizes_unset_phrasings(self):
        for phrase in [
            "unset primary user",
            "Unset the primary user",
            "unset primary",
            "remove me as primary",
            "remove me as the primary user",
            "clear the primary user",
            "i'm no longer the primary",
            "I am no longer the primary user",
            "i'm not the primary user anymore",
            "step down as primary",
        ]:
            intent = classify_intent(phrase)
            self.assertEqual(intent, {"type": "unset"}, phrase)

    def test_recognizes_transfer_phrasings(self):
        cases = [
            ("transfer primary user to Bob", "Bob"),
            ("Transfer the primary user to bob", "bob"),
            ("transfer primary to Alex", "Alex"),
            ("make Sam the primary user", "Sam"),
            ("make sam primary", "sam"),
            ("set Mary as the primary user", "Mary"),
            ("set jane as primary", "jane"),
            ("Bob is the primary user now", "Bob"),
        ]
        for phrase, expected in cases:
            intent = classify_intent(phrase)
            self.assertIsNotNone(intent, phrase)
            self.assertEqual(intent["type"], "transfer", phrase)
            self.assertEqual(intent["target"], expected, phrase)

    def test_rejects_non_matching_input(self):
        for phrase in [
            "",
            "hello",
            "tell me about the weather",
            "primary user is great",  # mentions but no action
            "transfer the files to Bob",  # transfer but no primary
        ]:
            self.assertIsNone(classify_intent(phrase), phrase)

    def test_unset_wins_over_transfer(self):
        # "remove me as primary" could in principle parse as a
        # transfer to "me"; unset matching first guards against that.
        self.assertEqual(
            classify_intent("remove me as primary"),
            {"type": "unset"},
        )


# ---------------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------------


class VerificationGate(unittest.TestCase):
    def test_no_primary_set_refuses(self):
        controller, _, profile, tts, _ = _make(primary=None)
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["event"], "refused")
        self.assertEqual(resp["identity_action"]["reason"], "no_primary_set")
        self.assertEqual(profile.unset_calls, 0)
        self.assertEqual(len(tts.spoken), 1)

    def test_no_face_visible_refuses(self):
        controller, face, profile, tts, _ = _make()
        face.snapshot = None
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["reason"], "no_face_visible")
        self.assertEqual(profile.unset_calls, 0)
        self.assertIn("look at the camera", tts.spoken[0][1].lower())

    def test_wrong_identity_refuses(self):
        controller, face, profile, tts, _ = _make()
        face.snapshot = FaceSnapshot(
            identity="someone_else", confidence=0.9,
            pose="frontal", face_group_id="ug_2",
        )
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["reason"], "not_primary")
        self.assertEqual(profile.unset_calls, 0)

    def test_low_confidence_refuses(self):
        controller, face, profile, tts, _ = _make(face_conf=0.5)
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["reason"], "low_confidence")
        self.assertEqual(profile.unset_calls, 0)

    def test_min_confidence_threshold_is_inclusive_at_default(self):
        # Confidence exactly at threshold (0.7) should pass.
        controller, _, profile, _, _ = _make(face_conf=0.7)
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["event"], "unset")
        self.assertEqual(profile.unset_calls, 1)


# ---------------------------------------------------------------------------
# Unset
# ---------------------------------------------------------------------------


class UnsetAction(unittest.TestCase):
    def test_unset_clears_primary_silently(self):
        controller, _, profile, tts, on_unset = _make()
        resp = controller.try_handle("unset primary user", "n")
        self.assertEqual(resp["identity_action"]["event"], "unset")
        self.assertEqual(resp["identity_action"]["previous_primary"], "alex")
        self.assertEqual(profile.unset_calls, 1)
        self.assertIsNone(profile.primary)
        self.assertEqual(resp["message"], "")  # silent
        self.assertEqual(tts.spoken, [])  # nothing spoken
        # Runtime callback fired so awaiting_primary_user can re-activate.
        self.assertEqual(on_unset, [True])


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


class TransferAction(unittest.TestCase):
    def test_transfer_to_enrolled_target_flips_pointer(self):
        controller, _, profile, tts, on_unset = _make(
            enrolled={"alex", "bob"},
        )
        resp = controller.try_handle("transfer primary user to Bob", "n")
        self.assertEqual(resp["identity_action"]["event"], "transferred")
        self.assertEqual(resp["identity_action"]["previous_primary"], "alex")
        self.assertEqual(resp["identity_action"]["target"], "bob")
        self.assertEqual(profile.primary, "bob")
        self.assertEqual(profile.designate_calls, ["bob"])
        # Transfer DOES speak — confirmation to old primary.
        self.assertIn("primary user", tts.spoken[0][1].lower())
        # on_unset NOT fired — primary is still set (to new target).
        self.assertEqual(on_unset, [])

    def test_transfer_to_unenrolled_refuses_with_hint(self):
        controller, _, profile, tts, _ = _make(enrolled={"alex"})
        resp = controller.try_handle("make Charlie the primary user", "n")
        self.assertEqual(resp["identity_action"]["event"], "transfer_unenrolled")
        self.assertEqual(resp["identity_action"]["target"], "charlie")
        # Primary unchanged.
        self.assertEqual(profile.primary, "alex")
        self.assertEqual(profile.designate_calls, [])
        self.assertIn("unset primary user", tts.spoken[0][1].lower())

    def test_transfer_to_self_is_noop(self):
        controller, _, profile, tts, _ = _make(enrolled={"alex"})
        resp = controller.try_handle("make Alex the primary user", "n")
        self.assertEqual(resp["identity_action"]["event"], "transfer_noop")
        self.assertEqual(profile.primary, "alex")
        self.assertEqual(profile.designate_calls, [])
        self.assertIn("already", tts.spoken[0][1].lower())


# ---------------------------------------------------------------------------
# Non-matching input passes through
# ---------------------------------------------------------------------------


class NonMatchingInput(unittest.TestCase):
    def test_returns_none_when_no_intent_matched(self):
        controller, _, profile, tts, _ = _make()
        self.assertIsNone(controller.try_handle("hi how are you", "n"))
        self.assertIsNone(controller.try_handle("what's the weather", "n"))
        self.assertEqual(profile.unset_calls, 0)
        self.assertEqual(profile.designate_calls, [])
        self.assertEqual(tts.spoken, [])


if __name__ == "__main__":
    unittest.main()
