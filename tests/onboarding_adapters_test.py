#!/usr/bin/env python3
"""Tests for vault/onboarding_adapters.py.

Currently focused on the ``VaultFaceObserver`` pose-hold accumulator —
the stateful piece that filters mid-rotation transient poses by
requiring N consecutive matching reads before ``confirm_pose`` returns
True. The TTS and ProfileStore adapters are thin pass-throughs over
existing vault bridge / node_registry surfaces and are exercised end-to-end at
integration time.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from onboarding_adapters import VaultFaceObserver  # noqa: E402


class _FakeBridge:
    """Minimal vault-to-camera-node HTTP bridge stand-in (the real one
    is ``ScoutVaultBridge`` — naming is historical, the bridge isn't
    scout-specific). ``meta`` is a dict mutated by tests to simulate
    successive camera reads."""

    def __init__(self):
        self.meta: dict = {"detections": []}
        self.learn_face_calls: list[str] = []
        self.learn_face_result: dict = {"ok": True}

    def camera_state_for_node(self, node_id):
        return dict(self.meta)

    def learn_face(self, identity):
        self.learn_face_calls.append(identity)
        return dict(self.learn_face_result)


def _face(*, identity="unknown", pose="frontal", face_group_id="ug_1",
          identity_confidence=0.0, bbox=(10, 10, 80, 80)):
    return {
        "label": "face",
        "identity": identity,
        "reference_pose": pose,
        "face_group_id": face_group_id,
        "vault_face_group_id": face_group_id,
        "identity_confidence": identity_confidence,
        "bbox": list(bbox),
        "id": face_group_id,
    }


class PoseHoldAccumulator(unittest.TestCase):
    def test_holds_for_required_consecutive_reads(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=3)
        bridge.meta = {"detections": [_face(pose="left")]}
        # Two reads — not yet confirmed.
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        # Third read — confirmed.
        self.assertTrue(obs.confirm_pose("n", "left", "unknown_0001"))
        # And subsequent reads start fresh (counter reset on confirm),
        # so re-confirming this pose requires another full hold.
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertTrue(obs.confirm_pose("n", "left", "unknown_0001"))

    def test_non_match_resets_counter(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=3)
        bridge.meta = {"detections": [_face(pose="left")]}
        obs.confirm_pose("n", "left", "unknown_0001")
        obs.confirm_pose("n", "left", "unknown_0001")
        # User briefly rotates past — pose changes.
        bridge.meta = {"detections": [_face(pose="frontal")]}
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        # Back to left — starts from zero again, doesn't carry the
        # previous two.
        bridge.meta = {"detections": [_face(pose="left")]}
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertTrue(obs.confirm_pose("n", "left", "unknown_0001"))

    def test_face_disappearing_resets_counter(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=3)
        bridge.meta = {"detections": [_face(pose="left")]}
        obs.confirm_pose("n", "left", "unknown_0001")
        obs.confirm_pose("n", "left", "unknown_0001")
        # Face vanishes.
        bridge.meta = {"detections": []}
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        # Comes back — counter restarted.
        bridge.meta = {"detections": [_face(pose="left")]}
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertFalse(obs.confirm_pose("n", "left", "unknown_0001"))
        self.assertTrue(obs.confirm_pose("n", "left", "unknown_0001"))

    def test_different_node_has_independent_counter(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=3)
        bridge.meta = {"detections": [_face(pose="left")]}
        # Two reads on node 'a' — not confirmed.
        obs.confirm_pose("a", "left", "unknown_0001")
        obs.confirm_pose("a", "left", "unknown_0001")
        # First read on node 'b' should NOT count node 'a''s history.
        self.assertFalse(obs.confirm_pose("b", "left", "unknown_0001"))
        # 'a' is still one read away.
        self.assertTrue(obs.confirm_pose("a", "left", "unknown_0001"))

    def test_hold_required_one_confirms_instantly(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=1)
        bridge.meta = {"detections": [_face(pose="frontal")]}
        self.assertTrue(obs.confirm_pose("n", "frontal", "unknown_0001"))

    def test_wrong_identity_match_rejects(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=2)
        # Different known identity in frame — not the onboarding subject.
        bridge.meta = {"detections": [_face(
            pose="left",
            identity="someone_else",  # already-enrolled, not the subject
            face_group_id="ug_2",
            identity_confidence=0.9,
        )]}
        # Should never confirm because identity_hint doesn't match and
        # the face isn't unknown.
        self.assertFalse(obs.confirm_pose("n", "left", "alex"))
        self.assertFalse(obs.confirm_pose("n", "left", "alex"))
        self.assertFalse(obs.confirm_pose("n", "left", "alex"))


class CaptureReferenceResetsHold(unittest.TestCase):
    def test_capture_clears_hold_counter(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge, hold_required=3)
        bridge.meta = {"detections": [_face(pose="left")]}
        # Drive to a confirmed state (3 reads), then capture.
        obs.confirm_pose("n", "left", "unknown_0001")
        obs.confirm_pose("n", "left", "unknown_0001")
        self.assertTrue(obs.confirm_pose("n", "left", "unknown_0001"))
        ok = obs.capture_reference("n", "alex", "left")
        self.assertTrue(ok)
        self.assertEqual(bridge.learn_face_calls, ["alex"])
        # After capture, counter is back to zero — onboarding moves on
        # to next pose; if it ever came back here it'd require fresh hold.
        self.assertNotIn(("n", "left"), obs._hold_counts)


class CurrentFacePicksBest(unittest.TestCase):
    def test_prefers_higher_identity_confidence(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge)
        bridge.meta = {"detections": [
            _face(pose="left", identity_confidence=0.4, face_group_id="a"),
            _face(pose="right", identity_confidence=0.8, face_group_id="b"),
        ]}
        snap = obs.current_face("n")
        self.assertIsNotNone(snap)
        self.assertEqual(snap.face_group_id, "b")
        self.assertEqual(snap.pose, "right")

    def test_falls_back_to_largest_bbox_when_no_confidence(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge)
        bridge.meta = {"detections": [
            _face(pose="left", bbox=(0, 0, 30, 30), face_group_id="small"),
            _face(pose="right", bbox=(0, 0, 100, 100), face_group_id="big"),
        ]}
        for det in bridge.meta["detections"]:
            det.pop("identity_confidence", None)
        snap = obs.current_face("n")
        self.assertEqual(snap.face_group_id, "big")

    def test_returns_none_when_no_face_detections(self):
        bridge = _FakeBridge()
        obs = VaultFaceObserver(bridge)
        bridge.meta = {"detections": [
            {"label": "person", "identity": None, "bbox": [0, 0, 50, 50]},
        ]}
        self.assertIsNone(obs.current_face("n"))


if __name__ == "__main__":
    unittest.main()
