#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from mood_engine import MoodEngine


class MoodEngineTest(unittest.TestCase):
    def test_style_updates_resolve_contradictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            base = engine.style_state()["resolved"]

            engine.apply_style_update({"preference": "be more sarcastic"})
            after_sarcasm = engine.style_state()["resolved"]
            self.assertGreater(after_sarcasm["sarcasm"], base["sarcasm"])

            engine.apply_style_update({"preference": "be less rude"})
            after_less_rude = engine.style_state()["resolved"]
            self.assertLess(after_less_rude["rudeness"], after_sarcasm["rudeness"])
            self.assertGreaterEqual(after_less_rude["sarcasm"], after_sarcasm["sarcasm"])

    def test_unverified_user_caps_rudeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            for _ in range(5):
                engine.apply_style_update({"preference": "be more rude and harsher"})

            unverified = engine.voice_state({"may_address_primary_user": False})["voice"]
            verified = engine.voice_state({"may_address_primary_user": True})["voice"]

            self.assertLessEqual(unverified["rudeness"], 0.15)
            self.assertLessEqual(verified["rudeness"], 0.35)
            self.assertGreaterEqual(verified["rudeness"], unverified["rudeness"])

    def test_imports_legacy_overrides_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            settings = {
                "behavior": {
                    "overrides": [
                        {"preference": "be more sarcastic"},
                        {"preference": "be less rude"},
                    ]
                }
            }
            imported = engine.import_legacy_response_settings(settings)
            imported_again = engine.import_legacy_response_settings(settings)

            self.assertEqual(len(imported["history"]), 2)
            self.assertEqual(len(imported_again["history"]), 2)
            self.assertGreater(imported["resolved"]["sarcasm"], 0.55)
            self.assertLess(imported["resolved"]["rudeness"], 0.20)

    def test_tone_it_down_changes_resolved_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            before = engine.style_state()["resolved"]
            after = engine.apply_style_update({"preference": "tone it down"})["resolved"]

            self.assertLess(after["sarcasm"], before["sarcasm"])
            self.assertLess(after["rudeness"], before["rudeness"])
            self.assertLess(after["directness"], before["directness"])

    def test_mood_records_interaction_without_making_cruelty_a_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            before = engine.mood()["state"]
            engine.record_interaction(
                {"ok": True, "route": {"route": "direction"}, "actions": [{"ok": False}]},
                identity_verified=False,
            )
            after = engine.mood()["state"]

            self.assertGreater(after["irritation"], before["irritation"])
            self.assertLess(after["patience"], before["patience"])
            voice = engine.voice_state({"may_address_primary_user": False})["voice"]
            self.assertLessEqual(voice["rudeness"], 0.15)

    def test_mood_changes_voice_contract_without_exceeding_audience_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = MoodEngine(Path(tmp))
            calm_voice = engine.voice_state({"may_address_primary_user": False})["voice"]

            mood = engine.mood()
            mood["state"]["irritation"] = 1.0
            mood["state"]["playfulness"] = 0.0
            mood["last_updated"] = time.time()
            engine.write_mood(mood)

            tense = engine.voice_state({"may_address_primary_user": False})
            tense_voice = tense["voice"]
            contract = "\n".join(engine.voice_contract_lines({"may_address_primary_user": False}))

            self.assertGreater(tense_voice["brevity"], calm_voice["brevity"])
            self.assertLess(tense_voice["sarcasm"], calm_voice["sarcasm"])
            self.assertLessEqual(tense_voice["rudeness"], 0.15)
            self.assertIn("Friction may make answers shorter, never crueler.", contract)


if __name__ == "__main__":
    unittest.main(verbosity=2)
