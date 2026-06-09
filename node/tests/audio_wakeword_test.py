#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_node import service as audio_service


class AudioWakewordTest(unittest.TestCase):
    def test_spaced_luhkas_misrecognition_opens_listen_window(self) -> None:
        self.assertTrue(audio_service._contains_audio_wakeword("the look as status report"))

    def test_non_wake_phrase_does_not_match_substring(self) -> None:
        self.assertFalse(audio_service._contains_audio_wakeword("please look at status report"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
