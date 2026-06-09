#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_node import service as audio_service


class FakeTTS:
    name = "fake"
    available = True

    def __init__(self) -> None:
        self.interrupted = 0
        self.spoken: list[str] = []

    def interrupt(self) -> None:
        self.interrupted += 1

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class AudioOutputMuteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_env = dict(os.environ)
        self.old_muted = audio_service._output_muted
        self.old_generation = audio_service._tts_generation
        self.old_run = audio_service.subprocess.run
        self.old_update_state = audio_service.update_state
        self.old_restore = dict(audio_service._hardware_volume_restore)
        os.environ["AUDIO_HARDWARE_MUTE_ENABLE"] = "1"
        os.environ["AUDIO_OUTPUT_DEVICE"] = "plughw:CARD=wm8960soundcard,DEV=0"
        os.environ.pop("AUDIO_OUTPUT_MUTE_COMMAND", None)
        os.environ.pop("AUDIO_WM8960_CARD", None)
        self.runs: list[list[str]] = []
        audio_service.subprocess.run = self._fake_run
        audio_service.update_state = lambda payload: None
        audio_service._output_muted = False
        audio_service._tts_generation = 0
        audio_service._hardware_volume_restore.clear()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)
        audio_service.subprocess.run = self.old_run
        audio_service.update_state = self.old_update_state
        audio_service._output_muted = self.old_muted
        audio_service._tts_generation = self.old_generation
        audio_service._hardware_volume_restore.clear()
        audio_service._hardware_volume_restore.update(self.old_restore)

    def _fake_run(self, args, **kwargs):
        self.runs.append(list(args))
        stdout = ""
        if "sget" in args:
            stdout = "Front Left: Playback 73 [57%] [-8.00dB]\nFront Right: Playback 73 [57%] [-8.00dB]\n"
        return types.SimpleNamespace(returncode=0, stdout=stdout)

    def test_output_mute_interrupts_tts_and_mutes_alsa_controls(self) -> None:
        tts = FakeTTS()

        muted = audio_service._set_output_muted(True, tts)

        self.assertTrue(muted)
        self.assertEqual(tts.interrupted, 1)
        self.assertTrue(audio_service._is_output_muted())
        self.assertIn(["amixer", "-q", "-c", "wm8960soundcard", "sset", "Speaker", "mute"], self.runs)
        self.assertIn(["amixer", "-q", "-c", "wm8960soundcard", "sset", "Playback", "mute"], self.runs)
        self.assertIn(["amixer", "-q", "-c", "wm8960soundcard", "sset", "Speaker", "0%"], self.runs)

    def test_output_unmute_asks_alsa_to_unmute_controls(self) -> None:
        tts = FakeTTS()
        audio_service._hardware_volume_restore[("wm8960soundcard", "Speaker")] = "73"

        muted = audio_service._set_output_muted(False, tts)

        self.assertFalse(muted)
        self.assertEqual(tts.interrupted, 0)
        self.assertIn(["amixer", "-q", "-c", "wm8960soundcard", "sset", "Speaker", "unmute"], self.runs)
        self.assertIn(["amixer", "-q", "-c", "wm8960soundcard", "sset", "Speaker", "73"], self.runs)

    def test_custom_hardware_mute_command_can_replace_default_amixer_controls(self) -> None:
        os.environ["AUDIO_OUTPUT_MUTE_COMMAND"] = "echo {state} {card} {muted}"
        tts = FakeTTS()

        audio_service._set_output_muted(True, tts)

        self.assertEqual(self.runs, [["echo", "mute", "wm8960soundcard", "1"]])

    def test_start_tts_is_noop_while_output_muted(self) -> None:
        tts = FakeTTS()
        audio_service._output_muted = True

        audio_service._start_tts(tts, "hello")

        self.assertEqual(tts.spoken, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
