#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_node import service as audio_service


class FakeStream:
    def __init__(self, events: list[dict]) -> None:
        self.lines = [(json.dumps(event) + "\n").encode("utf-8") for event in events]
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self) -> None:
        self.closed = True


class FakeTTS:
    available = True

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class MutedStreamCaptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_urlopen = audio_service.urlopen
        self.old_notify = audio_service._notify_ui_event
        self.old_update_state = audio_service.update_state
        self.old_muted = audio_service._output_muted
        self.events: list[dict] = []
        self.state_updates: list[dict] = []
        audio_service._output_muted = True
        audio_service._notify_ui_event = lambda _url, payload: self.events.append(dict(payload))
        audio_service.update_state = lambda payload: self.state_updates.append(payload)

    def tearDown(self) -> None:
        audio_service.urlopen = self.old_urlopen
        audio_service._notify_ui_event = self.old_notify
        audio_service.update_state = self.old_update_state
        audio_service._output_muted = self.old_muted

    def test_muted_stream_sends_accumulated_text_to_display(self) -> None:
        final = (
            "one two three four five six seven eight "
            "nine ten eleven twelve thirteen fourteen fifteen sixteen"
        )
        stream = FakeStream([
            {"type": "delta", "text": "one two three four five six seven eight "},
            {"type": "delta", "text": "nine ten eleven twelve thirteen fourteen fifteen sixteen "},
            {"type": "done", "text": final},
        ])
        audio_service.urlopen = lambda *_args, **_kwargs: stream

        result = audio_service._stream_presence_to_tts(
            "http://vault/presence/message/stream",
            {"message": "test"},
            FakeTTS(),
            "http://display/ui/event",
        )

        assistant_texts = [
            event["text"]
            for event in self.events
            if event.get("type") == "assistant_message"
        ]
        self.assertEqual(result, {"tts": final, "message": final})
        self.assertEqual(assistant_texts, [
            "one two three four five six seven eight",
            final,
        ])
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
