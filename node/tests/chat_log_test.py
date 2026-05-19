#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera_node.chat_log import ChatLog


class ChatLogTest(unittest.TestCase):
    def test_init_file_loads_existing_history_before_session_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_session.jsonl"
            existing = [
                {"id": 4, "timestamp": 1.0, "role": "user", "source": "chat_input", "text": "Good evening", "meta": {}},
                {"id": 5, "timestamp": 2.0, "role": "assistant", "source": "vault_chat", "text": "Evening. I'm here, but not for you.", "meta": {}},
            ]
            path.write_text("\n".join(json.dumps(entry) for entry in existing) + "\n", encoding="utf-8")

            log = ChatLog(path, max_entries=10)
            log.init_file()

            snapshot = log.snapshot()
            self.assertEqual(snapshot[0]["text"], "Good evening")
            self.assertEqual(snapshot[1]["text"], "Evening. I'm here, but not for you.")
            self.assertEqual(snapshot[-1]["role"], "system")

            added = log.add("user", "Why not?", source="chat_input")
            self.assertEqual(added["id"], 7)

    def test_zero_max_entries_keeps_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_session.jsonl"
            log = ChatLog(path, max_entries=0)

            for idx in range(250):
                log.add("user", f"turn {idx}", source="test")

            snapshot = log.snapshot()
            self.assertEqual(len(snapshot), 250)
            self.assertEqual(snapshot[0]["text"], "turn 0")
            self.assertEqual(snapshot[-1]["text"], "turn 249")


if __name__ == "__main__":
    unittest.main(verbosity=2)
