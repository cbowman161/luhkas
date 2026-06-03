#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.pop("luhkas_node", None)
LUHKAS_NODE_SERVICE = ROOT / "luhkas_node" / "service.py"

from luhkas_node.chat_context import build_presence_payload


def entry(role: str, text: str) -> dict:
    return {"role": role, "source": "test", "text": text}


class ChatContextTest(unittest.TestCase):
    def test_payload_includes_full_retained_chat_context(self) -> None:
        entries = [entry("user", f"old turn {idx}") for idx in range(20)]
        entries.append(entry("assistant", "The old context still matters."))
        entries.append(entry("user", "Why?"))

        payload = build_presence_payload("Why?", entries, "scout")

        self.assertEqual(len(payload["chat_context"]), 22)
        self.assertEqual(payload["chat_context"][0]["text"], "old turn 0")
        self.assertEqual(payload["chat_context"][-1]["text"], "Why?")

    def test_short_followup_links_to_previous_assistant_message(self) -> None:
        payload = build_presence_payload(
            "Why not?",
            [
                entry("user", "Good evening"),
                entry("assistant", "Evening. I'm here, but not for you."),
                entry("user", "Why not?"),
            ],
            "scout",
        )

        self.assertTrue(payload["conversation_continuity"])
        self.assertEqual(payload["reply_context"]["type"], "reply_to_previous_assistant")
        self.assertEqual(payload["reply_context"]["previous_user_message"], "Good evening")
        self.assertEqual(
            payload["reply_context"]["previous_assistant_message"],
            "Evening. I'm here, but not for you.",
        )

    def test_affirmative_reply_to_route_confirmation_restores_original_request(self) -> None:
        payload = build_presence_payload(
            "yes",
            [
                entry("user", "memory usage"),
                entry("assistant", "I think you mean a status or self-check. Is that right?"),
                entry("user", "yes"),
            ],
            "scout",
        )

        self.assertTrue(payload["clarification"])
        self.assertEqual(payload["clarified_request"], "memory usage")
        self.assertEqual(payload["routing_feedback"]["type"], "route_confirmation")
        self.assertEqual(payload["routing_feedback"]["previous_user_message"], "memory usage")

    def test_correction_reply_to_route_confirmation_builds_corrected_request(self) -> None:
        payload = build_presence_payload(
            "no, I meant memory usage",
            [
                entry("user", "memory usage"),
                entry("assistant", "I think you mean a vision analysis request. Is that right?"),
                entry("user", "no, I meant memory usage"),
            ],
            "scout",
        )

        self.assertTrue(payload["clarification"])
        self.assertEqual(payload["routing_feedback"]["type"], "route_correction")
        self.assertIn("memory usage", payload["clarified_request"])
        self.assertIn("Correction from user: no, I meant memory usage", payload["clarified_request"])

    def test_ui_chat_attempts_local_commands_before_vault(self) -> None:
        source = LUHKAS_NODE_SERVICE.read_text(encoding="utf-8")

        local_index = source.index("local_response = _local_command_handle(message)")
        vault_index = source.index("reply = _json_post(PRESENCE_URL")

        self.assertLess(local_index, vault_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
