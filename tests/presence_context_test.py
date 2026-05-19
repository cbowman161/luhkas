#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOUT_INTEGRATION = ROOT / "vault" / "scout_integration.py"


def load_function(name: str):
    module = ast.parse(SCOUT_INTEGRATION.read_text(encoding="utf-8"))
    function_nodes = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {}
            imports = [
                ast.Import(names=[ast.alias(name="re")]),
            ]
            dependencies = [
                function_nodes[dep]
                for dep in (
                    "_normalize_command_text",
                    "_self_topic_from_text",
                    "_extract_light_brightness",
                    "_has_any",
                    "_canonical_intent_text",
                    "_presence_conversation_context",
                    "_conversation_user_turns",
                    "_extract_context_phrase",
                    "_asks_recent_conversation",
                )
                if dep in function_nodes and dep != name
            ]
            future_annotations = ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            )
            for import_node in imports:
                ast.fix_missing_locations(import_node)
            ast.fix_missing_locations(future_annotations)
            test_module = ast.Module(body=[future_annotations, *imports, *dependencies, node], type_ignores=[])
            ast.fix_missing_locations(test_module)
            exec(
                compile(
                    test_module,
                    str(SCOUT_INTEGRATION),
                    "exec",
                ),
                namespace,
            )
            return namespace[name]
    raise AssertionError(f"missing function {name}")


class PresenceContextTest(unittest.TestCase):
    def test_model_context_preserves_full_chat_history(self) -> None:
        presence_conversation_context = load_function("_presence_conversation_context")
        chat_context = [
            {"role": "user", "text": f"turn {idx}", "source": "test"}
            for idx in range(20)
        ]
        presence = {
            "chat_context": chat_context,
            "reply_context": {
                "type": "reply_to_previous_assistant",
                "current_user_message": "Why not?",
                "previous_user_message": "Good evening",
                "previous_assistant_message": "Evening. I'm here, but not for you.",
            },
        }

        result = presence_conversation_context(presence)

        self.assertEqual(len(result["chat_context"]), 20)
        self.assertEqual(result["chat_context"][0], {"role": "user", "text": "turn 0"})
        self.assertEqual(result["chat_context"][-1], {"role": "user", "text": "turn 19"})
        self.assertEqual(
            result["reply_context"]["previous_assistant_message"],
            "Evening. I'm here, but not for you.",
        )

    def test_recent_conversation_phrases_are_not_vision_requests(self) -> None:
        asks_recent_conversation = load_function("_asks_recent_conversation")

        self.assertTrue(asks_recent_conversation("what did i just say the marker was"))
        self.assertTrue(asks_recent_conversation("what test phrase did i just say"))
        self.assertTrue(asks_recent_conversation("what did i ask immediately before this"))
        self.assertTrue(asks_recent_conversation("why not"))
        self.assertTrue(asks_recent_conversation("why that word"))
        self.assertTrue(asks_recent_conversation("what was my last question"))
        self.assertFalse(asks_recent_conversation("what do you see"))

    def test_context_setup_detects_plain_marker_phrase(self) -> None:
        is_conversation_context_setup = load_function("_is_conversation_context_setup")
        conversation_setup_answer = load_function("_conversation_setup_answer")
        recent_conversation_answer = load_function("_recent_conversation_answer")

        self.assertTrue(is_conversation_context_setup("the test phrase is blue comet"))
        self.assertEqual(
            conversation_setup_answer("The test phrase is blue comet."),
            "Got it. The test phrase is blue comet.",
        )
        self.assertIsNone(recent_conversation_answer("The test phrase is blue comet.", {}))

    def test_recent_conversation_answer_uses_chat_context(self) -> None:
        recent_conversation_answer = load_function("_recent_conversation_answer")
        presence = {
            "chat_context": [
                {"role": "user", "text": "The test phrase is blue comet."},
                {"role": "assistant", "text": "Got it. The test phrase is blue comet."},
                {"role": "user", "text": "What test phrase did I just say?"},
            ]
        }

        self.assertEqual(
            recent_conversation_answer("What test phrase did I just say?", presence),
            "The test phrase was blue comet.",
        )

    def test_context_setup_and_personality_state_route_deterministically(self) -> None:
        is_conversation_context_setup = load_function("_is_conversation_context_setup")
        self_topic_from_text = load_function("_self_topic_from_text")
        looks_like_scout_action = load_function("_looks_like_scout_action")

        self.assertTrue(
            is_conversation_context_setup(
                "full loop context test the test token is cedar reply with ok only"
            )
        )
        self.assertEqual(self_topic_from_text("what is your current personality state"), "personality")
        self.assertFalse(looks_like_scout_action("what is your current personality state"))

    def test_voice_band_formats_personality_state_values(self) -> None:
        voice_band = load_function("_voice_band")

        self.assertEqual(voice_band(0.1), "very low")
        self.assertEqual(voice_band(0.35), "low")
        self.assertEqual(voice_band(0.55), "medium")
        self.assertEqual(voice_band(0.75), "medium-high")
        self.assertEqual(voice_band(0.9), "high")
        self.assertEqual(voice_band(None), "unknown")

    def test_node_identity_claims_are_rejected(self) -> None:
        claims_assistant_is_node_identity = load_function("_claims_assistant_is_node_identity")
        asks_assistant_identity_topic = load_function("_asks_assistant_identity_topic")

        self.assertTrue(claims_assistant_is_node_identity("I am scout, active and ready."))
        self.assertTrue(claims_assistant_is_node_identity("I'm the scout node."))
        self.assertFalse(claims_assistant_is_node_identity("I'm Luhkas. Scout is one body I can use."))
        self.assertTrue(asks_assistant_identity_topic("are you scout"))

    def test_casual_assistant_state_is_detected(self) -> None:
        asks_casual_assistant_state = load_function("_asks_casual_assistant_state")

        self.assertTrue(asks_casual_assistant_state("how are you"))
        self.assertTrue(asks_casual_assistant_state("are you okay"))
        self.assertFalse(asks_casual_assistant_state("is tracking on"))

    def test_foreign_character_guard_exists_for_response_policy(self) -> None:
        has_excessive_foreign_chars = load_function("_has_excessive_foreign_chars")

        self.assertFalse(has_excessive_foreign_chars("You said the marker word was alder."))
        self.assertTrue(has_excessive_foreign_chars("Привет мир это полный русский ответ"))

    def test_scout_state_explanation_uses_luhkas_voice(self) -> None:
        scout_state_explanation = load_function("_scout_state_explanation")

        result = scout_state_explanation({
            "ok": True,
            "behavior": {"state": "MANUAL"},
            "target_state": "manual",
            "tracking_enabled": False,
            "wheel_enabled": False,
        })

        self.assertIn("I'm using Scout", result)
        self.assertNotIn("Scout is manual", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
