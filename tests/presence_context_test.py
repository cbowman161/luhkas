#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOUT_INTEGRATION = ROOT / "vault" / "scout_integration.py"


def load_class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse(SCOUT_INTEGRATION.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing method {class_name}.{method_name}")


def calls_method(node: ast.AST, method_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and func.attr == method_name:
            return True
    return False


def load_function(name: str):
    module = ast.parse(SCOUT_INTEGRATION.read_text(encoding="utf-8"))
    function_nodes = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    constant_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"SCOUT_TOGGLE_DEFINITIONS", "_VISION_TRIGGER_PHRASES"}
            for target in node.targets
        )
    ]
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
                    "_presence_uses_previous_turn",
                    "_conversation_user_turns",
                    "_extract_context_phrase",
                    "_requested_context_phrase_label",
                    "_extract_context_fact",
                    "_matching_context_fact",
                    "_asks_recent_conversation",
                    "_normalize_source",
                    "_source_is_scout",
                    "_asks_feeling_state",
                    "_asks_casual_assistant_state",
                    "_asks_broad_status_report",
                    "_status_report_response_violation",
                    "_mood_statement_response_violation",
                    "_echoes_generation_instruction",
                    "_volunteers_node_boundary",
                    "_scout_status_facts",
                    "_status_report_statement",
                    "_operational_status_facts",
                    "_operational_status_statement",
                    "_rag_ingestion_runtime_status",
                    "_parse_scout_toggle_request",
                    "_toggle_state_value",
                    "_source_node_id",
                    "_self_route",
                    "_looks_like_scout_action",
                    "_looks_like_scout_hardware_command",
                    "_has_vision_trigger",
                    "_is_plain_general_question",
                    "_personality_preference_answer",
                    "_is_social_greeting",
                    "_is_conversation_context_setup",
                    "_asks_assistant_name",
                    "_asks_assistant_identity_topic",
                    "_asks_user_identity",
                    "_asks_registered_or_active_nodes",
                    "_asks_tracking_status",
                    "_asks_pose_interval",
                    "_asks_skill_inventory",
                    "_asks_capability_inventory",
                    "_asks_stored_knowledge_owner",
                    "_asks_camera_action_owner",
                    "_asks_why_here",
                    "_strip_emoji",
                    "_phrase_in_text",
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
            test_module = ast.Module(body=[future_annotations, *imports, *constant_nodes, *dependencies, node], type_ignores=[])
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

    def test_model_context_marks_new_topic_without_full_chat_but_keeps_stack(self) -> None:
        presence_conversation_context = load_function("_presence_conversation_context")
        conversation_user_turns = load_function("_conversation_user_turns")
        presence_uses_previous_turn = load_function("_presence_uses_previous_turn")
        presence = {
            "conversation_flow": {"mode": "new_topic", "uses_previous_turn": False},
            "chat_context": [
                {"role": "user", "text": "What is seven plus one?"},
                {"role": "assistant", "text": "8"},
                {"role": "user", "text": "What is nine minus four?"},
            ],
            "recent_contexts": [
                {"index": 1, "user": "What is seven plus one?", "assistant": "8"},
            ],
        }

        result = presence_conversation_context(presence)

        self.assertEqual(result["flow"]["mode"], "new_topic")
        self.assertFalse(presence_uses_previous_turn(presence))
        self.assertNotIn("chat_context", result)
        self.assertEqual(result["recent_contexts"][0]["assistant"], "8")
        self.assertEqual(conversation_user_turns(presence, "What is nine minus four?"), ["What is seven plus one?", "8"])

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

    def test_recent_conversation_answer_uses_assistant_side_of_context_stack(self) -> None:
        recent_conversation_answer = load_function("_recent_conversation_answer")
        presence = {
            "conversation_flow": {"mode": "followup", "uses_previous_turn": True},
            "recent_contexts": [
                {
                    "index": 1,
                    "user": "What test phrase did I just say?",
                    "assistant": "The test phrase was heliotrope canyon.",
                },
                {"index": 2, "user": "status", "assistant": "No unread updates."},
                {"index": 3, "user": "What is nine minus four?", "assistant": "5"},
            ],
        }

        self.assertEqual(
            recent_conversation_answer("What test phrase did I just say?", presence),
            "The test phrase was heliotrope canyon.",
        )

    def test_recent_conversation_answer_uses_assistant_side_fact_summaries(self) -> None:
        recent_conversation_answer = load_function("_recent_conversation_answer")
        presence = {
            "conversation_flow": {"mode": "followup", "uses_previous_turn": True},
            "recent_contexts": [
                {"index": 1, "user": "What is my favorite color?", "assistant": "Your favorite color is viridian."},
                {"index": 2, "user": "What kind of art did I say I like?", "assistant": "You said you like stark architecture best."},
                {"index": 3, "user": "What is nine minus four?", "assistant": "5"},
            ],
        }

        self.assertEqual(
            recent_conversation_answer("What is my favorite color?", presence),
            "Your favorite color is viridian.",
        )
        self.assertEqual(
            recent_conversation_answer("What kind of art did I say I like?", presence),
            "You said you like stark architecture best.",
        )

    def test_recent_conversation_answer_matches_requested_phrase_label(self) -> None:
        recent_conversation_answer = load_function("_recent_conversation_answer")
        presence = {
            "conversation_flow": {"mode": "followup", "uses_previous_turn": True},
            "recent_contexts": [
                {"index": 1, "user": "The test phrase is heliotrope canyon.", "assistant": "Got it. The test phrase is heliotrope canyon."},
                {"index": 2, "user": "The marker word is viridian lantern.", "assistant": "Got it. The marker word is viridian lantern."},
                {"index": 3, "user": "The token is stark architecture.", "assistant": "Got it. The token is stark architecture."},
            ],
        }

        self.assertEqual(
            recent_conversation_answer("What was the test phrase?", presence),
            "The test phrase was heliotrope canyon.",
        )
        self.assertEqual(
            recent_conversation_answer("What was the marker word?", presence),
            "The marker word was viridian lantern.",
        )
        self.assertEqual(
            recent_conversation_answer("What was the token?", presence),
            "The token was stark architecture.",
        )

    def test_recent_conversation_answer_uses_user_preference_context(self) -> None:
        recent_conversation_answer = load_function("_recent_conversation_answer")
        conversation_setup_answer = load_function("_conversation_setup_answer")
        presence = {
            "chat_context": [
                {"role": "user", "text": "My favorite color is green."},
                {"role": "assistant", "text": "I remember your favorite color is green."},
                {"role": "user", "text": "I like stark architecture best."},
                {"role": "assistant", "text": "I remember your like is stark architecture best."},
            ]
        }

        self.assertEqual(
            recent_conversation_answer("What is my favorite color?", presence),
            "Your favorite color is green.",
        )
        self.assertEqual(
            recent_conversation_answer("What kind of art did I say I like?", presence),
            "You said you like stark architecture best.",
        )
        self.assertEqual(
            conversation_setup_answer("I like stark architecture best."),
            "I remember you like stark architecture best.",
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
        self.assertTrue(is_conversation_context_setup("My favorite color is green."))
        self.assertTrue(is_conversation_context_setup("I like stark architecture best."))

    def test_mood_statement_compiles_values_into_feeling(self) -> None:
        mood_statement_from_state = load_function("_mood_statement_from_state")

        result = mood_statement_from_state(
            {"warmth": 0.7, "playfulness": 0.7, "brevity": 0.5},
            {"patience": 0.8, "irritation": 0.05, "arousal": 0.35, "valence": 0.2, "social_energy": 0.75},
            {"warmth": 0.55, "brevity": 0.6},
            False,
        )

        self.assertTrue(result.startswith("I feel "))
        self.assertIn("calm and steady", result)
        self.assertIn("patient", result)
        self.assertIn("open to people", result)
        self.assertIn("curious", result)
        self.assertNotIn("Personality state:", result)
        self.assertNotIn("Mood:", result)
        self.assertNotIn("0.", result)
        self.assertNotIn("thread", result.lower())
        self.assertNotIn("edges", result.lower())

    def test_assistant_intro_does_not_volunteer_creator_or_scout(self) -> None:
        assistant_identity_response_violation = load_function("_assistant_identity_response_violation")

        self.assertTrue(assistant_identity_response_violation("I'm Luhkas. You are not Chris."))
        self.assertTrue(assistant_identity_response_violation("I'm Luhkas through the Scout body."))
        self.assertFalse(assistant_identity_response_violation("I'm Luhkas, a local AI for memory and connected action."))

    def test_social_how_are_you_statement_does_not_claim_vision_state(self) -> None:
        mood_statement_from_state = load_function("_mood_statement_from_state")
        mood_statement_response_violation = load_function("_mood_statement_response_violation")

        result = mood_statement_from_state(
            {"warmth": 0.5, "playfulness": 0.5, "brevity": 0.7},
            {"patience": 0.7, "irritation": 0.1, "arousal": 0.3, "valence": 0.0, "social_energy": 0.5},
            {"warmth": 0.45, "brevity": 0.75},
            False,
        )

        lowered = result.lower()
        self.assertNotIn("tracking", lowered)
        self.assertNotIn("face", lowered)
        self.assertNotIn("chris", lowered)
        self.assertNotIn("scout", lowered)
        self.assertTrue(mood_statement_response_violation("steady, tracking three visible detections."))
        self.assertTrue(mood_statement_response_violation("steady, patient with the thread, edges softened."))
        self.assertFalse(mood_statement_response_violation(result))

    def test_feeling_state_is_distinct_from_plain_status(self) -> None:
        asks_feeling_state = load_function("_asks_feeling_state")
        asks_casual_assistant_state = load_function("_asks_casual_assistant_state")
        asks_broad_status_report = load_function("_asks_broad_status_report")

        self.assertTrue(asks_feeling_state("how do you feel"))
        self.assertTrue(asks_feeling_state("what is your current mood"))
        self.assertFalse(asks_feeling_state("how are you"))
        self.assertFalse(asks_casual_assistant_state("how do you feel"))
        self.assertTrue(asks_casual_assistant_state("how are you"))
        self.assertTrue(asks_broad_status_report("status report"))
        self.assertTrue(asks_broad_status_report("give me a status report"))

    def test_broad_status_routes_as_status_not_pure_personality(self) -> None:
        fast_route_message = load_function("_fast_route_message")
        source_node_id = load_function("_source_node_id")

        self.assertEqual(fast_route_message("How are you?")["self_route"]["route"], "personality")
        self.assertEqual(fast_route_message("Status report")["self_route"]["route"], "status")
        self.assertEqual(fast_route_message("How do you feel?")["self_route"]["route"], "personality")
        self.assertEqual(source_node_id(None, {"node_id": "scout"}), "scout")

    def test_common_chat_inputs_have_deterministic_fast_routes(self) -> None:
        fast_route_message = load_function("_fast_route_message")

        for text in ("Good morning", "How are you?", "Status report", "is tracking on?"):
            with self.subTest(text=text):
                route = fast_route_message(text)
                self.assertIsNotNone(route)
                self.assertTrue(route.get("deterministic"))
                self.assertIn(route.get("route"), {"self_question", "greeting"})

    def test_plain_world_questions_have_deterministic_fast_route(self) -> None:
        fast_route_message = load_function("_fast_route_message")

        for text in (
            "What causes the tides?",
            "Where is the Atacama Desert?",
            "Who wrote The Origin of Species?",
        ):
            with self.subTest(text=text):
                route = fast_route_message(text)
                self.assertIsNotNone(route)
                self.assertTrue(route.get("deterministic"))
                self.assertEqual(route.get("route"), "general_question")

    def test_plain_world_fast_route_preserves_vision_and_self_paths(self) -> None:
        fast_route_message = load_function("_fast_route_message")

        self.assertIsNone(fast_route_message("What do you see?"))
        self.assertEqual(fast_route_message("What is your name?")["self_route"]["route"], "assistant_identity")
        self.assertEqual(fast_route_message("Is tracking on?")["self_route"]["route"], "status")

    def test_broad_status_is_not_taken_as_direct_scout_action(self) -> None:
        looks_like_scout_action = load_function("_looks_like_scout_action")

        self.assertFalse(looks_like_scout_action("Status report"))
        self.assertFalse(looks_like_scout_action("How are you?"))
        self.assertFalse(looks_like_scout_action("How do you feel?"))
        self.assertTrue(looks_like_scout_action("start tracking"))

    def test_ui_toggle_requests_route_deterministically(self) -> None:
        parse_toggle = load_function("_parse_scout_toggle_request")
        toggle_state_value = load_function("_toggle_state_value")
        looks_like_scout_action = load_function("_looks_like_scout_action")

        cases = {
            "is face recognition on": ("face_recognition_enabled", None),
            "turn face recognition off": ("face_recognition_enabled", False),
            "face recognition on": ("face_recognition_enabled", True),
            "is wheel motion on": ("wheel_enabled", None),
            "wheel motion off": ("wheel_enabled", False),
            "enable collision avoidance": ("collision_avoidance_enabled", True),
            "guard mode off": ("guard_mode", False),
            "pose estimation on": ("pose_enabled", True),
            "auto reference capture disabled": ("auto_reference_capture_enabled", False),
            "filter ghosts by pose on": ("pose_filter_persons", True),
            "manual control off": ("manual_controller_enabled", False),
            "auto low-light on": ("camera_light_auto_enabled", True),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = parse_toggle(text)
                self.assertIsNotNone(result)
                self.assertEqual((result["state_key"], result["desired"]), expected)
                self.assertTrue(looks_like_scout_action(text))

        self.assertTrue(toggle_state_value({"gamepad": {"enabled": True}}, "manual_controller_enabled"))
        self.assertFalse(toggle_state_value({"wheel_enabled": False}, "wheel_enabled"))

    def test_standalone_confirmation_does_not_route_as_new_request(self) -> None:
        standalone_confirmation_answer = load_function("_standalone_confirmation_answer")

        self.assertEqual(
            standalone_confirmation_answer("yes"),
            "I do not have a pending confirmation for that.",
        )
        self.assertIsNone(standalone_confirmation_answer("yes, check the CPU"))

    def test_art_preference_asks_curiosity_followup_fast(self) -> None:
        personality_preference_answer = load_function("_personality_preference_answer")

        result = personality_preference_answer("What kind of art do you like?")

        self.assertIsNotNone(result)
        self.assertIn("Which direction should I learn from first", result)

    def test_hot_self_answers_do_not_call_generation_model(self) -> None:
        for method_name in (
            "fast_self_answer",
            "_assistant_identity_answer",
            "_assistant_status_answer",
            "_hardware_summary_answer",
            "_sensors_summary_answer",
            "_registered_nodes_answer",
            "_registered_capabilities_answer",
            "_registered_skills_answer",
        ):
            with self.subTest(method_name=method_name):
                method = load_class_method("ScoutVaultBridge", method_name)
                self.assertFalse(calls_method(method, "_generated_fact_answer"))

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

    def test_scout_origin_status_has_live_body_facts_without_identity_claims(self) -> None:
        scout_status_facts = load_function("_scout_status_facts")
        status_report_statement = load_function("_status_report_statement")
        status_report_response_violation = load_function("_status_report_response_violation")

        facts = scout_status_facts({
            "ok": True,
            "behavior": {"state": "MANUAL"},
            "target_state": "manual",
            "tracking_enabled": False,
            "follow_enabled": False,
            "guard_mode": False,
            "wheel_enabled": False,
            "collision_blocked": False,
            "detections": [
                {"label": "chair", "identity": "Chris"},
                {"label": "chair"},
            ],
        })
        result = status_report_statement("I feel steady.", facts)

        self.assertIn("Scout is in manual mode", result)
        self.assertIn("tracking off", result)
        self.assertIn("wheel drive off", result)
        self.assertIn("2 visible detections", result)
        self.assertNotIn("Chris", result)
        self.assertNotIn("face", result.lower())
        self.assertTrue(status_report_response_violation("tracking three faces, all known. Chris is among them."))
        self.assertFalse(status_report_response_violation(result))

    def test_operational_status_report_lists_running_services_and_option_states(self) -> None:
        operational_status_statement = load_function("_operational_status_statement")

        facts = {
            "nodes": {
                "vault": {"ok": True, "reachable": True, "services_down": []},
                "kiosk": {"ok": True, "reachable": True, "services_down": []},
                "scout": {"ok": True, "reachable": True, "services_down": []},
            }
        }
        result = operational_status_statement(facts)

        self.assertIn("All services running", result)
        self.assertLessEqual(len(result.split()), 45)
        self.assertNotIn("RTX 3090", result)
        self.assertNotIn("qwen3:8b", result)
        self.assertNotIn("Raspberry Pi", result)
        self.assertNotIn("I feel", result)
        self.assertNotIn("patient", result)

    def test_generated_responses_reject_prompt_echo_and_unasked_node_boundary(self) -> None:
        echoes_generation_instruction = load_function("_echoes_generation_instruction")
        volunteers_node_boundary = load_function("_volunteers_node_boundary")

        self.assertTrue(echoes_generation_instruction("The deterministic answer is present. I'll keep the same meaning."))
        self.assertFalse(echoes_generation_instruction("I feel steady, and Scout is in manual mode."))
        self.assertTrue(volunteers_node_boundary("Morning. I'm Luhkas, a system, not a node."))
        self.assertFalse(volunteers_node_boundary("Morning. I'm Luhkas."))

    def test_sanitizer_preserves_luhkas_name_mentions(self) -> None:
        sanitize_generated_response = load_function("_sanitize_generated_response")

        self.assertEqual(sanitize_generated_response("Luhkas. I feel steady."), "Luhkas. I feel steady.")
        self.assertEqual(sanitize_generated_response("Luhkas, and Scout is in manual mode."), "Luhkas, and Scout is in manual mode.")
        self.assertEqual(sanitize_generated_response("I'm Luhkas, a local AI."), "I'm Luhkas, a local AI.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
