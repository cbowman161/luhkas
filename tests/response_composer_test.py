#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from response_composer import FALLBACK_PREFIX, ResponseComposer


class FakeModel:
    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, *args, **kwargs) -> str:
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response or ""


class ResponseComposerTest(unittest.TestCase):
    def test_model_failure_fallback_is_explicit(self) -> None:
        composer = ResponseComposer(FakeModel(error=RuntimeError("offline")))

        result = composer.compose(
            response_type="test",
            user_message="hello",
            facts={"deterministic_answer": "Hello."},
            fallback="Hello.",
        )

        self.assertTrue(result.startswith(FALLBACK_PREFIX))
        self.assertIn("Hello.", result)

    def test_success_uses_generated_response(self) -> None:
        composer = ResponseComposer(FakeModel(response="I have the fact, neatly."))

        result = composer.compose(
            response_type="test",
            user_message="hello",
            facts={"deterministic_answer": "The fact is loaded."},
            fallback="The fact is loaded.",
        )

        self.assertEqual(result, "I have the fact, neatly.")

    def test_policy_failure_uses_clean_fallback_without_internal_reason(self) -> None:
        composer = ResponseComposer(FakeModel(response="bad"))

        result = composer.compose(
            response_type="test",
            user_message="hello",
            facts={"deterministic_answer": "Good."},
            fallback="Good.",
            validator=lambda text: "nope" if text == "bad" else None,
        )

        self.assertEqual(result, "Good.")

    def test_identity_uses_generated_wording_not_canned_fallback(self) -> None:
        model = FakeModel(response="I'm Luhkas, a local AI for memory, thinking, and connected actions.")
        composer = ResponseComposer(model)

        result = composer.compose(
            response_type="assistant_identity",
            user_message="Who are you?",
            facts={
                "identity_name": "Luhkas",
                "self_description": "local AI",
                "capability_summary": "helps think, remember, and act through connected systems",
                "do_not_volunteer": ["creator", "Scout"],
            },
            fallback="I'm Luhkas: a local AI that helps think, remember, and act through the systems connected to me.",
            required_terms=("Luhkas",),
        )

        self.assertEqual(result, "I'm Luhkas, a local AI for memory, thinking, and connected actions.")
        self.assertNotEqual(result, "I'm Luhkas: a local AI that helps think, remember, and act through the systems connected to me.")
        self.assertTrue(model.calls)

    def test_identity_model_failure_is_explicit_fallback(self) -> None:
        composer = ResponseComposer(FakeModel(error=RuntimeError("offline")))

        result = composer.compose(
            response_type="assistant_identity",
            user_message="Who are you?",
            facts={"identity_name": "Luhkas"},
            fallback="I'm Luhkas: a local AI that helps think, remember, and act through the systems connected to me.",
            required_terms=("Luhkas",),
        )

        self.assertTrue(result.startswith(FALLBACK_PREFIX))

    def test_prompt_uses_direct_reply_instruction_not_manual_name_strip(self) -> None:
        model = FakeModel(response="I feel steady.")
        composer = ResponseComposer(model)

        composer.compose(
            response_type="status_report",
            user_message="how are you",
            facts={"deterministic_answer": "I feel steady."},
            fallback="I feel steady.",
        )

        prompt = model.calls[0][0][0]
        self.assertIn("direct conversational reply from Luhkas", prompt)
        self.assertIn("not format the reply like a transcript or speaker label", prompt)
        self.assertIn("Use the name Luhkas naturally only when the user asks", prompt)
        self.assertNotIn('Do not start with "Luhkas."', prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
