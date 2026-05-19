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

    def generate(self, *args, **kwargs) -> str:
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

    def test_policy_failure_fallback_is_explicit(self) -> None:
        composer = ResponseComposer(FakeModel(response="bad"))

        result = composer.compose(
            response_type="test",
            user_message="hello",
            facts={"deterministic_answer": "Good."},
            fallback="Good.",
            validator=lambda text: "nope" if text == "bad" else None,
        )

        self.assertTrue(result.startswith(FALLBACK_PREFIX))


if __name__ == "__main__":
    unittest.main(verbosity=2)
