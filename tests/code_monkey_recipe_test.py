#!/usr/bin/env python3
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))
sys.modules.setdefault("requests", types.SimpleNamespace())

from code_monkey import recipe_generator


class FakeModel:
    response = "{}"

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        self.__class__.last_prompt = prompt
        return self.__class__.response


class CodeMonkeyRecipeGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.original_model = recipe_generator.LocalModel
        recipe_generator.LocalModel = FakeModel
        self.addCleanup(setattr, recipe_generator, "LocalModel", self.original_model)

    def test_generates_valid_single_bash_recipe_from_standard_prompt(self):
        FakeModel.response = (
            '{"type":"bash","command":"uname -a",'
            '"required_facts":["kernel"],"summary_hint":"kernel version"}'
        )

        result = recipe_generator.generate_learned_command_recipe({
            "input": "What kernel is the vault running?",
            "intent": "vault_learned_command",
            "description": "Vault operating system version",
            "target": "vault",
        })

        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["recipe"]["type"], "bash")
        self.assertEqual(result["recipe"]["command"], "uname -a")
        self.assertIn("Confirmed request context", FakeModel.last_prompt)
        self.assertIn("Return ONLY one compact JSON object", FakeModel.last_prompt)

    def test_rejects_destructive_generated_command(self):
        FakeModel.response = '{"type":"bash","command":"sudo rm -rf /","required_facts":[]}'

        result = recipe_generator.generate_learned_command_recipe({
            "input": "Check the disk",
            "intent": "vault_learned_command",
        })

        self.assertFalse(result["ok"])
        self.assertIn("blocked", result["error"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
