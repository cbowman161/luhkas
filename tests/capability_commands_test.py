#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))
sys.modules.setdefault("requests", types.SimpleNamespace())

from command_agent import CommandAgent
from vault_runtime import VaultRuntime


class FakeBlackboard:
    def __init__(self) -> None:
        self.pending = None
        self.session = {}

    def get_pending_decision(self):
        return self.pending

    def set(self, *args, **kwargs) -> None:
        pass


class FakeRegistry:
    def __init__(self) -> None:
        self.capabilities = [
            {
                "name": "show_updates",
                "description": "Show unread updates.",
                "subsystem": "event_log",
                "action": "show_updates",
                "examples": ["show updates"],
            },
            {
                "name": "memory_usage",
                "description": "Get memory usage.",
                "subsystem": "system_agent",
                "action": "dynamic_command",
                "examples": ["memory usage"],
                "command": "free -h",
            },
        ]

    def list(self):
        return list(self.capabilities)


class FakeSkillRegistry:
    def list(self):
        return [
            {
                "name": "smoke_skill",
                "description": "A test skill.",
                "filename": "skills/smoke_skill.py",
                "examples": ["smoke skill"],
            }
        ]


class FakeNodeRegistry:
    def update_capabilities(self, *args, **kwargs) -> None:
        pass

    def update_activity(self, *args, **kwargs) -> None:
        pass

    def has_display(self, node_id: str) -> bool:
        return True


class FakeSystemAgent:
    def run_direct(self, action, capability=None):
        return {"ok": True, "message": "Mem: 1G used", "action": action, "capability": capability}


class FakeRouter:
    def __init__(self) -> None:
        self.system_agent = FakeSystemAgent()
        self.last_skill = None

    def show_updates(self, active_task_id=None):
        return {"mode": "direct", "message": "No unread updates.", "active_task_id": active_task_id}

    def show_jobs(self, active_task_id=None):
        return {"mode": "direct", "message": "No jobs yet.", "active_task_id": active_task_id}

    def show_code_monkey_health(self, active_task_id=None):
        return {"mode": "direct", "message": "Code Monkey service: ok", "active_task_id": active_task_id}

    def set_last_skill(self, name, filename):
        self.last_skill = {"skill": name, "filename": filename}

    def confirm_existing_skill(self, plan, user_input, active_task_id=None):
        return {
            "mode": "direct",
            "message": f"I already have a skill for that: {plan['skill']}",
            "active_task_id": active_task_id,
        }

    def run_pending_skill(self, pending, active_task_id=None):
        return {
            "mode": "direct",
            "message": f"What arguments should I pass to `{pending['skill']}`?",
            "active_task_id": active_task_id,
        }

    def start_code_monkey_requirements(self, user_input, active_task_id=None):
        return {"mode": "direct", "message": "started coding task", "active_task_id": active_task_id}


class FakeCommandAgent:
    def handle(self, message):
        if message == "ping smoke":
            return {"mode": "direct", "message": "pong"}
        return None


def fake_runtime() -> VaultRuntime:
    runtime = VaultRuntime.__new__(VaultRuntime)
    runtime.registry = FakeRegistry()
    runtime.skill_registry = FakeSkillRegistry()
    runtime.blackboard = FakeBlackboard()
    runtime.router = FakeRouter()
    runtime.command_agent = FakeCommandAgent()
    runtime.node_registry = FakeNodeRegistry()
    runtime.active_task_id = None
    runtime._node_task_ids = {}
    runtime._current_node_id = "scout"
    return runtime


class CapabilityCommandsTest(unittest.TestCase):
    def test_presence_show_updates_uses_vault_capability_not_scout_routing(self) -> None:
        result = fake_runtime().handle_presence("show updates", node_id="scout", presence_context={})

        self.assertEqual(result["message"], "No unread updates.")
        self.assertEqual(result["node_id"], "scout")

    def test_presence_exact_capability_name_runs_system_capability(self) -> None:
        result = fake_runtime().handle_presence("memory usage", node_id="scout", presence_context={})

        self.assertEqual(result["message"], "Mem: 1G used")
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["deterministic_source"], "capability:memory_usage")

    def test_presence_installed_command_agent_runs_before_scout_routing(self) -> None:
        result = fake_runtime().handle_presence("ping smoke", node_id="scout", presence_context={})

        self.assertEqual(result["message"], "pong")
        self.assertEqual(result["deterministic_source"], "installed_capability_command")

    def test_presence_skill_name_routes_deterministically(self) -> None:
        result = fake_runtime().handle_presence("smoke skill", node_id="scout", presence_context={})

        self.assertIn("smoke_skill", result["message"])

    def test_command_agent_can_create_and_call_new_capability_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cap_dir = Path(tmp) / "smoke_capability"
            cap_dir.mkdir()
            (cap_dir / "commands.json").write_text(
                json.dumps({
                    "commands": [
                        {
                            "triggers": ["smoke capability", "echo smoke {value}"],
                            "action": "smoke",
                            "args": {"value": "string"},
                            "description": "smoke test capability",
                        }
                    ]
                }),
                encoding="utf-8",
            )
            (cap_dir / "api.py").write_text(
                "def smoke(value='ok'):\n"
                "    return {'ok': True, 'message': 'smoke ran', 'data': {'value': value}}\n",
                encoding="utf-8",
            )

            agent = CommandAgent(Path(tmp))
            direct = agent.handle("smoke capability")
            with_arg = agent.handle("echo smoke amber")

            self.assertIn("smoke ran", direct["message"])
            self.assertIn("value: ok", direct["message"])
            self.assertIn("value: amber", with_arg["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
