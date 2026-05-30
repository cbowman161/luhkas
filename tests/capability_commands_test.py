#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))
sys.modules.setdefault("requests", types.SimpleNamespace())

from command_agent import CommandAgent
from router import Router
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
    def flush_pending_to(self, *args, **kwargs):
        return []

    def pop_alerts(self, *args, **kwargs):
        return []

    def update_capabilities(self, *args, **kwargs) -> None:
        pass

    def update_activity(self, *args, **kwargs) -> None:
        pass

    def has_display(self, node_id: str) -> bool:
        return True

    def has_audio(self, node_id: str) -> bool:
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


class FakeCodeMonkeyUpdates:
    def unread_updates(self):
        return [
            {
                "state": "test_failed",
                "message": "Verification failed after repairs",
                "task_id": "task-1",
                "goal": "Generate a deterministic learned-command recipe for Luhkas Vault.\n\nConfirmed user input: What is your RAM hardware brand?\nIntent: vault_system_info",
            },
            {
                "state": "verified",
                "message": "Tests passed",
                "task_id": "task-2",
                "goal": "Generate a deterministic learned-command recipe for Luhkas Vault.\n\nConfirmed user input: How much disk space does the vault have left?\nIntent: vault_disk_status",
            },
            {
                "state": "building",
                "message": "Building files",
                "task_id": "task-3",
                "goal": "Overwrite existing skill 'snakegame'. Original request: build a playable Snake game with pygame. Modification request: add tests.",
            },
            {
                "state": "queued",
                "message": "Queued",
                "task_id": "task-4",
                "goal": "Complete self-contained specification a developer can implement without further questions",
            },
            {
                "state": "build_failed",
                "message": "File generation failed",
                "task_id": "task-5",
                "goal": "Confirmed user input: What is the vault hostname?",
            },
        ]


class FakeEventLog:
    def unread(self):
        return []

    def mark_read(self, ids):
        self.marked = ids


class FakeBlackboardForRouter:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


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
    runtime._async_job_lock = threading.Lock()
    runtime._active_learn_jobs = {}
    runtime._active_install_jobs = {}
    runtime._async_jobs = {}
    runtime._async_job_seq = 0
    runtime._inline_alerts_tls = threading.local()
    return runtime


class CapabilityCommandsTest(unittest.TestCase):
    def test_presence_show_updates_uses_vault_capability_not_scout_routing(self) -> None:
        result = fake_runtime().handle_presence("show updates", node_id="scout", presence_context={})

        self.assertEqual(result["message"], "No unread updates.")
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["deterministic_source"], "code_monkey_updates")
        self.assertFalse(result["compose"])
        self.assertTrue(result["response_composed"])
        self.assertEqual(result["node_id"], "scout")

    def test_presence_updates_aliases_are_deterministic_runtime_commands(self) -> None:
        for user_input in ("updates", "notification", "notifications", "check notifications", "any updates"):
            with self.subTest(user_input=user_input):
                result = fake_runtime().handle_presence(user_input, node_id="scout", presence_context={})

                self.assertEqual(result["message"], "No unread updates.")
                self.assertTrue(result["deterministic"])
                self.assertEqual(result["deterministic_source"], "code_monkey_updates")
                self.assertFalse(result["compose"])

    def test_presence_job_aliases_are_deterministic_runtime_commands(self) -> None:
        for user_input in ("jobs", "tasks", "queue", "show jobs", "active jobs"):
            with self.subTest(user_input=user_input):
                result = fake_runtime().handle_presence(user_input, node_id="scout", presence_context={})

                self.assertEqual(result["message"], "No jobs yet.")
                self.assertTrue(result["deterministic"])
                self.assertEqual(result["deterministic_source"], "code_monkey_jobs")
                self.assertFalse(result["compose"])

    def test_presence_code_monkey_health_is_deterministic_runtime_command(self) -> None:
        result = fake_runtime().handle_presence("code monkey status", node_id="scout", presence_context={})

        self.assertEqual(result["message"], "Code Monkey service: ok")
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["deterministic_source"], "code_monkey_health")
        self.assertFalse(result["compose"])

    def test_presence_deterministic_commands_remember_the_presence_node(self) -> None:
        runtime = fake_runtime()
        delattr(runtime, "_current_node_id")

        result = runtime.handle_presence("updates", node_id="scout", presence_context={})

        self.assertEqual(result["node_id"], "scout")
        self.assertIn("scout", runtime._node_task_ids)

    def test_cli_updates_and_jobs_are_deterministic_runtime_commands(self) -> None:
        runtime = fake_runtime()

        updates = runtime.handle("What's the status?", node_id="cli")
        jobs = runtime.handle("jobs", node_id="cli")

        self.assertEqual(updates["message"], "No unread updates.")
        self.assertEqual(updates["deterministic_source"], "code_monkey_updates")
        self.assertFalse(updates["compose"])
        self.assertEqual(jobs["message"], "No jobs yet.")
        self.assertEqual(jobs["deterministic_source"], "code_monkey_jobs")
        self.assertFalse(jobs["compose"])

    def test_router_notifications_are_spoken_concise_summaries(self) -> None:
        router = Router.__new__(Router)
        router.code_monkey = FakeCodeMonkeyUpdates()
        router.event_log = FakeEventLog()
        router.blackboard = FakeBlackboardForRouter()

        result = router.show_updates()
        message = result["message"]

        self.assertIn("5 unread notifications.", message)
        self.assertIn("Code Monkey:", message)
        self.assertIn("Tests failed: What is your RAM hardware brand?", message)
        self.assertIn("Passed: How much disk space does the vault have left?", message)
        self.assertIn("1 more notification hidden.", message)
        self.assertIn("Say review for details.", message)
        self.assertNotIn("Generate a deterministic learned-command recipe", message)
        self.assertNotIn("Return an API-first capability", message)
        self.assertLess(len(message), 520)

    def test_presence_exact_capability_name_runs_system_capability(self) -> None:
        result = fake_runtime().handle_presence("memory usage", node_id="scout", presence_context={})

        self.assertIn("Mem: 1G used", result["message"])
        self.assertTrue(result["message"].startswith("Fallback response:"))
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["deterministic_source"], "capability:memory_usage")

    def test_presence_installed_command_agent_runs_before_scout_routing(self) -> None:
        result = fake_runtime().handle_presence("ping smoke", node_id="scout", presence_context={})

        self.assertIn("pong", result["message"])
        self.assertTrue(result["message"].startswith("Fallback response:"))
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
