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

from learned_capabilities import LearnedCapabilityEngine, LearnedCapabilityStore
from vault_runtime import VaultRuntime


class TestLearnedCapabilityEngine(LearnedCapabilityEngine):
    def _run_bash(self, command: str) -> dict:
        return {
            "ok": True,
            "stdout": f"ran {command}",
            "stderr": "",
            "returncode": 0,
            "error": None,
            "ran_at": 1.0,
        }

    def _run_python_script(self, execution: dict) -> dict:
        source = execution.get("source") or ""
        stdout = "script ran\n"
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
            "returncode": 0,
            "error": None,
            "ran_at": 1.0,
        }


class FakeCodeMonkeyClient:
    def __init__(self) -> None:
        self.goals = []
        self.statuses = {}
        self.recipes = []

    def start_task(self, goal: str) -> str:
        self.goals.append(goal)
        return "cm-task-1"

    def generate_learned_command_recipe(self, user_input, proposal):
        self.recipes.append({"input": user_input, "proposal": proposal})
        normalized = str(user_input).lower()
        if "python" in normalized:
            return {
                "ok": True,
                "generator": "code_monkey_single_recipe",
                "recipe": {
                    "type": "python_script",
                    "filename": "python_runtime.py",
                    "source": "import sys\nprint('Python ' + sys.version.split()[0])\n",
                    "required_facts": ["Python"],
                },
            }
        return {
            "ok": True,
            "generator": "code_monkey_single_recipe",
            "recipe": {
                "type": "bash",
                "command": "echo learned",
                "required_facts": ["learned"],
            },
        }

    def safe_status(self, task_id: str) -> dict:
        return self.statuses.get(task_id, {"task_id": task_id, "state": "queued"})

    def get_artifacts(self, task_id: str) -> dict:
        return {"task_id": task_id, "state": self.statuses.get(task_id, {}).get("state", "unknown")}


class FakeBlackboard:
    def __init__(self) -> None:
        self.pending = None

    def get_pending_decision(self):
        return self.pending

    def set_pending_decision(self, value):
        self.pending = value

    def clear_pending_decision(self):
        self.pending = None


class FakeNodeRegistry:
    def flush_pending_to(self, *args, **kwargs):
        return []

    def pop_alerts(self, *args, **kwargs):
        return []

    def has_display(self, node_id: str) -> bool:
        return True

    def has_audio(self, node_id: str) -> bool:
        return True


class FakeEventLog:
    def unread(self):
        return []


class FakeRegistry:
    def list(self):
        return []

    def lookup_by_alias(self, normalized_text: str):
        return None


class FakeSkillRegistry:
    def list(self):
        return []


class FakeCommandAgent:
    def handle(self, message):
        return None


class FakeRouter:
    def show_updates(self, active_task_id=None):
        return {"mode": "direct", "message": "updates", "active_task_id": active_task_id}

    def show_jobs(self, active_task_id=None):
        return {"mode": "direct", "message": "jobs", "active_task_id": active_task_id}

    def show_code_monkey_health(self, active_task_id=None):
        return {"mode": "direct", "message": "coder", "active_task_id": active_task_id}


class FakeClassroom:
    def maybe_handle_turn(self, message: str, node_id: str):
        return None


class FakeChatSessions:
    def __init__(self) -> None:
        self.awaiting = {}
        self.closed = []

    def set_awaiting(self, node_id: str, value):
        self.awaiting[node_id] = value

    def get_active(self, node_id: str):
        return None

    def close(self, node_id: str, outcome=None):
        self.closed.append({"node_id": node_id, "outcome": outcome})


def fake_runtime(engine: LearnedCapabilityEngine) -> VaultRuntime:
    runtime = VaultRuntime.__new__(VaultRuntime)
    runtime.registry = FakeRegistry()
    runtime.skill_registry = FakeSkillRegistry()
    runtime.blackboard = FakeBlackboard()
    runtime.router = FakeRouter()
    runtime.command_agent = FakeCommandAgent()
    runtime.node_registry = FakeNodeRegistry()
    runtime.event_log = FakeEventLog()
    runtime.classroom = FakeClassroom()
    runtime.chat_sessions = FakeChatSessions()
    runtime.active_task_id = None
    runtime._node_task_ids = {}
    runtime._current_node_id = "scout"
    runtime._node_pendings = {}
    runtime._node_pendings_lock = threading.RLock()
    runtime.learned_capabilities = engine
    runtime._async_job_lock = threading.Lock()
    runtime._active_learn_jobs = {}
    runtime._active_install_jobs = {}
    runtime._inline_alerts_tls = threading.local()

    def _sync_spawn_async_learn(self, *, original_message: str, proposal: dict, confirmed_by: str, node_id: str) -> None:
        engine.learn_and_execute(original_message, proposal, confirmed_by=confirmed_by)

    runtime._spawn_async_learn = types.MethodType(_sync_spawn_async_learn, runtime)
    return runtime


class LearnedCapabilitiesTest(unittest.TestCase):
    def make_engine(self):
        tmp = tempfile.TemporaryDirectory()
        store = LearnedCapabilityStore(Path(tmp.name) / "capabilities.json")
        engine = TestLearnedCapabilityEngine(store, code_monkey_client=FakeCodeMonkeyClient())
        self.addCleanup(tmp.cleanup)
        return engine

    def test_cpu_request_proposes_vault_capability_not_scout_action(self) -> None:
        engine = self.make_engine()

        proposal = engine.propose("What's your CPU?")

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["intent"], "vault_learned_command")
        self.assertEqual(proposal["inferred"], {"topic": "cpu", "aspect": "hardware"})
        self.assertEqual(proposal["target"], "vault")
        self.assertFalse(proposal["queue_code_monkey"])
        self.assertEqual(proposal["planner"], "code_monkey_single_recipe")
        self.assertEqual(engine.propose("CPU status")["intent"], "vault_learned_command")
        self.assertEqual(engine.propose("What CPU usage is the vault at right now?")["intent"], "vault_learned_command")
        self.assertEqual(engine.propose("How long has the vault machine been up?")["intent"], "vault_learned_command")
        self.assertEqual(
            engine.propose("Can you inspect the Python executable path and version for the vault?")["intent"],
            "vault_learned_command",
        )
        self.assertIsNone(engine.propose("turn your camera light on"))

    def test_ram_hardware_correction_is_inferred_without_specific_route(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        first = runtime._handle_deterministic_presence_command("What's your RAM?", "scout")
        self.assertIn("Vault memory usage", first["message"])

        corrected = runtime._handle_deterministic_presence_command("no, RAM hardware", "scout")

        self.assertIn("Vault RAM hardware", corrected["message"])
        pending = runtime._get_pending("scout")
        self.assertEqual(pending["proposal"]["intent"], "vault_learned_command")
        self.assertEqual(pending["proposal"]["inferred"], {"topic": "memory", "aspect": "hardware"})
        self.assertFalse(pending["proposal"]["queue_code_monkey"])
        self.assertEqual(engine.store.all()["capabilities"], {})

    def test_fragment_correction_inherits_pending_topic(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        runtime._handle_deterministic_presence_command("What's your RAM?", "scout")
        corrected = runtime._handle_deterministic_presence_command("no, hardware", "scout")

        self.assertIn("Vault RAM hardware", corrected["message"])
        pending = runtime._get_pending("scout")
        self.assertEqual(pending["original_message"], "What's your RAM?")
        self.assertEqual(pending["proposal"]["inferred"], {"topic": "memory", "aspect": "hardware"})

        confirmed = runtime._handle_deterministic_presence_command("yes", "scout")
        saved = engine.store.all()["capabilities"]
        self.assertIn("whats your ram", saved)
        self.assertIn("I'll work on that", confirmed["message"])
        self.assertEqual(confirmed["deterministic_source"], "learned_capability_async")
        self.assertEqual(engine.code_monkey.recipes[-1]["input"], "What's your RAM?")

    def test_hardware_recipe_is_not_predefined_and_is_queued_for_learning(self) -> None:
        engine = self.make_engine()

        proposal = engine.propose("RAM hardware")

        self.assertEqual(proposal["intent"], "vault_learned_command")
        self.assertEqual(proposal["inferred"], {"topic": "memory", "aspect": "hardware"})
        self.assertFalse(proposal["queue_code_monkey"])
        with self.assertRaisesRegex(ValueError, "Local learned-command recipe templates are disabled"):
            engine.build_recipe("RAM hardware", proposal)

    def test_gpu_hardware_correction_is_inferred_and_safe(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        first = runtime._handle_deterministic_presence_command("What's your GPU?", "scout")
        self.assertIn("Vault GPU usage", first["message"])

        corrected = runtime._handle_deterministic_presence_command("no, hardware", "scout")

        self.assertIn("Vault GPU hardware", corrected["message"])
        pending = runtime._get_pending("scout")
        self.assertEqual(pending["proposal"]["intent"], "vault_learned_command")
        self.assertEqual(pending["proposal"]["inferred"], {"topic": "gpu", "aspect": "hardware"})
        self.assertFalse(pending["proposal"]["queue_code_monkey"])

        confirmed = runtime._handle_deterministic_presence_command("yes", "scout")
        self.assertIn("I'll work on that", confirmed["message"])

    def test_confirmation_creates_executable_recipe_and_reuses_it(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        first = runtime._handle_deterministic_presence_command("What's your CPU?", "scout")
        self.assertIsNotNone(first)
        self.assertIn("I think you mean", first["message"])
        self.assertEqual(runtime._get_pending("scout")["type"], "learned_capability_confirmation")

        confirmed = runtime._handle_deterministic_presence_command("yes", "scout")
        self.assertIsNotNone(confirmed)
        self.assertEqual("learned_capability_async", confirmed["deterministic_source"])
        self.assertIn("I'll work on that", confirmed["message"])
        self.assertIsNone(runtime._get_pending("scout"))

        saved = engine.store.all()["capabilities"]
        self.assertIn("whats your cpu", saved)
        self.assertEqual(len(engine.code_monkey.recipes), 1)

    def test_cpu_status_uses_code_monkey_single_recipe_planner(self) -> None:
        engine = self.make_engine()

        proposal = engine.propose("CPU status")

        self.assertEqual(proposal["intent"], "vault_learned_command")
        self.assertEqual(proposal["planner"], "code_monkey_single_recipe")

    def test_finished_code_monkey_work_order_is_attached_to_next_learned_interaction(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        task = {"ok": True, "task_id": "cm-task-1", "goal": "build a special learned command", "state": "queued"}
        proposal = {
            "intent": "custom_recipe",
            "description": "a special learned command",
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.7,
        }
        engine.store.remember_pending_code_monkey("build a special learned command", proposal, task)
        engine.code_monkey.statuses["cm-task-1"] = {"task_id": "cm-task-1", "state": "verified"}

        update = runtime._handle_deterministic_presence_command("How much disk space does the vault have left?", "scout")

        self.assertIn("Code Monkey finished task cm-task-1", update["message"])
        self.assertEqual(update["deterministic_source"], "learned_capability_confirmation")
        self.assertIn("Is that right?", update["message"])
        pending = engine.store.pending_code_monkey()["cm-task-1"]
        self.assertTrue(pending["notified"])

    def test_failed_code_monkey_learning_task_removes_learned_command(self) -> None:
        engine = self.make_engine()
        proposal = {
            "intent": "vault_system_info",
            "description": "Vault CPU hardware",
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.84,
            "inferred": {"topic": "cpu", "aspect": "hardware"},
        }
        engine.store.remember(
            "check CPU",
            {
                **proposal,
                "execution": {"type": "bash", "command": "lscpu"},
                "code_monkey_task": {"ok": True, "task_id": "cm-task-1"},
            },
        )
        engine.store.remember_pending_code_monkey(
            "check CPU",
            proposal,
            {"ok": True, "task_id": "cm-task-1", "goal": "learn CPU hardware", "state": "queued"},
        )
        engine.code_monkey.statuses["cm-task-1"] = {"task_id": "cm-task-1", "state": "test_failed"}

        updates = engine.check_pending_code_monkey()

        self.assertEqual(updates[0]["state"], "test_failed")
        self.assertTrue(engine.store.pending_code_monkey()["cm-task-1"]["learned_removed"])
        self.assertNotIn("check cpu", engine.store.all()["capabilities"])
        self.assertIn("no learned command was saved", engine.summarize_pending_update(updates[0]))

    def test_denial_does_not_save_recipe(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        runtime._handle_deterministic_presence_command("What's your CPU?", "scout")
        denied = runtime._handle_deterministic_presence_command("no", "scout")

        self.assertIn("will not save", denied["message"])
        self.assertEqual(engine.store.all()["capabilities"], {})

    def test_correction_requires_second_confirmation_before_saving(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)

        runtime._handle_deterministic_presence_command("What's your CPU?", "scout")
        corrected = runtime._handle_deterministic_presence_command("No, I mean memory usage", "scout")

        self.assertIn("memory", corrected["message"].lower())
        self.assertEqual(engine.store.all()["capabilities"], {})

        runtime._handle_deterministic_presence_command("yes", "scout")
        saved = engine.store.all()["capabilities"]
        self.assertIn("memory usage", saved)
        self.assertEqual(saved["memory usage"]["intent"], "vault_learned_command")
        self.assertEqual(saved["memory usage"]["inferred"], {"topic": "memory", "aspect": "usage"})

    def test_python_recipe_is_supported_for_python_runtime_queries(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = LearnedCapabilityEngine(
            LearnedCapabilityStore(Path(tmp.name) / "capabilities.json"),
            code_monkey_client=FakeCodeMonkeyClient(),
        )
        proposal = engine.propose("What Python version are you running?")
        result = engine.learn_and_execute("What Python version are you running?", proposal)

        self.assertTrue(result["ok"], result.get("error"))
        capability = result["capability"]
        self.assertEqual(capability["execution"]["type"], "python_script")
        self.assertTrue(Path(capability["execution"]["path"]).exists())
        self.assertIn("Python", result["stdout"])
        self.assertTrue(capability["code_monkey_task"]["single_recipe"])

    def test_old_builtin_records_do_not_repeat_code_monkey_note(self) -> None:
        engine = self.make_engine()
        capability = {
            "intent": "vault_memory_status",
            "description": "Vault memory usage",
            "execution": {"type": "bash", "command": "free -h"},
            "code_monkey_task": {"ok": True, "task_id": "old-task"},
        }
        result = {"ok": True, "stdout": "Mem: 1Gi 512Mi 512Mi", "stderr": "", "returncode": 0}

        summary = engine.summarize_result("Can you check memory usage?", capability, result)

        self.assertNotIn("Code Monkey task", summary)
        self.assertNotIn("For now:", summary)
        self.assertIn("Vault memory usage", summary)

    def test_builtin_learned_summary_answers_directly_without_for_now_prefix(self) -> None:
        engine = self.make_engine()
        capability = {
            "intent": "vault_cpu_usage",
            "description": "Vault CPU usage",
            "execution": {"type": "python_script"},
            "code_monkey_task": {"ok": False, "skipped": True},
        }
        result = {"ok": True, "stdout": "CPU usage percent: 7.4\n", "stderr": "", "returncode": 0}

        summary = engine.summarize_result("CPU status", capability, result)

        self.assertEqual(summary, "Vault CPU usage: 7.4%.")
        self.assertNotIn("For now:", summary)

    def test_queued_code_monkey_learned_summary_uses_current_result_prefix(self) -> None:
        engine = self.make_engine()
        capability = {
            "intent": "custom_recipe",
            "description": "custom Vault check",
            "execution": {"type": "bash"},
            "code_monkey_task": {"ok": True, "task_id": "cm-task-1"},
        }
        result = {"ok": True, "stdout": "custom output\n", "stderr": "", "returncode": 0}

        summary = engine.summarize_result("custom check", capability, result)

        self.assertIn("I queued Code Monkey task cm-task-1", summary)
        self.assertIn("Current result: custom Vault check: custom output", summary)
        self.assertNotIn("For now:", summary)

    def test_new_learned_command_bypasses_previous_execution_review(self) -> None:
        engine = self.make_engine()
        runtime = fake_runtime(engine)
        cpu_cap = {
            "intent": "vault_cpu_usage",
            "description": "Vault CPU usage",
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.9,
            "inferred": {"topic": "cpu", "aspect": "usage"},
            "execution": {"type": "bash", "command": "echo cpu usage", "required_facts": ["cpu"]},
            "code_monkey_task": {"ok": False, "skipped": True},
        }
        disk_cap = {
            "intent": "vault_disk_usage",
            "description": "Vault disk usage",
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.9,
            "inferred": {"topic": "disk", "aspect": "usage"},
            "execution": {"type": "bash", "command": "echo disk usage", "required_facts": ["disk"]},
            "code_monkey_task": {"ok": False, "skipped": True},
        }
        engine.store.remember("cpu usage", cpu_cap)
        engine.store.remember("disk usage", disk_cap)

        first = runtime._handle_deterministic_presence_command("cpu usage", "scout")
        self.assertIn("Learned command.", first["message"])
        self.assertEqual(runtime._get_pending("scout")["type"], "learned_execution_review")

        second = runtime._handle_deterministic_presence_command("disk usage", "scout")

        self.assertIn("Learned command.", second["message"])
        self.assertEqual(second["deterministic_source"], "learned_capability:vault_disk_usage")
        caps = engine.store.all()["capabilities"]
        self.assertIn("cpu usage", caps)
        self.assertIn("disk usage", caps)
        self.assertIsNone(caps["cpu usage"].get("alias_of"))
        self.assertIsNone(caps["disk usage"].get("alias_of"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
