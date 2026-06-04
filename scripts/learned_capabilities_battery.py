#!/usr/bin/env python3
"""Isolated learned-capabilities E2E battery.

This runs the learned-capability proposal, confirmation, execution, alias,
correction, persistence, and safety paths against a temporary store. It does
not touch the live learned-capabilities database or call the live Code Monkey.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("lancedb", types.SimpleNamespace(connect=lambda *args, **kwargs: None))
sys.modules.setdefault(
    "pyarrow",
    types.SimpleNamespace(
        schema=lambda *args, **kwargs: None,
        field=lambda *args, **kwargs: None,
        string=lambda *args, **kwargs: None,
        float64=lambda *args, **kwargs: None,
        float32=lambda *args, **kwargs: None,
        list_=lambda *args, **kwargs: None,
    ),
)

from learned_capabilities import LearnedCapabilityEngine, LearnedCapabilityStore, normalize_text
from vault_runtime import VaultRuntime


@dataclass
class Result:
    label: str
    ok: bool
    seconds: float
    detail: Any = None


class BatteryFailure(AssertionError):
    pass


class BatteryEngine(LearnedCapabilityEngine):
    def _run_bash(self, command: str) -> dict:
        safety = self.safety.validate_command(command)
        if not safety.get("allowed"):
            return self._error(safety.get("reason") or "Command failed safety validation.")
        return {
            "ok": True,
            "stdout": f"ran {command}",
            "stderr": "",
            "returncode": 0,
            "error": None,
            "ran_at": time.time(),
        }

    def _run_python_script(self, execution: dict) -> dict:
        return {
            "ok": True,
            "stdout": "Python 3.test\n",
            "stderr": "",
            "returncode": 0,
            "error": None,
            "ran_at": time.time(),
        }


class FakeCodeMonkeyClient:
    def __init__(self) -> None:
        self.recipes: list[dict] = []
        self.statuses: dict[str, dict] = {}
        self.goals: list[str] = []

    def start_task(self, goal: str) -> str:
        self.goals.append(goal)
        return f"cm-task-{len(self.goals)}"

    def generate_learned_command_recipe(self, user_input: str, proposal: dict) -> dict:
        self.recipes.append({"input": user_input, "proposal": proposal})
        normalized = normalize_text(user_input)
        if "dangerous" in normalized:
            return {
                "ok": True,
                "generator": "fake_code_monkey",
                "recipe": {"type": "bash", "command": "rm -rf /tmp/luhkas-danger", "required_facts": []},
            }
        if "python" in normalized:
            return {
                "ok": True,
                "generator": "fake_code_monkey",
                "recipe": {
                    "type": "python_script",
                    "filename": "python_runtime.py",
                    "source": "print('Python 3.test')\n",
                    "required_facts": ["Python"],
                },
            }
        inferred = proposal.get("inferred") or {}
        topic = inferred.get("topic") or "system"
        aspect = inferred.get("aspect") or "status"
        return {
            "ok": True,
            "generator": "fake_code_monkey",
            "recipe": {
                "type": "bash",
                "command": f"printf '{topic} {aspect}\\n'",
                "required_facts": [topic, aspect],
            },
        }

    def safe_status(self, task_id: str) -> dict:
        return self.statuses.get(task_id, {"task_id": task_id, "state": "queued"})

    def get_artifacts(self, task_id: str) -> dict:
        return {"task_id": task_id, "state": self.safe_status(task_id).get("state")}


class FakeBlackboard:
    def __init__(self) -> None:
        self.pending = None

    def get_pending_decision(self):
        return self.pending

    def set_pending_decision(self, value):
        self.pending = value

    def clear_pending_decision(self):
        self.pending = None


class FakeChatSessions:
    def __init__(self) -> None:
        self.awaiting: dict[str, Any] = {}
        self.active: dict[str, Any] = {}
        self.closed: list[dict] = []

    def set_awaiting(self, node_id: str, value: Any) -> None:
        self.awaiting[node_id] = value

    def get_active(self, node_id: str):
        return self.active.get(node_id)

    def close(self, node_id: str, outcome: dict | None = None) -> None:
        self.closed.append({"node_id": node_id, "outcome": outcome})
        self.active.pop(node_id, None)


class FakeRouter:
    def show_updates(self, active_task_id=None):
        return {"mode": "direct", "message": "updates", "active_task_id": active_task_id}

    def show_jobs(self, active_task_id=None):
        return {"mode": "direct", "message": "jobs", "active_task_id": active_task_id}

    def show_code_monkey_health(self, active_task_id=None):
        return {"mode": "direct", "message": "coder", "active_task_id": active_task_id}


class FakeRegistry:
    def list(self):
        return []

    def lookup_by_alias(self, normalized_text: str):
        return None


class FakeNodeRegistry:
    def has_display(self, node_id: str) -> bool:
        return True

    def has_audio(self, node_id: str) -> bool:
        return True

    def pop_alerts(self, node_id: str):
        return []


class FakeEventLog:
    def unread(self):
        return []


class FakeClassroom:
    def maybe_handle_turn(self, message: str, node_id: str):
        return None


class FakeCommandAgent:
    def handle(self, message):
        return None


def fake_runtime(engine: LearnedCapabilityEngine) -> VaultRuntime:
    runtime = VaultRuntime.__new__(VaultRuntime)
    runtime.registry = FakeRegistry()
    runtime.skill_registry = FakeRegistry()
    runtime.blackboard = FakeBlackboard()
    runtime.router = FakeRouter()
    runtime.command_agent = FakeCommandAgent()
    runtime.node_registry = FakeNodeRegistry()
    runtime.event_log = FakeEventLog()
    runtime.classroom = FakeClassroom()
    runtime.active_task_id = None
    runtime._node_task_ids = {}
    runtime._current_node_id = "scout"
    runtime._node_pendings = {}
    runtime._node_pendings_lock = threading.RLock()
    runtime.learned_capabilities = engine
    runtime.chat_sessions = FakeChatSessions()
    runtime._async_job_lock = threading.Lock()
    runtime._active_learn_jobs = {}
    runtime._active_install_jobs = {}
    runtime._inline_alerts_tls = threading.local()

    def _sync_spawn_async_learn(self, *, original_message: str, proposal: dict, confirmed_by: str, node_id: str) -> None:
        engine.learn_and_execute(original_message, proposal, confirmed_by=confirmed_by)

    runtime._spawn_async_learn = types.MethodType(_sync_spawn_async_learn, runtime)
    return runtime


class LearnedCapabilitiesBattery:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.results: list[Result] = []

    def run(self) -> int:
        for label, scenario in (
            ("proposal classifier boundaries", self.proposal_classifier_boundaries),
            ("confirm save and exact reuse", self.confirm_save_and_exact_reuse),
            ("correction requires second confirmation", self.correction_requires_second_confirmation),
            ("denial does not persist", self.denial_does_not_persist),
            ("concept alias reuse", self.concept_alias_reuse),
            ("execution review reroutes correction", self.execution_review_reroutes_correction),
            ("new learned command bypasses review", self.new_learned_command_bypasses_review),
            ("python recipe materialization", self.python_recipe_materialization),
            ("dangerous generated recipe rejected", self.dangerous_generated_recipe_rejected),
            ("store persistence and pending updates", self.store_persistence_and_pending_updates),
        ):
            self._record(label, scenario)
        self.print_summary()
        return 1 if any(not result.ok for result in self.results) else 0

    def make_engine(self) -> tuple[tempfile.TemporaryDirectory, BatteryEngine]:
        tmp = tempfile.TemporaryDirectory()
        store = LearnedCapabilityStore(Path(tmp.name) / "capabilities.json")
        engine = BatteryEngine(store, code_monkey_client=FakeCodeMonkeyClient(), model=None)
        return tmp, engine

    def proposal_classifier_boundaries(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            cpu = self.require(engine.propose("What's your CPU?"), "CPU proposal missing")
            ram = self.require(engine.propose("RAM hardware"), "RAM hardware proposal missing")
            py = self.require(engine.propose("Can you inspect the Python executable path and version for the vault?"), "Python proposal missing")
            self.eq(cpu["inferred"], {"topic": "cpu", "aspect": "hardware"})
            self.eq(ram["inferred"], {"topic": "memory", "aspect": "hardware"})
            self.eq(py["inferred"], {"topic": "python", "aspect": "version"})
            self.is_none(engine.propose("turn your camera light on"))
            self.is_none(engine.propose("who am I"))
            return {"cpu": cpu["description"], "ram": ram["description"], "python": py["description"]}
        finally:
            tmp.cleanup()

    def confirm_save_and_exact_reuse(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            first = self.require(runtime._handle_deterministic_presence_command("What's your CPU?", "scout"))
            self.contains(first["message"], "Is that right?")
            self.eq(runtime._get_pending("scout")["type"], "learned_capability_confirmation")
            confirmed = self.require(runtime._handle_deterministic_presence_command("yes", "scout"))
            self.eq(confirmed["deterministic_source"], "learned_capability_async")
            saved = engine.store.all()["capabilities"]
            self.contains(" ".join(saved.keys()), "whats your cpu")
            reused = self.require(runtime._handle_deterministic_presence_command("What's your CPU?", "scout"))
            self.contains(reused["message"], "Learned command.")
            self.contains(reused["deterministic_source"], "learned_capability:")
            return {"saved_keys": sorted(saved.keys()), "reuse_source": reused["deterministic_source"]}
        finally:
            tmp.cleanup()

    def correction_requires_second_confirmation(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            runtime._handle_deterministic_presence_command("What's your RAM?", "scout")
            corrected = self.require(runtime._handle_deterministic_presence_command("no, hardware", "scout"))
            self.contains(corrected["message"], "Vault RAM hardware")
            self.eq(engine.store.all()["capabilities"], {})
            confirmed = self.require(runtime._handle_deterministic_presence_command("yes", "scout"))
            self.contains(confirmed["message"], "I'll work on that")
            cap = engine.store.all()["capabilities"]["whats your ram"]
            self.eq(cap["inferred"], {"topic": "memory", "aspect": "hardware"})
            return {"saved": cap["description"]}
        finally:
            tmp.cleanup()

    def denial_does_not_persist(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            runtime._handle_deterministic_presence_command("What's your GPU?", "scout")
            denied = self.require(runtime._handle_deterministic_presence_command("no", "scout"))
            self.contains(denied["message"], "will not save")
            self.eq(engine.store.all()["capabilities"], {})
            self.is_none(runtime._get_pending("scout"))
            return {"message": denied["message"]}
        finally:
            tmp.cleanup()

    def concept_alias_reuse(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            runtime._handle_deterministic_presence_command("CPU usage", "scout")
            runtime._handle_deterministic_presence_command("yes", "scout")
            reused = self.require(runtime._handle_deterministic_presence_command("current CPU usage", "scout"))
            self.contains(reused["message"], "Learned command.")
            caps = engine.store.all()["capabilities"]
            self.require(caps.get("current cpu usage"), "concept alias was not recorded")
            self.eq(caps["current cpu usage"].get("alias_of"), "cpu usage")
            return {"aliases": [k for k, v in caps.items() if v.get("alias_of")]}
        finally:
            tmp.cleanup()

    def execution_review_reroutes_correction(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            runtime._handle_deterministic_presence_command("CPU usage", "scout")
            runtime._handle_deterministic_presence_command("yes", "scout")
            runtime._handle_deterministic_presence_command("show me processor load", "scout")
            correction = self.require(runtime._handle_deterministic_presence_command("no, hardware", "scout"))
            self.contains(correction["message"], "Vault CPU hardware")
            self.require("show me processor load" not in engine.store.all()["capabilities"], "bad alias was not removed")
            runtime._handle_deterministic_presence_command("yes", "scout")
            caps = engine.store.all()["capabilities"]
            self.eq(caps["show me processor load"]["inferred"], {"topic": "cpu", "aspect": "hardware"})
            return {"corrected_key": "show me processor load", "description": caps["show me processor load"]["description"]}
        finally:
            tmp.cleanup()

    def new_learned_command_bypasses_review(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            runtime = fake_runtime(engine)
            engine.store.remember("cpu usage", self.seed_cap("vault_cpu_usage", "Vault CPU usage", "cpu", "usage"))
            engine.store.remember("disk usage", self.seed_cap("vault_disk_usage", "Vault disk usage", "disk", "usage"))
            first = self.require(runtime._handle_deterministic_presence_command("cpu usage", "scout"))
            self.contains(first["message"], "Learned command.")
            self.eq(runtime._get_pending("scout")["type"], "learned_execution_review")
            second = self.require(runtime._handle_deterministic_presence_command("disk usage", "scout"))
            self.contains(second["message"], "Learned command.")
            self.eq(second["deterministic_source"], "learned_capability:vault_disk_usage")
            caps = engine.store.all()["capabilities"]
            self.require(caps.get("cpu usage"), "cpu usage cap missing")
            self.require(caps.get("disk usage"), "disk usage cap missing")
            self.eq(caps["cpu usage"].get("alias_of"), None)
            self.eq(caps["disk usage"].get("alias_of"), None)
            return {"first": first["deterministic_source"], "second": second["deterministic_source"]}
        finally:
            tmp.cleanup()

    def seed_cap(self, intent: str, description: str, topic: str, aspect: str) -> dict:
        return {
            "intent": intent,
            "description": description,
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.9,
            "inferred": {"topic": topic, "aspect": aspect},
            "execution": {"type": "bash", "command": f"echo {topic} {aspect}", "required_facts": [topic, aspect]},
            "code_monkey_task": {"ok": False, "skipped": True},
        }

    def python_recipe_materialization(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            proposal = self.require(engine.propose("What Python version are you running?"))
            result = engine.learn_and_execute("What Python version are you running?", proposal)
            self.truthy(result["ok"], result.get("error"))
            cap = result["capability"]
            self.eq(cap["execution"]["type"], "python_script")
            self.truthy(Path(cap["execution"]["path"]).exists(), "python script was not materialized")
            self.contains(result["stdout"], "Python")
            return {"path": cap["execution"]["path"], "stdout": result["stdout"].strip()}
        finally:
            tmp.cleanup()

    def dangerous_generated_recipe_rejected(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            proposal = engine._code_monkey_recipe_proposal("dangerous", "status")
            result = engine.learn_and_execute("dangerous system check", proposal)
            self.eq(result["ok"], False)
            self.contains(result["error"], "safety")
            self.eq(engine.store.all()["capabilities"], {})
            return {"error": result["error"]}
        finally:
            tmp.cleanup()

    def store_persistence_and_pending_updates(self) -> dict:
        tmp, engine = self.make_engine()
        try:
            proposal = self.require(engine.propose("disk usage"))
            result = engine.learn_and_execute("disk usage", proposal)
            self.truthy(result["saved"], "capability was not saved")
            reloaded = LearnedCapabilityStore(engine.store.path).lookup("disk usage")
            self.require(reloaded, "saved capability did not reload")
            task = {"ok": True, "task_id": "cm-task-verified", "goal": "build cap", "state": "queued"}
            engine.store.remember_pending_code_monkey("custom disk check", proposal, task)
            engine.code_monkey.statuses["cm-task-verified"] = {"task_id": "cm-task-verified", "state": "verified"}
            updates = engine.check_pending_code_monkey()
            self.eq(updates[0]["state"], "verified")
            summary = engine.summarize_pending_update(updates[0])
            self.contains(summary, "Code Monkey finished task cm-task-verified")
            return {"reloaded": reloaded["description"], "update": summary}
        finally:
            tmp.cleanup()

    def _record(self, label: str, scenario: Callable[[], Any]) -> None:
        started = time.time()
        try:
            detail = scenario()
            ok = True
        except Exception as exc:
            detail = {"error": repr(exc)}
            ok = False
        result = Result(label, ok, round(time.time() - started, 3), detail)
        self.results.append(result)
        if self.verbose or not ok:
            print(json.dumps(result.__dict__, sort_keys=True), flush=True)

    def print_summary(self) -> None:
        serializable = [result.__dict__ for result in self.results]
        print(json.dumps(serializable, indent=2, sort_keys=True))
        failed = [result.label for result in self.results if not result.ok]
        print("SUMMARY", json.dumps({"total": len(self.results), "failed": len(failed), "failed_labels": failed}, sort_keys=True))

    def require(self, value: Any, message: str = "required value missing") -> Any:
        if not value:
            raise BatteryFailure(message)
        return value

    def truthy(self, value: Any, message: str = "expected truthy value") -> None:
        if not value:
            raise BatteryFailure(message)

    def eq(self, actual: Any, expected: Any) -> None:
        if actual != expected:
            raise BatteryFailure(f"expected {expected!r}, got {actual!r}")

    def is_none(self, value: Any) -> None:
        if value is not None:
            raise BatteryFailure(f"expected None, got {value!r}")

    def contains(self, text: str, expected: str) -> None:
        if expected.casefold() not in str(text).casefold():
            raise BatteryFailure(f"expected {expected!r} in {text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated LUHKAS learned-capabilities battery.")
    parser.add_argument("--verbose", action="store_true", help="Print each scenario result as it completes.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(LearnedCapabilitiesBattery(verbose=parse_args().verbose).run())
