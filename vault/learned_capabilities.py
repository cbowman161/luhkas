from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from code_monkey_client import CodeMonkeyClient
from models import get_model
from safety_policy import SafetyPolicy


DEFAULT_STORE = Path(__file__).parent / "data" / "learned_capabilities" / "capabilities.json"

SYSTEM_TOPICS = (
    "cpu", "memory", "gpu", "disk", "uptime", "os", "hostname",
    "python", "process", "network", "service", "temperature",
)
SYSTEM_ASPECTS = ("usage", "hardware", "status", "version")

_CLASSIFIER_PROMPT = """You classify whether a user request is asking about a Linux server's runtime state.

Output strict JSON: {"topic": <topic_or_none>, "aspect": <aspect_or_none>}

Topics:
- cpu: processors, cores, threads
- memory: RAM, swap
- gpu: graphics cards
- disk: storage, filesystems, volumes
- uptime: how long the system has been up
- os: kernel, distro, OS version
- hostname: machine name
- python: Python interpreter
- process: running processes, tasks (NOT "how long it's been running")
- network: interfaces, IPs, network state
- service: systemd services
- temperature: thermals, sensors
- none: NOT a Linux-server-state question (identity, chitchat, memory recall, math, gibberish, scout/camera/robot actions)

Aspects:
- usage: live percent or current activity
- hardware: specs, capacity, models
- status: current state, what's running
- version: which version is installed
- none

Examples:
INPUT: "cpu usage"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

INPUT: "what processes are running"
OUTPUT: {"topic": "process", "aspect": "status"}

INPUT: "how long has the box been up"
OUTPUT: {"topic": "uptime", "aspect": "status"}

INPUT: "how much ram is installed"
OUTPUT: {"topic": "memory", "aspect": "hardware"}

INPUT: "tell me a joke"
OUTPUT: {"topic": "none", "aspect": "none"}

INPUT: "what is my favorite color"
OUTPUT: {"topic": "none", "aspect": "none"}

INPUT: "remember that my code is 4321"
OUTPUT: {"topic": "none", "aspect": "none"}

INPUT: "what's eating my disk"
OUTPUT: {"topic": "disk", "aspect": "usage"}

INPUT: "what's my current CPU usage?"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

INPUT: "what's my CPU?"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

INPUT: "show me processor load"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

INPUT: "how busy is the box right now"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

INPUT: "what cpu does this machine have"
OUTPUT: {"topic": "cpu", "aspect": "hardware"}

INPUT: "tell me about the cpu model"
OUTPUT: {"topic": "cpu", "aspect": "hardware"}

INPUT: "look at me"
OUTPUT: {"topic": "none", "aspect": "none"}

Now classify:
INPUT: %s
OUTPUT:"""


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text or "").casefold()).strip()


class LearnedCapabilityStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_STORE)

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "capabilities": {}, "pending_code_monkey": {}}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def lookup(self, text: str) -> dict | None:
        key = normalize_text(text)
        if not key:
            return None
        data = self.load()
        cap = (data.get("capabilities") or {}).get(key)
        if not isinstance(cap, dict):
            return None
        cap["hits"] = int(cap.get("hits") or 0) + 1
        cap["last_hit_at"] = time.time()
        data["capabilities"][key] = cap
        self.save(data)
        return dict(cap)

    def remember(self, text: str, capability: dict) -> dict:
        key = normalize_text(text)
        if not key:
            raise ValueError("Cannot save a learned capability for empty input.")
        data = self.load()
        caps = data.setdefault("capabilities", {})
        existing = caps.get(key) if isinstance(caps.get(key), dict) else {}
        examples = existing.get("examples") if isinstance(existing.get("examples"), list) else []
        example = {
            "input": str(text or ""),
            "normalized_input": key,
            "confirmed_at": time.time(),
            "confidence": capability.get("confidence"),
            "route": capability.get("route"),
            "intent": capability.get("intent"),
        }
        examples.append(example)
        saved = {
            **capability,
            "input": str(text or ""),
            "normalized_input": key,
            "confirmed": True,
            "created_at": existing.get("created_at") or time.time(),
            "updated_at": time.time(),
            "hits": int(existing.get("hits") or 0),
            "examples": examples[-20:],
        }
        caps[key] = saved
        self.save(data)
        return dict(saved)

    def remember_alias(self, new_text: str, source_cap: dict) -> dict | None:
        """Save a new phrasing as an alias entry that reuses an existing capability's
        execution recipe. Returns the stored capability dict, or None if invalid."""
        key = normalize_text(new_text)
        if not key:
            return None
        data = self.load()
        caps = data.setdefault("capabilities", {})
        existing = caps.get(key) if isinstance(caps.get(key), dict) else None
        examples = list((existing or {}).get("examples") or [])
        examples.append({
            "input": str(new_text),
            "normalized_input": key,
            "confirmed_at": time.time(),
            "source": "concept_match",
            "intent": source_cap.get("intent"),
            "confidence": source_cap.get("confidence"),
        })
        saved = {
            **(existing or source_cap),
            "name": source_cap.get("name") or source_cap.get("intent"),
            "intent": source_cap.get("intent"),
            "description": source_cap.get("description"),
            "route": source_cap.get("route"),
            "target": source_cap.get("target"),
            "confidence": source_cap.get("confidence"),
            "reason": source_cap.get("reason"),
            "inferred": source_cap.get("inferred"),
            "execution": source_cap.get("execution"),
            "response": source_cap.get("response"),
            "code_monkey_task": source_cap.get("code_monkey_task"),
            "confirmed_by": "concept_match",
            "input": str(new_text),
            "normalized_input": key,
            "confirmed": True,
            "alias_of": source_cap.get("normalized_input"),
            "created_at": (existing or {}).get("created_at") or time.time(),
            "updated_at": time.time(),
            "hits": int((existing or {}).get("hits") or 0),
            "examples": examples[-20:],
        }
        caps[key] = saved
        self.save(data)
        return dict(saved)

    def forget(self, text: str) -> bool:
        key = normalize_text(text)
        if not key:
            return False
        data = self.load()
        caps = data.setdefault("capabilities", {})
        existed = key in caps
        if existed:
            del caps[key]
            self.save(data)
        return existed

    def remember_pending_code_monkey(self, text: str, proposal: dict, task: dict) -> dict:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            return {}
        data = self.load()
        pending = data.setdefault("pending_code_monkey", {})
        entry = {
            "task_id": task_id,
            "input": str(text or ""),
            "normalized_input": normalize_text(text),
            "proposal": proposal,
            "goal": task.get("goal"),
            "state": task.get("state") or "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "notified": False,
        }
        pending[task_id] = entry
        self.save(data)
        return dict(entry)

    def pending_code_monkey(self) -> dict:
        data = self.load()
        pending = data.get("pending_code_monkey")
        return dict(pending) if isinstance(pending, dict) else {}

    def update_pending_code_monkey(self, task_id: str, updates: dict) -> dict | None:
        data = self.load()
        pending = data.setdefault("pending_code_monkey", {})
        entry = pending.get(task_id)
        if not isinstance(entry, dict):
            return None
        entry.update(updates)
        entry["updated_at"] = time.time()
        pending[task_id] = entry
        self.save(data)
        return dict(entry)

    def all(self) -> dict:
        return self.load()


class LearnedCapabilityEngine:
    """Confirm, execute, and persist safe non-Scout deterministic recipes."""

    def __init__(self, store: LearnedCapabilityStore | None = None, code_monkey_client=None, model=None):
        self.store = store or LearnedCapabilityStore()
        self.safety = SafetyPolicy()
        self.scripts_dir = self.store.path.parent / "scripts"
        self.code_monkey = code_monkey_client if code_monkey_client is not None else CodeMonkeyClient(timeout=3)
        self.model = model if model is not None else get_model("router")
        self._inference_cache: dict = {}

    def _infer_topic_and_aspect(self, text: str) -> tuple[str | None, str | None]:
        """Classify a request semantically into (topic, aspect) using the router LLM.

        Memoized per normalized input so each unique phrasing only hits the LLM once.
        Returns (None, None) when the input is not a system-state request.
        """
        key = normalize_text(text)
        if not key:
            return None, None
        if key in self._inference_cache:
            return self._inference_cache[key]
        result = self._llm_classify(text)
        self._inference_cache[key] = result
        return result

    def _llm_classify(self, text: str) -> tuple[str | None, str | None]:
        if self.model is None:
            return None, None
        prompt = _CLASSIFIER_PROMPT % json.dumps(str(text or ""))
        try:
            raw = self.model.generate(prompt, think=False, timeout=10)
        except Exception:
            return None, None
        parsed = self._parse_json_object(raw)
        if not isinstance(parsed, dict):
            return None, None
        topic = parsed.get("topic")
        aspect = parsed.get("aspect")
        if isinstance(topic, str):
            topic = topic.strip().lower() or None
        else:
            topic = None
        if isinstance(aspect, str):
            aspect = aspect.strip().lower() or None
        else:
            aspect = None
        if topic == "none":
            topic = None
        if aspect == "none":
            aspect = None
        if topic and topic not in SYSTEM_TOPICS:
            topic = None
        if aspect and aspect not in SYSTEM_ASPECTS:
            aspect = None
        if topic is None:
            return None, None
        return topic, aspect

    @staticmethod
    def _parse_json_object(raw: str) -> dict | None:
        if not raw:
            return None
        text = raw.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def is_scout_specific(self, text: str) -> bool:
        normalized = normalize_text(text)
        scout_terms = {
            "scout", "camera", "vision", "tracking", "follow", "face", "recognize",
            "light", "lamp", "wheel", "wheels", "drive", "move", "turn", "pan",
            "tilt", "look", "guard",
        }
        return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in scout_terms)

    def lookup(self, text: str) -> dict | None:
        return self.store.lookup(text)

    def lookup_by_concept(self, text: str) -> dict | None:
        """Find an existing capability whose inferred (topic, aspect) matches the
        LLM-classified concept of *text*. Falls back to parsing the intent name
        (vault_<topic>_<aspect>) for legacy capabilities that pre-date LLM
        classification. Returns None if no inference or no match."""
        topic, aspect = self._infer_topic_and_aspect(text)
        if topic is None:
            return None
        data = self.store.load()
        caps = (data.get("capabilities") or {}).values()
        topic_aspect_matches = []
        topic_only_matches = []
        for cap in caps:
            if not isinstance(cap, dict):
                continue
            if (cap.get("execution") or {}).get("type") not in {"bash", "python_script"}:
                continue
            cap_topic, cap_aspect = self._cap_concept(cap)
            if cap_topic != topic:
                continue
            if cap_aspect == aspect:
                topic_aspect_matches.append(cap)
            else:
                topic_only_matches.append(cap)
        candidates = topic_aspect_matches or topic_only_matches
        if not candidates:
            return None
        candidates.sort(
            key=lambda c: (int(c.get("hits") or 0), float(c.get("updated_at") or 0)),
            reverse=True,
        )
        return dict(candidates[0])

    @staticmethod
    def _cap_concept(cap: dict) -> tuple[str | None, str | None]:
        inferred = cap.get("inferred") or {}
        topic = inferred.get("topic")
        aspect = inferred.get("aspect")
        if topic and topic in SYSTEM_TOPICS:
            return topic, (aspect if aspect in SYSTEM_ASPECTS else None)
        intent = str(cap.get("intent") or "")
        match = re.match(r"vault_([a-z]+)(?:_([a-z]+))?$", intent)
        if not match:
            return None, None
        parsed_topic = match.group(1)
        parsed_aspect = match.group(2)
        if parsed_topic not in SYSTEM_TOPICS:
            # Allow common single-word legacy intents that ARE a topic on their own.
            legacy_topic_only = {"uptime", "hostname"}
            if parsed_topic in legacy_topic_only:
                return parsed_topic, "status"
            return None, None
        if parsed_aspect and parsed_aspect not in SYSTEM_ASPECTS:
            parsed_aspect = None
        return parsed_topic, parsed_aspect

    def record_alias(self, new_text: str, source_cap: dict) -> dict | None:
        return self.store.remember_alias(new_text, source_cap)

    def propose(self, text: str) -> dict | None:
        if self.is_scout_specific(text):
            return None
        normalized = normalize_text(text)
        if not normalized:
            return None
        return self._propose_code_monkey_recipe(normalized)

    def propose_correction(self, correction: str, previous_proposal: dict | None = None) -> dict | None:
        new_topic, new_aspect = self._infer_topic_and_aspect(correction)
        previous_inferred = (previous_proposal or {}).get("inferred") or {}
        previous_topic = previous_inferred.get("topic")
        if new_topic is None:
            if previous_topic is None:
                return None
            return self._code_monkey_recipe_proposal(previous_topic, new_aspect or previous_inferred.get("aspect") or "usage")
        safe = self.safety.classify_capability_request(normalize_text(correction))
        if not safe.get("allowed"):
            return None
        return self._code_monkey_recipe_proposal(new_topic, new_aspect or self._default_aspect_for(new_topic))

    def correction_updates_previous_request(self, proposal: dict, previous_proposal: dict | None = None) -> bool:
        inferred = proposal.get("inferred") or {}
        previous_inferred = (previous_proposal or {}).get("inferred") or {}
        return bool(
            inferred.get("topic")
            and previous_inferred.get("topic")
            and inferred.get("topic") == previous_inferred.get("topic")
        )

    def _propose_code_monkey_recipe(self, normalized: str) -> dict | None:
        topic, aspect = self._infer_topic_and_aspect(normalized)
        if topic is None:
            return None
        safe = self.safety.classify_capability_request(normalized)
        if not safe.get("allowed"):
            return None
        return self._code_monkey_recipe_proposal(topic, aspect or self._default_aspect_for(topic))

    def _default_aspect_for(self, topic: str | None) -> str:
        if topic in {"python", "os"}:
            return "version"
        if topic in {"hostname", "network", "service", "process", "temperature", "uptime"}:
            return "status"
        return "usage"

    def _code_monkey_recipe_proposal(self, topic: str, aspect: str) -> dict:
        description = self._describe_inferred_system_info(topic, aspect)
        confidence = 0.86 if topic else 0.72
        return {
            "ok": True,
            "intent": "vault_learned_command",
            "description": description,
            "route": "learned_capability",
            "target": "vault",
            "confidence": confidence,
            "reason": "Code Monkey will generate and validate a safe single-command learned recipe.",
            "planner": "code_monkey_single_recipe",
            "inferred": {
                "topic": topic,
                "aspect": aspect,
            },
            "queue_code_monkey": False,
        }

    def _describe_inferred_system_info(self, topic: str, aspect: str) -> str:
        topic_names = {
            "cpu": "CPU",
            "memory": "RAM" if aspect == "hardware" else "memory",
            "gpu": "GPU",
            "disk": "disk",
            "uptime": "uptime",
            "os": "operating system",
            "hostname": "hostname",
            "python": "Python runtime",
            "process": "process",
            "network": "network",
            "service": "service",
            "temperature": "temperature",
            "system": "system",
        }
        aspect_names = {
            "hardware": "hardware",
            "usage": "usage",
            "status": "status",
            "version": "version",
        }
        return f"Vault {topic_names.get(topic, topic)} {aspect_names.get(aspect, aspect)}"

    def build_recipe(self, text: str, proposal: dict) -> dict:
        raise ValueError(
            "Local learned-command recipe templates are disabled. "
            "Use Code Monkey's /learned-command-recipe endpoint."
        )

    def learn_and_execute(self, text: str, proposal: dict, *, confirmed_by: str = "user_confirmation") -> dict:
        if proposal.get("queue_code_monkey"):
            code_monkey_task = self.request_code_monkey_recipe(text, proposal)
            result = {
                "ok": bool(code_monkey_task.get("ok")),
                "stdout": "",
                "stderr": "",
                "returncode": 0 if code_monkey_task.get("ok") else -1,
                "error": code_monkey_task.get("error"),
                "ran_at": time.time(),
                "code_monkey_task": code_monkey_task,
                "capability": {
                    "name": proposal.get("intent"),
                    "intent": proposal.get("intent"),
                    "description": proposal.get("description"),
                    "route": proposal.get("route"),
                    "target": proposal.get("target"),
                    "confidence": proposal.get("confidence"),
                    "reason": proposal.get("reason"),
                    "inferred": proposal.get("inferred"),
                    "confirmed_by": confirmed_by,
                    "code_monkey_task": code_monkey_task,
                    "execution": {"type": "code_monkey_pending"},
                },
                "saved": False,
                "learning_queued": bool(code_monkey_task.get("ok")),
            }
            return result
        if proposal.get("planner") == "code_monkey_single_recipe":
            recipe_result = self.generate_single_command_recipe(text, proposal)
            if not recipe_result.get("ok"):
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": recipe_result.get("error") or "Code Monkey could not generate a learned command recipe.",
                    "returncode": -1,
                    "error": recipe_result.get("error"),
                    "ran_at": time.time(),
                    "code_monkey_task": {"ok": False, "single_recipe": True, "error": recipe_result.get("error")},
                    "capability": {
                        "name": proposal.get("intent"),
                        "intent": proposal.get("intent"),
                        "description": proposal.get("description"),
                        "route": proposal.get("route"),
                        "target": proposal.get("target"),
                        "confidence": proposal.get("confidence"),
                        "reason": proposal.get("reason"),
                        "inferred": proposal.get("inferred"),
                        "confirmed_by": confirmed_by,
                    },
                    "saved": False,
                }
            code_monkey_task = {
                "ok": True,
                "single_recipe": True,
                "generator": recipe_result.get("generator") or "code_monkey_single_recipe",
            }
            recipe = self._materialize_generated_recipe(text, recipe_result.get("recipe") or {})
        else:
            code_monkey_task = {"ok": False, "skipped": True, "reason": "legacy_builtin_recipe"}
            recipe = self.build_recipe(text, proposal)
        capability = {
            "name": proposal.get("intent"),
            "intent": proposal.get("intent"),
            "description": proposal.get("description"),
            "route": proposal.get("route"),
            "target": proposal.get("target"),
            "confidence": proposal.get("confidence"),
            "reason": proposal.get("reason"),
            "inferred": proposal.get("inferred"),
            "confirmed_by": confirmed_by,
            "code_monkey_task": code_monkey_task,
            "execution": recipe,
            "response": {
                "type": "compose_from_output",
                "required_facts": recipe.get("required_facts") or [],
            },
        }
        result = self.execute_capability(capability)
        capability["last_result"] = {
            "ok": result.get("ok"),
            "returncode": result.get("returncode"),
            "ran_at": result.get("ran_at"),
            "error": result.get("error"),
        }
        saved = self.store.remember(text, capability) if result.get("ok") else None
        result["capability"] = saved or capability
        result["saved"] = bool(saved)
        return result

    def generate_single_command_recipe(self, text: str, proposal: dict) -> dict:
        if self.code_monkey is None or not hasattr(self.code_monkey, "generate_learned_command_recipe"):
            return {"ok": False, "error": "code_monkey_single_recipe_endpoint_not_configured"}
        try:
            return self.code_monkey.generate_learned_command_recipe(text, proposal)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _materialize_generated_recipe(self, text: str, recipe: dict) -> dict:
        kind = recipe.get("type")
        if kind == "bash":
            command = str(recipe.get("command") or "").strip()
            return self._bash_recipe(command, recipe.get("required_facts") or [])
        if kind == "python_script":
            source = str(recipe.get("source") or "")
            filename = str(recipe.get("filename") or normalize_text(text) or "learned_command")
            if not filename.endswith(".py"):
                filename += ".py"
            return self._python_recipe(filename, source, recipe.get("required_facts") or [])
        raise ValueError(f"Unsupported generated recipe type: {kind}")

    def request_code_monkey_recipe(self, text: str, proposal: dict) -> dict:
        if self.code_monkey is None:
            return {"ok": False, "error": "code_monkey_client_not_configured"}
        goal = self._code_monkey_goal(text, proposal)
        try:
            task_id = self.code_monkey.start_task(goal)
            task = {
                "ok": True,
                "task_id": task_id,
                "goal": goal,
                "state": "queued",
            }
            self.store.remember_pending_code_monkey(text, proposal, task)
            return task
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "goal": goal,
            }

    def check_pending_code_monkey(self) -> list[dict]:
        if self.code_monkey is None:
            return []
        updates = []
        final_states = {"verified", "build_failed", "test_failed", "failed", "cancelled"}
        for task_id, entry in self.store.pending_code_monkey().items():
            if entry.get("notified"):
                continue
            try:
                status = self.code_monkey.safe_status(task_id)
            except Exception as exc:
                status = {"task_id": task_id, "error": str(exc)}
            state = status.get("state") or ("error" if status.get("error") else entry.get("state"))
            self.store.update_pending_code_monkey(task_id, {"state": state, "last_status": status})
            if state not in final_states:
                continue
            artifacts = {}
            if state == "verified" and hasattr(self.code_monkey, "get_artifacts"):
                try:
                    artifacts = self.code_monkey.get_artifacts(task_id)
                except Exception as exc:
                    artifacts = {"error": str(exc)}
            learned_removed = False
            if state != "verified":
                learned_removed = self.store.forget(entry.get("input") or "")
            updated = self.store.update_pending_code_monkey(
                task_id,
                {
                    "notified": True,
                    "completed_at": time.time(),
                    "state": state,
                    "artifacts": artifacts,
                    "learned_removed": learned_removed,
                },
            )
            updates.append(updated or {**entry, "state": state, "artifacts": artifacts})
        return updates

    def execute_capability(self, capability: dict) -> dict:
        execution = capability.get("execution") or {}
        kind = execution.get("type")
        if kind == "bash":
            return self._run_bash(execution.get("command") or "")
        if kind == "python_script":
            return self._run_python_script(execution)
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "error": f"Unsupported learned execution type: {kind}",
            "ran_at": time.time(),
        }

    def summarize_result(self, text: str, capability: dict, result: dict) -> str:
        task = capability.get("code_monkey_task") or result.get("code_monkey_task") or {}
        task_note = ""
        if self._is_builtin_simple_capability(capability):
            task_note = ""
        elif task.get("ok") and task.get("task_id"):
            task_note = f"I queued Code Monkey task {task.get('task_id')} to build the polished deterministic path. "
        elif task.get("error"):
            task_note = f"Code Monkey did not accept the work order: {task.get('error')}. "
        if not result.get("ok"):
            return f"{task_note}I could not run the immediate learned path for {capability.get('description')}: {result.get('error') or result.get('stderr')}"
        if result.get("learning_queued"):
            task_id = task.get("task_id")
            if task_id:
                return f"I am learning this command as {capability.get('description')}. Code Monkey task {task_id} is building the deterministic path."
            return f"I am learning this command as {capability.get('description')}."
        output = (result.get("stdout") or "").strip()
        if not output:
            return f"{task_note}The immediate learned path for {capability.get('description')} ran, but returned no output."
        parsed = self._parsed_summary(capability, output)
        if task_note:
            return f"{task_note}Current result: {parsed}"
        return parsed

    def _is_builtin_simple_capability(self, capability: dict) -> bool:
        if (capability.get("code_monkey_task") or {}).get("skipped"):
            return True
        intent = capability.get("intent") or capability.get("name")
        builtin_intents = {
            "vault_cpu_status",
            "vault_cpu_usage",
            "vault_cpu_hardware",
            "vault_memory_status",
            "vault_disk_status",
            "vault_uptime",
            "vault_os_status",
            "vault_hostname",
            "vault_python_version",
        }
        return intent in builtin_intents and (capability.get("execution") or {}).get("type") in {"bash", "python_script"}

    def summarize_pending_update(self, update: dict) -> str:
        proposal = update.get("proposal") or {}
        task_id = update.get("task_id")
        state = update.get("state") or "unknown"
        description = proposal.get("description") or update.get("input") or "that learned command"
        if state == "verified":
            return f"Code Monkey finished task {task_id} for {description}. The generated capability is ready for review."
        return f"Learning failed for {description}. Code Monkey task {task_id} ended with state {state}, so no learned command was saved."

    def _parsed_summary(self, capability: dict, output: str) -> str:
        intent = capability.get("intent")
        if intent == "vault_cpu_usage":
            match = re.search(r"CPU usage percent:\s*([0-9]+(?:\.[0-9]+)?)", output)
            if match:
                return f"Vault CPU usage: {match.group(1)}%."
        if intent in {"vault_cpu_hardware", "vault_cpu_status"}:
            facts = self._key_value_lines(output)
            model = facts.get("Model name")
            cpus = facts.get("CPU(s)")
            threads = facts.get("Thread(s) per core")
            cores = facts.get("Core(s) per socket")
            bits = []
            if model:
                bits.append(model)
            if cpus:
                bits.append(f"{cpus} logical CPUs")
            if cores:
                bits.append(f"{cores} cores per socket")
            if threads:
                bits.append(f"{threads} threads per core")
            if bits:
                return "Vault CPU hardware: " + ", ".join(bits) + "."
        if intent == "vault_memory_status":
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return "Vault memory usage: " + " | ".join(lines[:2])
        if intent == "vault_system_info":
            inferred = capability.get("inferred") or {}
            if inferred.get("topic") == "memory" and inferred.get("aspect") == "hardware":
                return f"Vault RAM hardware: {output}"
            if inferred.get("topic") == "memory":
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                return "Vault memory usage: " + " | ".join(lines[:2])
            if inferred.get("topic") == "gpu" and inferred.get("aspect") == "hardware":
                return f"Vault GPU hardware: {output}"
            if inferred.get("topic") == "gpu":
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                if lines:
                    return "Vault GPU status: " + " | ".join(lines[:3])
        if intent == "vault_disk_status":
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return "Vault disk usage: " + " | ".join(lines[:2])
        return f"{capability.get('description')}: {output}"

    def _key_value_lines(self, output: str) -> dict:
        facts = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = re.sub(r"\s+", " ", key).strip()
            value = re.sub(r"\s+", " ", value).strip()
            if key and value:
                facts[key] = value
        return facts

    def _bash_recipe(self, command: str, required_facts: list[str]) -> dict:
        safety = self.safety.validate_command(command)
        if not safety.get("allowed"):
            raise ValueError(safety.get("reason") or "Command failed safety validation.")
        return {
            "type": "bash",
            "command": command,
            "required_facts": required_facts,
            "timeout_seconds": 10,
            "safety": safety,
        }

    def _code_monkey_goal(self, text: str, proposal: dict) -> str:
        return (
            "Generate a deterministic learned-command recipe for Luhkas Vault.\n\n"
            "The recipe must answer a confirmed user request by running safe local commands "
            "or a small Python script. It must be read-only, non-destructive, fast, and "
            "suitable for saving as a deterministic path.\n\n"
            f"Confirmed user input: {text}\n"
            f"Intent: {proposal.get('intent')}\n"
            f"Description: {proposal.get('description')}\n"
            f"Target: {proposal.get('target')}\n"
            f"Confidence: {proposal.get('confidence')}\n\n"
            "Return an API-first capability whose public function executes the recipe and "
            "returns a dict with ok, action, message, data, and error. Include commands.json "
            "triggers using the confirmed input and close variants. Do not use network calls, "
            "destructive commands, background processes, sudo, package installs, or writes "
            "outside the capability data directory."
        )

    def _python_recipe(self, filename: str, source: str, required_facts: list[str]) -> dict:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        path = self.scripts_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
        path.write_text(source, encoding="utf-8")
        return {
            "type": "python_script",
            "path": str(path),
            "source": source,
            "required_facts": required_facts,
            "timeout_seconds": 10,
        }

    def _run_bash(self, command: str) -> dict:
        command = str(command or "").strip()
        safety = self.safety.validate_command(command)
        if not safety.get("allowed"):
            return self._error(safety.get("reason") or "Command failed safety validation.")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return self._error(str(exc))
        if not argv:
            return self._error("Empty command.")
        if not shutil.which(argv[0]):
            return self._error(f"Command not found: {argv[0]}", returncode=127)
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return {
                "ok": completed.returncode == 0,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "returncode": completed.returncode,
                "error": completed.stderr.strip() if completed.returncode else None,
                "ran_at": time.time(),
            }
        except subprocess.TimeoutExpired:
            return self._error("Command timed out.")
        except Exception as exc:
            return self._error(str(exc))

    def _run_python_script(self, execution: dict) -> dict:
        path = Path(execution.get("path") or "")
        try:
            resolved = path.resolve()
            root = self.scripts_dir.resolve()
            if root not in resolved.parents and resolved != root:
                return self._error("Python script path is outside the learned scripts directory.")
            if not resolved.exists():
                return self._error(f"Python script not found: {resolved}")
            completed = subprocess.run(
                ["python3", str(resolved)],
                capture_output=True,
                text=True,
                timeout=int(execution.get("timeout_seconds") or 10),
                check=False,
            )
            return {
                "ok": completed.returncode == 0,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "returncode": completed.returncode,
                "error": completed.stderr.strip() if completed.returncode else None,
                "ran_at": time.time(),
            }
        except subprocess.TimeoutExpired:
            return self._error("Python script timed out.")
        except Exception as exc:
            return self._error(str(exc))

    def _error(self, message: str, returncode: int = -1) -> dict:
        return {
            "ok": False,
            "stdout": "",
            "stderr": message,
            "returncode": returncode,
            "error": message,
            "ran_at": time.time(),
        }
