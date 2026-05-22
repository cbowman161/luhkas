from __future__ import annotations

import json
import os
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


_CORRECTION_PROMPT = """You classify a user CORRECTION to a previous request about a Linux server's runtime state.

Output strict JSON: {"topic": <topic_noun_or_none>, "aspect": <aspect_noun_or_none>}

- "topic" is the singular lowercase noun for what the user is asking about
  (e.g. cpu, memory, disk, temperature, process, port, user, log, package,
  time, kernel, drive, fan — anything that fits). Use "none" if the
  correction provides no new topic and you cannot tell from context.
- "aspect" is the singular lowercase noun for what about that topic
  (e.g. usage, hardware, status, version, list, count, recent). Use "none"
  if the correction provides no new aspect.

Use the previous topic/aspect as defaults when the correction is partial:
- aspect-only correction ("the hardware") → keep the previous topic
- topic-only correction ("actually disk") → keep the previous aspect
- both specified → use both from the correction
- empty / just "no" → output "none"/"none"

Examples:
PREVIOUS topic=cpu aspect=usage
ORIGINAL: "what is your CPU"
CORRECTION: "no, the hardware"
OUTPUT: {"topic": "cpu", "aspect": "hardware"}

PREVIOUS topic=cpu aspect=usage
ORIGINAL: "tell me about the cpu"
CORRECTION: "actually disk"
OUTPUT: {"topic": "disk", "aspect": "usage"}

PREVIOUS topic=memory aspect=usage
ORIGINAL: "ram free"
CORRECTION: "no I meant the processor"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

PREVIOUS topic=cpu aspect=hardware
ORIGINAL: "what's my cpu"
CORRECTION: "no, current usage"
OUTPUT: {"topic": "cpu", "aspect": "usage"}

PREVIOUS topic=disk aspect=usage
ORIGINAL: "disk space"
CORRECTION: "no, by user"
OUTPUT: {"topic": "user", "aspect": "usage"}

PREVIOUS topic=disk aspect=usage
ORIGINAL: "disk space"
CORRECTION: "no"
OUTPUT: {"topic": "none", "aspect": "none"}

Now classify:
PREVIOUS topic=%(prev_topic)s aspect=%(prev_aspect)s
ORIGINAL: %(original)s
CORRECTION: %(correction)s
OUTPUT:"""


_CLASSIFIER_PROMPT = """You classify whether a user request is asking about a Linux server's runtime state, hardware, or configuration.

Output strict JSON: {"topic": <topic_noun_or_none>, "aspect": <aspect_noun_or_none>}

- "topic" is a SINGULAR LOWERCASE one-word noun for what the user is asking
  about. Use the most natural common noun. Some examples (not an exhaustive
  list): cpu, memory, gpu, disk, uptime, os, kernel, hostname, python,
  process, network, port, service, temperature, fan, user, login, log,
  package, time, timezone, locale, drive, partition, mount, volume, ip,
  route, dns, firewall, battery, bluetooth, wifi, audio, usb, pci, sensor,
  cgroup, namespace, swap, gpu, nvidia, vault, scout. Use "none" if the
  user is NOT asking about a Linux server's state (greetings, identity,
  chitchat, math, gibberish, scout/camera/robot actions, chat-context
  memory recall).

- "aspect" is a SINGULAR LOWERCASE one-word noun for what they want about
  the topic. Common ones: usage (live activity/percent), hardware
  (specs/capacity/model), status (current state, what's running), version
  (which version installed), list (enumeration of items), count (how many),
  recent (latest events). Use "none" only if they truly didn't specify.

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

INPUT: "no, the hardware"
OUTPUT: {"topic": "none", "aspect": "hardware"}

INPUT: "actually I meant the version"
OUTPUT: {"topic": "none", "aspect": "version"}

INPUT: "no the usage"
OUTPUT: {"topic": "none", "aspect": "usage"}

INPUT: "what's the cpu temperature"
OUTPUT: {"topic": "temperature", "aspect": "status"}

INPUT: "list my disks"
OUTPUT: {"topic": "disk", "aspect": "list"}

INPUT: "what time is it"
OUTPUT: {"topic": "time", "aspect": "status"}

INPUT: "who's logged in"
OUTPUT: {"topic": "user", "aspect": "list"}

INPUT: "any errors in the journal"
OUTPUT: {"topic": "log", "aspect": "recent"}

INPUT: "what ports are open"
OUTPUT: {"topic": "port", "aspect": "list"}

INPUT: "is ollama running"
OUTPUT: {"topic": "service", "aspect": "status"}

INPUT: "any failed services"
OUTPUT: {"topic": "service", "aspect": "status"}

INPUT: "what timezone are we in"
OUTPUT: {"topic": "timezone", "aspect": "status"}

INPUT: "how many processes are running"
OUTPUT: {"topic": "process", "aspect": "count"}

INPUT: "what's eating cpu"
OUTPUT: {"topic": "process", "aspect": "usage"}

INPUT: "show fan speeds"
OUTPUT: {"topic": "fan", "aspect": "status"}

INPUT: "is the firewall on"
OUTPUT: {"topic": "firewall", "aspect": "status"}

INPUT: "what bluetooth devices are paired"
OUTPUT: {"topic": "bluetooth", "aspect": "list"}

INPUT: "is bluetooth on"
OUTPUT: {"topic": "bluetooth", "aspect": "status"}

INPUT: "what wifi am I connected to"
OUTPUT: {"topic": "wifi", "aspect": "status"}

INPUT: "show audio devices"
OUTPUT: {"topic": "audio", "aspect": "list"}

INPUT: "what's my battery percentage"
OUTPUT: {"topic": "battery", "aspect": "usage"}

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
        # Defensive: refuse to overwrite a populated file with an empty one.
        # The whole learned-capabilities feature depends on this file, and a
        # bug elsewhere (e.g. a load failure followed by a save) could silently
        # nuke every confirmed cap.
        new_caps = (data.get("capabilities") or {}) if isinstance(data, dict) else {}
        if not new_caps and self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                existing_caps = (existing.get("capabilities") or {}) if isinstance(existing, dict) else {}
            except Exception:
                existing_caps = {}
            if existing_caps:
                import sys
                print(
                    f"[learned_capabilities] REFUSING to save: would replace "
                    f"{len(existing_caps)} caps with 0. Caller bug — investigate.",
                    file=sys.stderr,
                    flush=True,
                )
                return
        # Keep a rolling backup of the previous good state.
        if self.path.exists():
            try:
                bak = self.path.with_suffix(".bak")
                bak.write_bytes(self.path.read_bytes())
            except Exception:
                pass
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
        # Recipe generation can run the planner LLM multiple times (initial
        # attempt + retries) and uses format-constrained decoding, which is
        # slower than plain generation. The default 3s timeout was too tight
        # and aborted legitimate calls mid-retry. 60s is generous but bounded.
        self.code_monkey = code_monkey_client if code_monkey_client is not None else CodeMonkeyClient(timeout=60)
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

    def classify_correction(
        self,
        correction: str,
        original: str = "",
        previous_topic: str | None = None,
        previous_aspect: str | None = None,
    ) -> tuple[str | None, str | None]:
        """LLM classification for a CORRECTION turn, with full prior context.

        Unlike `_infer_topic_and_aspect`, this is not memoized: the result
        depends on the previous proposal/cap, so each correction is fresh."""
        if self.model is None:
            return None, None
        prompt = _CORRECTION_PROMPT % {
            "prev_topic": previous_topic or "none",
            "prev_aspect": previous_aspect or "none",
            "original": json.dumps(str(original or "")),
            "correction": json.dumps(str(correction or "")),
        }
        try:
            raw = self.model.generate(prompt, think=False, timeout=10)
        except Exception:
            return None, None
        return self._parse_classification(raw)

    def _llm_classify(self, text: str) -> tuple[str | None, str | None]:
        if self.model is None:
            return None, None
        prompt = _CLASSIFIER_PROMPT % json.dumps(str(text or ""))
        try:
            raw = self.model.generate(prompt, think=False, timeout=10)
        except Exception:
            return None, None
        topic, aspect = self._parse_classification(raw)
        # Plain classify path requires a topic; aspect-only doesn't make sense
        # for a fresh request.
        if topic is None:
            return None, None
        return topic, aspect

    @staticmethod
    def _parse_classification(raw: str) -> tuple[str | None, str | None]:
        parsed = LearnedCapabilityEngine._parse_json_object(raw)
        if not isinstance(parsed, dict):
            return None, None
        topic = LearnedCapabilityEngine._clean_noun(parsed.get("topic"))
        aspect = LearnedCapabilityEngine._clean_noun(parsed.get("aspect"))
        return topic, aspect

    @staticmethod
    def _clean_noun(value) -> str | None:
        """Normalize an LLM-returned topic/aspect noun: lowercase singular
        single token. Returns None for 'none', empty, multi-word, or
        non-string values."""
        if not isinstance(value, str):
            return None
        token = value.strip().lower()
        if not token or token == "none" or token == "null":
            return None
        # Require a single short alphabetic word — guards against the LLM
        # returning a sentence or hallucinated structure.
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", token):
            return None
        return token

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

    def same_topic_caps(self, topic: str) -> list[dict]:
        """All caps stored under the given topic, ranked by hits then recency.
        Used to surface alternatives when the user proposes a *new* aspect
        for a topic we already have caps for — lets them pick the existing
        one instead of fragmenting the topic into near-duplicates."""
        if not topic:
            return []
        data = self.store.load()
        caps = (data.get("capabilities") or {}).values()
        matches = []
        for cap in caps:
            if not isinstance(cap, dict):
                continue
            if (cap.get("execution") or {}).get("type") not in {"bash", "python_script"}:
                continue
            cap_topic, _ = self._cap_concept(cap)
            if cap_topic != topic:
                continue
            matches.append(cap)
        matches.sort(
            key=lambda c: (int(c.get("hits") or 0), float(c.get("updated_at") or 0)),
            reverse=True,
        )
        return matches

    def lookup_by_concept(self, text: str) -> dict | None:
        """Find an existing capability whose inferred (topic, aspect) matches the
        LLM-classified concept of *text*. Falls back to parsing the intent name
        (vault_<topic>_<aspect>) for legacy capabilities that pre-date LLM
        classification.

        Matching rules:
        - If the LLM gave a specific aspect, ONLY caps with that exact aspect
          qualify. We never silently substitute a different aspect (e.g. usage
          for hardware), because that gives the user the wrong answer when they
          explicitly asked for a particular facet.
        - If the LLM left aspect unset, any cap on the same topic is a
          candidate, ranked by hits.

        Returns None when no candidate matches.
        """
        topic, aspect = self._infer_topic_and_aspect(text)
        if topic is None:
            return None
        data = self.store.load()
        caps = (data.get("capabilities") or {}).values()
        candidates = []
        for cap in caps:
            if not isinstance(cap, dict):
                continue
            if (cap.get("execution") or {}).get("type") not in {"bash", "python_script"}:
                continue
            cap_topic, cap_aspect = self._cap_concept(cap)
            if cap_topic != topic:
                continue
            if aspect and cap_aspect != aspect:
                continue
            candidates.append(cap)
        if not candidates:
            return None
        candidates.sort(
            key=lambda c: (int(c.get("hits") or 0), float(c.get("updated_at") or 0)),
            reverse=True,
        )
        return dict(candidates[0])

    @staticmethod
    def _cap_concept(cap: dict) -> tuple[str | None, str | None]:
        """Return the (topic, aspect) for a stored capability. Reads the
        inferred field first; falls back to parsing the intent name
        (vault_<topic>_<aspect> or vault_<topic>) for legacy caps that
        pre-date LLM classification."""
        inferred = cap.get("inferred") or {}
        topic = LearnedCapabilityEngine._clean_noun(inferred.get("topic"))
        aspect = LearnedCapabilityEngine._clean_noun(inferred.get("aspect"))
        if topic:
            return topic, aspect
        intent = str(cap.get("intent") or "")
        match = re.match(r"vault_([a-z][a-z0-9_]*?)(?:_([a-z][a-z0-9_]*))?$", intent)
        if not match:
            return None, None
        return match.group(1), match.group(2) or "status"

    def record_alias(self, new_text: str, source_cap: dict) -> dict | None:
        return self.store.remember_alias(new_text, source_cap)

    def propose(self, text: str) -> dict | None:
        if self.is_scout_specific(text):
            return None
        normalized = normalize_text(text)
        if not normalized:
            return None
        return self._propose_code_monkey_recipe(normalized)

    def propose_correction(
        self,
        correction: str,
        previous_proposal: dict | None = None,
        original_message: str = "",
    ) -> dict | None:
        """Classify a correction turn with full prior context (original
        message + previous proposal) so the LLM knows what's being
        corrected. Falls back to plain classification if context-aware
        inference fails to find a topic."""
        previous_inferred = (previous_proposal or {}).get("inferred") or {}
        previous_topic = previous_inferred.get("topic")
        previous_aspect = previous_inferred.get("aspect")

        new_topic, new_aspect = self.classify_correction(
            correction,
            original=original_message,
            previous_topic=previous_topic,
            previous_aspect=previous_aspect,
        )

        # Fall back to plain classification if the context-aware call gave
        # nothing useful and we have no previous topic to anchor to.
        if new_topic is None and new_aspect is None and previous_topic is None:
            return None

        if new_topic is None and previous_topic is not None:
            chosen_aspect = new_aspect or previous_aspect or self._default_aspect_for(previous_topic)
            return self._code_monkey_recipe_proposal(previous_topic, chosen_aspect)

        if new_topic is not None:
            safe = self.safety.classify_capability_request(normalize_text(correction))
            if not safe.get("allowed"):
                return None
            return self._code_monkey_recipe_proposal(
                new_topic,
                new_aspect or previous_aspect or self._default_aspect_for(new_topic),
            )

        return None

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
        """Pick a sane default aspect when one isn't supplied. The classifier
        almost always returns an aspect for system queries, so this is only
        a last-resort fallback when topic is known but aspect was 'none'."""
        return "status"

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
        # Topic/aspect are free-form LLM-inferred nouns. Pretty-print a handful
        # of common acronyms but otherwise display the LLM's noun directly so
        # any topic the model recognized shows up reasonably.
        acronyms = {"cpu": "CPU", "gpu": "GPU", "ram": "RAM", "os": "operating system", "ip": "IP", "dns": "DNS"}
        topic_display = acronyms.get(topic, topic)
        if topic == "memory" and aspect == "hardware":
            topic_display = "RAM"
        return f"Vault {topic_display} {aspect}".strip()

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
                # If every attempt failed because a specific binary isn't
                # installed, look up the apt package and surface that to the
                # caller — they'll ask the user for permission to install.
                missing_binary = recipe_result.get("missing_binary")
                suggested_package = (
                    self.lookup_package_for_binary(missing_binary)
                    if missing_binary
                    else None
                )
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": recipe_result.get("error") or "Code Monkey could not generate a learned command recipe.",
                    "returncode": -1,
                    "error": recipe_result.get("error"),
                    "ran_at": time.time(),
                    "code_monkey_task": {
                        "ok": False,
                        "single_recipe": True,
                        "error": recipe_result.get("error"),
                        "missing_binary": missing_binary,
                        "suggested_package": suggested_package,
                    },
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
                    "missing_binary": missing_binary,
                    "suggested_package": suggested_package,
                }
            code_monkey_task = {
                "ok": True,
                "single_recipe": True,
                "generator": recipe_result.get("generator") or "code_monkey_single_recipe",
            }
            try:
                recipe = self._materialize_generated_recipe(text, recipe_result.get("recipe") or {})
            except ValueError as exc:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": str(exc),
                    "returncode": -1,
                    "error": f"Generated recipe failed safety/format check: {exc}",
                    "ran_at": time.time(),
                    "code_monkey_task": {"ok": False, "single_recipe": True, "error": str(exc)},
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

    def lookup_package_for_binary(self, binary: str) -> str | None:
        """Ask the LLM which Debian/Ubuntu apt package provides a binary.
        Returns the package name (validated as a sane apt name) or None when
        the model can't identify one."""
        binary = (binary or "").strip()
        if not binary or self.model is None:
            return None
        prompt = (
            "On a Debian or Ubuntu system, which apt package provides the "
            f"binary named {binary!r}? Respond with ONLY the package name as "
            "a single lowercase token, or the literal token 'unknown' if you "
            "don't know.\n\nPackage:"
        )
        try:
            raw = self.model.generate(prompt, think=False, timeout=10)
        except Exception:
            return None
        candidate = (raw or "").strip().splitlines()[0].strip() if raw else ""
        candidate = candidate.strip("`'\"").strip().lower()
        if not candidate or candidate in {"unknown", "none", "n/a", "null"}:
            return None
        if not re.fullmatch(r"[a-z0-9][a-z0-9+\-.]{0,63}", candidate):
            return None
        return candidate

    def install_package(self, package: str, timeout: int = 180) -> dict:
        """Install an apt package non-interactively. Returns a dict with
        ok/stdout/stderr/returncode. The caller is responsible for getting
        user confirmation BEFORE calling this — this method just runs the
        install."""
        package = (package or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9+\-.]{0,63}", package):
            return {"ok": False, "error": f"refusing to install: bad package name {package!r}"}
        try:
            completed = subprocess.run(
                ["sudo", "-n", "apt-get", "install", "-y",
                 "--no-install-recommends", package],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
                check=False,
            )
            return {
                "ok": completed.returncode == 0,
                "package": package,
                "stdout": (completed.stdout or "")[-2000:],
                "stderr": (completed.stderr or "")[-2000:],
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "package": package, "error": f"install timed out after {timeout}s"}
        except FileNotFoundError as exc:
            return {"ok": False, "package": package, "error": f"sudo or apt-get not found: {exc}"}
        except Exception as exc:
            return {"ok": False, "package": package, "error": str(exc)}

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

    _FORBIDDEN_SHELL_TOKENS = frozenset({
        "|", "||", "&&", "&", ";", ">", ">>", "<", "<<", "<<<",
    })

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
        for token in argv:
            if token in self._FORBIDDEN_SHELL_TOKENS:
                return self._error(
                    f"Refusing to run: command contains shell metachar {token!r} "
                    "but the executor runs argv without a shell."
                )
        binary = argv[0]
        if "/" not in binary and not shutil.which(binary):
            return self._error(f"Command not found: {binary}", returncode=127)
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
