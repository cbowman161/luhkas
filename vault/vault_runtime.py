from __future__ import annotations

import subprocess
import threading

from blackboard import Blackboard
from background_manager import BackgroundManager
from capability_registry import CapabilityRegistry
from command_agent import CommandAgent
from config import INSTALLED_CAPABILITIES_DIR
from event_log import EventLog
from interaction_interpreter import InteractionInterpreter
from job_manager import JobManager
from learned_capabilities import LearnedCapabilityEngine, normalize_text as _learned_normalize
from models import model_manifest, warm_models
from node_health_monitor import NodeHealthMonitor
from node_registry import NodeRegistry
from planner import Planner
from router import Router
from scout_integration import ScoutVaultBridge, _sanitize_generated_response
from skill_registry import SkillRegistry
from tts_formatter import format_for_tts


_UPDATES_KEYWORDS = {
    "updates", "notification", "notifications", "status", "progress", "news", "alerts",
}

_JOBS_KEYWORDS = {
    "jobs", "tasks", "queue", "running",
}

_UPDATES_COMMANDS = {
    "updates",
    "status",
    "progress",
    "whats the status",
    "any updates",
    "notification",
    "notifications",
    "show notifications",
    "check notifications",
    "show updates",
    "get updates",
    "check updates",
}

_JOBS_COMMANDS = {
    "jobs",
    "tasks",
    "queue",
    "running",
    "list jobs",
    "show jobs",
    "my jobs",
    "active jobs",
}

_CODE_MONKEY_HEALTH_COMMANDS = {
    "code monkey",
    "code monkey health",
    "code monkey status",
    "coder health",
    "coder status",
}

SESSION_COMMANDS = {
    "new": "new",
    "new task": "new",
    "new session": "new",
    "reset": "new",
    "reset session": "new",
    "clear session": "new",
}


RUN_COMMANDS = {
    "run",
    "run it",
    "execute",
    "execute it",
}


def _command_text(text: str) -> str:
    import re
    return re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()


def _item_aliases(item: dict) -> set[str]:
    aliases = set()
    for key in ("name", "display_name"):
        value = item.get(key)
        if value:
            normalized = _command_text(str(value).replace("_", " "))
            aliases.add(normalized)
            aliases.add(_command_text(value))
    for example in item.get("examples") or []:
        if example:
            aliases.add(_command_text(str(example).replace("_", " ")))
    return {alias for alias in aliases if alias}


def _matches_named_item(text: str, item: dict) -> bool:
    aliases = _item_aliases(item)
    return text in aliases


def _is_affirmative(text: str) -> bool:
    normalized = _command_text(text)
    return normalized in {
        "yes", "yeah", "yep", "yup", "correct", "right", "sure", "ok", "okay",
        "sounds right", "thats right", "that is right", "exactly", "affirmative",
    }


def _is_denial(text: str) -> bool:
    return _command_text(text) in {"no", "nope", "nah", "wrong", "not right", "negative"}


def _extract_correction(text: str) -> str | None:
    import re
    raw = str(text or "").strip()
    lowered = raw.lower()
    for pattern in (
        r"^no[,.]?\s+i\s+mean\s+(.+)$",
        r"^no[,.]?\s+i\s+meant\s+(.+)$",
        r"^no[,.]?\s+(.+)$",
        r"^nope[,.]?\s+(.+)$",
        r"^nah[,.]?\s+(.+)$",
        r"^actually[,.]?\s+(.+)$",
        r"^not\s+that[,.]?\s+(.+)$",
    ):
        match = re.match(pattern, lowered)
        if match:
            return match.group(1).strip()
    return None


class VaultRuntime:
    """Stateful main-vault orchestrator used by CLI and service frontends."""

    def __init__(self):
        self.registry = CapabilityRegistry()
        self.skill_registry = SkillRegistry()
        self.blackboard = Blackboard()
        self.planner = Planner(self.registry, self.skill_registry)
        self.interpreter = InteractionInterpreter()
        self.event_log = EventLog()
        self.job_manager = JobManager(self.event_log)
        self.router = Router(
            self.blackboard,
            self.event_log,
            self.job_manager,
            self.registry,
            self.skill_registry,
        )
        self.node_registry = NodeRegistry()
        self.scout = ScoutVaultBridge()
        self.scout.node_registry = self.node_registry
        self.scout.capability_registry = self.registry
        self.scout.skill_registry = self.skill_registry
        self.command_agent = CommandAgent(INSTALLED_CAPABILITIES_DIR)
        self.background_manager = BackgroundManager(self.event_log)
        self.background_manager.start_all_from_dir(INSTALLED_CAPABILITIES_DIR)
        self.router.command_agent = self.command_agent
        self.router.background_manager = self.background_manager
        self.model_warmup = warm_models()
        self.node_health_monitor = NodeHealthMonitor(self.node_registry, self.event_log)
        self.node_health_monitor.start()
        self.active_task_id = None
        self._last_active_node_id = "cli"
        self.learned_capabilities = LearnedCapabilityEngine()
        # Per-node active_task_id so multi-node sessions don't clobber each other
        self._node_task_ids: dict = {}

    def handle(self, user_input, node_id: str = "cli"):
        user_input = (user_input or "").strip()
        # Use per-node active_task_id so concurrent sessions don't clobber each other
        self.active_task_id = self._node_task_ids.get(node_id)

        if not user_input:
            return self._enrich({
                "mode": "direct",
                "message": "",
                "active_task_id": self.active_task_id,
            }, node_id)

        self._current_node_id = node_id
        self._last_active_node_id = node_id

        lowered = user_input.lower()
        command_text = _command_text(user_input)

        if command_text in _UPDATES_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_updates(self.active_task_id),
                "code_monkey_updates",
            ))

        if command_text in _JOBS_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_jobs(self.active_task_id),
                "code_monkey_jobs",
            ))

        if command_text in _CODE_MONKEY_HEALTH_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_code_monkey_health(self.active_task_id),
                "code_monkey_health",
            ))

        if lowered in {"review", "review code", "review build", "review project", "review results"} or lowered.startswith("review "):
            parts = user_input.strip().split(None, 1)
            explicit_id = parts[1].strip() if len(parts) > 1 and parts[0].lower() == "review" else None
            task_id = (
                explicit_id
                or self.active_task_id
                or self.blackboard.get_session_value("last_completed_task_id")
            )
            response = self.router.enter_review_session(task_id, task_id or self.active_task_id)
            return self._remember_active(response)

        # Single-word command keywords — catch before any LLM is invoked.
        if lowered in _UPDATES_KEYWORDS and " " not in lowered:
            return self._remember_active(self._runtime_command_response(
                self.router.show_updates(self.active_task_id),
                "code_monkey_updates",
            ))

        if lowered in _JOBS_KEYWORDS and " " not in lowered:
            return self._remember_active(self._runtime_command_response(
                self.router.show_jobs(self.active_task_id),
                "code_monkey_jobs",
            ))

        command = SESSION_COMMANDS.get(lowered)

        if command == "new":
            self.active_task_id = None
            self.blackboard.reset_session()
            return {
                "mode": "direct",
                "message": "Cleared active session.",
                "active_task_id": self.active_task_id,
            }

        pending = self.blackboard.get_pending_decision()

        if isinstance(pending, dict) and pending.get("type") == "learned_capability_confirmation":
            learned_flow = self._handle_learned_capability_confirmation(user_input, node_id)
            if learned_flow is not None:
                return learned_flow

        if isinstance(pending, dict) and pending.get("type") == "learned_execution_review":
            review = self._handle_learned_execution_review(user_input, node_id)
            if review is not None:
                return review

        # Requirements gathering and review sessions receive raw user text — no intent
        # classification needed since the agents handle their own conversation logic.
        if pending and pending.get("type") in {
            "code_monkey_requirements", "code_monkey_review",
            "code_monkey_pick_review", "code_monkey_overlap_decision",
        }:
            response = self.router.resolve_pending_decision(
                pending=pending,
                user_input=user_input,
                active_task_id=self.active_task_id,
            )
            return self._remember_active(response)

        interpretation = self.interpreter.interpret(
            user_input=user_input,
            pending=pending,
            session=getattr(self.blackboard, "session", {}),
        )

        if pending:
            response = self.router.route_interpreted_pending(
                interpretation=interpretation,
                user_input=user_input,
                active_task_id=self.active_task_id,
            )
            return self._remember_active(response)

        if lowered in RUN_COMMANDS:
            last = self.blackboard.get_session_value("last_skill")

            if last:
                response = self.router.run_pending_skill(
                    pending=last,
                    active_task_id=self.active_task_id,
                )
                return self._remember_active(response)

        # Deterministic command routing — zero LLM cost for known capability commands.
        cmd_response = self.command_agent.handle(user_input)
        if cmd_response is not None:
            cmd_response["active_task_id"] = self.active_task_id
            return self._remember_active(cmd_response)

        learned_response = self._handle_learned_capability_request(user_input, node_id)
        if learned_response is not None:
            return learned_response

        plan = self.planner.decide(user_input)

        response = self.router.route(
            plan=plan,
            user_input=user_input,
            active_task_id=self.active_task_id,
        )
        response["plan"] = plan
        return self._remember_active(response)

    def dispatch_guard_alert(self, payload: dict) -> None:
        """Called when the scout detects a person in guard mode.

        Routing priority:
          1. Node where primary user was last identified (within 5 min)
          2. All nodes with detected people
          3. Most recently active node
        """
        confidence = payload.get("confidence", 0)
        primary_user = self.scout.identity_profile.get("primary_user") if self.scout else None
        target_nodes = self.node_registry.find_alert_targets(primary_user=primary_user)

        alert = {
            "type": "guard_alert",
            "severity": "critical",
            "confidence": confidence,
            "payload": payload,
        }

        self.event_log.notify(
            "guard_alert",
            "critical",
            f"Person detected (confidence {confidence:.0%}) — routing to: {target_nodes}",
            payload,
        )

        for nid in target_nodes:
            self.node_registry.queue_alert(nid, alert)

        threading.Thread(target=self._guard_os_alert, args=(target_nodes,), daemon=True).start()

    def _guard_os_alert(self, node_ids: list) -> None:
        try:
            subprocess.run(
                ["notify-send", "-u", "critical", "-t", "0",
                 "LUHKAS GUARD ALERT",
                 f"Person detected by scout\nRouted to: {', '.join(node_ids)}"],
                timeout=5,
            )
        except Exception:
            pass

    def handle_presence(self, message: str, node_id: str = "scout", presence_context: dict | None = None):
        """Route a presence/chat message through the scout bridge and return an
        enriched response with the same shape as handle()."""
        self.active_task_id = self._node_task_ids.get(node_id)
        self._current_node_id = node_id
        self._last_active_node_id = node_id
        if isinstance(presence_context, dict):
            self.node_registry.update_capabilities(
                node_id,
                capabilities=presence_context.get("node_capabilities") or {},
                modules=presence_context.get("modules") or {},
            )
        direct = self._handle_deterministic_presence_command(message, node_id)
        if direct is not None:
            self.node_registry.update_activity(node_id, identity=None)
            return direct
        result = self.scout.handle_message(
            message,
            source=node_id,
            node_id=node_id,
            presence_context=presence_context,
        )
        # Scout bridge stores the reply text in "response" and the input in "message"
        reply_text = result.get("response") or ""
        result["message"] = reply_text
        result["response_composed"] = True
        active_id = result.get("active_identity")
        self.node_registry.update_activity(node_id, identity=active_id)
        return self._enrich(result, node_id)

    def _handle_deterministic_presence_command(self, message: str, node_id: str) -> dict | None:
        """Run known capability/skill commands before Scout chat routing.

        Presence messages normally go through ScoutVaultBridge, but the vault
        owns capabilities, skills, jobs, and updates. This mirrors the zero-LLM
        command surface used by handle() so phrases like "show updates" never
        fall through to vision analysis.
        """
        text = _command_text(message)
        if not text:
            return None

        learned_flow = self._handle_learned_capability_confirmation(message, node_id)
        if learned_flow is not None:
            return learned_flow

        review = self._handle_learned_execution_review(message, node_id)
        if review is not None:
            return review

        pending = self.blackboard.get_pending_decision()
        if pending and pending.get("type") in {
            "existing_skill",
            "modify_skill_details",
            "skill_evolution",
            "capability_evolution",
            "run_skill_args",
            "code_monkey_pick_review",
            "code_monkey_overlap_decision",
            "code_monkey_requirements",
            "code_monkey_review",
        }:
            return self.handle(message, node_id=node_id)

        if text in _UPDATES_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_updates(self.active_task_id),
                "code_monkey_updates",
            ))

        if text in _JOBS_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_jobs(self.active_task_id),
                "code_monkey_jobs",
            ))

        if text in _CODE_MONKEY_HEALTH_COMMANDS:
            return self._remember_active(self._runtime_command_response(
                self.router.show_code_monkey_health(self.active_task_id),
                "code_monkey_health",
            ))

        command_response = self.command_agent.handle(message)
        if command_response is not None:
            command_response["active_task_id"] = self.active_task_id
            command_response["deterministic"] = True
            command_response["deterministic_source"] = "installed_capability_command"
            return self._remember_active(command_response)

        capability_response = self._handle_named_capability_command(text)
        if capability_response is not None:
            return self._remember_active(capability_response)

        skill_response = self._handle_named_skill_command(text, message)
        if skill_response is not None:
            return self._remember_active(skill_response)

        learned_response = self._handle_learned_capability_request(message, node_id)
        if learned_response is not None:
            return learned_response

        return None

    def _learned_engine(self) -> LearnedCapabilityEngine:
        engine = getattr(self, "learned_capabilities", None)
        if engine is None:
            engine = LearnedCapabilityEngine()
            self.learned_capabilities = engine
        return engine

    def _set_pending(self, value: dict | None) -> None:
        if hasattr(self.blackboard, "set_pending_decision"):
            self.blackboard.set_pending_decision(value)
        else:
            self.blackboard.pending = value

    def _clear_pending(self) -> None:
        if hasattr(self.blackboard, "clear_pending_decision"):
            self.blackboard.clear_pending_decision()
        else:
            self.blackboard.pending = None

    def _learned_capability_pending_update(self) -> dict | None:
        engine = self._learned_engine()
        updates = engine.check_pending_code_monkey()
        if not updates:
            return None
        message = engine.summarize_pending_update(updates[0])
        return {
            "message": message,
            "update": updates[0],
        }

    def _attach_learned_capability_update(self, response: dict) -> dict:
        update = self._learned_capability_pending_update()
        if update is None:
            return response
        response = dict(response)
        response["message"] = f"{update['message']} {response.get('message') or ''}".strip()
        data = dict(response.get("data") or {})
        raw_update = update["update"]
        data["learned_capability_update"] = {
            "task_id": raw_update.get("task_id"),
            "state": raw_update.get("state"),
            "input": raw_update.get("input"),
            "proposal": raw_update.get("proposal"),
            "notified": raw_update.get("notified"),
        }
        response["data"] = data
        return response

    def _handle_learned_execution_review(self, message: str, node_id: str) -> dict | None:
        """One-turn handler that runs immediately after a concept-match
        execution. If the user says 'no' or 'no, X' here, the freshly-saved
        alias is removed and a new propose flow is started using the
        correction text. Any other message clears the review state and falls
        through to normal handling."""
        pending = self.blackboard.get_pending_decision()
        if not isinstance(pending, dict) or pending.get("type") != "learned_execution_review":
            return None

        correction = _extract_correction(message)
        if not correction and not _is_denial(message):
            # User moved on without correcting — review window closes.
            self._clear_pending()
            return None

        engine = self._learned_engine()
        alias_key = pending.get("alias_key") or ""
        removed = engine.store.forget(alias_key) if alias_key else False

        if correction:
            combined = f"{pending.get('original_message') or ''} {correction}".strip()
            proposal = engine.propose(combined) or engine.propose(correction)
        else:
            proposal = None

        if proposal is None:
            self._clear_pending()
            note = (
                "Got it, I removed that wrong learned command."
                if removed
                else "Got it."
            )
            tail = " Try rephrasing what you wanted."
            return self._remember_active({
                "mode": "direct",
                "message": note + tail,
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_execution_review",
                "compose": False,
                "response_composed": True,
            })

        original_message = pending.get("original_message") or message
        self._set_pending({
            "type": "learned_capability_confirmation",
            "original_message": original_message,
            "proposal": proposal,
            "node_id": node_id,
            "correction": correction,
        })
        prefix = "Removed the wrong learned command." if removed else "Got it."
        return self._remember_active({
            "mode": "direct",
            "message": f"{prefix} I think you actually mean {proposal['description']}. Is that right?",
            "data": {"proposal": proposal, "alias_removed": removed},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "learned_execution_review",
            "compose": False,
            "response_composed": True,
        })

    def _handle_learned_capability_confirmation(self, message: str, node_id: str) -> dict | None:
        pending = self.blackboard.get_pending_decision()
        if not isinstance(pending, dict) or pending.get("type") != "learned_capability_confirmation":
            return None
        engine = self._learned_engine()
        if _is_affirmative(message):
            original = pending.get("original_message") or ""
            proposal = pending.get("proposal") or {}
            result = engine.learn_and_execute(
                original,
                proposal,
                confirmed_by=str(message or "user_confirmation"),
            )
            self._clear_pending()
            capability = result.get("capability") or proposal
            summary = engine.summarize_result(original, capability, result)
            if result.get("saved"):
                summary = f"Learned command saved. {summary}"
            return self._remember_active({
                "mode": "direct",
                "message": summary,
                "data": {
                    "learned_capability": capability,
                    "execution_result": result,
                },
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": f"learned_capability:{capability.get('name') or capability.get('intent')}",
                "compose": False,
                "response_composed": True,
            })
        correction = _extract_correction(message)
        if correction:
            previous_proposal = pending.get("proposal") or {}
            proposal = engine.propose_correction(correction, previous_proposal)
            if proposal is None:
                self._clear_pending()
                return self._remember_active({
                    "mode": "direct",
                    "message": "I do not see a safe Vault-side deterministic path for that correction yet.",
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": "learned_capability_confirmation",
                    "compose": False,
                    "response_composed": True,
                })
            original_message = (
                pending.get("original_message")
                if engine.correction_updates_previous_request(proposal, previous_proposal)
                else correction
            )
            self._set_pending({
                "type": "learned_capability_confirmation",
                "original_message": original_message,
                "proposal": proposal,
                "node_id": node_id,
                "correction": correction,
            })
            return self._remember_active({
                "mode": "direct",
                "message": f"I think you mean {proposal['description']}. Is that right?",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_capability_confirmation",
                "compose": False,
                "response_composed": True,
            })
        if _is_denial(message):
            self._clear_pending()
            return self._remember_active({
                "mode": "direct",
                "message": "Got it. I will not save a deterministic path for that.",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_capability_confirmation",
                "compose": False,
                "response_composed": True,
            })
        return None

    def _handle_learned_capability_request(self, message: str, node_id: str) -> dict | None:
        engine = self._learned_engine()
        learned = engine.lookup(message)
        alias_source = None
        if learned is None:
            concept = engine.lookup_by_concept(message)
            if concept is not None:
                learned = concept
                alias_source = concept
        if learned is not None:
            result = engine.execute_capability(learned)
            alias_recorded = False
            if result.get("ok") and alias_source is not None:
                stored_alias = engine.record_alias(message, alias_source)
                if stored_alias is not None:
                    learned = stored_alias
                    alias_recorded = True
            summary = engine.summarize_result(message, learned, result)
            summary = f"Learned command. {summary}"
            # Set a one-turn review state so the user can correct a wrong
            # concept-match by saying "no" / "no, X" — that will remove the
            # just-saved alias and propose a fresh learn flow.
            if alias_recorded:
                self._set_pending({
                    "type": "learned_execution_review",
                    "original_message": message,
                    "alias_key": _learned_normalize(message),
                    "executed_cap_intent": learned.get("intent"),
                    "node_id": node_id,
                })
            return self._remember_active(self._attach_learned_capability_update({
                "mode": "direct",
                "message": summary,
                "data": {
                    "learned_capability": learned,
                    "execution_result": result,
                    "alias_recorded": alias_recorded,
                },
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": f"learned_capability:{learned.get('name') or learned.get('intent')}",
                "compose": False,
                "response_composed": True,
            }))
        proposal = engine.propose(message)
        if proposal is None:
            return None
        self._set_pending({
            "type": "learned_capability_confirmation",
            "original_message": message,
            "proposal": proposal,
            "node_id": node_id,
        })
        return self._remember_active(self._attach_learned_capability_update({
            "mode": "direct",
            "message": f"I think you mean {proposal['description']}. Is that right?",
            "data": {"proposal": proposal},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "learned_capability_confirmation",
            "compose": False,
            "response_composed": True,
        }))

    def _handle_named_capability_command(self, text: str) -> dict | None:
        for capability in self.registry.list():
            if not _matches_named_item(text, capability):
                continue
            subsystem = capability.get("subsystem")
            action = capability.get("action")
            if subsystem == "event_log":
                return self._runtime_command_response(
                    self.router.show_updates(self.active_task_id),
                    f"capability:{capability.get('name')}",
                )
            if subsystem == "job_manager":
                return self._runtime_command_response(
                    self.router.show_jobs(self.active_task_id),
                    f"capability:{capability.get('name')}",
                )
            if subsystem == "system_agent":
                result = self.router.system_agent.run_direct(action, capability=capability)
                return {
                    "mode": "direct",
                    "message": result.get("message", ""),
                    "data": result,
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": f"capability:{capability.get('name')}",
                }
            if subsystem == "code_monkey":
                return self.router.start_code_monkey_requirements(
                    user_input=capability.get("description") or capability.get("name"),
                    active_task_id=self.active_task_id,
                )
            if subsystem == "chat_agent":
                return None
            return {
                "mode": "direct",
                "message": f"Capability `{capability.get('name')}` is registered, but I do not have a deterministic executor for subsystem `{subsystem}`.",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": f"capability:{capability.get('name')}",
            }
        return None

    def _handle_named_skill_command(self, text: str, message: str) -> dict | None:
        run_prefix = None
        for prefix in ("run ", "execute ", "start ", "use "):
            if text.startswith(prefix):
                run_prefix = prefix
                break
        target = text[len(run_prefix):].strip() if run_prefix else text
        for skill in self.skill_registry.list():
            if not _matches_named_item(target, skill):
                continue
            pending = {
                "type": "existing_skill",
                "skill": skill.get("name"),
                "filename": skill.get("filename"),
                "description": skill.get("description"),
                "original_request": message,
                "active_task_id": self.active_task_id,
            }
            self.router.set_last_skill(skill.get("name"), skill.get("filename"))
            if run_prefix:
                return self.router.run_pending_skill(pending, self.active_task_id)
            return self.router.confirm_existing_skill(
                {
                    "skill": skill.get("name"),
                    "action": "confirm_existing_skill",
                },
                message,
                self.active_task_id,
            )
        return None

    def handle_presence_message(self, message, source=None):
        return self.scout.handle_message(message, source=source)

    def handle_chat(self, message, source=None):
        return self.handle_presence_message(message, source=source)

    def health(self):
        try:
            code_monkey = self.router.code_monkey.health()
        except Exception as exc:
            code_monkey = {
                "ok": False,
                "error": str(exc),
            }

        return {
            "ok": True,
            "service": "vault_runtime",
            "presence_owner": "vault_pc",
            "active_task_id": self.active_task_id,
            "models": model_manifest(),
            "model_warmup": self.model_warmup,
            "code_monkey": code_monkey,
            "scout": {
                "url": self.scout.scout_url,
                "active_identity": self.scout.active_identity,
            },
        }

    def _remember_active(self, response):
        self.active_task_id = response.get("active_task_id", self.active_task_id)
        node_id = getattr(self, "_current_node_id", "cli")
        # Persist per-node task id
        self._node_task_ids[node_id] = self.active_task_id
        return self._enrich(response, node_id)

    def _runtime_command_response(self, response: dict, source: str) -> dict:
        """Mark operational router views as deterministic, not generated prose."""
        result = dict(response or {})
        result.setdefault("mode", "direct")
        result["active_task_id"] = result.get("active_task_id", self.active_task_id)
        result["deterministic"] = True
        result["deterministic_source"] = source
        result["compose"] = False
        result["response_composed"] = True
        return result

    def pop_alerts(self, node_id: str) -> list:
        """Return and clear pending alerts queued for a node."""
        return self.node_registry.pop_alerts(node_id)

    def _enrich(self, response: dict, node_id: str) -> dict:
        """Add tts, display, and has_display fields to every response."""
        message = response.get("message") or ""
        if message and response.get("compose", True) and not response.get("response_composed"):
            response["raw_message"] = message
            response["message"] = self._compose_runtime_message(response, node_id)
            response["response_composed"] = True
            message = response["message"]
        if response.get("response_composed") or "tts" not in response:
            response["tts"] = format_for_tts(message)
        has_display = self.node_registry.has_display(node_id)
        response["has_display"] = has_display
        response["node_id"] = node_id
        if not has_display:
            response["message"] = response["tts"]
        return response

    def _compose_runtime_message(self, response: dict, node_id: str) -> str:
        message = str(response.get("message") or "")
        if not message:
            return message
        scout = getattr(self, "scout", None)
        composer = getattr(scout, "response_composer", None)
        if composer is None:
            return f"Fallback response: {message} (response composer unavailable)"
        recent = []
        try:
            recent = [
                str(turn.get("response") or "").strip()
                for turn in scout.turns[-8:]
                if str(turn.get("response") or "").strip()
            ]
        except Exception:
            recent = []
        return composer.compose(
            response_type="runtime_direct",
            user_message="",
            facts={
                "deterministic_answer": message,
                "mode": response.get("mode"),
                "node_id": node_id,
                "deterministic": response.get("deterministic"),
                "source": response.get("deterministic_source"),
                "data": response.get("data"),
            },
            fallback=message,
            contract=scout.response_contract("runtime_direct", {"ok": True}),
            recent_responses=recent,
            options={"num_predict": 90, "temperature": 0.62, "top_p": 0.9},
            validator=lambda text: scout.response_policy_violation(text, {"ok": True}, "runtime_direct"),
            sanitizer=_sanitize_generated_response,
        )
