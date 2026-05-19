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
from models import model_manifest, warm_models
from node_health_monitor import NodeHealthMonitor
from node_registry import NodeRegistry
from planner import Planner
from router import Router
from scout_integration import ScoutVaultBridge
from skill_registry import SkillRegistry
from tts_formatter import format_for_tts


_UPDATES_KEYWORDS = {
    "updates", "notifications", "status", "progress", "news", "alerts",
}

_JOBS_KEYWORDS = {
    "jobs", "tasks", "queue", "running",
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

        if lowered in {
            "updates", "status", "progress", "what's the status", "any updates",
            "notifications", "show notifications", "check notifications",
            "show updates", "get updates", "check updates",
        }:
            return self._remember_active(self.router.show_updates(self.active_task_id))

        if lowered in {"jobs", "list jobs", "show jobs", "my jobs", "active jobs"}:
            return self._remember_active(self.router.show_jobs(self.active_task_id))

        if lowered in {"code monkey", "code monkey health", "coder health", "coder status"}:
            return self._remember_active(self.router.show_code_monkey_health(self.active_task_id))

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
            return self._remember_active(self.router.show_updates(self.active_task_id))

        if lowered in _JOBS_KEYWORDS and " " not in lowered:
            return self._remember_active(self.router.show_jobs(self.active_task_id))

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

        if text in {
            "updates", "status", "progress", "whats the status", "any updates",
            "notifications", "show notifications", "check notifications",
            "show updates", "get updates", "check updates",
        }:
            return self._remember_active(self.router.show_updates(self.active_task_id))

        if text in {"jobs", "list jobs", "show jobs", "my jobs", "active jobs"}:
            return self._remember_active(self.router.show_jobs(self.active_task_id))

        if text in {"code monkey", "code monkey health", "coder health", "coder status"}:
            return self._remember_active(self.router.show_code_monkey_health(self.active_task_id))

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

        return None

    def _handle_named_capability_command(self, text: str) -> dict | None:
        for capability in self.registry.list():
            if not _matches_named_item(text, capability):
                continue
            subsystem = capability.get("subsystem")
            action = capability.get("action")
            if subsystem == "event_log":
                return self.router.show_updates(self.active_task_id)
            if subsystem == "job_manager":
                return self.router.show_jobs(self.active_task_id)
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

    def pop_alerts(self, node_id: str) -> list:
        """Return and clear pending alerts queued for a node."""
        return self.node_registry.pop_alerts(node_id)

    def _enrich(self, response: dict, node_id: str) -> dict:
        """Add tts, display, and has_display fields to every response."""
        message = response.get("message") or ""
        if "tts" not in response:
            response["tts"] = format_for_tts(message)
        has_display = self.node_registry.has_display(node_id)
        response["has_display"] = has_display
        response["node_id"] = node_id
        if not has_display:
            response["message"] = response["tts"]
        return response
