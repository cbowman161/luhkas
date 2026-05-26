from __future__ import annotations

import re
import subprocess
import threading
import time

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

_AUDIT_CAPS_COMMANDS = {
    "audit caps",
    "audit learned caps",
    "audit learned commands",
    "consolidate caps",
    "consolidate learned caps",
    "consolidate learned commands",
    "merge duplicate caps",
}

# `install <pkg>` admin command — single-arg form. Recognized when the
# remainder validates as an apt-style package name. Anything not matching
# falls through to normal routing (so "install a fence in the yard" stays
# chat).
_INSTALL_COMMAND_PREFIX = "install "

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
    """Deprecated fast-path heuristic — only catches the most unambiguous
    leading-"no"/"nope"/"nah" forms. Anything subtler is now decided by the
    LLM intent classifier (LearnedCapabilityEngine.classify_pending_intent)
    called from the handlers when this returns None.

    Kept as a cheap pre-check so plain "no, hardware" responses don't pay an
    extra LLM round-trip."""
    import re
    raw = str(text or "").strip()
    lowered = raw.lower()
    for pattern in (
        r"^no[,.]?\s+(.+)$",
        r"^nope[,.]?\s+(.+)$",
        r"^nah[,.]?\s+(.+)$",
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
        # In-flight background workers — description/package → started_at unix.
        # Surfaced in "any updates" so the user knows learning is still cooking.
        self._active_learn_jobs: dict = {}
        self._active_install_jobs: dict = {}
        self._async_job_lock = threading.Lock()
        # Per-request stash for alerts to inline into the /ui or
        # /presence/message response. Thread-local so concurrent
        # requests (ThreadingHTTPServer spawns one thread per call)
        # don't bleed alerts across each other.
        self._inline_alerts_tls = threading.local()

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
                self._show_updates_with_progress(),
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

        if command_text in _AUDIT_CAPS_COMMANDS:
            return self._remember_active(self._start_audit_caps(node_id))

        install_response = self._maybe_handle_install_command(user_input, node_id)
        if install_response is not None:
            return install_response

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
                self._show_updates_with_progress(),
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

        pending = self._get_pending(node_id)

        if isinstance(pending, dict) and pending.get("type") == "learned_capability_confirmation":
            learned_flow = self._handle_learned_capability_confirmation(user_input, node_id)
            if learned_flow is not None:
                return learned_flow

        if isinstance(pending, dict) and pending.get("type") == "learned_execution_review":
            review = self._handle_learned_execution_review(user_input, node_id)
            if review is not None:
                return review

        if isinstance(pending, dict) and pending.get("type") == "learned_install_confirmation":
            install_flow = self._handle_learned_install_confirmation(user_input, node_id)
            if install_flow is not None:
                return install_flow

        if isinstance(pending, dict) and pending.get("type") == "audit_merge_confirmation":
            audit_flow = self._handle_audit_merge_confirmation(user_input, node_id)
            if audit_flow is not None:
                return audit_flow

        if isinstance(pending, dict) and pending.get("type") == "memory_update_confirmation":
            memory_flow = self._handle_memory_update_confirmation(user_input, node_id)
            if memory_flow is not None:
                return memory_flow

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
        # User-is-here signal: bump activity FIRST so the registry's
        # currently_active_node_ids() check sees this node as active,
        # then drain any deferred alerts onto this node's queue, then
        # IMMEDIATELY pop them onto a thread-local so they ride back
        # out attached to THIS response. Without the immediate pop,
        # Scout's background `/alerts/pending` poller (in
        # presence_client_service.py, every 3s) frequently wins the
        # race during the LLM processing that follows, draining the
        # queue before `_enrich` can read it — alerts then sit in
        # Scout's local cache (which the web UI doesn't poll) and
        # never reach the user.
        self.node_registry.update_activity(node_id, identity=None)
        try:
            drained = self.node_registry.flush_pending_to(node_id)
            if drained:
                print(
                    f"[runtime] flushed {drained} deferred alert(s) to {node_id}",
                    flush=True,
                )
            inline = self.node_registry.pop_alerts(node_id)
        except Exception as exc:
            print(f"[runtime] flush_pending_to({node_id}) failed: {exc}", flush=True)
            inline = []
        self._inline_alerts_tls.alerts = list(inline or [])
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
        # Vision short-circuit: if a previous turn left us in
        # vision_full_analysis_confirmation, a yes here re-routes through
        # scout WITH force_full_vision so the heavy vision LLM runs.
        pending = self._get_pending(node_id)
        force_full_vision = False
        if isinstance(pending, dict) and pending.get("type") == "vision_full_analysis_confirmation":
            if _is_affirmative(message):
                force_full_vision = True
                message_for_scout = pending.get("original_message") or message
                self._clear_pending()
                presence_context = dict(presence_context or {})
                presence_context["force_full_vision"] = True
                message = message_for_scout
            elif _is_denial(message):
                self._clear_pending()
                return self._remember_active({
                    "mode": "direct",
                    "message": "OK, I won't run the full vision analysis.",
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": "vision_full_analysis_confirmation",
                    "compose": False,
                    "response_composed": True,
                })
            else:
                # Ambiguous → drop the pending and let the new message route
                # normally; the short-circuit will re-arm if it's a vision ask.
                self._clear_pending()
        # Clear any prior short-circuit marker on scout before calling.
        for attr in ("_stash_vision_short_circuit_marker", "_stash_memory_conflict_marker"):
            if hasattr(self.scout, attr):
                try:
                    delattr(self.scout, attr)
                except Exception:
                    pass
        result = self.scout.handle_message(
            message,
            source=node_id,
            node_id=node_id,
            presence_context=presence_context,
        )
        # If scout's analyze_vision dispatch short-circuited and asked the
        # user whether to do the heavy analysis, install the pending state
        # here so the user's next yes/no is routed correctly.
        marker = getattr(self.scout, "_stash_vision_short_circuit_marker", None)
        if isinstance(marker, dict) and marker.get("needs_vision_confirmation") and not force_full_vision:
            self._set_pending({
                "type": "vision_full_analysis_confirmation",
                "original_message": marker.get("original_message") or message,
                "summary": marker.get("vision_summary"),
                "node_id": node_id,
            })
            try:
                delattr(self.scout, "_stash_vision_short_circuit_marker")
            except Exception:
                pass
        # If scout detected a memory conflict (new fact contradicts stored one),
        # install pending state so the next turn's yes/no replaces or keeps.
        mem_marker = getattr(self.scout, "_stash_memory_conflict_marker", None)
        if isinstance(mem_marker, dict) and mem_marker.get("new_fact"):
            self._set_pending({
                "type": "memory_update_confirmation",
                "original_message": message,
                "conflict": mem_marker,
                "node_id": node_id,
            })
            try:
                delattr(self.scout, "_stash_memory_conflict_marker")
            except Exception:
                pass
        # Scout bridge stores the reply text in "response" and the input in "message"
        reply_text = result.get("response") or ""
        result["message"] = reply_text
        result["response_composed"] = True
        active_id = result.get("active_identity")
        self.node_registry.update_activity(node_id, identity=active_id)
        result = self._attach_notification_alert(result)
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

        install_flow = self._handle_learned_install_confirmation(message, node_id)
        if install_flow is not None:
            return install_flow

        audit_flow = self._handle_audit_merge_confirmation(message, node_id)
        if audit_flow is not None:
            return audit_flow

        memory_flow = self._handle_memory_update_confirmation(message, node_id)
        if memory_flow is not None:
            return memory_flow

        pending = self._get_pending(node_id)
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
                self._show_updates_with_progress(),
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

        if text in _AUDIT_CAPS_COMMANDS:
            return self._remember_active(self._start_audit_caps(node_id))

        install_response = self._maybe_handle_install_command(message, node_id)
        if install_response is not None:
            return install_response

        world_ingest_response = self._maybe_handle_world_ingest_command(message)
        if world_ingest_response is not None:
            return world_ingest_response

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

    def _resolve_pending_intent(
        self,
        message: str,
        *,
        previous_inferred: dict | None = None,
        previous_description: str | None = None,
        original_message: str = "",
    ) -> dict:
        """Ask the LLM what the user means with `message` in the context of a
        pending proposal or just-executed cap. Returns a dict with intent +
        (when intent is 'correct') topic/aspect.

        Cheap fast-paths first:
          - leading "no, ..." regex → returns intent=correct with the rest as
            the correction text (LLM still classifies topic/aspect from it).
          - plain `_is_affirmative` / `_is_denial` → returns those intents
            without an LLM call.

        Anything else → defer to the LLM intent classifier."""
        if _is_affirmative(message):
            return {"intent": "affirm", "topic": None, "aspect": None}
        if _is_denial(message):
            return {"intent": "deny", "topic": None, "aspect": None}
        engine = self._learned_engine()
        prev = previous_inferred or {}
        return engine.classify_pending_intent(
            message,
            original=original_message,
            previous_topic=prev.get("topic"),
            previous_aspect=prev.get("aspect"),
            previous_description=previous_description,
        )

    def _spawn_async_learn(self, *, original_message: str, proposal: dict,
                           confirmed_by: str, node_id: str) -> None:
        """Run the planner+smoke+save loop on a background thread. Tracked in
        _active_learn_jobs so 'any updates' can show in-flight work."""
        description = proposal.get("description") or "that capability"

        def _worker():
            engine = self._learned_engine()
            try:
                result = engine.learn_and_execute(
                    original_message,
                    proposal,
                    confirmed_by=confirmed_by,
                )
            except Exception as exc:
                self.event_log.write(
                    job_id=f"learn:{_learned_normalize(original_message)}:{int(time.time())}",
                    event_type="learn_failed",
                    message=f"Learning {description} crashed: {exc}"[:1200],
                    data={"original_message": original_message, "error": str(exc)},
                )
            else:
                self._notify_learn_result(original_message, proposal, result)
            finally:
                with self._async_job_lock:
                    self._active_learn_jobs.pop(description, None)

        with self._async_job_lock:
            self._active_learn_jobs[description] = time.time()
        threading.Thread(target=_worker, daemon=True).start()

    def _spawn_async_install(self, package: str, node_id: str) -> dict:
        """Install an apt package in the background. Same pattern as
        _spawn_async_learn: immediate ack to the user, notification on
        completion."""
        engine = self._learned_engine()

        def _worker():
            try:
                result = engine.install_package(package)
            except Exception as exc:
                self.event_log.write(
                    job_id=f"install:{package}:{int(time.time())}",
                    event_type="install_failed",
                    message=f"Install of {package} crashed: {exc}"[:1200],
                    data={"package": package, "error": str(exc)},
                )
                return
            finally:
                with self._async_job_lock:
                    self._active_install_jobs.pop(package, None)
            if result.get("ok"):
                self.event_log.write(
                    job_id=f"install:{package}:{int(time.time())}",
                    event_type="install_succeeded",
                    message=f"Installed {package}.",
                    data={"package": package, "result": result},
                )
            else:
                err = result.get("error") or result.get("stderr") or f"rc={result.get('returncode')}"
                self.event_log.write(
                    job_id=f"install:{package}:{int(time.time())}",
                    event_type="install_failed",
                    message=f"Install of {package} failed: {err}"[:1200],
                    data={"package": package, "result": result},
                )

        with self._async_job_lock:
            self._active_install_jobs[package] = time.time()
        threading.Thread(target=_worker, daemon=True).start()
        return self._remember_active({
            "mode": "direct",
            "message": f"I'll install {package} and let you know when it's done.",
            "data": {"install_async": True, "package": package},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "install_async",
            "compose": False,
            "response_composed": True,
        })

    def _maybe_handle_world_ingest_command(self, user_input: str) -> dict | None:
        """Recognize chat commands controlling the Wikipedia ingest:
        'start the wikipedia ingest', 'wikipedia progress', 'stop the wiki
        ingest', etc. Falls through to normal routing for anything else."""
        try:
            from world.ingest_admin import handle as world_ingest_handle
        except Exception:
            return None
        response = world_ingest_handle(user_input)
        if response is None:
            return None
        return self._remember_active(self._runtime_command_response(
            response, "world_ingest_admin",
        ))

    def _maybe_handle_install_command(self, user_input: str, node_id: str) -> dict | None:
        """Recognize `install <pkg>` admin commands. Returns a response when
        the input matches the strict pattern, None otherwise so normal routing
        proceeds (e.g. "install a screen door" stays chat)."""
        raw = (user_input or "").strip()
        lowered = raw.lower()
        if not lowered.startswith(_INSTALL_COMMAND_PREFIX):
            return None
        remainder = raw[len(_INSTALL_COMMAND_PREFIX):].strip()
        # Strip trailing punctuation
        remainder = remainder.rstrip(".!?,")
        # Strict: a single apt-style package name.
        if not re.fullmatch(r"[a-z0-9][a-z0-9+\-.]{0,63}", remainder.lower()):
            return None
        package = remainder.lower()
        # Idempotent: if already installing the same package, just say so.
        with self._async_job_lock:
            if package in self._active_install_jobs:
                return self._remember_active({
                    "mode": "direct",
                    "message": f"Already installing {package}; I'll notify you when it finishes.",
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": "install_async",
                    "compose": False,
                    "response_composed": True,
                })
        return self._spawn_async_install(package, node_id)

    def _show_updates_with_progress(self) -> dict:
        """Wrap router.show_updates so the response also reports any
        in-flight background jobs (async learning, async installs)."""
        response = self.router.show_updates(self.active_task_id)
        if not isinstance(response, dict):
            return response
        progress = self._async_progress_summary()
        if progress:
            msg = response.get("message") or ""
            response = dict(response)
            response["message"] = (
                f"{progress}\n{msg}" if msg.strip() else progress
            )
        return response

    def _async_progress_summary(self) -> str:
        """Human-readable summary of in-flight background jobs, or '' if none."""
        with self._async_job_lock:
            learn_jobs = list(self._active_learn_jobs.items())
            install_jobs = list(self._active_install_jobs.items())
        now = time.time()
        lines = []
        if learn_jobs:
            entries = ", ".join(
                f"'{desc}' ({int(now - started)}s)" for desc, started in learn_jobs
            )
            label = "learn job" if len(learn_jobs) == 1 else "learn jobs"
            lines.append(f"{len(learn_jobs)} {label} in progress: {entries}")
        if install_jobs:
            entries = ", ".join(
                f"{pkg} ({int(now - started)}s)" for pkg, started in install_jobs
            )
            label = "install" if len(install_jobs) == 1 else "installs"
            lines.append(f"{len(install_jobs)} {label} in progress: {entries}")
        return "\n".join(lines)

    def _notify_learn_result(self, original_message: str, proposal: dict, result: dict) -> None:
        """Write a notification summarizing an async learn job."""
        engine = self._learned_engine()
        description = proposal.get("description") or "that"
        capability = result.get("capability") or proposal
        if result.get("saved"):
            summary = engine.summarize_result(original_message, capability, result)
            msg = f"Learned: {description}. {summary}"
            event_type = "learn_succeeded"
        elif result.get("missing_binary") and result.get("suggested_package"):
            msg = (
                f"Couldn't learn {description}: '{result.get('missing_binary')}' "
                f"isn't installed. Suggested package: "
                f"{result.get('suggested_package')}."
            )
            event_type = "learn_needs_install"
        elif result.get("missing_binary"):
            msg = (
                f"Couldn't learn {description}: '{result.get('missing_binary')}' "
                "isn't installed and I couldn't identify its apt package."
            )
            event_type = "learn_needs_install"
        else:
            err = result.get("error") or result.get("stderr") or "unknown error"
            msg = f"Couldn't learn {description}: {err}"
            event_type = "learn_failed"
        # Write to the events table so it surfaces via the existing
        # "any updates" router (which queries event_log.unread()).
        self.event_log.write(
            job_id=f"learn:{_learned_normalize(original_message)}:{int(time.time())}",
            event_type=event_type,
            message=msg[:1200],
            data={
                "original_message": original_message,
                "description": description,
                "saved": bool(result.get("saved")),
                "missing_binary": result.get("missing_binary"),
                "suggested_package": result.get("suggested_package"),
            },
        )

    def _attach_notification_alert(self, response: dict) -> dict:
        """Add a separate notification-alert field to the response when the
        event_log has unread async-work events. The main `message` (and
        `tts`) field is left untouched — the alert is its own thing in
        `notification_alert`, plus a count under `data.unread_async_events`,
        so the /ui client can render or speak it on a separate channel from
        the primary reply.

        Silenced on the responses that ARE about notifications (so reading
        updates doesn't carry an alert about itself)."""
        if not isinstance(response, dict):
            return response
        src = str(response.get("deterministic_source") or "")
        if src in {"code_monkey_updates", "learned_capability_async"}:
            return response
        try:
            unread = self.event_log.unread() or []
        except Exception:
            return response
        background_types = {
            "learn_succeeded", "learn_failed", "learn_needs_install",
            "install_succeeded", "install_failed",
            # World vault ingest health (written by the
            # luhkas-world-watchdog systemd timer).
            "world_ingest_stalled", "world_ingest_completed",
        }
        relevant = [e for e in unread if e.get("event_type") in background_types]
        count = len(relevant)
        if count <= 0:
            return response
        response = dict(response)
        response["notification_alert"] = (
            f"New notifications received ({count}). Say 'any updates' to read them."
        )
        response["data"] = {
            **(response.get("data") or {}),
            "unread_async_events": count,
        }
        return response

    def _learned_engine(self) -> LearnedCapabilityEngine:
        engine = getattr(self, "learned_capabilities", None)
        if engine is None:
            engine = LearnedCapabilityEngine()
            self.learned_capabilities = engine
        return engine

    PENDING_TTL_SECONDS = 300  # 5 minutes — abandoned confirmations auto-expire

    def _set_pending(self, value: dict | None, node_id: str | None = None) -> None:
        if isinstance(value, dict):
            # Tag with owner + TTL so abandoned confirmations can't trap
            # other sessions or linger forever.
            value = dict(value)
            value.setdefault("_node_id", node_id or value.get("node_id"))
            value["_expires_at"] = time.time() + self.PENDING_TTL_SECONDS
        if hasattr(self.blackboard, "set_pending_decision"):
            self.blackboard.set_pending_decision(value)
        else:
            self.blackboard.pending = value

    def _clear_pending(self) -> None:
        if hasattr(self.blackboard, "clear_pending_decision"):
            self.blackboard.clear_pending_decision()
        else:
            self.blackboard.pending = None

    def _get_pending(self, node_id: str | None = None) -> dict | None:
        """Read pending decision scoped to a node. Returns None if the
        pending was set by a different node OR if it has expired (in which
        case the expired pending is cleared as a side-effect).

        Pass node_id=None to bypass node scoping (still respects TTL) -- used
        by background sweeps that need to see the raw state."""
        raw = self.blackboard.get_pending_decision() if hasattr(self.blackboard, "get_pending_decision") else getattr(self.blackboard, "pending", None)
        if not isinstance(raw, dict):
            return raw
        expires_at = raw.get("_expires_at")
        if expires_at is not None and time.time() > expires_at:
            self._clear_pending()
            return None
        if node_id is not None:
            owner = raw.get("_node_id")
            if owner is not None and owner != node_id:
                return None
        return raw

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

    def _start_audit_caps(self, node_id: str) -> dict:
        """Build the duplicate-merge queue and present the first proposed
        merge. Sets pending state so subsequent yes/no/skip turns process
        the queue one pair at a time."""
        engine = self._learned_engine()
        pairs = engine.find_duplicate_caps()
        if not pairs:
            return {
                "mode": "direct",
                "message": "Audit complete. No duplicate or near-duplicate caps found.",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "audit_caps",
                "compose": False,
                "response_composed": True,
            }
        # Stash a compact serializable form of each pair on the blackboard
        # so it survives node-id transitions and re-loads of the engine.
        queue = []
        for pair in pairs:
            queue.append({
                "primary_key": pair["primary_key"],
                "dup_key": pair["dup_key"],
                "similarity_score": float(pair["similarity"]["score"]),
                "similarity_reason": pair["similarity"]["reason"],
                "kind": pair["kind"],
                "primary_description": pair["primary_cap"].get("description"),
                "dup_description": pair["dup_cap"].get("description"),
                "primary_hits": int(pair["primary_cap"].get("hits") or 0),
                "dup_hits": int(pair["dup_cap"].get("hits") or 0),
                "primary_execution": pair["primary_cap"].get("execution") or {},
                "dup_execution": pair["dup_cap"].get("execution") or {},
            })
        self._set_pending({
            "type": "audit_merge_confirmation",
            "queue": queue,
            "decisions": [],
            "node_id": node_id,
        })
        return {
            "mode": "direct",
            "message": self._format_audit_pair(queue, idx=0, total=len(queue)),
            "data": {"pending_pairs": len(queue)},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "audit_caps",
            "compose": False,
            "response_composed": True,
        }

    def _format_audit_pair(self, queue: list, idx: int, total: int) -> str:
        pair = queue[idx] if idx < len(queue) else None
        if pair is None:
            return "Audit complete."
        kind = pair.get("kind") or "near"
        sim = pair.get("similarity_score") or 0.0
        reason = pair.get("similarity_reason") or ""
        primary = pair["primary_execution"] or {}
        dup = pair["dup_execution"] or {}
        def render_exec(exe: dict) -> str:
            if exe.get("type") == "bash":
                return f"bash: {exe.get('command') or ''}"
            if exe.get("type") == "python_script":
                src = (exe.get("source") or "").strip()
                # Show full python source (up to 800 chars) so the user can
                # judge similarity directly.
                if len(src) > 800:
                    src = src[:797] + "..."
                return f"python_script:\n----\n{src}\n----"
            return repr(exe)
        progress = f"[{idx + 1}/{total}]"
        kind_label = "EXACT MATCH" if kind == "exact" else f"NEAR MATCH ({sim:.2f})"
        return (
            f"Audit {progress} — {kind_label} ({reason}).\n\n"
            f"PRIMARY (kept): {pair['primary_description']!r}  "
            f"[key={pair['primary_key']!r}, hits={pair['primary_hits']}]\n"
            f"  runs: {render_exec(primary)}\n\n"
            f"DUPLICATE (to merge into primary): {pair['dup_description']!r}  "
            f"[key={pair['dup_key']!r}, hits={pair['dup_hits']}]\n"
            f"  runs: {render_exec(dup)}\n\n"
            "Merge them? (yes / no / skip — skip leaves both as-is, no rejects "
            "this merge but keeps auditing.)"
        )

    def _handle_audit_merge_confirmation(self, message: str, node_id: str) -> dict | None:
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "audit_merge_confirmation":
            return None
        queue = list(pending.get("queue") or [])
        decisions = list(pending.get("decisions") or [])
        if not queue:
            self._clear_pending()
            return self._audit_complete_summary(decisions)

        text = _command_text(message)
        is_yes = _is_affirmative(message)
        is_no = _is_denial(message) or text in {"no", "skip", "next", "pass"}
        is_cancel = text in {"cancel", "stop audit", "abort", "abort audit", "quit"}

        if is_cancel:
            self._clear_pending()
            decisions.append({"action": "cancel", "remaining": len(queue)})
            return self._audit_complete_summary(decisions, cancelled=True)

        if not (is_yes or is_no):
            # Ambiguous — re-prompt without advancing.
            return self._remember_active({
                "mode": "direct",
                "message": (
                    "Please answer with yes (merge), no (keep both separate), "
                    "skip (move on), or cancel (stop the audit)."
                ),
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "audit_merge_confirmation",
                "compose": False,
                "response_composed": True,
            })

        engine = self._learned_engine()
        head = queue.pop(0)
        if is_yes:
            result = engine.merge_caps(head["primary_key"], head["dup_key"])
            decisions.append({
                "action": "merge",
                "primary": head["primary_description"],
                "dup": head["dup_description"],
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
            })
        else:
            decisions.append({
                "action": "skip",
                "primary": head["primary_description"],
                "dup": head["dup_description"],
            })

        if not queue:
            self._clear_pending()
            return self._audit_complete_summary(decisions)

        self._set_pending({
            "type": "audit_merge_confirmation",
            "queue": queue,
            "decisions": decisions,
            "node_id": node_id,
        })
        last = decisions[-1]
        prefix = (
            ("Merged." if last.get("ok") else f"Merge failed: {last.get('error')}.")
            if last.get("action") == "merge"
            else "Skipped."
        )
        body = self._format_audit_pair(queue, idx=0, total=len(queue) + len(decisions))
        return self._remember_active({
            "mode": "direct",
            "message": f"{prefix}\n\n{body}",
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "audit_merge_confirmation",
            "compose": False,
            "response_composed": True,
        })

    def _audit_complete_summary(self, decisions: list, cancelled: bool = False) -> dict:
        merged = sum(1 for d in decisions if d.get("action") == "merge" and d.get("ok"))
        failed = sum(1 for d in decisions if d.get("action") == "merge" and not d.get("ok"))
        skipped = sum(1 for d in decisions if d.get("action") == "skip")
        cancel_note = " (audit cancelled by user)" if cancelled else ""
        lines = [
            f"Audit complete{cancel_note}.",
            f"  merged: {merged}",
            f"  skipped: {skipped}",
        ]
        if failed:
            lines.append(f"  merge errors: {failed}")
        if merged:
            lines.append("\nMerged pairs:")
            for d in decisions:
                if d.get("action") == "merge" and d.get("ok"):
                    lines.append(f"  • {d.get('dup')!r} → {d.get('primary')!r}")
        return self._remember_active({
            "mode": "direct",
            "message": "\n".join(lines),
            "data": {"decisions": decisions},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "audit_caps_complete",
            "compose": False,
            "response_composed": True,
        })

    def _handle_memory_update_confirmation(self, message: str, node_id: str) -> dict | None:
        """One-turn handler for 'Oh, I thought you said X — should I update to Y?'
        prompts triggered when a new speaker-fact contradicts a stored one.

        - affirm → replace the old fact with the new fact in the speaker's namespace.
        - deny   → discard the new fact, keep the old one.
        - anything else → leave pending alive (TTL handles cleanup) and let
          the message route normally. We DON'T clear on neutral messages
          because the user might say something fact-related between the
          conflict prompt and their decision — clearing would silently
          forget the open question. The TTL on _set_pending caps how long
          we keep it.
        """
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "memory_update_confirmation":
            return None
        conflict = pending.get("conflict") or {}
        new_fact = conflict.get("new_fact") or ""
        old_fact = conflict.get("old_fact") or ""
        old_id = conflict.get("old_id") or ""

        store = getattr(self.scout, "memory_store", None)
        if store is None or not new_fact or not old_id:
            self._clear_pending()
            return None

        bridge = self.scout
        old_phrase = bridge._third_to_second_person(old_fact)
        new_phrase = bridge._third_to_second_person(new_fact)

        if _is_denial(message):
            self._clear_pending()
            return self._remember_active({
                "mode": "direct",
                "message": f"OK, I'll keep it as {old_phrase}.",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "memory_update_confirmation",
                "compose": False,
                "response_composed": True,
            })

        if not _is_affirmative(message):
            # Leave pending alive; let the message route normally so the user
            # can talk about something else without losing the open question.
            # TTL on the pending will reclaim it if abandoned.
            return None

        result = store.replace(
            old_id,
            new_fact,
            identity=conflict.get("identity"),
            unidentified_face_ref=conflict.get("unidentified_face_ref"),
            category="fact",
            source_message=conflict.get("source_message", ""),
        )
        self._clear_pending()
        if not result.get("ok"):
            return self._remember_active({
                "mode": "direct",
                "message": f"I couldn't update that ({result.get('error', 'unknown error')}).",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "memory_update_confirmation",
                "compose": False,
                "response_composed": True,
            })
        return self._remember_active({
            "mode": "direct",
            "message": f"Got it — updated. {new_phrase}.",
            "data": {"replaced": old_phrase, "now": new_phrase},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "memory_update_confirmation",
            "compose": False,
            "response_composed": True,
        })

    def _handle_learned_install_confirmation(self, message: str, node_id: str) -> dict | None:
        """One-turn handler for 'should I install <pkg>?' prompts.

        - 'yes' → install via apt, then retry learn_and_execute end-to-end.
        - 'no' / denial → clear pending, tell user nothing was installed.
        - any other input → clear pending and fall through.
        """
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "learned_install_confirmation":
            return None
        engine = self._learned_engine()
        package = pending.get("package") or ""

        if _is_denial(message):
            self._clear_pending()
            return self._remember_active({
                "mode": "direct",
                "message": f"OK, I won't install {package}. The capability wasn't learned.",
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_install_confirmation",
                "compose": False,
                "response_composed": True,
            })

        if not _is_affirmative(message):
            # Treat anything else as backing out — too risky to install on a vague reply.
            self._clear_pending()
            return None

        install_result = engine.install_package(package)
        if not install_result.get("ok"):
            self._clear_pending()
            err = (
                install_result.get("error")
                or install_result.get("stderr")
                or f"exited rc={install_result.get('returncode')}"
            )
            return self._remember_active({
                "mode": "direct",
                "message": f"Install of {package} failed: {err}",
                "data": {"install_result": install_result},
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_install_confirmation",
                "compose": False,
                "response_composed": True,
            })

        # Install succeeded — retry the learn flow end-to-end.
        original = pending.get("original_message") or ""
        proposal = pending.get("proposal") or {}
        self._clear_pending()
        result = engine.learn_and_execute(
            original,
            proposal,
            confirmed_by=f"user_confirmation_after_install:{package}",
        )
        capability = result.get("capability") or proposal
        summary = engine.summarize_result(original, capability, result)
        if result.get("saved"):
            summary = f"Installed {package} and learned the command. {summary}"
        else:
            summary = f"Installed {package} but the recipe still didn't work: {summary}"
        return self._remember_active({
            "mode": "direct",
            "message": summary,
            "data": {
                "install_result": install_result,
                "learned_capability": capability,
                "execution_result": result,
            },
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": f"learned_capability:{capability.get('name') or capability.get('intent')}",
            "compose": False,
            "response_composed": True,
        })

    def _handle_learned_execution_review(self, message: str, node_id: str) -> dict | None:
        """One-turn handler that runs immediately after a concept-match
        execution. If the user says 'no' or 'no, X' here, the freshly-saved
        alias is removed and a new propose flow is started using the
        correction text. Any other message clears the review state and falls
        through to normal handling."""
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "learned_execution_review":
            return None

        # LLM-driven intent classification — no hardcoded "is this a
        # correction" patterns. Fast-paths still handle plain yes/no.
        intent_info = self._resolve_pending_intent(
            message,
            previous_inferred=pending.get("executed_cap_inferred") or {},
            previous_description=pending.get("executed_cap_description"),
            original_message=pending.get("original_message") or "",
        )
        intent = intent_info.get("intent")

        if intent in {"affirm", "unrelated"}:
            # Affirm = user is happy with the cap; unrelated = moved on.
            # Either way, close the review window and stop intercepting.
            self._clear_pending()
            return None

        engine = self._learned_engine()
        alias_key = pending.get("alias_key") or ""
        removed = engine.store.forget(alias_key) if alias_key else False

        if intent == "correct":
            corrected_topic = intent_info.get("topic")
            corrected_aspect = intent_info.get("aspect")
            if corrected_topic is None:
                proposal = None
            else:
                proposal = engine._code_monkey_recipe_proposal(
                    corrected_topic,
                    corrected_aspect or engine._default_aspect_for(corrected_topic),
                )
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

        # If the corrected (topic, aspect) matches an existing cap, MOVE the
        # original phrase to that cap instead of starting a new propose flow.
        # This is the explicit "remove from bad command, add to good command"
        # behaviour.
        corrected_topic = ((proposal.get("inferred") or {}).get("topic") or "")
        corrected_aspect = ((proposal.get("inferred") or {}).get("aspect") or "")
        existing_cap = None
        if corrected_topic:
            for cap in engine.same_topic_caps(corrected_topic):
                ct, ca = engine._cap_concept(cap)
                if ct == corrected_topic and ca == corrected_aspect:
                    existing_cap = cap
                    break
        if existing_cap is not None:
            original_msg = pending.get("original_message") or message
            self._clear_pending()
            result = engine.execute_capability(existing_cap)
            if result.get("ok"):
                engine.record_alias(original_msg, existing_cap)
            summary = engine.summarize_result(original_msg, existing_cap, result)
            prefix = "Removed the wrong learned command and routed to" if removed else "Routed to"
            return self._remember_active({
                "mode": "direct",
                "message": f"{prefix} the existing {existing_cap.get('description')}. {summary}",
                "data": {
                    "learned_capability": existing_cap,
                    "execution_result": result,
                    "alias_recorded": result.get("ok"),
                    "alias_removed": removed,
                },
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": f"learned_capability:{existing_cap.get('name') or existing_cap.get('intent')}",
                "compose": False,
                "response_composed": True,
            })

        original_message = pending.get("original_message") or message
        self._set_pending({
            "type": "learned_capability_confirmation",
            "original_message": original_message,
            "proposal": proposal,
            "node_id": node_id,
            "correction": message,
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
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "learned_capability_confirmation":
            return None
        engine = self._learned_engine()
        if _is_affirmative(message):
            original = pending.get("original_message") or ""
            proposal = pending.get("proposal") or {}
            confirmed_by = str(message or "user_confirmation")
            self._clear_pending()
            # Learning is async: spawn the planner+smoke+save work in a
            # background thread and ack the user immediately. When the
            # background finishes it writes a notification to the event log
            # so subsequent /ui calls report "new notifications received".
            self._spawn_async_learn(
                original_message=original,
                proposal=proposal,
                confirmed_by=confirmed_by,
                node_id=node_id,
            )
            description = proposal.get("description") or "that"
            return self._remember_active({
                "mode": "direct",
                "message": f"I'll work on that ({description}). I'll let you know when it's ready.",
                "data": {"learning_async": True, "proposal": proposal},
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_capability_async",
                "compose": False,
                "response_composed": True,
            })

        # Legacy synchronous path (kept for fallback inspection; not used)
        if False and _is_affirmative(message):
            original = pending.get("original_message") or ""
            proposal = pending.get("proposal") or {}
            result = engine.learn_and_execute(
                original,
                proposal,
                confirmed_by=str(message or "user_confirmation"),
            )
            if not result.get("ok") and result.get("missing_binary"):
                missing = result.get("missing_binary")
                pkg = result.get("suggested_package")
                if pkg:
                    self._set_pending({
                        "type": "learned_install_confirmation",
                        "original_message": original,
                        "proposal": proposal,
                        "node_id": node_id,
                        "missing_binary": missing,
                        "package": pkg,
                    })
                    return self._remember_active({
                        "mode": "direct",
                        "message": (
                            f"I need the '{missing}' tool to do that, but it's not "
                            f"installed. I can install it via apt — that would run "
                            f"`sudo apt-get install -y {pkg}`. OK to install?"
                        ),
                        "data": {"missing_binary": missing, "suggested_package": pkg},
                        "active_task_id": self.active_task_id,
                        "deterministic": True,
                        "deterministic_source": "learned_install_confirmation",
                        "compose": False,
                        "response_composed": True,
                    })
                self._clear_pending()
                return self._remember_active({
                    "mode": "direct",
                    "message": (
                        f"I need the '{missing}' tool to do that, but it's not installed "
                        "and I'm not sure which apt package provides it. Install it manually "
                        "and I can try again."
                    ),
                    "data": {"missing_binary": missing},
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": "learned_capability_missing_tool",
                    "compose": False,
                    "response_composed": True,
                })
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
        # Non-affirmative response: ask the LLM whether the user is correcting
        # the pending proposal, denying it, or moving on. No hardcoded patterns
        # — the LLM has full context of what was proposed and what the user
        # originally asked for.
        previous_proposal = pending.get("proposal") or {}
        intent_info = self._resolve_pending_intent(
            message,
            previous_inferred=previous_proposal.get("inferred") or {},
            previous_description=previous_proposal.get("description"),
            original_message=pending.get("original_message") or "",
        )
        intent = intent_info.get("intent")
        if intent == "correct":
            corrected_topic = intent_info.get("topic")
            corrected_aspect = intent_info.get("aspect")
            if corrected_topic is None and corrected_aspect is None:
                proposal = None
            else:
                # Keep the previous topic when the correction only specified
                # an aspect, and vice versa.
                previous_inferred = previous_proposal.get("inferred") or {}
                use_topic = corrected_topic or previous_inferred.get("topic")
                use_aspect = (
                    corrected_aspect
                    or previous_inferred.get("aspect")
                    or engine._default_aspect_for(use_topic)
                )
                if not use_topic:
                    proposal = None
                else:
                    proposal = engine._code_monkey_recipe_proposal(use_topic, use_aspect)
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
            # If the corrected (topic, aspect) matches a cap we already have,
            # execute that directly and record the ORIGINAL message as an
            # alias of it — no need to learn a new cap.
            corrected_topic = ((proposal.get("inferred") or {}).get("topic") or "")
            corrected_aspect = ((proposal.get("inferred") or {}).get("aspect") or "")
            existing_cap = None
            if corrected_topic:
                for cap in engine.same_topic_caps(corrected_topic):
                    ct, ca = engine._cap_concept(cap)
                    if ct == corrected_topic and ca == corrected_aspect:
                        existing_cap = cap
                        break
            if existing_cap is not None:
                original_msg = pending.get("original_message") or message
                self._clear_pending()
                result = engine.execute_capability(existing_cap)
                if result.get("ok"):
                    engine.record_alias(original_msg, existing_cap)
                summary = engine.summarize_result(original_msg, existing_cap, result)
                return self._remember_active({
                    "mode": "direct",
                    "message": f"Got it — that's the existing {existing_cap.get('description')}. {summary}",
                    "data": {
                        "learned_capability": existing_cap,
                        "execution_result": result,
                        "alias_recorded": result.get("ok"),
                    },
                    "active_task_id": self.active_task_id,
                    "deterministic": True,
                    "deterministic_source": f"learned_capability:{existing_cap.get('name') or existing_cap.get('intent')}",
                    "compose": False,
                    "response_composed": True,
                })
            original_message = (
                pending.get("original_message")
                if engine.correction_updates_previous_request(proposal, previous_proposal)
                else message
            )
            self._set_pending({
                "type": "learned_capability_confirmation",
                "original_message": original_message,
                "proposal": proposal,
                "node_id": node_id,
                "correction": message,
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
        if intent == "deny":
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
        # "unrelated" or anything else — let normal routing take over.
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
            # Set a one-turn review state on EVERY successful learned-cap
            # execution, regardless of how the cap was matched (exact-key
            # hit OR concept-match-with-alias-recording). Without this the
            # user could only correct on a phrase's first use, not on
            # subsequent uses where lookup hit an existing key directly.
            if result.get("ok"):
                # Prefer the source cap's inferred (when concept-match), else
                # the learned cap's own inferred, else derive from intent.
                cap_inferred = (
                    (alias_source or {}).get("inferred")
                    or learned.get("inferred")
                    or {}
                )
                if not cap_inferred.get("topic"):
                    cap_topic, cap_aspect = engine._cap_concept(alias_source or learned or {})
                    if cap_topic:
                        cap_inferred = {"topic": cap_topic, "aspect": cap_aspect}
                self._set_pending({
                    "type": "learned_execution_review",
                    "original_message": message,
                    "alias_key": _learned_normalize(message),
                    "executed_cap_intent": learned.get("intent"),
                    "executed_cap_description": learned.get("description"),
                    "executed_cap_inferred": cap_inferred,
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
        # Surface existing same-topic alternatives so the user can redirect
        # to an existing cap instead of fragmenting the topic into a near-
        # duplicate. Common case: user says "available ram" -> LLM proposes
        # memory/hardware; we already have memory/usage. Tell them.
        proposed_topic = ((proposal.get("inferred") or {}).get("topic") or "")
        proposed_aspect = ((proposal.get("inferred") or {}).get("aspect") or "")
        same_topic = engine.same_topic_caps(proposed_topic) if proposed_topic else []
        # Don't list the proposed (topic, aspect) itself if a cap happens to
        # already match it (lookup_by_concept would have caught that, but
        # cheap to filter here).
        alternatives = [
            cap for cap in same_topic
            if engine._cap_concept(cap) != (proposed_topic, proposed_aspect)
        ]
        alt_descriptions = []
        seen_descs = set()
        for cap in alternatives[:3]:
            desc = cap.get("description") or cap.get("name") or ""
            if desc and desc not in seen_descs and desc != proposal.get("description"):
                alt_descriptions.append(desc)
                seen_descs.add(desc)
        if alt_descriptions:
            alt_clause = ", ".join(f"\"{d}\"" for d in alt_descriptions)
            message_text = (
                f"I think you mean {proposal['description']}. "
                f"You also already have {alt_clause}. "
                f"Is that right, or did you mean one of those?"
            )
        else:
            message_text = f"I think you mean {proposal['description']}. Is that right?"
        self._set_pending({
            "type": "learned_capability_confirmation",
            "original_message": message,
            "proposal": proposal,
            "node_id": node_id,
            "alternative_descriptions": alt_descriptions,
        })
        return self._remember_active(self._attach_learned_capability_update({
            "mode": "direct",
            "message": message_text,
            "data": {"proposal": proposal, "alternatives": alt_descriptions},
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
        response = self._attach_notification_alert(response)
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
        """Populate channel-independent rendering fields on every response.

        `message` is the rich text (markdown, citations); `tts` is the
        speech-clean version. Both are always emitted so multi-channel
        surfaces (e.g. a node with both a screen and a speaker) can
        render both. The node-side renderer is responsible for picking
        which channel(s) to use based on its own `has_display` /
        `has_audio` capabilities — vault does NOT clobber `message`
        with `tts` for audio-only nodes anymore. A node that genuinely
        cannot render text reads `tts` instead; one that has neither
        ignores both.

        Any alerts queued for this node (background-job results, ingest
        stalls, presence-triggered pushes) are drained and attached as
        `response['pending_alerts']` so request-response clients (web
        UIs, curl) get them on the same HTTP round-trip instead of
        having to separately poll /alerts/pending. The /alerts/pending
        endpoint remains for poll-only surfaces (presence proxies, the
        persistent chat daemon's between-input window); both code paths
        share the same per-node queue under a lock, so each alert is
        delivered exactly once."""
        message = response.get("message") or ""
        if message and response.get("compose", True) and not response.get("response_composed"):
            response["raw_message"] = message
            response["message"] = self._compose_runtime_message(response, node_id)
            response["response_composed"] = True
            message = response["message"]
        if response.get("response_composed") or "tts" not in response:
            response["tts"] = format_for_tts(message)
        response["has_display"] = self.node_registry.has_display(node_id)
        response["has_audio"] = self.node_registry.has_audio(node_id)
        response["node_id"] = node_id
        # Inline alerts captured at the top of handle_presence (see the
        # race-fix comment there). Consume once per request so deterministic
        # short-circuits don't show the same alerts twice if they happen
        # to traverse _enrich more than once.
        try:
            inline = getattr(self._inline_alerts_tls, "alerts", None) or []
            if inline:
                existing = response.get("pending_alerts") or []
                response["pending_alerts"] = list(existing) + list(inline)
                self._inline_alerts_tls.alerts = []
        except Exception:
            pass
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
