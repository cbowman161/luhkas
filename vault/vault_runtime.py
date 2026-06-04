from __future__ import annotations

import logging
import re
import subprocess
import threading
import time

log = logging.getLogger("vault.runtime")

from blackboard import Blackboard
from background_manager import BackgroundManager
from capability_registry import CapabilityRegistry
from chat_sessions import ChatSessionManager
from classroom import ClassroomController
from command_agent import CommandAgent
from config import DATA_DIR, INSTALLED_CAPABILITIES_DIR
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


_VAULT_EXPECTED_UNITS = [
    "vault-runtime.service",
    "code-monkey.service",
    "world-ingest-supervisor.service",
    "vault-autosync.timer",
    "luhkas-world-watchdog.timer",
    "world-compact.timer",
]


def _systemd_unit_status(unit: str) -> dict:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.5,
        )
    except Exception as exc:
        return {"ok": False, "active": False, "state": "unknown", "error": str(exc)}
    state = (result.stdout or result.stderr or "").strip() or "unknown"
    return {
        "ok": result.returncode == 0 and state == "active",
        "active": result.returncode == 0 and state == "active",
        "state": state,
    }


def _vault_own_services_status() -> dict:
    units = {unit: _systemd_unit_status(unit) for unit in _VAULT_EXPECTED_UNITS}
    down = [unit for unit, status in units.items() if not status.get("ok")]
    return {
        "ok": not down,
        "expected": list(_VAULT_EXPECTED_UNITS),
        "down": down,
        "units": units,
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

_LEARNED_STATUS_COMMANDS = {
    "learned status",
    "learning status",
    "learned growth status",
    "learning growth status",
    "learned commands status",
    "show learned status",
    "show learning status",
}

_LEARNED_FIX_COMMANDS = {
    "fix failed learned attempt",
    "fix failed learned attempts",
    "fix failed learned command",
    "fix failed learned commands",
    "retry failed learned attempt",
    "retry failed learned attempts",
    "retry failed learned command",
    "retry failed learned commands",
}

_LEARNED_INSTALL_MISSING_COMMANDS = {
    "install missing learned packages",
    "install missing learning packages",
    "install learned missing packages",
    "install missing packages for learned commands",
    "install missing packages for learning",
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


# Phase B: detect when the user expresses that the previous response
# was wrong/off-target — used to flag the most recent session as
# user-corrected so the learning aggregator can decay confidence on
# whatever it thinks it just "learned". Anchored at message start so
# tail occurrences like "not really, but tell me anyway" don't trigger.
_CORRECTION_OF_PREVIOUS_RE = re.compile(
    r"""
    ^\s*
    (?:
        no [,.!]? \s+ (?: that(?:'s|s|\s+is|\s+was)? \s+ (?: wrong | not | incorrect)
                       | i \s+ meant
                       | i \s+ asked
                       | i \s+ said
                       | not \s+ what )
      | that(?:'s|s|\s+is|\s+was)? \s+ (?: not \s+ (?: right | what | it | correct)
                                  | wrong
                                  | incorrect )
      | wrong (?: \s+ answer)?
      | incorrect
      | nope[,.! ]+ (?: that | i )
      | not \s+ quite
      | not \s+ what \s+ i
      | you(?:'re|re|\s+are) \s+ wrong
      | actually [,.]? \s+ (?: i \s+ meant | no | that(?:'s|s|\s+is|\s+was)?\s+not )
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_correction_of_previous(text: str) -> bool:
    if not text:
        return False
    return bool(_CORRECTION_OF_PREVIOUS_RE.search(str(text).strip()))


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
        # Phase 1A shadow tracking — observational only, no behavior
        # change. Sessions persist to data/chat_sessions/{node_id}.jsonl
        # for the learning aggregator (Layer 3) to consume later. Disable
        # via VAULT_CHAT_SESSIONS_ENABLE=0.
        self.chat_sessions = ChatSessionManager(DATA_DIR / "chat_sessions")
        # Per-node pending-decision state. Previously the Blackboard
        # held a single ``pending_decision`` slot; with two presence
        # nodes confirming capabilities concurrently the second writer
        # silently overwrote the first. Now vault-set pendings (every
        # type registered in ``_PENDING_HANDLERS``) live here, keyed by
        # node_id; the Blackboard slot is kept as a fallback for
        # router-set pendings (code_monkey_requirements etc.) which
        # don't know node_id today.
        self._node_pendings: dict[str, dict] = {}
        self._node_pendings_lock = threading.Lock()
        # Classroom mode controller. While a session has mode=classroom,
        # every user turn routes through this instead of the planner,
        # and the chat path is replaced with a pedagogy-aware prompt.
        # Lazy models — first lesson pays the load.
        self.classroom = ClassroomController(self.chat_sessions)
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
        # Last time a user message landed at handle_presence / handle. Read by
        # the world-ingest supervisor to decide whether the system is idle
        # enough to resume background ingestion. Updated atomically; zero
        # locking needed for a single float read/write.
        self._last_user_activity_at: float = 0.0

    def _touch_user_activity(self) -> None:
        self._last_user_activity_at = time.time()

    def handle(self, user_input, node_id: str = "cli"):
        user_input = (user_input or "").strip()
        self._touch_user_activity()
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

        cmd_response = self._try_runtime_command_dispatch(command_text, node_id)
        if cmd_response is not None:
            return cmd_response

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

        pending_result = self._try_pending_handler(user_input, node_id, pending=pending)
        if pending_result is not None:
            return pending_result

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

        # If the active session is in classroom mode, route the turn
        # through the controller. It handles exit/pause phrases
        # deterministically and otherwise calls the chat model with a
        # pedagogy-aware prompt. Returns None if no classroom session
        # is active, in which case we fall through to normal routing.
        try:
            classroom_response = self.classroom.maybe_handle_turn(user_input, node_id)
            if classroom_response is not None:
                classroom_response.setdefault("active_task_id", self.active_task_id)
                return self._remember_active(classroom_response)
        except Exception as exc:
            log.warning("classroom.maybe_handle_turn failed: %s", exc)

        # Deterministic command routing — zero LLM cost for known capability commands.
        cmd_response = self.command_agent.handle(user_input)
        if cmd_response is not None:
            cmd_response["active_task_id"] = self.active_task_id
            return self._remember_active(cmd_response)

        learned_response = self._handle_learned_capability_request(user_input, node_id)
        if learned_response is not None:
            return learned_response

        plan = self.planner.decide(user_input)

        # Intercept classroom subsystem before router.route, because the
        # controller needs node_id (which router doesn't currently
        # thread through) to operate on chat_sessions.
        if plan.get("subsystem") == "classroom":
            classroom_plan_response = self._dispatch_classroom_plan(
                plan, user_input, node_id,
            )
            if classroom_plan_response is not None:
                return self._remember_active(classroom_plan_response)

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
        except Exception as exc:
            # The notification is the user's last-resort signal in a
            # guard event; silent failure was hiding real misconfigs
            # (notify-send not installed, no $DISPLAY, dbus unreachable).
            log.warning("guard OS notify-send failed (nodes=%s): %s",
                        node_ids, exc)

    def handle_presence(self, message: str, node_id: str = "scout", presence_context: dict | None = None):
        """Route a presence/chat message through the scout bridge and return an
        enriched response with the same shape as handle()."""
        self._touch_user_activity()
        self.active_task_id = self._node_task_ids.get(node_id)
        self._current_node_id = node_id
        self._last_active_node_id = node_id
        # Phase B: if the user is correcting the previous response
        # ("no, that was wrong" / "incorrect" / "not what i asked"),
        # flag the most recent session so the eventual learning
        # aggregator can decay confidence on whatever it "learned".
        # Done BEFORE session bookkeeping so we mark the OUTGOING
        # session, not the new one we're about to open.
        if _is_correction_of_previous(message):
            self._safe_chat_sessions(
                "flag_last_wrong", self.chat_sessions.flag_last_wrong, node_id, message,
            )
        # Phase 1A: decide whether this message extends the active
        # session (the user is answering a pending question) or starts a
        # new one. Observational only — does not affect dispatch.
        try:
            active = self.chat_sessions.get_active(node_id)
            if active is not None and active.state == "awaiting":
                # User is responding to an outstanding prompt — extend
                # the active session by NOT opening a new one. The
                # response will be attached as another turn via
                # add_turn() at the end.
                pass
            elif active is not None and active.mode == "classroom":
                # Classroom sessions span many turns by design — never
                # close-and-reopen mid-lesson. The controller closes the
                # session explicitly on end/complete; for any other turn
                # we just extend.
                pass
            else:
                # Either no active session, or the previous one is no
                # longer waiting — close/park it and start fresh.
                identity = self.scout.active_identity if hasattr(self.scout, "active_identity") else None
                self.chat_sessions.open_session(
                    node_id=node_id,
                    identity=identity,
                    original_message=message,
                )
            # Lazy idle sweep — keeps stale sessions from lingering
            # without needing a background thread.
            self.chat_sessions.sweep_idle()
        except Exception as exc:
            log.warning("chat_sessions bookkeeping failed: %s", exc)
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
                log.info("flushed %d deferred alert(s) to %s", drained, node_id)
            inline = self.node_registry.pop_alerts(node_id)
        except Exception as exc:
            log.warning("flush_pending_to(%s) failed: %s", node_id, exc)
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
                # Phase 1B: session is source of truth for original input.
                message_for_scout = (
                    self._session_original_message(node_id)
                    or pending.get("original_message")
                    or message
                )
                self._resolve_pending(node_id, outcome={
                    "action": "vision_analysis_confirmed",
                    "result": {"original_message": message_for_scout,
                               "vision_summary": pending.get("summary")},
                    "learned": [],
                })
                presence_context = dict(presence_context or {})
                presence_context["force_full_vision"] = True
                message = message_for_scout
            elif _is_denial(message):
                self._resolve_pending(node_id, outcome={
                    "action": "vision_analysis_declined",
                    "result": {"original_message": pending.get("original_message")},
                    "learned": [],
                })
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
                self._resolve_pending(node_id, outcome={
                    "action": "vision_analysis_abandoned",
                    "result": {"original_message": pending.get("original_message"),
                               "user_said": message},
                    "learned": [],
                })
        # Clear any prior short-circuit marker on scout before calling.
        # hasattr is sufficient; redundant try/except was masking real errors.
        for attr in ("_stash_vision_short_circuit_marker", "_stash_memory_conflict_marker"):
            if hasattr(self.scout, attr):
                delattr(self.scout, attr)
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
            if hasattr(self.scout, "_stash_vision_short_circuit_marker"):
                delattr(self.scout, "_stash_vision_short_circuit_marker")
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
            if hasattr(self.scout, "_stash_memory_conflict_marker"):
                delattr(self.scout, "_stash_memory_conflict_marker")
        # Scout bridge stores the reply text in "response" and the input in "message"
        reply_text = result.get("response") or ""
        result["message"] = reply_text
        result["response_composed"] = True
        active_id = result.get("active_identity")
        self.node_registry.update_activity(node_id, identity=active_id)
        result = self._attach_notification_alert(result)
        # Phase 1A: attach this turn to the active session for diagnostics
        # / future learning. The turn_index is the index in scout's turns
        # deque (best-effort — scout owns the canonical list).
        try:
            turn_idx = len(getattr(self.scout, "turns", []) or []) - 1
            if turn_idx >= 0:
                self.chat_sessions.add_turn(node_id, turn_idx)
        except Exception as exc:
            log.warning("chat_sessions.add_turn failed: %s", exc)
        # Promotion-engine capture: record what the router picked for
        # this phrase. Becomes a silent_route candidate in the session's
        # outcome.learned at close time; the learning_promoter (future)
        # will threshold-count these and auto-promote high-confidence
        # (phrase → route) pairs to the deterministic_router. Skip the
        # routes where the phrase isn't meaningful learning material:
        # bare confirmations ("yes"), wakeword, and routing errors.
        try:
            route_info = result.get("route") or {}
            route_name = route_info.get("route")
            skip = {"wakeword", "routing_error"}
            msg_low = (message or "").strip().lower()
            if (
                route_name
                and route_name not in skip
                and not _is_affirmative(msg_low)
                and not _is_denial(msg_low)
                # Don't capture corrections as silent-route candidates —
                # they're noise that would teach the system to associate
                # "no that was wrong" with whatever route handled it.
                and not _is_correction_of_previous(message)
                and len(msg_low) >= 3
            ):
                self.chat_sessions.record_route(node_id, phrase=message, route=route_name)
        except Exception as exc:
            log.warning("chat_sessions.record_route failed: %s", exc)
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

        pending_result = self._try_pending_handler(message, node_id)
        if pending_result is not None:
            return pending_result

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

        # Active classroom session intercept — same as in handle(). Lets
        # exit phrases and on-topic chat run before any other deterministic
        # routing competes for these turns.
        try:
            classroom_response = self.classroom.maybe_handle_turn(message, node_id)
            if classroom_response is not None:
                classroom_response.setdefault("active_task_id", self.active_task_id)
                return self._remember_active(classroom_response)
        except Exception as exc:
            log.warning("classroom.maybe_handle_turn (presence) failed: %s", exc)

        cmd_response = self._try_runtime_command_dispatch(text, node_id)
        if cmd_response is not None:
            return cmd_response

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

        capability_response = self._handle_named_capability_command(text, node_id=node_id)
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
                "proposal": proposal,
                "saved": bool(result.get("saved")),
                "missing_binary": result.get("missing_binary"),
                "suggested_package": result.get("suggested_package"),
                "error": result.get("error") or result.get("stderr"),
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

    def _try_runtime_command_dispatch(self, text: str, node_id: str) -> dict | None:
        """Four direct command-set dispatches that were bit-identical in
        ``handle()`` and ``_handle_deterministic_presence_command``:
        updates, jobs, code-monkey health, audit-caps. Returns the
        response, or None when nothing matches. Lives here so the two
        callers can never drift apart.
        """
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
        if text in _LEARNED_STATUS_COMMANDS:
            return self._remember_active(self._learned_growth_status_response())
        if text in _LEARNED_FIX_COMMANDS:
            return self._remember_active(self._fix_failed_learned_attempts(node_id))
        if text in _LEARNED_INSTALL_MISSING_COMMANDS:
            return self._remember_active(self._install_missing_learned_packages(node_id))
        return None

    def _learned_growth_snapshot(self) -> dict:
        engine = self._learned_engine()
        data = engine.store.load()
        caps = data.get("capabilities") if isinstance(data, dict) else {}
        pending = data.get("pending_code_monkey") if isinstance(data, dict) else {}
        caps = caps if isinstance(caps, dict) else {}
        pending = pending if isinstance(pending, dict) else {}
        unread = []
        try:
            unread = self.event_log.unread() or []
        except Exception:
            unread = []
        with self._async_job_lock:
            active_learn = dict(self._active_learn_jobs)
            active_install = dict(self._active_install_jobs)
        final_failed = {"build_failed", "test_failed", "failed", "cancelled"}
        pending_tasks = {
            task_id: entry for task_id, entry in pending.items()
            if isinstance(entry, dict) and not entry.get("notified")
        }
        failed_tasks = {
            task_id: entry for task_id, entry in pending.items()
            if isinstance(entry, dict) and str(entry.get("state") or "") in final_failed
        }
        missing_events = [
            event for event in unread
            if event.get("event_type") == "learn_needs_install"
            and (event.get("data") or {}).get("suggested_package")
        ]
        failed_events = [
            event for event in unread
            if event.get("event_type") in {"learn_failed", "learn_needs_install"}
        ]
        root_caps = [
            (key, cap) for key, cap in caps.items()
            if isinstance(cap, dict) and not cap.get("alias_of")
        ]
        aliases = [
            (key, cap) for key, cap in caps.items()
            if isinstance(cap, dict) and cap.get("alias_of")
        ]
        root_caps.sort(key=lambda item: float(item[1].get("updated_at") or item[1].get("created_at") or 0), reverse=True)
        aliases.sort(key=lambda item: float(item[1].get("updated_at") or item[1].get("created_at") or 0), reverse=True)
        duplicate_count = 0
        try:
            duplicate_count = len(engine.find_duplicate_caps())
        except Exception:
            duplicate_count = 0
        return {
            "active_learn_jobs": active_learn,
            "active_install_jobs": active_install,
            "pending_tasks": pending_tasks,
            "failed_tasks": failed_tasks,
            "missing_package_events": missing_events,
            "failed_events": failed_events,
            "recent_learned": root_caps[:8],
            "recent_aliases": aliases[:8],
            "capability_count": len(caps),
            "alias_count": len(aliases),
            "duplicate_count": duplicate_count,
            "unread_events": unread,
        }

    def _learned_growth_status_response(self) -> dict:
        snapshot = self._learned_growth_snapshot()
        lines = [
            "Learned growth status:",
            f"  capabilities: {snapshot['capability_count']} ({snapshot['alias_count']} aliases)",
            f"  active learn jobs: {len(snapshot['active_learn_jobs'])}",
            f"  active installs: {len(snapshot['active_install_jobs'])}",
            f"  pending Code Monkey tasks: {len(snapshot['pending_tasks'])}",
            f"  failed learned attempts: {len(snapshot['failed_tasks'])}",
            f"  missing-package prompts: {len(snapshot['missing_package_events'])}",
            f"  duplicate/audit candidates: {snapshot['duplicate_count']}",
        ]
        if snapshot["active_learn_jobs"]:
            lines.append("\nActive learning:")
            for desc, started in list(snapshot["active_learn_jobs"].items())[:5]:
                lines.append(f"  - {desc} ({int(time.time() - started)}s)")
        if snapshot["failed_tasks"]:
            lines.append("\nFailed attempts:")
            for task_id, entry in list(snapshot["failed_tasks"].items())[:5]:
                lines.append(f"  - {entry.get('input') or task_id}: {entry.get('state')}")
        if snapshot["missing_package_events"]:
            lines.append("\nMissing packages:")
            for event in snapshot["missing_package_events"][:5]:
                data = event.get("data") or {}
                lines.append(
                    f"  - {data.get('suggested_package')} for {data.get('missing_binary')} "
                    f"({data.get('description') or data.get('original_message')})"
                )
        if snapshot["recent_learned"]:
            lines.append("\nRecently learned:")
            for key, cap in snapshot["recent_learned"][:5]:
                lines.append(f"  - {key}: {cap.get('description')}")
        if snapshot["recent_aliases"]:
            lines.append("\nRecent aliases:")
            for key, cap in snapshot["recent_aliases"][:5]:
                lines.append(f"  - {key} -> {cap.get('alias_of')}")
        lines.append("\nCommands: 'fix failed learned attempts' or 'install missing learned packages'.")
        return {
            "mode": "direct",
            "message": "\n".join(lines),
            "data": snapshot,
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "learned_growth_status",
            "compose": False,
            "response_composed": True,
        }

    def _fix_failed_learned_attempts(self, node_id: str) -> dict:
        snapshot = self._learned_growth_snapshot()
        retry_items = []
        seen_inputs = set()
        for task_id, entry in snapshot["failed_tasks"].items():
            if not isinstance(entry, dict):
                continue
            original = entry.get("input") or ""
            proposal = entry.get("proposal") or {}
            if original and isinstance(proposal, dict) and proposal and original not in seen_inputs:
                retry_items.append((original, proposal, task_id))
                seen_inputs.add(original)
        for event in snapshot["failed_events"]:
            data = event.get("data") or {}
            original = data.get("original_message") or ""
            proposal = data.get("proposal") or {}
            if original and isinstance(proposal, dict) and proposal and original not in seen_inputs:
                retry_items.append((original, proposal, event.get("job_id")))
                seen_inputs.add(original)
        if not retry_items:
            return {
                "mode": "direct",
                "message": "I do not see any failed learned attempts with enough stored context to retry.",
                "data": {"retried": 0},
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_growth_fix",
                "compose": False,
                "response_composed": True,
            }
        for original, proposal, _source in retry_items[:5]:
            self._spawn_async_learn(
                original_message=original,
                proposal=proposal,
                confirmed_by="user_requested_failed_learn_retry",
                node_id=node_id,
            )
        return {
            "mode": "direct",
            "message": f"I restarted {min(len(retry_items), 5)} failed learned attempt(s). Say 'any updates' for progress.",
            "data": {"retried": [item[0] for item in retry_items[:5]], "available": len(retry_items)},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "learned_growth_fix",
            "compose": False,
            "response_composed": True,
        }

    def _install_missing_learned_packages(self, node_id: str) -> dict:
        snapshot = self._learned_growth_snapshot()
        packages = []
        for event in snapshot["missing_package_events"]:
            pkg = str((event.get("data") or {}).get("suggested_package") or "").strip().lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9+\-.]{0,63}", pkg) and pkg not in packages:
                packages.append(pkg)
        if not packages:
            return {
                "mode": "direct",
                "message": "I do not see any learned attempts waiting on an identifiable missing package.",
                "data": {"installed": 0},
                "active_task_id": self.active_task_id,
                "deterministic": True,
                "deterministic_source": "learned_growth_install_missing",
                "compose": False,
                "response_composed": True,
            }
        started = []
        with self._async_job_lock:
            active = set(self._active_install_jobs)
        for package in packages[:5]:
            if package in active:
                continue
            self._spawn_async_install(package, node_id)
            started.append(package)
        return {
            "mode": "direct",
            "message": (
                f"I started installing {', '.join(started)} for failed learned attempts. "
                "Say 'any updates' for progress."
                if started
                else "Those missing packages are already being installed."
            ),
            "data": {"packages": packages, "started": started},
            "active_task_id": self.active_task_id,
            "deterministic": True,
            "deterministic_source": "learned_growth_install_missing",
            "compose": False,
            "response_composed": True,
        }

    # Maps pending-state types to their handler method name and whether
    # the handler's result should be wrapped with _remember_active before
    # returning. Used by _try_pending_handler so handle() and
    # _handle_deterministic_presence_command don't need to duplicate the
    # cascade. To add a new pending type, register it here and write the
    # handler — both code paths pick it up automatically.
    _PENDING_HANDLERS: dict[str, tuple[str, bool]] = {
        "learned_capability_confirmation": ("_handle_learned_capability_confirmation", False),
        "learned_execution_review":        ("_handle_learned_execution_review", False),
        "learned_install_confirmation":    ("_handle_learned_install_confirmation", False),
        "audit_merge_confirmation":        ("_handle_audit_merge_confirmation", False),
        "memory_update_confirmation":      ("_handle_memory_update_confirmation", False),
        "classroom_subject_prompt":        ("_handle_classroom_subject_prompt", True),
    }

    def _try_pending_handler(self, message: str, node_id: str,
                              *, pending: dict | None = None) -> dict | None:
        """Run the registered handler for the current pending state, if
        any. Returns the handler's response (optionally wrapped with
        _remember_active per the registry), or None when no pending
        state is active OR the handler chooses to no-op.

        ``pending`` is accepted as a parameter so callers that already
        looked it up (handle()) don't pay a second _get_pending call.
        """
        if pending is None:
            pending = self._get_pending(node_id)
        if not isinstance(pending, dict):
            return None
        entry = self._PENDING_HANDLERS.get(pending.get("type"))
        if entry is None:
            return None
        handler_name, wrap = entry
        handler = getattr(self, handler_name, None)
        if handler is None:
            return None
        result = handler(message, node_id)
        if result is None:
            return None
        return self._remember_active(result) if wrap else result

    def _learned_engine(self) -> LearnedCapabilityEngine:
        engine = getattr(self, "learned_capabilities", None)
        if engine is None:
            engine = LearnedCapabilityEngine()
            self.learned_capabilities = engine
        return engine

    PENDING_TTL_SECONDS = 300  # 5 minutes — abandoned confirmations auto-expire

    def _safe_chat_sessions(self, label: str, fn, *args, **kwargs):
        """Wrap chat_sessions calls in one error-logging boundary.

        chat_sessions is observational; if it fails we don't want to break
        the main flow — but we also don't want to silently lose visibility
        into what broke. Logs to stderr (journalctl picks it up).
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.warning("chat_sessions.%s failed: %s", label, exc)
            return None

    def _set_pending(self, value: dict | None, node_id: str | None = None) -> None:
        if isinstance(value, dict):
            # Tag with owner + TTL so abandoned confirmations can't trap
            # other sessions or linger forever.
            value = dict(value)
            value.setdefault("_node_id", node_id or value.get("node_id"))
            value["_expires_at"] = time.time() + self.PENDING_TTL_SECONDS
        nid = node_id or (value.get("_node_id") if isinstance(value, dict) else None) or self._current_node_id
        if nid:
            # Per-node slot — survives concurrent confirmations from
            # other nodes without race-overwriting.
            with self._node_pendings_lock:
                if value is None:
                    self._node_pendings.pop(nid, None)
                else:
                    self._node_pendings[nid] = value
        else:
            # Caller couldn't supply a node_id (background sweeps, CLI
            # without context). Fall back to the legacy single slot —
            # only safe when at most one such caller is active.
            self.blackboard.set_pending_decision(value)
        # Shadow-mirror to chat_session.awaiting so the session record
        # carries the same prompt the user is being asked.
        try:
            if nid:
                self.chat_sessions.set_awaiting(nid, value)
        except Exception as exc:
            log.warning("chat_sessions.set_awaiting (set) failed: %s", exc)

    def _clear_pending(self) -> None:
        nid = self._current_node_id
        if nid:
            with self._node_pendings_lock:
                self._node_pendings.pop(nid, None)
        # Also clear the legacy single slot only when it belongs to this node.
        # Router-set pendings can still live there without node ownership; a
        # node-local confirmation must not silently erase another flow.
        blackboard_pending = self.blackboard.get_pending_decision()
        if isinstance(blackboard_pending, dict):
            pending_node_id = blackboard_pending.get("_node_id")
            if (nid and pending_node_id == nid) or (not nid and not pending_node_id):
                self.blackboard.clear_pending_decision()
        elif not nid and blackboard_pending is not None:
            self.blackboard.clear_pending_decision()
        # Just clear the awaiting prompt — does NOT close the session.
        # Use _resolve_pending(node_id, outcome) when the confirmation
        # actually resolved and the session should close with a record.
        try:
            if nid:
                self.chat_sessions.set_awaiting(nid, None)
        except Exception as exc:
            log.warning("chat_sessions.set_awaiting (clear) failed: %s", exc)

    # ---- Phase 1B: session as source of truth -------------------------

    def _session_original_message(self, node_id: str | None) -> str | None:
        """Return the active session's original_message (the user's first
        input in this conversation thread). This is the durable source
        of truth — pending dicts hold the same value but only for as
        long as the 5-minute TTL.
        """
        if not node_id:
            return None
        try:
            session = self.chat_sessions.get_active(node_id)
            return session.original_message if session else None
        except Exception:
            return None

    def _resolve_pending(self, node_id: str | None, outcome: dict | None = None) -> None:
        """Clear pending dict AND close the active session with a
        structured outcome. Confirmation handlers should call this
        instead of ``_clear_pending`` when their resolution is final
        (yes/no/correct accepted, action executed). The outcome becomes
        the session's canonical record of what got learned.
        """
        self._clear_pending()
        try:
            if node_id:
                self.chat_sessions.close(node_id, outcome=outcome)
        except Exception as exc:
            log.warning("chat_sessions.close failed: %s", exc)

    def _get_pending(self, node_id: str | None = None) -> dict | None:
        """Read pending decision scoped to a node. Checks the per-node
        store first, then falls back to the Blackboard's single slot
        (which holds router-set pendings — code_monkey_requirements
        etc. — that don't currently know node_id).

        Returns None if the pending was set by a different node OR if
        it has expired (in which case the expired pending is cleared
        as a side-effect).

        Pass node_id=None to bypass node scoping (still respects TTL);
        used by background sweeps that need to see the raw state.
        """
        raw: dict | None = None
        if node_id is not None:
            with self._node_pendings_lock:
                raw = self._node_pendings.get(node_id)
        if raw is None:
            raw = self.blackboard.get_pending_decision()
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
            self._resolve_pending(node_id, outcome={
                "action": "memory_update_unavailable",
                "result": {"reason": "no_store_or_missing_fact"},
                "learned": [],
            })
            return None

        bridge = self.scout
        old_phrase = bridge._third_to_second_person(old_fact)
        new_phrase = bridge._third_to_second_person(new_fact)
        original = self._session_original_message(node_id) or pending.get("original_message")

        if _is_denial(message):
            self._resolve_pending(node_id, outcome={
                "action": "memory_update_declined",
                "result": {"original_message": original, "kept": old_fact,
                           "rejected": new_fact},
                "learned": [],
            })
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
        self._resolve_pending(node_id, outcome={
            "action": "memory_update_confirmed" if result.get("ok") else "memory_update_failed",
            "result": {"original_message": original, "replaced": old_fact,
                       "now": new_fact, "error": result.get("error")},
            "learned": [{"kind": "fact_replacement", "old": old_fact, "new": new_fact}]
                       if result.get("ok") else [],
        })
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

        if self._should_bypass_learned_execution_review(message):
            self._clear_pending()
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

    def _should_bypass_learned_execution_review(self, message: str) -> bool:
        """Return True when a review-window turn is itself a learned command.

        Learned-cap executions open a one-turn correction window. Without this
        guard, a user who asks another learned command immediately after the
        first can accidentally have the second phrase interpreted as a
        correction of the first, causing alias churn in the store.
        """
        if _is_affirmative(message) or _is_denial(message):
            return False
        text = _command_text(message)
        if (
            _is_correction_of_previous(message)
            or text.startswith("actually ")
            or text.startswith("no ")
            or text.startswith("nope ")
            or text.startswith("nah ")
            or text.startswith("wrong ")
            or text.startswith("not ")
        ):
            return False
        engine = self._learned_engine()
        key = _learned_normalize(message)
        caps = (engine.store.load().get("capabilities") or {})
        if key and isinstance(caps.get(key), dict):
            return True
        return engine.lookup_by_concept(message) is not None

    def _handle_learned_capability_confirmation(self, message: str, node_id: str) -> dict | None:
        pending = self._get_pending(node_id)
        if not isinstance(pending, dict) or pending.get("type") != "learned_capability_confirmation":
            return None
        engine = self._learned_engine()
        if _is_affirmative(message):
            # Phase 1B: prefer the session's original_message over the
            # pending dict's. The session value survives the 5-minute
            # pending TTL; the pending value can age out mid-confirmation
            # if the user takes too long to reply.
            original = (
                self._session_original_message(node_id)
                or pending.get("original_message")
                or ""
            )
            proposal = pending.get("proposal") or {}
            confirmed_by = str(message or "user_confirmation")
            # _resolve_pending closes the session with a structured
            # outcome — the canonical record of what got learned in
            # this thread. The async learn happens after, but the
            # session is closed now with the proposal info; Layer 3
            # (learning aggregator) can correlate async completions
            # back to this session via original_message later.
            self._resolve_pending(node_id, outcome={
                "action": "learned_capability_confirmed",
                "result": {
                    "original_message": original,
                    "proposal": proposal,
                    "confirmed_by": confirmed_by,
                    "async": True,
                },
                "learned": [{"kind": "pending_capability", "phrase": original,
                             "canonical": proposal.get("description")}],
            })
            # Spawn the planner+smoke+save work in a background thread
            # and ack the user immediately. When the background finishes
            # it writes a notification to the event log so subsequent
            # /ui calls report "new notifications received".
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
                else intent_info.get("correction_text") or message
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

    def _handle_classroom_subject_prompt(self, message: str, node_id: str) -> dict | None:
        """Resolve a ``classroom_subject_prompt`` pending state by handing
        the user's reply to the controller as the lesson subject. The
        controller cancels gracefully on cancel-words, or routes to
        ``start_lesson`` otherwise. Always clears the pending state."""
        pending = self._get_pending(node_id)
        if not (isinstance(pending, dict) and pending.get("type") == "classroom_subject_prompt"):
            return None
        identity = self.scout.active_identity if hasattr(self.scout, "active_identity") else None
        # Clear the pending FIRST so a failed start_lesson doesn't leave
        # the user stuck in the prompt.
        self._resolve_pending(node_id, outcome={
            "action": "classroom_subject_resolved",
            "result": {"subject_text": message[:200]},
            "learned": [],
        })
        try:
            response = self.classroom.resolve_subject_prompt(message, node_id, identity=identity)
        except Exception as exc:
            log.warning("classroom resolve_subject_prompt failed: %s", exc)
            return {
                "mode": "direct",
                "message": f"Classroom subject prompt failed: {exc}",
                "active_task_id": self.active_task_id,
            }
        response.setdefault("active_task_id", self.active_task_id)
        response.setdefault("deterministic_source", "classroom:subject_prompt_resolved")
        return response

    def _dispatch_classroom_plan(self, plan: dict, user_input: str,
                                  node_id: str) -> dict | None:
        """Route a planner decision whose subsystem is "classroom" to the
        ClassroomController. Returns the chat-shaped response, or None if
        the action isn't recognized (caller falls back to router.route)."""
        action = plan.get("action") or ""
        try:
            if action == "start":
                response = self.classroom.start_lesson(user_input, node_id)
            elif action == "open":
                response = self.classroom.prompt_for_subject(node_id)
            elif action == "resume":
                response = self.classroom.resume_lesson(node_id)
            elif action == "end":
                response = self.classroom.end_lesson(node_id)
            else:
                return None
        except Exception as exc:
            log.warning("classroom dispatch (%s) failed: %s", action, exc)
            return {
                "mode": "direct",
                "message": f"Classroom action '{action}' failed: {exc}",
                "active_task_id": self.active_task_id,
            }
        # If the controller asked for a pending state (subject_prompt
        # flow), install it so the next user turn routes back through
        # the classroom resolver.
        pending = response.pop("pending", None)
        if isinstance(pending, dict):
            self._set_pending(pending, node_id=node_id)
        response.setdefault("active_task_id", self.active_task_id)
        response.setdefault("deterministic_source", f"classroom:{action}")
        return response

    def _handle_named_capability_command(self, text: str, node_id: str | None = None) -> dict | None:
        # O(1) alias lookup via the registry's precomputed index — was
        # an O(N × alias-count) linear scan per presence turn.
        capability = self.registry.lookup_by_alias(text)
        if capability is not None:
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
            if subsystem == "classroom":
                if node_id is None:
                    node_id = self._last_active_node_id or "cli"
                fake_plan = {"subsystem": "classroom", "action": action}
                return self._dispatch_classroom_plan(fake_plan, text, node_id)
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

    def health(self):
        try:
            code_monkey = self.router.code_monkey.health()
        except Exception as exc:
            code_monkey = {
                "ok": False,
                "error": str(exc),
            }

        own_services = _vault_own_services_status()
        last_user = float(self._last_user_activity_at or 0.0)

        # Lightweight ingestion liveness for the kiosk display (visual
        # cue: yellow outer ring + faster spin when active). Pulled
        # from ingest_admin so detection covers both the manual runner
        # and the always-on supervisor service. Wrapped in try/except
        # so a libzim/state-file glitch can never break /health.
        ingestion_running = False
        try:
            from world import ingest_admin
            ingestion_running, _ = ingest_admin._process_running()
        except Exception:
            pass

        # Time of the last Ollama dispatch (chat, embed, generate). The
        # wiki-ingest supervisor uses this — not `last_user_activity_at`
        # — to decide when to pause: VRAM contention only matters when
        # an actual GPU model is running, so deterministic routes that
        # don't invoke Ollama no longer pause ingest unnecessarily.
        last_ollama = 0.0
        try:
            import models as _models
            last_ollama = float(_models.get_last_ollama_activity_at() or 0.0)
        except Exception:
            pass

        return {
            "ok": bool(own_services.get("ok")) and bool(code_monkey.get("ok", True)),
            "service": "vault_runtime",
            "presence_owner": "vault_pc",
            "active_task_id": self.active_task_id,
            "models": model_manifest(),
            "model_warmup": self.model_warmup,
            "code_monkey": code_monkey,
            "own_services": own_services,
            "services_down": own_services.get("down", []),
            "scout": {
                "url": self.scout.scout_url,
                "active_identity": self.scout.active_identity,
            },
            "ingestion": {"running": ingestion_running},
            # Used by the world-ingest supervisor (and useful for debugging
            # idle detection generally). 0 means no user message seen yet.
            "last_user_activity_at": last_user,
            "seconds_since_user_activity": (time.time() - last_user) if last_user > 0 else None,
            # Precise GPU-contention signal: timestamp of the last Ollama
            # call from this process. 0 means no LLM/embed activity yet.
            "last_ollama_activity_at": last_ollama,
            "seconds_since_ollama_activity": (time.time() - last_ollama) if last_ollama > 0 else None,
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
