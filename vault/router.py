from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

from agents.system_agent import SystemAgent
from agents.chat_agent import ChatAgent
from agents.requirements_agent import RequirementsAgent
from agents.review_agent import ReviewAgent
from capability_builder import CapabilityBuilder
from config import DATA_DIR, INSTALLED_CAPABILITIES_DIR
from executor import execute_action
from code_monkey_client import CodeMonkeyClient

_NAMES_FILE = DATA_DIR / "code_monkey_names.json"

_FINAL_CODE_MONKEY_STATES = {"verified", "build_failed", "test_failed", "failed", "cancelled"}


def _load_names() -> dict:
    try:
        if _NAMES_FILE.exists():
            return json.loads(_NAMES_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_names(names: dict) -> None:
    try:
        _NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _NAMES_FILE.write_text(json.dumps(names, indent=2))
    except Exception:
        pass

class Router:
    def __init__(self, blackboard, event_log, job_manager, registry, skill_registry):
        self.blackboard = blackboard
        self.event_log = event_log
        self.job_manager = job_manager
        self.registry = registry
        self.skill_registry = skill_registry

        self.system_agent = SystemAgent(registry)
        self.chat_agent = ChatAgent()
        self.code_monkey = CodeMonkeyClient()
        self.capability_builder = CapabilityBuilder(registry)
        self.requirements_agent = RequirementsAgent()
        self.review_agent = ReviewAgent()

    def route(self, plan, user_input, active_task_id=None):
        pending = self.blackboard.get_pending_decision()

        if pending:
            return self.resolve_pending_decision(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        intent = plan.get("intent")
        subsystem = plan.get("subsystem")

        if intent == "use_skill":
            return self.confirm_existing_skill(plan, user_input, active_task_id)

        if intent == "evolve_skill":
            return self.propose_skill_evolution(plan, user_input, active_task_id)

        if intent == "evolve_capability":
            return self.propose_capability_evolution(plan, user_input, active_task_id)

        if intent == "create_capability":
            return self.propose_capability(user_input, active_task_id)

        capability = self.registry.get(plan.get("capability"))

        if subsystem == "event_log":
            return self.show_updates(active_task_id)

        if subsystem == "job_manager":
            return self.show_jobs(active_task_id)

        if subsystem == "system_agent":
            result = self.system_agent.run_direct(
                plan.get("action"),
                capability=capability,
            )

            return {
                "mode": "direct",
                "message": result["message"],
                "data": result,
                "active_task_id": active_task_id,
            }

        if subsystem == "chat_agent":
            message = self.chat_agent.answer(user_input)

            return {
                "mode": "direct",
                "message": message,
                "active_task_id": active_task_id,
            }

        if subsystem == "code_monkey":
            return self.start_code_monkey_requirements(
                user_input=user_input,
                active_task_id=active_task_id,
            )

        return {
            "mode": "direct",
            "message": "I could not route that request.",
            "active_task_id": active_task_id,
        }

    # ----------------------------
    # PENDING DECISIONS
    # ----------------------------

    def confirm_existing_skill(self, plan, user_input, active_task_id=None):
        skill = self.skill_registry.get(plan.get("skill"))

        if not skill:
            return {
                "mode": "direct",
                "message": "I thought a matching skill existed, but I could not find it.",
                "active_task_id": active_task_id,
            }

        self.blackboard.set_pending_decision({
            "type": "existing_skill",
            "skill": skill.get("name"),
            "filename": skill.get("filename"),
            "description": skill.get("description"),
            "original_request": user_input,
            "active_task_id": active_task_id,
        })

        self.set_last_skill(skill.get("name"), skill.get("filename"))

        return {
            "mode": "direct",
            "message": (
                f"I already have a skill for that:\n"
                f"- Skill: {skill.get('name')}\n"
                f"- File: {skill.get('filename')}\n"
                f"- Description: {skill.get('description')}\n\n"
                f"Do you want me to run it, modify it, overwrite it, or leave it as-is?"
            ),
            "active_task_id": active_task_id,
        }
    def resolve_run_skill_args(self, pending, user_input, active_task_id=None):
        import shlex

        filename = pending.get("filename")
        skill_name = pending.get("skill")
        args = user_input.strip()

        self.blackboard.clear_pending_decision()

        if not filename:
            return {
                "mode": "direct",
                "message": f"Skill {skill_name} has no filename to run.",
                "active_task_id": active_task_id,
            }

        if args.lower() in {"default", "none", "no args", "no arguments", "nothing"}:
            command = f"python3 {filename}"
        else:
            command = f"python3 {filename} {shlex.quote(args)}"

        result = execute_action({
            "type": "command",
            "path": None,
            "content": command,
        })

        output = result.get("stdout") or result.get("stderr") or result.get("error") or ""

        return {
            "mode": "direct",
            "message": output or f"Ran skill: {skill_name}",
            "data": result,
            "active_task_id": active_task_id,
        }

    def resolve_pending_decision(self, pending, user_input, active_task_id=None, interpretation=None):
        decision_type = pending.get("type")

        if decision_type == "existing_skill":
            return self.resolve_existing_skill_decision(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
                interpretation=interpretation,
            )

        if decision_type == "modify_skill_details":
            return self.resolve_modify_skill_details(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "skill_evolution":
            return self.resolve_skill_evolution(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
                interpretation=interpretation,
            )

        if decision_type == "capability_evolution":
            return self.resolve_capability_evolution(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "run_skill_args":
            return self.resolve_run_skill_args(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "code_monkey_pick_review":
            return self.resolve_pick_review(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "code_monkey_overlap_decision":
            return self.resolve_overlap_decision(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "code_monkey_requirements":
            return self.resolve_code_monkey_requirements(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        if decision_type == "code_monkey_review":
            return self.resolve_code_monkey_review(
                pending=pending,
                user_input=user_input,
                active_task_id=active_task_id,
            )

        self.blackboard.clear_pending_decision()

        return {
            "mode": "direct",
            "message": "Cleared unknown pending decision.",
            "active_task_id": active_task_id,
        }

    def resolve_existing_skill_decision(self, pending, user_input, active_task_id=None, interpretation=None):
        choice = self.normalize_pending_choice(user_input, interpretation)

        if choice == "run":
            return self.run_pending_skill(pending, active_task_id)

        if choice == "modify":
            return self.ask_how_to_modify_skill(pending, active_task_id)

        if choice == "overwrite":
            return self.ask_how_to_overwrite_skill(pending, active_task_id)

        if choice == "leave":
            self.blackboard.clear_pending_decision()

            return {
                "mode": "direct",
                "message": "Okay, leaving the existing skill unchanged.",
                "active_task_id": active_task_id,
            }

        return {
            "mode": "direct",
            "message": (
                "I’m waiting on that skill decision. Please say: "
                "run it, modify it, overwrite it, or leave it."
            ),
            "active_task_id": active_task_id,
        }

    def normalize_pending_choice(self, user_input, interpretation=None):
        if interpretation:
            intent = interpretation.get("intent")

            if intent == "run_existing_skill":
                return "run"

            if intent == "modify_skill":
                return "modify"

            if intent == "overwrite_skill":
                return "overwrite"

            if intent == "leave_skill":
                return "leave"

            if interpretation.get("kind") == "cancel":
                return "leave"

        text = user_input.lower().strip()

        if text in {
            "run",
            "run it",
            "execute",
            "execute it",
            "start it",
            "use it",
            "yes run it",
            "go ahead and run it",
        } or text.startswith("run "):
            return "run"

        if text in {
            "modify",
            "modify it",
            "change",
            "change it",
            "edit",
            "edit it",
            "update",
            "update it",
            "improve",
            "improve it",
        } or text.startswith(("modify ", "change ", "edit ", "update ", "improve ")):
            return "modify"

        if text in {
            "overwrite",
            "overwrite it",
            "replace",
            "replace it",
            "redo",
            "redo it",
            "rewrite",
            "rewrite it",
        } or text.startswith(("overwrite ", "replace ", "redo ", "rewrite ")):
            return "overwrite"

        if text in {
            "leave",
            "leave it",
            "leave it as-is",
            "leave as-is",
            "keep",
            "keep it",
            "nothing",
            "cancel",
            "never mind",
            "nevermind",
        }:
            return "leave"

        return None

    # ----------------------------
    # MODIFY / OVERWRITE DETAILS
    # ----------------------------

    def ask_how_to_modify_skill(self, pending, active_task_id=None):
        self.blackboard.set_pending_decision({
            **pending,
            "type": "modify_skill_details",
            "operation": "modify",
        })

        return {
            "mode": "direct",
            "message": (
                f"How do you want to modify `{pending.get('skill')}`?\n\n"
                f"Suggested changes:\n"
                f"1. Change what it prints\n"
                f"2. Print the message multiple times\n"
                f"3. Add comments or a function wrapper\n"
                f"4. Rename or clean up the skill\n\n"
                f"Tell me what you want, or pick 1-4."
            ),
            "active_task_id": active_task_id,
        }

    def ask_how_to_overwrite_skill(self, pending, active_task_id=None):
        self.blackboard.set_pending_decision({
            **pending,
            "type": "modify_skill_details",
            "operation": "overwrite",
        })

        return {
            "mode": "direct",
            "message": (
                f"What should replace `{pending.get('skill')}`?\n\n"
                f"Suggested replacements:\n"
                f"1. A new print script\n"
                f"2. A reusable Python function\n"
                f"3. A small CLI-style script\n"
                f"4. Something else you describe\n\n"
                f"Tell me what you want, or pick 1-4."
            ),
            "active_task_id": active_task_id,
        }

    def resolve_modify_skill_details(self, pending, user_input, active_task_id=None):
        text = user_input.strip()
        lowered = text.lower()

        if lowered in {"cancel", "never mind", "nevermind", "leave it", "stop"}:
            self.blackboard.clear_pending_decision()

            return {
                "mode": "direct",
                "message": "Okay, cancelled the modification.",
                "active_task_id": active_task_id,
            }

        instruction = self.expand_modify_choice(text)
        operation = pending.get("operation", "modify")

        self.blackboard.clear_pending_decision()

        goal = (
            f"{operation.capitalize()} existing skill '{pending.get('skill')}' "
            f"in file '{pending.get('filename')}'. "
            f"Current description: {pending.get('description')}. "
            f"Original request: {pending.get('original_request')}. "
            f"Modification request: {instruction}."
        )

        self.set_last_skill(pending.get("skill"), pending.get("filename"))

        return self.start_or_continue_coding_task(
            goal=goal,
            active_task_id=active_task_id,
        )

    def expand_modify_choice(self, user_input):
        text = user_input.strip().lower()

        if text == "1":
            return "Change what the script prints. Ask the coder to infer the new text from the user's details if provided."

        if text == "2":
            return "Modify the script so it prints the message multiple times."

        if text == "3":
            return "Refactor the script by adding comments and wrapping the behavior in a simple main function."

        if text == "4":
            return "Clean up or rename the skill while preserving its basic behavior."

        return user_input

    def run_pending_skill(self, pending, active_task_id=None):
        filename = pending.get("filename")
        skill_name = pending.get("skill")

        self.blackboard.set_pending_decision({
            "type": "run_skill_args",
            "skill": skill_name,
            "filename": filename,
            "active_task_id": active_task_id,
        })

        return {
            "mode": "direct",
            "message": (
                f"What arguments should I pass to `{skill_name}`?\n"
                f"Example: `Hello there!`\n"
                f"Say `default` to run it with no arguments."
            ),
            "active_task_id": active_task_id,
        }
    def set_last_skill(self, skill_name, filename):
        self.blackboard.set("last_skill", {
            "skill": skill_name,
            "filename": filename,
        })

    # ----------------------------
    # SKILL / CAPABILITY EVOLUTION
    # ----------------------------

    def propose_skill_evolution(self, plan, user_input, active_task_id=None):
        skill_name = plan.get("skill") or (
            plan.get("related_skills", [None])[0]
        )
        related_skills = plan.get("related_skills", [])
        action = plan.get("action")

        self.blackboard.set_pending_decision({
            "type": "skill_evolution",
            "operation": action,
            "skill": skill_name,
            "related_skills": related_skills,
            "original_request": user_input,
            "suggested_change": plan.get("suggested_change", ""),
            "reason": plan.get("reason", ""),
            "active_task_id": active_task_id,
        })

        if action == "combine_skills":
            return {
                "mode": "direct",
                "message": (
                    "This looks related to multiple existing skills:\n"
                    + "\n".join(f"- {name}" for name in related_skills)
                    + "\n\nSuggested action: combine these into one broader skill.\n"
                    f"Reason: {plan.get('reason')}\n\n"
                    "Do you want to combine them, expand one of them, create a new skill, or cancel?"
                ),
                "active_task_id": active_task_id,
            }

        return {
            "mode": "direct",
            "message": (
                f"This looks similar to existing skill `{skill_name}`.\n\n"
                f"Reason: {plan.get('reason')}\n"
                f"Suggested change: {plan.get('suggested_change')}\n\n"
                "Do you want to expand that skill, create a new skill, run the existing one, or cancel?"
            ),
            "active_task_id": active_task_id,
        }

    def route_interpreted_pending(self, interpretation, user_input, active_task_id=None):
        pending = self.blackboard.get_pending_decision()

        if not pending:
            return {
                "mode": "direct",
                "message": "There is no pending decision to resolve.",
                "active_task_id": active_task_id,
            }

        return self.resolve_pending_decision(
            pending=pending,
            user_input=user_input,
            active_task_id=active_task_id,
            interpretation=interpretation,
        )

    def propose_capability_evolution(self, plan, user_input, active_task_id=None):
        capability_name = plan.get("capability")

        self.blackboard.set_pending_decision({
            "type": "capability_evolution",
            "operation": "expand_capability",
            "capability": capability_name,
            "related_capabilities": plan.get("related_capabilities", []),
            "original_request": user_input,
            "suggested_change": plan.get("suggested_change", ""),
            "reason": plan.get("reason", ""),
            "active_task_id": active_task_id,
        })

        return {
            "mode": "direct",
            "message": (
                f"This looks related to existing capability `{capability_name}`.\n\n"
                f"Reason: {plan.get('reason')}\n"
                f"Suggested change: {plan.get('suggested_change')}\n\n"
                "Do you want to expand that capability, create a new capability, use the existing one, or cancel?"
            ),
            "active_task_id": active_task_id,
        }

    def resolve_skill_evolution(self, pending, user_input, active_task_id=None, interpretation=None):
        intent = interpretation.get("intent") if interpretation else "unknown"
        kind = interpretation.get("kind") if interpretation else "unknown"

        if kind == "cancel" or intent == "leave_skill":
            self.blackboard.clear_pending_decision()

            return {
                "mode": "direct",
                "message": "Okay, cancelled skill evolution.",
                "active_task_id": active_task_id,
            }

        if intent == "compare_options":
            skill_name = pending.get("skill")
            related_skills = pending.get("related_skills", [])

            lines = [
                f"`{skill_name}` looks like the best target skill to expand.",
                "",
                f"Reason: {pending.get('reason')}",
                f"Suggested change: {pending.get('suggested_change')}",
                "",
            ]

            if related_skills:
                lines.append("Related skills:")

                for name in related_skills:
                    skill = self.skill_registry.get(name)

                    if skill:
                        lines.append(
                            f"- `{name}`: {skill.get('description')} "
                            f"(file: {skill.get('filename')})"
                        )
                    else:
                        lines.append(f"- `{name}`")

                lines.append("")

            lines.append(
                "Choose one: expand the target skill, combine related skills, "
                "create a new skill, run the existing skill, or cancel."
            )

            return {
                "mode": "direct",
                "message": "\n".join(lines),
                "active_task_id": active_task_id,
            }

        if intent == "create_new":
            self.blackboard.clear_pending_decision()

            return self.start_or_continue_coding_task(
                goal=pending.get("original_request"),
                active_task_id=active_task_id,
            )

        if intent == "run_existing_skill":
            skill_name = pending.get("skill")

            if not skill_name:
                related = pending.get("related_skills", [])
                skill_name = related[0] if related else None

            if not skill_name:
                return {
                    "mode": "direct",
                    "message": "I could not determine which existing skill to run.",
                    "active_task_id": active_task_id,
                }

            skill = self.skill_registry.get(skill_name)
            self.blackboard.clear_pending_decision()

            if not skill:
                return {
                    "mode": "direct",
                    "message": "I could not find that skill to run.",
                    "active_task_id": active_task_id,
                }

            if hasattr(self, "set_last_skill"):
                self.set_last_skill(skill.get("name"), skill.get("filename"))

            return self.run_pending_skill({
                "skill": skill.get("name"),
                "filename": skill.get("filename"),
            }, active_task_id)

        if intent not in {"expand_skill", "combine_skills"}:
            return {
                "mode": "direct",
                "message": (
                    "I’m still waiting on your decision. You can ask me to compare, "
                    "expand it, combine them, create a new skill, run the existing one, or cancel."
                ),
                "active_task_id": active_task_id,
            }

        self.blackboard.clear_pending_decision()

        target_skill = pending.get("skill")

        if not target_skill:
            related = pending.get("related_skills", [])
            target_skill = related[0] if related else None

        target_filename = "skills/file.py"

        if target_skill and hasattr(self, "set_last_skill"):
            skill = self.skill_registry.get(target_skill)
            target_filename = skill.get("filename") if skill else target_filename
            self.set_last_skill(target_skill, target_filename)

        operation = "combine_skills" if intent == "combine_skills" else pending.get("operation")

        if operation == "combine_skills":
            goal = (
                f"Combine related skills into one reusable Python skill in skills/file.py. "
                f"Related skills: {pending.get('related_skills')}. "
                f"The new combined skill should satisfy this new request: "
                f"{pending.get('original_request')}. "
                f"It should preserve the behavior of related skills where possible. "
                f"It must accept command-line arguments using sys.argv, so when the skill is run "
                f"the user can provide the message to print. "
                f"If no arguments are provided, use a sensible default. "
                f"Support existing related behaviors and the new requested behavior. "
                f"Use a main function. Avoid creating duplicate skill files."
            )
        else:
            goal = (
                f"Expand existing skill '{target_skill}' in skills/file.py. "
                f"Original request: {pending.get('original_request')}. "
                f"User decision/details: {user_input}. "
                f"Suggested change: {pending.get('suggested_change')}. "
                f"Prefer updating the existing skill instead of creating duplicates. "
                f"If this skill accepts inputs, make those inputs explicit command-line arguments "
                f"and keep the registry arguments accurate."
            )

        return self.start_or_continue_coding_task(
            goal=goal,
            active_task_id=active_task_id,
        )

    def resolve_capability_evolution(self, pending, user_input, active_task_id=None):
        choice = user_input.lower().strip()

        if choice in {"cancel", "stop", "never mind", "nevermind"}:
            self.blackboard.clear_pending_decision()

            return {
                "mode": "direct",
                "message": "Okay, cancelled capability evolution.",
                "active_task_id": active_task_id,
            }

        self.blackboard.clear_pending_decision()

        if "new" in choice:
            return self.propose_capability(
                pending.get("original_request"),
                active_task_id,
            )

        goal = (
            f"Expand existing capability `{pending.get('capability')}`. "
            f"Original request: {pending.get('original_request')}. "
            f"User decision/details: {user_input}. "
            f"Suggested change: {pending.get('suggested_change')}."
        )

        return self.start_or_continue_coding_task(
            goal=goal,
            active_task_id=active_task_id,
        )

    # ----------------------------
    # CODE MONKEY REQUIREMENTS GATHERING
    # ----------------------------

    def _installed_capability_summaries(self) -> list:
        """Return a list of dicts describing each installed capability for overlap checking."""
        summaries = []
        if not INSTALLED_CAPABILITIES_DIR.exists():
            return summaries
        for cap_dir in sorted(INSTALLED_CAPABILITIES_DIR.iterdir()):
            if not cap_dir.is_dir():
                continue
            cmd_file = cap_dir / "commands.json"
            if not cmd_file.exists():
                continue
            try:
                data = json.loads(cmd_file.read_text(encoding="utf-8"))
                triggers = []
                for cmd in data.get("commands", []):
                    for t in cmd.get("triggers", []):
                        if "{" not in t:
                            triggers.append(t)
                # Resolve display name from capability registry
                cap_name = cap_dir.name
                reg_cap = self.registry.get(cap_name)
                display_name = (reg_cap or {}).get("display_name") or cap_name
                summaries.append({
                    "capability_name": cap_name,
                    "display_name": display_name,
                    "description": data.get("description", "")[:300],
                    "commands": triggers[:8],
                })
            except Exception:
                continue
        return summaries

    def start_code_monkey_requirements(self, user_input, active_task_id=None):
        # Check for overlap with existing installed capabilities before starting.
        installed = self._installed_capability_summaries()
        if installed:
            overlap = self.requirements_agent.check_overlap(user_input, installed)
            if overlap["has_overlap"]:
                display = overlap["display_name"] or overlap["capability_name"] or "an existing capability"
                overlap_type = overlap["overlap_type"]
                reason = overlap["reason"]
                if overlap_type == "same":
                    prompt = (
                        f"[Code Monkey] '{display}' already covers this.\n"
                        f"({reason})\n\n"
                        f"Options:\n"
                        f"  1 — Use it as-is (I'll show you what it can do)\n"
                        f"  2 — Extend it with new functionality\n"
                        f"  3 — Build something new anyway\n"
                    )
                else:
                    prompt = (
                        f"[Code Monkey] '{display}' is related to what you're describing.\n"
                        f"({reason})\n\n"
                        f"Options:\n"
                        f"  1 — Extend '{display}' to include this\n"
                        f"  2 — Build something new\n"
                    )
                self.blackboard.set_pending_decision({
                    "type": "code_monkey_overlap_decision",
                    "user_input": user_input,
                    "overlap": overlap,
                    "overlap_type": overlap_type,
                    "active_task_id": active_task_id,
                })
                return {
                    "mode": "direct",
                    "message": prompt,
                    "active_task_id": active_task_id,
                }

        result = self.requirements_agent.start(user_input)
        if result["done"]:
            return self.start_or_continue_coding_task(
                goal=result["goal"],
                active_task_id=active_task_id,
            )
        self.blackboard.set_pending_decision({
            "type": "code_monkey_requirements",
            "initial_request": user_input,
            "conversation": [{"role": "assistant", "text": result["question"]}],
            "active_task_id": active_task_id,
        })
        return {
            "mode": "direct",
            "message": f"[Code Monkey] {result['question']}",
            "active_task_id": active_task_id,
        }

    def resolve_overlap_decision(self, pending, user_input, active_task_id=None):
        lowered = user_input.lower().strip()
        overlap = pending.get("overlap", {})
        original_request = pending.get("user_input", "")
        overlap_type = pending.get("overlap_type", "partial")
        cap_name = overlap.get("capability_name")
        display_name = overlap.get("display_name") or cap_name or "existing capability"

        # Parse choice
        use_existing = lowered in {"1", "use", "use it", "run it", "run"} and overlap_type == "same"
        extend = lowered in {"1", "extend", "extend it", "yes", "expand"} and overlap_type == "partial"
        extend = extend or (lowered in {"2", "extend", "extend it"} and overlap_type == "same")
        build_new = lowered in {"2", "new", "build new", "start fresh", "no"} and overlap_type == "partial"
        build_new = build_new or lowered in {"3", "new", "build new", "start fresh"}
        cancel = lowered in {"cancel", "stop", "nevermind", "never mind", "abort"}

        if cancel:
            self.blackboard.clear_pending_decision()
            return {"mode": "direct", "message": "Cancelled.", "active_task_id": active_task_id}

        if use_existing:
            self.blackboard.clear_pending_decision()
            # Show what the existing capability can do
            cmd_file = INSTALLED_CAPABILITIES_DIR / cap_name / "commands.json"
            try:
                data = json.loads(cmd_file.read_text(encoding="utf-8"))
                triggers = []
                for cmd in data.get("commands", []):
                    desc = cmd.get("description", "")
                    examples = [t for t in cmd.get("triggers", []) if "{" not in t][:1]
                    if examples:
                        triggers.append(f"  - \"{examples[0]}\" — {desc}")
                lines = [f"Here's what '{display_name}' can do:"] + triggers
                return {"mode": "direct", "message": "\n".join(lines), "active_task_id": active_task_id}
            except Exception:
                return {"mode": "direct", "message": f"'{display_name}' is installed and ready.",
                        "active_task_id": active_task_id}

        if extend:
            self.blackboard.clear_pending_decision()
            # Load existing capability context and start requirements gathering for extension
            context = f"Extend the existing '{display_name}' capability to also: {original_request}"
            cap_reg = self.registry.get(cap_name)
            if cap_reg:
                existing_goal = cap_reg.get("description", "")
                context = (
                    f"Extend the existing '{display_name}' capability.\n"
                    f"Current goal: {existing_goal}\n"
                    f"New request: {original_request}\n"
                    f"Build a unified capability that covers both the existing functionality and the new request."
                )
            return self.start_code_monkey_requirements(context, active_task_id)

        if build_new:
            self.blackboard.clear_pending_decision()
            return self.start_code_monkey_requirements(original_request, active_task_id)

        # Unrecognised input — re-show the options
        if overlap_type == "same":
            msg = "Please choose: 1 (use as-is), 2 (extend it), or 3 (build new)."
        else:
            msg = "Please choose: 1 (extend it) or 2 (build something new)."
        return {"mode": "direct", "message": f"[Code Monkey] {msg}", "active_task_id": active_task_id}

    def resolve_code_monkey_requirements(self, pending, user_input, active_task_id=None):
        if user_input.lower().strip() in {"cancel", "stop", "nevermind", "never mind", "abort"}:
            self.blackboard.clear_pending_decision()
            return {
                "mode": "direct",
                "message": "Requirements gathering cancelled.",
                "active_task_id": active_task_id,
            }

        conversation = list(pending.get("conversation") or [])
        conversation.append({"role": "user", "text": user_input})

        initial_request = pending.get("initial_request", user_input)
        result = self.requirements_agent.continue_conversation(initial_request, conversation)

        if result["done"]:
            self.blackboard.clear_pending_decision()
            return self.start_or_continue_coding_task(
                goal=result["goal"],
                active_task_id=active_task_id,
            )

        conversation.append({"role": "assistant", "text": result["question"]})
        self.blackboard.set_pending_decision({
            **pending,
            "conversation": conversation,
        })
        return {
            "mode": "direct",
            "message": f"[Code Monkey] {result['question']}",
            "active_task_id": active_task_id,
        }

    # ----------------------------
    # CODE MONKEY REVIEW SESSION
    # ----------------------------

    def _offer_completed_task_review(self, active_task_id=None):
        try:
            tasks = self.code_monkey.recent_completed_tasks(limit=10)
        except Exception as exc:
            return {
                "mode": "direct",
                "message": f"Could not retrieve completed tasks: {exc}",
                "active_task_id": active_task_id,
            }

        if not tasks:
            return {
                "mode": "direct",
                "message": "No completed Code Monkey tasks found to review.",
                "active_task_id": active_task_id,
            }

        # Resolve display names (generate + cache any missing ones)
        names = _load_names()
        changed = False
        for t in tasks:
            tid = t["task_id"]
            if tid not in names:
                goal = t.get("goal") or t.get("message") or ""
                names[tid] = self.review_agent.generate_name(goal)
                changed = True
            t["display_name"] = names[tid]
        if changed:
            _save_names(names)

        if len(tasks) == 1:
            return self.enter_review_session(tasks[0]["task_id"], active_task_id)

        lines = ["[Code Monkey] Here are the tasks eligible for review:\n"]
        for i, t in enumerate(tasks, 1):
            state_label = "✓" if t["state"] == "verified" else "✗"
            lines.append(f"  {i}. {t['display_name']} {state_label}")
        lines.append("\nSay the number or the project name.")

        self.blackboard.set_pending_decision({
            "type": "code_monkey_pick_review",
            "tasks": tasks,
            "active_task_id": active_task_id,
        })
        return {
            "mode": "direct",
            "message": "\n".join(lines),
            "active_task_id": active_task_id,
        }

    def resolve_pick_review(self, pending, user_input, active_task_id=None):
        tasks = pending.get("tasks", [])
        text = user_input.strip()
        lowered = text.lower()

        if lowered in {"cancel", "stop", "nevermind", "never mind"}:
            self.blackboard.clear_pending_decision()
            return {"mode": "direct", "message": "Cancelled.", "active_task_id": active_task_id}

        # Numeric pick
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(tasks):
                self.blackboard.clear_pending_decision()
                return self.enter_review_session(tasks[idx]["task_id"], active_task_id)

        # Exact display name match (case-insensitive)
        for t in tasks:
            if t.get("display_name", "").lower() == lowered:
                self.blackboard.clear_pending_decision()
                return self.enter_review_session(t["task_id"], active_task_id)

        # Partial name match — all words in input appear in the display name
        input_words = set(lowered.split())
        for t in tasks:
            name_words = set(t.get("display_name", "").lower().split())
            if input_words and input_words.issubset(name_words):
                self.blackboard.clear_pending_decision()
                return self.enter_review_session(t["task_id"], active_task_id)

        # task_id prefix match
        for t in tasks:
            if t["task_id"].startswith(text):
                self.blackboard.clear_pending_decision()
                return self.enter_review_session(t["task_id"], active_task_id)

        names_list = ", ".join(f"{i+1}. {t.get('display_name', t['task_id'][:8])}" for i, t in enumerate(tasks))
        return {
            "mode": "direct",
            "message": f"I didn't recognize that. Options: {names_list}",
            "active_task_id": active_task_id,
        }

    def enter_review_session(self, task_id, active_task_id=None):
        if not task_id:
            return self._offer_completed_task_review(active_task_id)

        artifacts = self.code_monkey.get_artifacts(task_id)
        state = artifacts.get("state", "unknown")

        if state not in {"verified", "build_failed", "test_failed", "failed"}:
            return {
                "mode": "direct",
                "message": f"Task {task_id} is still in progress (state: {state}). Check back when it completes.",
                "active_task_id": active_task_id,
            }

        if state != "verified":
            return {
                "mode": "direct",
                "message": (
                    f"Task {task_id} did not complete successfully (state: {state}).\n"
                    "Say 'rebuild' to try again with the same goal, or describe what you want changed."
                ),
                "active_task_id": active_task_id,
            }

        explanation = self.review_agent.explain(
            goal=artifacts["goal"],
            readme=artifacts["readme"],
            api_code=artifacts["api_code"],
            workspace=artifacts["workspace"],
            test_code=artifacts["test_code"],
            commands_json=artifacts.get("commands_json", ""),
        )

        names = _load_names()
        if task_id not in names:
            names[task_id] = self.review_agent.generate_name(artifacts["goal"])
            _save_names(names)
        display_name = names.get(task_id) or "Code Project"

        self.blackboard.set_pending_decision({
            "type": "code_monkey_review",
            "task_id": task_id,
            "display_name": display_name,
            "goal": artifacts["goal"],
            "capability_name": artifacts["capability_name"],
            "readme": artifacts["readme"],
            "api_code": artifacts["api_code"],
            "workspace": artifacts["workspace"],
            "active_task_id": task_id,
        })

        api_code = artifacts.get("api_code", "")
        commands_json = artifacts.get("commands_json", "")

        # Count functions in api.py for spoken summary
        try:
            _tree = ast.parse(api_code)
            fn_names = [n.name for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        except Exception:
            fn_names = []

        fn_summary = ""
        if fn_names:
            fn_summary = f" It has {len(fn_names)} function{'s' if len(fn_names) != 1 else ''}: {', '.join(fn_names)}."

        tts_message = (
            f"Build complete. I've built {display_name}.{fn_summary} "
            "Say commit to save it, discard to remove it, or describe changes to revise it."
        )

        display_message = (
            f"[Code Monkey — {display_name}] Build complete.\n\n{explanation}\n\n"
            "How does this look? Give me feedback, say 'commit' to save it, "
            "or 'discard' to throw it away."
        )

        display_content = ""
        if api_code:
            display_content += f"### api.py\n```python\n{api_code}\n```\n\n"
        if commands_json:
            display_content += f"### commands.json\n```json\n{commands_json}\n```\n"

        return {
            "mode": "direct",
            "message": display_message,
            "tts": tts_message,
            "display_content": display_content.strip(),
            "active_task_id": task_id,
        }

    def resolve_code_monkey_review(self, pending, user_input, active_task_id=None):
        lowered = user_input.lower().strip()

        if lowered in {"cancel", "stop", "exit review", "back"}:
            self.blackboard.clear_pending_decision()
            return {
                "mode": "direct",
                "message": "Exited review session. Back to normal.",
                "active_task_id": active_task_id,
            }

        goal = pending.get("goal", "")
        task_id = pending.get("task_id", active_task_id)

        result = self.review_agent.classify_feedback(user_input, goal)
        action = result["action"]

        if action == "commit":
            self.blackboard.clear_pending_decision()
            display_name = pending.get("display_name") or pending.get("capability_name") or "generated_capability"
            capability_name = pending.get("capability_name") or "generated_capability"
            self._register_capability(
                name=capability_name,
                goal=goal,
                workspace=pending.get("workspace", ""),
                task_id=task_id,
                display_name=display_name,
            )
            return {
                "mode": "direct",
                "message": (
                    f"'{display_name}' committed to the capability registry.\n"
                    f"Workspace: {pending.get('workspace', '')}\n\n"
                    "Want to build something else? Just describe it, or keep chatting normally."
                ),
                "active_task_id": task_id,
            }

        if action == "discard":
            self.blackboard.clear_pending_decision()
            return {
                "mode": "direct",
                "message": "Discarded. Want to build something else? Just describe it, or keep chatting.",
                "active_task_id": active_task_id,
            }

        # action == "change"
        change_request = result["details"] or user_input
        revised_goal = self.review_agent.synthesize_change_goal(
            original_goal=goal,
            readme=pending.get("readme", ""),
            api_code=pending.get("api_code", ""),
            change_request=change_request,
        )

        self.blackboard.clear_pending_decision()
        response = self.start_or_continue_coding_task(
            goal=revised_goal,
            active_task_id=task_id,
        )
        response["message"] = (
            "[Code Monkey] Got it. Sending the revision to the build queue.\n"
            + response["message"]
            + "\nSay 'review' when the build completes to see the updated result."
        )
        return response

    def _register_capability(self, name, goal, workspace, task_id, display_name=""):
        installed_dir = self._install_capability_bundle(name, workspace)
        commands = self._load_commands_from_bundle(installed_dir)
        self.registry.add({
            "name": name,
            "display_name": display_name or name,
            "description": goal,
            "subsystem": "code_monkey_workspace",
            "action": "run",
            "examples": self._triggers_from_commands(commands) or [goal],
            "source": "code_monkey",
            "task_id": task_id,
            "workspace": workspace,
            "installed_dir": str(installed_dir) if installed_dir else None,
            "has_background": (installed_dir / "background.py").exists() if installed_dir else False,
        })
        if installed_dir and hasattr(self, "command_agent") and self.command_agent:
            self.command_agent.register(name, installed_dir)
        if installed_dir and hasattr(self, "background_manager") and self.background_manager:
            bg = installed_dir / "background.py"
            if bg.exists():
                self.background_manager.start(name, bg)

    def _install_capability_bundle(self, name: str, workspace: str) -> Path | None:
        """Copy build artifacts to a permanent installed_capabilities directory."""
        workspace_path = Path(workspace)
        if not workspace_path.exists():
            # workspace may be relative to the brain root
            from config import ROOT_DIR
            workspace_path = ROOT_DIR / workspace
        if not workspace_path.exists():
            return None
        try:
            dest = INSTALLED_CAPABILITIES_DIR / name
            dest.mkdir(parents=True, exist_ok=True)
            # Copy api.py
            src_api = workspace_path / "src" / "api.py"
            if src_api.exists():
                shutil.copy2(src_api, dest / "api.py")
            # Copy commands.json
            cmd_json = workspace_path / "commands.json"
            if cmd_json.exists():
                shutil.copy2(cmd_json, dest / "commands.json")
            # Copy background.py if present
            bg_py = workspace_path / "src" / "background.py"
            if bg_py.exists():
                shutil.copy2(bg_py, dest / "background.py")
            # Carry over data directory (symlink or copy)
            data_src = workspace_path / "src" / "data"
            data_dst = dest / "data"
            if data_src.exists() and not data_dst.exists():
                shutil.copytree(data_src, data_dst)
            elif not data_dst.exists():
                data_dst.mkdir(parents=True, exist_ok=True)
            return dest
        except Exception:
            return None

    def _load_commands_from_bundle(self, installed_dir: Path | None) -> list:
        if installed_dir is None:
            return []
        cmd_file = installed_dir / "commands.json"
        if not cmd_file.exists():
            return []
        try:
            return json.loads(cmd_file.read_text()).get("commands", [])
        except Exception:
            return []

    def _triggers_from_commands(self, commands: list) -> list:
        triggers = []
        for cmd in commands:
            for t in cmd.get("triggers", []):
                if "{" not in t:
                    triggers.append(t)
        return triggers[:5]

    # ----------------------------
    # CODING TASKS
    # ----------------------------

    def start_or_continue_coding_task(self, goal, active_task_id=None):
        try:
            if active_task_id and self.code_monkey.is_known_task(active_task_id):
                task_id = self.code_monkey.continue_task(active_task_id, goal)
                message = f"Started Code Monkey continuation: {task_id}"
            else:
                task_id = self.code_monkey.start_task(goal)
                message = f"Started Code Monkey coding job: {task_id}"
        except Exception as exc:
            return {
                "mode": "direct",
                "message": str(exc),
                "active_task_id": active_task_id,
            }

        self.event_log.write(
            task_id,
            "started",
            message,
            {"goal": goal},
        )

        return {
            "mode": "async",
            "message": message,
            "active_task_id": task_id,
        }

    # ----------------------------
    # CAPABILITIES
    # ----------------------------

    def propose_capability(self, user_input, active_task_id):
        proposal = self.capability_builder.propose(user_input)

        if not proposal["ok"]:
            return {
                "mode": "direct",
                "message": proposal["message"],
                "active_task_id": active_task_id,
            }

        capability = proposal["capability"]

        if proposal["requires_approval"]:
            return {
                "mode": "direct",
                "message": (
                    "I can probably add that capability, but it needs approval first.\n"
                    f"Proposed capability: {capability}"
                ),
                "active_task_id": active_task_id,
            }

        self.capability_builder.install(capability)

        return {
            "mode": "direct",
            "message": (
                f"Added new capability: {capability['name']}\n"
                f"Try asking again now."
            ),
            "active_task_id": active_task_id,
        }

    # ----------------------------
    # SYSTEM VIEWS
    # ----------------------------

    def show_updates(self, active_task_id=None):
        try:
            code_monkey_updates = self.code_monkey.unread_updates()
        except Exception as exc:
            code_monkey_updates = []
            code_monkey_error = str(exc)
        else:
            code_monkey_error = None

        events = self.event_log.unread()

        if not code_monkey_updates and not events and not code_monkey_error:
            return {
                "mode": "direct",
                "message": "No unread updates.",
                "active_task_id": active_task_id,
            }

        total = len(code_monkey_updates) + len(events)
        label = "notification" if total == 1 else "notifications"
        lines = [f"{total} unread {label}."]

        if code_monkey_error:
            lines.append(f"Code Monkey is unavailable: {code_monkey_error}")

        has_reviewable = False
        state_counts = {}
        for update in code_monkey_updates:
            state = update.get("state")
            task_id = update.get("task_id")
            state_counts[state or "unknown"] = state_counts.get(state or "unknown", 0) + 1
            if state in _FINAL_CODE_MONKEY_STATES:
                has_reviewable = True
                self.blackboard.set("last_completed_task_id", task_id)

        if state_counts:
            lines.append("Code Monkey: " + self._spoken_state_counts(state_counts) + ".")

        for update in code_monkey_updates[:4]:
            lines.append(self._spoken_update_line(update))

        remaining = len(code_monkey_updates) - 4
        if remaining > 0:
            more = "more notification" if remaining == 1 else "more notifications"
            lines.append(f"{remaining} {more} hidden.")

        for event in events:
            lines.append(f"{event['event_type']}: {self._short_text(event['message'], 140)}")

        self.event_log.mark_read([event["id"] for event in events])

        if has_reviewable:
            lines.append("Say review for details.")

        return {
            "mode": "direct",
            "message": "\n".join(lines),
            "active_task_id": active_task_id,
        }

    def _spoken_state_counts(self, counts: dict) -> str:
        names = {
            "verified": "passed",
            "build_failed": "build failed",
            "test_failed": "tests failed",
            "failed": "failed",
            "cancelled": "cancelled",
            "queued": "queued",
            "building": "building",
            "repairing": "repairing",
        }
        parts = []
        for state in sorted(counts):
            count = counts[state]
            noun = names.get(state, state.replace("_", " "))
            parts.append(f"{count} {noun}")
        return ", ".join(parts)

    def _spoken_update_line(self, update: dict) -> str:
        state = str(update.get("state") or "unknown")
        message = self._short_text(update.get("message") or state.replace("_", " "), 80)
        goal = self._short_goal(update.get("goal") or "")
        prefix = {
            "verified": "Passed",
            "build_failed": "Build failed",
            "test_failed": "Tests failed",
            "failed": "Failed",
            "cancelled": "Cancelled",
            "queued": "Queued",
            "building": "Building",
            "repairing": "Repairing",
        }.get(state, state.replace("_", " ").title())
        return f"{prefix}: {goal or message}."

    def _short_goal(self, goal: str) -> str:
        text = str(goal or "")
        confirmed = re.search(r"Confirmed user input:\s*(.+)", text)
        if confirmed:
            return self._short_text(confirmed.group(1), 90)
        original = re.search(r"Original request:\s*(.+?)(?:\. Modification request:|$)", text, re.S)
        if original:
            return self._short_text(original.group(1), 90)
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return self._short_text(first, 90)

    def _short_text(self, text: str, limit: int) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 3)].rstrip(" .,;:") + "..."

    def show_jobs(self, active_task_id=None):
        try:
            code_monkey_jobs = self.code_monkey.list_jobs()
        except Exception as exc:
            code_monkey_jobs = []
            code_monkey_error = str(exc)
        else:
            code_monkey_error = None

        legacy_jobs = self.job_manager.list_jobs()

        if not code_monkey_jobs and not legacy_jobs and not code_monkey_error:
            return {
                "mode": "direct",
                "message": "No jobs yet.",
                "active_task_id": active_task_id,
            }

        lines = ["Jobs:"]

        if code_monkey_error:
            lines.append(f"- code_monkey unavailable: {code_monkey_error}")

        for job in code_monkey_jobs[:20]:
            lines.append(
                f"- {job['task_id']} | code_monkey:{job['state']} | {job['goal']}"
            )

        for job in legacy_jobs[:20]:
            lines.append(f"- {job['id']} | legacy:{job['status']} | {job['title']}")

        return {
            "mode": "direct",
            "message": "\n".join(lines),
            "active_task_id": active_task_id,
        }

    def show_code_monkey_health(self, active_task_id=None):
        try:
            health = self.code_monkey.health()
        except Exception as exc:
            return {
                "mode": "direct",
                "message": f"Code Monkey unavailable: {exc}",
                "active_task_id": active_task_id,
            }

        lines = [
            "Code Monkey service:",
            f"- ok: {health.get('ok')}",
            f"- workers: {health.get('workers')}",
            f"- active workers: {health.get('active_workers')}",
            f"- queued tasks: {health.get('queued_tasks')}",
            f"- unread notifications: {health.get('unread_notifications')}",
            f"- data dir: {health.get('data_dir')}",
        ]

        return {
            "mode": "direct",
            "message": "\n".join(lines),
            "active_task_id": active_task_id,
        }
