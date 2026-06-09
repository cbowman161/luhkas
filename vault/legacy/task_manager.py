import os
import threading
import uuid
import traceback
import json
from datetime import datetime

from executor import execute_action
from validator import validate_action
from action_policy import ActionPolicy

from agents.coder_agent import CoderAgent
from agents.analyst_agent import AnalystAgent
from skill_registrar import SkillRegistrar


class TaskManager:
    def __init__(self, blackboard, skill_registry=None, event_log=None):
        self.blackboard = blackboard
        self.event_log = event_log
        self.coder = CoderAgent()
        self.analyst = AnalystAgent()
        self.policy = ActionPolicy()
        self.skill_registrar = SkillRegistrar(skill_registry) if skill_registry else None
        self.unregistered_skill_files = {}
        self.verbose_task_stdout = False

    # ----------------------------
    # LOGGING / NOTIFICATIONS
    # ----------------------------

    def log_task_event(self, task_id, message, data=None, event_type="task_progress"):
        data = data or {}

        if not isinstance(message, str):
            message = str(message)

        if not isinstance(event_type, str):
            event_type = "task_progress"

        self.append_task_log_file(
            task_id=task_id,
            message=message,
            data=data,
            event_type=event_type,
        )

        # User-facing latest-update sink. EventLog.write should upsert one row per task_id.
        if getattr(self, "event_log", None):
            self.event_log.write(
                task_id,
                event_type,
                message,
                data,
            )

        # Optional live debug sink. Keep False for normal CLI use.
        if getattr(self, "verbose_task_stdout", False):
            print(f"[TASK {task_id}] {message}: {data}")

    def append_task_log_file(self, task_id, message, data=None, event_type="task_progress"):
        os.makedirs("logs/tasks", exist_ok=True)
        log_path = os.path.join("logs/tasks", f"{task_id}.log")

        entry = {
            "time": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def notify_task_terminal(self, task_id, level, message, data=None):
        """
        Terminal-state notification-center sink.

        Preferred EventLog API:
            notify(job_id, level, message, data=None)

        This is intentionally separate from progress updates. Progress goes to:
            - logs/tasks/<task_id>.log
            - events latest-per-task row

        Terminal notifications go to:
            - notifications table, consumed by main.py on next user interaction
        """
        if not getattr(self, "event_log", None):
            return

        if hasattr(self.event_log, "notify"):
            self.event_log.notify(task_id, level, message, data or {})
            return

        # Compatibility fallback: if notifications table is not implemented yet,
        # make the terminal message visible through normal `updates`.
        self.event_log.write(task_id, level, message, data or {})

    def task_logger(self, task_id):
        return lambda message, data=None: self.log_task_event(
            task_id,
            message,
            data or {},
            event_type="coder_internal",
        )

    # ----------------------------
    # TASK LIFECYCLE
    # ----------------------------

    def create_task(self, goal, task_type):
        task_id = str(uuid.uuid4())
        self.blackboard.init_task(task_id, goal)
        self.unregistered_skill_files[task_id] = set()

        self.log_task_event(task_id, "Task created", {
            "goal": goal,
            "task_type": task_type,
        }, event_type="started")

        thread = threading.Thread(
            target=self.run_task,
            args=(task_id, goal, task_type, False),
            daemon=True,
        )
        thread.start()

        return task_id

    def continue_task(self, task_id, user_input):
        self.log_task_event(task_id, "Task follow-up received", {
            "user_input": user_input,
        })

        self.blackboard.update(task_id, {
            "iteration": None,
            "followup": True,
            "action": {
                "type": "knowledge",
                "path": None,
                "content": f"User follow-up: {user_input}",
            },
            "result": {
                "status": "success",
                "stdout": user_input,
                "stderr": "",
                "error": "",
                "returncode": 0,
            },
            "analysis": {
                "status": "retry",
                "reason": "User added follow-up instruction",
                "failure_type": "none",
                "confidence": 1.0,
                "suggestion": user_input,
                "message_to_coder": user_input,
            },
        })

        thread = threading.Thread(
            target=self.run_task,
            args=(task_id, user_input, "task", True),
            daemon=True,
        )
        thread.start()

    def history_after_latest_followup(self, history):
        latest = None

        for i, entry in enumerate(history):
            if entry.get("followup"):
                latest = i

        if latest is None:
            return history

        return history[latest + 1:]

    def run_task(self, task_id, goal, task_type, is_followup=False):
        try:
            max_iterations = 10
            last_error = ""

            self.log_task_event(task_id, "Task loop started", {
                "goal": goal,
                "task_type": task_type,
                "is_followup": is_followup,
                "max_iterations": max_iterations,
            })

            for iteration in range(max_iterations):
                iteration_number = iteration + 1

                self.log_task_event(task_id, "Iteration started", {
                    "iteration": iteration_number,
                })

                context = self.blackboard.get(task_id) or {}
                history = context.get("history", [])

                policy_history = (
                    self.history_after_latest_followup(history)
                    if is_followup
                    else history
                )

                allowed_actions = self.policy.allowed_actions(policy_history)

                self.log_task_event(task_id, "Allowed actions", {
                    "iteration": iteration_number,
                    "allowed_actions": allowed_actions,
                })

                try:
                    raw_output = self.coder.act(
                        goal=goal,
                        context=context,
                        iteration=iteration,
                        last_error=last_error,
                        task_id=task_id,
                        allowed_actions=allowed_actions,
                        logger=self.task_logger(task_id),
                    )

                    self.log_task_event(task_id, "Coder output", {
                        "iteration": iteration_number,
                        "raw_output_preview": raw_output[:2000] if raw_output else "",
                    })

                except Exception as exc:
                    error = f"Coder failed: {exc}"
                    failure_result = {
                        "status": "error",
                        "error": error,
                        "traceback": traceback.format_exc(),
                    }
                    self.log_task_event(task_id, "Coder failed", {
                        "iteration": iteration_number,
                        **failure_result,
                    }, event_type="failed")
                    self.blackboard.set_result(task_id, failure_result)
                    self.cleanup_unregistered_skill_files(task_id, reason="coder failure")
                    self.notify_task_terminal(
                        task_id,
                        "error",
                        "Task failed during coding. Check `updates` for details.",
                        {"task_id": task_id, "error": error},
                    )
                    return

                try:
                    action = validate_action(raw_output)

                    terminal_actions = {"final", "fail"}

                    if action["type"] not in allowed_actions and action["type"] not in terminal_actions:
                        last_error = (
                            f"Action {action['type']} is not allowed now. "
                            f"Allowed actions: {allowed_actions}"
                        )

                        self.log_task_event(task_id, "Policy rejected action", {
                            "iteration": iteration_number,
                            "error": last_error,
                            "action": action,
                        })

                        self.coder.receive_feedback(
                            task_id,
                            {
                                "status": "error",
                                "stdout": "",
                                "stderr": last_error,
                                "error": last_error,
                                "returncode": -1,
                            },
                            {
                                "status": "retry",
                                "reason": "Policy rejection",
                                "failure_type": "policy",
                                "confidence": 1.0,
                                "suggestion": last_error,
                                "message_to_coder": last_error,
                            },
                        )

                        continue

                    self.log_task_event(task_id, "Validated action", {
                        "iteration": iteration_number,
                        "action": self.safe_action_for_log(action),
                    })
                    last_error = ""

                except Exception as exc:
                    last_error = f"VALIDATION ERROR: {str(exc)}\nOUTPUT:\n{raw_output}"
                    self.log_task_event(task_id, "Validation failed", {
                        "iteration": iteration_number,
                        "error": str(exc),
                        "raw_output_preview": raw_output[:2000] if raw_output else "",
                    })
                    continue

                try:
                    result = execute_action(action)
                    self.log_task_event(task_id, "Execution result", {
                        "iteration": iteration_number,
                        "result": self.safe_result_for_log(result),
                    })
                except Exception as exc:
                    error = f"Execution failed: {exc}"
                    failure_result = {
                        "status": "error",
                        "error": error,
                        "traceback": traceback.format_exc(),
                    }
                    self.log_task_event(task_id, "Execution failed", {
                        "iteration": iteration_number,
                        **failure_result,
                    }, event_type="failed")
                    self.blackboard.set_result(task_id, failure_result)
                    self.cleanup_unregistered_skill_files(task_id, reason="execution failure")
                    self.notify_task_terminal(
                        task_id,
                        "error",
                        "Task failed during execution. Check `updates` for details.",
                        {"task_id": task_id, "error": error},
                    )
                    return

                self.track_unregistered_skill_file(task_id, action, result)

                try:
                    context_for_analysis = {
                        **(self.blackboard.get(task_id) or {}),
                    }

                    analysis_history = context_for_analysis.get("history", []) + [
                        {
                            "iteration": iteration,
                            "action": action,
                            "result": result,
                            "registered_skill": None,
                            "analysis": None,
                        }
                    ]

                    context_for_analysis["history"] = analysis_history

                    analysis = self.analyst.analyze(
                        goal=goal,
                        context=context_for_analysis,
                        last_result=result,
                    )

                    self.log_task_event(task_id, "Analyst output", {
                        "iteration": iteration_number,
                        "analysis": analysis,
                    })

                except Exception as exc:
                    error = f"Analyst failed: {exc}"
                    failure_result = {
                        "status": "error",
                        "error": error,
                        "traceback": traceback.format_exc(),
                    }
                    self.log_task_event(task_id, "Analyst failed", {
                        "iteration": iteration_number,
                        **failure_result,
                    }, event_type="failed")
                    self.blackboard.set_result(task_id, failure_result)
                    self.cleanup_unregistered_skill_files(task_id, reason="analyst failure")
                    self.notify_task_terminal(
                        task_id,
                        "error",
                        "Task failed during analysis. Check `updates` for details.",
                        {"task_id": task_id, "error": error},
                    )
                    return

                registered_skill = None

                if analysis.get("status") == "success":
                    registered_skill = self.register_latest_skill_after_success(
                        task_id=task_id,
                        goal=goal,
                        context_for_analysis=context_for_analysis,
                        final_result=result,
                    )

                self.blackboard.update(task_id, {
                    "iteration": iteration,
                    "action": action,
                    "result": result,
                    "registered_skill": registered_skill,
                    "analysis": analysis,
                })

                self.coder.receive_feedback(task_id, result, analysis)

                if result.get("needs_dependency_approval"):
                    pause_result = {
                        "status": "paused",
                        "reason": "Waiting for dependency installation approval.",
                        "package": result.get("package"),
                    }

                    self.blackboard.set_pending_decision({
                        "type": "dependency_install",
                        "task_id": task_id,
                        "goal": goal,
                        "package": result.get("package"),
                        "missing_module": result.get("missing_module"),
                        "install_command": result.get("install_command"),
                        "original_command": action.get("content"),
                    })

                    self.blackboard.set_result(task_id, pause_result)

                    self.log_task_event(task_id, "Task paused for dependency approval", {
                        "package": result.get("package"),
                        "missing_module": result.get("missing_module"),
                        "install_command": result.get("install_command"),
                    }, event_type="paused")

                    self.notify_task_terminal(
                        task_id,
                        "warning",
                        "Task paused and needs your input. Check `updates` for details.",
                        {"task_id": task_id, **pause_result},
                    )
                    return

                if analysis.get("status") == "success":
                    final_result = {
                        **result,
                        "registered_skill": registered_skill,
                    }
                    self.blackboard.set_result(task_id, final_result)
                    self.mark_registered_file(task_id, registered_skill)

                    self.log_task_event(
                        task_id,
                        "Task completed successfully",
                        final_result,
                        event_type="completed",
                    )
                    self.notify_task_terminal(
                        task_id,
                        "success",
                        "Task completed. Check `updates` for details.",
                        {"task_id": task_id, "status": "completed"},
                    )
                    return

                if analysis.get("status") == "failure":
                    failure_result = {
                        "status": "error",
                        "error": analysis.get("reason"),
                    }
                    self.blackboard.set_result(task_id, failure_result)
                    self.log_task_event(task_id, "Task failed", failure_result, event_type="failed")
                    self.cleanup_unregistered_skill_files(task_id, reason="analysis failure")
                    self.notify_task_terminal(
                        task_id,
                        "error",
                        "Task failed. Check `updates` for details.",
                        {"task_id": task_id, **failure_result},
                    )
                    return

                last_error = (
                    analysis.get("message_to_coder")
                    or analysis.get("suggestion")
                    or ""
                )

            max_result = {
                "status": "error",
                "error": "Max iterations reached",
            }
            self.blackboard.set_result(task_id, max_result)
            self.log_task_event(task_id, "Task max iterations reached", max_result, event_type="failed")
            self.cleanup_unregistered_skill_files(task_id, reason="max iterations reached")
            self.notify_task_terminal(
                task_id,
                "error",
                "Task stopped after max iterations. Check `updates` for details.",
                {"task_id": task_id, **max_result},
            )

        except Exception as exc:
            fatal_result = {
                "status": "error",
                "error": f"Fatal error in task: {exc}",
                "traceback": traceback.format_exc(),
            }
            self.blackboard.set_result(task_id, fatal_result)

            try:
                self.log_task_event(task_id, "Fatal error in task", fatal_result, event_type="failed")
                self.notify_task_terminal(
                    task_id,
                    "error",
                    "Task crashed. Check `updates` for details.",
                    {"task_id": task_id, **fatal_result},
                )
                self.cleanup_unregistered_skill_files(task_id, reason="fatal error")
            except Exception:
                # Last-resort stderr visibility. Avoid crashing the background thread.
                print("[FATAL ERROR IN TASK]")
                traceback.print_exc()

    # ----------------------------
    # REGISTRATION
    # ----------------------------

    def register_latest_skill_after_success(self, task_id, goal, context_for_analysis, final_result):
        if not self.skill_registrar:
            return None

        latest_write = self.latest_write_or_patch_action(context_for_analysis)

        if not latest_write:
            return None

        try:
            registered_skill = self.skill_registrar.maybe_register(
                goal=goal,
                action=latest_write,
                result=final_result,
            )

            if registered_skill:
                self.log_task_event(task_id, "Skill registered", registered_skill, event_type="skill_registered")

            return registered_skill

        except Exception as exc:
            self.log_task_event(task_id, "Skill registration failed", {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }, event_type="failed")
            return None

    def latest_write_or_patch_action(self, context):
        history = context.get("history", []) if context else []

        for entry in reversed(history):
            action = entry.get("action") or {}

            if action.get("type") in {"write_file", "patch_file"}:
                return action

        return None

    # ----------------------------
    # UNREGISTERED FILE CLEANUP
    # ----------------------------

    def track_unregistered_skill_file(self, task_id, action, result):
        if result.get("status") != "success":
            return

        if action.get("type") not in {"write_file", "patch_file"}:
            return

        path = action.get("path")

        if not path or not path.startswith("skills/"):
            return

        self.unregistered_skill_files.setdefault(task_id, set()).add(path)
        self.log_task_event(task_id, "Tracked unregistered created skill file", {
            "path": path,
        })

    def mark_registered_file(self, task_id, registered_skill):
        if not registered_skill:
            return

        filename = registered_skill.get("filename")

        if not filename:
            return

        files = self.unregistered_skill_files.setdefault(task_id, set())
        files.discard(filename)

    def cleanup_unregistered_skill_files(self, task_id, reason=""):
        paths = list(self.unregistered_skill_files.get(task_id, set()))

        if not paths:
            return

        removed = []
        failed = []

        for path in paths:
            try:
                if not path.startswith("skills/"):
                    failed.append({"path": path, "error": "Refusing to delete non-skill path"})
                    continue

                if os.path.exists(path):
                    os.remove(path)
                    removed.append(path)
            except Exception as exc:
                failed.append({"path": path, "error": str(exc)})

        self.unregistered_skill_files[task_id] = set()

        self.log_task_event(task_id, "Cleaned up unregistered skill files", {
            "reason": reason,
            "removed": removed,
            "failed": failed,
        })

    # ----------------------------
    # SAFE LOG REPRESENTATIONS
    # ----------------------------

    def safe_action_for_log(self, action):
        if not action:
            return {}

        safe = dict(action)

        if safe.get("content") and len(safe["content"]) > 2000:
            safe["content"] = safe["content"][:2000] + "...<truncated>"

        return safe

    def safe_result_for_log(self, result):
        if not result:
            return {}

        safe = dict(result)

        for key in ["stdout", "stderr", "error"]:
            value = safe.get(key)
            if isinstance(value, str) and len(value) > 2000:
                safe[key] = value[:2000] + "...<truncated>"

        return safe
