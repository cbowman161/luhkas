import ast
import re

from models import get_model

try:
    from environment.environment_context import build_coder_environment_context
except Exception:
    def build_coder_environment_context():
        return "LOCAL ENVIRONMENT CONTEXT: unavailable"


DEFAULT_SKILL_PATH = "skills/generated_skill.py"
MAX_INTERNAL_GENERATION_ATTEMPTS = 4


class CoderAgent:
    def __init__(self):
        self.model = get_model("coder")
        self.sessions = {}

    def get_session(self, task_id, goal):
        task_id = task_id or "default"

        if task_id not in self.sessions:
            self.sessions[task_id] = CoderSession(task_id, goal, self.model)

        session = self.sessions[task_id]

        if session.goal != goal:
            session.reset(goal)

        return session

    def act(self, goal, context, iteration, last_error, task_id=None, allowed_actions=None, logger=None):
        session = self.get_session(task_id, goal)
        session.logger = logger

        return session.produce_action(
            context=context,
            iteration=iteration,
            last_error=last_error,
            allowed_actions=allowed_actions or [],
        )

    def receive_feedback(self, task_id, result, analysis):
        session = self.sessions.get(task_id or "default")

        if session:
            session.receive_feedback(result, analysis)


class CoderSession:
    def __init__(self, task_id, goal, model):
        self.task_id = task_id
        self.model = model
        self.logger = None
        self.reset(goal)

    def reset(self, goal):
        self.goal = goal
        self.feedback_log = []
        self.rejected_actions = []
        self.last_written_path = None
        self.last_code = ""
        self.test_command = None
        self.internal_generation_failures = []

    def log(self, message, data=None):
        if not self.logger:
            return

        try:
            self.logger(message, data or {})
        except Exception:
            pass

    def environment_context(self):
        try:
            return build_coder_environment_context()
        except Exception as exc:
            return f"LOCAL ENVIRONMENT CONTEXT: unavailable: {exc}"

    def receive_feedback(self, result, analysis):
        self.feedback_log.append({
            "result": result,
            "analysis": analysis,
        })

        stdout = result.get("stdout") or ""

        if stdout.startswith("File written: "):
            self.last_written_path = stdout.replace("File written: ", "").strip()

        if stdout.startswith("File patched: "):
            self.last_written_path = stdout.replace("File patched: ", "").strip()

        if analysis.get("failure_type") == "policy":
            self.rejected_actions.append(analysis.get("message_to_coder", ""))

    # ----------------------------
    # ACTION SELECTION
    # ----------------------------

    def produce_action(self, context, iteration, last_error, allowed_actions):
        history = context.get("history", []) if context else []

        last_action = history[-1].get("action") if history else {}
        last_analysis = history[-1].get("analysis") if history else {}

        last_type = (last_action or {}).get("type")
        failure_type = (last_analysis or {}).get("failure_type")
        suggestion = (last_analysis or {}).get("suggestion", "").lower()

        self.log("Coder action selection", {
            "iteration": iteration,
            "allowed_actions": allowed_actions,
            "last_action_type": last_type,
            "failure_type": failure_type,
            "suggestion_preview": suggestion[:300],
        })

        if failure_type in {"runtime", "syntax", "test", "requirement", "environment"}:
            if "patch_file" in allowed_actions:
                code = self.generate_valid_repair_code(context, last_error)

                if code:
                    self.last_code = code
                    self.log("Repair code accepted", {
                        "path": self.last_written_path,
                        "code_preview": code[:1000],
                    })
                    return self.patch_file_action(code)

                return self.fail_action("Unable to generate valid Python repair after internal validation.")

            return self.fail_action("Runtime or requirement failure occurred, but patch_file is not currently allowed.")

        if last_type in {"write_file", "patch_file"} and "command" in allowed_actions:
            return self.command_action()

        if ("run" in suggestion or "test" in suggestion) and "command" in allowed_actions:
            return self.command_action()

        if iteration == 0:
            self.last_written_path = self.safe_skill_path(self.generate_path_with_llm())
            code = self.generate_valid_implementation_code()

            if not code:
                return self.fail_action("Unable to generate valid Python after internal validation.")

            self.last_code = code
            self.test_command = self.generate_test_command_with_llm()

            if "write_file" in allowed_actions:
                return self.write_file_action(self.last_code)

            return self.fail_action("Generated code, but write_file is not currently allowed.")

        if "command" in allowed_actions:
            return self.command_action()

        return self.fail_action("Unable to proceed with the available actions.")

    # ----------------------------
    # GENERATION WITH INTERNAL RETRY
    # ----------------------------

    def generate_valid_implementation_code(self):
        last_failure = ""

        self.log("Internal implementation generation started", {
            "max_attempts": MAX_INTERNAL_GENERATION_ATTEMPTS,
        })

        for attempt in range(MAX_INTERNAL_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1
            self.log("Internal implementation attempt", {"attempt": attempt_number})

            code = self.generate_code_with_llm(last_failure=last_failure)
            code = self.clean_generated_code(code)

            self.log("Implementation draft generated", {
                "attempt": attempt_number,
                "path": self.last_written_path,
                "code_preview": code[:1200],
            })

            ok, error = self.validate_python_code(code)

            if not ok:
                self.log("Implementation draft failed Python validation", {
                    "attempt": attempt_number,
                    "error": error,
                })
                last_failure = error
                self.record_internal_failure("implementation", attempt_number, error, code)
                continue

            self.log("Implementation draft accepted", {
                "attempt": attempt_number,
                "path": self.last_written_path,
            })
            return code

        self.log("Implementation generation exhausted", {
            "failures": self.internal_generation_failures[-8:],
        })
        return None

    def generate_valid_repair_code(self, context, last_error):
        last_failure = ""

        self.log("Internal repair generation started", {
            "max_attempts": MAX_INTERNAL_GENERATION_ATTEMPTS,
        })

        for attempt in range(MAX_INTERNAL_GENERATION_ATTEMPTS):
            attempt_number = attempt + 1
            self.log("Internal repair attempt", {"attempt": attempt_number})

            code = self.generate_repair_code(
                context=context,
                last_error=self.combine_errors(last_error, last_failure),
            )
            code = self.clean_generated_code(code)

            self.log("Repair draft generated", {
                "attempt": attempt_number,
                "code_preview": code[:1200],
            })

            ok, error = self.validate_python_code(code)

            if not ok:
                self.log("Repair draft failed Python validation", {
                    "attempt": attempt_number,
                    "error": error,
                })
                last_failure = error
                self.record_internal_failure("repair", attempt_number, error, code)
                continue

            self.log("Repair draft accepted", {"attempt": attempt_number})
            return code

        self.log("Repair generation exhausted", {
            "failures": self.internal_generation_failures[-8:],
        })
        return None

    def record_internal_failure(self, phase, attempt, error, code):
        self.internal_generation_failures.append({
            "phase": phase,
            "attempt": attempt,
            "error": error,
            "code_preview": (code or "")[:500],
        })

    # ----------------------------
    # GENERATION PROMPTS
    # ----------------------------

    def generate_path_with_llm(self):
        prompt = f"""
Choose a clean Python file path for this local skill.

Request:
{self.goal}

Return ONLY one path.

Rules:
- Must start with skills/
- Must end with .py
- Use snake_case.
- Describe what the program does.
- Avoid generic names.
- Do not include prose, backticks, labels, or explanations.
"""

        raw = self.model.generate(prompt).strip()
        return self.safe_skill_path(raw)

    def generate_code_with_llm(self, last_failure=""):
        retry_guidance = ""

        if last_failure:
            retry_guidance = f"""
Previous draft was rejected before execution.
Validation failure:
{last_failure}

Try again with ONLY complete valid Python source code.
"""

        prompt = f"""
Generate a complete Python script for this request:

{self.goal}

{retry_guidance}

Environment / hardware / OS / software context:
{self.environment_context()}

Rules:
- Output ONLY valid Python code.
- No markdown.
- No explanations.
- No leading language label like python or python3.
- If input is needed, use sys.argv and validate arguments.
- The code should directly satisfy the request.
- Prefer standard library.
- Prefer Ubuntu/Linux commands.
- Never assume a hardware path, command, sensor, GPU, camera, or device exists.
- Probe/check existence first and degrade gracefully.
- Prefer discovery over hardcoded volatile paths.
- Do not replace the requested behavior with placeholder output, boolean checks, constant values, or test-only code.
- If stderr contains meaningful errors, the test should not be considered successful.
"""

        return self.model.generate(prompt)

    def generate_test_command_with_llm(self):
        path = self.last_written_path or DEFAULT_SKILL_PATH

        prompt = f"""
Generate one safe shell command to test this Python script.

Goal:
{self.goal}

Path:
{path}

Code:
{self.last_code}

Environment / hardware / OS / software context:
{self.environment_context()}

Return ONLY the command.

Rules:
- Must start with: python3 {path}
- Add sample command-line arguments if the script requires them.
- Do not use destructive commands.
- Do not include markdown or explanations.
- Use only local safe commands.
- The command must validate real requested behavior, not only that Python starts.
"""

        raw = self.model.generate(prompt).strip()
        return self.safe_test_command(raw) or f"python3 {path}"

    def generate_repair_code(self, context, last_error):
        history = context.get("history", []) if context else []
        latest_analysis = history[-1].get("analysis") if history else {}
        latest_analysis = latest_analysis or {}
        repair_context = latest_analysis.get("repair_context") or {}

        current_code = (
            repair_context.get("code")
            or self.last_code
            or self.latest_written_code(context)
        )

        prompt = f"""
Repair this Python script.

Goal:
{self.goal}

Current code:
{current_code}

Error or feedback:
{last_error}

Environment / hardware / OS / software context:
{self.environment_context()}

Return ONLY complete valid Python code.

Rules:
- Return a full replacement script, not a diff.
- No markdown.
- No explanations.
- No leading language label like python or python3.
- Preserve the original goal.
- Fix the actual failure.
- Do not remove functionality to make the test pass.
- Do not replace the requested behavior with placeholders, booleans, constants, existence checks, or test-only code.
- Prefer Ubuntu/Linux commands.
- Never assume a hardware path, command, sensor, GPU, camera, or device exists.
- Probe/check existence first and degrade gracefully.
"""

        return self.model.generate(prompt)

    # ----------------------------
    # ACTION ENVELOPES
    # ----------------------------

    def write_file_action(self, code):
        path = self.last_written_path or DEFAULT_SKILL_PATH
        return self.envelope(
            action="write_file",
            path=path,
            reason="Create implementation file.",
            content=code,
            content_type="text/x-python",
        )

    def patch_file_action(self, code):
        path = self.last_written_path or DEFAULT_SKILL_PATH
        return self.envelope(
            action="patch_file",
            path=path,
            reason="Repair implementation file.",
            content=code,
            content_type="text/x-python",
        )

    def command_action(self):
        path = self.last_written_path or DEFAULT_SKILL_PATH

        if not self.test_command:
            self.test_command = self.generate_test_command_with_llm()

        command = self.safe_test_command(self.test_command) or f"python3 {path}"

        self.log("Command action selected", {"command": command})
        return self.envelope(
            action="command",
            command=command,
            reason="Run verification command.",
        )

    def final_action(self, message):
        return self.envelope(
            action="final",
            reason="Task verified complete.",
            content=message,
            content_type="text/plain",
        )

    def fail_action(self, message):
        return self.envelope(
            action="fail",
            reason="Task cannot continue under current contract.",
            content=message,
            content_type="text/plain",
        )

    def envelope(self, action, path=None, command=None, reason="", content=None, content_type=None):
        lines = [f"ACTION: {action}"]

        if path:
            lines.append(f"PATH: {path}")

        if command:
            lines.append(f"COMMAND: {command}")

        if reason:
            safe_reason = str(reason).replace("\n", " ").strip()
            lines.append(f"REASON: {safe_reason}")

        if content_type:
            lines.append(f"CONTENT_TYPE: {content_type}")

        if content is None:
            return "\n".join(lines)

        return "\n".join(lines) + "\n\n---BEGIN CONTENT---\n" + str(content).rstrip("\n") + "\n---END CONTENT---"

    # ----------------------------
    # CLEANING / SAFETY
    # ----------------------------

    def clean_generated_code(self, raw):
        if not raw:
            return ""

        text = str(raw).strip()

        if "```" in text:
            parts = text.split("```")
            code_parts = []

            for i in range(1, len(parts), 2):
                block = parts[i].strip()
                lines = block.splitlines()

                if lines and lines[0].strip().lower() in {"python", "py", "python3"}:
                    lines = lines[1:]

                code_parts.append("\n".join(lines))

            if code_parts:
                text = "\n".join(code_parts).strip()
            else:
                text = text.replace("```", "")

        lines = text.splitlines()

        while lines:
            first = lines[0].strip()
            lowered = first.lower()

            if lowered in {"python", "py", "python3"}:
                lines = lines[1:]
                continue

            if self.is_obvious_prose_line(first) and not self.looks_like_python_start(first):
                lines = lines[1:]
                continue

            break

        while lines and self.is_obvious_prose_line(lines[-1].strip()):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def is_obvious_prose_line(self, line):
        if not line:
            return False

        lowered = line.lower()
        prose_markers = [
            "here is",
            "here's",
            "the error",
            "this error",
            "you can",
            "try again",
            "updated version",
            "this could be",
            "usually means",
            "explanation",
            "note:",
        ]
        return any(marker in lowered for marker in prose_markers)

    def looks_like_python_start(self, line):
        stripped = line.strip()
        return (
            stripped.startswith("#!")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("if ")
            or stripped.startswith("try:")
            or stripped.startswith("#")
            or stripped.startswith("@")
        )

    def validate_python_code(self, code):
        if not code or not code.strip():
            return False, "Code is empty."

        stripped = code.strip()
        bad_prefixes = [
            "the error",
            "here is",
            "here's",
            "this error",
            "updated version",
            "action:",
        ]

        lowered = stripped.lower()

        if lowered in {"python", "python3", "py"} or any(lowered.startswith(prefix) for prefix in bad_prefixes):
            return False, "Code appears to contain prose or an action/header label, not Python source."

        try:
            ast.parse(stripped)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}"
        except Exception as e:
            return False, f"Invalid Python: {e}"

        return True, ""

    def safe_skill_path(self, path):
        if not path:
            return DEFAULT_SKILL_PATH

        path = str(path).strip().strip('"').strip("'")
        path = path.replace("\\", "/")

        if "```" in path:
            path = path.replace("```", "")

        path = path.splitlines()[0].strip()
        path = path.replace("`", "")

        if "skills/" in path and not path.startswith("skills/"):
            path = path[path.index("skills/"):]

        if not path.startswith("skills/"):
            path = "skills/" + path

        if not path.endswith(".py"):
            path += ".py"

        directory, filename = path.rsplit("/", 1)

        filename = filename.lower()
        filename = re.sub(r"[^a-z0-9_.-]+", "_", filename)
        filename = re.sub(r"_+", "_", filename).strip("_")

        bad = {
            "file.py",
            "script.py",
            "skill.py",
            "task.py",
            "program.py",
            "main.py",
        }

        if filename in bad or len(filename) > 80:
            filename = self.generate_safe_fallback_filename()

        return f"{directory}/{filename}"

    def generate_safe_fallback_filename(self):
        raw = re.sub(r"[^a-z0-9]+", "_", self.goal.lower()).strip("_")
        raw = raw[:50].strip("_") or "generated_skill"
        return f"{raw}.py"

    def safe_test_command(self, command):
        if not command:
            return None

        command = str(command).strip().strip('"').strip("'")

        if "```" in command:
            command = command.replace("```", "")

        command = command.splitlines()[0].strip()
        command = command.replace("`", "")

        if not command.startswith("python3 skills/"):
            return None

        blocked = [
            "rm ",
            "rm-",
            "shutdown",
            "reboot",
            "mkfs",
            "dd ",
            "chmod",
            "chown",
            "sudo",
            "curl ",
            "wget ",
            "pip install",
        ]

        lowered = command.lower()

        if any(token in lowered for token in blocked):
            return None

        return command

    def latest_written_code(self, context):
        history = context.get("history", []) if context else []

        for entry in reversed(history):
            action = entry.get("action") or {}

            if action.get("type") in {"write_file", "patch_file"}:
                return action.get("content") or ""

        return ""

    def combine_errors(self, runtime_error, generation_error):
        parts = []

        if runtime_error:
            parts.append(str(runtime_error))

        if generation_error:
            parts.append(
                "Previous generated draft was rejected before execution because "
                f"it failed validation: {generation_error}"
            )

        return "\n\n".join(parts)
