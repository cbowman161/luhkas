import json
from models import get_model


ANALYST_PROMPT = """
You are a strict execution evaluator inside a persistent coding session.

Return ONLY JSON:

{
  "status": "success" | "retry" | "failure",
  "reason": "...",
  "failure_type": "none" | "syntax" | "runtime" | "test" | "requirement" | "environment" | "policy" | "unknown",
  "confidence": 0.0,
  "suggestion": "...",
  "message_to_coder": "..."
}

Rules:
- Success is only allowed after an explicit final action or a successful verification command.
- File writes, patches, reads, and listings are progress, not success.
- A fail action is always failure.
- If stdout/stderr contains failure markers, do not return success.
- Never explain outside JSON.
"""


FAILURE_MARKERS = [
    "traceback",
    "exception",
    "syntaxerror",
    "modulenotfounderror",
    "importerror",
    "error:",
    "failed",
    "failure",
    "assertionerror",
    "permission denied",
    "command timed out",
    "unable to proceed",
    "unable to generate",
]


class AnalystAgent:
    def __init__(self):
        self.model = get_model("analyst")

    def build_prompt(self, goal, context, result):
        return f"""{ANALYST_PROMPT}

GOAL:
{goal}

RESULT:
{json.dumps(result, indent=2)}

CONTEXT:
{json.dumps(context, indent=2)}

OUTPUT:
"""

    def analyze(self, goal, context, last_result):
        history = context.get("history", []) if context else []
        last_action = history[-1].get("action") if history else {}
        last_action = last_action or {}
        action_type = last_action.get("type")

        status = last_result.get("status")
        stdout = last_result.get("stdout") or ""
        stderr = last_result.get("stderr") or ""
        error = last_result.get("error") or ""
        combined = f"{stdout}\n{stderr}\n{error}".lower()

        if action_type == "fail":
            return {
                "status": "failure",
                "reason": last_action.get("content") or error or stderr or "Coder returned explicit failure.",
                "failure_type": self.detect_failure_type(combined),
                "confidence": 1.0,
                "suggestion": "",
                "message_to_coder": "",
            }

        if action_type == "final":
            if self.contains_failure_marker(combined):
                return {
                    "status": "failure",
                    "reason": "Final action contained failure language.",
                    "failure_type": self.detect_failure_type(combined),
                    "confidence": 0.95,
                    "suggestion": "Do not report final success unless the task was verified.",
                    "message_to_coder": "Your final message looked like a failure. Continue only if there is a valid fix, otherwise return fail.",
                }

            return {
                "status": "success",
                "reason": "Coder returned explicit final status.",
                "failure_type": "none",
                "confidence": 0.85,
                "suggestion": "",
                "message_to_coder": "Task completed.",
            }

        if status != "success":
            return {
                "status": "retry",
                "reason": "Execution failed",
                "failure_type": self.detect_failure_type(combined),
                "confidence": 0.9,
                "suggestion": stderr or error or stdout or "Fix the error and try a different approach.",
                "message_to_coder": stderr or error or stdout or "Execution failed. Inspect and repair the implementation.",
            }

        if stderr and stderr.strip():
            return {
                "status": "retry",
                "reason": "Execution produced stderr",
                "failure_type": self.detect_failure_type(combined) if self.contains_failure_marker(combined) else "runtime",
                "confidence": 0.9,
                "suggestion": stderr,
                "message_to_coder": stderr,
            }

        if self.contains_failure_marker(combined):
            return {
                "status": "retry",
                "reason": "Failure marker detected in output",
                "failure_type": self.detect_failure_type(combined),
                "confidence": 0.95,
                "suggestion": stdout or stderr or "Fix the detected failure.",
                "message_to_coder": stdout or stderr or "The output contains a failure marker. Inspect and patch the code.",
            }

        if action_type in {"write_file", "patch_file"}:
            path = last_action.get("path") or "file.py"
            return {
                "status": "retry",
                "reason": "File changed; verification required",
                "failure_type": "none",
                "confidence": 0.95,
                "suggestion": f"Run or test the changed file: python3 {path}",
                "message_to_coder": f"The file {path} was changed successfully. Run or test it next.",
            }

        if action_type == "read_file":
            return {
                "status": "retry",
                "reason": "File inspected",
                "failure_type": "none",
                "confidence": 0.8,
                "suggestion": "Use the inspected file contents to patch, run, or finish.",
                "message_to_coder": "You inspected a file. Continue with the next implementation or verification step.",
            }

        if action_type == "list_files":
            return {
                "status": "retry",
                "reason": "Files listed",
                "failure_type": "none",
                "confidence": 0.8,
                "suggestion": "Read the relevant file or create the needed implementation.",
                "message_to_coder": "Use the file listing to decide whether to read, patch, or create files.",
            }

        if action_type == "command":
            stdout = (last_result.get("stdout") or "").strip()
            stderr = (last_result.get("stderr") or "").strip()

            if stderr:
                return {
                    "status": "retry",
                    "reason": "Command produced stderr output",
                    "failure_type": "runtime",
                    "confidence": 0.95,
                    "suggestion": stderr,
                    "message_to_coder": stderr,
                }

            if not stdout:
                return {
                    "status": "retry",
                    "reason": "Command produced no output; cannot verify behavior",
                    "failure_type": "requirement",
                    "confidence": 0.9,
                    "suggestion": "Ensure the program prints meaningful output",
                    "message_to_coder": "The script ran but produced no output. Fix it to display the result.",
                }

            return {
                "status": "success",
                "reason": "Command produced output and no errors",
                "failure_type": "none",
                "confidence": 0.9,
                "suggestion": "",
                "message_to_coder": "Execution succeeded with visible output.",
            }

        prompt = self.build_prompt(goal, context, last_result)
        raw = self.model.generate(prompt)

        try:
            parsed = json.loads(raw)
        except Exception:
            return {
                "status": "retry",
                "reason": "Bad analysis output",
                "failure_type": "unknown",
                "confidence": 0.4,
                "suggestion": "Try a different implementation approach.",
                "message_to_coder": "The analysis output was invalid. Continue with a safer next step.",
            }

        return self.validate_analysis(parsed)

    def validate_analysis(self, analysis):
        status = analysis.get("status")
        if status not in {"success", "retry", "failure"}:
            analysis["status"] = "retry"
            analysis["reason"] = "Invalid analyst status normalized to retry"
            analysis["failure_type"] = "unknown"
            analysis["confidence"] = 0.4
        return analysis

    def contains_failure_marker(self, text):
        text = (text or "").lower()
        return any(marker in text for marker in FAILURE_MARKERS)

    def detect_failure_type(self, text):
        text = (text or "").lower()

        if "syntaxerror" in text:
            return "syntax"

        if "modulenotfounderror" in text or "no module named" in text or "importerror" in text:
            return "environment"

        if "assert" in text or "test failed" in text or "failed" in text or "failure" in text:
            return "test"

        if "traceback" in text or "exception" in text or "runtimeerror" in text:
            return "runtime"

        if "permission denied" in text:
            return "environment"

        if "requirement" in text:
            return "requirement"

        return "unknown"
