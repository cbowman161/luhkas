import json
import re
from models import get_model


INTERPRETER_PROMPT = """
You interpret user input for a local agent system.

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{
  "kind": "new_request" | "answer_pending_decision" | "clarification_question" | "cancel" | "session_command",
  "intent": "create_skill" | "create_capability" | "run_existing_skill" | "modify_skill" | "overwrite_skill" | "leave_skill" | "expand_skill" | "combine_skills" | "create_new" | "compare_options" | "provide_arguments" | "unknown",
  "arguments": "string_or_null",
  "confidence": 0.0,
  "reason": "short explanation"
}

Rules:
- If pending_decision is not empty, interpret the user input as an answer to that pending decision unless clearly unrelated.
- If pending_decision.type is "run_skill_args", the user is probably providing runtime arguments.
- If no pending decision exists, classify the user input as a new_request unless it is a session command.
- A request to write/create/make/build a script, file, program, function, or tool is intent create_skill.
- A request to add a system command/capability is intent create_capability.
- "run", "run it", "use existing", "use the existing one" means run_existing_skill when answering a pending decision.
- "what's different", "compare", "why", "explain" means compare_options when answering a pending decision.
- "cancel", "never mind", "stop" means cancel.

Examples:

User input: write a python script that prints hello
Pending decision: {}
Output:
{"kind":"new_request","intent":"create_skill","arguments":"prints hello","confidence":0.95,"reason":"User wants to create a Python script skill."}

User input: make a script that adds two numbers
Pending decision: {}
Output:
{"kind":"new_request","intent":"create_skill","arguments":"adds two numbers","confidence":0.95,"reason":"User wants to create a Python script skill."}

User input: run
Pending decision: {"type":"existing_skill","skill":"print_hello"}
Output:
{"kind":"answer_pending_decision","intent":"run_existing_skill","arguments":null,"confidence":0.95,"reason":"User chose to run the existing skill."}

User input: 2 3
Pending decision: {"type":"run_skill_args","skill":"add_numbers"}
Output:
{"kind":"answer_pending_decision","intent":"provide_arguments","arguments":"2 3","confidence":0.95,"reason":"User provided runtime arguments."}

User input: what are the differences
Pending decision: {"type":"skill_evolution"}
Output:
{"kind":"answer_pending_decision","intent":"compare_options","arguments":null,"confidence":0.9,"reason":"User asked to compare the options."}
"""


class InteractionInterpreter:
    def __init__(self):
        self.model = get_model("router")

    def interpret(self, user_input, pending=None, session=None):
        prompt = f"""{INTERPRETER_PROMPT}

USER INPUT:
{user_input}

PENDING DECISION:
{json.dumps(pending or {}, indent=2)}

SESSION:
{json.dumps(session or {}, indent=2)}

OUTPUT:
"""

        raw = self.model.generate(prompt, think=False)
        parsed = self.parse_json(raw)

        if parsed is None:
            return self.fallback(user_input, pending, raw)

        return self.sanitize(parsed, user_input, pending)

    def parse_json(self, raw):
        if not raw:
            return None

        text = raw.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def sanitize(self, data, user_input, pending=None):
        valid_kinds = {
            "new_request",
            "answer_pending_decision",
            "clarification_question",
            "cancel",
            "session_command",
        }

        valid_intents = {
            "create_skill",
            "create_capability",
            "run_existing_skill",
            "modify_skill",
            "overwrite_skill",
            "leave_skill",
            "expand_skill",
            "combine_skills",
            "create_new",
            "compare_options",
            "provide_arguments",
            "unknown",
        }

        kind = data.get("kind")
        intent = data.get("intent")

        if kind not in valid_kinds:
            kind = "new_request"

        if intent not in valid_intents:
            intent = "unknown"

        pending = pending or {}

        if pending and kind == "new_request":
            kind = "answer_pending_decision"

        if pending.get("type") == "run_skill_args":
            kind = "answer_pending_decision"
            intent = "provide_arguments"

        if kind == "new_request" and intent == "unknown":
            inferred = self.infer_new_request_intent(user_input)

            if inferred:
                intent = inferred

        if kind == "answer_pending_decision" and intent == "unknown":
            inferred = self.infer_pending_intent(user_input, pending)

            if inferred:
                intent = inferred

        confidence = float(data.get("confidence", 0.0) or 0.0)

        if intent != "unknown" and confidence < 0.7:
            confidence = 0.7

        return {
            "kind": kind,
            "intent": intent,
            "arguments": data.get("arguments") if data.get("arguments") is not None else self.default_arguments(user_input, intent),
            "confidence": confidence,
            "reason": data.get("reason") or "Interpreted user input.",
        }

    def infer_new_request_intent(self, user_input):
        lowered = user_input.lower()

        creation_phrases = [
            "write a",
            "create a",
            "make a",
            "build a",
            "generate a",
            "script",
            "program",
            "function",
            "tool",
        ]

        if any(phrase in lowered for phrase in creation_phrases):
            return "create_skill"

        return None

    def infer_pending_intent(self, user_input, pending):
        lowered = user_input.lower().strip()

        if lowered in {"cancel", "stop", "never mind", "nevermind"}:
            return "leave_skill"

        if any(phrase in lowered for phrase in {
            "run",
            "run it",
            "use existing",
            "use the existing one",
            "existing",
        }):
            return "run_existing_skill"

        if any(phrase in lowered for phrase in {
            "modify",
            "change",
            "edit",
            "update",
            "improve",
        }):
            return "modify_skill"

        if any(phrase in lowered for phrase in {
            "overwrite",
            "replace",
            "rewrite",
            "redo",
        }):
            return "overwrite_skill"

        if any(phrase in lowered for phrase in {
            "compare",
            "difference",
            "differences",
            "why",
            "explain",
        }):
            return "compare_options"

        if any(phrase in lowered for phrase in {
            "combine",
            "merge",
            "consolidate",
        }):
            return "combine_skills"

        if any(phrase in lowered for phrase in {
            "expand",
            "extend",
            "yes",
            "do it",
            "go ahead",
            "proceed",
        }):
            return "expand_skill"

        if pending.get("type") == "run_skill_args":
            return "provide_arguments"

        return None

    def default_arguments(self, user_input, intent):
        if intent in {"create_skill", "provide_arguments"}:
            return user_input

        return None

    def fallback(self, user_input, pending=None, raw=None):
        pending = pending or {}

        if pending:
            intent = self.infer_pending_intent(user_input, pending) or "unknown"

            return {
                "kind": "answer_pending_decision",
                "intent": intent,
                "arguments": self.default_arguments(user_input, intent),
                "confidence": 0.6 if intent != "unknown" else 0.2,
                "reason": "Fallback pending response.",
            }

        intent = self.infer_new_request_intent(user_input) or "unknown"

        return {
            "kind": "new_request",
            "intent": intent,
            "arguments": self.default_arguments(user_input, intent),
            "confidence": 0.6 if intent != "unknown" else 0.2,
            "reason": "Fallback after invalid interpreter output.",
        }