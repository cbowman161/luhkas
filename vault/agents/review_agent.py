import json

from agents._json_utils import extract_json as _extract_json
from models import get_model


_EXPLAIN_PROMPT = """You are Code Monkey debriefing the user on a capability bundle you just built.

The brain's command_agent will route user phrases directly to Python functions — there is no HTTP server.
Write a clear, structured plain-text report covering ALL of the following sections:

FILES CREATED
List every file created (read from the API source and README — do not guess).
Format: path — one sentence on what it does.

WHAT IT DOES
2-3 sentences describing what this capability does and what data it stores.

VOICE COMMANDS
List every command the user can now say to the brain to use this capability.
Read these from the commands.json section provided. Format exactly:
- "trigger phrase" → calls function_name(args)
  Description of what it does.
If a trigger contains {varname}, show a concrete example like: delete reminder standup

PYTHON API
List every public function with its signature and one-line description.
Read from the API source — do not invent functions. Format:
- function_name(args) — what it does

BACKGROUND SERVICE
State whether a background.py was generated.
- If yes: describe what it does and that it starts automatically on brain restart.
- If no: say "No background service — this capability is request-driven only."

DATA STORAGE
Where data is stored, whether it persists across restarts, and any known limits.
Be accurate: files on disk DO persist across restarts. Only say "lost on restart" if data is in memory only.

LIMITATIONS / NOTES
Features from the original goal that were NOT implemented. Known edge cases from the code.
If everything was implemented, say so explicitly.

FORMAT RULES — strictly enforced:
- Plain text only. No markdown whatsoever.
- No ** bold **, no backticks, no code blocks, no # headers.
- Section names are plain uppercase labels on their own line.
- Bullet points use a plain dash (-). One level only."""

_FEEDBACK_PROMPT = """You are classifying user feedback on a just-built Code Monkey capability.

Classify the user's response as one of:
- "change": they want modifications ("add X", "fix Y", "also support Z", "change the...")
- "commit": they are satisfied ("looks good", "commit it", "perfect", "save it", "ship it")
- "discard": they want to throw it away ("not what I wanted", "start over", "scrap it")

Return ONLY valid JSON:
{"action": "change", "details": "<specific change requested>"}
{"action": "commit", "details": ""}
{"action": "discard", "details": ""}
"""

_NAME_PROMPT = """Generate a short, memorable human-readable name for a software project based on its goal.

Rules:
- 2-5 words maximum
- Title Case
- No underscores, no snake_case, no repeating the goal verbatim
- No punctuation, no quotes, no "API" or "System" unless truly essential
- Examples: "Reminder Scheduler", "Note Taker", "Weather Fetcher", "Task Manager", "Face Recognition Cache"

Return ONLY the name on a single line, nothing else.

GOAL:
"""

_CHANGE_GOAL_PROMPT = """You are writing a revised build goal for Code Monkey.

Given the original goal, the current README, the current API source, and the user's change request,
write a complete revised specification that a developer can implement without asking further questions.

Include everything from the original that should be kept, plus the requested changes.

Return ONLY the goal as plain text (no JSON, no headers, no code blocks).
"""


class ReviewAgent:
    def __init__(self):
        self.model = get_model("chat")

    def generate_name(self, goal: str) -> str:
        prompt = _NAME_PROMPT + goal.strip() + "\n\nName:"
        try:
            raw = self.model.generate(prompt, options={"temperature": 0.1, "num_predict": 60}, think=False)
            name = raw.strip().strip('"\'').split("\n")[0].strip()
            return name if name else "Code Project"
        except Exception:
            return "Code Project"

    def explain(self, goal: str, readme: str, api_code: str, workspace: str,
                test_code: str = "", commands_json: str = "") -> str:
        prompt = (
            _EXPLAIN_PROMPT
            + "\n\nORIGINAL GOAL:\n" + goal
            + "\n\nAPI SOURCE (src/api.py):\n" + (api_code[:6000] or "(not available)")
            + ("\n\nCOMMANDS (commands.json):\n" + commands_json[:2000] if commands_json else "")
            + "\n\nREADME:\n" + (readme[:1500] or "(not available)")
            + "\n\nReport:"
        )
        return self.model.generate(prompt, options={"temperature": 0.2, "num_predict": 4000}, think=False)

    def classify_feedback(self, feedback: str, goal: str) -> dict:
        prompt = (
            _FEEDBACK_PROMPT
            + "\n\nORIGINAL GOAL:\n" + goal
            + "\n\nUSER FEEDBACK:\n" + feedback
            + "\n\nReturn JSON:"
        )
        raw = self.model.generate(prompt, options={"temperature": 0.0, "num_predict": 200}, think=False)
        try:
            data = json.loads(_extract_json(raw))
            action = data.get("action")
            if action not in {"change", "commit", "discard"}:
                lower = feedback.lower()
                if any(w in lower for w in {"good", "great", "perfect", "commit", "save", "keep", "ship", "nice", "yes"}):
                    action = "commit"
                elif any(w in lower for w in {"scrap", "discard", "throw", "start over", "trash", "no"}):
                    action = "discard"
                else:
                    action = "change"
            return {"action": action, "details": str(data.get("details") or "")}
        except Exception:
            return {"action": "change", "details": feedback}

    def synthesize_change_goal(self, original_goal: str, readme: str, api_code: str, change_request: str) -> str:
        prompt = (
            _CHANGE_GOAL_PROMPT
            + "\n\nORIGINAL GOAL:\n" + original_goal
            + "\n\nCURRENT README:\n" + (readme[:2000] or "(none)")
            + "\n\nCURRENT API SOURCE:\n" + (api_code[:2000] or "(none)")
            + "\n\nUSER'S CHANGE REQUEST:\n" + change_request
            + "\n\nRevised goal:"
        )
        return self.model.generate(prompt, options={"temperature": 0.15, "num_predict": 2000}, think=False)
