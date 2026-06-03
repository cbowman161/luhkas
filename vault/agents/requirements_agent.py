import json

from agents._json_utils import extract_json as _extract_json
from models import get_model


_OVERLAP_PROMPT = """You are checking whether a new build request overlaps with existing installed capabilities.

Return ONLY valid JSON (no markdown, no prose):
{
  "has_overlap": true or false,
  "capability_name": "internal_name or null",
  "display_name": "Human Name or null",
  "overlap_type": "same" | "partial" | "none",
  "reason": "one sentence"
}

overlap_type meanings:
- "same": existing capability already does exactly what is requested — no need to build
- "partial": existing capability is related and should be extended rather than replaced
- "none": no meaningful overlap — build something new

Only return has_overlap=true for "same" or "partial".
Be strict: only flag overlap if the core purpose genuinely matches, not just because both involve Python or files.
"""


_SYSTEM = """You are the Code Monkey requirements analyst for a local AI robotics system.

Code Monkey builds Python HTTP API capabilities (src/api.py) that:
- Accept JSON POST requests and return JSON responses
- Use only the Python standard library (no pip installs)
- Persist data under src/data/ when storage is needed

Your job: collect enough detail to write a complete, unambiguous build specification.

Ask targeted questions about:
- What endpoints the API needs and what each one does
- What data goes in and what comes out (field names, types)
- Whether anything needs to persist between calls
- Any constraints or special requirements

Rules:
- Ask ONE question per turn
- Stop after 1-4 questions when you have enough
- If the initial request already contains enough detail, go straight to done
- If the user says "go ahead", "start building", "that's all", or similar — synthesize immediately

Return ONLY valid JSON (no markdown, no prose):
{"done": false, "question": "Your single clarifying question"}
or
{"done": true, "goal": "Complete self-contained specification a developer can implement without further questions"}
"""


class RequirementsAgent:
    def __init__(self):
        self.model = get_model("chat")

    def check_overlap(self, user_request: str, installed_capabilities: list) -> dict:
        """Check if any installed capability already covers or overlaps the request.

        installed_capabilities: list of dicts with keys:
            capability_name, display_name, description, commands (list of trigger strings)
        """
        if not installed_capabilities:
            return {"has_overlap": False, "capability_name": None, "display_name": None,
                    "overlap_type": "none", "reason": "No installed capabilities."}

        cap_descriptions = []
        for cap in installed_capabilities:
            triggers = ", ".join(f'"{t}"' for t in (cap.get("commands") or [])[:6])
            cap_descriptions.append(
                f"- {cap['display_name']} (internal: {cap['capability_name']})\n"
                f"  Description: {cap.get('description', '')[:200]}\n"
                f"  Commands: {triggers or 'none'}"
            )

        prompt = (
            _OVERLAP_PROMPT
            + "\n\nINSTALLED CAPABILITIES:\n" + "\n".join(cap_descriptions)
            + "\n\nNEW REQUEST:\n" + user_request
            + "\n\nJSON:"
        )
        raw = self.model.generate(
            prompt,
            options={"temperature": 0.0, "num_predict": 300},
            think=False,
        )
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            overlap_type = data.get("overlap_type", "none")
            has_overlap = bool(data.get("has_overlap")) and overlap_type in {"same", "partial"}
            return {
                "has_overlap": has_overlap,
                "capability_name": data.get("capability_name"),
                "display_name": data.get("display_name"),
                "overlap_type": overlap_type,
                "reason": str(data.get("reason") or ""),
            }
        except Exception:
            return {"has_overlap": False, "capability_name": None, "display_name": None,
                    "overlap_type": "none", "reason": "Overlap check failed."}

    def start(self, initial_request: str) -> dict:
        prompt = (
            _SYSTEM
            + "\n\nUSER'S INITIAL REQUEST:\n" + initial_request
            + "\n\nFirst turn. Decide whether you need clarification or have enough to synthesize.\nReturn JSON:"
        )
        return self._call(prompt, initial_request)

    def continue_conversation(self, initial_request: str, conversation: list) -> dict:
        lines = []
        for turn in conversation:
            prefix = "Code Monkey" if turn["role"] == "assistant" else "User"
            lines.append(f"{prefix}: {turn['text']}")
        prompt = (
            _SYSTEM
            + "\n\nUSER'S INITIAL REQUEST:\n" + initial_request
            + "\n\nCONVERSATION SO FAR:\n" + "\n".join(lines)
            + "\n\nDecide: ask another question or synthesize the final goal.\nReturn JSON:"
        )
        return self._call(prompt, initial_request)

    def _call(self, prompt: str, fallback_goal: str) -> dict:
        raw = self.model.generate(prompt, options={"temperature": 0.15, "num_predict": 800}, think=False)
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            if data.get("done"):
                return {"done": True, "goal": str(data.get("goal") or fallback_goal).strip()}
            question = str(data.get("question") or "").strip()
            if not question:
                return {"done": True, "goal": fallback_goal}
            return {"done": False, "question": question}
        except Exception:
            return {"done": True, "goal": fallback_goal}
