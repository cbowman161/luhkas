import json
import re
from models import get_model
from safety_policy import SafetyPolicy


BUILDER_PROMPT = """
You design safe local assistant capabilities.

Return ONLY JSON:

{
  "name": "short_snake_case_name",
  "description": "...",
  "subsystem": "system_agent",
  "action": "dynamic_command",
  "mode": "direct",
  "command": "safe shell command",
  "examples": ["..."]
}

Rules:
- Only propose read-only commands.
- Prefer common Linux commands.
- Do not use destructive commands.
- Do not install packages.
- Do not modify files.
- If impossible safely, return:
{
  "name": null,
  "error": "reason"
}
"""


class CapabilityBuilder:
    def __init__(self, registry):
        self.registry = registry
        self.model = get_model("planner")
        self.safety = SafetyPolicy()

    def propose(self, user_input):
        safety = self.safety.classify_capability_request(user_input)

        if not safety["allowed"]:
            return {
                "ok": False,
                "requires_approval": False,
                "message": safety["reason"],
                "capability": None,
            }

        prompt = f"""{BUILDER_PROMPT}

USER REQUEST:
{user_input}

OUTPUT:
"""

        raw = self.model.generate(prompt)

        try:
            capability = json.loads(raw)
        except Exception:
            capability = self.fallback_capability(user_input)

        if capability.get("error") or not capability.get("name"):
            return {
                "ok": False,
                "requires_approval": False,
                "message": capability.get("error", "Could not design safe capability."),
                "capability": None,
            }

        command_check = self.safety.validate_command(capability.get("command", ""))

        if not command_check["allowed"]:
            return {
                "ok": False,
                "requires_approval": False,
                "message": command_check["reason"],
                "capability": None,
            }

        capability["requires_approval"] = safety["requires_approval"]

        return {
            "ok": True,
            "requires_approval": safety["requires_approval"],
            "message": "Proposed new capability.",
            "capability": capability,
        }

    def install(self, capability):
        self.registry.add(capability)

    def fallback_capability(self, user_input):
        lowered = user_input.lower()

        if "gpu" in lowered and "temp" in lowered:
            return {
                "name": "gpu_temperature",
                "description": "Show GPU temperature if supported by local tools.",
                "subsystem": "system_agent",
                "action": "dynamic_command",
                "mode": "direct",
                "command": "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits",
                "examples": [
                    "what is my gpu temperature",
                    "show gpu temp"
                ],
            }

        if "process" in lowered:
            return {
                "name": "list_processes",
                "description": "List top running processes.",
                "subsystem": "system_agent",
                "action": "dynamic_command",
                "mode": "direct",
                "command": "ps aux --sort=-%cpu | head -10",
                "examples": [
                    "show running processes",
                    "what processes are using cpu"
                ],
            }

        safe_name = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")[:40] or "custom_system_info"

        return {
            "name": safe_name,
            "error": "No safe fallback capability available.",
        }