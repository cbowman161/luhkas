"""
CommandAgent — deterministic command routing for installed capability bundles.

Sits before the LLM planner in vault_runtime.handle(). If the user input
matches a registered command trigger, the api function is called directly with
no LLM involvement. Falls through to None if no match.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


class CommandAgent:
    def __init__(self, installed_dir: Path):
        self.installed_dir = Path(installed_dir)
        self._commands = []   # {trigger, action, args, capability_name, description, has_vars}
        self._modules = {}    # capability_name → imported module
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self):
        self._commands = []
        self._modules = {}
        if not self.installed_dir.exists():
            return
        for cap_dir in sorted(self.installed_dir.iterdir()):
            if cap_dir.is_dir():
                self._load_capability(cap_dir)

    def _load_capability(self, cap_dir: Path):
        commands_file = cap_dir / "commands.json"
        api_file = cap_dir / "api.py"
        if not commands_file.exists() or not api_file.exists():
            return
        try:
            data = json.loads(commands_file.read_text(encoding="utf-8"))
            capability_name = cap_dir.name
            module = self._import_api(capability_name, api_file)
            if module is None:
                return
            self._modules[capability_name] = module
            for cmd in data.get("commands", []):
                triggers = cmd.get("triggers", [])
                action = cmd.get("action", "")
                args = cmd.get("args", {})
                if not action or not triggers:
                    continue
                for trigger in triggers:
                    self._commands.append({
                        "trigger": trigger.lower(),
                        "action": action,
                        "args": args,
                        "capability_name": capability_name,
                        "description": cmd.get("description", ""),
                        "has_vars": "{" in trigger,
                    })
        except Exception:
            pass

    def _import_api(self, capability_name: str, api_path: Path):
        try:
            mod_name = f"_installed_cap_{capability_name}"
            spec = importlib.util.spec_from_file_location(mod_name, api_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, user_input: str) -> dict | None:
        lowered = user_input.lower().strip()

        # Two passes: exact/prefix first (no-var triggers), then pattern (var triggers).
        # This prevents partial pattern matches shadowing exact commands.
        for pass_num in (0, 1):
            for cmd in self._commands:
                trigger = cmd["trigger"]
                has_vars = cmd["has_vars"]

                if pass_num == 0 and has_vars:
                    continue
                if pass_num == 1 and not has_vars:
                    continue

                if not has_vars:
                    if lowered == trigger or lowered.startswith(trigger + " "):
                        extracted = lowered[len(trigger):].strip()
                        return {"cmd": cmd, "extracted": extracted, "args": {}}
                else:
                    var_names = re.findall(r"\{([^}]+)\}", trigger)
                    regex = re.escape(trigger)
                    regex = re.sub(r"\\\{[^}]+\\\}", r"(.+?)", regex)
                    regex = regex.rstrip(r"\?") + r"(.*)$"
                    # Build a pattern where all but last group are non-greedy,
                    # last captures the remainder.
                    parts = trigger.split("{")
                    pattern = "^"
                    for i, part in enumerate(parts):
                        if "}" in part:
                            varpart, rest = part.split("}", 1)
                            if i < len(parts) - 1:
                                pattern += r"(.+?)" + re.escape(rest)
                            else:
                                pattern += r"(.+)" + re.escape(rest)
                        else:
                            pattern += re.escape(part)
                    pattern += "$"
                    m = re.match(pattern, lowered, re.IGNORECASE)
                    if m:
                        captured = {var_names[i]: m.group(i + 1).strip()
                                    for i in range(len(var_names))}
                        return {"cmd": cmd, "extracted": "", "args": captured}

        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, match_result: dict) -> str:
        cmd = match_result["cmd"]
        extracted = match_result.get("extracted", "")
        captured = match_result.get("args", {})
        capability_name = cmd["capability_name"]
        action = cmd["action"]
        args_spec = cmd.get("args", {})

        module = self._modules.get(capability_name)
        if module is None:
            return f"Capability '{capability_name}' is not loaded."

        fn = getattr(module, action, None)
        if fn is None:
            return f"Action '{action}' not found in '{capability_name}'."

        try:
            if captured:
                result = fn(**captured)
            elif not args_spec:
                result = fn()
            elif extracted:
                first_arg = next(iter(args_spec))
                result = fn(**{first_arg: extracted})
            else:
                result = fn()
        except Exception as exc:
            return f"Error running {action}: {exc}"

        return self._format_result(result)

    def _format_result(self, result) -> str:
        if not isinstance(result, dict):
            return str(result)

        if not result.get("ok"):
            return result.get("error") or result.get("message") or "Error."

        data = result.get("data")
        message = result.get("message", "")

        if data is None:
            return message or "Done."

        if isinstance(data, list):
            if not data:
                return (message + " — none found.").strip()
            lines = [message] if message else []
            for item in data:
                if isinstance(item, dict):
                    parts = [f"{k}: {v}" for k, v in item.items() if v is not None]
                    lines.append("- " + ", ".join(parts))
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)

        if isinstance(data, dict):
            lines = [message] if message else []
            for k, v in data.items():
                if v is not None:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        return f"{message}: {data}".strip(": ")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def handle(self, user_input: str) -> dict | None:
        """Return response dict if handled, None to fall through to LLM."""
        m = self.match(user_input)
        if m is None:
            return None
        message = self.execute(m)
        return {"mode": "direct", "message": message}

    def register(self, capability_name: str, bundle_dir: Path) -> bool:
        """Hot-register a newly installed capability without full reload."""
        self._commands = [c for c in self._commands
                          if c["capability_name"] != capability_name]
        self._modules.pop(capability_name, None)
        self._load_capability(Path(bundle_dir))
        return capability_name in self._modules

    def describe(self) -> str:
        """Human-readable list of all registered commands."""
        if not self._commands:
            return "No custom commands registered."
        seen = set()
        lines = []
        for cmd in self._commands:
            key = (cmd["capability_name"], cmd["action"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  [{cmd['capability_name']}] {cmd['trigger']} → {cmd['action']}")
        return "\n".join(lines)
