import subprocess
import shutil
from safety_policy import SafetyPolicy


class SystemAgent:
    def __init__(self, registry=None):
        self.registry = registry
        self.safety = SafetyPolicy()

    def run_direct(self, action, capability=None):
        if action != "dynamic_command":
            return {
                "ok": False,
                "message": f"Unsupported system action: {action}",
            }

        return self.dynamic_command(capability)

    def dynamic_command(self, capability):
        if not capability:
            return {
                "ok": False,
                "message": "Missing capability.",
            }

        command = capability.get("command")

        if not command:
            return {
                "ok": False,
                "message": f"Capability {capability.get('name')} has no command.",
            }

        safety = self.safety.validate_command(command)

        if not safety["allowed"]:
            return {
                "ok": False,
                "message": safety["reason"],
                "command": command,
            }

        return self._run_named(command)

    def _run_named(self, command):
        result = self._run(command)

        return {
            "ok": result["returncode"] == 0,
            "message": result["stdout"] or result["stderr"],
            "command": command,
        }

    def _run(self, command):
        first = command.split()[0]

        if not shutil.which(first):
            return {
                "stdout": "",
                "stderr": f"Command not found: {first}",
                "returncode": 127,
            }

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }