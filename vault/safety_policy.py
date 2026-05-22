class SafetyPolicy:
    # Substring matches against the lower-cased command. Keep narrow —
    # generic words like "format" / "wipe" appear in legitimate read-only
    # flags (e.g. `nvidia-smi --format=csv`) and produce false positives.
    BLOCKED_WORDS = [
        "rm -rf",
        "shutdown",
        "reboot",
        "mkfs",
        "mkfs.",
        "dd if=",
        "dd of=",
        "wipefs",
        "shred ",
        "delete all",
        "fork bomb",
        ":(){",
        "chmod -R 777 /",
        "chown -R",
    ]

    READ_ONLY_HINTS = [
        "show",
        "get",
        "list",
        "check",
        "read",
        "display",
        "measure",
        "usage",
        "status",
        "temperature",
        "version",
        "info",
    ]

    def classify_capability_request(self, text):
        lowered = text.lower()

        for blocked in self.BLOCKED_WORDS:
            if blocked in lowered:
                return {
                    "allowed": False,
                    "requires_approval": False,
                    "reason": f"Blocked dangerous request: {blocked}",
                }

        if any(word in lowered for word in self.READ_ONLY_HINTS):
            return {
                "allowed": True,
                "requires_approval": False,
                "reason": "Read-only system capability appears safe.",
            }

        return {
            "allowed": True,
            "requires_approval": True,
            "reason": "Capability may change system state and needs approval.",
        }

    def validate_command(self, command):
        lowered = command.lower()

        for blocked in self.BLOCKED_WORDS:
            if blocked in lowered:
                return {
                    "allowed": False,
                    "reason": f"Blocked dangerous command: {blocked}",
                }

        return {
            "allowed": True,
            "reason": "Command passed basic safety validation.",
        }