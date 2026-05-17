import json
import os

from config import LEARNED_CAPABILITIES_PATH, SYSTEM_CAPABILITIES_PATH


class CapabilityRegistry:
    def __init__(
        self,
        system_path=SYSTEM_CAPABILITIES_PATH,
        learned_path=LEARNED_CAPABILITIES_PATH,
    ):
        self.system_path = system_path
        self.learned_path = learned_path
        self.system_capabilities = []
        self.learned_capabilities = []
        self.load()

    def load(self):
        self.system_capabilities = self._load_json(self.system_path)
        self.learned_capabilities = self._load_json(self.learned_path)

    def _load_json(self, path):
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")

        return data

    def _save_learned(self):
        os.makedirs(os.path.dirname(self.learned_path), exist_ok=True)

        with open(self.learned_path, "w", encoding="utf-8") as f:
            json.dump(self.learned_capabilities, f, indent=2)

    def list(self):
        return list(self.system_capabilities) + list(self.learned_capabilities)

    def get(self, name):
        for capability in self.list():
            if capability.get("name") == name:
                return capability
        return None

    def add(self, capability):
        if not capability.get("name"):
            raise ValueError("Capability requires name")

        existing = None

        for cap in self.learned_capabilities:
            if cap.get("name") == capability["name"]:
                existing = cap
                break

        if existing:
            existing.update(capability)
        else:
            self.learned_capabilities.append(capability)

        self._save_learned()
        self.load()

    def describe_for_prompt(self):
        lines = []

        for cap in self.list():
            name = cap.get("name")
            desc = cap.get("description")
            examples = ", ".join(cap.get("examples", [])[:3])
            source = "learned" if cap in self.learned_capabilities else "system"

            lines.append(
                f"- {name} [{source}]: {desc} (examples: {examples})"
            )

        return "\n".join(lines)
