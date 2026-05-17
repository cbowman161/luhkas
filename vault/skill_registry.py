import json
import os
from datetime import datetime

from config import SKILLS_REGISTRY_PATH


class SkillRegistry:
    def __init__(self, path=SKILLS_REGISTRY_PATH):
        self.path = path
        self.skills = []
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.skills = []
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{self.path} must contain a JSON list")

        self.skills = data

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, indent=2)

    def list(self):
        return list(self.skills)

    def get(self, name):
        for skill in self.skills:
            if skill.get("name") == name:
                return skill
        return None

    def add_or_update(self, skill):
        if not skill.get("name"):
            raise ValueError("Skill requires name")

        skill.setdefault("created_at", datetime.utcnow().isoformat())
        skill["updated_at"] = datetime.utcnow().isoformat()

        existing = self.get(skill["name"])

        if existing:
            existing.update(skill)
        else:
            self.skills.append(skill)

        self.save()
        self.load()

    def describe_for_prompt(self):
        if not self.skills:
            return "No registered skills."

        lines = []

        for skill in self.skills:
            lines.append(
                f"- {skill.get('name')}: {skill.get('description')} "
                f"(file: {skill.get('filename')}, examples: {', '.join(skill.get('examples', [])[:3])})"
            )

        return "\n".join(lines)
