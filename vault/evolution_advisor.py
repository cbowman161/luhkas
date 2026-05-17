import json
from models import get_model


ADVISOR_PROMPT = """
You are a registry evolution advisor for a local agent system.

Your job is to decide whether a new user request is conceptually related to existing
skills or capabilities.

IMPORTANT:
Do NOT treat two requests as related just because they both mention:
- Python
- script
- command line arguments
- file
- run
- CLI

Those are implementation details, not conceptual skill purpose.

Focus on the actual user goal:
- printing text
- adding numbers
- checking system status
- reading files
- converting data
- etc.

Return ONLY JSON:

{
  "decision": "use_skill" | "expand_skill" | "combine_skills" | "use_capability" | "expand_capability" | "create_new" | "chat",
  "target_skill": "skill_name_or_null",
  "target_capability": "capability_name_or_null",
  "related_skills": ["skill_name"],
  "related_capabilities": ["capability_name"],
  "relationship": "same_purpose" | "variant" | "adjacent" | "implementation_overlap_only" | "unrelated",
  "confidence": 0.0,
  "reason": "short reason",
  "suggested_change": "what should be updated, expanded, or combined"
}

Critical:
- If the input is conversational (greetings, questions, opinions, small talk — not a build/code/task request), choose "chat" regardless of existing skills.
- If relationship is "implementation_overlap_only" or "unrelated" AND the input is a genuine build/task request, choose create_new.
- Never invent skill or capability names.
"""


class EvolutionAdvisor:
    def __init__(self, skill_registry, capability_registry):
        self.advisor_model = get_model("chat")
        self.verify_model = get_model("router")
        self.skill_registry = skill_registry
        self.capability_registry = capability_registry

    def advise(self, user_input):
        prompt = f"""{ADVISOR_PROMPT}

USER REQUEST:
{user_input}

EXISTING SKILLS:
{self.describe_skills()}

EXISTING CAPABILITIES:
{self.describe_capabilities()}

OUTPUT:
"""

        raw = self.advisor_model.generate(prompt, think=False)

        try:
            parsed = json.loads(raw)
        except Exception:
            return self.fallback(user_input)

        return self.sanitize(parsed, user_input)

    def describe_skills(self):
        skills = self.skill_registry.list()

        if not skills:
            return "No registered skills."

        lines = []

        for skill in skills:
            lines.append(json.dumps({
                "name": skill.get("name"),
                "description": skill.get("description"),
                "filename": skill.get("filename"),
                "examples": self.clean_examples(skill.get("examples", [])),
                "arguments": skill.get("arguments", []),
            }))

        return "\n".join(lines)

    def describe_capabilities(self):
        capabilities = self.capability_registry.list()

        if not capabilities:
            return "No registered capabilities."

        lines = []

        for capability in capabilities:
            lines.append(json.dumps({
                "name": capability.get("name"),
                "description": capability.get("description"),
                "examples": capability.get("examples", []),
                "subsystem": capability.get("subsystem"),
                "action": capability.get("action"),
            }))

        return "\n".join(lines)

    def clean_examples(self, examples):
        cleaned = []

        for example in examples:
            if not example:
                continue

            lowered = example.lower()

            noisy = [
                "update skill registry intelligently",
                "operation:",
                "target skill:",
                "related skills:",
                "suggested change:",
                "prefer updating",
            ]

            if any(token in lowered for token in noisy):
                continue

            cleaned.append(example)

        return cleaned[-5:]

    def verify_relationship(self, user_input, skill):
        prompt = f"""
You are verifying whether two tasks have the SAME PURPOSE.

TASK A:
{user_input}

TASK B:
Name: {skill.get("name")}
Description: {skill.get("description")}
Examples: {self.clean_examples(skill.get("examples", []))}
Arguments: {skill.get("arguments", [])}

Answer ONLY one word:
same
different

Be strict:
- "print hello" vs "add two numbers" = different
- "print hello" vs "print hello world" = same
- "print hello" vs "print custom message" = same
- "add two numbers" vs "multiply numbers" = different
"""

        raw = self.verify_model.generate(prompt).strip().lower()

        first_word = raw.split()[0] if raw.split() else ""

        return first_word == "same"

    def sanitize(self, data, user_input):
        decision = data.get("decision")

        valid_decisions = {
            "use_skill",
            "expand_skill",
            "combine_skills",
            "use_capability",
            "expand_capability",
            "create_new",
            "chat",
        }

        if decision not in valid_decisions:
            decision = "create_new"

        relationship = data.get("relationship")

        valid_relationships = {
            "same_purpose",
            "variant",
            "adjacent",
            "implementation_overlap_only",
            "unrelated",
        }

        if relationship not in valid_relationships:
            relationship = "unrelated"

        if relationship in {"implementation_overlap_only", "unrelated", "adjacent"}:
            decision = "create_new"

        target_skill = data.get("target_skill")
        target_capability = data.get("target_capability")

        if target_skill and not self.skill_registry.get(target_skill):
            target_skill = None

        if target_capability and not self.capability_registry.get(target_capability):
            target_capability = None

        related_skills = [
            name for name in data.get("related_skills", [])
            if self.skill_registry.get(name)
        ]

        related_capabilities = [
            name for name in data.get("related_capabilities", [])
            if self.capability_registry.get(name)
        ]

        if decision in {"use_skill", "expand_skill", "combine_skills"}:
            if not target_skill and related_skills:
                target_skill = related_skills[0]

            if not target_skill:
                decision = "create_new"

        if decision in {"use_capability", "expand_capability"}:
            if not target_capability and related_capabilities:
                target_capability = related_capabilities[0]

            if not target_capability:
                decision = "create_new"

        if decision in {"use_skill", "expand_skill", "combine_skills"} and target_skill:
            skill = self.skill_registry.get(target_skill)

            if skill and not self.verify_relationship(user_input, skill):
                decision = "create_new"
                target_skill = None
                related_skills = []
                relationship = "unrelated"

        return {
            "decision": decision,
            "target_skill": target_skill,
            "target_capability": target_capability,
            "related_skills": related_skills,
            "related_capabilities": related_capabilities,
            "relationship": relationship,
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "reason": data.get("reason", ""),
            "suggested_change": data.get("suggested_change", ""),
        }

    def fallback(self, user_input):
        return {
            "decision": "create_new",
            "target_skill": None,
            "target_capability": None,
            "related_skills": [],
            "related_capabilities": [],
            "relationship": "unrelated",
            "confidence": 0.0,
            "reason": "Advisor fallback.",
            "suggested_change": "",
        }