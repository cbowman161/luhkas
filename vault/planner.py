import json
from models import get_model
from evolution_advisor import EvolutionAdvisor


PLANNER_PROMPT = """
You are a capability and skill router.

You receive:
1. A user request
2. Available capabilities
3. Existing skills

Return ONLY JSON:

{
  "intent": "use_capability" | "use_skill" | "create_capability" | "chat",
  "capability": "capability_name_or_null",
  "skill": "skill_name_or_null",
  "reason": "short reason"
}

Rules:
- If the user is asking for something an existing skill already does, choose use_skill.
- If the user asks to overwrite, modify, change, or improve an existing skill, choose use_capability with coding_task.
- If a capability directly matches, choose use_capability.
- If nothing matches but it is something the system could learn to do locally, choose create_capability.
- Otherwise choose chat.
- Never invent skill names or capability names.
"""


class Planner:
    def __init__(self, registry, skill_registry):
        self.model = get_model("planner")
        self.registry = registry
        self.skill_registry = skill_registry
        self.evolution_advisor = EvolutionAdvisor(skill_registry, registry)

    def decide(self, user_input):
        evolution = self.evolution_advisor.advise(user_input)

        if evolution.get("decision") == "chat":
            chat = self.registry.get("chat")
            if chat:
                return {
                    "intent": "use_capability",
                    "capability": "chat",
                    "subsystem": chat["subsystem"],
                    "action": chat["action"],
                    "mode": chat.get("mode", "direct"),
                    "reason": evolution.get("reason", "evolution advisor: chat"),
                    "evolution": evolution,
                }

        routed = self.route_evolution_decision(user_input, evolution)

        if routed:
            return routed

        prompt = f"""{PLANNER_PROMPT}

AVAILABLE CAPABILITIES:
{self.registry.describe_for_prompt()}

EXISTING SKILLS:
{self.skill_registry.describe_for_prompt()}

USER REQUEST:
{user_input}

OUTPUT:
"""

        raw = self.model.generate(prompt)

        try:
            parsed = json.loads(raw)
        except Exception:
            return self.fallback(user_input)

        if parsed.get("intent") == "use_skill":
            skill = self.skill_registry.get(parsed.get("skill"))

            if skill:
                return {
                    "intent": "use_skill",
                    "skill": skill["name"],
                    "subsystem": "skill_registry",
                    "action": "confirm_existing_skill",
                    "mode": "direct",
                    "reason": parsed.get("reason", ""),
                }

        if parsed.get("intent") == "use_capability":
            cap = self.registry.get(parsed.get("capability"))

            if cap:
                return {
                    "intent": "use_capability",
                    "capability": cap["name"],
                    "subsystem": cap["subsystem"],
                    "action": cap["action"],
                    "mode": cap.get("mode", "direct"),
                    "reason": parsed.get("reason", ""),
                }

        if parsed.get("intent") == "create_capability":
            return {
                "intent": "create_capability",
                "subsystem": "capability_builder",
                "mode": "direct",
                "reason": parsed.get("reason", ""),
            }

        return self.fallback(user_input)

    def route_evolution_decision(self, user_input, evolution):
        decision = evolution.get("decision")
        confidence = evolution.get("confidence", 0.0)

        if decision == "use_skill" and evolution.get("target_skill"):
            return {
                "intent": "use_skill",
                "skill": evolution["target_skill"],
                "subsystem": "skill_registry",
                "action": "confirm_existing_skill",
                "mode": "direct",
                "reason": evolution.get("reason", "evolution advisor skill match"),
                "evolution": evolution,
            }

        if decision in {"expand_skill", "combine_skills"} and confidence >= 0.25:
            target_skill = evolution.get("target_skill")

            if not target_skill:
                related = evolution.get("related_skills", [])
                if related:
                    target_skill = related[0]

            if not target_skill:
                return None

            return {
                "intent": "evolve_skill",
                "skill": target_skill,
                "related_skills": evolution.get("related_skills", []),
                "subsystem": "skill_registry",
                "action": decision,
                "mode": "direct",
                "reason": evolution.get("reason", ""),
                "suggested_change": evolution.get("suggested_change", ""),
                "evolution": evolution,
            }

        if decision == "use_capability" and evolution.get("target_capability"):
            cap = self.registry.get(evolution["target_capability"])

            if cap:
                return {
                    "intent": "use_capability",
                    "capability": cap["name"],
                    "subsystem": cap["subsystem"],
                    "action": cap["action"],
                    "mode": cap.get("mode", "direct"),
                    "reason": evolution.get("reason", "evolution advisor capability match"),
                    "evolution": evolution,
                }

        if decision == "expand_capability" and evolution.get("target_capability"):
            return {
                "intent": "evolve_capability",
                "capability": evolution.get("target_capability"),
                "related_capabilities": evolution.get("related_capabilities", []),
                "subsystem": "capability_builder",
                "action": "expand_capability",
                "mode": "direct",
                "reason": evolution.get("reason", ""),
                "suggested_change": evolution.get("suggested_change", ""),
                "evolution": evolution,
            }

        return None

    def fallback(self, user_input):
        lowered = user_input.lower()

        for skill in self.skill_registry.list():
            text = (
                skill.get("name", "")
                + " "
                + skill.get("description", "")
                + " "
                + " ".join(skill.get("examples", []))
            ).lower()

            if any(word in text for word in lowered.split() if len(word) > 2):
                return {
                    "intent": "use_skill",
                    "skill": skill["name"],
                    "subsystem": "skill_registry",
                    "action": "confirm_existing_skill",
                    "mode": "direct",
                    "reason": "fallback skill match",
                }

        for cap in self.registry.list():
            text = (
                cap.get("name", "")
                + " "
                + cap.get("description", "")
                + " "
                + " ".join(cap.get("examples", []))
            ).lower()

            if any(word in text for word in lowered.split() if len(word) > 2):
                return {
                    "intent": "use_capability",
                    "capability": cap["name"],
                    "subsystem": cap["subsystem"],
                    "action": cap["action"],
                    "mode": cap.get("mode", "direct"),
                    "reason": "fallback capability match",
                }

        chat = self.registry.get("chat")

        return {
            "intent": "use_capability",
            "capability": "chat",
            "subsystem": chat["subsystem"],
            "action": chat["action"],
            "mode": chat.get("mode", "direct"),
            "reason": "fallback to chat",
        }