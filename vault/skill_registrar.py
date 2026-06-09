import ast
import json
import re
from datetime import datetime

from models import get_model


class SkillRegistrar:
    def __init__(self, skill_registry):
        self.skill_registry = skill_registry
        self.model = get_model("planner")

    def maybe_register(self, goal, action, result):
        if result.get("status") != "success":
            return None

        action_type = action.get("type")

        if action_type not in {"write_file", "patch_file"}:
            return None

        path = action.get("path")

        if not path or not path.startswith("skills/"):
            return None

        content = action.get("content") or ""
        existing = self.find_existing_skill_for_path(path)

        metadata = self.infer_metadata(
            goal=goal,
            path=path,
            content=content,
            result=result,
            existing=existing,
        )

        skill = {
            "name": metadata["name"],
            "description": metadata["description"],
            "purpose": metadata["purpose"],
            "filename": path,
            "examples": self.merge_examples(existing, metadata.get("examples", [])),
            "arguments": metadata.get("arguments", []),
            "last_action": action_type,
            "created_or_updated_at": datetime.utcnow().isoformat(),
        }

        self.skill_registry.add_or_update(skill)

        return skill

    def find_existing_skill_for_path(self, path):
        for skill in self.skill_registry.list():
            if skill.get("filename") == path:
                return skill

        return None

    # ----------------------------
    # LLM METADATA
    # ----------------------------

    def infer_metadata(self, goal, path, content, result, existing=None):
        static_args = self.static_detect_arguments(content)

        prompt = f"""
You create clean registry metadata for a local Python skill.

User request:
{goal}

File path:
{path}

Code:
{content}

Execution result:
{result}

Existing registry entry:
{existing}

Static argument detection:
{static_args}

Return ONLY valid JSON:

{{
  "name": "short_snake_case_name",
  "purpose": "stable_snake_case_purpose",
  "description": "short human-readable description of what the skill does",
  "examples": ["short user-facing examples"],
  "arguments": [
    {{
      "name": "argument_name",
      "type": "string|int|float|bool|path|unknown",
      "required": true,
      "position": "argv[1]",
      "description": "what this argument controls",
      "default": null,
      "accepted_values": [],
      "examples": []
    }}
  ]
}}

Rules:
- Describe what the code actually does, not how it was requested.
- Do not include phrases like "write a python script", "skill created for", or "modify existing skill".
- Name should describe the behavior.
- Purpose should be stable and reusable.
- Examples should be natural user requests, not implementation instructions.
- Arguments must reflect actual runtime inputs accepted by the code.
- If the code does not require runtime input, arguments must be [].
- Prefer static argument detection when it clearly identifies sys.argv usage.
"""

        raw = self.model.generate(prompt)
        parsed = self.parse_json(raw)

        if not parsed:
            parsed = self.fallback_metadata(goal, path, content, static_args)

        return self.sanitize_metadata(parsed, goal, path, content, static_args)

    def parse_json(self, raw):
        if not raw:
            return None

        text = raw.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    # ----------------------------
    # SANITIZE
    # ----------------------------

    def sanitize_metadata(self, metadata, goal, path, content, static_args):
        name = self.clean_snake(metadata.get("name"))
        purpose = self.clean_snake(metadata.get("purpose"))
        description = self.clean_text(metadata.get("description"), max_len=200)
        examples = self.clean_examples(metadata.get("examples", []))
        arguments = self.clean_arguments(metadata.get("arguments", []))

        if not name or self.is_bad_name(name):
            name = self.name_from_path(path)

        if not purpose:
            purpose = name

        if not description or self.is_bad_text(description):
            description = self.fallback_description(content, name)

        if not examples:
            examples = [self.fallback_example(goal)]

        if static_args:
            arguments = self.merge_static_arguments(arguments, static_args)
        else:
            arguments = []

        return {
            "name": name,
            "purpose": purpose,
            "description": description,
            "examples": examples,
            "arguments": arguments,
        }

    def clean_snake(self, value):
        if not value:
            return None

        value = str(value).strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")

        return value[:80] or None

    def clean_text(self, value, max_len=200):
        if not value:
            return None

        value = str(value).strip()
        value = re.sub(r"\s+", " ", value)

        return value[:max_len] or None

    def clean_examples(self, examples):
        if not isinstance(examples, list):
            return []

        cleaned = []

        for example in examples:
            example = self.clean_text(example, max_len=120)

            if not example:
                continue

            if self.is_bad_text(example):
                continue

            if example not in cleaned:
                cleaned.append(example)

        return cleaned[:10]

    def clean_arguments(self, arguments):
        if not isinstance(arguments, list):
            return []

        cleaned = []

        for arg in arguments:
            if not isinstance(arg, dict):
                continue

            name = self.clean_snake(arg.get("name")) or f"arg{len(cleaned) + 1}"
            arg_type = str(arg.get("type") or "unknown").lower()

            if arg_type not in {"string", "int", "float", "bool", "path", "unknown"}:
                arg_type = "unknown"

            position = str(arg.get("position") or f"argv[{len(cleaned) + 1}]")

            cleaned.append({
                "name": name,
                "type": arg_type,
                "required": bool(arg.get("required", True)),
                "position": position,
                "description": self.clean_text(arg.get("description"), max_len=160) or "Runtime argument.",
                "default": arg.get("default"),
                "accepted_values": arg.get("accepted_values", []) if isinstance(arg.get("accepted_values"), list) else [],
                "examples": arg.get("examples", []) if isinstance(arg.get("examples"), list) else [],
            })

        return cleaned

    def is_bad_name(self, name):
        return name in {
            "file",
            "script",
            "skill",
            "task",
            "program",
            "main",
            "python_script",
            "generated_skill",
            "coding_task",
        }

    def is_bad_text(self, text):
        lowered = text.lower()

        bad = [
            "write a python script",
            "skill created for",
            "modify existing skill",
            "update skill registry",
            "operation:",
            "target skill:",
            "related skills:",
            "suggested change:",
            "prefer updating",
        ]

        return any(token in lowered for token in bad)

    # ----------------------------
    # STATIC ARGUMENT DETECTION
    # ----------------------------

    def static_detect_arguments(self, content):
        argv_usage = self.extract_argv_usage(content)

        if not argv_usage:
            return []

        args = []

        for item in argv_usage:
            index = item["index"]

            args.append({
                "name": f"arg{index}",
                "type": item["type"],
                "required": item["required"],
                "position": f"argv[{index}]",
                "description": f"Command-line argument {index}.",
                "default": item.get("default"),
                "accepted_values": [],
                "examples": [],
            })

        return args

    def extract_argv_usage(self, content):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        usages = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue

            if not self.is_sys_argv(node.value):
                continue

            index = self.extract_index(node.slice)

            if not isinstance(index, int) or index <= 0:
                continue

            usages[index] = {
                "index": index,
                "type": self.infer_type_from_parent(content, index),
                "required": True,
                "default": None,
            }

        return [usages[index] for index in sorted(usages)]

    def is_sys_argv(self, node):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "argv"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        )

    def extract_index(self, node):
        if isinstance(node, ast.Constant):
            return node.value

        if hasattr(ast, "Index") and isinstance(node, ast.Index):
            return self.extract_index(node.value)

        return None

    def infer_type_from_parent(self, content, index):
        if re.search(rf"int\s*\(\s*sys\.argv\[{index}\]\s*\)", content):
            return "int"

        if re.search(rf"float\s*\(\s*sys\.argv\[{index}\]\s*\)", content):
            return "float"

        return "string"

    def merge_static_arguments(self, llm_args, static_args):
        if not llm_args:
            return static_args

        by_position = {
            arg.get("position"): arg
            for arg in llm_args
            if arg.get("position")
        }

        merged = []

        for static in static_args:
            position = static.get("position")
            llm = by_position.get(position, {})

            merged.append({
                **static,
                "name": llm.get("name") or static.get("name"),
                "description": llm.get("description") or static.get("description"),
                "examples": llm.get("examples") or static.get("examples", []),
                "accepted_values": llm.get("accepted_values") or static.get("accepted_values", []),
            })

        return merged

    # ----------------------------
    # FALLBACKS
    # ----------------------------

    def fallback_metadata(self, goal, path, content, static_args):
        name = self.name_from_path(path)

        return {
            "name": name,
            "purpose": name,
            "description": self.fallback_description(content, name),
            "examples": [self.fallback_example(goal)],
            "arguments": static_args,
        }

    def name_from_path(self, path):
        filename = path.split("/")[-1]
        filename = filename.rsplit(".", 1)[0]
        filename = self.clean_snake(filename)

        if not filename or self.is_bad_name(filename):
            filename = "local_skill"

        return filename

    def fallback_description(self, content, name):
        docstring = self.extract_docstring(content)

        if docstring:
            return docstring[:200]

        return f"Runs the local skill `{name}`."

    def fallback_example(self, goal):
        text = str(goal).strip()
        text = re.sub(r"write a python script that\s+", "", text, flags=re.I)
        text = re.sub(r"write a python script to\s+", "", text, flags=re.I)
        text = re.sub(r"create a python script that\s+", "", text, flags=re.I)
        text = re.sub(r"make a python script that\s+", "", text, flags=re.I)

        return text[:120] or "run skill"

    def extract_docstring(self, content):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        return ast.get_docstring(tree)

    def merge_examples(self, existing, new_examples):
        examples = []

        if existing:
            examples.extend(existing.get("examples", []))

        examples.extend(new_examples or [])

        cleaned = []

        for example in examples:
            example = self.clean_text(example, max_len=120)

            if not example:
                continue

            if self.is_bad_text(example):
                continue

            if example not in cleaned:
                cleaned.append(example)

        return cleaned[-10:]