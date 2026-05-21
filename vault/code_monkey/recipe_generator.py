from __future__ import annotations

import json
import re
import shlex
from typing import Any, Dict

from .coder import LocalModel
from .config import PLANNER_MODEL


BLOCKED_COMMAND_BITS = (
    "rm -rf",
    "sudo",
    "su ",
    "shutdown",
    "reboot",
    "mkfs",
    "dd ",
    "format",
    "wipe",
    "curl ",
    "wget ",
    "apt ",
    "apt-get",
    "pip install",
    "npm install",
    ">",
    ">>",
    "| sh",
    "| bash",
    "&",
)


def generate_learned_command_recipe(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_input = str(payload.get("input") or payload.get("user_input") or "").strip()
    if not user_input:
        return {"ok": False, "error": "Missing required field: input"}

    prompt = _recipe_prompt(user_input=user_input, payload=payload)
    model = LocalModel(model=PLANNER_MODEL, timeout=120, temperature=0.05, num_ctx=4096)
    raw = model.generate(prompt)
    parsed = _parse_json_object(raw)
    recipe = _normalize_recipe(parsed)
    validation = _validate_recipe(recipe)
    if not validation.get("ok"):
        return {
            "ok": False,
            "error": validation.get("error"),
            "raw_response": raw,
            "recipe": recipe,
        }
    return {
        "ok": True,
        "recipe": recipe,
        "raw_response": raw,
        "generator": "code_monkey_single_recipe",
    }


def _recipe_prompt(*, user_input: str, payload: Dict[str, Any]) -> str:
    context = {
        "input": user_input,
        "intent": payload.get("intent"),
        "description": payload.get("description"),
        "target": payload.get("target") or "vault",
        "confidence": payload.get("confidence"),
        "inferred": payload.get("inferred") or {},
    }
    return (
        "You are Code Monkey's learned-command recipe generator for LUHKAS Vault.\n"
        "Return ONLY one compact JSON object. No markdown. No commentary.\n\n"
        "Goal: choose the safest fast deterministic way to answer the confirmed user request.\n"
        "The recipe must be read-only, non-destructive, local-only, and suitable for reuse.\n\n"
        "Allowed recipe forms:\n"
        "1. Bash command:\n"
        "{\"type\":\"bash\",\"command\":\"...\",\"required_facts\":[\"...\"],\"summary_hint\":\"...\"}\n"
        "2. Python script:\n"
        "{\"type\":\"python_script\",\"filename\":\"learned_command.py\",\"source\":\"...\","
        "\"required_facts\":[\"...\"],\"summary_hint\":\"...\"}\n\n"
        "Rules:\n"
        "- Use one command or one Python script.\n"
        "- Prefer simple Linux read-only sources: /proc, /sys, hostname, uname, df, free, lscpu, lsmem, nvidia-smi, systemctl status/list views.\n"
        "- Do not use sudo, package installs, network calls, background processes, writes outside stdout, destructive commands, or shell scripts downloaded from anywhere.\n"
        "- Do not use pipelines unless absolutely necessary; prefer direct commands with arguments.\n"
        "- Python scripts may read local files and print facts, but must not write files or call the network.\n"
        "- If a command may not exist, choose a more portable Python script when practical.\n"
        "- required_facts should name the facts the caller should expect in stdout.\n\n"
        f"Confirmed request context:\n{json.dumps(context, indent=2, sort_keys=True)}\n"
    )


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Model did not return a JSON object.")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model returned JSON, but not an object.")
    return parsed


def _normalize_recipe(parsed: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(parsed.get("type") or "").strip()
    required = parsed.get("required_facts")
    if not isinstance(required, list):
        required = []
    recipe = {
        "type": kind,
        "required_facts": [str(item) for item in required if str(item).strip()],
        "summary_hint": str(parsed.get("summary_hint") or "").strip(),
        "timeout_seconds": int(parsed.get("timeout_seconds") or 10),
        "generator": "code_monkey_single_recipe",
    }
    if kind == "bash":
        recipe["command"] = str(parsed.get("command") or "").strip()
    elif kind == "python_script":
        recipe["filename"] = str(parsed.get("filename") or "learned_command.py").strip()
        recipe["source"] = str(parsed.get("source") or "")
    return recipe


def _validate_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    kind = recipe.get("type")
    if kind == "bash":
        command = str(recipe.get("command") or "").strip()
        if not command:
            return {"ok": False, "error": "Generated bash recipe is missing command."}
        lowered = command.lower()
        for blocked in BLOCKED_COMMAND_BITS:
            if blocked in lowered:
                return {"ok": False, "error": f"Generated command uses blocked operation: {blocked}"}
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return {"ok": False, "error": f"Generated command is not parseable: {exc}"}
        if not argv:
            return {"ok": False, "error": "Generated command is empty."}
        return {"ok": True}
    if kind == "python_script":
        source = str(recipe.get("source") or "")
        if not source.strip():
            return {"ok": False, "error": "Generated python recipe is missing source."}
        lowered = source.lower()
        for blocked in ("subprocess", "socket", "requests", "urllib", "open(", "pathlib.path("):
            if blocked in lowered and "('/proc/" not in lowered and "(\"/proc/" not in lowered and "('/sys/" not in lowered and "(\"/sys/" not in lowered:
                return {"ok": False, "error": f"Generated python recipe uses blocked source token: {blocked}"}
        return {"ok": True}
    return {"ok": False, "error": f"Unsupported recipe type: {kind}"}
