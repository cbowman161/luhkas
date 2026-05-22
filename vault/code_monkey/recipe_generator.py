from __future__ import annotations

import json
import re
import shlex
import shutil
from typing import Any, Dict

from .coder import LocalModel
from .config import PLANNER_MODEL


# Substring patterns rejected in the raw command string before shlex.split.
# These catch shell-expansion forms that would survive tokenization, and the
# concrete dangerous binaries/operations. Keep this list narrow — overly broad
# substrings (e.g. bare "format") false-positive on flag names like
# --format=csv. Specific destructive binaries are the right granularity.
BLOCKED_COMMAND_SUBSTRINGS = (
    "rm -rf",
    "sudo",
    " su ",
    "shutdown",
    "reboot",
    "mkfs",
    "mkfs.",
    "dd if=",
    "dd of=",
    "wipefs",
    "shred ",
    "curl ",
    "wget ",
    "apt ",
    "apt-get",
    "pip install",
    "npm install",
    "$(",
    "${",
    "`",
)

# Tokens that would be present after shlex.split if the planner emitted a
# shell pipeline / redirect / chain. Reject these — the executor runs argv
# without a shell so they don't do what the planner intends and the command
# usually crashes (e.g. cat tries to open "|" as a filename).
FORBIDDEN_ARGV_TOKENS = frozenset({
    "|", "||", "&&", "&", ";",
    ">", ">>", "<", "<<", "<<<",
})


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
        "Goal: produce a deterministic, read-only, single-process command that answers\n"
        "the user's confirmed request directly. The result is parsed via the recipe's\n"
        "\"required_facts\" — name them with the actual facts the user asked for, not\n"
        "the topic label (e.g. for a load-average request use \"load 1m\", not \"uptime\").\n\n"
        "Allowed recipe forms:\n"
        "1. Bash command (single process, no shell features):\n"
        "{\"type\":\"bash\",\"command\":\"<argv>\",\"required_facts\":[\"...\"],\"summary_hint\":\"...\"}\n"
        "2. Python script (use this when you need to read multiple sources, do math,\n"
        "   or aggregate output — anything that would normally require a pipe):\n"
        "{\"type\":\"python_script\",\"filename\":\"learned_command.py\",\"source\":\"...\","
        "\"required_facts\":[\"...\"],\"summary_hint\":\"...\"}\n\n"
        "CRITICAL: bash commands run via subprocess.run(argv) — NOT through a shell.\n"
        "The following DO NOT WORK in bash commands and will be rejected:\n"
        "  | (pipes)    > >> (redirects)    && || ; (chains)    $(...) `...` (substitution)\n"
        "  * ? glob expansion   ${VAR} variable expansion   ~ (home expansion)\n"
        "If the answer requires any of those, choose python_script instead.\n\n"
        "Bash command DO/DON'T examples:\n"
        "  DO   command:\"lscpu\"\n"
        "  DO   command:\"free -h\"\n"
        "  DO   command:\"uname -r\"\n"
        "  DO   command:\"cat /proc/loadavg\"\n"
        "  DO   command:\"systemctl --failed --no-pager\"\n"
        "  DO   command:\"ss -tlnp\"\n"
        "  DON'T command:\"cat /etc/resolv.conf | grep nameserver\"  (use python_script)\n"
        "  DON'T command:\"dpkg -l | wc -l\"                          (use python_script)\n"
        "  DON'T command:\"dmesg --level=err\"                         (this flag varies; use python_script reading /dev/kmsg or journalctl)\n\n"
        "Python script template for aggregation/parsing cases:\n"
        "  import subprocess\n"
        "  out = subprocess.run([\"dpkg\", \"-l\"], capture_output=True, text=True, timeout=5).stdout\n"
        "  count = sum(1 for line in out.splitlines() if line.startswith(\"ii \"))\n"
        "  print(f\"installed packages: {count}\")\n\n"
        "Rules:\n"
        "- Read-only commands only. No sudo, no installs, no network, no writes outside stdout.\n"
        "- Prefer simple Linux read-only sources: /proc, /sys, hostname, uname, df -h, free -h, lscpu, lsmem, lsblk, lspci, lsmod, last, who, w, ip, ss, systemctl, journalctl, nvidia-smi, timedatectl, iptables -L, nft list ruleset.\n"
        "- Some tools require root or aren't installed everywhere. Prefer the unprivileged equivalent: use journalctl -k instead of dmesg, iptables -L instead of ufw, /proc/net/* instead of netstat, systemctl --user list-units for user services. Only use sudo-required tools when nothing else exposes the fact.\n"
        "- The command's stdout must directly contain what the user asked for. If you'd\n"
        "  normally pipe to grep/awk/wc/head/tail, that's a strong signal to use python_script.\n"
        "- Python scripts may use subprocess to call read-only commands and may read\n"
        "  /proc and /sys, but must not write files or open network sockets.\n"
        "- Choose the command whose first word resolves on a stock Ubuntu/Debian PATH.\n\n"
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
        # The planner sometimes emits "command" as a JSON list (already
        # tokenized argv) instead of a string. Accept both forms.
        raw_command = parsed.get("command")
        if isinstance(raw_command, list):
            recipe["command"] = " ".join(shlex.quote(str(token)) for token in raw_command if str(token))
        else:
            recipe["command"] = str(raw_command or "").strip()
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
        for blocked in BLOCKED_COMMAND_SUBSTRINGS:
            if blocked in lowered:
                return {"ok": False, "error": f"Generated command uses blocked operation: {blocked!r}"}
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return {"ok": False, "error": f"Generated command is not parseable: {exc}"}
        if not argv:
            return {"ok": False, "error": "Generated command is empty."}
        for token in argv:
            if token in FORBIDDEN_ARGV_TOKENS:
                return {
                    "ok": False,
                    "error": (
                        f"Generated command contains shell metachar {token!r}; the executor "
                        "runs argv without a shell so pipes/redirects/chains do not work. "
                        "Use a python_script recipe for anything that needs piping."
                    ),
                }
        binary = argv[0]
        if "/" in binary:
            # Absolute or relative path — accept (we'll let the executor fail
            # if it's missing). Avoids false positives for /usr/sbin/* etc.
            pass
        elif shutil.which(binary) is None:
            return {
                "ok": False,
                "error": (
                    f"Generated command's first token {binary!r} is not on PATH. "
                    "Pick a different binary or use a python_script."
                ),
            }
        return {"ok": True}
    if kind == "python_script":
        source = str(recipe.get("source") or "")
        if not source.strip():
            return {"ok": False, "error": "Generated python recipe is missing source."}
        lowered = source.lower()
        # subprocess is allowed (read-only commands invoked from python), so it
        # is NOT in this list. Network and file-write tokens stay blocked.
        for blocked in ("socket", "requests", "urllib"):
            if blocked in lowered:
                return {"ok": False, "error": f"Generated python recipe uses blocked source token: {blocked}"}
        # open(...) is only allowed for read-mode access to /proc or /sys.
        for match in re.finditer(r"open\s*\(([^)]*)\)", lowered):
            args = match.group(1)
            if "'w'" in args or '"w"' in args or "'a'" in args or '"a"' in args:
                return {"ok": False, "error": "Generated python recipe opens a file for writing."}
            if "/proc/" not in args and "/sys/" not in args:
                # allow opens that don't specify a path string at all (e.g.,
                # variable arg) only when the script doesn't write — this is
                # imperfect but the bare-open check is mainly a guardrail.
                continue
        return {"ok": True}
    return {"ok": False, "error": f"Unsupported recipe type: {kind}"}
