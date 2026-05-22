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


MAX_PLANNER_RETRIES = 2


def generate_learned_command_recipe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Plan a recipe for the confirmed request. If the generated recipe fails
    validation, re-prompt the planner with the rejection reason as feedback
    (up to MAX_PLANNER_RETRIES). Temperature is bumped on retries so the
    planner doesn't just repeat the same mistake."""
    user_input = str(payload.get("input") or payload.get("user_input") or "").strip()
    if not user_input:
        return {"ok": False, "error": "Missing required field: input"}

    last_recipe: Dict[str, Any] | None = None
    last_error: str = ""
    last_raw: str = ""

    for attempt in range(MAX_PLANNER_RETRIES + 1):
        if attempt == 0:
            prompt = _recipe_prompt(user_input=user_input, payload=payload)
            temperature = 0.05
        else:
            prompt = _retry_prompt(
                user_input=user_input,
                payload=payload,
                previous_recipe=last_recipe or {},
                rejection=last_error,
            )
            temperature = 0.25 + 0.15 * (attempt - 1)

        model = LocalModel(
            model=PLANNER_MODEL,
            timeout=120,
            temperature=temperature,
            num_ctx=4096,
            # Recipes are short JSON; cap output well above the longest
            # python_script template we'd produce, but not so high that
            # format=json's per-token validation balloons latency.
            num_predict=1024,
        )
        try:
            raw = model.generate(prompt, response_format="json")
        except Exception as exc:
            last_error = f"planner call failed: {exc}"
            last_raw = ""
            continue
        last_raw = raw

        try:
            parsed = _parse_json_object(raw)
        except ValueError as exc:
            last_error = f"planner returned unparseable JSON: {exc}"
            continue

        recipe = _normalize_recipe(parsed)
        validation = _validate_recipe(recipe)
        if validation.get("ok"):
            return {
                "ok": True,
                "recipe": recipe,
                "raw_response": raw,
                "generator": "code_monkey_single_recipe",
                "attempts": attempt + 1,
            }
        last_recipe = recipe
        last_error = validation.get("error") or "unknown validator error"

    return {
        "ok": False,
        "error": f"After {MAX_PLANNER_RETRIES + 1} planner attempts: {last_error}",
        "raw_response": last_raw,
        "recipe": last_recipe,
        "attempts": MAX_PLANNER_RETRIES + 1,
    }


def _retry_prompt(*, user_input: str, payload: Dict[str, Any], previous_recipe: Dict[str, Any], rejection: str) -> str:
    return (
        _recipe_prompt(user_input=user_input, payload=payload)
        + "\n\n----- RETRY -----\n"
        + "Your previous attempt was REJECTED by the validator:\n"
        + f"  Rejected recipe: {json.dumps(previous_recipe, indent=2, sort_keys=True)}\n"
        + f"  Rejection reason: {rejection}\n\n"
        + "Try AGAIN with a different approach. Concretely:\n"
        + "- If rejection mentions a shell metachar (|, >, ;, &&, etc.) or piping, "
        + "switch to type=python_script and aggregate inside Python.\n"
        + "- If rejection mentions PATH or 'not on PATH', pick a binary that exists "
        + "on stock Ubuntu/Debian (e.g. systemctl, ip, ss, lsblk, lsmod, lspci, "
        + "journalctl) or use a python_script.\n"
        + "- If rejection mentions a blocked operation, choose a non-destructive "
        + "equivalent.\n\n"
        + "Return the corrected JSON object only.\n"
    )


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
        "Common patterns the planner often gets wrong — pick the CORRECT form:\n"
        "  • DNS servers — DON'T `cat /etc/resolv.conf | grep nameserver`.\n"
        "    DO python_script that reads /etc/resolv.conf line-by-line and prints lines starting with 'nameserver'.\n"
        "  • Firewall active — DON'T `iptables -L` (needs root) or `ufw status` (often not installed).\n"
        "    DO python_script that calls `subprocess.run([\"systemctl\", \"is-active\", svc], ...)` for each of ufw, firewalld, nftables, iptables and reports which are active.\n"
        "  • Counting items in command output — DON'T `dpkg -l | wc -l` etc.\n"
        "    DO python_script that runs the producer command via subprocess, splits stdout lines in Python, and counts.\n"
        "  • Grepping a file — DON'T `grep PATTERN /path` if you want only matching lines from a config file.\n"
        "    DO python_script that opens the file and filters in Python (better error handling, no shell).\n"
        "  • dmesg — DON'T use it (requires root on most kernels).\n"
        "    DO `journalctl -k --no-pager -n 50` (unprivileged equivalent for recent kernel messages).\n\n"
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
        # Syntax-check first. The planner occasionally emits truncated source
        # (cut off mid-f-string when generation is bounded). compile() catches
        # those before they get saved and fail at execution time.
        try:
            compile(source, "<learned-recipe>", "exec")
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": (
                    f"Generated python source has a syntax error at line {exc.lineno}: "
                    f"{exc.msg}. The recipe is likely truncated — emit the whole script."
                ),
            }
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
