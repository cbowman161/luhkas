from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from .coder import LocalModel
from .config import PLANNER_MODEL


SMOKE_TIMEOUT_SECONDS = 8


# Substring patterns rejected in the raw command string before shlex.split.
# These catch shell-expansion forms that would survive tokenization, and the
# concrete dangerous binaries/operations. Keep this list narrow — overly broad
# substrings (e.g. bare "format") false-positive on flag names like
# --format=csv. Specific destructive binaries are the right granularity.
#
# Note: `sudo` and `apt-get` are NOT in this list. Recipes are allowed to
# install missing tools via `sudo apt-get install ...`; that path is gated by
# _is_safe_sudo_install below — any OTHER sudo use is rejected.
BLOCKED_COMMAND_SUBSTRINGS = (
    "rm -rf",
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
    "pip install",
    "npm install",
    "$(",
    "${",
    "`",
)


# Recognized safe sudo patterns inside a recipe. The recipe runs via
# subprocess.run(argv) with no shell, so argv[0] must be exactly "sudo" and
# argv[1:] must form one of these whitelisted operations. Anything else with
# sudo is rejected.
SAFE_SUDO_PATTERNS = (
    # apt-get install ...
    ("apt-get", "install"),
    ("apt", "install"),
    # apt-get update (often needed before an install)
    ("apt-get", "update"),
    ("apt", "update"),
)


def _is_safe_sudo_install(argv: list[str]) -> bool:
    """Return True only when argv represents a safe sudo invocation —
    currently limited to apt-get install/update style commands."""
    if not argv or argv[0] != "sudo":
        return False
    # Strip benign sudo flags like -n, -E, -H from the front.
    rest = list(argv[1:])
    while rest and rest[0].startswith("-"):
        rest.pop(0)
    if len(rest) < 2:
        return False
    for prefix in SAFE_SUDO_PATTERNS:
        if tuple(rest[:len(prefix)]) == prefix:
            return True
    return False

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
        if not validation.get("ok"):
            last_recipe = recipe
            last_error = validation.get("error") or "unknown validator error"
            continue
        smoke = _smoke_test_recipe(recipe)
        if not smoke.get("ok"):
            last_recipe = recipe
            last_error = smoke.get("error") or "unknown smoke error"
            continue
        return {
            "ok": True,
            "recipe": recipe,
            "raw_response": raw,
            "generator": "code_monkey_single_recipe",
            "attempts": attempt + 1,
        }

    # If every attempt died because a binary the planner picked isn't on the
    # host, surface that as structured data so the brain can offer to install
    # it rather than just reporting a generic failure.
    missing_binary = _detect_missing_binary(last_error)
    return {
        "ok": False,
        "error": f"After {MAX_PLANNER_RETRIES + 1} planner attempts: {last_error}",
        "raw_response": last_raw,
        "recipe": last_recipe,
        "attempts": MAX_PLANNER_RETRIES + 1,
        "missing_binary": missing_binary,
    }


_MISSING_BINARY_PATTERNS = (
    re.compile(r"(?:first token )?'([A-Za-z0-9_.\-]+)'(?: is)? not on PATH", re.IGNORECASE),
    re.compile(r"binary not found.*?'([A-Za-z0-9_.\-]+)'", re.IGNORECASE),
    re.compile(r"FileNotFoundError.*?'([A-Za-z0-9_.\-]+)'", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_.\-]+): command not found", re.IGNORECASE),
    re.compile(r"No such file or directory: '([A-Za-z0-9_.\-]+)'", re.IGNORECASE),
)


def _detect_missing_binary(error_text: str) -> str | None:
    """Extract a binary name from a smoke/validator error if the failure was
    'tool not installed'. Returns None for other failure modes."""
    if not error_text:
        return None
    for pattern in _MISSING_BINARY_PATTERNS:
        match = pattern.search(error_text)
        if match:
            candidate = match.group(1).strip()
            # Filter out obvious non-binaries — paths, python keywords, etc.
            if candidate and "/" not in candidate and candidate not in {
                "python3", "python", "sh", "bash"
            }:
                return candidate
    return None


def _retry_prompt(*, user_input: str, payload: Dict[str, Any], previous_recipe: Dict[str, Any], rejection: str) -> str:
    return (
        _recipe_prompt(user_input=user_input, payload=payload)
        + "\n\n----- RETRY -----\n"
        + "Your previous attempt was REJECTED:\n"
        + f"  Rejected recipe: {json.dumps(previous_recipe, indent=2, sort_keys=True)}\n"
        + f"  Rejection reason: {rejection}\n\n"
        + "Pick a structurally DIFFERENT approach for this attempt. Do not repeat\n"
        + "the same shape of recipe that was just rejected. Use the rejection text\n"
        + "above as the only ground truth for what didn't work — adjust to address\n"
        + "exactly that.\n\n"
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
        "Execution model — read this carefully:\n"
        "- bash recipes run via subprocess.run(argv, shell=False). There is NO shell.\n"
        "  Pipes (|), redirects (> >> <), chains (&& || ;), command substitution\n"
        "  ($(...) `...`), glob expansion (* ?), variable expansion (${VAR}, ~) all\n"
        "  FAIL — those characters are passed to the command as literal arguments.\n"
        "  If the task needs any of those, use a python_script instead.\n"
        "- python_script recipes run via `python3 <file>` and may call\n"
        "  subprocess.run([...]) themselves to invoke other read-only tools.\n"
        "- Recipes must otherwise be read-only and local: no destructive commands,\n"
        "  no network sockets, no writes outside stdout.\n"
        "- The only privileged operation allowed is installing missing packages\n"
        "  via apt — `sudo apt-get install -y <package>` (or `sudo apt install ...`).\n"
        "  Use this only when a needed tool isn't on PATH. Anything else with sudo\n"
        "  is rejected.\n"
        "- Output must go to stdout. required_facts names what the caller should\n"
        "  expect to parse out of stdout.\n\n"
        "How you'll learn what works:\n"
        "- After you emit a recipe, the system smoke-tests it: it runs the recipe\n"
        "  with a short timeout and checks exit code and stdout.\n"
        "- If the recipe fails (wrong flag, missing binary, empty output, syntax\n"
        "  error, etc.), you'll be re-prompted with the rejected recipe and the\n"
        "  actual stderr from the process. Pick a structurally DIFFERENT approach\n"
        "  for the retry — change the tool, change the form (bash → python_script),\n"
        "  or read the underlying /proc or /sys file directly instead of invoking\n"
        "  a high-level tool whose syntax varies by distro/version.\n"
        "- Prefer reading kernel-exposed files (/proc, /sys) over tools when there\n"
        "  is a direct equivalent — those files are stable across distros.\n\n"
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


def _smoke_test_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Actually run the recipe in a sandboxed subprocess with a short timeout.
    Returns {"ok": True} if it ran clean (rc=0, non-empty stdout); otherwise
    {"ok": False, "error": "..."} with the runtime failure as the reason so
    the retry loop can pass it back to the planner.

    Catches the class of bug a structural validator can't — semantically
    wrong flags (e.g. `ps -o etime,pid,comm --sort=-etime --no-headers` on a
    busybox vs procps where flag names differ), missing optional features,
    permission requirements that only show up at runtime, etc.
    """
    kind = recipe.get("type")
    # Recipes that perform an apt install can legitimately take 30-90s; bump
    # the smoke timeout adaptively when we see those tokens.
    def _timeout_for(command_or_source: str) -> int:
        lowered = (command_or_source or "").lower()
        if "apt-get install" in lowered or "apt install" in lowered:
            return 120
        return SMOKE_TIMEOUT_SECONDS

    try:
        if kind == "bash":
            command = str(recipe.get("command") or "").strip()
            try:
                argv = shlex.split(command)
            except ValueError as exc:
                return {"ok": False, "error": f"smoke: shlex failed: {exc}"}
            if not argv:
                return {"ok": False, "error": "smoke: empty argv"}
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_timeout_for(command),
                check=False,
            )
        elif kind == "python_script":
            source = str(recipe.get("source") or "")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as fh:
                fh.write(source)
                script_path = fh.name
            try:
                result = subprocess.run(
                    ["python3", script_path],
                    capture_output=True,
                    text=True,
                    timeout=_timeout_for(source),
                    check=False,
                )
            finally:
                try:
                    Path(script_path).unlink()
                except Exception:
                    pass
        else:
            return {"ok": False, "error": f"smoke: unsupported recipe type {kind!r}"}
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": (
                f"smoke: recipe exceeded {SMOKE_TIMEOUT_SECONDS}s timeout. "
                "Pick a fast non-blocking variant or trim the work."
            ),
        }
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"smoke: binary not found: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"smoke: unexpected error: {exc}"}

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "").strip()[:300]
        return {
            "ok": False,
            "error": (
                f"smoke: recipe exited rc={result.returncode}. "
                f"stderr: {stderr_snippet or '(empty)'}. "
                "The command is syntactically OK but the flags/arguments are wrong; "
                "pick a different invocation or use python_script."
            ),
        }
    if not (result.stdout or "").strip():
        return {
            "ok": False,
            "error": (
                "smoke: recipe ran cleanly but produced no stdout. "
                "Choose a command/script that actually prints the requested fact."
            ),
        }
    return {"ok": True}


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
        # Sudo is allowed only for the apt-install path. Any other sudo
        # invocation is a privilege-escalation surface and is rejected.
        if argv[0] == "sudo":
            if not _is_safe_sudo_install(argv):
                return {
                    "ok": False,
                    "error": (
                        "sudo is only permitted for `apt-get install` / `apt install` / "
                        "`apt-get update` (and `apt update`). Use the unprivileged form of "
                        "your command, or restrict the sudo usage to an apt install."
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
