"""
Action Envelope validator/parser for Brain_V2.

LLM-facing format:

ACTION: write_file | patch_file | read_file | list_files | command | final | fail
PATH: optional
COMMAND: optional
REASON: optional short text
CONTENT_TYPE: optional

---BEGIN CONTENT---
raw content, never JSON
---END CONTENT---

Internal normalized format returned by validate_action():
{
    "type": str,
    "path": str | None,
    "content": str,
    "reason": str,
    "content_type": str | None,
}
"""

from __future__ import annotations

import re
from typing import Dict, Optional


VALID_TYPES = {
    "write_file",
    "patch_file",
    "read_file",
    "list_files",
    "command",
    "final",
    "fail",
}

REQUIRES_PATH = {"write_file", "patch_file", "read_file"}
REQUIRES_CONTENT = {"write_file", "patch_file", "final", "fail"}
REQUIRES_COMMAND = {"command"}

BEGIN = "---BEGIN CONTENT---"
END = "---END CONTENT---"


class ActionValidationError(ValueError):
    """Raised when a coder action envelope is invalid."""


def _strip_outer_noise(text: str) -> str:
    if text is None:
        raise ActionValidationError("Empty output")

    text = str(text).strip()
    if not text:
        raise ActionValidationError("Empty output")

    # Do not parse markdown or prose. Only allow leading whitespace before ACTION:.
    match = re.search(r"(?m)^ACTION:\s*\w+", text)
    if not match:
        raise ActionValidationError(f"No ACTION envelope found:\n{text[:1000]}")

    if text[: match.start()].strip():
        raise ActionValidationError("Unexpected text before ACTION header")

    return text[match.start() :].strip()


def _split_headers_and_content(text: str) -> tuple[str, str]:
    has_begin = BEGIN in text
    has_end = END in text

    if has_begin != has_end:
        raise ActionValidationError("Content block must include both BEGIN and END markers")

    if not has_begin:
        return text.strip(), ""

    before, rest = text.split(BEGIN, 1)
    content, after = rest.split(END, 1)

    if after.strip():
        raise ActionValidationError("Unexpected text after END CONTENT marker")

    return before.strip(), content


def _parse_headers(header_text: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    seen_action_count = 0

    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" not in line:
            raise ActionValidationError(f"Invalid header line: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip().upper().replace("-", "_")
        value = value.strip()

        if not key:
            raise ActionValidationError(f"Invalid header line: {raw_line}")

        if key == "ACTION":
            seen_action_count += 1
            if seen_action_count > 1:
                raise ActionValidationError("Multiple ACTION headers detected")

        if key in headers and key != "REASON":
            raise ActionValidationError(f"Duplicate header: {key}")

        if key == "REASON" and key in headers:
            headers[key] = headers[key] + "\n" + value
        else:
            headers[key] = value

    return headers


def parse_action_envelope(text: str) -> dict:
    text = _strip_outer_noise(text)

    # A second ACTION line outside content means multiple actions.
    action_lines = re.findall(r"(?m)^ACTION:\s*\w+", text)
    if len(action_lines) > 1:
        raise ActionValidationError("Multiple ACTION envelopes detected")

    header_text, body = _split_headers_and_content(text)
    headers = _parse_headers(header_text)

    action_type = (headers.get("ACTION") or "").strip().lower()
    path = (headers.get("PATH") or "").strip() or None
    command = (headers.get("COMMAND") or "").strip()
    reason = (headers.get("REASON") or "").strip()
    content_type = (headers.get("CONTENT_TYPE") or headers.get("CONTENT-TYPE") or "").strip() or None

    content = body
    if action_type == "command":
        content = command

    return {
        "type": action_type,
        "path": path,
        "content": content.rstrip("\n"),
        "reason": reason,
        "content_type": content_type,
    }


def validate_action(raw_output: str | dict) -> dict:
    """
    Validate a coder action. Accepts either an Action dict or an envelope string.
    The dict path is useful for future non-LLM callers; LLM output should be envelope text.
    """
    if isinstance(raw_output, dict):
        action = {
            "type": (raw_output.get("type") or raw_output.get("action") or "").strip().lower(),
            "path": raw_output.get("path"),
            "content": raw_output.get("content") or raw_output.get("command") or "",
            "reason": raw_output.get("reason") or "",
            "content_type": raw_output.get("content_type"),
        }
    else:
        action = parse_action_envelope(raw_output)

    action_type = action.get("type")

    if not action_type:
        raise ActionValidationError("Missing ACTION type")

    if action_type not in VALID_TYPES:
        raise ActionValidationError(f"Invalid action type: {action_type}")

    if action_type in REQUIRES_PATH and not action.get("path"):
        raise ActionValidationError(f"{action_type} requires PATH")

    if action_type in REQUIRES_CONTENT and not str(action.get("content") or "").strip():
        raise ActionValidationError(f"{action_type} requires non-empty content block")

    if action_type in REQUIRES_COMMAND and not str(action.get("content") or "").strip():
        raise ActionValidationError("command requires COMMAND")

    if action_type in {"read_file", "list_files"}:
        action["content"] = action.get("content") or ""

    if action_type == "list_files" and not action.get("path"):
        action["path"] = "."

    return action
