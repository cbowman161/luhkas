"""Shared JSON-extraction helpers for the agent prompts.

Both requirements_agent and review_agent ask their LLM for JSON-shaped
output and have to peel that JSON back out of whatever wrapping the
model produced (raw, code-fenced, or with leading/trailing prose).
Previously each agent kept its own copy of the same parser — moved here
so the next agent doesn't add a third copy.
"""
from __future__ import annotations


def extract_json(text: str) -> str:
    """Pull the first JSON object out of an LLM response.

    Handles three common shapes:
      * raw ``{...}``
      * triple-fenced ``​```json\n{...}\n```​``
      * prose-wrapped ``Sure, here it is: {...}``

    Returns the candidate JSON substring (still a string — caller is
    responsible for ``json.loads``). On no-match returns the input
    unchanged so the caller's parser surfaces its own error message.
    """
    text = text.strip()
    if "```" in text:
        for block in text.split("```")[1::2]:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                return block
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text
