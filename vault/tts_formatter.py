"""
TTS formatter — strips markdown, emojis, and visual-only content from responses
so they read naturally when spoken aloud.
"""
from __future__ import annotations

import re


# Emoji ranges to strip
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def format_for_tts(text: str) -> str:
    """Return a TTS-safe version of text: no markdown, no emojis, natural phrasing."""
    if not text:
        return text

    # Fenced code blocks — replace with a spoken placeholder
    text = re.sub(r"```[a-z]*\n?[\s\S]*?```", "(code block)", text)

    # Inline code — strip backticks, keep the text
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Bold / italic
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)

    # Links — keep the label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Bullet / numbered list markers — strip the symbol, keep the content
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Section dividers like "---"
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

    # Emojis
    text = _EMOJI_RE.sub("", text)

    # Collapse extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim
    return text.strip()


def is_tts_clean(text: str) -> bool:
    """True if the text needs no TTS transformation."""
    return text == format_for_tts(text)
