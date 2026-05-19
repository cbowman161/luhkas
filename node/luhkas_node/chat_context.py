"""Context helpers for node-local chat forwarding."""
from __future__ import annotations

from .routing_intent import classify_request_target

try:
    from .keyword_extractor import extract_keywords as _extract_keywords
except Exception:
    _extract_keywords = None


def build_presence_payload(message: str, entries: list[dict], node_id: str) -> dict:
    context = _chat_context(entries)
    clarification = _clarification(message, context)
    reply_context = _reply_context(message, context)
    routing_intent = classify_request_target(message, node_id)
    payload = {
        "message": message,
        "node_id": node_id,
        "chat_context": context,
        "routing_intent": routing_intent,
        "request_owner": routing_intent.get("request_owner"),
        "target_node": routing_intent.get("target_node"),
    }
    if _extract_keywords is not None:
        try:
            kw = _extract_keywords(message)
            if kw.get("people") or kw.get("nodes"):
                payload["keywords"] = kw
        except Exception:
            pass
    if clarification:
        payload["original_message"] = message
        payload["clarification"] = True
        payload["clarified_request"] = clarification["clarified_request"]
        payload["routing_feedback"] = clarification
    if reply_context:
        payload["reply_context"] = reply_context
        payload["conversation_continuity"] = True
    return payload


def _chat_context(entries: list[dict]) -> list[dict]:
    result = []
    for entry in entries:
        role = entry.get("role")
        text = entry.get("text")
        source = entry.get("source")
        if role not in {"user", "assistant", "error"} or not text:
            continue
        result.append({
            "role": role,
            "source": source,
            "text": str(text),
        })
    return result


def _clarification(message: str, context: list[dict]) -> dict | None:
    correction = _looks_like_correction(message)
    confirmation = _looks_like_confirmation(message)
    if not correction and not confirmation:
        return None
    previous_assistant = None
    previous_user = None
    for entry in reversed(context[:-1]):
        if previous_assistant is None and entry.get("role") == "assistant":
            previous_assistant = str(entry.get("text") or "")
            continue
        if previous_assistant is not None and entry.get("role") == "user":
            previous_user = str(entry.get("text") or "")
            break
    if not previous_user or not previous_assistant:
        return None
    if not _looks_like_interpretation(previous_assistant):
        return None
    if confirmation:
        return {
            "type": "route_confirmation",
            "previous_user_message": previous_user,
            "assistant_interpretation": previous_assistant,
            "user_confirmation": message,
            "clarified_request": previous_user,
        }
    return {
        "type": "route_correction",
        "previous_user_message": previous_user,
        "assistant_interpretation": previous_assistant,
        "user_correction": message,
        "clarified_request": (
            f"{previous_user}\n"
            f"Correction from user: {message}\n"
            "Use the correction to choose the request owner and route."
        ),
    }


def _looks_like_correction(message: str) -> bool:
    text = " ".join(str(message).casefold().split())
    return (
        text.startswith("no")
        or text.startswith("not ")
        or text.startswith("actually ")
        or text.startswith("i mean ")
        or text.startswith("that's not ")
        or text.startswith("that is not ")
    )


def _reply_context(message: str, context: list[dict]) -> dict | None:
    if not _looks_like_contextual_reply(message):
        return None
    previous_assistant = None
    previous_user = None
    for entry in reversed(context[:-1]):
        if previous_assistant is None and entry.get("role") == "assistant":
            previous_assistant = str(entry.get("text") or "")
            continue
        if previous_assistant is not None and entry.get("role") == "user":
            previous_user = str(entry.get("text") or "")
            break
    if not previous_assistant:
        return None
    return {
        "type": "reply_to_previous_assistant",
        "current_user_message": message,
        "previous_user_message": previous_user,
        "previous_assistant_message": previous_assistant,
    }


def _looks_like_confirmation(message: str) -> bool:
    text = " ".join(str(message).casefold().split())
    return text in {
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "right",
        "sure",
        "ok",
        "okay",
        "sounds right",
        "that's right",
        "that is right",
        "exactly",
    }


def _looks_like_contextual_reply(message: str) -> bool:
    text = " ".join(str(message).casefold().split())
    if not text:
        return False
    if _looks_like_confirmation(text) or _looks_like_correction(text):
        return True
    if len(text.split()) <= 5 and (
        text.startswith("why")
        or text.startswith("how")
        or text.startswith("what do you mean")
        or text in {"what?", "why?", "how?", "which one", "that one", "do it"}
    ):
        return True
    references_previous = {
        "that",
        "this",
        "it",
        "they",
        "them",
        "those",
        "one",
        "not",
    }
    words = set(text.replace("?", "").replace(".", "").split())
    return len(words) <= 8 and bool(words & references_previous)


def _looks_like_interpretation(message: str) -> bool:
    text = str(message).casefold()
    return (
        "i think you mean" in text
        or "is that right" in text
        or "did you mean" in text
        or "do you mean" in text
    )
