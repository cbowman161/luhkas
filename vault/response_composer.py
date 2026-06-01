from __future__ import annotations

import json

from streaming import get_stream_sink


FALLBACK_PREFIX = "Fallback response:"


class ResponseComposer:
    """Generate final user-facing wording from structured facts.

    The caller owns routing, actions, and fact collection. This class owns the
    last inch: fast phrasing, anti-repeat behavior, and explicit fallbacks.
    """

    def __init__(self, model):
        self.model = model

    def compose(
        self,
        *,
        response_type: str,
        user_message: str,
        facts: dict,
        fallback: str,
        contract: str = "",
        recent_responses: list[str] | None = None,
        options: dict | None = None,
        timeout: float = 8.0,
        validator=None,
        sanitizer=None,
        required_terms: tuple[str, ...] = (),
    ) -> str:
        recent = [str(item).strip() for item in (recent_responses or []) if str(item).strip()]
        prompt = self._prompt(response_type, user_message, facts, recent)
        if contract:
            prompt = f"{prompt}\n\nNon-negotiable response contract:\n{contract}\n"
        try:
            # Streaming path: when a sink is installed for the current
            # request and the model supports streaming, claim the sink
            # and stream raw tokens directly to the consumer. Skip
            # sanitizer / validator / required_terms / recent-dedup
            # entirely — once a token is sent to the audio_node it has
            # already been queued for TTS, so post-hoc validation can't
            # take it back. The node speaks whatever the LLM produces.
            # claim() is single-shot per sink so intermediate compose
            # calls in the same request fall through to the sync path
            # and don't also broadcast their tokens.
            sink = get_stream_sink()
            stream_fn = (
                getattr(self.model, "generate_stream", None)
                if sink is not None and sink.claim()
                else None
            )
            if stream_fn is not None:
                parts: list[str] = []
                try:
                    for chunk in stream_fn(
                        prompt,
                        options=self._options(options),
                        timeout=timeout,
                        think=False,
                    ):
                        parts.append(chunk)
                        sink.emit("delta", chunk)
                except Exception as stream_exc:
                    # The audio_node has already heard whatever streamed
                    # up to the error — can't take it back. Log with
                    # context (the outer try would catch this too but
                    # with less specific framing) and let what we have
                    # be returned. Empty parts -> outer fallback path.
                    print(
                        f"[response_composer] stream interrupted after "
                        f"{len(parts)} tokens: {stream_exc}",
                        flush=True,
                    )
                return "".join(parts).strip() or self.fallback(fallback, "empty model response")
            # Sync path: full validation as before.
            text = self.model.generate(
                prompt,
                options=self._options(options),
                timeout=timeout,
                think=False,
            ).strip()
            text = sanitizer(text) if sanitizer is not None else text
            if not text:
                return self.fallback(fallback, "empty model response")
            lowered = text.lower()
            if any(term.lower() not in lowered for term in required_terms):
                return self.clean_fallback(fallback, recent)
            if text in recent:
                varied = self.varied_fallback(fallback, recent)
                return varied if varied != fallback else self.clean_fallback(fallback, recent)
            if validator is not None:
                violation = validator(text)
                if violation:
                    return self.clean_fallback(fallback, recent)
            return text
        except Exception as exc:
            return self.fallback(fallback, str(exc))

    def fallback(self, fallback: str, reason: str = "") -> str:
        text = str(fallback or "I could not generate that cleanly.").strip()
        if text.startswith(FALLBACK_PREFIX):
            return text
        suffix = f" ({reason})" if reason else ""
        return f"{FALLBACK_PREFIX} {text}{suffix}"

    def varied_fallback(self, fallback: str, recent: list[str]) -> str:
        base = str(fallback or "").strip()
        variants = [base]
        if base.startswith("Got it. The "):
            variants.append(base.replace("Got it. The ", "Logged for this chat: the ", 1))
            variants.append(base.replace("Got it. The ", "I have it: the ", 1))
        elif base.startswith("The ") and " was " in base:
            variants.append(base.replace("The ", "You gave me the ", 1).replace(" was ", ": ", 1))
            variants.append(base.replace("The ", "I have the ", 1).replace(" was ", " as ", 1))
        elif base.startswith("The live node registry"):
            variants.append(base.replace("The live node registry currently shows", "Right now the live registry has", 1))
            variants.append(base.replace("The live node registry currently shows", "I see", 1))
        elif base.startswith("I'm using Scout"):
            variants.append(base.replace("I'm using Scout", "Scout is the body I'm using", 1))
            variants.append(base.replace("I'm using Scout", "From Scout's body, I'm", 1))
        for variant in variants:
            if variant and variant not in recent:
                return variant
        return base

    def clean_fallback(self, fallback: str, recent: list[str] | None = None) -> str:
        """Use a safe deterministic wording without exposing validation internals."""
        base = str(fallback or "I could not generate that cleanly.").strip()
        if base.startswith(FALLBACK_PREFIX):
            base = base[len(FALLBACK_PREFIX):].strip()
        recent = recent or []
        varied = self.varied_fallback(base, recent)
        return varied or base

    def _prompt(
        self,
        response_type: str,
        user_message: str,
        facts: dict,
        recent_responses: list[str],
    ) -> str:
        return f"""Write the final user-facing answer as a direct conversational reply from Luhkas.
Type: {response_type}
User: {user_message}

Facts:
{json.dumps(facts, separators=(",", ":"), default=str)}

Recent answers to avoid repeating exactly:
{json.dumps(recent_responses[-5:], separators=(",", ":"), default=str)}

Rules:
- Preserve the facts exactly. Do not invent state, actions, memories, people, nodes, or capabilities.
- Keep deterministic commands accurate; only the wording should vary.
- One short sentence unless the facts require a compact list.
- First person when talking about yourself.
- Do not format the reply like a transcript or speaker label. Use the name Luhkas naturally only when the user asks who you are or what your name is.
- No emojis, no customer-service closer, no generic offer to help.
- If deterministic_answer is present, keep the same meaning without copying it verbatim when possible.
"""

    def _options(self, options: dict | None) -> dict:
        merged = {
            "num_predict": 80,
            "temperature": 0.72,
            "top_p": 0.9,
            "repeat_penalty": 1.18,
            "num_ctx": 2048,
        }
        if options:
            merged.update(options)
        return merged
