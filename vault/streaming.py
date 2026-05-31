"""Request-scoped streaming sink for vault responses.

Several layers below the HTTP handler (notably ``ResponseComposer.compose``)
can produce text incrementally if their model is asked to stream. Threading
a sink object explicitly through every caller is invasive, so we expose a
``contextvars.ContextVar`` that the producer reads opportunistically.

Per-thread isolation is preserved automatically: each request handler thread
gets its own context, so two simultaneous requests can each install their
own sink without bleeding into each other.

The ``StreamSink`` exposes:

* ``__call__(kind, text)`` and ``emit(kind, text)`` — same as the older
  callable contract; producers send ``("delta", token_chunk)``,
  ``("working", "progress hint")``, etc.
* ``claim()`` — returns True exactly once per sink. The producer should
  only start streaming if ``claim()`` succeeds. This prevents the case
  where one request triggers multiple compose calls (intermediate fact
  extraction, retries, etc.) and each call would otherwise broadcast its
  raw tokens to the audio output, even though only one of them is the
  user-facing reply. The FIRST compose to claim wins.

Usage from the HTTP handler::

    def _emit(kind, text):
        emit_event_to_client({"type": kind, "text": text})

    sink = StreamSink(_emit)
    token = set_stream_sink(sink)
    try:
        runtime.handle_presence(message, ...)
    finally:
        reset_stream_sink(token)

Usage from the producer (response_composer.compose)::

    sink = get_stream_sink()
    if sink is not None and sink.claim():
        sink.emit("working", "composing")
        for chunk in self.model.generate_stream(prompt, ...):
            sink.emit("delta", chunk)
    else:
        text = self.model.generate(prompt, ...)
"""
from __future__ import annotations

import contextvars
from typing import Callable, Optional


EmitCallable = Callable[[str, str], None]


class StreamSink:
    """Per-request emit callable with single-claim semantics."""

    def __init__(self, emit: EmitCallable) -> None:
        self._emit = emit
        self._claimed = False

    def __call__(self, kind: str, text: str) -> None:
        try:
            self._emit(kind, text)
        except Exception:
            # Producers should never fail because the consumer disconnected.
            pass

    def emit(self, kind: str, text: str) -> None:
        self.__call__(kind, text)

    def claim(self) -> bool:
        """Return True once per sink, False on subsequent calls.

        Used by the producer to decide whether to take responsibility for
        the request's streamed output. Without this, every compose call
        in a request would broadcast its raw tokens — including
        intermediate compositions (fact extractors, retries) that the
        user shouldn't hear.
        """
        if self._claimed:
            return False
        self._claimed = True
        return True

    @property
    def claimed(self) -> bool:
        return self._claimed


_stream_sink_var: contextvars.ContextVar[Optional[StreamSink]] = contextvars.ContextVar(
    "vault_stream_sink", default=None
)


def set_stream_sink(sink: StreamSink):
    """Install ``sink`` for the current context. Returns a reset token."""
    return _stream_sink_var.set(sink)


def reset_stream_sink(token) -> None:
    _stream_sink_var.reset(token)


def get_stream_sink() -> Optional[StreamSink]:
    return _stream_sink_var.get()
