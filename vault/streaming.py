"""Request-scoped streaming sink for vault responses.

Several layers below the HTTP handler (notably ``ResponseComposer.compose``)
can produce text incrementally if their model is asked to stream. Threading
a sink object explicitly through every caller is invasive, so we expose a
``contextvars.ContextVar`` that the producer reads opportunistically.

Per-thread isolation is preserved automatically: each request handler thread
gets its own context, so two simultaneous requests can each install their
own sink without bleeding into each other.

Usage::

    # in the HTTP handler for a streaming endpoint
    def sink(kind, text):
        emit_event_to_client({"type": kind, "text": text})

    token = set_stream_sink(sink)
    try:
        runtime.handle_presence(message, ...)
    finally:
        reset_stream_sink(token)

The producer side (see ``response_composer.compose``) checks for a sink and,
if present, calls it with ``("delta", chunk)`` for each token group.
"""
from __future__ import annotations

import contextvars
from typing import Callable, Optional


StreamSink = Callable[[str, str], None]

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
