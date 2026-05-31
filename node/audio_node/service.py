#!/usr/bin/env python3
"""audio_node HTTP service.

Owns the mic-to-speaker loop on a node:

  mic (arecord) → VAD/streaming STT → POST /presence/message → TTS → aplay

Endpoints:
  GET  /health      — engine + capture status
  POST /tts         — body {"text": "..."}; synthesize and play locally
  POST /listen      — body {"muted": bool}; pause/resume mic capture
  GET  /transcripts — last N recognized utterances (debug)

Configuration is fully env-driven so the same systemd unit works on any
node that has the RaspAudio HAT or a USB mic+speaker.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_node.capture import MicCapture
from audio_node.engines import load_stt, load_tts
from luhkas_node.wakeword import is_wakeword_only, response as wakeword_response
from presence_state import update_state


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] audio_node: %(message)s")
log = logging.getLogger("audio_node")


_transcripts: deque = deque(maxlen=20)
_transcripts_lock = threading.Lock()

# ---- Noise / short-utterance filter -----------------------------------------
# Vosk emits false-positive single-word transcripts ("the", "a", "is")
# from faint background noise and from gaps around our own TTS playback.
# Forwarding those to vault triggers a hallucinated LLM response that goes
# back through TTS, which the mic picks up — feedback loop.
# Tunable via env:
#   AUDIO_MIN_TRANSCRIPT_CHARS    min trimmed length  (default 6)
#   AUDIO_MIN_TRANSCRIPT_WORDS    min word count      (default 2)
#   AUDIO_STOPWORD_REJECT         csv stopwords       (default the,a,an,...)
_MIN_TRANSCRIPT_CHARS = int(os.environ.get("AUDIO_MIN_TRANSCRIPT_CHARS", "4"))
_MIN_TRANSCRIPT_WORDS = int(os.environ.get("AUDIO_MIN_TRANSCRIPT_WORDS", "1"))
_STOPWORD_REJECT = {
    w.strip().lower()
    for w in os.environ.get(
        "AUDIO_STOPWORD_REJECT",
        "the,a,an,of,is,and,to,in,it,on,or,um,uh,hmm",
    ).split(",")
    if w.strip()
}


_WAKEWORD_VARIANTS = {w.strip().lower() for w in os.environ.get(
    "AUDIO_WAKEWORD_VARIANTS",
    # Vosk's English model mistranscribes 'luhkas' to several near-words.
    "luhkas,luhkus,lewis,lucas,lookus,luca,lukas,loukas,looks,louise",
).split(",") if w.strip()}


def _is_noise_transcript(text: str) -> tuple[bool, str]:
    """Return (rejected, reason) for transcripts we want to silently drop."""
    stripped = (text or "").strip()
    # Wake-word bypass: any transcript containing a wake-word variant always
    # passes the filter so the user can summon LUHKAS even with mistranscription.
    low = stripped.lower()
    if any(w in low for w in _WAKEWORD_VARIANTS):
        return False, ""
    if not stripped:
        return True, "empty"
    if len(stripped) < _MIN_TRANSCRIPT_CHARS:
        return True, f"too short ({len(stripped)} < {_MIN_TRANSCRIPT_CHARS} chars)"
    words = [w for w in stripped.lower().split() if w]
    if len(words) < _MIN_TRANSCRIPT_WORDS:
        return True, f"too few words ({len(words)} < {_MIN_TRANSCRIPT_WORDS})"
    if len(words) == 1 and words[0] in _STOPWORD_REJECT:
        return True, f"single stopword '{words[0]}'"
    return False, ""


def _contains_audio_wakeword(text: str) -> bool:
    words = set(_words(text))
    return bool(words & _WAKEWORD_VARIANTS) or _is_audio_wakeword_only(text)
_tts_lock = threading.Lock()
_tts_speaking = threading.Event()
# Generation counter for queue-drain. Bumping it invalidates any _speak
# thread that hasn't started speaking yet (queued on _tts_lock) AND short-
# circuits the inter-chunk loop inside a running _speak. Used by the
# streaming "redo" path so chunks queued behind the now-invalid speech
# don't play after the corrective fallback.
_tts_gen_lock = threading.Lock()
_tts_generation = 0
_tts_text_lock = threading.Lock()
_tts_current_text = ""
_tts_last_started_at = 0.0
_tts_last_ended_at = 0.0
_tts_recent_texts: deque = deque(maxlen=6)
_tts_threads: deque = deque(maxlen=8)

_WORD_RE = re.compile(r"[a-z0-9']+")
_WAKE_PHRASE_VARIANTS = {
    phrase.strip().casefold()
    for phrase in os.environ.get(
        "AUDIO_WAKE_PHRASE_VARIANTS",
        "luhkas,luhkus,lucas,lukas,loukas,hey luhkas,hey lucas,okay luhkas,they say",
    ).split(",")
    if phrase.strip()
}

# Vosk's "final" results are often clause fragments rather than the whole
# thing the person said. Buffer adjacent fragments briefly so presence receives
# phrases instead of stray one- and two-word scraps.
_PHRASE_GAP_SECONDS = float(os.environ.get("AUDIO_PHRASE_GAP_SECONDS", "1.1"))
_PHRASE_MAX_SECONDS = float(os.environ.get("AUDIO_PHRASE_MAX_SECONDS", "8.0"))
_PHRASE_MIN_WORDS = int(os.environ.get("AUDIO_PHRASE_MIN_WORDS", "2"))
_PHRASE_MIN_CHARS = int(os.environ.get("AUDIO_PHRASE_MIN_CHARS", "8"))

# TTS playback chunking. Split assistant replies on sentence boundaries so
# the first sentence starts playing while later sentences are still
# synthesizing. If the leading sentence is itself long, take a one-time
# clause break (just first clause + rest) so audio starts sooner without
# shredding lists like "apples, oranges, bananas, and pears".
#   AUDIO_TTS_CHUNK_ENABLE       toggle (default 1)
#   AUDIO_TTS_CHUNK_MIN_WORDS    target min words per chunk (default 6)
_TTS_CHUNK_ENABLE = os.environ.get("AUDIO_TTS_CHUNK_ENABLE", "1").lower() not in ("0", "false", "no", "")
_TTS_CHUNK_MIN_WORDS = int(os.environ.get("AUDIO_TTS_CHUNK_MIN_WORDS", "6"))

# Words that end with a period but never end a sentence. Lowercased, trailing
# period stripped. Keeps "Mr. Smith", "8 a.m. Tomorrow", "e.g. foo", "etc. so"
# from being mis-split. Note: decimals ("3.14") and IPs ("192.168.1.5") are
# already safe because the period has no whitespace after it.
_TTS_SENTENCE_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "fr", "jr", "sr", "st", "vs", "no",
    "etc", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k",
    "fig", "vol", "ch", "approx", "min", "max", "sec", "msec",
    "inc", "ltd", "corp", "co",
}
# A sentence-end candidate: a non-space token ending in .!?, then optional
# closing quote/bracket, then whitespace, then a capital letter or digit
# (sentence-start signal). We capture the token to look up abbreviations.
_TTS_SENT_END_RE = re.compile(
    r"(\S+?)([.!?]+)(['\")\]]*)\s+(?=[\"'(\[]*[A-Z0-9])"
)
_TTS_CLAUSE_BREAK_RE = re.compile(r"[,;:]\s+(?=\S)")


def _split_sentences(text: str) -> list[str]:
    """Sentence-segment text, leaving abbreviations and embedded decimals intact."""
    sentences: list[str] = []
    last = 0
    for m in _TTS_SENT_END_RE.finditer(text):
        token = m.group(1).lower().rstrip(".")
        if token in _TTS_SENTENCE_ABBREVIATIONS:
            continue
        end = m.end()  # past the trailing whitespace
        chunk = text[last:end].strip()
        if chunk:
            sentences.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_speech_chunks(text: str, min_words: int = _TTS_CHUNK_MIN_WORDS) -> list[str]:
    """Split text into incrementally-speakable chunks.

    Sentences first (.!? respecting abbreviations). If the leading sentence is
    notably long, take a single clause break so audio starts sooner — but only
    one break, so lists stay intact. Tiny adjacent fragments are coalesced so
    we never emit a one-word chunk.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]
    head = sentences[0]
    if len(head.split()) > max(min_words * 3, min_words + 6):
        for m in _TTS_CLAUSE_BREAK_RE.finditer(head):
            first = head[: m.start() + 1].strip()  # keep the comma/colon
            if len(first.split()) < max(2, min_words):
                continue
            rest = head[m.end():].strip()
            if rest:
                sentences = [first, rest] + sentences[1:]
            break
    merged: list[str] = []
    pending = ""
    for chunk in sentences:
        candidate = (pending + " " + chunk).strip() if pending else chunk
        if len(candidate.split()) < 2:
            pending = candidate
            continue
        if merged and len(merged[-1].split()) < 2:
            merged[-1] = (merged[-1] + " " + candidate).strip()
        else:
            merged.append(candidate)
        pending = ""
    if pending:
        if merged:
            merged[-1] = (merged[-1] + " " + pending).strip()
        else:
            merged.append(pending)
    return merged


# ---- Vault streaming consumer ----------------------------------------------
# When the vault exposes /presence/message/stream, we POST there instead of
# the legacy synchronous endpoint and pull NDJSON events. As each complete
# sentence accumulates in the buffer we hand it to TTS, so the user hears
# the first sentence while the model is still generating later ones.
#
# Env:
#   AUDIO_PRESENCE_STREAM        toggle (default 1)
#   AUDIO_PRESENCE_STREAM_URL    explicit override; otherwise derived from
#                                AUDIO_PRESENCE_URL by appending "/stream"
_PRESENCE_STREAM_ENABLE = os.environ.get("AUDIO_PRESENCE_STREAM", "1").lower() not in ("0", "false", "no", "")


def _derive_stream_url(presence_url: str, override: str = "") -> str:
    if override:
        return override
    base = presence_url.rstrip("/")
    if base.endswith("/presence/message"):
        return base + "/stream"
    return ""


# Streaming chunk dispatch: word-count threshold instead of "wait for the
# full sentence." Once the buffer has at least this many words, dispatch
# whatever is bounded by the latest natural break (sentence > clause >
# word). This minimises time-to-first-audio at the cost of occasionally
# cutting at a clause/word boundary mid-sentence, which Piper handles
# without audible artifacts.
_TTS_STREAM_MIN_WORDS = int(os.environ.get("AUDIO_TTS_STREAM_MIN_WORDS", "8"))


def _find_natural_break(buffer: str) -> int:
    """Index (just past) the latest natural break in ``buffer``.

    Preference order: sentence end (``.!?`` + whitespace) > clause end
    (``,;:`` + whitespace) > word boundary (trailing whitespace position).
    Returns 0 if no clean break exists (the buffer is one long unbroken
    token). Abbreviations like "Mr." are honored — those don't count as
    sentence ends.
    """
    n = len(buffer)
    # Sentence boundary, abbreviation-aware
    for i in range(n - 1, -1, -1):
        if buffer[i] in ".!?" and i + 1 < n and buffer[i + 1].isspace():
            word_start = buffer.rfind(" ", 0, i) + 1
            word = buffer[word_start:i].lower().rstrip(".")
            if word not in _TTS_SENTENCE_ABBREVIATIONS:
                return i + 1
    # Clause boundary
    for i in range(n - 1, -1, -1):
        if buffer[i] in ",;:" and i + 1 < n and buffer[i + 1].isspace():
            return i + 1
    # Word boundary (split on the last full whitespace position)
    last_space = buffer.rfind(" ")
    return last_space + 1 if last_space >= 0 else 0


def _drain_streamed_chunks(
    buffer: str,
    min_words: int = _TTS_STREAM_MIN_WORDS,
) -> tuple[list[str], str]:
    """Pull speakable chunks from a streaming buffer.

    Strategy: only dispatch when the buffer has at least ``min_words``
    complete words AND a natural break is available. This gets the first
    chunk out fast (no waiting for a sentence to complete) while keeping
    chunk boundaries at points where Piper synthesizes naturally.

    Returns (chunks, remaining_buffer).
    """
    chunks: list[str] = []
    while True:
        if len(buffer.split()) < min_words:
            break
        break_at = _find_natural_break(buffer)
        if break_at <= 0:
            break  # no break point yet — wait for whitespace/punct
        chunk = buffer[:break_at].strip()
        if not chunk:
            break
        buffer = buffer[break_at:].lstrip()
        chunks.append(chunk)
    return chunks, buffer


def _stream_presence_to_tts(
    stream_url: str,
    payload: dict,
    tts,
    event_url: str,
) -> dict | None:
    """POST a user message to the vault streaming endpoint and pipeline TTS.

    Returns:
      * ``{"tts": str, "message": str}`` on success (TTS dispatch already done)
      * ``None`` to signal "fall back to the legacy synchronous endpoint":
        404, transport error, or the server doesn't actually stream.
    """
    req = Request(
        stream_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    try:
        upstream = urlopen(req, timeout=180)
    except HTTPError as exc:
        if exc.code == 404:
            log.info("stream endpoint not available (404); falling back")
            return None
        log.warning("stream POST failed (%s): %s", exc.code, exc)
        return None
    except (URLError, OSError, TimeoutError) as exc:
        log.warning("stream POST failed: %s", exc)
        return None

    speech_buffer = ""
    final_text = ""
    saw_terminal = False
    spoken_anything = False
    redo_text = None
    try:
        for raw_line in upstream:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "delta":
                speech_buffer += str(event.get("text") or "")
                chunks, speech_buffer = _drain_streamed_chunks(speech_buffer)
                for chunk in chunks:
                    _start_tts(tts, chunk)
                    if not spoken_anything:
                        # First chunk: mirror to UI as the assistant message
                        # so the display starts showing text alongside the
                        # audio rather than waiting for "done".
                        _notify_ui_event(
                            event_url,
                            {"type": "assistant_message", "text": chunk},
                        )
                    spoken_anything = True
            elif etype == "done":
                final_text = str(event.get("text") or "")
                tail = speech_buffer.strip()
                if tail:
                    _start_tts(tts, tail)
                    spoken_anything = True
                    speech_buffer = ""
                saw_terminal = True
                break
            elif etype == "truncate":
                # Vault sanitizer trimmed a trailing portion. The streamed
                # prefix IS the validated content. Two cases:
                #   1. We already dispatched some chunks (spoken_anything):
                #      discard the buffer — what's spoken is correct.
                #   2. Nothing dispatched yet (e.g., short reply with no
                #      natural break before the closer): speak final_text
                #      so the user isn't left in silence.
                final_text = str(event.get("text") or "")
                speech_buffer = ""
                if not spoken_anything and final_text:
                    _start_tts(tts, final_text)
                    _notify_ui_event(
                        event_url,
                        {"type": "assistant_message", "text": final_text},
                    )
                    spoken_anything = True
                saw_terminal = True
                break
            elif etype == "redo":
                # Validator rejected the streamed text and replaced it with
                # final. Cancel everything queued/playing, then speak the
                # replacement. _cancel_queued_tts bumps the generation so
                # threads waiting on _tts_lock bail out instead of playing
                # stale chunks after our fallback.
                redo_text = str(event.get("text") or "")
                _cancel_queued_tts(tts)
                speech_buffer = ""
                if redo_text:
                    _start_tts(tts, redo_text)
                    _notify_ui_event(
                        event_url,
                        {"type": "assistant_message", "text": redo_text},
                    )
                    spoken_anything = True
                final_text = redo_text
                saw_terminal = True
                break
            elif etype == "working":
                # Vault-side progress hint ("composing", "checking vision",
                # etc.). Don't speak; just keep the UI alive. Useful when
                # the pre-LLM phase is slow.
                hint = str(event.get("text") or "")
                if hint:
                    log.debug("vault working: %s", hint)
            elif etype == "error":
                log.warning("vault stream error: %s", event.get("error"))
                break
            # "start" and unknown types ignored
    finally:
        try:
            upstream.close()
        except Exception:
            pass

    if not saw_terminal:
        tail = speech_buffer.strip()
        if tail:
            _start_tts(tts, tail)
            final_text = final_text or tail
        if not final_text:
            return None  # bail to legacy endpoint
    return {"tts": final_text, "message": final_text}


# ---- openWakeWord runtime ---------------------------------------------------
_WAKEWORD_ENABLED = os.environ.get('AUDIO_WAKEWORD_ENABLED', '1').lower() not in ('0', 'false', 'no', '')
_WAKEWORD_REQUIRE = os.environ.get("AUDIO_REQUIRE_WAKEWORD", "1").lower() not in ("0", "false", "no", "")
_WAKEWORD_THRESHOLD = float(os.environ.get('AUDIO_WAKEWORD_THRESHOLD', '0.3'))
_WAKEWORD_LISTEN_SECONDS = float(os.environ.get('AUDIO_WAKEWORD_LISTEN_SECONDS', '8.0'))
_WAKEWORD_MODEL_PATH = os.environ.get('AUDIO_WAKEWORD_MODEL', '')
_wakeword_model = None
_wakeword_buffer = bytearray()
_WAKEWORD_CHUNK_BYTES = 1280 * 2
_listening_until = 0.0
_wakeword_last_score = 0.0
_wakeword_recent_scores = deque(maxlen=400)  # ~32s @ 80ms chunks
_wakeword_lock = threading.Lock()


def _init_wakeword() -> None:
    global _wakeword_model
    if not _WAKEWORD_ENABLED:
        log.info('wakeword: disabled')
        return
    try:
        import openwakeword
        from openwakeword.model import Model
        if _WAKEWORD_MODEL_PATH:
            paths = [_WAKEWORD_MODEL_PATH]
        else:
            log.info('wakeword: no custom AUDIO_WAKEWORD_MODEL configured; requiring Vosk wake phrase=%s', _WAKEWORD_REQUIRE)
            return
        _wakeword_model = Model(wakeword_model_paths=paths)
        log.info('wakeword: loaded openWakeWord with %s', list(_wakeword_model.models.keys()))
    except Exception as exc:
        log.warning('wakeword: openWakeWord unavailable (%s)', exc)
        _wakeword_model = None


def _on_audio_chunk(pcm) -> None:
    global _listening_until, _wakeword_last_score
    if _wakeword_model is None:
        return
    import numpy as np
    _wakeword_buffer.extend(pcm)
    while len(_wakeword_buffer) >= _WAKEWORD_CHUNK_BYTES:
        chunk = bytes(_wakeword_buffer[:_WAKEWORD_CHUNK_BYTES])
        del _wakeword_buffer[:_WAKEWORD_CHUNK_BYTES]
        samples = np.frombuffer(chunk, dtype=np.int16)
        try:
            preds = _wakeword_model.predict(samples)
        except Exception as exc:
            log.warning('wakeword predict failed: %s', exc)
            return
        top = float(max(preds.values())) if preds else 0.0
        _wakeword_last_score = top
        _wakeword_recent_scores.append(top)
        if top > 0.1:
            log.info("wakeword score=%.3f (threshold %.2f)", top, _WAKEWORD_THRESHOLD)
        if top >= _WAKEWORD_THRESHOLD:
            now = time.time()
            with _wakeword_lock:
                was = _listening_until > now
                _listening_until = now + _WAKEWORD_LISTEN_SECONDS
            if not was:
                name = max(preds, key=preds.get)
                log.info('wakeword DETECTED (%s, score=%.3f); listening %.1fs', name, top, _WAKEWORD_LISTEN_SECONDS)


def _is_listening_now() -> bool:
    with _wakeword_lock:
        return time.time() < _listening_until


def _extend_listening() -> None:
    global _listening_until
    with _wakeword_lock:
        _listening_until = max(_listening_until, time.time() + _WAKEWORD_LISTEN_SECONDS)


def _normalized_phrase(text: str) -> str:
    return " ".join(_words(text))


def _is_audio_wakeword_only(text: str) -> bool:
    phrase = _normalized_phrase(text)
    return phrase in _WAKE_PHRASE_VARIANTS or is_wakeword_only(text)


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict | None:
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        log.warning("POST %s failed: %s", url, exc)
        return None


def _notify_ui_event(event_url: str, payload: dict) -> None:
    if not event_url:
        return
    try:
        url = event_url.rstrip("/")
        if not url.endswith("/ui/event"):
            url += "/ui/event"
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2.0):
            pass
    except Exception:
        # UI/display events are best-effort; never block the audio loop on them.
        pass


def _make_transcript_handler(presence_url: str, source: str, node_id: str, tts, event_url: str, stream_url: str = "") -> "callable":
    phrase_lock = threading.Lock()
    phrase_parts: list[str] = []
    phrase_started_at = 0.0
    phrase_timer: threading.Timer | None = None

    def _cancel_phrase_timer() -> None:
        nonlocal phrase_timer
        if phrase_timer is not None:
            phrase_timer.cancel()
            phrase_timer = None

    def _phrase_text() -> str:
        return " ".join(part.strip() for part in phrase_parts if part.strip()).strip()

    def _phrase_is_too_short(text: str) -> tuple[bool, str]:
        words = _words(text)
        if len(text.strip()) < _PHRASE_MIN_CHARS:
            return True, f"too short phrase ({len(text.strip())} < {_PHRASE_MIN_CHARS} chars)"
        if len(words) < _PHRASE_MIN_WORDS:
            return True, f"too few phrase words ({len(words)} < {_PHRASE_MIN_WORDS})"
        return False, ""

    def _process_phrase(text: str) -> None:
        if not text:
            return
        if _tts_speaking.is_set():
            if _is_likely_self_speech(text):
                log.info("ignored self-speech phrase during TTS: %s", text)
                return
            log.info("ignored phrase during TTS: %s", text)
            return
        elif _is_likely_self_speech(text):
            log.info("ignored delayed self-speech phrase after TTS: %s", text)
            return

        with _transcripts_lock:
            _transcripts.append({"text": text, "timestamp": time.time()})
        log.info("transcript phrase: %s", text)
        update_state({
            "audio": {
                "hearing": True,
                "last_transcript_at": time.time(),
                "last_transcript_text": text,
            },
            "latest_user": {"text": text, "source": source, "timestamp": time.time()},
        })
        _notify_ui_event(event_url, {"type": "user_message", "text": text, "source": source})
        update_state({"conversation": {"thinking": True, "thinking_started_at": time.time()}})
        request_payload = {"message": text, "source": source, "node_id": node_id}
        streamed_reply = None
        if _PRESENCE_STREAM_ENABLE and stream_url:
            streamed_reply = _stream_presence_to_tts(stream_url, request_payload, tts, event_url)
        if streamed_reply is not None:
            update_state({"conversation": {"thinking": False, "thinking_ended_at": time.time()}})
            spoken = streamed_reply.get("tts") or streamed_reply.get("message") or ""
            if spoken:
                update_state({"latest_assistant": {"text": spoken, "source": "presence", "timestamp": time.time()}})
                _notify_ui_event(event_url, {"type": "assistant_message", "text": spoken})
            return
        reply = _post_json(presence_url, request_payload)
        update_state({"conversation": {"thinking": False, "thinking_ended_at": time.time()}})
        if not reply:
            return
        response = reply.get("response") or reply
        spoken = response.get("tts") or response.get("message") or ""
        if spoken:
            update_state({"latest_assistant": {"text": spoken, "source": "presence", "timestamp": time.time()}})
            _notify_ui_event(event_url, {"type": "assistant_message", "text": spoken})
        if not spoken:
            return
        _start_tts(tts, spoken)

    def _flush_phrase(reason: str) -> None:
        nonlocal phrase_started_at
        with phrase_lock:
            text = _phrase_text()
            phrase_parts.clear()
            phrase_started_at = 0.0
            _cancel_phrase_timer()
        if not text:
            return
        rejected, reject_reason = _phrase_is_too_short(text)
        if rejected:
            log.info("dropped short phrase after %s (%s): %r", reason, reject_reason, text)
            return
        log.info("flushed transcript phrase after %s: %s", reason, text)
        _process_phrase(text)

    def _schedule_phrase_flush() -> None:
        nonlocal phrase_timer
        _cancel_phrase_timer()
        phrase_timer = threading.Timer(_PHRASE_GAP_SECONDS, _flush_phrase, args=("pause",))
        phrase_timer.daemon = True
        phrase_timer.start()

    def _queue_phrase_part(text: str) -> None:
        nonlocal phrase_started_at
        now = time.time()
        flush_now = False
        with phrase_lock:
            if not phrase_parts:
                phrase_started_at = now
            phrase_parts.append(text)
            flush_now = bool(phrase_started_at and now - phrase_started_at >= _PHRASE_MAX_SECONDS)
            if not flush_now:
                _schedule_phrase_flush()
        if flush_now:
            _flush_phrase("max duration")

    def _on_transcript(text: str) -> None:
        heard_wakeword = _contains_audio_wakeword(text)
        if _WAKEWORD_REQUIRE and not _is_listening_now():
            if not heard_wakeword:
                log.debug('dropped transcript (no wakeword window): %r', text)
                return
            log.info("wakeword transcript opened listen window: %s", text)
        if _WAKEWORD_REQUIRE or _wakeword_model is not None or heard_wakeword:
            _extend_listening()
        wakeword_only = _is_audio_wakeword_only(text)
        if wakeword_only:
            if _tts_speaking.is_set() and _is_likely_self_speech(text):
                log.info("ignored self-speech wakeword during TTS: %s", text)
                return
            if _tts_speaking.is_set():
                log.info("ignored wakeword during TTS: %s", text)
                return
            response = wakeword_response()
            spoken = response.get("tts") or response.get("message") or ""
            with _transcripts_lock:
                _transcripts.append({"text": text, "timestamp": time.time(), "wakeword": True})
            log.info("wakeword transcript: %s", text)
            update_state({
                "audio": {
                    "hearing": True,
                    "last_transcript_at": time.time(),
                    "last_transcript_text": text,
                },
                "latest_user": {"text": text, "source": source, "timestamp": time.time()},
            })
            _notify_ui_event(event_url, {"type": "user_message", "text": text, "source": source})
            if spoken:
                update_state({"latest_assistant": {"text": spoken, "source": "wakeword", "timestamp": time.time()}})
                _notify_ui_event(event_url, {"type": "assistant_message", "text": spoken, "source": "wakeword"})
                _start_tts(tts, spoken)
            return
        # Drop obvious noise first — single stopwords from Vosk false-positives,
        # sub-threshold lengths. These never reach vault, never trigger TTS,
        # never enter _transcripts history.
        rejected, reason = _is_noise_transcript(text)
        if rejected:
            log.info("dropped noise transcript (%s): %r", reason, text)
            return
        if _tts_speaking.is_set() and not _is_likely_self_speech(text):
            log.info("ignored transcript fragment during TTS: %s", text)
            return
        _queue_phrase_part(text)
    return _on_transcript


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or "").casefold())


def _is_likely_self_speech(text: str) -> bool:
    heard = _words(text)
    if not heard:
        return True
    with _tts_text_lock:
        recent = list(_tts_recent_texts)
        if _tts_current_text:
            recent.append((_tts_current_text, time.time()))
    if not recent:
        return False
    linger_seconds = float(os.environ.get("AUDIO_SELF_ECHO_LINGER_SECONDS", "12"))
    threshold = float(os.environ.get("AUDIO_SELF_ECHO_OVERLAP", "0.55"))
    now = time.time()
    for spoken_text, ended_at in recent:
        if ended_at and now - ended_at > linger_seconds:
            continue
        spoken = _words(spoken_text)
        if not spoken:
            continue
        spoken_set = set(spoken)
        overlap = sum(1 for word in heard if word in spoken_set)
        if len(heard) <= 2 and overlap == len(heard):
            return True
        if (overlap / max(1, len(set(heard)))) >= threshold:
            return True
    return False


def _current_tts_generation() -> int:
    with _tts_gen_lock:
        return _tts_generation


def _cancel_queued_tts(tts) -> None:
    """Drain queued TTS chunks and stop current playback.

    Bumps the generation counter so any thread queued on _tts_lock — or
    looping through internal chunks inside a running _speak — short-
    circuits and returns. Also signals the engine to abort its current
    subprocess pair. Use before queueing replacement speech (e.g., when
    the vault emits a `redo` event).
    """
    global _tts_generation
    with _tts_gen_lock:
        _tts_generation += 1
    try:
        if hasattr(tts, "interrupt"):
            tts.interrupt()
    except Exception:
        pass


def _speak(tts, text: str, my_gen: int | None = None) -> None:
    # If this _speak was queued behind speech that has since been cancelled
    # (gen bump), bail before holding the lock — keeps the queue drained.
    if my_gen is not None and my_gen != _current_tts_generation():
        return
    with _tts_lock:
        if my_gen is not None and my_gen != _current_tts_generation():
            return
        global _tts_current_text, _tts_last_started_at, _tts_last_ended_at
        with _tts_text_lock:
            _tts_current_text = str(text or "")
            _tts_last_started_at = time.time()
        update_state({
            "audio": {
                "speaking": True,
                "speaking_started_at": _tts_last_started_at,
                "speaking_text": _tts_current_text,
                "interrupt_enabled": False,
            }
        })
        _tts_speaking.set()
        try:
            chunks = _split_speech_chunks(text) if _TTS_CHUNK_ENABLE else [text]
            if not chunks:
                chunks = [text or ""]
            for chunk in chunks:
                # Cancellation can happen mid-utterance (long reply,
                # streaming "redo" fires after several chunks). Check each
                # iteration so we abandon the remaining chunks.
                if my_gen is not None and my_gen != _current_tts_generation():
                    break
                chunk = chunk.strip()
                if not chunk:
                    continue
                tts.speak(chunk)
        finally:
            _tts_speaking.clear()
            with _tts_text_lock:
                _tts_last_ended_at = time.time()
                if _tts_current_text:
                    _tts_recent_texts.append((_tts_current_text, _tts_last_ended_at))
                _tts_current_text = ""
            update_state({
                "audio": {
                    "speaking": False,
                    "speaking_ended_at": _tts_last_ended_at,
                    "speaking_text": "",
                    "interrupt_enabled": False,
                }
            })


def _start_tts(tts, text: str) -> None:
    my_gen = _current_tts_generation()
    thread = threading.Thread(
        target=_speak, args=(tts, text), kwargs={"my_gen": my_gen},
        daemon=True, name="audio-tts",
    )
    _tts_threads.append(thread)
    thread.start()


class Handler(BaseHTTPRequestHandler):
    capture: MicCapture | None = None
    tts = None
    stt = None
    event_url = ""
    stream_url = ""

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._json(self._health_payload())
        elif path == "/transcripts":
            with _transcripts_lock:
                items = list(_transcripts)
            self._json({"ok": True, "transcripts": items})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_json()
        if body is None:
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/tts":
            self._handle_tts(body)
        elif path == "/listen":
            self._handle_listen(body)
        elif path == "/interrupt":
            self._handle_interrupt()
        else:
            self.send_error(404)

    def _handle_tts(self, body: dict) -> None:
        text = str(body.get("text") or "").strip()
        if not text:
            self.send_error(400, "missing text")
            return
        if self.tts is None or not self.tts.available:
            self._json({"ok": False, "error": "tts_unavailable", "engine": getattr(self.tts, "name", None)}, status=503)
            return
        if not bool(body.get("silent")):
            _notify_ui_event(
                self.event_url,
                {
                    "type": "assistant_message",
                    "text": text,
                    "source": str(body.get("source") or "audio_node_tts"),
                },
            )
        try:
            _speak(self.tts, text)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, status=500)
            return
        self._json({"ok": True, "engine": self.tts.name})

    def _handle_listen(self, body: dict) -> None:
        muted = bool(body.get("muted"))
        if self.capture is None:
            self._json({"ok": False, "error": "capture_unavailable"}, status=503)
            return
        if muted:
            self.capture.mute()
        else:
            self.capture.unmute()
        self._json({"ok": True, "muted": self.capture.muted})

    def _handle_interrupt(self) -> None:
        self._json({"ok": False, "error": "interrupt_disabled", "speaking": _tts_speaking.is_set()}, status=410)

    def _health_payload(self) -> dict:
        stt_name = getattr(self.stt, "name", "none")
        tts_name = getattr(self.tts, "name", "none")
        capture_running = bool(self.capture and self.capture.running)
        return {
            "ok": True,
            "stt": {
                "engine": stt_name,
                "available": bool(getattr(self.stt, "available", False)),
                "init_error": getattr(self.stt, "_init_error", None),
            },
            "tts": {
                "engine": tts_name,
                "available": bool(getattr(self.tts, "available", False)),
                "speaking": _tts_speaking.is_set(),
                "self_echo_filter": True,
                "interrupt_enabled": False,
                "init_error": getattr(self.tts, "_init_error", None),
                "chunking": {
                    "enabled": _TTS_CHUNK_ENABLE,
                    "min_words": _TTS_CHUNK_MIN_WORDS,
                },
                "streaming": {
                    "enabled": _PRESENCE_STREAM_ENABLE,
                    "stream_url": self.stream_url or None,
                    "min_words": _TTS_STREAM_MIN_WORDS,
                },
            },
            "capture": {
                "running": capture_running,
                "muted": bool(self.capture and self.capture.muted),
                "last_error": getattr(self.capture, "last_error", None),
                "last_transcript_at": getattr(self.capture, "last_transcript_at", 0.0),
                "last_transcript_text": getattr(self.capture, "last_transcript_text", ""),
                "wakeword": {
                    "enabled": _WAKEWORD_ENABLED,
                    "required": _WAKEWORD_REQUIRE,
                    "engine": "openwakeword" if _wakeword_model else "none",
                    "threshold": _WAKEWORD_THRESHOLD,
                    "listen_window_s": _WAKEWORD_LISTEN_SECONDS,
                    "is_listening": _is_listening_now(),
                    "last_score": float(_wakeword_last_score),
                    "recent_max_score": float(max(_wakeword_recent_scores)) if _wakeword_recent_scores else 0.0,
                    "recent_window_chunks": len(_wakeword_recent_scores),
                },
                "noise_filter": {
                    "min_chars": _MIN_TRANSCRIPT_CHARS,
                    "min_words": _MIN_TRANSCRIPT_WORDS,
                    "stopword_reject": sorted(_STOPWORD_REJECT),
                },
                "phrase_buffer": {
                    "gap_seconds": _PHRASE_GAP_SECONDS,
                    "max_seconds": _PHRASE_MAX_SECONDS,
                    "min_chars": _PHRASE_MIN_CHARS,
                    "min_words": _PHRASE_MIN_WORDS,
                },
            },
        }

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return None

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)


def main() -> None:
    host = os.environ.get("AUDIO_HOST", "0.0.0.0")
    port = int(os.environ.get("AUDIO_PORT", "5004"))
    stt_name = os.environ.get("AUDIO_STT_ENGINE", "vosk")
    tts_name = os.environ.get("AUDIO_TTS_ENGINE", "espeak")
    presence_url = os.environ.get(
        "AUDIO_PRESENCE_URL",
        f"http://127.0.0.1:{os.environ.get('PRESENCE_PORT', '5002')}/presence/message",
    )
    stream_url = _derive_stream_url(
        presence_url,
        os.environ.get("AUDIO_PRESENCE_STREAM_URL", ""),
    )
    event_url = os.environ.get("AUDIO_UI_EVENT_URL") or os.environ.get("AUDIO_DISPLAY_URL", "")
    source = os.environ.get("AUDIO_SOURCE", "audio_node")
    node_id = os.environ.get("LUHKAS_NODE_ID", "kiosk")

    stt = load_stt(stt_name)
    tts = load_tts(tts_name)
    log.info("stt=%s available=%s; tts=%s available=%s", stt.name, stt.available, tts.name, tts.available)
    if not stt.available:
        log.warning("STT unavailable (%s) — running output-only", getattr(stt, "_init_error", "?"))

    _init_wakeword()
    capture = MicCapture(
        stt=stt,
        on_transcript=_make_transcript_handler(presence_url, source, node_id, tts, event_url, stream_url),
        on_chunk=_on_audio_chunk,
    )
    capture.start()

    Handler.capture = capture
    Handler.tts = tts
    Handler.stt = stt
    Handler.event_url = event_url

    Handler.stream_url = stream_url
    log.info(
        "listening on http://%s:%s (presence=%s, stream=%s)",
        host, port, presence_url,
        stream_url if (stream_url and _PRESENCE_STREAM_ENABLE) else "disabled",
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
