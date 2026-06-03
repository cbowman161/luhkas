#!/usr/bin/env python3
"""Run the wiki ingest only when the vault GPU is idle.

The native embedder pulls ~11 GB of VRAM which collides with the chat /
vision / router models when a user interaction triggers an LLM call.
This supervisor watches a single, precise signal:

  ``last_ollama_activity_at`` from vault's /health — set at the start of
  every Ollama dispatch (chat, embed, vision) made by vault. If an
  Ollama call happened within ``USER_ACTIVITY_WINDOW`` seconds, the
  system is considered busy.

Why this signal specifically: deterministic routes (mute toggle, service
info, learned silent_routes that bypass the LLM) don't contend with the
ingest for VRAM, so they shouldn't trigger a pause. The older
``last_user_activity_at`` signal was a coarse proxy ("user interacting"
≈ "GPU about to be busy") that paused ingest for any interaction —
wasteful for non-LLM routes. Tracking Ollama dispatches directly fixes
that.

On busy, the ingest is stopped via SIGTERM (its CLI uses
``--resume-from-state`` so progress is preserved). When the signal has
been quiet for ``MIN_IDLE_SECONDS``, the ingest is launched. The
supervisor restarts the child if it exits on its own.

Config (env overrides):
  VAULT_INGEST_POLL_INTERVAL          seconds between checks (default 30)
  VAULT_INGEST_MIN_IDLE_SECONDS       sustained idle window required to start
                                      (default 300 — 5 min)
  VAULT_INGEST_USER_ACTIVITY_WINDOW   seconds since last user activity that
                                      still counts as busy (default 180)
  VAULT_INGEST_OLLAMA_BUSY_WINDOW     seconds since last Ollama touch that
                                      still counts as busy (default 90)
  VAULT_INGEST_HEALTH_URL             health endpoint (default
                                      http://127.0.0.1:7000/health)
  VAULT_INGEST_OLLAMA_URL             ollama base (default
                                      http://127.0.0.1:11434)
  VAULT_INGEST_ZIM_PATH               required, path to the .zim corpus
  VAULT_INGEST_STATE_FILE             required, JSON state file path
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


# Poll cadence. Was 30s — too coarse: meant up to 30s of "ingest
# technically still running" after a user's LLM call landed, which
# the kiosk display surfaces as yellow ring during the gap. 5s gives
# near-immediate pause response without measurable overhead (each
# poll is one cheap /health + one fragment-count via iterdir).
POLL_INTERVAL = float(os.environ.get("VAULT_INGEST_POLL_INTERVAL", "5"))
MIN_IDLE_SECONDS = float(os.environ.get("VAULT_INGEST_MIN_IDLE_SECONDS", "120"))
USER_ACTIVITY_WINDOW = float(os.environ.get("VAULT_INGEST_USER_ACTIVITY_WINDOW", "120"))
HEALTH_URL = os.environ.get("VAULT_INGEST_HEALTH_URL", "http://127.0.0.1:7000/health")
OLLAMA_URL = os.environ.get("VAULT_INGEST_OLLAMA_URL", "http://127.0.0.1:11434")
ZIM_PATH = os.environ.get("VAULT_INGEST_ZIM_PATH", "")
STATE_FILE = os.environ.get("VAULT_INGEST_STATE_FILE", "")
EMBEDDER = os.environ.get("VAULT_INGEST_EMBEDDER", "native")
BATCH = os.environ.get("VAULT_INGEST_BATCH", "64")

# Fragment-triggered compaction. ingest_wiki appends a new lance
# fragment per flush; past ~10k fragments per table, write throughput
# collapses (observed 21/s -> 0.2/s at 40k fragments). Compact when we
# cross this threshold rather than relying on the time-based timer
# alone — keeps performance bounded regardless of ingest rate.
COMPACT_FRAGMENT_THRESHOLD = int(os.environ.get("VAULT_INGEST_COMPACT_FRAGMENT_THRESHOLD", "8000"))
# Largest write-volume table to watch. wiki_chunks always has roughly
# the same fragment count as wiki_articles, so monitoring one suffices.
COMPACT_WATCH_TABLE = os.environ.get(
    "VAULT_INGEST_COMPACT_WATCH_TABLE", "wiki_chunks"
)
WORLD_DB_PATH = os.environ.get("WORLD_DB_PATH", "/home/vault/world_data/world.lance")
# Throughput tunables surfaced from ingest_wiki args. Defaults match the
# ingest_wiki defaults so behavior is unchanged unless overridden.
# - concurrency: parallel embed workers. >1 lets one batch's embed run
#   while another batch flushes to LanceDB. Useful when DB-write latency
#   matters; redundant if the embed itself is the bottleneck.
# - prefetch_queue: max parsed articles buffered between parser and
#   embed/write loop. queue=24/24 in logs means raise this to give the
#   producer more headroom while the writer drains.
CONCURRENCY = os.environ.get("VAULT_INGEST_CONCURRENCY", "")
PREFETCH_QUEUE = os.environ.get("VAULT_INGEST_PREFETCH_QUEUE", "")
WORKDIR = os.environ.get("VAULT_INGEST_WORKDIR", str(Path(__file__).resolve().parents[1]))
GRACEFUL_STOP_TIMEOUT = float(os.environ.get("VAULT_INGEST_STOP_TIMEOUT", "45"))


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] ingest_supervisor: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_supervisor")


# ---------------------------------------------------------------------------
# Busy signals
# ---------------------------------------------------------------------------


def _now_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return "?"


def _parse_iso_z(value: str) -> float:
    """Parse Ollama's ``expires_at`` (RFC3339 with 'Z' or +00:00) to epoch."""
    if not value:
        return 0.0
    s = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def vault_idle_seconds() -> float | None:
    """Seconds since last Ollama dispatch, per vault /health.

    Returns ``None`` if vault is unreachable (we'll skip this poll) or if
    no Ollama activity has ever been recorded (then we assume idle so
    the supervisor doesn't deadlock waiting for an LLM call after a
    fresh vault restart).

    Falls back to ``last_user_activity_at`` if the newer
    ``last_ollama_activity_at`` field isn't present, so an old vault
    build keeps working until it's rolled forward.
    """
    try:
        with urlopen(HEALTH_URL, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        log.warning("vault /health unreachable (%s); skipping poll", exc)
        return None
    last = data.get("last_ollama_activity_at")
    if last is None:
        # Backwards-compat with vault builds that don't expose the
        # GPU-precise signal yet.
        last = data.get("last_user_activity_at") or 0.0
    last = float(last or 0.0)
    if last <= 0:
        return float("inf")  # nothing ever ran → fully idle
    return max(0.0, time.time() - last)


def system_status() -> tuple[bool, float | None, str]:
    """Return (busy, idle_seconds, reason).

    busy=True means an Ollama call happened within
    ``USER_ACTIVITY_WINDOW`` seconds. (The env name is unchanged for
    config compatibility but the semantic is now "Ollama activity
    window" — non-LLM interactions are intentionally ignored so
    deterministic routes don't pause the ingest.)
    """
    idle = vault_idle_seconds()
    if idle is None:
        return False, None, "vault unreachable"
    if idle == float("inf"):
        return False, idle, "no Ollama activity recorded yet"
    if idle < USER_ACTIVITY_WINDOW:
        return True, idle, f"ollama call {idle:.0f}s ago"
    return False, idle, f"last ollama call {idle:.0f}s ago"


# ---------------------------------------------------------------------------
# Child lifecycle
# ---------------------------------------------------------------------------


def build_ingest_argv() -> list[str]:
    if not ZIM_PATH:
        log.error("VAULT_INGEST_ZIM_PATH is required")
        sys.exit(2)
    if not STATE_FILE:
        log.error("VAULT_INGEST_STATE_FILE is required")
        sys.exit(2)
    argv = [
        sys.executable, "-u", "-m", "world.ingest_wiki",
        ZIM_PATH,
        "--embedder", EMBEDDER,
        "--batch", str(BATCH),
        "--state-file", STATE_FILE,
        "--resume-from-state",
    ]
    if CONCURRENCY:
        argv += ["--concurrency", str(CONCURRENCY)]
    if PREFETCH_QUEUE:
        argv += ["--prefetch-queue", str(PREFETCH_QUEUE)]
    return argv


def state_is_complete() -> bool:
    try:
        with open(STATE_FILE) as fh:
            data = json.load(fh)
        return bool(data.get("completed"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def start_ingest() -> subprocess.Popen | None:
    if state_is_complete():
        log.info("ingest already completed (per state file); not starting")
        return None
    argv = build_ingest_argv()
    log.info("starting ingest: %s", " ".join(argv))
    return subprocess.Popen(
        argv,
        cwd=WORKDIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )


def stop_ingest(proc: subprocess.Popen) -> None:
    log.info("stopping ingest pid=%s (SIGTERM, %ss grace)", proc.pid, GRACEFUL_STOP_TIMEOUT)
    try:
        proc.terminate()
    except Exception as exc:
        log.warning("terminate failed: %s", exc)
    try:
        proc.wait(timeout=GRACEFUL_STOP_TIMEOUT)
        log.info("ingest exited cleanly (rc=%s)", proc.returncode)
    except subprocess.TimeoutExpired:
        log.warning("ingest did not exit in %ss; sending SIGKILL", GRACEFUL_STOP_TIMEOUT)
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


_stop = False


def _on_signal(signum, _frame):
    global _stop
    log.info("received signal %s; shutting down", signum)
    _stop = True


def _count_fragments(table_name: str) -> int:
    """Cheap fragment count from sysfs-style iterdir on the lance
    table's data dir. Mirrors compact._count_fragments — duplicated
    here to avoid importing the heavy lancedb module just to read a
    directory."""
    data_dir = Path(WORLD_DB_PATH) / f"{table_name}.lance" / "data"
    if not data_dir.is_dir():
        return 0
    try:
        return sum(1 for _ in data_dir.iterdir())
    except OSError:
        return -1


def _run_compaction() -> None:
    """Run the compaction script synchronously. Ingest must be stopped
    first — calling code is responsible for that ordering. Compaction
    typically takes 30-120s depending on fragment counts and the index
    rebuild cost; we wait it out and resume the loop afterward."""
    log.info("running compaction (world.compact)")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "world.compact"],
            cwd=WORKDIR,
            env=os.environ.copy(),
            timeout=600,  # IVF_PQ rebuild on a fresh chunks table is the slowest case (~90s)
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            log.info("compaction completed in %.1fs", elapsed)
        else:
            log.warning("compaction returned rc=%s in %.1fs: %s",
                        result.returncode, elapsed, result.stderr.strip()[:400])
    except subprocess.TimeoutExpired:
        log.error("compaction timed out after 600s")
    except Exception as exc:
        log.error("compaction failed to start: %s", exc)


def _post_warm_models() -> None:
    """Ask vault to warm chat/router/vision models. Best-effort — never
    raises into the supervisor loop. Used after a pause and after the
    ingest completes so the first user interaction afterwards isn't a
    cold model load."""
    url = HEALTH_URL.replace("/health", "/admin/warm_models")
    try:
        import urllib.request
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()
        log.info("warm_models requested (%s)", url)
    except Exception as exc:
        log.warning("warm_models request failed (%s): %s", url, exc)


def main() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info(
        "config: poll=%ss min_idle=%ss user_window=%ss zim=%s state=%s",
        POLL_INTERVAL, MIN_IDLE_SECONDS, USER_ACTIVITY_WINDOW,
        ZIM_PATH, STATE_FILE,
    )

    proc: subprocess.Popen | None = None
    last_state = None  # one of: "busy", "idle_waiting", "running", "complete"
    completion_warm_sent = False

    while not _stop:
        busy, idle_seconds, reason = system_status()
        proc_alive = bool(proc and proc.poll() is None)

        if busy:
            if proc_alive:
                log.info("system busy (%s) — stopping ingest", reason)
                stop_ingest(proc)
                proc = None
                # Pre-warm models now that bge-m3 is freeing VRAM.
                # Fires once per pause transition; subsequent polls
                # during the wait window won't re-trigger because
                # proc is already None.
                _post_warm_models()
            if last_state != "busy":
                log.info("state=busy (%s)", reason)
                last_state = "busy"
        elif idle_seconds is None:
            # vault unreachable — be conservative, don't change state.
            if last_state != "unreachable":
                log.info("state=unreachable (%s)", reason)
                last_state = "unreachable"
        else:
            if proc_alive:
                # Fragment-triggered compaction check: when the watched
                # table's fragment count crosses the threshold, stop
                # ingest, run compaction inline (~30-120s), then let
                # the next poll restart ingest naturally. This keeps
                # fragment counts bounded without relying solely on
                # the time-based timer.
                frag_count = _count_fragments(COMPACT_WATCH_TABLE)
                if frag_count >= COMPACT_FRAGMENT_THRESHOLD:
                    log.info(
                        "fragment threshold crossed: %s=%d >= %d — stopping ingest to compact",
                        COMPACT_WATCH_TABLE, frag_count, COMPACT_FRAGMENT_THRESHOLD,
                    )
                    stop_ingest(proc)
                    proc = None
                    last_state = "compacting"
                    _run_compaction()
                    # Fall through; next loop iteration will check
                    # ollama-idle and restart ingest if appropriate.
                    # Avoid the per-state log noise by continuing.
                    continue
                if last_state != "running":
                    log.info("state=running ingest (idle for %.0fs)", idle_seconds)
                    last_state = "running"
                completion_warm_sent = False
            elif state_is_complete():
                if last_state != "complete":
                    log.info("ingest already completed per state file — supervisor dormant")
                    last_state = "complete"
                if not completion_warm_sent:
                    # Ingest is done — keep models warm so post-
                    # completion interactions stay snappy.
                    _post_warm_models()
                    completion_warm_sent = True
                for _ in range(60):
                    if _stop:
                        break
                    time.sleep(POLL_INTERVAL)
                continue
            elif idle_seconds >= MIN_IDLE_SECONDS:
                idle_str = "∞" if idle_seconds == float("inf") else f"{idle_seconds:.0f}s"
                log.info(
                    "state=starting ingest (idle %s >= %ss; %s)",
                    idle_str, MIN_IDLE_SECONDS, reason,
                )
                proc = start_ingest()
                last_state = "running"
                completion_warm_sent = False
            else:
                if last_state != "idle_waiting":
                    log.info(
                        "state=idle_waiting (%.0fs/%ss before start; %s)",
                        idle_seconds, MIN_IDLE_SECONDS, reason,
                    )
                    last_state = "idle_waiting"

        # If the child exited on its own, reset.
        if proc and proc.poll() is not None:
            log.info("ingest exited on its own (rc=%s)", proc.returncode)
            proc = None
            last_state = None

        # Sleep in small slices so signals are responsive.
        slept = 0.0
        while slept < POLL_INTERVAL and not _stop:
            time.sleep(min(2.0, POLL_INTERVAL - slept))
            slept += 2.0

    # Shutdown
    if proc and proc.poll() is None:
        stop_ingest(proc)
    log.info("supervisor exited")


if __name__ == "__main__":
    main()
