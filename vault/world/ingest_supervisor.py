#!/usr/bin/env python3
"""Run the wiki ingest only when the vault is idle.

The native embedder pulls ~11 GB of VRAM which collides with the chat /
vision / router models when a user is interacting. This supervisor watches
two signals and starts or stops the ingest accordingly:

  1. ``last_user_activity_at`` from vault's /health — set every time
     ``VaultRuntime.handle`` or ``handle_presence`` is invoked. If a user
     message arrived within the recent activity window, the system is busy.

  2. Recently-touched Ollama models — by querying ``/api/ps`` and looking at
     each model's ``expires_at``. Ollama refreshes ``expires_at`` on every
     request, so a freshly-renewed keep-alive means recent inference even
     if no presence message landed (e.g. background fact-relation classifier
     calls during a turn).

If either signal indicates recent activity, the ingest is stopped via
SIGTERM (its CLI uses ``--resume-from-state`` so progress is preserved).
When both signals have been quiet for ``MIN_IDLE_SECONDS``, the ingest is
launched. The supervisor restarts the child if it exits on its own.

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


POLL_INTERVAL = float(os.environ.get("VAULT_INGEST_POLL_INTERVAL", "30"))
MIN_IDLE_SECONDS = float(os.environ.get("VAULT_INGEST_MIN_IDLE_SECONDS", "120"))
USER_ACTIVITY_WINDOW = float(os.environ.get("VAULT_INGEST_USER_ACTIVITY_WINDOW", "120"))
HEALTH_URL = os.environ.get("VAULT_INGEST_HEALTH_URL", "http://127.0.0.1:7000/health")
OLLAMA_URL = os.environ.get("VAULT_INGEST_OLLAMA_URL", "http://127.0.0.1:11434")
ZIM_PATH = os.environ.get("VAULT_INGEST_ZIM_PATH", "")
STATE_FILE = os.environ.get("VAULT_INGEST_STATE_FILE", "")
EMBEDDER = os.environ.get("VAULT_INGEST_EMBEDDER", "native")
BATCH = os.environ.get("VAULT_INGEST_BATCH", "64")
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
    """Seconds since last user activity, per vault /health.

    Returns ``None`` if vault is unreachable (we'll skip this poll) or if
    no user activity has ever been recorded (then we assume idle so the
    supervisor doesn't deadlock waiting for a message after a fresh
    vault restart).
    """
    try:
        with urlopen(HEALTH_URL, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        log.warning("vault /health unreachable (%s); skipping poll", exc)
        return None
    last = float(data.get("last_user_activity_at") or 0.0)
    if last <= 0:
        return float("inf")  # never any user activity → fully idle
    return max(0.0, time.time() - last)


def system_status() -> tuple[bool, float | None, str]:
    """Return (busy, idle_seconds, reason).

    busy=True means a user message arrived within ``USER_ACTIVITY_WINDOW``.
    idle_seconds is the real duration since the last user message (read
    from vault, not tracked locally) — so a freshly-started supervisor
    knows the system was already idle for a long time.
    """
    idle = vault_idle_seconds()
    if idle is None:
        return False, None, "vault unreachable"
    if idle == float("inf"):
        return False, idle, "no user activity recorded yet"
    if idle < USER_ACTIVITY_WINDOW:
        return True, idle, f"user message {idle:.0f}s ago"
    return False, idle, f"last user message {idle:.0f}s ago"


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
    return [
        sys.executable, "-u", "-m", "world.ingest_wiki",
        ZIM_PATH,
        "--embedder", EMBEDDER,
        "--batch", str(BATCH),
        "--state-file", STATE_FILE,
        "--resume-from-state",
    ]


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

    while not _stop:
        busy, idle_seconds, reason = system_status()
        proc_alive = bool(proc and proc.poll() is None)

        if busy:
            if proc_alive:
                log.info("system busy (%s) — stopping ingest", reason)
                stop_ingest(proc)
                proc = None
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
                if last_state != "running":
                    log.info("state=running ingest (idle for %.0fs)", idle_seconds)
                    last_state = "running"
            elif state_is_complete():
                if last_state != "complete":
                    log.info("ingest already completed per state file — supervisor dormant")
                    last_state = "complete"
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
