"""Periodic health check for the Wikipedia ingest.

Runs as a systemd-user timer (default: every 2 minutes). Reads the
ingest state file + checks the systemd-user scope; on a transition
from "running" to "stopped without completion" or "running → completed",
writes an event to vault_v2.db so the chat surface auto-surfaces a
``notification_alert`` on the user's next interaction, and so the
deterministic "any updates" command lists it.

Stateless across reboots except for ``watchdog_state.json``, which
records the last-observed status to detect transitions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


VAULT_DIR = Path(os.environ.get("VAULT_DIR", "/home/vault/luhkas/vault"))
STATE_FILE = Path(os.environ.get(
    "WORLD_INGEST_STATE_FILE",
    "/home/vault/world_data/logs/ingest_wiki.state.json",
))
SCOPE_FILE = Path(os.environ.get(
    "WORLD_INGEST_SCOPE_FILE",
    "/home/vault/world_data/logs/ingest_wiki.pid.scope",
))
WATCHDOG_STATE = Path(os.environ.get(
    "WORLD_WATCHDOG_STATE",
    "/home/vault/world_data/logs/ingest_watchdog.state.json",
))
LOG_FILE = Path(os.environ.get(
    "WORLD_WATCHDOG_LOG",
    "/home/vault/world_data/logs/ingest_watchdog.log",
))
# A "stall" alert is only sent if the ingest stays stopped for at least
# this many seconds. Avoids noisy alerts during the brief gap when the
# user runs `stop` then `start` via chat.
STALL_GRACE_SECONDS = int(os.environ.get("WORLD_WATCHDOG_STALL_GRACE_S", "180"))


def _log(msg: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
    except Exception:
        pass


def _scope_active() -> bool:
    if not SCOPE_FILE.exists():
        return False
    try:
        scope = SCOPE_FILE.read_text().strip()
    except Exception:
        return False
    if not scope:
        return False
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-active", f"{scope}.scope"],
            capture_output=True, text=True, timeout=4,
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)


VAULT_URL = os.environ.get("VAULT_URL", "http://127.0.0.1:7000")


def _emit_event(event_type: str, message: str, data: dict) -> bool:
    """Persist to vault_v2.db AND push to vault's alert router.

    Two-step delivery:
      1. event_log.write — persistent in vault_v2.db, surfaces via the
         deterministic 'any updates' command and the auto-attach
         notification_alert on the next chat response.
      2. POST /alerts/enqueue — sends to vault_runtime.node_registry,
         which decides immediate-vs-deferred routing based on whether
         any node has a currently-interacting user. Deferred alerts
         wait in a persistent queue until a user-present signal fires.

    Both writes are best-effort; either one alone still lets the user
    see the alert eventually."""
    sys.path.insert(0, str(VAULT_DIR))
    ok_event = False
    try:
        from event_log import EventLog
        log = EventLog()
        job_id = f"world_ingest:{event_type}:{int(time.time())}"
        log.write(
            job_id=job_id,
            event_type=event_type,
            message=message[:1200],
            data=data,
        )
        _log(f"event written: {event_type} job_id={job_id}")
        ok_event = True
    except Exception as exc:
        _log(f"event_log write FAILED: {exc!r}")

    # Push to the live runtime's alert router. If vault-runtime is
    # down this is a no-op (and the event still landed in the DB above).
    try:
        import json as _json
        import urllib.request as _ur
        payload = _json.dumps({
            "alert": {
                "event_type": event_type,
                "message": message,
                "data": data,
                "source": "world_ingest_watchdog",
            }
        }).encode("utf-8")
        req = _ur.Request(
            f"{VAULT_URL.rstrip('/')}/alerts/enqueue",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=5) as r:
            body = _json.loads(r.read())
        _log(f"alerts/enqueue: {body}")
    except Exception as exc:
        _log(f"alerts/enqueue POST FAILED: {exc!r}")

    return ok_event


def _snapshot(state: dict | None) -> dict:
    if not state:
        return {}
    now = time.time()
    elapsed = state.get("elapsed_s") or 0
    if not elapsed and state.get("started_at"):
        try:
            elapsed = now - float(state["started_at"])
        except Exception:
            pass
    return {
        "articles_seen": state.get("articles_seen"),
        "articles_new": state.get("articles_new"),
        "articles_replaced": state.get("articles_replaced"),
        "articles_skipped_unchanged": state.get("articles_skipped_unchanged"),
        "chunks_written": state.get("chunks_written"),
        "last_committed_index": state.get("last_committed_index"),
        "elapsed_s": round(elapsed, 1),
        "started_at_iso": state.get("started_at_iso"),
    }


def _classify(scope_active: bool, state: dict | None) -> str:
    if scope_active:
        return "running"
    if state and state.get("completed"):
        return "completed"
    if state is not None:
        return "stopped"
    return "none"


def main() -> int:
    state = _read_json(STATE_FILE)
    watchdog = _read_json(WATCHDOG_STATE) or {}
    last_status = watchdog.get("status")
    first_stopped_at = watchdog.get("first_stopped_at")

    scope_active = _scope_active()
    current = _classify(scope_active, state)
    snap = _snapshot(state)
    now = time.time()

    alert_fired = None

    if current == "running":
        # Reset stall timer when we go back to running.
        first_stopped_at = None

    elif current == "stopped":
        # Start the grace timer on the first observation that it's
        # stopped without completion. Only fire the alert if it's been
        # stopped for >= STALL_GRACE_SECONDS (avoids noise during user
        # stop/start cycles via chat).
        if last_status == "running":
            first_stopped_at = now
            _log(f"transition running -> stopped; grace timer started ({STALL_GRACE_SECONDS}s)")
        if first_stopped_at and (now - first_stopped_at) >= STALL_GRACE_SECONDS:
            # Only fire once per stall: check whether we already alerted
            # for this stopped epoch.
            if watchdog.get("last_stall_alert_at") != first_stopped_at:
                cursor = snap.get("last_committed_index")
                new = snap.get("articles_new") or 0
                msg = (
                    f"Wikipedia ingest stopped unexpectedly at entry "
                    f"{cursor}. {new} articles ingested this session. "
                    f"Say 'start the wikipedia ingest' to resume "
                    f"from the saved cursor."
                )
                if _emit_event("world_ingest_stalled", msg, snap):
                    alert_fired = "world_ingest_stalled"
                    watchdog["last_stall_alert_at"] = first_stopped_at

    elif current == "completed":
        # Fire-once: only on the first observation that completion happened.
        if last_status != "completed":
            new = snap.get("articles_new") or 0
            chunks = snap.get("chunks_written") or 0
            msg = (
                f"Wikipedia ingest completed. "
                f"{new} articles, {chunks} chunks ingested this session. "
                f"Auto-build of the wiki search index will run next "
                f"(or has already run — say 'wiki index status')."
            )
            if _emit_event("world_ingest_completed", msg, snap):
                alert_fired = "world_ingest_completed"
        first_stopped_at = None

    # "none" → no state file yet, nothing to alert.

    _atomic_write_json(WATCHDOG_STATE, {
        "status": current,
        "checked_at": now,
        "checked_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "first_stopped_at": first_stopped_at,
        "last_stall_alert_at": watchdog.get("last_stall_alert_at"),
        "last_alert_fired": alert_fired or watchdog.get("last_alert_fired"),
        "snapshot": snap,
    })
    _log(f"check: status={current} prev={last_status} alert={alert_fired or '-'} cursor={snap.get('last_committed_index')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
