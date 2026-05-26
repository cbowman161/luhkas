"""Chat-driven admin surface for the Wikipedia ingest.

Recognizes a small set of phrasings ("start the wiki ingest", "wikipedia
progress", "stop the wikipedia ingestion", ...) and dispatches to the
runner script. Status reads the live state file and formats an ETA.

The handler is plain Python — no LLM, no router involvement. It returns
either a dict the vault runtime can ship straight to the user, or None
to let the message fall through to normal routing."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


RUNNER_PATH = Path(os.environ.get(
    "WORLD_INGEST_RUNNER",
    "/home/vault/luhkas/vault/world/run_full_ingest.sh",
))
STATE_FILE = Path(os.environ.get(
    "WORLD_INGEST_STATE_FILE",
    "/home/vault/world_data/logs/ingest_wiki.state.json",
))
PID_FILE = Path(os.environ.get(
    "WORLD_INGEST_PID_FILE",
    "/home/vault/world_data/logs/ingest_wiki.pid",
))
SCOPE_FILE = Path(str(PID_FILE) + ".scope")
LOG_FILE = Path(os.environ.get(
    "WORLD_INGEST_LOG_FILE",
    "/home/vault/world_data/logs/ingest_wiki.log",
))
# Conservative pre-filter estimate of how many ZIM entries are real
# articles vs soft-redirects/non-html. Used only for ETA display.
ZIM_REAL_ARTICLE_FRACTION = float(os.environ.get("WORLD_ZIM_REAL_FRACTION", "0.55"))
ZIM_TOTAL_ENTRIES_FALLBACK = int(os.environ.get("WORLD_ZIM_TOTAL_ENTRIES", "19551505"))


# Action triggers — each is a tight regex so chat lines like
# "what is wikipedia" don't accidentally match.
_START_RE = re.compile(
    r"\b(?:start|begin|kick\s*off|launch|resume)\s+(?:the\s+)?"
    r"(?:wiki(?:pedia)?\s+)?(?:ingest(?:ion)?|ingestion|indexing|index)\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"\b(?:stop|halt|pause|cancel|kill)\s+(?:the\s+)?"
    r"(?:wiki(?:pedia)?\s+)?(?:ingest(?:ion)?|ingestion|indexing|index)\b",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"\b(?:"
    r"wiki(?:pedia)?\s+(?:ingest(?:ion)?\s+)?(?:status|progress|state|how['’]?s)"
    r"|(?:ingest(?:ion)?|indexing)\s+(?:status|progress|state)"
    r"|how(?:\s+is|\s*['’]?s)\s+(?:the\s+)?(?:wiki(?:pedia)?|ingest(?:ion)?)"
    r"(?:\s+(?:ingest(?:ion)?|going|coming|doing|progressing))?"
    r"|world\s+(?:status|progress)"
    r")\b",
    re.IGNORECASE,
)
_BUILD_INDEX_RE = re.compile(
    r"\b(?:build|create|rebuild|make)\s+(?:the\s+)?"
    r"(?:wiki(?:pedia)?|world)\s+(?:search\s+)?index\b",
    re.IGNORECASE,
)
_INDEX_STATUS_RE = re.compile(
    r"\b(?:is\s+(?:the\s+)?(?:wiki(?:pedia)?|world)\s+(?:search\s+)?index|"
    r"(?:wiki(?:pedia)?|world)\s+(?:search\s+)?index\s+(?:status|state|built))\b",
    re.IGNORECASE,
)


def detect_action(message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None
    if _BUILD_INDEX_RE.search(text):
        return "build_index"
    if _INDEX_STATUS_RE.search(text):
        return "index_status"
    if _START_RE.search(text):
        return "start"
    if _STOP_RE.search(text):
        return "stop"
    if _STATUS_RE.search(text):
        return "status"
    return None


def _runner_call(action: str, timeout: float = 20.0) -> tuple[int, str, str]:
    if not RUNNER_PATH.exists():
        return 127, "", f"runner not found at {RUNNER_PATH}"
    try:
        result = subprocess.run(
            ["bash", str(RUNNER_PATH), action],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:
        return 1, "", str(exc)


def _read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return None


def _scope_python_pid(scope: str) -> int | None:
    """Find the world.ingest_wiki python pid inside a transient scope.

    Tries the cgroup.procs path first (fast, no fork), falls back to
    pgrep filtered to user."""
    uid = os.getuid()
    candidate = Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/"
        f"app.slice/{scope}.scope/cgroup.procs"
    )
    pids: list[int] = []
    if candidate.exists():
        try:
            pids = [int(line.strip()) for line in candidate.read_text().splitlines() if line.strip().isdigit()]
        except Exception:
            pids = []
    # Prefer the actual python ingest process over the bash wrapper that
    # exec'd it (the wrapper exits, but if both are still listed for any
    # reason we want the python).
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"world.ingest_wiki" in cmdline:
                return pid
        except Exception:
            continue
    if pids:
        return pids[0]
    # Last-resort: pgrep -f
    try:
        r = subprocess.run(
            ["pgrep", "-u", str(uid), "-f", "world.ingest_wiki"],
            capture_output=True, text=True, timeout=2,
        )
        first = (r.stdout.splitlines() or [""])[0].strip()
        return int(first) if first.isdigit() else None
    except Exception:
        return None


def _process_running() -> tuple[bool, int | None]:
    """Liveness check.

    Source of truth is the systemd user scope (if recorded) because the
    pid file can lag behind reality — bash subshells exit and the actual
    python ingest pid is only knowable after the scope settles. Falls
    back to the pid file for non-scope launches."""
    scope = None
    if SCOPE_FILE.exists():
        try:
            scope = SCOPE_FILE.read_text().strip()
        except Exception:
            scope = None
    if scope:
        try:
            r = subprocess.run(
                ["systemctl", "--user", "is-active", f"{scope}.scope"],
                capture_output=True, text=True, timeout=3,
            )
            if r.stdout.strip() == "active":
                # Systemd scopes don't expose MainPID like services do.
                # Read the cgroup's process list and pick the python
                # ingest pid (filter out the shell wrapper if present).
                pid = _scope_python_pid(scope)
                return True, pid or 0
        except Exception:
            pass
    # Fallback: stale pid-file path for non-scope launches.
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except OSError:
        return False, pid


def _humanize_seconds(s: float) -> str:
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def _format_status() -> str:
    state = _read_state()
    running, pid = _process_running()
    if state is None and not running:
        return (
            "Wikipedia ingest hasn't been started yet. "
            "Say 'start the wikipedia ingest' to launch it."
        )

    parts: list[str] = []
    starting = bool(state and state.get("phase") == "starting" and not state.get("articles_seen"))
    if running and starting:
        elapsed = time.time() - float(state.get("started_at") or time.time())
        parts.append(
            f"Ingest is starting (pid {pid}) — loading the bge-m3 model on the GPU "
            f"(~15s). {_humanize_seconds(elapsed)} elapsed."
        )
        return " ".join(parts)
    if running:
        parts.append(f"Ingest running (pid {pid}).")
    elif state and state.get("completed"):
        parts.append("Ingest completed.")
    elif state and not state.get("articles_seen"):
        parts.append("Ingest was started but exited before processing any articles.")
    else:
        parts.append("Ingest is not running.")

    if state and not starting:
        seen = state.get("articles_seen") or 0
        new = state.get("articles_new") or 0
        replaced = state.get("articles_replaced") or 0
        skipped = state.get("articles_skipped_unchanged") or 0
        empty = state.get("articles_empty") or 0
        chunks = state.get("chunks_written") or 0
        cursor = state.get("last_committed_index") or 0
        elapsed = state.get("elapsed_s") or 0
        if not elapsed and state.get("started_at"):
            elapsed = time.time() - float(state["started_at"])

        parts.append(
            f"This session: seen {seen}, new {new}, replaced {replaced}, "
            f"skipped-unchanged {skipped}, empty {empty}; "
            f"{chunks} chunks written."
        )

        scan_rate = (seen / elapsed) if elapsed > 0 else 0
        ingested = new + replaced
        ingest_rate = (ingested / elapsed) if elapsed > 0 else 0
        entries_left = max(0, ZIM_TOTAL_ENTRIES_FALLBACK - cursor)

        if running:
            # Two rates because they answer different questions:
            #   scan_rate  = how fast we move the cursor (inflated when
            #                resume-skip means most entries are cheap
            #                hash-checks, not embed work)
            #   ingest_rate = how fast we actually embed new articles
            # ETA from scan_rate during a skip-heavy phase lies; ETA from
            # ingest_rate during steady-state is the real number.
            if ingest_rate >= 0.1:
                real_articles_left = entries_left * ZIM_REAL_ARTICLE_FRACTION
                eta_s = real_articles_left / ingest_rate
                parts.append(
                    f"Scanning {scan_rate:.1f} entries/s, ingesting "
                    f"{ingest_rate:.1f} new articles/s. "
                    f"Cursor {cursor:,}/{ZIM_TOTAL_ENTRIES_FALLBACK:,}. "
                    f"ETA ~{_humanize_seconds(eta_s)} "
                    f"(assumes {ZIM_REAL_ARTICLE_FRACTION:.0%} of remaining "
                    f"entries are real articles)."
                )
            else:
                # In a resume-skip phase OR very early (before first
                # batch flushes), ingest rate is ~0 until we catch up to
                # fresh territory. Don't surface a meaningless cursor of
                # -1 when nothing's happened yet.
                cursor_str = (
                    f"Cursor {max(cursor, 0):,}/{ZIM_TOTAL_ENTRIES_FALLBACK:,}. "
                    if cursor >= 0 else ""
                )
                parts.append(
                    f"Warming up (no flush yet — model load + resume scan). "
                    f"{cursor_str}"
                    f"ETA available once past resume zone."
                )
        else:
            parts.append(
                f"Last cursor: entry {cursor:,}. "
                f"Re-run start to resume from here."
            )
    return " ".join(parts)


def _format_started(stdout: str) -> str:
    # The runner prints pid + paths on start.
    pid_match = re.search(r"started ingest pid=(\d+)", stdout)
    pid = pid_match.group(1) if pid_match else "?"
    return (
        f"Started the Wikipedia ingest (pid {pid}). "
        f"It runs detached and resumes from the last cursor on restart. "
        f"Ask 'wikipedia ingest status' anytime, or 'stop the wikipedia ingest' to halt."
    )


def _format_already_running(stdout: str, stderr: str) -> str:
    blob = (stderr or stdout or "").strip()
    pid_match = re.search(r"pid=(\d+)", blob)
    pid = pid_match.group(1) if pid_match else "?"
    return f"Wikipedia ingest is already running (pid {pid}). Use 'status' to see progress."


def _format_stopped() -> str:
    return (
        "Sent stop signal. The current batch will finish writing, then "
        "the ingest will exit cleanly. Resume cursor is preserved."
    )


def handle(message: str) -> dict | None:
    """Returns a runtime-shaped response dict or None to let the message
    fall through to normal routing."""
    action = detect_action(message)
    if action is None:
        return None

    if action == "status":
        return {
            "message": _format_status(),
            "data": {"world_ingest_action": "status"},
        }

    if action == "start":
        running, pid = _process_running()
        if running:
            return {
                "message": f"Wikipedia ingest is already running (pid {pid}). Say 'stop the wikipedia ingest' if you want to halt it.",
                "data": {"world_ingest_action": "start", "already_running": True, "pid": pid},
            }
        rc, out, err = _runner_call("start")
        if rc != 0:
            return {
                "message": f"I couldn't start the ingest: {(err or out).strip()[:300]}",
                "data": {"world_ingest_action": "start", "error": err or out, "rc": rc},
            }
        if "already running" in (out + err).lower():
            return {
                "message": _format_already_running(out, err),
                "data": {"world_ingest_action": "start", "already_running": True},
            }
        return {
            "message": _format_started(out),
            "data": {"world_ingest_action": "start", "stdout": out.strip()},
        }

    if action == "stop":
        running, pid = _process_running()
        if not running:
            return {
                "message": "Wikipedia ingest isn't running. Nothing to stop.",
                "data": {"world_ingest_action": "stop", "was_running": False},
            }
        rc, out, err = _runner_call("stop")
        if rc != 0:
            return {
                "message": f"I couldn't stop the ingest cleanly: {(err or out).strip()[:300]}",
                "data": {"world_ingest_action": "stop", "error": err or out, "rc": rc},
            }
        return {
            "message": _format_stopped(),
            "data": {"world_ingest_action": "stop", "pid": pid},
        }

    if action == "index_status":
        try:
            from models import get_model
            from world import WorldKnowledgeStore
            store = WorldKnowledgeStore(text_embedder=get_model("embed"))
            rows = store._tables["wiki_chunks"].count_rows()
            st = store.wiki_index_status()
            if st.get("has_vector_index"):
                return {
                    "message": (
                        f"The wiki search index is built. "
                        f"{rows:,} chunks indexed; queries should run in well under 100 ms."
                    ),
                    "data": {"world_ingest_action": "index_status", "rows": rows, "status": st},
                }
            return {
                "message": (
                    f"No ANN index on wiki_chunks yet. "
                    f"{rows:,} chunks present; queries currently use brute-force scan "
                    f"(fine under ~50k chunks, several seconds per query at millions). "
                    f"Say 'build the wiki index' to create one when you're ready."
                ),
                "data": {"world_ingest_action": "index_status", "rows": rows, "status": st},
            }
        except Exception as exc:
            return {
                "message": f"Couldn't read index status: {exc}",
                "data": {"world_ingest_action": "index_status", "error": str(exc)},
            }

    if action == "build_index":
        # Build runs in the chat-runtime process. For small corpora it's
        # fast; for millions of rows it takes minutes-to-hours and the
        # request will block until done. The user explicitly asked for it,
        # so this is acceptable.
        try:
            from models import get_model
            from world import WorldKnowledgeStore
            store = WorldKnowledgeStore(text_embedder=get_model("embed"))
            rows = store._tables["wiki_chunks"].count_rows()
            if rows < 256:
                return {
                    "message": (
                        f"Only {rows:,} chunks in the store — too few to build "
                        f"a meaningful IVF index. Ingest more first."
                    ),
                    "data": {"world_ingest_action": "build_index", "rows": rows},
                }
            result = store.build_wiki_index(force=False)
            if result.get("skipped"):
                return {
                    "message": (
                        f"Wiki index already exists ({rows:,} chunks). "
                        f"Say 'rebuild the wiki index' if you want to force a fresh build."
                    ),
                    "data": {"world_ingest_action": "build_index", **result},
                }
            parts_info = (
                f"partitions={result.get('num_partitions')}, "
                f"sub_vectors={result.get('num_sub_vectors')}"
            )
            return {
                "message": (
                    f"Built IVF_PQ index on {result.get('rows'):,} wiki chunks "
                    f"({parts_info}) in {result.get('elapsed_s')}s. "
                    f"Searches now run via approximate nearest-neighbor."
                ),
                "data": {"world_ingest_action": "build_index", **result},
            }
        except Exception as exc:
            return {
                "message": f"Index build failed: {exc}",
                "data": {"world_ingest_action": "build_index", "error": str(exc)},
            }

    return None
