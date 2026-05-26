"""Drive a fresh edge node from empty Pi OS to fully-running LUHKAS node.

Called from ``/node/register`` in the vault HTTP service when a node first
appears, and also exposed as a CLI for manual re-orchestration:

    python3 -m vault.node_orchestrator <node_id> <lan_ip>

The orchestrator runs over SSH against ``<user>@<lan_ip>``. Vault's deploy
key (``~/.ssh/id_ed25519_nodes``) must be in the node's authorized_keys —
this happens automatically because ``prep_node_sd.sh`` injects the vault's
public key into Pi Imager's cloud-init ``ssh_authorized_keys`` list.

Idempotent: rerunning on an already-orchestrated node is safe. apt installs
skip already-present packages, ``git clone`` becomes ``git pull``, install
scripts are themselves idempotent, render_units.py overwrites the same
unit files.

Flow per ``node/profiles/<node_id>.json``:

  1. SSH-check the host (sanity)
  2. Push the repo (rsync — same logic as ``push_node``)
  3. apt baseline (git, python3-pip, ...)
  4. For each module in profile.modules: run ``node/<module>/install.sh``
  5. Push Tailscale auth-key and run ``setup_tailscale.sh`` (joins tailnet)
  6. Render systemd units via ``install_user_services.sh`` (renders +
     enables + starts always-on services)
  7. Wait for re-registration over tailnet (best-effort)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NODE_DIR = _REPO_ROOT / "node"
_SSH_KEY = Path.home() / ".ssh" / "id_ed25519_nodes"
_SSH_OPTS = [
    "-i", str(_SSH_KEY),
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]

# Use the canonical profile loader.
if str(_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(_NODE_DIR))
from profile_loader import load_profile  # noqa: E402


_state: dict[str, dict] = {}  # node_id -> latest orchestration result
_inflight: set[str] = set()


def status() -> dict:
    return {"orchestrations": dict(_state), "inflight": sorted(_inflight)}


def orchestrate(
    node_id: str,
    host: str,
    *,
    user: str = "luhkas",
    node_dir: str = "luhkas/node",
    log: Optional[list[str]] = None,
) -> dict:
    """Run the full first-time setup for *node_id* at *host* over SSH."""
    record: list[str] = log if log is not None else []
    started = time.time()

    def step(name: str) -> None:
        record.append(f"[{time.strftime('%H:%M:%S')}] >> {name}")
        print(f"[orchestrator] {node_id}@{host}: {name}", flush=True)

    def report(ok: bool, error: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        result = {
            "ok": ok,
            "node_id": node_id,
            "host": host,
            "elapsed_s": round(time.time() - started, 1),
            "log": record,
        }
        if error:
            result["error"] = error
        if extra:
            result.update(extra)
        _state[node_id] = result
        return result

    # ── 0. profile sanity ─────────────────────────────────────────────────
    try:
        profile = load_profile(node_id)
    except Exception as exc:
        return report(False, f"profile load failed: {exc}")

    modules = list(profile.get("modules") or [])
    if not modules:
        return report(False, "profile has no modules")
    step(f"profile loaded: modules={modules}")

    # ── 1. SSH reachability ───────────────────────────────────────────────
    step("ssh reachability check")
    if not _ssh_run(host, user, "echo ok", record, timeout=15).get("ok"):
        return report(False, f"ssh to {user}@{host} failed (is vault's pubkey in authorized_keys?)")

    # ── 2. baseline apt deps ──────────────────────────────────────────────
    step("apt baseline (git, python3-pip, ...)")
    baseline_cmd = (
        "set -euo pipefail; "
        "for i in $(seq 1 60); do "
        "  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 3; "
        "done; "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get update -y; "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "git curl ca-certificates python3 python3-pip python3-venv rsync"
    )
    if not _ssh_run(host, user, baseline_cmd, record, timeout=600).get("ok"):
        return report(False, "baseline apt install failed")

    # ── 3. rsync the repo onto the node ───────────────────────────────────
    step(f"rsync repo to {user}@{host}:{node_dir}")
    rsync_result = _rsync_node(host, user, node_dir, record)
    if not rsync_result.get("ok"):
        return report(False, f"rsync failed: {rsync_result.get('error')}")

    # ── 4. per-module install.sh scripts (NODE_USER=user, NODE_ID=node_id) ─
    for mod in modules:
        install_path_remote = f"~/{node_dir}/{mod}/install.sh"
        check = _ssh_run(host, user, f"test -x {install_path_remote}", record, timeout=10)
        if not check.get("ok"):
            step(f"(no install.sh for {mod})")
            continue
        step(f"running {mod}/install.sh on {host}")
        install_cmd = (
            "set -euo pipefail; "
            f"sudo NODE_USER={user!r} NODE_ID={node_id!r} bash {install_path_remote}"
        )
        result = _ssh_run(host, user, install_cmd, record, timeout=1800)
        if not result.get("ok"):
            record.append(f"  WARN: {mod}/install.sh exited non-zero: {result.get('error', '')[:200]}")

    # ── 5. Tailscale join (push key + run setup_tailscale.sh) ─────────────
    step("provisioning Tailscale (push key + setup_tailscale.sh)")
    try:
        from sync_manager import provision_tailscale_for_node
        ts = provision_tailscale_for_node(
            node_id=node_id, host=host, user=user, node_dir=node_dir,
        )
        record.append(f"  tailscale: {'ok' if ts.get('ok') else ts.get('error')}")
        if not ts.get("ok"):
            record.append(f"  WARN: tailscale provisioning failed; node may stay on LAN")
    except Exception as exc:
        record.append(f"  WARN: tailscale provisioning errored: {exc}")

    # ── 6. install + start user systemd services ──────────────────────────
    step("install_user_services.sh (render units + enable + start)")
    svc_cmd = (
        "set -euo pipefail; "
        f"bash ~/{node_dir}/scripts/install_user_services.sh"
    )
    result = _ssh_run(host, user, svc_cmd, record, timeout=600)
    if not result.get("ok"):
        return report(False, f"install_user_services.sh failed: {result.get('error')}")

    # ── 7. (best-effort) wait for re-registration over tailnet ────────────
    # The presence_service should come up and re-register with its tailnet
    # IP within ~30s. We don't block on this — the orchestrator returns OK
    # once installation is complete; tailnet visibility is a follow-up.

    return report(True, extra={"modules": modules})


# ──────────────────────────────────────────────────────────────────────────
# helpers


def _ssh_run(host: str, user: str, command: str, record: list[str], *, timeout: int = 60) -> dict:
    try:
        result = subprocess.run(
            ["ssh"] + _SSH_OPTS + [f"{user}@{host}", command],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        record.append(f"  ssh timeout after {timeout}s")
        return {"ok": False, "error": f"timeout after {timeout}s"}
    if result.returncode != 0:
        snippet = (result.stderr or result.stdout).strip().splitlines()[-3:]
        record.append("  ssh err: " + " | ".join(snippet))
        return {"ok": False, "error": "\n".join(snippet), "rc": result.returncode}
    return {"ok": True, "stdout": result.stdout.strip()}


def _rsync_node(host: str, user: str, node_dir: str, record: list[str]) -> dict:
    # First make sure the parent dir exists on the target.
    parent = node_dir.rsplit("/", 1)[0] if "/" in node_dir else "."
    if parent:
        mk = _ssh_run(host, user, f"mkdir -p ~/{parent}", record, timeout=15)
        if not mk.get("ok"):
            return mk
    dest = f"{user}@{host}:~/{node_dir}/"
    try:
        result = subprocess.run(
            [
                "rsync", "-a", "--delete",
                "--exclude=__pycache__/",
                "--exclude=*.pyc",
                "--exclude=._*",
                "--exclude=.DS_Store",
                "--exclude=*.log",
                "--exclude=captures/",
                "--exclude=config/faces/",
                "--exclude=config/people/",
                "--exclude=config/vault_faces/",
                "--exclude=config/unknown_faces/",
                "--exclude=config/brain_faces/",
                "--exclude=config/telemetry.db",
                "--exclude=config/telemetry.db-shm",
                "--exclude=config/telemetry.db-wal",
                "--exclude=luhkas_node/data/",
                "-e", "ssh " + " ".join(_SSH_OPTS),
                str(_NODE_DIR) + "/",
                dest,
            ],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "rsync timeout"}
    if result.returncode != 0:
        return {"ok": False, "error": (result.stderr or result.stdout).strip()}
    return {"ok": True}


def orchestrate_async(node_id: str, host: str, **kwargs) -> None:
    """Run orchestration on a background thread; safe to call from a handler."""
    import threading
    if node_id in _inflight:
        print(f"[orchestrator] {node_id} already in flight; skipping", flush=True)
        return

    def _run() -> None:
        _inflight.add(node_id)
        try:
            result = orchestrate(node_id, host, **kwargs)
            outcome = "ok" if result.get("ok") else f"failed: {result.get('error')}"
            print(f"[orchestrator] {node_id}@{host} done: {outcome}", flush=True)
        finally:
            _inflight.discard(node_id)

    threading.Thread(target=_run, daemon=True, name=f"orchestrate-{node_id}").start()


# ──────────────────────────────────────────────────────────────────────────
# CLI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("node_id", help="Node id (matches node/profiles/<id>.json)")
    parser.add_argument("host", help="LAN IP or tailnet hostname of the node")
    parser.add_argument("--user", default="luhkas")
    parser.add_argument("--node-dir", default="luhkas/node")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human-readable log")
    args = parser.parse_args()

    result = orchestrate(args.node_id, args.host, user=args.user, node_dir=args.node_dir)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print(f"=== orchestration result: {args.node_id}@{args.host} ===")
        print(f"ok           : {result['ok']}")
        print(f"elapsed      : {result['elapsed_s']}s")
        if result.get("error"):
            print(f"error        : {result['error']}")
        if result.get("modules"):
            print(f"modules      : {', '.join(result['modules'])}")
        print()
        print("log:")
        for line in result.get("log", []):
            print(f"  {line}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
