#!/usr/bin/env python3
"""Read-only audit for the production learned-capabilities store."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from learned_capabilities import DEFAULT_STORE, LearnedCapabilityEngine, LearnedCapabilityStore
from safety_policy import SafetyPolicy


@dataclass
class Finding:
    severity: str
    label: str
    detail: dict[str, Any]


class StoreAuditor:
    def __init__(self, store_path: Path, *, stale_pending_days: float, duplicate_threshold: float) -> None:
        self.store_path = store_path
        self.stale_pending_seconds = stale_pending_days * 86400
        self.duplicate_threshold = duplicate_threshold
        self.safety = SafetyPolicy()
        self.findings: list[Finding] = []

    def run(self) -> int:
        data = self.load_store()
        caps = data.get("capabilities") if isinstance(data, dict) else None
        pending = data.get("pending_code_monkey") if isinstance(data, dict) else None
        if not isinstance(caps, dict):
            self.add("error", "capabilities map missing", {"path": str(self.store_path)})
            caps = {}
        if pending is not None and not isinstance(pending, dict):
            self.add("error", "pending_code_monkey map invalid", {"path": str(self.store_path)})
            pending = {}
        self.audit_capabilities(caps)
        self.audit_pending(pending or {})
        self.audit_duplicates()
        self.print_summary(caps, pending or {})
        return 1 if any(f.severity == "error" for f in self.findings) else 0

    def load_store(self) -> dict:
        if not self.store_path.exists():
            self.add("error", "store file missing", {"path": str(self.store_path)})
            return {}
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.add("error", "store JSON unreadable", {"path": str(self.store_path), "error": repr(exc)})
            return {}

    def audit_capabilities(self, caps: dict[str, Any]) -> None:
        keys = set(caps)
        for key, cap in sorted(caps.items()):
            if not isinstance(cap, dict):
                self.add("error", "capability entry invalid", {"key": key, "type": type(cap).__name__})
                continue
            normalized = cap.get("normalized_input")
            if normalized and normalized != key:
                self.add("warn", "normalized_input does not match key", {"key": key, "normalized_input": normalized})
            alias_of = cap.get("alias_of")
            if alias_of and alias_of not in keys:
                self.add("error", "alias target missing", {"key": key, "alias_of": alias_of})
            execution = cap.get("execution")
            if not isinstance(execution, dict):
                self.add("error", "execution missing", {"key": key})
                continue
            kind = execution.get("type")
            if kind == "bash":
                self.audit_bash(key, execution)
            elif kind == "python_script":
                self.audit_python_script(key, execution)
            elif kind == "code_monkey_pending":
                self.add("warn", "capability still points at pending Code Monkey execution", {"key": key})
            else:
                self.add("error", "unsupported execution type", {"key": key, "type": kind})

    def audit_bash(self, key: str, execution: dict) -> None:
        command = str(execution.get("command") or "").strip()
        if not command:
            self.add("error", "bash command empty", {"key": key})
            return
        safety = self.safety.validate_command(command)
        if not safety.get("allowed"):
            self.add("error", "unsafe bash command", {"key": key, "command": command, "reason": safety.get("reason")})
        try:
            shlex.split(command)
        except ValueError as exc:
            self.add("error", "bash command does not parse", {"key": key, "command": command, "error": str(exc)})

    def audit_python_script(self, key: str, execution: dict) -> None:
        path = str(execution.get("path") or "").strip()
        source = execution.get("source")
        if not path:
            self.add("error", "python_script path missing", {"key": key})
            return
        script_path = Path(path)
        if not script_path.exists():
            self.add("error", "python_script path missing on disk", {"key": key, "path": path})
        if source is not None and not isinstance(source, str):
            self.add("error", "python_script source is not text", {"key": key, "source_type": type(source).__name__})

    def audit_pending(self, pending: dict[str, Any]) -> None:
        now = time.time()
        for task_id, entry in sorted(pending.items()):
            if not isinstance(entry, dict):
                self.add("error", "pending entry invalid", {"task_id": task_id, "type": type(entry).__name__})
                continue
            entry_task_id = str(entry.get("task_id") or "")
            if entry_task_id and entry_task_id != task_id:
                self.add("warn", "pending task id mismatch", {"key": task_id, "task_id": entry_task_id})
            updated = float(entry.get("updated_at") or entry.get("created_at") or 0)
            if updated and now - updated > self.stale_pending_seconds:
                self.add(
                    "warn",
                    "stale pending Code Monkey entry",
                    {"task_id": task_id, "state": entry.get("state"), "age_days": round((now - updated) / 86400, 2)},
                )
            if not entry.get("input"):
                self.add("warn", "pending entry missing input", {"task_id": task_id})

    def audit_duplicates(self) -> None:
        engine = LearnedCapabilityEngine(LearnedCapabilityStore(self.store_path), code_monkey_client=None, model=None)
        for dup in engine.find_duplicate_caps(near_threshold=self.duplicate_threshold)[:20]:
            self.add(
                "warn",
                "duplicate or near-duplicate recipe",
                {
                    "primary_key": dup.get("primary_key"),
                    "dup_key": dup.get("dup_key"),
                    "kind": dup.get("kind"),
                    "score": round(float((dup.get("similarity") or {}).get("score") or 0), 3),
                    "reason": (dup.get("similarity") or {}).get("reason"),
                },
            )

    def add(self, severity: str, label: str, detail: dict[str, Any]) -> None:
        self.findings.append(Finding(severity, label, detail))

    def print_summary(self, caps: dict, pending: dict) -> None:
        counts = {
            "error": sum(1 for f in self.findings if f.severity == "error"),
            "warn": sum(1 for f in self.findings if f.severity == "warn"),
        }
        payload = {
            "ok": counts["error"] == 0,
            "store": str(self.store_path),
            "capabilities": len(caps),
            "pending_code_monkey": len(pending),
            "counts": counts,
            "findings": [f.__dict__ for f in self.findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "SUMMARY",
            json.dumps(
                {
                    "ok": payload["ok"],
                    "capabilities": len(caps),
                    "pending_code_monkey": len(pending),
                    "errors": counts["error"],
                    "warnings": counts["warn"],
                },
                sort_keys=True,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for learned-capabilities store health.")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--stale-pending-days", type=float, default=7.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(StoreAuditor(args.store, stale_pending_days=args.stale_pending_days, duplicate_threshold=args.duplicate_threshold).run())
