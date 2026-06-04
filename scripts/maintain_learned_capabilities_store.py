#!/usr/bin/env python3
"""Dry-run/apply maintenance for the learned-capabilities store."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))

from learned_capabilities import DEFAULT_STORE, LearnedCapabilityEngine, LearnedCapabilityStore


FINAL_PENDING_STATES = {"verified", "build_failed", "test_failed", "failed", "cancelled"}


@dataclass
class PlannedAction:
    action: str
    detail: dict[str, Any]


class StoreMaintainer:
    def __init__(self, store_path: Path, *, stale_pending_days: float, apply: bool) -> None:
        self.store_path = store_path
        self.stale_pending_seconds = stale_pending_days * 86400
        self.apply = apply
        self.actions: list[PlannedAction] = []

    def run(self) -> int:
        store = LearnedCapabilityStore(self.store_path)
        data = store.load()
        caps = data.get("capabilities") if isinstance(data, dict) else {}
        pending = data.get("pending_code_monkey") if isinstance(data, dict) else {}
        if not isinstance(caps, dict) or not isinstance(pending, dict):
            print(json.dumps({"ok": False, "error": "store shape is invalid", "store": str(self.store_path)}, indent=2))
            return 1
        self.plan_alias_repairs(caps)
        self.plan_duplicate_merges(caps)
        self.plan_stale_pending_removals(pending)
        backup = None
        applied: list[dict[str, Any]] = []
        if self.apply and self.actions:
            backup = self.backup_store()
            data = store.load()
            caps = data.setdefault("capabilities", {})
            alias_updates = []
            for planned in self.actions:
                detail = planned.detail
                if planned.action == "make_alias_root":
                    cap = caps.get(detail["key"])
                    if isinstance(cap, dict):
                        cap.pop("alias_of", None)
                        cap["updated_at"] = time.time()
                        caps[detail["key"]] = cap
                        alias_updates.append(detail)
                elif planned.action == "repoint_alias":
                    cap = caps.get(detail["key"])
                    if isinstance(cap, dict):
                        cap["alias_of"] = detail["root"]
                        cap["updated_at"] = time.time()
                        caps[detail["key"]] = cap
                        alias_updates.append(detail)
            if alias_updates:
                store.save(data)
                applied.append({"action": "alias_repair", "updates": alias_updates})
            engine = LearnedCapabilityEngine(store, code_monkey_client=None, model=None)
            for planned in self.actions:
                if planned.action == "merge_exact_duplicate":
                    detail = planned.detail
                    applied.append(
                        {
                            "action": planned.action,
                            "result": engine.merge_caps(detail["primary_key"], detail["dup_key"]),
                        }
                    )
            data = store.load()
            pending = data.setdefault("pending_code_monkey", {})
            removed = []
            for planned in self.actions:
                if planned.action == "remove_stale_pending" and pending.pop(planned.detail["task_id"], None) is not None:
                    removed.append(planned.detail["task_id"])
            if removed:
                store.save(data)
                applied.append({"action": "remove_stale_pending", "task_ids": removed})
        payload = {
            "ok": True,
            "mode": "apply" if self.apply else "dry_run",
            "store": str(self.store_path),
            "backup": str(backup) if backup else None,
            "planned": [action.__dict__ for action in self.actions],
            "applied": applied,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "SUMMARY",
            json.dumps(
                {
                    "mode": payload["mode"],
                    "planned": len(self.actions),
                    "applied": len(applied),
                    "backup": payload["backup"],
                },
                sort_keys=True,
            ),
        )
        return 0

    def plan_alias_repairs(self, caps: dict[str, Any]) -> None:
        planned_keys = set()
        for key, cap in sorted(caps.items()):
            if not isinstance(cap, dict) or not cap.get("alias_of"):
                continue
            root, cycle = self.resolve_alias_root(key, caps)
            if cycle:
                root_key = self.choose_alias_cycle_root(cycle, caps)
                if root_key not in planned_keys:
                    self.actions.append(
                        PlannedAction(
                            "make_alias_root",
                            {"key": root_key, "reason": "break_alias_cycle", "cycle": cycle},
                        )
                    )
                    planned_keys.add(root_key)
                for alias_key in cycle:
                    if alias_key != root_key and alias_key not in planned_keys:
                        self.actions.append(
                            PlannedAction(
                                "repoint_alias",
                                {"key": alias_key, "root": root_key, "reason": "break_alias_cycle", "cycle": cycle},
                            )
                        )
                        planned_keys.add(alias_key)
                continue
            alias_of = cap.get("alias_of")
            if root and root != alias_of and key not in planned_keys:
                self.actions.append(
                    PlannedAction(
                        "repoint_alias",
                        {"key": key, "root": root, "previous_alias_of": alias_of, "reason": "flatten_alias_chain"},
                    )
                )
                planned_keys.add(key)

    def resolve_alias_root(self, key: str, caps: dict[str, Any]) -> tuple[str | None, list[str] | None]:
        seen = []
        current = key
        while True:
            if current in seen:
                return None, seen[seen.index(current):] + [current]
            seen.append(current)
            cap = caps.get(current)
            if not isinstance(cap, dict):
                return current, None
            alias_of = cap.get("alias_of")
            if not alias_of:
                return current, None
            current = alias_of

    def choose_alias_cycle_root(self, cycle: list[str], caps: dict[str, Any]) -> str:
        unique = sorted(set(cycle))

        def score(key: str) -> tuple[int, float, int, str]:
            cap = caps.get(key) if isinstance(caps.get(key), dict) else {}
            return (-(int(cap.get("hits") or 0)), float(cap.get("created_at") or 0), len(key), key)

        return sorted(unique, key=score)[0]

    def plan_duplicate_merges(self, caps: dict[str, Any]) -> None:
        groups: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        for key, cap in caps.items():
            if not isinstance(cap, dict) or cap.get("alias_of"):
                continue
            execution = cap.get("execution") or {}
            kind = execution.get("type")
            signature = None
            if kind == "bash":
                command = str(execution.get("command") or "").strip()
                if command:
                    signature = ("bash", command)
            elif kind == "python_script":
                source = str(execution.get("source") or "").strip()
                path = str(execution.get("path") or "").strip()
                if source:
                    signature = ("python_source", source)
                elif path:
                    signature = ("python_path", path)
            if signature is not None:
                groups[signature].append((key, cap))
        for signature, entries in sorted(groups.items()):
            if len(entries) < 2:
                continue
            primary_key, primary = self.choose_primary(entries)
            for dup_key, dup in sorted(entries, key=lambda item: item[0]):
                if dup_key == primary_key:
                    continue
                self.actions.append(
                    PlannedAction(
                        "merge_exact_duplicate",
                        {
                            "primary_key": primary_key,
                            "dup_key": dup_key,
                            "primary_description": primary.get("description"),
                            "dup_description": dup.get("description"),
                            "recipe_type": signature[0],
                        },
                    )
                )

    def choose_primary(self, entries: list[tuple[str, dict]]) -> tuple[str, dict]:
        def score(item: tuple[str, dict]) -> tuple[int, int, float, str]:
            key, cap = item
            hits = int(cap.get("hits") or 0)
            alias_count = sum(1 for other_key, other_cap in entries if other_key != key and other_cap.get("alias_of") == key)
            created = float(cap.get("created_at") or 0)
            return (-hits, -alias_count, created, key)

        return sorted(entries, key=score)[0]

    def plan_stale_pending_removals(self, pending: dict[str, Any]) -> None:
        now = time.time()
        for task_id, entry in sorted(pending.items()):
            if not isinstance(entry, dict):
                continue
            state = str(entry.get("state") or "")
            updated = float(entry.get("updated_at") or entry.get("completed_at") or entry.get("created_at") or 0)
            if state not in FINAL_PENDING_STATES:
                continue
            if not updated or now - updated < self.stale_pending_seconds:
                continue
            self.actions.append(
                PlannedAction(
                    "remove_stale_pending",
                    {
                        "task_id": task_id,
                        "state": state,
                        "age_days": round((now - updated) / 86400, 2),
                        "input": entry.get("input"),
                    },
                )
            )

    def backup_store(self) -> Path:
        backup = self.store_path.with_suffix(f".maintain-{int(time.time())}.bak")
        shutil.copy2(self.store_path, backup)
        return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply learned-capabilities store maintenance.")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--stale-pending-days", type=float, default=7.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply planned maintenance after creating a backup.")
    mode.add_argument("--dry-run", action="store_true", help="Show planned maintenance without changing the store.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(StoreMaintainer(args.store, stale_pending_days=args.stale_pending_days, apply=args.apply).run())
