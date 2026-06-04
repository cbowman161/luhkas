#!/usr/bin/env python3
"""Live probe for learned execution review correction behavior.

The probe temporarily seeds two disposable learned capabilities into the
production store, exercises the chat route that corrects a just-run learned
command, verifies the phrase is repointed to the existing capability, and then
restores the original store bytes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "vault" / "data" / "learned_capabilities" / "capabilities.json"


@dataclass
class Result:
    label: str
    ok: bool
    status: int | None = None
    seconds: float = 0.0
    detail: Any = None


class LiveReviewCorrectionProbe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.base = args.web.rstrip("/")
        self.store = args.store
        self.run_id = str(int(time.time()))
        self.bad_phrase = f"e2e review wrong cpu {self.run_id}"
        self.good_phrase = f"e2e review disk usage {self.run_id}"
        self.results: list[Result] = []

    def run(self) -> int:
        original = self.store.read_bytes()
        try:
            self.seed_store()
            self.chat("run disposable wrong cap", self.bad_phrase, contains="Learned command.", require_source="learned_capability")
            self.chat(
                "correct to disposable disk cap",
                "no, disk usage",
                contains="Removed the wrong learned command and routed to the existing Vault disk usage",
                require_source="learned_capability",
            )
            self.assert_store_repointed()
        finally:
            self.store.write_bytes(original)
        self.run_audit()
        self.print_summary()
        return 1 if any(not result.ok for result in self.results) else 0

    def seed_store(self) -> None:
        data = json.loads(self.store.read_text(encoding="utf-8"))
        caps = data.setdefault("capabilities", {})
        now = time.time()
        caps[self.bad_phrase] = self.capability(
            self.bad_phrase,
            "vault_e2e_cpu_usage",
            "Vault CPU usage",
            "cpu",
            "usage",
            "printf 'CPU usage percent: 1.0\\n'",
            now,
        )
        caps[self.good_phrase] = self.capability(
            self.good_phrase,
            "vault_e2e_disk_usage",
            "Vault disk usage",
            "disk",
            "usage",
            "printf 'Filesystem Size Used Avail Use%% Mounted on\\n/e2e 1G 1M 999M 1%% /e2e\\n'",
            now,
            hits=9999,
        )
        self.store.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def capability(
        self,
        phrase: str,
        intent: str,
        description: str,
        topic: str,
        aspect: str,
        command: str,
        now: float,
        hits: int = 0,
    ) -> dict:
        return {
            "name": intent,
            "intent": intent,
            "description": description,
            "route": "learned_capability",
            "target": "vault",
            "confidence": 0.99,
            "reason": "live review correction probe",
            "inferred": {"topic": topic, "aspect": aspect},
            "input": phrase,
            "normalized_input": phrase,
            "confirmed": True,
            "created_at": now,
            "updated_at": now,
            "hits": hits,
            "examples": [],
            "execution": {
                "type": "bash",
                "command": command,
                "required_facts": [topic, aspect],
                "timeout_seconds": 10,
                "safety": {"allowed": True, "reason": "Probe read-only command."},
            },
            "code_monkey_task": {"ok": False, "skipped": True},
            "response": {"type": "compose_from_output", "required_facts": [topic, aspect]},
        }

    def chat(self, label: str, message: str, contains: str, require_source: str) -> None:
        started = time.time()
        status = None
        detail: dict[str, Any] = {"message": message}
        ok = False
        try:
            req = urllib.request.Request(
                self.base + "/chat",
                data=json.dumps({"message": message}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                status = resp.status
                payload = json.loads(resp.read(65536).decode("utf-8", "replace"))
            response = payload.get("response") if isinstance(payload, dict) else {}
            text = str((response or {}).get("message") or "")
            source = str((response or {}).get("deterministic_source") or "")
            detail.update({"response": text, "deterministic_source": source})
            ok = contains.casefold() in text.casefold() and require_source in source
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail["error"] = exc.read(1024).decode("utf-8", "replace")
        except Exception as exc:
            detail["error"] = repr(exc)
        self.results.append(Result(label, ok, status, round(time.time() - started, 2), detail))

    def assert_store_repointed(self) -> None:
        started = time.time()
        data = json.loads(self.store.read_text(encoding="utf-8"))
        caps = data.get("capabilities") or {}
        bad = caps.get(self.bad_phrase) or {}
        good = caps.get(self.good_phrase) or {}
        ok = (
            isinstance(bad, dict)
            and isinstance(good, dict)
            and bad.get("alias_of") == self.good_phrase
            and not good.get("alias_of")
            and (bad.get("inferred") or {}).get("topic") == "disk"
        )
        self.results.append(
            Result(
                "store repointed disposable alias",
                ok,
                seconds=round(time.time() - started, 2),
                detail={
                    "bad_alias_of": bad.get("alias_of"),
                    "bad_inferred": bad.get("inferred"),
                    "good_alias_of": good.get("alias_of"),
                },
            )
        )

    def run_audit(self) -> None:
        started = time.time()
        completed = subprocess.run(
            [sys.executable or "python3", "scripts/audit_learned_capabilities_store.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.results.append(
            Result(
                "post-restore learned store audit",
                completed.returncode == 0,
                seconds=round(time.time() - started, 2),
                detail={"returncode": completed.returncode, "tail": (completed.stdout or completed.stderr)[-500:]},
            )
        )

    def print_summary(self) -> None:
        serializable = [result.__dict__ for result in self.results]
        print(json.dumps(serializable, indent=2, sort_keys=True))
        failed = [result.label for result in self.results if not result.ok]
        print("SUMMARY", json.dumps({"total": len(self.results), "failed": len(failed), "failed_labels": failed}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live learned execution-review correction probe.")
    parser.add_argument("--web", default="http://100.81.45.83:5005")
    parser.add_argument("--store", type=Path, default=STORE)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(LiveReviewCorrectionProbe(parse_args()).run())
