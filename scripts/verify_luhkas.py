#!/usr/bin/env python3
"""Run the LUHKAS verification battery from one entrypoint."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    label: str
    command: list[str]
    seconds: float = 0.0
    returncode: int | None = None


class Verifier:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = sys.executable or "python3"
        self.checks: list[Check] = []

    def run(self) -> int:
        self.run_check("compile", [self.python, "-m", "compileall", "-q", "vault", "tests", "scripts"])
        self.run_check("learned capability unit tests", [self.python, "-m", "unittest", "tests/learned_capabilities_test.py"])
        self.run_check("isolated learned capabilities battery", [self.python, "scripts/learned_capabilities_battery.py"])
        self.run_check("learned store maintenance dry-run", [self.python, "scripts/maintain_learned_capabilities_store.py", "--dry-run"])
        self.run_check("production learned store audit", [self.python, "scripts/audit_learned_capabilities_store.py"])
        if self.args.live:
            self.run_check("live learned capability probes", [self.python, "scripts/live_learned_capabilities_probe.py"])
            self.run_check("live context E2E", [self.python, "scripts/live_context_e2e.py"])
            self.run_check("post-live learned store maintenance dry-run", [self.python, "scripts/maintain_learned_capabilities_store.py", "--dry-run"])
            self.run_check("post-live production learned store audit", [self.python, "scripts/audit_learned_capabilities_store.py"])
        self.print_summary()
        return 1 if any(check.returncode for check in self.checks) else 0

    def run_check(self, label: str, command: list[str]) -> None:
        check = Check(label, command)
        self.checks.append(check)
        print(f"\n== {label} ==", flush=True)
        print(" ".join(command), flush=True)
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
        check.seconds = round(time.time() - started, 2)
        check.returncode = completed.returncode
        if completed.returncode != 0 and self.args.stop_on_failure:
            self.print_summary()
            raise SystemExit(1)

    def print_summary(self) -> None:
        payload = [
            {
                "label": check.label,
                "command": check.command,
                "seconds": check.seconds,
                "returncode": check.returncode,
                "ok": check.returncode == 0,
            }
            for check in self.checks
        ]
        failed = [check.label for check in self.checks if check.returncode]
        print("\nVERIFY_SUMMARY", json.dumps({"total": len(self.checks), "failed": len(failed), "failed_labels": failed}, sort_keys=True))
        print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LUHKAS compile, learned-capability, audit, and optional live checks.")
    parser.add_argument("--live", action="store_true", help="Also run scripts/live_context_e2e.py against reachable nodes.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop at the first failing check.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(Verifier(parse_args()).run())
