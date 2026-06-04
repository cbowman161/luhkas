#!/usr/bin/env python3
"""Live learned-capability route probes for kiosk and scout chat."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    label: str
    ok: bool
    status: int | None = None
    seconds: float = 0.0
    detail: Any = None


class LiveLearnedProbe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.kiosk_web = args.kiosk_web.rstrip("/")
        self.scout_web = args.scout_web.rstrip("/")
        self.results: list[Result] = []

    def run(self) -> int:
        for label, base, message in (
            ("kiosk cpu alias", self.kiosk_web, "how busy is the processor currently"),
            ("kiosk cpu alias 2", self.kiosk_web, "what is the CPU utilization right now"),
            ("kiosk disk root", self.kiosk_web, "what is the disk usage"),
            ("kiosk disk alias", self.kiosk_web, "root filesystem space"),
            ("scout memory root", self.scout_web, "ram free right now"),
            ("scout memory alias", self.scout_web, "swap usage"),
        ):
            if self.chat(label, base, message, contains="Learned command."):
                self.chat(label + " review ack", base, "yes", contains=None, require_learned=False)
        self.print_summary()
        return 1 if any(not result.ok for result in self.results) else 0

    def chat(
        self,
        label: str,
        base: str,
        message: str,
        contains: str | None,
        require_learned: bool = True,
    ) -> bool:
        started = time.time()
        detail: dict[str, Any] = {"message": message}
        status = None
        ok = False
        try:
            req = urllib.request.Request(
                base + "/chat",
                data=json.dumps({"message": message}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                status = resp.status
                body = resp.read(65536).decode("utf-8", "replace")
                payload = json.loads(body)
            response = payload.get("response") if isinstance(payload, dict) else {}
            text = str((response or {}).get("message") or "")
            source = str((response or {}).get("deterministic_source") or "")
            detail.update({"response": text, "deterministic_source": source})
            ok = True
            if contains is not None and contains.casefold() not in text.casefold():
                ok = False
            if require_learned and "learned_capability" not in source:
                ok = False
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail["error"] = exc.read(1024).decode("utf-8", "replace")
        except Exception as exc:
            detail["error"] = repr(exc)
        self.results.append(Result(label, ok, status, round(time.time() - started, 2), detail))
        return ok

    def print_summary(self) -> None:
        serializable = [result.__dict__ for result in self.results]
        print(json.dumps(serializable, indent=2, sort_keys=True))
        failed = [result.label for result in self.results if not result.ok]
        print("SUMMARY", json.dumps({"total": len(self.results), "failed": len(failed), "failed_labels": failed}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live learned-capability route probes.")
    parser.add_argument("--kiosk-web", default="http://100.81.45.83:5005")
    parser.add_argument("--scout-web", default="http://100.112.87.59:5005")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(LiveLearnedProbe(parse_args()).run())
