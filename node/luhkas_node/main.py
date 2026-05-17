#!/usr/bin/env python3
"""Interactive CLI entry point for the LUHKAS node thin client."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from node import NodeRuntime


def main() -> None:
    runtime = NodeRuntime()
    runtime.start()

    health = runtime.health()
    status = "connected" if health["brain_reachable"] else "brain unreachable"
    print(f"[{runtime.node_id}] {status} — {runtime.brain_url}", flush=True)

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                sys.exit(0)
            t0 = time.monotonic()
            runtime.handle(user_input)
            print(f"[{time.monotonic() - t0:.1f}s]")
            print()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
