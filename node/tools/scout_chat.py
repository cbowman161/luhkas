#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI chat client for the unified vault PC presence service.")
    parser.add_argument(
        "--url",
        default=os.environ.get("VAULT_CHAT_URL", "http://127.0.0.1:7000"),
        help="Brain chat service URL. Defaults to VAULT_CHAT_URL or http://127.0.0.1:7000.",
    )
    parser.add_argument("--show-actions", action="store_true", help="Print planned/executed actions after each response.")
    parser.add_argument("--show-route", action="store_true", help="Print the brain intent route after each response.")
    parser.add_argument("--source", default=os.environ.get("VAULT_CHAT_SOURCE", "rover_cli"), help="Edge source name sent to the unified vault presence endpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")
    print("Brain presence chat. Type /help for commands, /quit to exit.")
    show_status(base_url)

    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            return
        if message == "/help":
            print("/status  show active identity and recent session state")
            print("/who     show who the rover currently thinks is present")
            print("/whoami  ask the brain who it thinks you are")
            print("/debug   show identity debug details")
            print("/caps    show action capabilities")
            print("/quit    exit")
            continue
        if message == "/status":
            show_status(base_url)
            continue
        if message == "/who":
            show_who(base_url)
            continue
        if message == "/whoami":
            print_json(get_json(base_url + "/whoami"))
            continue
        if message == "/debug":
            print_json(get_json(base_url + "/debug/identity"))
            continue
        if message == "/caps":
            print_json(get_json(base_url + "/capabilities"))
            continue

        result = post_json_with_wait_hint(base_url + "/presence/message", {"message": message, "source": args.source}, message)
        if not result:
            print("No response from brain chat service.")
            continue
        if args.show_route:
            route = result.get("route") or {}
            label = route.get("route") or "unknown"
            confidence = route.get("confidence")
            attempts = route.get("attempts")
            suffix = f" ({confidence:.2f})" if isinstance(confidence, (int, float)) else ""
            attempt_suffix = f", attempts: {attempts}" if attempts else ""
            print(f"[route: {label}{suffix}{attempt_suffix}]")
            self_route = route.get("self_route") or {}
            if isinstance(self_route, dict) and self_route:
                self_label = self_route.get("route") or "unknown"
                self_confidence = self_route.get("confidence")
                self_attempts = self_route.get("attempts")
                self_suffix = f" ({self_confidence:.2f})" if isinstance(self_confidence, (int, float)) else ""
                self_attempt_suffix = f", attempts: {self_attempts}" if self_attempts else ""
                print(f"[self-route: {self_label}{self_suffix}{self_attempt_suffix}]")
                if not self_route.get("ok"):
                    reason = self_route.get("error")
                    if reason:
                        print(f"[self-route error: {reason}]")
            if not route.get("ok"):
                reason = route.get("error")
                if reason:
                    print(f"[route error: {reason}]")
                raw_attempts = route.get("raw_attempts") or []
                for index, raw in enumerate(raw_attempts, start=1):
                    print(f"[route raw {index}: {raw}]")
        print(result.get("response") or result.get("answer") or "")
        if args.show_actions:
            print_json(result.get("actions", []))


def show_status(base_url: str) -> None:
    session = get_json(base_url + "/session")
    if not session:
        print("Brain chat service is not reachable.")
        return
    active = session.get("active_identity") or session.get("identity") or "unknown"
    print(f"active identity: {active}")


def show_who(base_url: str) -> None:
    state = get_json(base_url + "/scout/state") or get_json(base_url + "/session") or {}
    active = state.get("active_identity") or state.get("identity") or "unknown"
    print(f"active identity: {active}")
    target = state.get("target")
    if target:
        print("target:")
        print_json(target)
    memory = state.get("tracking_memory") or state.get("object_memory") or []
    if memory:
        print("tracking memory:")
        print_json(memory)
    else:
        print("tracking memory: none")


def get_json(url: str):
    try:
        with urlopen(url, timeout=4.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return None


def post_json(url: str, payload: dict):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return None


def post_json_with_wait_hint(url: str, payload: dict, message: str):
    done = threading.Event()
    result = {"value": None}

    def worker():
        try:
            result["value"] = post_json(url, payload)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    if not done.wait(0.5):
        print(random.choice(wait_hints(message)))
    thread.join()
    return result["value"]


def wait_hints(message: str):
    if is_question(message):
        return ["Let me think...", "Hmm, let me think...", "One second, thinking..."]
    return ["Hmmm...", "Well...", "One second..."]


def is_question(message: str):
    lowered = message.lower().strip()
    question_starters = (
        "who ",
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "would ",
        "should ",
        "is ",
        "are ",
    )
    return lowered.endswith("?") or lowered.startswith(question_starters)


def print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
