#!/usr/bin/env python3
"""Live LUHKAS context/routing E2E battery.

Run from the vault or any host that can reach the Tailscale node URLs.
The script uses low-impact chat probes, health checks, snapshots, and
/learn_face validation errors. It does not enroll faces or move hardware.
"""
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


class LiveE2E:
    def __init__(self, args: argparse.Namespace) -> None:
        self.vault = args.vault.rstrip("/")
        self.kiosk_web = args.kiosk_web.rstrip("/")
        self.kiosk_vision = args.kiosk_vision.rstrip("/")
        self.kiosk_audio = args.kiosk_audio.rstrip("/")
        self.kiosk_display = args.kiosk_display.rstrip("/")
        self.scout_web = args.scout_web.rstrip("/")
        self.scout_vision = args.scout_vision.rstrip("/")
        self.scout_robot = args.scout_robot.rstrip("/")
        self.scout_battery = args.scout_battery.rstrip("/")
        self.results: list[Result] = []

    def get(self, label: str, url: str, timeout: float = 12.0, binary: bool = False) -> Any:
        return self._request(label, "GET", url, timeout=timeout, binary=binary)

    def post(self, label: str, url: str, payload: dict, timeout: float = 30.0, expect_status: int | None = None) -> Any:
        return self._request(label, "POST", url, payload=payload, timeout=timeout, expect_status=expect_status)

    def chat(self, label: str, base: str, message: str, timeout: float = 45.0, contains: str | None = None, exact: str | None = None) -> str:
        data = self.post(label, base + "/chat", {"message": message}, timeout=timeout)
        response = data.get("response") if isinstance(data, dict) else {}
        text = str((response or {}).get("message") or "")
        ok = True
        detail: dict[str, Any] = {"message": message, "response": text}
        if exact is not None and text.strip() != exact:
            ok = False
            detail["expected_exact"] = exact
        if contains is not None and contains.casefold() not in text.casefold():
            ok = False
            detail["expected_contains"] = contains
        if not ok:
            self.results.append(Result(label + " assertion", False, detail=detail))
        return text

    def _request(
        self,
        label: str,
        method: str,
        url: str,
        payload: dict | None = None,
        timeout: float = 12.0,
        expect_status: int | None = None,
        binary: bool = False,
    ) -> Any:
        attempts = 4 if expect_status is None else 1
        last_result: Result | None = None
        for attempt in range(attempts):
            data = None
            headers = {}
            if payload is not None:
                data = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read(4096 if binary else 65536)
                    elapsed = round(time.time() - started, 2)
                    detail: Any
                    if binary:
                        detail = {"bytes_sampled": len(body), "content_type": resp.headers.get("Content-Type")}
                    else:
                        text = body.decode("utf-8", "replace")
                        try:
                            detail = json.loads(text)
                        except json.JSONDecodeError:
                            detail = text[:300]
                    ok = expect_status is None or resp.status == expect_status
                    result = Result(label, ok, resp.status, elapsed, self._compact(detail))
                    self.results.append(result)
                    return detail
            except urllib.error.HTTPError as exc:
                body = exc.read(1024).decode("utf-8", "replace")
                elapsed = round(time.time() - started, 2)
                ok = expect_status is not None and exc.code == expect_status
                last_result = Result(label, ok, exc.code, elapsed, body[:500])
                if ok:
                    self.results.append(last_result)
                    return {}
                if exc.code != 503 or attempt == attempts - 1:
                    self.results.append(last_result)
                    return {}
            except Exception as exc:
                elapsed = round(time.time() - started, 2)
                last_result = Result(label, False, None, elapsed, repr(exc))
                if attempt == attempts - 1:
                    self.results.append(last_result)
                    return {}
            time.sleep(1.0 + attempt)
        if last_result is not None:
            self.results.append(last_result)
        return {}

    def _compact(self, detail: Any) -> Any:
        if not isinstance(detail, dict):
            return detail
        out = {
            key: detail.get(key)
            for key in ("ok", "service", "node_id", "message", "error", "history_size", "surface", "latest_user", "latest_assistant")
            if key in detail
        }
        response = detail.get("response")
        if isinstance(response, dict):
            out["response"] = {
                key: response.get(key)
                for key in ("ok", "message", "mode", "capability", "error")
                if key in response
            }
        meta = detail.get("meta")
        if isinstance(meta, dict):
            out["meta"] = {
                key: meta.get(key)
                for key in ("ok", "frame_id", "frame_shape", "target_id")
                if key in meta
            }
        return out or str(detail)[:500]

    def run(self) -> int:
        self.health_matrix()
        self.topic_stack_switching()
        self.followup_new_topic_separation()
        self.route_variety_with_context()
        self.print_summary()
        return 1 if any(not result.ok for result in self.results) else 0

    def health_matrix(self) -> None:
        for label, base in (
            ("vault", self.vault),
            ("kiosk_web", self.kiosk_web),
            ("kiosk_vision", self.kiosk_vision),
            ("kiosk_audio", self.kiosk_audio),
            ("kiosk_display", self.kiosk_display),
            ("scout_web", self.scout_web),
            ("scout_vision", self.scout_vision),
            ("scout_robot", self.scout_robot),
            ("scout_battery", self.scout_battery),
        ):
            self.get(label + " /health", base + "/health", timeout=10)

    def topic_stack_switching(self) -> None:
        self.chat("stack setup phrase", self.kiosk_web, "The test phrase is heliotrope canyon.", contains="heliotrope canyon")
        self.chat("stack setup marker", self.kiosk_web, "The marker word is viridian lantern.", contains="viridian lantern")
        self.chat("stack setup token", self.kiosk_web, "The token is stark architecture.", contains="stark architecture")
        self.chat("stack switch phrase", self.kiosk_web, "What was the test phrase?", contains="heliotrope canyon")
        self.chat("stack switch token", self.kiosk_web, "What was the token?", contains="stark architecture")
        self.chat("stack switch marker", self.kiosk_web, "What was the marker word?", contains="viridian lantern")

    def followup_new_topic_separation(self) -> None:
        self.chat("new topic arithmetic first", self.kiosk_web, "what is seven plus one? answer with just the number", exact="8")
        self.chat("local command between topics", self.kiosk_web, "status", timeout=20)
        self.chat("new topic arithmetic second", self.kiosk_web, "what is nine minus four? answer with just the number", exact="5")
        self.chat("return to earlier context", self.kiosk_web, "What was the token?", contains="stark architecture")

    def route_variety_with_context(self) -> None:
        self.chat("kiosk wakeword", self.kiosk_web, "luhkas", exact="Yes? What can I do for you?")
        self.chat("scout wakeword", self.scout_web, "luhkas", exact="Yes? What can I do for you?")
        self.chat("scout battery local", self.scout_web, "battery", contains="Battery is at", timeout=20)

        for node, base in (("kiosk", self.kiosk_vision), ("scout", self.scout_vision)):
            self.get(node + " /meta", base + "/meta", timeout=10)
            self.get(node + " /capabilities", base + "/capabilities", timeout=10)
            self.get(node + " /snapshot", base + "/snapshot", timeout=15, binary=True)
            self.post(node + " /learn_face missing name", base + "/learn_face", {}, timeout=10, expect_status=400)
            self.post(node + " /learn_face invalid face", base + "/learn_face?name=E2EValidation&face_id=-99999", {}, timeout=10, expect_status=409)

        self.post("kiosk display event", self.kiosk_display + "/ui/event", {"type": "e2e_probe", "source": "live_context_e2e", "message": "probe"}, timeout=10)
        self.get("kiosk display face state", self.kiosk_display + "/presence/face/state", timeout=10)

    def print_summary(self) -> None:
        serializable = [
            {
                "label": result.label,
                "ok": result.ok,
                "status": result.status,
                "seconds": result.seconds,
                "detail": result.detail,
            }
            for result in self.results
        ]
        print(json.dumps(serializable, indent=2, sort_keys=True))
        failed = [result.label for result in self.results if not result.ok]
        print("SUMMARY", json.dumps({"total": len(self.results), "failed": len(failed), "failed_labels": failed}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live LUHKAS context/routing E2E probes.")
    parser.add_argument("--vault", default="http://127.0.0.1:7000")
    parser.add_argument("--kiosk-web", default="http://100.81.45.83:5005")
    parser.add_argument("--kiosk-vision", default="http://100.81.45.83:5000")
    parser.add_argument("--kiosk-audio", default="http://100.81.45.83:5004")
    parser.add_argument("--kiosk-display", default="http://100.81.45.83:5006")
    parser.add_argument("--scout-web", default="http://100.112.87.59:5005")
    parser.add_argument("--scout-vision", default="http://100.112.87.59:5000")
    parser.add_argument("--scout-robot", default="http://100.112.87.59:5001")
    parser.add_argument("--scout-battery", default="http://100.112.87.59:5003")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(LiveE2E(parse_args()).run())
