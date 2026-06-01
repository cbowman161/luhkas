#!/usr/bin/env python3
"""GPIO button watcher for the RaspAudio MIC Ultra 3 (or compatible HAT).

Watches a configurable GPIO line on gpiochip0 for falling edges (button
press; the line is active-low with a pull-up bias) and POSTs to the
local audio_node ``/mute`` endpoint on each press. audio_node toggles
the output mute state and emits a ``mute_changed`` UI event so the
display switches into caption mode.

Env:
  AUDIO_BUTTON_GPIO      GPIO line offset on gpiochip0 (default 23 — the
                         middle button on the RaspAudio MIC Ultra 3,
                         confirmed by gpiomon discovery on kiosk).
  AUDIO_BUTTON_CHIP      chip path (default /dev/gpiochip0).
  AUDIO_MUTE_URL         audio_node /mute endpoint (default
                         http://127.0.0.1:5004/mute).
  AUDIO_BUTTON_DEBOUNCE_MS  ignore consecutive presses within this many
                         ms (default 250, since gpiomon discovery showed
                         ~180ms between falling and rising edges).

Runs forever. On signal exit cleanly. On unrecoverable error logs and
exits non-zero so systemd restarts us.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] button_watcher: %(message)s")
log = logging.getLogger("button_watcher")


BUTTON_GPIO = int(os.environ.get("AUDIO_BUTTON_GPIO", "23"))
CHIP_PATH = os.environ.get("AUDIO_BUTTON_CHIP", "/dev/gpiochip0")
MUTE_URL = os.environ.get("AUDIO_MUTE_URL", "http://127.0.0.1:5004/mute")
DEBOUNCE_MS = int(os.environ.get("AUDIO_BUTTON_DEBOUNCE_MS", "250"))


def _post_mute() -> None:
    """POST an empty body to /mute, which toggles the current state."""
    try:
        req = Request(
            MUTE_URL,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            log.info("toggled mute -> %s", parsed.get("muted"))
        except Exception:
            log.info("mute response: %s", body[:120])
    except (URLError, OSError, TimeoutError) as exc:
        log.warning("POST %s failed: %s", MUTE_URL, exc)


def _run_with_gpiod() -> int:
    """Monitor BUTTON_GPIO for falling edges and call _post_mute() per press."""
    try:
        import gpiod
    except ImportError as exc:
        log.error("python gpiod unavailable: %s — install python3-gpiod or run via system python", exc)
        return 2

    # gpiod v2 API (Pi OS Bookworm+, libgpiod ≥ 2.0). Older v1 ("gpiod.Chip(...).get_line(...)")
    # uses a totally different signature; detect by attribute presence.
    if hasattr(gpiod, "request_lines"):
        # v2 path. Pull-up bias is essential — RaspAudio buttons short
        # GPIO to ground when pressed.
        from gpiod.line import Bias, Direction, Edge
        request = gpiod.request_lines(
            CHIP_PATH,
            consumer="audio_button_watcher",
            config={
                BUTTON_GPIO: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    bias=Bias.PULL_UP,
                    edge_detection=Edge.FALLING,
                ),
            },
        )
        log.info(
            "watching %s line %d (debounce %dms) -> %s",
            CHIP_PATH, BUTTON_GPIO, DEBOUNCE_MS, MUTE_URL,
        )
        last_press_ns = 0
        debounce_ns = DEBOUNCE_MS * 1_000_000
        try:
            while True:
                # wait_edge_events blocks; chunked so SIGTERM can wake us.
                if not request.wait_edge_events(timeout=1.0):
                    continue
                for ev in request.read_edge_events():
                    now_ns = ev.timestamp_ns
                    if now_ns - last_press_ns < debounce_ns:
                        continue
                    last_press_ns = now_ns
                    log.info("button press (GPIO %d) at %s", BUTTON_GPIO, time.strftime("%H:%M:%S"))
                    _post_mute()
        finally:
            request.release()
        return 0

    # v1 fallback — older systems
    chip = gpiod.Chip(CHIP_PATH.replace("/dev/", ""))
    line = chip.get_line(BUTTON_GPIO)
    line.request(
        consumer="audio_button_watcher",
        type=gpiod.LINE_REQ_EV_FALLING_EDGE,
        flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP,
    )
    log.info("watching %s line %d (v1 API)", CHIP_PATH, BUTTON_GPIO)
    last_press = 0.0
    try:
        while True:
            if line.event_wait(sec=1):
                line.event_read()
                now = time.time()
                if (now - last_press) * 1000 < DEBOUNCE_MS:
                    continue
                last_press = now
                log.info("button press (GPIO %d)", BUTTON_GPIO)
                _post_mute()
    finally:
        line.release()
    return 0


def main() -> int:
    def _sigterm(_signum, _frame):
        log.info("shutdown requested, exiting")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    try:
        return _run_with_gpiod()
    except Exception as exc:
        log.exception("button watcher crashed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
