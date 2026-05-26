"""Centralized vault chat surface.

Acts as a registered node (default node_id="vault") that:
  * accepts typed input → POSTs to vault `/ui`
  * polls vault's `/alerts/pending?node_id=vault` for messages the brain
    pushed proactively (notifications from background work, presence-
    triggered greetings) and prints them into the chat feed

The push channel is the standard `node_registry.queue_alert` /
`/alerts/pending` mechanism the rest of the node fleet already uses
(see `node/services/presence_client_service.py`). The vault's local
chat surface is just another node in that fleet — same primitive,
no new pubsub layer.

Run via the `luhkas-chat.service` systemd unit (persistent across
reboots, inside a tmux session). Attach with the `chat` command."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def _http_json(url: str, *, method: str = "GET", body: dict | None = None,
               timeout: float = 5.0) -> dict | None:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def register_with_vault(base_url: str, node_id: str) -> None:
    """Refresh this surface's entry in node_registry on startup. Even
    though `vault` is a static intrinsic node, registering re-asserts
    its current capabilities (display=True, no audio module yet)."""
    _http_json(
        f"{base_url}/node/register",
        method="POST",
        body={
            "node_id": node_id,
            "node_name": "Vault Chat Surface",
            "ip": "127.0.0.1",
            "services": {"presence": 0},  # this surface doesn't host services
            "display": {
                "has_display": True,
                "type": "vault_runtime",
                "can_show_code": True,
                "can_open_browser": False,
                "can_show_images": False,
            },
            "modules": {},  # no audio module yet → vault won't route TTS-only here
            "capabilities": {"surface": "tmux-cli"},
        },
        timeout=5,
    )


def post_ui(base_url: str, message: str, node_id: str) -> dict:
    payload = json.dumps({"message": message, "node_id": node_id}).encode()
    req = urllib.request.Request(
        f"{base_url}/ui",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def print_response(data: dict) -> None:
    response = data.get("response") or {}
    has_display = response.get("has_display", True)
    text = response.get("message") if has_display else (
        response.get("tts") or response.get("message")
    )
    if text:
        print(text)
        extra = response.get("display_content", "")
        if extra:
            print()
            print(extra)


def _format_alert(alert: dict) -> str:
    """Render one pushed alert for display in the chat feed."""
    if not isinstance(alert, dict):
        return f"[alert] {alert}"
    etype = alert.get("event_type") or alert.get("type") or "alert"
    msg = (alert.get("message") or "").strip()
    queued_at = alert.get("queued_at")
    age_note = ""
    if queued_at:
        try:
            age = int(time.time() - float(queued_at))
            if age > 60:
                if age >= 3600:
                    age_note = f" (queued {age // 3600}h{(age % 3600) // 60}m ago)"
                else:
                    age_note = f" (queued {age // 60}m ago)"
        except Exception:
            pass
    return f"[{etype}{age_note}] {msg}" if msg else f"[{etype}{age_note}]"


def _alert_subscriber(base_url: str, node_id: str, prompt: str,
                      poll_interval: float) -> None:
    """Background thread: poll /alerts/pending and print incoming pushes
    above the prompt. Each poll pops queued alerts atomically on the
    server side, so the same alert never appears twice."""
    while True:
        time.sleep(poll_interval)
        body = _http_json(f"{base_url}/alerts/pending?node_id={node_id}")
        if not body or not body.get("ok"):
            continue
        alerts = body.get("alerts") or []
        if not alerts:
            continue
        # Clear the current prompt line, print each alert, re-draw prompt.
        sys.stdout.write("\r\033[2K")
        for alert in alerts:
            sys.stdout.write(f"\n{_format_alert(alert)}\n")
        sys.stdout.write(f"\n{prompt}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(prog="ui_client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--node-id", default="vault",
                        help="Surface identity in the node registry. "
                             "Defaults to 'vault' (the static intrinsic "
                             "entry for the vault PC's local chat).")
    parser.add_argument("--no-push", dest="push", action="store_false",
                        help="Disable background /alerts/pending subscription.")
    parser.add_argument("--poll-interval", type=float, default=3.0,
                        help="Seconds between /alerts/pending polls "
                             "(default 3, matching node/services/"
                             "presence_client_service.py).")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
            health = json.loads(r.read())
        name = (health.get("identity") or {}).get("name") or "Vault"
        print(f"Connected to {name} at {base_url} as node_id={args.node_id!r}")
        push_note = " · push notifications on" if args.push else ""
        print(f"Type your message, or 'exit' to quit.{push_note}\n")
    except Exception as exc:
        print(f"Could not reach vault service at {base_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Refresh our registry entry so node_registry knows our current caps.
    try:
        register_with_vault(base_url, args.node_id)
    except Exception:
        pass  # non-fatal; the static `vault` default is already present

    if args.push:
        sub = threading.Thread(
            target=_alert_subscriber,
            args=(base_url, args.node_id, "> ", args.poll_interval),
            name="alert-subscriber",
            daemon=True,
        )
        sub.start()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            sys.exit(0)

        t0 = time.monotonic()
        try:
            data = post_ui(base_url, user_input, node_id=args.node_id)
        except urllib.error.URLError as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            continue
        elapsed = time.monotonic() - t0

        print_response(data)
        print(f"[{elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()
