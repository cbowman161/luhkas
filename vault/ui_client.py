"""Interactive CLI that sends input to the running vault service via POST /ui."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_ui(base_url: str, message: str, node_id: str = "ui") -> dict:
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
    if has_display:
        print(response.get("message", ""))
        extra = response.get("display_content", "")
        if extra:
            print()
            print(extra)
    else:
        print(response.get("tts") or response.get("message", ""))


def main():
    parser = argparse.ArgumentParser(prog="ui_client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--node-id", default="scout")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
            health = json.loads(r.read())
        name = (health.get("identity") or {}).get("name") or "Vault"
        print(f"Connected to {name} at {base_url}")
        print("Type your message, or 'exit' to quit.\n")
    except Exception as exc:
        print(f"Could not reach vault service at {base_url}: {exc}", file=sys.stderr)
        sys.exit(1)

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
