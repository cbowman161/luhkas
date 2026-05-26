#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_DIR = REPO_ROOT / "vault" / "secrets"
STATE_FILE = SECRETS_DIR / "tailscale_authkey_state.json"
ROTATE_SCRIPT = REPO_ROOT / "scripts" / "rotate_tailscale_authkey.sh"
DEFAULT_THRESHOLD_SECONDS = 24 * 60 * 60


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _load_expiry() -> datetime | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return _parse_time(str(data.get("expires") or ""))


def main() -> int:
    threshold = int(os.environ.get("TAILSCALE_AUTHKEY_ROTATE_BEFORE_SECONDS", str(DEFAULT_THRESHOLD_SECONDS)))
    now = datetime.now(timezone.utc)
    expires = _load_expiry()
    if expires is not None and (expires - now).total_seconds() > threshold:
        print(f"Tailscale auth key is still valid until {expires.isoformat()}; no rotation needed.")
        return 0

    reason = "missing expiry state" if expires is None else f"expires at {expires.isoformat()}"
    print(f"Rotating Tailscale auth key because {reason}.")
    result = subprocess.run([str(ROTATE_SCRIPT)], cwd=REPO_ROOT, text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
