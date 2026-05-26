#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_DIR = REPO_ROOT / "vault" / "secrets"
DEFAULT_OAUTH_FILE = SECRETS_DIR / "tailscale_oauth.env"
DEFAULT_AUTHKEY_FILE = SECRETS_DIR / "tailscale.authkey"
DEFAULT_STATE_FILE = SECRETS_DIR / "tailscale_authkey_state.json"
OAUTH_URL = "https://api.tailscale.com/api/v2/oauth/token"
API_BASE = "https://api.tailscale.com/api/v2"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise SystemExit(f"Missing OAuth credentials file: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = shlex.split(value, posix=True)[0] if value.strip() else ""
    return values


def _truthy(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _request_json(request: Request) -> dict:
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Tailscale API HTTP {exc.code}: {body}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Tailscale API request failed: {exc}") from exc


def _oauth_token(client_id: str, client_secret: str, tag: str) -> str:
    data = urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "auth_keys",
        "tags": tag,
    }).encode("utf-8")
    request = Request(
        OAUTH_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    payload = _request_json(request)
    token = str(payload.get("access_token") or "")
    if not token:
        raise SystemExit("Tailscale OAuth response did not include access_token")
    return token


def _create_auth_key(env: dict[str, str], access_token: str) -> dict:
    tag = env.get("TAILSCALE_AUTHKEY_TAG", "tag:luhkas-node")
    tailnet = env.get("TAILSCALE_TAILNET", "-")
    expiry = int(env.get("TAILSCALE_AUTHKEY_EXPIRY_SECONDS", "7776000"))
    description = env.get("TAILSCALE_AUTHKEY_DESCRIPTION", "luhkas-node-bootstrap")
    body = {
        "capabilities": {
            "devices": {
                "create": {
                    "reusable": _truthy(env.get("TAILSCALE_AUTHKEY_REUSABLE", ""), True),
                    "ephemeral": _truthy(env.get("TAILSCALE_AUTHKEY_EPHEMERAL", ""), False),
                    "preauthorized": _truthy(env.get("TAILSCALE_AUTHKEY_PREAUTHORIZED", ""), True),
                    "tags": [tag],
                }
            }
        },
        "expirySeconds": expiry,
        "description": description,
    }
    request = Request(
        f"{API_BASE}/tailnet/{tailnet}/keys",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    payload = _request_json(request)
    if not str(payload.get("key") or "").startswith("tskey-auth-"):
        raise SystemExit("Tailscale key response did not include an auth key")
    return payload


def main() -> int:
    oauth_file = Path(os.environ.get("TAILSCALE_OAUTH_FILE", DEFAULT_OAUTH_FILE))
    authkey_file = Path(os.environ.get("TAILSCALE_AUTHKEY_FILE", DEFAULT_AUTHKEY_FILE))
    state_file = Path(os.environ.get("TAILSCALE_AUTHKEY_STATE_FILE", DEFAULT_STATE_FILE))
    env = _load_env_file(oauth_file)
    client_id = env.get("TAILSCALE_OAUTH_CLIENT_ID", "")
    client_secret = env.get("TAILSCALE_OAUTH_CLIENT_SECRET", "")
    tag = env.get("TAILSCALE_AUTHKEY_TAG", "tag:luhkas-node")
    if not client_id or not client_secret:
        raise SystemExit(f"Missing client id/secret in {oauth_file}")

    token = _oauth_token(client_id, client_secret, tag)
    key_payload = _create_auth_key(env, token)

    authkey_file.parent.mkdir(parents=True, exist_ok=True)
    authkey_file.parent.chmod(0o700)
    tmp = authkey_file.with_suffix(".tmp")
    tmp.write_text(str(key_payload["key"]) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(authkey_file)

    state = {
        "id": key_payload.get("id"),
        "created": key_payload.get("created"),
        "expires": key_payload.get("expires"),
        "description": key_payload.get("description"),
        "capabilities": key_payload.get("capabilities"),
    }
    state_tmp = state_file.with_suffix(".tmp")
    state_tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state_tmp.chmod(0o600)
    state_tmp.replace(state_file)

    print(f"Saved new Tailscale auth key to {authkey_file}")
    print(f"Key id: {key_payload.get('id', '(unknown)')}")
    print(f"Expires: {key_payload.get('expires', '(unknown)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
