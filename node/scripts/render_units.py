#!/usr/bin/env python3
"""Render per-node systemd unit files from shared templates + a profile.

Reads ``node/profiles/<NODE_ID>.json`` and writes ``<NODE_ID>-<svc>.service``
unit files to the destination directory (default: the user systemd unit
dir). Called by ``install_user_services.sh`` and ``bootstrap_node.sh`` so
the profile JSON is the only file you author per node.

Profile schema (both shorthand and verbose forms accepted):

    {
      "services": {
        "vision": 5000,                       // shorthand: port only
        "battery": {                          // verbose: port + env
          "port": 5003,
          "env": {"BATTERY_BACKEND": "max17040"}
        }
      },
      "extra_units": ["browser"]              // units without a network port
    }

Templates live at ``node/systemd/templates/<svc>.service.tmpl`` and may use
``{NODE_ID}``, ``{NODE_DIR}``, ``{VAULT_URL}`` placeholders. The token
``{EXTRA_ENV}`` is replaced by additional ``Environment=KEY=VALUE`` lines
derived from the profile's per-service ``env`` mapping (and is removed
along with its trailing newline when no overrides are present).

Service entries in the profile that have no matching template are
silently skipped — that's the case for ``presence`` rolled into another
unit name, for example.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_NODE_DIR = Path(__file__).resolve().parents[1]
if str(REPO_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_NODE_DIR))

from profile_loader import load_profile

TEMPLATE_DIR = REPO_NODE_DIR / "systemd" / "templates"
PROFILE_DIR = REPO_NODE_DIR / "profiles"
DEFAULT_DEST = Path.home() / ".config" / "systemd" / "user"
DEFAULT_VAULT_URL = "http://100.70.245.116:7000"  # Tailscale IP — see sync_manager._DEFAULT_VAULT_URL for why


def _service_spec(value) -> dict:
    """Normalize a profile services entry to {port, env, after, wants}."""
    if isinstance(value, dict):
        return {
            "port": value.get("port"),
            "env": dict(value.get("env") or {}),
            "after": list(value.get("after") or []),
            "wants": list(value.get("wants") or []),
        }
    if isinstance(value, int):
        return {"port": value, "env": {}, "after": [], "wants": []}
    if value is None:
        return {"port": None, "env": {}, "after": [], "wants": []}
    raise ValueError(f"unsupported service spec: {value!r}")


def _format_extra_env(env: dict) -> str:
    """Render env mapping as Environment= lines for unit file insertion."""
    if not env:
        return ""
    lines = [f"Environment={key}={value}" for key, value in env.items()]
    return "\n".join(lines)


def _format_extra_unit(after: list, wants: list) -> str:
    """Render additional After=/Wants= directives for the [Unit] section."""
    lines = []
    if after:
        lines.append(f"After={' '.join(after)}")
    if wants:
        lines.append(f"Wants={' '.join(wants)}")
    return "\n".join(lines)


def _substitute(
    template: str,
    *,
    node_id: str,
    node_dir: str,
    vault_url: str,
    extra_env: str,
    extra_unit: str,
) -> str:
    text = template
    text = text.replace("{NODE_ID}", node_id)
    text = text.replace("{NODE_DIR}", node_dir)
    text = text.replace("{VAULT_URL}", vault_url)
    # Drop empty placeholders together with their newline to avoid blank lines.
    for placeholder, value in (("{EXTRA_ENV}", extra_env), ("{EXTRA_UNIT}", extra_unit)):
        if value:
            text = text.replace(placeholder, value)
        else:
            text = text.replace(placeholder + "\n", "")
            text = text.replace(placeholder, "")
    return text


def render_profile(
    node_id: str,
    *,
    node_dir: str,
    vault_url: str,
    dest_dir: Path,
    template_dir: Path = TEMPLATE_DIR,
    profile_path: Path | None = None,
) -> list[Path]:
    profile = load_profile(
        profile_path if profile_path else node_id,
        profiles_dir=(profile_path.parent if profile_path else None),
    )

    services = profile.get("services") or {}
    extra_units = profile.get("extra_units") or []

    targets: list[tuple[str, dict]] = []
    for name, value in services.items():
        targets.append((name, _service_spec(value)))
    for name in extra_units:
        if isinstance(name, dict):
            targets.append((name.get("name", ""), _service_spec(name)))
        else:
            targets.append((name, {"port": None, "env": {}, "after": [], "wants": []}))

    dest_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for svc_name, spec in targets:
        tmpl_path = template_dir / f"{svc_name}.service.tmpl"
        if not tmpl_path.exists():
            print(f"[render] no template for service '{svc_name}' — skipping", file=sys.stderr)
            continue
        unit_text = _substitute(
            tmpl_path.read_text(),
            node_id=node_id,
            node_dir=node_dir,
            vault_url=vault_url,
            extra_env=_format_extra_env(spec["env"]),
            extra_unit=_format_extra_unit(spec["after"], spec["wants"]),
        )
        out_path = dest_dir / f"{node_id}-{svc_name}.service"
        out_path.write_text(unit_text)
        rendered.append(out_path)
        print(f"[render] {out_path}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description="Render LUHKAS node systemd units from a profile.")
    parser.add_argument("node_id", help="Node id (matches node/profiles/<id>.json)")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination unit directory")
    parser.add_argument("--node-dir", default=str(REPO_NODE_DIR), help="Path to the node runtime on the target host")
    parser.add_argument("--vault-url", default=os.environ.get("VAULT_CHAT_URL", DEFAULT_VAULT_URL))
    parser.add_argument("--profile", default=None, help="Override the profile JSON path")
    parser.add_argument("--template-dir", default=str(TEMPLATE_DIR))
    args = parser.parse_args()

    try:
        render_profile(
            args.node_id,
            node_dir=args.node_dir,
            vault_url=args.vault_url,
            dest_dir=Path(args.dest),
            template_dir=Path(args.template_dir),
            profile_path=Path(args.profile) if args.profile else None,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
