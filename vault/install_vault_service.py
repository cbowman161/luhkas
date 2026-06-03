"""Install the brain runtime HTTP service as a user-level systemd service."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


UNIT_NAME = "vault-runtime.service"


def main():
    parser = argparse.ArgumentParser(prog="python3 install_vault_service.py")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--working-directory", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--no-enable", action="store_true", help="write the unit but do not enable/start it")
    args = parser.parse_args()

    user_dir = Path.home() / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    unit_path = user_dir / UNIT_NAME

    unit = f"""[Unit]
Description=LUHKAS brain runtime orchestrator service
After=network-online.target ollama.service code-monkey.service
Wants=network-online.target
Requires=code-monkey.service

[Service]
Type=simple
WorkingDirectory={args.working_directory}
ExecStart={args.python} vault_service.py --host {args.host} --port {args.port}
Restart=always
RestartSec=5
TimeoutStartSec=600
TimeoutStopSec=60
Environment=PYTHONUNBUFFERED=1
Environment=CODE_MONKEY_URL=http://127.0.0.1:8765
Environment=VAULT_IMMEDIATE_KEEP_ALIVE=30m
Environment=VAULT_BACKGROUND_KEEP_ALIVE=5m
Environment=VAULT_WARM_MODEL_ROLES=router,chat

[Install]
WantedBy=default.target
"""

    unit_path.write_text(unit)
    print(f"wrote {unit_path}")

    # Install drop-ins from repo (e.g. wait-for-ollama). The repo dir mirrors
    # ~/.config/systemd/user/vault-runtime.service.d/. We sync rather than
    # symlink so the user's systemd doesn't need read access to the repo path.
    repo_dropin_src = Path(args.working_directory) / "systemd" / f"{UNIT_NAME}.d"
    dropin_dst = user_dir / f"{UNIT_NAME}.d"
    if repo_dropin_src.is_dir():
        dropin_dst.mkdir(parents=True, exist_ok=True)
        # Wipe any stale files we may have written previously, then copy fresh.
        for stale in dropin_dst.glob("*.conf"):
            stale.unlink()
        for src in sorted(repo_dropin_src.glob("*.conf")):
            shutil.copy(src, dropin_dst / src.name)
            print(f"wrote {dropin_dst / src.name}")

    if args.no_enable:
        return 0

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", UNIT_NAME], check=True)
    print("installed and started vault-runtime.service")
    print("check status with: systemctl --user status vault-runtime.service")

    try:
        user = os.environ.get("USER")
        if user:
            subprocess.run(["loginctl", "enable-linger", user], check=False)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
