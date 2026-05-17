"""Install the brain runtime HTTP service as a user-level systemd service."""

import argparse
import os
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
Environment=VAULT_IMMEDIATE_KEEP_ALIVE=24h
Environment=VAULT_BACKGROUND_KEEP_ALIVE=5m
Environment=VAULT_WARM_MODEL_ROLES=router,chat,vision

[Install]
WantedBy=default.target
"""

    unit_path.write_text(unit)
    print(f"wrote {unit_path}")

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
