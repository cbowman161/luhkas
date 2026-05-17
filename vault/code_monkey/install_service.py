"""Install code_monkey as a user-level systemd service.

Run from the LUHKAS-BRAIN project root:
    python3 -m code_monkey.install_service --workers 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

UNIT_NAME = "code-monkey.service"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 -m code_monkey.install_service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--working-directory", default=str(Path.home() / "brain_v2"))
    parser.add_argument("--no-enable", action="store_true", help="write the unit but do not enable/start it")
    args = parser.parse_args()

    user_dir = Path.home() / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    unit_path = user_dir / UNIT_NAME
    unit = f"""[Unit]
Description=LUHKAS code_monkey async coder service
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={args.working_directory}
ExecStart={args.python} -m code_monkey service --host {args.host} --port {args.port} --workers {args.workers}
Restart=always
RestartSec=5
TimeoutStartSec=120
TimeoutStopSec=300
Environment=PYTHONUNBUFFERED=1
Environment=BRAIN_BACKGROUND_KEEP_ALIVE=5m

[Install]
WantedBy=default.target
"""
    unit_path.write_text(unit)
    print(f"wrote {unit_path}")
    if args.no_enable:
        return 0
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", UNIT_NAME], check=True)
    print("installed and started code-monkey.service")
    print("check status with: systemctl --user status code-monkey.service")
    try:
        user = os.environ.get("USER")
        if user:
            subprocess.run(["loginctl", "enable-linger", user], check=False)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
