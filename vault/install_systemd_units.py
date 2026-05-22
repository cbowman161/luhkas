"""Install all LUHKAS vault systemd user units from vault/systemd/.

Copies every .service and .timer file from this directory's `systemd/`
subdirectory into ~/.config/systemd/user/, daemon-reloads, then enables
and starts each unit so they survive reboots and run on first install.

Units installed:
  - code-monkey.service     (port 8765, code_monkey coder service)
  - vault-runtime.service   (port 7000, brain orchestrator; requires code-monkey)
  - vault-autosync.service  (oneshot: rsync node/ to each registered node)
  - vault-autosync.timer    (fires vault-autosync every 60s)

Run from any working directory:  python3 install_systemd_units.py
Pass --no-enable to install the unit files without enabling/starting them.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


UNITS_DIR = Path(__file__).resolve().parent / "systemd"

# Order matters for the initial start: bring up the dependencies first.
ENABLE_ORDER = [
    "code-monkey.service",
    "vault-runtime.service",
    "vault-autosync.service",
    "vault-autosync.timer",
]


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 install_systemd_units.py")
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="copy the unit files but do not enable or start them",
    )
    args = parser.parse_args()

    target_dir = Path.home() / ".config" / "systemd" / "user"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not UNITS_DIR.is_dir():
        raise SystemExit(f"unit source directory missing: {UNITS_DIR}")

    copied: list[str] = []
    for source in sorted(UNITS_DIR.iterdir()):
        if source.suffix not in {".service", ".timer"}:
            continue
        destination = target_dir / source.name
        shutil.copyfile(source, destination)
        copied.append(source.name)
        print(f"wrote {destination}")

    if not copied:
        raise SystemExit(f"no .service or .timer files found in {UNITS_DIR}")

    if args.no_enable:
        print("--no-enable set; skipping daemon-reload and enable")
        return 0

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

    for name in ENABLE_ORDER:
        if name not in copied:
            continue
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", name],
            check=True,
        )
        print(f"enabled and started {name}")

    try:
        user = os.environ.get("USER")
        if user:
            subprocess.run(["loginctl", "enable-linger", user], check=False)
    except Exception:
        pass

    print("install complete; check status with:")
    print("  systemctl --user status vault-runtime.service code-monkey.service vault-autosync.timer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
