import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ENV_DIR = Path("environment")

PROFILE_FILES = [
    ENV_DIR / "hardware_profile.md",
    ENV_DIR / "os_software_profile.md",
    ENV_DIR / "command_profile.md",
]


SAFE_PROBE_COMMANDS = {
    "python": [sys.executable, "--version"],
    "python3": ["python3", "--version"],
    "uname": ["uname", "-a"],
    "lsb_release": ["lsb_release", "-a"],
    "nvidia_smi": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,temperature.gpu", "--format=csv,noheader"],
    "ollama": ["ollama", "--version"],
}


def read_profile_file(path: Path) -> str:
    if not path.exists():
        return f"{path}: missing"

    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"{path}: unreadable: {exc}"


def run_probe(name: str, command: list[str], timeout: int = 5) -> str:
    executable = command[0]

    if shutil.which(executable) is None:
        return f"{name}: not found"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return f"{name}: returncode={result.returncode}; stderr={stderr or 'none'}"

        if stderr:
            return f"{name}: stdout={stdout or 'none'}; stderr={stderr}"

        return f"{name}: {stdout or 'ok'}"

    except subprocess.TimeoutExpired:
        return f"{name}: timed out"

    except Exception as exc:
        return f"{name}: probe failed: {exc}"


def build_runtime_probe_summary() -> str:
    lines = [
        f"platform.system: {platform.system()}",
        f"platform.platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        f"cwd: {os.getcwd()}",
    ]

    for name, command in SAFE_PROBE_COMMANDS.items():
        lines.append(run_probe(name, command))

    return "\n".join(lines)


def build_coder_environment_context() -> str:
    profile_sections = []

    for path in PROFILE_FILES:
        profile_sections.append(f"## {path}\n{read_profile_file(path)}")

    runtime = build_runtime_probe_summary()

    return f"""
LOCAL ENVIRONMENT CONTEXT

Use this context when writing, testing, or repairing code.
Do not assume Windows paths, PowerShell, cmd.exe, or Windows-only commands unless explicitly requested.
Prefer Ubuntu/Linux-compatible Python and shell commands.
Prefer python3 for commands.
Do not assume a Linux sensor path exists unless probing or checking fallbacks.
If using hardware, sensors, GPU, camera, audio, or OS commands, code must handle missing tools/files gracefully.

# Static Profiles

{chr(10).join(profile_sections)}

# Runtime Probe Snapshot

{runtime}
""".strip()