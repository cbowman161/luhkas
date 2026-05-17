from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TIMEOUT_SECONDS = 60


def run_command(command: str, cwd: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Run one local command inside a task workspace."""
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            'command': command,
            'status': 'success' if result.returncode == 0 else 'error',
            'returncode': result.returncode,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'error': result.stderr.strip() if result.returncode != 0 else '',
        }
    except subprocess.TimeoutExpired as exc:
        return {
            'command': command,
            'status': 'error',
            'returncode': -1,
            'stdout': (exc.stdout or '').strip() if isinstance(exc.stdout, str) else '',
            'stderr': 'Command timed out after {} seconds'.format(timeout),
            'error': 'Command timed out after {} seconds'.format(timeout),
        }
    except Exception as exc:
        return {
            'command': command,
            'status': 'error',
            'returncode': -1,
            'stdout': '',
            'stderr': str(exc),
            'error': str(exc),
        }


def run_verification_commands(
    root: Path,
    test_command: str,
    self_test_command: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the generic verification contract for a workspace.

    Step 7 intentionally validates behavior through the generated test suite
    only. The test suite must exercise the normal script/API behavior, create a
    test entry, verify it exists, call the script's delete/back-out path, and
    verify cleanup. The legacy self_test_command argument is accepted for API
    compatibility but is not executed.
    """
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    reset_native_data_dir(root)
    test_result = run_command(test_command, cwd=root)
    results.append(test_result)

    if test_result.get('status') != 'success':
        failures.append(test_result)

    return {
        'status': 'success' if not failures else 'error',
        'results': results,
        'failures': failures,
        'verification_mode': 'test_command_only_with_cleanup_contract',
    }


def reset_native_data_dir(root: Path) -> None:
    """Clear task-local native data before each verification attempt.

    Generated skills store persistent data under src/data during the build.
    Clearing that directory between verification attempts prevents failed prior
    attempts from contaminating later repaired attempts while still testing the
    exact native path the promoted skill will use relative to its own file.
    """
    data_dir = root / 'src' / 'data'
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
