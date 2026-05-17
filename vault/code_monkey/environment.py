import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict


def probe_command(command: str) -> Dict[str, Any]:
    exe = shutil.which(command)
    if not exe:
        return {'available': False, 'path': None, 'version': None}
    try:
        result = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=5)
        text = (result.stdout or result.stderr or '').strip().splitlines()
        return {'available': True, 'path': exe, 'version': text[0] if text else 'available'}
    except Exception as exc:
        return {'available': True, 'path': exe, 'version': f'probe failed: {exc}'}


def snapshot() -> Dict[str, Any]:
    return {
        'os': {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'cwd': os.getcwd(),
        },
        'python': {
            'executable': sys.executable,
            'version': sys.version.split()[0],
        },
        'commands': {name: probe_command(name) for name in ['python3', 'pip', 'git', 'ollama']},
        'constraints': [
            'Use local files only.',
            'Use Python standard library unless dependencies are declared later.',
            'Generated files must stay inside the task workspace.',
            'Tests must be non-interactive and deterministic.',
        ],
    }
