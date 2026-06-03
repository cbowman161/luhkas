import os
import tempfile
import subprocess

from path_policy import PathPolicy


BLOCKED_COMMANDS = [
    "rm -rf /",
    "shutdown",
    "reboot",
    ":(){ :|:& };:",
]

path_policy = PathPolicy()


def execute_action(action):
    action_type = action.get("type")
    content = action.get("content", "")

    if action_type == "write_file":
        return write_file(action)

    if action_type == "patch_file":
        return patch_file(action)

    if action_type == "read_file":
        return read_file(action)

    if action_type == "list_files":
        return list_files(action)

    if action_type == "command":
        return safe_execute_command(content)

    if action_type == "final":
        return {
            "status": "success",
            "stdout": content,
            "stderr": "",
            "error": None,
            "returncode": 0,
            "terminal_action": "final",
        }

    if action_type == "fail":
        return {
            "status": "error",
            "stdout": "",
            "stderr": content,
            "error": content,
            "returncode": -1,
            "terminal_action": "fail",
        }

    return error_result(f"Unknown action type: {action_type}")


def write_file(action):
    path = action.get("path")
    content = action.get("content", "")

    allowed = path_policy.can_write_file(path)

    if not allowed["allowed"]:
        return error_result(allowed["reason"])

    full_path = allowed["path"]

    try:
        directory = os.path.dirname(full_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "status": "success",
            "stdout": f"File written: {path}",
            "stderr": "",
            "error": None,
            "returncode": 0,
        }

    except Exception as e:
        return error_result(str(e))


def patch_file(action):
    path = action.get("path")
    content = action.get("content", "")

    allowed = path_policy.can_write_file(path)

    if not allowed["allowed"]:
        return error_result(allowed["reason"])

    full_path = allowed["path"]

    try:
        if not os.path.exists(full_path):
            return error_result(f"Cannot patch missing file: {path}")

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "status": "success",
            "stdout": f"File patched: {path}",
            "stderr": "",
            "error": None,
            "returncode": 0,
        }

    except Exception as e:
        return error_result(str(e))


def read_file(action):
    path = action.get("path")

    allowed = path_policy.can_read_file(path)

    if not allowed["allowed"]:
        return error_result(allowed["reason"])

    full_path = allowed["path"]

    try:
        if not os.path.exists(full_path):
            return error_result(f"File not found: {path}")

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "status": "success",
            "stdout": content,
            "stderr": "",
            "error": None,
            "returncode": 0,
        }

    except Exception as e:
        return error_result(str(e))


def list_files(action):
    root = action.get("path") or "."

    allowed = path_policy.can_read_file(root)

    if not allowed["allowed"]:
        return error_result(allowed["reason"])

    full_root = allowed["path"]

    try:
        if not os.path.exists(full_root):
            return error_result(f"Path not found: {root}")

        files = []

        for current_root, dirs, filenames in os.walk(full_root):
            dirs[:] = [
                d for d in dirs
                if d not in {
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                }
            ]

            for filename in filenames:
                absolute = os.path.join(current_root, filename)
                files.append(os.path.relpath(absolute, path_policy.root))

        return {
            "status": "success",
            "stdout": "\n".join(sorted(files)),
            "stderr": "",
            "error": None,
            "returncode": 0,
        }

    except Exception as e:
        return error_result(str(e))


def execute_code(code: str):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return format_completed_process(result)

    except subprocess.TimeoutExpired:
        return error_result("Code execution timed out")

    except Exception as e:
        return error_result(str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                # Best-effort cleanup. The tempfile is in /tmp so it
                # eventually gets cleaned up by the OS; logging would
                # spam if a parallel process already removed it.
                pass


def safe_execute_command(command: str):
    command = (command or "").strip()

    if not command:
        return error_result("Empty command")

    blocked = validate_shell_command(command)

    if blocked:
        return error_result(blocked)

    result = run_shell_command(command)
    return format_shell_result(result)


def validate_shell_command(command):
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return f"Blocked dangerous command: {command}"

    return None


def run_shell_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out",
            "returncode": -1,
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def format_completed_process(result):
    return {
        "status": "success" if result.returncode == 0 else "error",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": result.stderr.strip() or None,
        "returncode": result.returncode,
    }


def format_shell_result(result):
    return {
        "status": "success" if result["returncode"] == 0 else "error",
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
        "error": result["stderr"].strip() or None,
        "returncode": result["returncode"],
    }


def error_result(message: str):
    return {
        "status": "error",
        "stdout": "",
        "stderr": message,
        "error": message,
        "returncode": -1,
    }
