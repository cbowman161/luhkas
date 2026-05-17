"""
BackgroundManager — manages long-running background.py processes for capabilities.

Each installed capability that ships a background.py gets one supervised
subprocess. Crashed processes are restarted automatically every 15 seconds.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path


class BackgroundManager:
    def __init__(self, event_log=None):
        self._processes = {}   # capability_name → Popen
        self._scripts = {}     # capability_name → Path
        self._lock = threading.Lock()
        self.event_log = event_log
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start_all_from_dir(self, installed_dir: Path):
        """Start every background.py found in installed capabilities."""
        installed_dir = Path(installed_dir)
        if not installed_dir.exists():
            return
        for cap_dir in sorted(installed_dir.iterdir()):
            if not cap_dir.is_dir():
                continue
            bg = cap_dir / "background.py"
            if bg.exists():
                self.start(cap_dir.name, bg)

    def start(self, capability_name: str, script_path: Path) -> bool:
        script_path = Path(script_path)
        if not script_path.exists():
            return False
        with self._lock:
            self._stop_locked(capability_name)
            self._scripts[capability_name] = script_path
            self._launch_locked(capability_name)
        if not self._running:
            self._start_monitor()
        return True

    def stop(self, capability_name: str):
        with self._lock:
            self._stop_locked(capability_name)
            self._scripts.pop(capability_name, None)

    def stop_all(self):
        self._running = False
        with self._lock:
            for name in list(self._processes.keys()):
                self._stop_locked(name)

    def status(self) -> dict:
        with self._lock:
            return {
                name: "running" if proc.poll() is None else "stopped"
                for name, proc in self._processes.items()
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _launch_locked(self, capability_name: str):
        script = self._scripts.get(capability_name)
        if script is None:
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes[capability_name] = proc
        except Exception as exc:
            self._log(capability_name, "background_error",
                      f"Failed to start: {exc}")

    def _stop_locked(self, capability_name: str):
        proc = self._processes.pop(capability_name, None)
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _start_monitor(self):
        self._running = True
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()

    def _monitor_loop(self):
        while self._running:
            time.sleep(15)
            with self._lock:
                for name in list(self._processes):
                    if self._processes[name].poll() is not None:
                        self._launch_locked(name)
                        self._log(name, "background_restarted",
                                  f"Background process for '{name}' restarted")

    def _log(self, capability_name: str, event_type: str, message: str):
        if self.event_log is None:
            return
        try:
            self.event_log.write(capability_name, event_type, message, {})
        except Exception:
            pass
