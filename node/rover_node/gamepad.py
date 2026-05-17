"""USB gamepad/manual rover runtime."""
from __future__ import annotations

import glob
import os
import select
import struct
import threading
import time
from collections.abc import Callable


class GamepadRuntime:
    def __init__(
        self,
        get_robot: Callable[[], object | None],
        get_controller: Callable[[], object | None],
        save_snapshot: Callable[[], dict],
        save_clip: Callable[[], dict],
        toggle_light: Callable[[], int | None],
        adjust_light: Callable[[int], int | None],
    ) -> None:
        self._get_robot = get_robot
        self._get_controller = get_controller
        self._save_snapshot = save_snapshot
        self._save_clip = save_clip
        self._toggle_light = toggle_light
        self._adjust_light = adjust_light
        self._lock = threading.Lock()
        self._status = {
            "enabled": False,
            "connected": False,
            "device": None,
            "last_event": 0.0,
            "last_action": None,
            "axes": {},
            "buttons": {},
        }

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._status.get("enabled"))

    def device(self) -> str | None:
        with self._lock:
            return self._status.get("device")

    def set_status(self, connected: bool, device: str | None, action: str | None) -> None:
        with self._lock:
            self._status["connected"] = connected
            self._status["device"] = device
            self._status["last_event"] = time.time() if connected else self._status.get("last_event", 0.0)
            if action is not None:
                self._status["last_action"] = action

    def set_manual_enabled(self, enabled: bool, search_config=None) -> None:
        controller = self._get_controller()
        robot = self._get_robot()
        with self._lock:
            self._status["enabled"] = enabled
            self._status["last_action"] = "manual on" if enabled else "manual off"
        if enabled:
            if controller:
                controller.config.enabled = False
            if search_config is not None:
                search_config.enabled = False
            if robot:
                robot.move(0, 0)
        else:
            if robot:
                robot.move(0, 0)

    def loop(self) -> None:
        axes: dict[int, float] = {}
        buttons: dict[int, bool] = {}
        fd = None
        device = None
        last_camera_at = 0.0
        last_move_at = 0.0
        last_move = (0, 0)

        while True:
            if fd is None:
                paths = sorted(glob.glob("/dev/input/js*"))
                if not paths:
                    self.set_status(False, None, None)
                    time.sleep(1.0)
                    continue
                device = paths[0]
                try:
                    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
                    self.set_status(True, device, "connected")
                except OSError as exc:
                    self.set_status(False, device, f"open failed: {exc}")
                    fd = None
                    time.sleep(1.0)
                    continue

            try:
                ready, _, _ = select.select([fd], [], [], 0.02)
                if ready:
                    try:
                        data = os.read(fd, 8)
                    except BlockingIOError:
                        data = b""
                    if len(data) == 8:
                        _, value, event_type, number = struct.unpack("IhBB", data)
                        event_type &= 0x7F
                        with self._lock:
                            self._status["connected"] = True
                            self._status["device"] = device
                            self._status["last_event"] = time.time()
                        if event_type == 0x02:
                            axes[number] = self.axis(value)
                            with self._lock:
                                self._status["axes"] = {str(k): round(v, 3) for k, v in sorted(axes.items())}
                        elif event_type == 0x01:
                            down = bool(value)
                            was = buttons.get(number, False)
                            buttons[number] = down
                            with self._lock:
                                self._status["buttons"] = {str(k): v for k, v in sorted(buttons.items()) if v}
                            if down and not was:
                                self.button_action(number)

                now = time.time()
                lx = self.deadzone(axes.get(0, 0.0))
                ly = self.deadzone(axes.get(1, 0.0))
                rx = self.deadzone(self.right_stick_axis(axes, "x"))
                ry = self.deadzone(self.right_stick_axis(axes, "y"))
                if (lx or ly or rx or ry) and not self.enabled():
                    self.set_manual_enabled(True)

                if not self.enabled():
                    time.sleep(0.05)
                    continue

                if (rx or ry) and now - last_camera_at >= 0.14:
                    last_camera_at = now
                    self.camera(rx, ry)

                if now - last_move_at >= 0.12:
                    move = (int(round(-ly * 450)), int(round(lx * 320)))
                    if abs(move[0] - last_move[0]) > 15 or abs(move[1] - last_move[1]) > 15:
                        last_move_at = now
                        last_move = move
                        self.move(move[0], move[1])
            except OSError as exc:
                self.set_status(False, device, f"disconnected: {exc}")
                try:
                    os.close(fd)
                except Exception:
                    pass
                fd = None
                device = None
                axes.clear()
                buttons.clear()
                last_move = (0, 0)
                robot = self._get_robot()
                if robot:
                    robot.move(0, 0)

    def disable_tracking(self) -> None:
        controller = self._get_controller()
        if controller and controller.config.enabled:
            controller.config.enabled = False

    def camera(self, rx: float, ry: float) -> None:
        robot = self._get_robot()
        controller = self._get_controller()
        if robot is None or controller is None:
            return
        self.disable_tracking()
        next_pan = controller._clamp_pan(controller._estimated_pan + int(round(rx * 5)))
        next_tilt = controller._clamp_tilt(controller._estimated_tilt + int(round(-ry * 5)))
        robot.pantilt({"mode": "absolute", "pan": int(round(next_pan)), "tilt": int(round(next_tilt)), "spd": 0, "acc": 0})
        controller.notify_external_pantilt(next_pan, next_tilt)
        self.set_status(True, self.device(), "camera")

    def move(self, x: int, z: int) -> None:
        robot = self._get_robot()
        if robot is None:
            return
        if x or z:
            self.disable_tracking()
        robot.move(x, z)
        self.set_status(True, self.device(), "move" if (x or z) else "stop")

    def button_action(self, button: int) -> None:
        if button == 0:
            robot = self._get_robot()
            controller = self._get_controller()
            if robot and controller:
                self.disable_tracking()
                robot.pantilt(controller.center_command())
                self.set_status(True, self.device(), "center camera")
        elif button == 1:
            brightness = self._toggle_light()
            if brightness is not None:
                self.set_status(True, self.device(), f"light {brightness}")
        elif button == 2:
            self._save_snapshot()
        elif button == 3:
            self._save_clip()
        elif button == 4:
            brightness = self._adjust_light(-25)
            if brightness is not None:
                self.set_status(True, self.device(), f"light {brightness}")
        elif button == 5:
            brightness = self._adjust_light(25)
            if brightness is not None:
                self.set_status(True, self.device(), f"light {brightness}")

    @staticmethod
    def axis(value: int) -> float:
        return max(-1.0, min(1.0, float(value) / 32767.0))

    @staticmethod
    def deadzone(value: float, threshold: float = 0.18) -> float:
        return 0.0 if abs(value) < threshold else value

    @staticmethod
    def right_stick_axis(axes: dict[int, float], axis: str) -> float:
        if axis == "x":
            if 3 in axes:
                return axes.get(3, 0.0)
            return axes.get(2, 0.0)
        if 4 in axes:
            return axes.get(4, 0.0)
        return axes.get(3, 0.0)
