#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scout.config import RobotApiConfig
from scout.telemetry_logger import TelemetryLogger

try:
    import serial
except ImportError:  # pragma: no cover - handled at runtime on the Pi
    serial = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("robot_api")

config = RobotApiConfig()
telemetry_logger: TelemetryLogger | None = None
state_lock = threading.Lock()
serial_lock = threading.Lock()
ser = None
serial_buffer = ""
# Battery state is owned by battery_node. We publish each T:1001 reading to
# BATTERY_RAW_PATH for the uart_proxy backend to consume, and pull the
# canonical reading from battery_node's HTTP service when we need it
# locally (e.g. for the OLED display).
_BATTERY_RAW_DEFAULT_BASE = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
BATTERY_RAW_PATH = Path(
    os.environ.get("BATTERY_UART_PROXY_PATH")
    or f"{_BATTERY_RAW_DEFAULT_BASE}/luhkas-battery.json"
)
BATTERY_SERVICE_URL = os.environ.get("BATTERY_SERVICE_URL", "http://127.0.0.1:5003").rstrip("/")
latest_telemetry: dict = {}
last_heartbeat: float = 0.0
HEARTBEAT_TIMEOUT = 5.0  # seconds before wheels are stopped
pt_seq = 0
mv_seq = 0
oled_queue: queue.Queue = queue.Queue(maxsize=1)


def _publish_battery_reading(voltage: float, percent: int) -> None:
    """Write the latest UART battery sample for battery_node to read."""
    try:
        BATTERY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BATTERY_RAW_PATH.with_suffix(BATTERY_RAW_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "voltage": voltage,
            "percent": percent,
            "timestamp": time.time(),
            "source": "robot_api_uart",
        }))
        tmp.replace(BATTERY_RAW_PATH)
    except Exception as exc:
        log.debug("battery publish failed: %s", exc)


def _fetch_battery_for_display() -> dict:
    """Best-effort fetch of canonical battery reading from battery_node."""
    try:
        with urlopen(f"{BATTERY_SERVICE_URL}/battery", timeout=1.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

pt_cmd = {
    "mode": "stop",
    "pan": 0,
    "tilt": 0,
    "spd": 0,
    "acc": 0,
    "sx": 0,
    "sy": 0,
    "brightness": 0,
}

mv_cmd = {
    "x": 0,
    "z": 0,
}


def init_serial() -> None:
    global ser
    if serial is None:
        raise RuntimeError("pyserial is not installed in this Python environment")

    with serial_lock:
        if ser and ser.is_open:
            ser.close()
        ser = serial.Serial(config.serial_port, config.baud_rate, timeout=0)
    time.sleep(2)
    log.info("Serial connected on %s at %s", config.serial_port, config.baud_rate)


def write_serial(payload: dict) -> None:
    global ser
    line = (json.dumps(payload) + "\n").encode("utf-8")

    with serial_lock:
        try:
            if not ser or not ser.is_open:
                raise RuntimeError("serial not open")
            ser.write(line)
            return
        except Exception as exc:
            log.warning("Serial write failed, reconnecting: %s", exc)

    try:
        init_serial()
        with serial_lock:
            ser.write(line)
    except Exception as exc:
        log.error("Serial reconnect failed: %s", exc)


def _brain_reachable() -> bool:
    try:
        from urllib.request import urlopen
        with urlopen(f"{config.brain_url}/health", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def heartbeat_watchdog() -> None:
    global last_heartbeat, mv_seq
    while True:
        time.sleep(1.0)
        if _brain_reachable():
            last_heartbeat = time.time()
            continue
        if last_heartbeat == 0.0:
            continue
        elapsed = time.time() - last_heartbeat
        if elapsed > HEARTBEAT_TIMEOUT:
            with state_lock:
                if mv_cmd["x"] != 0 or mv_cmd["z"] != 0:
                    log.warning("Brain unreachable (%.1fs) — stopping wheels", elapsed)
                    mv_cmd["x"] = 0
                    mv_cmd["z"] = 0
                    mv_seq += 1


def serial_reader() -> None:
    global serial_buffer
    while True:
        try:
            if not ser or not ser.is_open:
                time.sleep(0.2)
                continue

            waiting = ser.in_waiting
            if waiting <= 0:
                time.sleep(0.01)
                continue

            chunk = ser.read(waiting).decode("utf-8", errors="ignore")
            serial_buffer += chunk

            while "\n" in serial_buffer:
                line, serial_buffer = serial_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("T") == 1001:
                    raw_voltage = float(data.get("v", 0))
                    voltage = raw_voltage / 100.0
                    percent = max(0, min(100, int((voltage - 9.9) / (12.6 - 9.9) * 100)))
                    _publish_battery_reading(voltage, percent)
                    latest_telemetry.update({
                        "timestamp": time.time(),
                        "motors": {"L": data.get("L", 0), "R": data.get("R", 0)},
                        "accel": {"x": data.get("ax", 0), "y": data.get("ay", 0), "z": data.get("az", 0)},
                        "gyro": {"x": data.get("gx", 0), "y": data.get("gy", 0), "z": data.get("gz", 0)},
                        "mag": {"x": data.get("mx", 0), "y": data.get("my", 0), "z": data.get("mz", 0)},
                        "encoders": {"left": data.get("odl", 0), "right": data.get("odr", 0)},
                    })
                    if telemetry_logger is not None:
                        telemetry_logger.log(data)

            if len(serial_buffer) > 4096:
                serial_buffer = ""
        except Exception as exc:
            log.error("Serial read error: %s", exc)
            time.sleep(1)
            try:
                init_serial()
            except Exception:
                pass


def oled_updater() -> None:
    show_ip = False
    ip_addr = _local_ip()
    while True:
        try:
            # Check for an override pushed via the HTTP /oled endpoint; drop if
            # the queue fills (serial is busy) rather than blocking motor commands.
            try:
                lines = oled_queue.get_nowait()
                write_serial({"T": 3, "lineNum": 0, "Text": str(lines[0])})
                time.sleep(0.05)
                write_serial({"T": 3, "lineNum": 1, "Text": str(lines[1])})
            except queue.Empty:
                line1 = "   L  U  H  K  A  S   "
                if show_ip:
                    line2 = f"IP: {ip_addr}"
                else:
                    reading = _fetch_battery_for_display()
                    percent = reading.get("percent") if reading.get("ok") else None
                    line2 = f"BATTERY ::: {percent} %" if percent is not None else "BATTERY ::: ?? %"
                write_serial({"T": 3, "lineNum": 0, "Text": line1})
                time.sleep(0.05)
                write_serial({"T": 3, "lineNum": 1, "Text": line2})
                show_ip = not show_ip
        except Exception as exc:
            log.warning("OLED update failed: %s", exc)
        time.sleep(2)


def serial_worker() -> None:
    last_pt_seq = -1
    last_mv_seq = -1
    last_light = None

    while True:
        time.sleep(0.03)
        with state_lock:
            current_pt = pt_cmd.copy()
            current_mv = mv_cmd.copy()
            current_pt_seq = pt_seq
            current_mv_seq = mv_seq

        brightness = int(current_pt.get("brightness", 0))
        if brightness != last_light:
            write_serial({"T": 132, "IO4": brightness, "IO5": brightness})
            last_light = brightness

        if current_pt_seq != last_pt_seq:
            serial_cmd = _pantilt_serial_command(current_pt)
            if serial_cmd:
                write_serial(serial_cmd)
            last_pt_seq = current_pt_seq

        if current_mv_seq != last_mv_seq:
            write_serial(_move_serial_command(current_mv))
            last_mv_seq = current_mv_seq


_MAX_WHEEL_SPEED = 1.0


def _clamp_wheel_speed(value: float) -> float:
    return max(-_MAX_WHEEL_SPEED, min(_MAX_WHEEL_SPEED, value))


def _move_serial_command(command: dict) -> dict:
    x = _clamp_wheel_speed(float(command.get("x", 0)) / 1000.0)
    z = _clamp_wheel_speed(float(command.get("z", 0)) / 1000.0)
    left = _clamp_wheel_speed(x + z)
    right = _clamp_wheel_speed(x - z)
    if left * right < 0:
        left *= 0.5
        right *= 0.5
    return {"T": 1, "L": round(left, 3), "R": round(right, 3)}


def _pantilt_serial_command(command: dict) -> dict | None:
    mode = command.get("mode", "stop")
    if mode == "absolute":
        return {
            "T": 133,
            "X": int(command.get("pan", 0)),
            "Y": int(command.get("tilt", 0)),
            "SPD": int(command.get("spd", 0)),
            "ACC": int(command.get("acc", 0)),
        }
    if mode == "relative":
        return {
            "T": 134,
            "X": int(command.get("pan", 0)),
            "Y": int(command.get("tilt", 0)),
            "SX": int(command.get("sx", 0)),
            "SY": int(command.get("sy", 0)),
        }
    if mode == "stop":
        return {"T": 135}
    if mode == "light":
        return None
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({
                "ok": True,
                "serial": bool(ser and ser.is_open),
            })
        elif self.path == "/telemetry":
            self._json(latest_telemetry)
        elif self.path.startswith("/telemetry/history"):
            self._telemetry_history()
        elif self.path == "/heartbeat":
            self._json({"ok": True, "last_heartbeat": last_heartbeat, "timeout": HEARTBEAT_TIMEOUT})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_json()
        if body is None:
            return

        if self.path == "/pantilt":
            self._handle_pantilt(body)
        elif self.path == "/heartbeat":
            self._handle_heartbeat()
        elif self.path == "/move":
            self._handle_move(body)
        elif self.path == "/oled":
            self._handle_oled(body)
        elif self.path == "/send":
            write_serial(body)
            self._json({"status": "sent", "cmd": body})
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)

    def _telemetry_history(self) -> None:
        if telemetry_logger is None:
            self._json({"ok": False, "error": "telemetry_logging_disabled"}, status=503)
            return
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(self.path).query)
        try:
            seconds = min(float((params.get("seconds") or ["60"])[0]), 3600.0)
        except (ValueError, IndexError):
            seconds = 60.0
        rows = telemetry_logger.recent(seconds)
        self._json(rows)

    def _handle_heartbeat(self) -> None:
        global last_heartbeat
        last_heartbeat = time.time()
        self._json({"ok": True})

    def _handle_pantilt(self, body: dict) -> None:
        global pt_seq
        mode = body.get("mode", "stop")
        with state_lock:
            pt_cmd["mode"] = mode
            pt_cmd["pan"] = body.get("pan", 0)
            pt_cmd["tilt"] = body.get("tilt", 0)
            pt_cmd["spd"] = body.get("spd", 0)
            pt_cmd["acc"] = body.get("acc", 0)
            pt_cmd["sx"] = body.get("sx", 0)
            pt_cmd["sy"] = body.get("sy", 0)
            pt_cmd["brightness"] = body.get("brightness", pt_cmd.get("brightness", 0))
            pt_seq += 1
        self._json({"ok": True, "status": "ok", "pantilt": pt_cmd})

    def _handle_move(self, body: dict) -> None:
        global mv_seq
        with state_lock:
            mv_cmd["x"] = int(body.get("x", 0))
            mv_cmd["z"] = int(body.get("z", 0))
            mv_seq += 1
        self._json({"ok": True, "status": "ok", "move": mv_cmd})

    def _handle_oled(self, body: dict) -> None:
        lines = body.get("lines", [])
        if len(lines) < 2:
            self.send_error(400, "expected JSON body with lines: [line0, line1]")
            return
        try:
            oled_queue.put_nowait(lines)
        except queue.Full:
            pass  # drop if updater thread is busy; cosmetic display, never block
        self._json({"status": "ok"})

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return None

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip_addr = sock.getsockname()[0]
        sock.close()
        return ip_addr
    except Exception:
        return "No IP"


def main() -> None:
    global telemetry_logger
    if config.telemetry_log_enabled:
        try:
            telemetry_logger = TelemetryLogger(config.telemetry_db_path)
            log.info("Telemetry logger active: %s", config.telemetry_db_path)
        except Exception as exc:
            log.warning("Telemetry logger init failed: %s", exc)

    try:
        init_serial()
    except Exception as exc:
        log.error("Starting without serial connection: %s", exc)

    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=oled_updater, daemon=True).start()
    threading.Thread(target=serial_worker, daemon=True).start()
    threading.Thread(target=heartbeat_watchdog, daemon=True).start()

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    log.info("Robot API listening on http://%s:%s", config.host, config.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
