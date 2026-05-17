# Optional manual-start service — NOT enabled by default.
# To use: systemctl --user start scout-controller
# Service file: scout_runtime/systemd/scout-controller.service
import requests
from inputs import get_gamepad

API = "http://localhost:5001"
MOVE_SCALE = 500    # max wheel speed sent to robot_api /move
PAN_SCALE = 100     # max pan degrees per full stick deflection
TILT_SCALE = 60     # max tilt degrees per full stick deflection
PT_SPEED = 300      # relative pantilt sx/sy speed


def send_move(x: float, z: float) -> None:
    try:
        requests.post(
            f"{API}/move",
            json={"x": int(x * MOVE_SCALE), "z": int(z * MOVE_SCALE)},
            timeout=0.1,
        )
    except Exception:
        pass


def send_pantilt(pan: float, tilt: float) -> None:
    try:
        requests.post(
            f"{API}/pantilt",
            json={
                "mode": "relative",
                "pan": int(pan * PAN_SCALE),
                "tilt": int(tilt * TILT_SCALE),
                "sx": PT_SPEED,
                "sy": PT_SPEED,
            },
            timeout=0.1,
        )
    except Exception:
        pass


def deadzone(val: float, dz: float = 0.1) -> float:
    return 0.0 if abs(val) < dz else val


def controller_loop() -> None:
    x = 0.0
    z = 0.0

    while True:
        events = get_gamepad()

        for event in events:
            if event.code == "ABS_Y":
                x = deadzone(-event.state / 32768)
            elif event.code == "ABS_X":
                z = deadzone(event.state / 32768)
            elif event.code == "ABS_RY":
                send_pantilt(0.0, -event.state / 32768)
            elif event.code == "ABS_RX":
                send_pantilt(event.state / 32768, 0.0)

        send_move(x, z)


if __name__ == "__main__":
    controller_loop()
