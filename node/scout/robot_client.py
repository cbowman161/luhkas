import json
import urllib.error
import urllib.request


class RobotClient:
    def __init__(self, base_url: str, timeout: float = 0.15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def pantilt(self, payload: dict) -> bool:
        return self._post("/pantilt", payload)

    def camera_light(self, brightness: int) -> bool:
        value = max(0, min(255, int(brightness)))
        return self._post("/pantilt", {"mode": "light", "brightness": value})

    def move(self, x: int, z: int) -> bool:
        return self._post("/move", {"x": int(x), "z": int(z)})

    def _post(self, path: str, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except (urllib.error.URLError, TimeoutError):
            return False
