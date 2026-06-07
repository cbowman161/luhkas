#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def _json_request(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class _ServerCase(unittest.TestCase):
    def start_server(self, server: ThreadingHTTPServer) -> str:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup() -> None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(cleanup)
        host, port = server.server_address
        return f"http://{host}:{port}"


class VaultSmokeTest(_ServerCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "vault"))
        sys.modules["vault_runtime"] = types.SimpleNamespace(VaultRuntime=object)
        sys.modules["sync_manager"] = types.SimpleNamespace(
            auto_push_if_new=lambda node_id: None,
            last_result=lambda: {"ok": None},
            pubkey=lambda: "",
        )
        cls.vault_service = importlib.import_module("vault_service")

    def test_health_node_register_and_presence_message(self) -> None:
        class Registry:
            def __init__(self) -> None:
                self.registered = {}

            def register(self, **kwargs) -> None:
                self.registered[kwargs["node_id"]] = kwargs

            def update_activity(self, *args, **kwargs) -> None:
                pass

            def node_url(self, *args, **kwargs):
                return None

        class Runtime:
            def __init__(self) -> None:
                self.node_registry = Registry()

            def health(self) -> dict:
                return {"ok": True, "service": "test-vault"}

            def handle_presence(self, message, node_id="scout", presence_context=None) -> dict:
                return {"message": f"heard {message}", "node_id": node_id}

        runtime = Runtime()
        server = self.vault_service.VaultHTTPServer(
            ("127.0.0.1", 0),
            self.vault_service.VaultRequestHandler,
            runtime,
        )
        base = self.start_server(server)

        status, payload = _json_request(base + "/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        status, payload = _json_request(
            base + "/node/register",
            method="POST",
            payload={
                "node_id": "scout",
                "display": {"has_display": False},
                "services": {"vision": 5000, "robot_api": 5001, "presence": 5002},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["node_id"], "scout")
        self.assertIn("scout", runtime.node_registry.registered)

        status, payload = _json_request(
            base + "/presence/message",
            method="POST",
            payload={"message": "hello", "node_id": "scout"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["response"]["node_id"], "scout")


class RobotApiSmokeTest(_ServerCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "node"))
        cls.robot_api = importlib.import_module("services.robot_api")

    def test_health_move_and_pantilt(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), self.robot_api.Handler)
        base = self.start_server(server)

        status, payload = _json_request(base + "/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        status, payload = _json_request(
            base + "/move",
            method="POST",
            payload={"x": 250, "z": -50},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["move"], {"x": 250, "z": -50})

        status, payload = _json_request(
            base + "/pantilt",
            method="POST",
            payload={"mode": "relative", "pan": 4, "tilt": -2},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["pantilt"]["mode"], "relative")


@unittest.skipIf(
    importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None,
    "vision service smoke tests require cv2 and numpy",
)
class VisionApiSmokeTest(_ServerCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "node"))
        cls.vision_service = importlib.import_module("services.vision_service")

    def test_meta_tracking_move_and_pantilt_contracts(self) -> None:
        class Robot:
            def __init__(self) -> None:
                self.moves = []
                self.pantilts = []

            def move(self, x, z):
                self.moves.append((x, z))
                return True

            def pantilt(self, command):
                self.pantilts.append(command)
                return True

        robot = Robot()
        self.vision_service.active_robot = robot
        self.vision_service.active_controller = None
        self.vision_service.latest_meta = {"ok": True, "detections": []}

        server = ThreadingHTTPServer(("127.0.0.1", 0), self.vision_service.Handler)
        base = self.start_server(server)

        status, payload = _json_request(base + "/meta")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        status, payload = _json_request(base + "/tracking", method="POST", payload={"enabled": True})
        self.assertEqual(status, 503)
        self.assertFalse(payload["ok"])

        status, payload = _json_request(base + "/move", method="POST", payload={"x": 600, "z": -600})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["x"], 500)
        self.assertEqual(payload["z"], -500)

        status, payload = _json_request(base + "/pantilt", method="POST", payload={"pan": 5, "tilt": 0})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(robot.pantilts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
