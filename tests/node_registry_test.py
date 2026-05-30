#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "vault"))
        self.node_registry = importlib.import_module("node_registry")

    def test_registry_prunes_synthetic_nodes_and_keeps_vault_and_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registered_file = Path(tmp) / "nodes_registered.json"
            nodes_file = Path(tmp) / "nodes.json"
            registered_file.write_text(
                json.dumps({
                    "batch123": {"node_name": "batch123"},
                    "testnode": {"node_name": "testnode"},
                    "scout": {
                        "node_name": "scout",
                        "ip": "192.168.4.38",
                        "services": {"presence": 5002},
                    },
                }),
                encoding="utf-8",
            )
            old_registered = self.node_registry.REGISTERED_FILE
            old_nodes = self.node_registry.NODES_FILE
            self.node_registry.REGISTERED_FILE = registered_file
            self.node_registry.NODES_FILE = nodes_file
            try:
                registry = self.node_registry.NodeRegistry()
            finally:
                self.node_registry.REGISTERED_FILE = old_registered
                self.node_registry.NODES_FILE = old_nodes

            nodes = registry.registered_nodes()
            self.assertEqual(sorted(nodes), ["scout", "vault"])
            self.assertTrue(nodes["vault"]["intrinsic"])

            persisted = json.loads(registered_file.read_text(encoding="utf-8"))
            self.assertEqual(sorted(persisted), ["scout", "vault"])

    def test_future_real_nodes_can_register(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_registered = self.node_registry.REGISTERED_FILE
            old_nodes = self.node_registry.NODES_FILE
            self.node_registry.REGISTERED_FILE = Path(tmp) / "nodes_registered.json"
            self.node_registry.NODES_FILE = Path(tmp) / "nodes.json"
            try:
                registry = self.node_registry.NodeRegistry()
                registry.register(
                    node_id="workshop",
                    display={"has_display": False},
                    node_name="Workshop",
                    ip="192.168.4.55",
                    network={
                        "lan_ip": "192.168.4.55",
                        "tailscale_ip": "100.64.12.34",
                        "preferred": "tailscale",
                    },
                    services={"presence": 5002},
                )
                registry.register(node_id="test999", display={}, node_name="test999")
            finally:
                self.node_registry.REGISTERED_FILE = old_registered
                self.node_registry.NODES_FILE = old_nodes

            nodes = registry.registered_nodes()
            self.assertEqual(sorted(nodes), ["vault", "workshop"])
            self.assertEqual(nodes["workshop"]["node_name"], "Workshop")
            self.assertEqual(
                registry.node_url("workshop", "presence"),
                "http://100.64.12.34:5002",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
