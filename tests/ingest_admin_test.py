#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))


class DetectActionTests(unittest.TestCase):
    def setUp(self):
        from world.ingest_admin import detect_action
        self.detect = detect_action

    def test_start_phrasings(self):
        for msg in (
            "start the wikipedia ingest",
            "start wiki ingest",
            "begin wikipedia ingestion",
            "kick off the wiki ingest",
            "launch the wikipedia indexing",
            "resume the wiki ingest",
        ):
            self.assertEqual(self.detect(msg), "start", msg=msg)

    def test_stop_phrasings(self):
        for msg in (
            "stop the wikipedia ingest",
            "stop wiki ingest",
            "halt the wikipedia ingestion",
            "cancel wiki indexing",
            "kill the wiki ingest",
            "pause the wikipedia ingest",
        ):
            self.assertEqual(self.detect(msg), "stop", msg=msg)

    def test_status_phrasings(self):
        for msg in (
            "wikipedia ingest status",
            "wiki ingest progress",
            "how is the wiki ingest going",
            "how's the wikipedia ingest",
            "ingestion progress",
            "wiki progress",
            "world status",
        ):
            self.assertEqual(self.detect(msg), "status", msg=msg)

    def test_chat_lines_do_not_match(self):
        for msg in (
            "what is wikipedia",
            "do you have wikipedia",
            "tell me about indexing in databases",
            "i want to stop talking about this",
            "how is the weather",
            "what's my favorite drink",
            "start the dishwasher",
        ):
            self.assertIsNone(self.detect(msg), msg=msg)

    def test_build_index_phrasings(self):
        for msg in (
            "build the wiki index",
            "build wiki index",
            "rebuild the wikipedia index",
            "create the wiki search index",
            "make the wikipedia index",
            "build the world index",
        ):
            self.assertEqual(self.detect(msg), "build_index", msg=msg)

    def test_index_status_phrasings(self):
        for msg in (
            "wiki index status",
            "is the wiki index built",
            "is the wikipedia search index built",
            "world index state",
        ):
            self.assertEqual(self.detect(msg), "index_status", msg=msg)


class FormatStatusTests(unittest.TestCase):
    def setUp(self):
        import os, tempfile
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["WORLD_INGEST_STATE_FILE"] = str(Path(self.tmp.name) / "state.json")
        os.environ["WORLD_INGEST_PID_FILE"] = str(Path(self.tmp.name) / "pid")
        os.environ["WORLD_INGEST_RUNNER"] = str(Path(self.tmp.name) / "runner.sh")
        for mod in ["world.ingest_admin"]:
            sys.modules.pop(mod, None)
        from world.ingest_admin import handle
        self.handle = handle

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_without_state(self):
        resp = self.handle("wikipedia ingest status")
        self.assertIsNotNone(resp)
        self.assertIn("hasn't been started", resp["message"])

    def test_status_with_running_state(self):
        import json, os, time
        state = {
            "articles_seen": 50000,
            "articles_new": 30000,
            "articles_replaced": 0,
            "articles_skipped_unchanged": 100,
            "articles_empty": 19900,
            "chunks_written": 175000,
            "last_committed_index": 200000,
            "started_at": time.time() - 600,
            "elapsed_s": 600,
            "completed": False,
        }
        with open(os.environ["WORLD_INGEST_STATE_FILE"], "w") as fh:
            json.dump(state, fh)
        # Fake-running pid
        pid = os.getpid()
        with open(os.environ["WORLD_INGEST_PID_FILE"], "w") as fh:
            fh.write(str(pid))
        resp = self.handle("how is the wiki ingest going")
        self.assertIsNotNone(resp)
        msg = resp["message"]
        self.assertIn("running", msg.lower())
        self.assertIn("30000", msg.replace(",", ""))
        self.assertIn("ETA", msg)


if __name__ == "__main__":
    unittest.main()
