from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from .config import VaultMemoryConfig

log = logging.getLogger(__name__)


class BrainMemoryClient:
    def __init__(self, config: VaultMemoryConfig | None = None) -> None:
        self.config = config or VaultMemoryConfig()
        self.enabled = bool(self.config.enabled and self.config.url)
        self.base_url = self.config.url.rstrip("/") + "/" if self.config.url else ""
        self.face_cache_dir = Path(self.config.face_cache_dir).expanduser()
        self.last_face_sync_at = 0.0
        self.last_face_sync_ok: bool | None = None
        self.last_error: str | None = None
        if self.enabled:
            self.face_cache_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "url": self.config.url,
            "face_cache_dir": str(self.face_cache_dir),
            "last_face_sync_at": self.last_face_sync_at,
            "last_face_sync_ok": self.last_face_sync_ok,
            "last_error": self.last_error,
        }

    def person_summary(self, identity: str) -> dict | None:
        if not self.enabled:
            return None
        result = self._request_json("GET", f"people/{quote(identity)}/summary")
        if not result or result.get("ok") is False:
            return None
        summary = result.get("summary") or result.get("person_memory") or result
        if not isinstance(summary, dict):
            return None
        summary.setdefault("identity", identity)
        summary.setdefault("source", "brain")
        return summary

    def get_person_memory(self, identity: str) -> dict | None:
        if not self.enabled:
            return None
        return self._request_json("GET", f"people/{quote(identity)}/memory")

    def remember(self, identity: str, payload: dict) -> dict | None:
        if not self.enabled:
            return None
        return self._request_json("POST", f"people/{quote(identity)}/remember", payload)

    def set_preference(self, identity: str, payload: dict) -> dict | None:
        if not self.enabled:
            return None
        return self._request_json("POST", f"people/{quote(identity)}/preference", payload)

    def sync_face_references_if_due(self, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        if not force and now - self.last_face_sync_at < self.config.face_sync_interval_seconds:
            return False
        changed = self.sync_face_references()
        self.last_face_sync_at = now
        return changed

    def sync_on_unknown_identity(self) -> bool:
        """Trigger a face sync when the recognizer sees an unknown person.
        Only syncs if enough time has passed since the last sync to avoid hammering the brain."""
        if not self.enabled:
            return False
        now = time.time()
        # Throttle: at most once every 30s for unknown-triggered syncs
        if now - self.last_face_sync_at < 30.0:
            return False
        changed = self.sync_face_references()
        self.last_face_sync_at = now
        return changed

    def sync_face_references(self) -> bool:
        payload = self._request_json("GET", "faces/sync")
        if not payload:
            self.last_face_sync_ok = False
            return False

        people = payload.get("people", [])
        if not isinstance(people, list):
            self.last_face_sync_ok = False
            self.last_error = "invalid_faces_sync_payload"
            return False

        changed = False
        for person in people:
            if not isinstance(person, dict):
                continue
            identity = _safe_path_name(str(person.get("identity", "")))
            if not identity:
                continue
            samples = person.get("samples", [])
            if not isinstance(samples, list):
                continue
            for sample in samples:
                if self._write_face_sample(identity, sample):
                    changed = True

        self.last_face_sync_ok = True
        self.last_error = None
        return changed

    def upload_face_reference(self, result: dict) -> None:
        if not self.enabled or not result.get("ok"):
            return
        path = result.get("saved_path")
        if not path:
            return
        image_path = Path(path)
        if not image_path.exists():
            return
        try:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            self.last_error = str(exc)
            return

        payload = {
            "identity": result.get("identity"),
            "reference_pose": result.get("reference_pose"),
            "auto": bool(result.get("auto")),
            "image_b64": image_b64,
            "filename": image_path.name,
        }
        self._request_json("POST", f"people/{quote(str(result.get('identity', '')))}/faces", payload)

    def upload_unknown_face_observation(self, payload: dict) -> dict | None:
        if not self.enabled:
            return None
        return self._request_json("POST", "faces/unknown", payload)

    def promote_unknown_face_group(self, group_id: str, identity: str) -> dict | None:
        if not self.enabled or not group_id or not identity:
            return None
        return self._request_json("POST", "faces/unknown/promote", {
            "group_id": group_id,
            "identity": identity,
        })

    def _write_face_sample(self, identity: str, sample: dict) -> bool:
        if not isinstance(sample, dict):
            return False
        rel_path = _safe_relative_path(str(sample.get("path") or sample.get("filename") or ""))
        image_b64 = sample.get("image_b64")
        image_url = sample.get("url")
        if not rel_path:
            rel_path = f"{int(time.time() * 1000)}.jpg"

        destination = self.face_cache_dir / identity / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            if image_b64:
                data = base64.b64decode(str(image_b64), validate=True)
            elif image_url:
                data = self._request_bytes(str(image_url))
                if data is None:
                    return False
            else:
                return False
            if destination.exists() and destination.read_bytes() == data:
                return False
            destination.write_bytes(data)
            return True
        except (OSError, ValueError) as exc:
            self.last_error = str(exc)
            log.warning("Could not write brain face sample %s: %s", destination, exc)
            return False

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict | None:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(urljoin(self.base_url, path), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read()
        except (OSError, URLError) as exc:
            self.last_error = str(exc)
            return None
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            return None
        return decoded if isinstance(decoded, dict) else None

    def _request_bytes(self, url: str) -> bytes | None:
        try:
            with urlopen(url, timeout=self.config.timeout_seconds) as response:
                return response.read()
        except (OSError, URLError) as exc:
            self.last_error = str(exc)
            return None


def _safe_path_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._-")[:64]


def _safe_relative_path(value: str) -> str:
    parts = [_safe_path_name(part) for part in value.replace("\\", "/").split("/")]
    parts = [part for part in parts if part]
    return "/".join(parts[:4])
