"""Face detection, unknown-face grouping, and identity prompt runtime."""
from __future__ import annotations

import base64
import re
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from scout.vision import bbox_iou, center_inside, is_face_label


class FaceRuntime:
    def __init__(
        self,
        face_config,
        recognition_config,
        unknown_face_dir: Path,
        chat_log_add: Callable[..., dict],
        vault_client_getter: Callable[[], object | None],
    ) -> None:
        self.face_config = face_config
        self.recognition_config = recognition_config
        self.unknown_face_dir = unknown_face_dir
        self._chat_log_add = chat_log_add
        self._vault_client_getter = vault_client_getter
        self.identity_prompt_lock = threading.Lock()
        self.identity_prompt_active_face_id: int | None = None
        self.identity_prompt_completed_face_ids: dict[int | str, float] = {}
        self.latest_identity_prompt: dict | None = None
        self.identity_prompt_text_value = "Who are you?"
        self.identity_prompt_complete_grace_seconds = 20.0
        self.unknown_face_lock = threading.Lock()
        self.unknown_face_groups: dict[str, dict] = {}
        self.unknown_face_next_id: int = 1

    def configure_identity_prompt(self, text: str, complete_grace_seconds: float) -> None:
        self.identity_prompt_text_value = (text or "").strip()
        self.identity_prompt_complete_grace_seconds = complete_grace_seconds

    def select_identity_prompt_target(
        self,
        tracker,
        target,
        tracked: list,
        frame_id: int,
        frame_w: int,
        frame_h: int,
        prompt_enabled: bool,
    ):
        now = time.monotonic()
        self.prune_identity_prompt_completions(now)
        entries = self.visible_unknown_face_entries(tracked)
        queue = {
            "visible_face_count": len([det for det in tracked if is_face_label(det.label)]),
            "unknown_face_count": len(entries),
            "active_face_id": self.identity_prompt_active_face_id,
            "active_face_group_id": None,
            "faces": [
                {
                    "face_id": face.id,
                    "face_group_id": face.face_group_id,
                    "vault_face_group_id": face.vault_face_group_id,
                    "person_id": person.id,
                    "index": idx + 1,
                    "bbox": [int(v) for v in face.bbox],
                }
                for idx, (face, person) in enumerate(entries)
            ],
        }
        if not entries:
            with self.identity_prompt_lock:
                self.identity_prompt_active_face_id = None
                self.latest_identity_prompt = None
                queue["active_face_id"] = None
            return target, queue
        if not prompt_enabled:
            with self.identity_prompt_lock:
                self.identity_prompt_active_face_id = None
                self.latest_identity_prompt = None
                queue["active_face_id"] = None
            return target, queue

        active_entry = next(
            (
                (face, person) for face, person in entries
                if face.id == self.identity_prompt_active_face_id
                or (
                    self.latest_identity_prompt
                    and face.face_group_id
                    and face.face_group_id == self.latest_identity_prompt.get("face_group_id")
                )
            ),
            None,
        )
        if active_entry is None:
            active_entry = entries[0]
            with self.identity_prompt_lock:
                self.identity_prompt_active_face_id = active_entry[0].id

        face, person = active_entry
        active_index = next((idx for idx, (candidate, _) in enumerate(entries) if candidate.id == face.id), 0)
        queue["active_face_id"] = face.id
        queue["active_face_group_id"] = face.face_group_id
        queue["active_index"] = active_index + 1

        prompt_text = self.identity_prompt_text(len(entries), active_index + 1).strip()
        if not prompt_text:
            with self.identity_prompt_lock:
                self.latest_identity_prompt = None
            selected = replace(person)
            selected.aim_x, selected.aim_y = face.center
            selected.aim_source = f"face:{face.id}"
            selected.identity = face.identity or self.recognition_config.unknown_label
            selected.identity_confidence = face.identity_confidence
            selected.recognition_distance = face.recognition_distance
            selected.recognition_method = face.recognition_method
            selected.reference_pose = face.reference_pose
            selected.missing_reference_poses = list(face.missing_reference_poses)
            tracker.selected_target_id = selected.id
            tracker.selected_memory_id = selected.memory_id
            return selected, queue
        prompt = None
        with self.identity_prompt_lock:
            previous = self.latest_identity_prompt or {}
            should_prompt = previous.get("face_group_id") != face.face_group_id if face.face_group_id else previous.get("face_id") != face.id
            if should_prompt:
                prompt = {
                    "id": f"{int(time.time() * 1000)}-{face.id}",
                    "type": "unknown_face",
                    "prompt": prompt_text,
                    "face_id": face.id,
                    "face_group_id": face.face_group_id,
                    "vault_face_group_id": face.vault_face_group_id,
                    "person_id": person.id,
                    "face_count": len(entries),
                    "face_index": active_index + 1,
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "monotonic_ts": now,
                }
                self.latest_identity_prompt = prompt
        if prompt is not None and prompt.get("prompt"):
            self._chat_log_add(
                "assistant",
                prompt.get("prompt", self.identity_prompt_text_value),
                source="identity_prompt",
                face_id=prompt.get("face_id"),
                face_group_id=prompt.get("face_group_id"),
                vault_face_group_id=prompt.get("vault_face_group_id"),
                person_id=prompt.get("person_id"),
                face_count=prompt.get("face_count"),
                face_index=prompt.get("face_index"),
                frame_id=frame_id,
            )
        selected = replace(person)
        selected.aim_x, selected.aim_y = face.center
        selected.aim_source = f"face:{face.id}"
        selected.identity = face.identity or self.recognition_config.unknown_label
        selected.identity_confidence = face.identity_confidence
        selected.recognition_distance = face.recognition_distance
        selected.recognition_method = face.recognition_method
        selected.reference_pose = face.reference_pose
        selected.missing_reference_poses = list(face.missing_reference_poses)
        tracker.selected_target_id = selected.id
        tracker.selected_memory_id = selected.memory_id
        return selected, queue

    def visible_unknown_face_entries(self, tracked: list) -> list[tuple]:
        with self.identity_prompt_lock:
            completed = set(self.identity_prompt_completed_face_ids)
        people = [det for det in tracked if det.label == "person"]
        entries = []
        for face in tracked:
            group_id = getattr(face, "face_group_id", None)
            if not is_face_label(face.label) or face.id is None:
                continue
            if group_id in completed or face.id in completed:
                continue
            if face.identity and face.identity != self.recognition_config.unknown_label:
                continue
            if group_id and not self.unknown_face_group_ready(group_id):
                continue
            elif not group_id and face.seen_count < self.face_config.intro_min_seen_frames:
                continue
            containing_people = [person for person in people if self.face_matches_person(face, person)]
            if not containing_people:
                continue
            person = min(containing_people, key=lambda candidate: self.bbox_area(candidate.bbox))
            entries.append((face, person))
        entries.sort(key=lambda item: (item[0].center[0], item[0].center[1], item[0].id or 0))
        return entries

    def identity_prompt_text(self, face_count: int, face_index: int) -> str:
        if face_count <= 1:
            return self.identity_prompt_text_value
        return f"I see {face_count} unknown faces. Face {face_index}, {self.identity_prompt_text_value}"

    def mark_identity_prompt_learned(self, face_id: int | None, identity: str) -> None:
        with self.identity_prompt_lock:
            prompt_group_id = self.latest_identity_prompt.get("face_group_id") if self.latest_identity_prompt else None
            if face_id is not None:
                self.identity_prompt_completed_face_ids[face_id] = time.monotonic()
            if prompt_group_id:
                self.identity_prompt_completed_face_ids[prompt_group_id] = time.monotonic()
            if self.identity_prompt_active_face_id == face_id:
                self.identity_prompt_active_face_id = None
            if self.latest_identity_prompt and (self.latest_identity_prompt.get("face_id") == face_id or prompt_group_id):
                self.latest_identity_prompt = {
                    **self.latest_identity_prompt,
                    "recognized_identity": identity,
                    "completed": True,
                    "completed_ts": time.time(),
                }

    def prune_identity_prompt_completions(self, now: float) -> None:
        with self.identity_prompt_lock:
            stale = [
                face_id for face_id, completed_at in self.identity_prompt_completed_face_ids.items()
                if now - completed_at > self.identity_prompt_complete_grace_seconds
            ]
            for face_id in stale:
                self.identity_prompt_completed_face_ids.pop(face_id, None)

    def detect_faces_for_people(self, face_detector, frame: np.ndarray, detections: list) -> list:
        people = [det for det in detections if det.label == "person"]
        faces = []
        for person in people:
            faces.extend(face_detector.detect_in_bbox(frame, person.bbox))
        return self.dedupe_face_detections(faces)

    def dedupe_face_detections(self, faces: list) -> list:
        selected = []
        for face in sorted(faces, key=lambda det: det.confidence, reverse=True):
            if any(bbox_iou(face.bbox, existing.bbox) > 0.45 for existing in selected):
                continue
            selected.append(face)
            if len(selected) >= self.face_config.max_faces:
                break
        return selected

    def attach_face_identities_to_people(self, detections) -> None:
        faces = [det for det in detections if is_face_label(det.label) and det.identity and det.identity != "unknown"]
        if not faces:
            return
        for det in detections:
            if det.label != "person" or det.identity:
                continue
            inside = [face for face in faces if center_inside(face.center, det.bbox)]
            if not inside:
                continue
            face = max(inside, key=lambda candidate: candidate.identity_confidence or 0.0)
            det.copy_identity_from(face)

    def filter_faces_inside_people(self, detections, people_source):
        people = [det for det in people_source if det.label == "person"]
        if not people:
            return [det for det in detections if not is_face_label(det.label)]

        filtered = []
        for det in detections:
            if not is_face_label(det.label):
                filtered.append(det)
                continue
            if any(self.face_matches_person(det, person) for person in people):
                filtered.append(det)
        return filtered

    def face_matches_person(self, face, person) -> bool:
        if not center_inside(face.center, person.bbox):
            return False
        px, py, pw, ph = person.bbox
        fx, fy, fw, fh = face.bbox
        if ph <= 0:
            return False
        cx, cy = face.center
        height_ratio = fh / ph
        if height_ratio < self.face_config.min_person_height_ratio or height_ratio > self.face_config.max_person_height_ratio:
            return False
        person_center_x = px + pw / 2.0
        max_x_offset = pw * 0.60
        return abs(cx - person_center_x) <= max_x_offset

    def update_unknown_face_groups(self, frame: np.ndarray, detections: list) -> None:
        now = time.monotonic()
        faces = [det for det in detections if is_face_label(det.label)]
        for face in faces:
            if face.identity and face.identity != self.recognition_config.unknown_label:
                continue
            crop = self.crop_bbox(frame, face.bbox)
            if crop is None:
                continue
            hist = self.face_sample_histogram(crop)
            group_id = self.match_or_create_unknown_face_group(face, hist, now)
            face.face_group_id = group_id
            vault_group_id = self.save_unknown_face_sample(group_id, crop, face, now)
            if vault_group_id:
                face.vault_face_group_id = vault_group_id
        self.prune_unknown_face_groups(now)

    def match_or_create_unknown_face_group(self, face, hist: np.ndarray, now: float) -> str:
        best_id = None
        best_score = float("inf")
        with self.unknown_face_lock:
            for group_id, group in self.unknown_face_groups.items():
                if group.get("promoted"):
                    continue
                hist_score = self.hist_distance(hist, group["hist"])
                iou_score = 1.0 - bbox_iou(face.bbox, group.get("last_bbox", face.bbox))
                score = min(hist_score, iou_score * 0.7)
                if score < best_score:
                    best_score = score
                    best_id = group_id
            if best_id is None or best_score > self.face_config.unknown_match_threshold:
                best_id = f"unknown_{self.unknown_face_next_id:04d}"
                self.unknown_face_next_id += 1
                self.unknown_face_groups[best_id] = {
                    "id": best_id,
                    "hist": hist,
                    "first_seen": now,
                    "last_seen": now,
                    "last_sample_at": 0.0,
                    "last_bbox": list(face.bbox),
                    "sample_paths": [],
                    "seen_count": 0,
                    "promoted": False,
                }
            group = self.unknown_face_groups[best_id]
            group["hist"] = group["hist"] * 0.85 + hist * 0.15
            group["last_seen"] = now
            group["last_bbox"] = list(face.bbox)
            group["seen_count"] = int(group.get("seen_count", 0)) + 1
        return best_id

    def save_unknown_face_sample(self, group_id: str, crop: np.ndarray, face, now: float) -> str | None:
        with self.unknown_face_lock:
            group = self.unknown_face_groups.get(group_id)
            if not group or group.get("promoted"):
                return None
            sample_paths = group.setdefault("sample_paths", [])
            if len(sample_paths) >= self.face_config.unknown_max_samples:
                return group.get("vault_group_id")
            if now - float(group.get("last_sample_at", 0.0)) < self.face_config.unknown_sample_interval_seconds:
                return group.get("vault_group_id")
            group["last_sample_at"] = now
        group_dir = self.unknown_face_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        path = group_dir / f"{int(time.time() * 1000)}_{face.id or 'face'}.jpg"
        if cv2.imwrite(str(path), crop):
            with self.unknown_face_lock:
                group = self.unknown_face_groups.get(group_id)
                if group is not None:
                    group.setdefault("sample_paths", []).append(str(path))
        vault_group_id = self.upload_unknown_face_observation(group_id, crop, face, path)
        if vault_group_id:
            with self.unknown_face_lock:
                group = self.unknown_face_groups.get(group_id)
                if group is not None:
                    group["vault_group_id"] = vault_group_id
        return vault_group_id

    def upload_unknown_face_observation(self, group_id: str, crop: np.ndarray, face, path: Path) -> str | None:
        brain_client = self._vault_client_getter()
        if brain_client is None:
            return None
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return None
        try:
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception:
            return None
        result = brain_client.upload_unknown_face_observation({
            "node_id": "scout",
            "source_track_id": group_id,
            "face_id": face.id,
            "bbox": [int(v) for v in face.bbox],
            "confidence": float(face.confidence),
            "image_b64": image_b64,
            "filename": path.name,
            "observed_at": time.time(),
        })
        if result and result.get("ok") and result.get("group_id"):
            return str(result.get("group_id"))
        return None

    def unknown_face_group_ready(self, group_id: str) -> bool:
        with self.unknown_face_lock:
            group = self.unknown_face_groups.get(group_id)
            if not group or group.get("promoted"):
                return False
            return int(group.get("seen_count", 0)) >= self.face_config.intro_min_seen_frames

    def prune_unknown_face_groups(self, now: float) -> None:
        with self.unknown_face_lock:
            stale = [
                group_id for group_id, group in self.unknown_face_groups.items()
                if not group.get("sample_paths") and now - float(group.get("last_seen", 0.0)) > self.face_config.unknown_persist_seconds
            ]
            for group_id in stale:
                self.unknown_face_groups.pop(group_id, None)

    def promote_unknown_face_group(self, group_id: str | None, identity: str, recognizer) -> dict:
        brain_client = self._vault_client_getter()
        if brain_client is not None and group_id:
            result = brain_client.promote_unknown_face_group(group_id, identity)
            if result and result.get("ok"):
                brain_client.sync_face_references_if_due(force=True)
                if recognizer is not None:
                    recognizer.reload_from_disk()
                return {
                    "promoted": bool(result.get("promoted")),
                    "sample_count": result.get("sample_count", 0),
                    "group_id": result.get("group_id", group_id),
                    "destination": result.get("destination"),
                    "source": "vault",
                }
        if not group_id or recognizer is None:
            return {"promoted": False, "sample_count": 0}
        with self.unknown_face_lock:
            group = self.unknown_face_groups.get(group_id)
            if not group:
                return {"promoted": False, "sample_count": 0}
            sample_paths = [Path(path) for path in group.get("sample_paths", [])]
            group["promoted"] = True
        safe_identity = self.safe_identity_name(identity)
        if not safe_identity:
            return {"promoted": False, "sample_count": 0}
        destination = recognizer.known_faces_dir / safe_identity / "_unknown" / group_id
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for sample in sample_paths:
            if not sample.exists():
                continue
            target = destination / sample.name
            try:
                shutil.copy2(sample, target)
                copied += 1
            except Exception:
                pass
        if copied:
            recognizer.reload_from_disk()
        return {"promoted": copied > 0, "sample_count": copied, "group_id": group_id, "destination": str(destination)}

    def unknown_face_group_status(self) -> list[dict]:
        now = time.monotonic()
        with self.unknown_face_lock:
            groups = list(self.unknown_face_groups.values())
        result = []
        for group in groups:
            if group.get("promoted"):
                continue
            result.append({
                "id": group["id"],
                "seen_count": group.get("seen_count", 0),
                "sample_count": len(group.get("sample_paths", [])),
                "age_seconds": round(now - float(group.get("first_seen", now)), 1),
                "last_seen_seconds_ago": round(now - float(group.get("last_seen", now)), 1),
                "last_bbox": group.get("last_bbox"),
                "ready": int(group.get("seen_count", 0)) >= self.face_config.intro_min_seen_frames,
            })
        return result

    def target_face_detection(self, detections, face_id: int | None):
        if face_id is None:
            return None
        for det in detections:
            if det.id == face_id and is_face_label(det.label):
                return det
        return None

    def learn_identity_from_active_prompt(self, message: str, get_latest_frame_detections, recognizer_getter, recognizer_lock) -> dict | None:
        with self.identity_prompt_lock:
            prompt = dict(self.latest_identity_prompt) if self.latest_identity_prompt and not self.latest_identity_prompt.get("completed") else None
        if not prompt:
            return None
        try:
            face_id = int(prompt.get("face_id"))
        except (TypeError, ValueError):
            return None
        group_id = prompt.get("face_group_id")
        vault_group_id = prompt.get("vault_face_group_id")
        if self.is_identity_intro_refusal(message):
            self.skip_identity_prompt_face(face_id, "refused", str(group_id) if group_id else None)
            return {
                "ok": True,
                "response": {
                    "message": "Okay, I will skip that face.",
                    "face_id": face_id,
                    "face_group_id": group_id,
                    "vault_face_group_id": vault_group_id,
                    "skipped": True,
                },
            }
        identity = self.identity_from_chat_reply(message)
        if not identity:
            return {
                "ok": True,
                "response": {
                    "message": "What name should I use for that face? You can also say skip.",
                    "face_id": face_id,
                    "face_group_id": group_id,
                    "vault_face_group_id": vault_group_id,
                    "waiting_for_identity": True,
                },
            }
        frame, detections = get_latest_frame_detections()
        if frame is None:
            return {"ok": False, "error": "no_frame_available"}
        face = self.target_face_detection(detections, face_id)
        if face is None:
            with recognizer_lock:
                recognizer = recognizer_getter()
                if recognizer is None:
                    return {"ok": False, "error": "recognizer_not_ready"}
                promotion = self.promote_unknown_face_group(
                    str(vault_group_id or group_id) if (vault_group_id or group_id) else None,
                    identity,
                    recognizer,
                )
                if promotion.get("promoted"):
                    learned_identity = self.safe_identity_name(identity) or identity
                    self.mark_identity_prompt_learned(face_id, learned_identity)
                    return {
                        "ok": True,
                        "response": {
                            "message": f"Nice to meet you, {learned_identity}.",
                            "face_id": face_id,
                            "face_group_id": group_id,
                            "vault_face_group_id": vault_group_id,
                            "unknown_group_promotion": promotion,
                        },
                    }
            available = [det.id for det in detections if is_face_label(det.label) and det.id is not None]
            return {
                "ok": False,
                "error": "target_face_not_available",
                "available_face_ids": available,
                "response": {"message": "I lost that face and do not have enough saved samples. Look at me again and I will ask once it is visible."},
            }
        with recognizer_lock:
            recognizer = recognizer_getter()
            if recognizer is None:
                return {"ok": False, "error": "recognizer_not_ready"}
            result = recognizer.enroll_face(identity, frame, face)
            if result.get("ok"):
                learned_identity = str(result.get("identity") or identity)
                self.mark_identity_prompt_learned(face_id, learned_identity)
                promote_group_id = (
                    getattr(face, "vault_face_group_id", None)
                    or vault_group_id
                    or getattr(face, "face_group_id", None)
                    or (str(group_id) if group_id else None)
                )
                promotion = self.promote_unknown_face_group(
                    promote_group_id,
                    learned_identity,
                    recognizer,
                )
                result["unknown_group_promotion"] = promotion
                if promotion.get("source") == "vault":
                    vault_group_id = promotion.get("group_id") or promote_group_id
                brain_client = self._vault_client_getter()
                if brain_client is not None:
                    brain_client.upload_face_reference(result)
                return {
                    "ok": True,
                    "response": {
                        "message": f"Nice to meet you, {learned_identity}.",
                        "learned_face": result,
                        "face_id": face_id,
                        "face_group_id": group_id,
                        "vault_face_group_id": vault_group_id,
                    },
                }
        return {"ok": False, "error": result.get("error", "learn_failed"), "learned_face": result}

    def skip_identity_prompt_face(self, face_id: int | None, reason: str = "skipped", group_id: str | None = None) -> None:
        if face_id is None and group_id is None:
            return
        with self.identity_prompt_lock:
            if face_id is not None:
                self.identity_prompt_completed_face_ids[face_id] = time.monotonic()
            if group_id:
                self.identity_prompt_completed_face_ids[group_id] = time.monotonic()
            if self.identity_prompt_active_face_id == face_id:
                self.identity_prompt_active_face_id = None
            if self.latest_identity_prompt and (
                self.latest_identity_prompt.get("face_id") == face_id
                or (group_id and self.latest_identity_prompt.get("face_group_id") == group_id)
            ):
                self.latest_identity_prompt = {
                    **self.latest_identity_prompt,
                    "skipped": True,
                    "skip_reason": reason,
                    "completed": True,
                    "completed_ts": time.time(),
                }

    @staticmethod
    def bbox_area(bbox: list[int]) -> int:
        return max(0, int(bbox[2])) * max(0, int(bbox[3]))

    @staticmethod
    def crop_bbox(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        if w <= 1 or h <= 1:
            return None
        return frame[y:y + h, x:x + w].copy()

    @staticmethod
    def face_sample_histogram(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten().astype(np.float32)
        hist /= max(float(np.linalg.norm(hist)), 1e-6)
        return hist

    @staticmethod
    def hist_distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(1.0 - np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-6))

    @staticmethod
    def safe_identity_name(identity: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(identity).strip()).strip("._-")

    @staticmethod
    def identity_from_chat_reply(message: str) -> str | None:
        text = " ".join(str(message).strip().strip("\"'").split())
        if not text or "?" in text or len(text) > 80:
            return None
        lowered = text.casefold()
        for prefix in ("i am ", "i'm ", "im ", "my name is ", "this is ", "it's ", "its "):
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        text = text.strip(" .,!;:")
        if not text or len(text) > 64:
            return None
        return text

    @staticmethod
    def is_identity_intro_refusal(message: str) -> bool:
        text = " ".join(str(message).strip().casefold().split()).strip(" .,!;:")
        if not text:
            return False
        refusals = {
            "no",
            "nope",
            "nah",
            "skip",
            "pass",
            "not now",
            "don't know",
            "i don't know",
            "i dont know",
            "unknown",
            "they don't want to say",
            "they dont want to say",
            "don't learn this face",
            "dont learn this face",
        }
        if text in refusals:
            return True
        return text.startswith("skip ") or text.startswith("no ")
