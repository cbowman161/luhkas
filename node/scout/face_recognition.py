from __future__ import annotations

import logging
import json
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from .config import FaceDetectionConfig, FaceRecognitionConfig
from .face_detection import FaceDetector
from .person_memory import _safe_identity
from .types import Detection
from .vision import is_face_label

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REFERENCE_MANIFEST = ".reference_poses.json"


class FaceRecognizer:
    def __init__(
        self,
        config: FaceRecognitionConfig | None = None,
        detector_config: FaceDetectionConfig | None = None,
    ) -> None:
        self.config = config or FaceRecognitionConfig()
        self.enabled = bool(self.config.enabled)
        self.known_faces_dir = Path(self.config.known_faces_dir).expanduser()
        self.detector_config = detector_config or FaceDetectionConfig()
        self.method = "none"
        self._labels_by_id: dict[int, str] = {}
        self._histograms_by_label: dict[str, list[np.ndarray]] = {}
        self._reference_coverage: dict[str, dict[str, int]] = {}
        self._last_auto_capture_at: dict[tuple[str, str], float] = {}
        self._recognizer = None
        self._pending_retrain: int = 0
        self._retrain_batch_size: int = 5

        if not self.enabled:
            return
        self._train()

    def recognize_faces(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if not self.config.enabled:
            return detections

        for det in detections:
            if not is_face_label(det.label):
                continue
            face = _normalized_face(frame, det.bbox, self.config.image_size_px)
            if face is None:
                continue
            if self.enabled and self.method != "none":
                identity, confidence, distance = self._predict(face)
            else:
                identity, confidence, distance = self.config.unknown_label, 0.0, -1.0
            det.identity = identity
            det.identity_confidence = confidence
            det.recognition_distance = distance
            det.recognition_method = self.method
            det.reference_pose = _reference_pose_for_bbox(frame, det.bbox)
            if identity != self.config.unknown_label:
                det.missing_reference_poses = self.missing_reference_poses(identity)
        return detections

    def reload_from_disk(self) -> None:
        if not self.config.enabled:
            return
        self.enabled = True
        self._train()

    def auto_capture_missing_references(self, frame: np.ndarray, detections: list[Detection]) -> list[dict]:
        if not self.enabled or not self.config.auto_reference_capture_enabled:
            return []

        captures = []
        changed = False
        for det in detections:
            if not is_face_label(det.label):
                continue
            if not det.identity or det.identity == self.config.unknown_label:
                continue
            confidence = float(det.identity_confidence or 0.0)
            if confidence < self.config.auto_reference_min_confidence:
                continue

            pose = det.reference_pose or _reference_pose_for_bbox(frame, det.bbox)
            if not self.needs_reference_pose(det.identity, pose):
                continue
            if not self._auto_capture_allowed(det.identity, pose):
                continue

            result = self._save_reference_sample(det.identity, frame, det.bbox, pose, auto=True)
            if result.get("ok"):
                captures.append(result)
                self._pending_retrain += 1

        if self._pending_retrain >= self._retrain_batch_size:
            self._train()
            self._pending_retrain = 0
        return captures

    def enroll_face(self, identity: str, frame: np.ndarray, face: Detection) -> dict:
        if not self.config.enabled:
            return {"ok": False, "error": "face_recognition_disabled"}

        identity = _safe_identity(identity)
        if not identity:
            return {"ok": False, "error": "missing_identity"}

        if not is_face_label(face.label):
            return {"ok": False, "error": "target_is_not_face"}

        crop = _face_crop(frame, face.bbox)
        if crop is None:
            return {"ok": False, "error": "invalid_face_crop"}

        person_dir = self.known_faces_dir / identity
        person_dir.mkdir(parents=True, exist_ok=True)
        image_path = person_dir / f"{int(time.time() * 1000)}.jpg"
        if not cv2.imwrite(str(image_path), crop):
            return {"ok": False, "error": "write_failed"}
        encoded_ok, encoded_crop = cv2.imencode(".jpg", crop)
        pose = _reference_pose_for_bbox(frame, face.bbox)
        self._record_reference_sample(identity, image_path, pose, auto=False)

        self.enabled = True
        self._train()
        sample_count = len([
            path for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ])
        return {
            "ok": True,
            "identity": identity,
            "saved_path": str(image_path),
            "filename": image_path.name,
            "image_b64": base64.b64encode(encoded_crop.tobytes()).decode("ascii") if encoded_ok else None,
            "sample_count": sample_count,
            "trained": bool(self.enabled and self.method != "none"),
            "method": self.method,
            "needed_samples": max(0, self.config.min_training_images_per_person - sample_count),
            "reference_pose": pose,
            "missing_reference_poses": self.missing_reference_poses(identity),
        }

    def reference_pose_coverage(self) -> dict:
        buckets = self._reference_buckets()
        coverage = {}
        for person_dir in sorted(path for path in self.known_faces_dir.iterdir() if path.is_dir()) if self.known_faces_dir.exists() else []:
            identity = person_dir.name
            counts = self._coverage_for_identity(identity)
            coverage[identity] = {
                "counts": counts,
                "missing": [bucket for bucket in buckets if counts.get(bucket, 0) < self.config.reference_samples_per_pose],
                "samples_per_pose": self.config.reference_samples_per_pose,
            }
        return coverage

    def missing_reference_poses(self, identity: str) -> list[str]:
        counts = self._coverage_for_identity(identity)
        return [
            bucket for bucket in self._reference_buckets()
            if counts.get(bucket, 0) < self.config.reference_samples_per_pose
        ]

    def needs_reference_pose(self, identity: str, pose: str) -> bool:
        if pose not in self._reference_buckets():
            return False
        counts = self._coverage_for_identity(identity)
        return counts.get(pose, 0) < self.config.reference_samples_per_pose

    def _train(self) -> None:
        self._reset_model()
        if not self.known_faces_dir.exists():
            log.warning("Face recognition disabled. Known faces directory does not exist: %s", self.known_faces_dir)
            self.enabled = False
            return

        detector = FaceDetector(self.detector_config) if self.config.crop_training_faces else None
        labels: list[int] = []
        faces: list[np.ndarray] = []
        label_names: list[str] = []

        for label_id, person_dir in enumerate(sorted(path for path in self.known_faces_dir.iterdir() if path.is_dir())):
            self._sync_reference_manifest(person_dir)
            person_faces = self._load_person_faces(person_dir, detector)
            if len(person_faces) < self.config.min_training_images_per_person:
                log.warning(
                    "Skipping %s. Need at least %s training images, found %s.",
                    person_dir.name,
                    self.config.min_training_images_per_person,
                    len(person_faces),
                )
                continue
            self._labels_by_id[len(label_names)] = person_dir.name
            label_names.append(person_dir.name)
            for face in person_faces:
                labels.append(len(label_names) - 1)
                faces.append(face)
                self._histograms_by_label.setdefault(person_dir.name, []).append(_face_histogram(face))

        if not faces:
            log.warning("Face recognition disabled. No usable known-face images were found in %s.", self.known_faces_dir)
            self.enabled = False
            return

        factory = getattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None)
        if factory is not None:
            self._recognizer = factory()
            self._recognizer.train(faces, np.array(labels, dtype=np.int32))
            self.method = "lbph"
        else:
            self.method = "histogram"

        log.info("Face recognition enabled with %s known people using %s.", len(label_names), self.method)

    def _reset_model(self) -> None:
        self.method = "none"
        self._labels_by_id = {}
        self._histograms_by_label = {}
        self._reference_coverage = {}
        self._recognizer = None

    def _load_person_faces(self, person_dir: Path, detector: FaceDetector | None) -> list[np.ndarray]:
        faces: list[np.ndarray] = []
        image_paths = sorted(
            path for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                log.warning("Skipping unreadable known-face image: %s", image_path)
                continue

            bbox = None
            if detector is not None and detector.enabled:
                detected = detector.detect(image)
                if detected:
                    bbox = max(detected, key=lambda det: det.bbox[2] * det.bbox[3]).bbox

            face = _normalized_face(image, bbox, self.config.image_size_px)
            if face is not None:
                faces.append(face)
        return faces

    def _save_reference_sample(self, identity: str, frame: np.ndarray, bbox: list[int], pose: str, auto: bool) -> dict:
        crop = _face_crop(frame, bbox)
        if crop is None:
            return {"ok": False, "error": "invalid_face_crop", "identity": identity, "reference_pose": pose}

        person_dir = self.known_faces_dir / identity
        sample_dir = person_dir / "_auto" / pose if auto else person_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / f"{int(time.time() * 1000)}.jpg"
        if not cv2.imwrite(str(image_path), crop):
            return {"ok": False, "error": "write_failed", "identity": identity, "reference_pose": pose}

        self._record_reference_sample(identity, image_path, pose, auto=auto)
        if auto:
            self._last_auto_capture_at[(identity, pose)] = time.time()
        return {
            "ok": True,
            "identity": identity,
            "reference_pose": pose,
            "saved_path": str(image_path),
            "auto": auto,
            "missing_reference_poses": self.missing_reference_poses(identity),
        }

    def _record_reference_sample(self, identity: str, image_path: Path, pose: str, auto: bool) -> None:
        person_dir = self.known_faces_dir / identity
        manifest = _read_manifest(person_dir)
        rel_path = str(image_path.relative_to(person_dir)) if image_path.is_relative_to(person_dir) else str(image_path)
        samples = [
            sample for sample in manifest.get("samples", [])
            if sample.get("path") != rel_path
        ]
        samples.append({
            "path": rel_path,
            "reference_pose": pose,
            "auto": bool(auto),
            "captured_at": time.time(),
        })
        manifest["identity"] = identity
        manifest["samples"] = samples
        _write_manifest(person_dir, manifest)
        self._reference_coverage[identity] = self._counts_from_manifest(manifest)

    def _sync_reference_manifest(self, person_dir: Path) -> None:
        manifest = _read_manifest(person_dir)
        known_paths = {sample.get("path") for sample in manifest.get("samples", [])}
        samples = list(manifest.get("samples", []))
        changed = False
        for image_path in sorted(path for path in person_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            rel_path = str(image_path.relative_to(person_dir))
            if rel_path in known_paths:
                continue
            samples.append({
                "path": rel_path,
                "reference_pose": _reference_pose_from_path(image_path),
                "auto": "_auto" in image_path.parts,
                "captured_at": image_path.stat().st_mtime,
            })
            changed = True
        manifest["identity"] = person_dir.name
        manifest["samples"] = samples
        if changed:
            _write_manifest(person_dir, manifest)
        self._reference_coverage[person_dir.name] = self._counts_from_manifest(manifest)

    def _coverage_for_identity(self, identity: str) -> dict[str, int]:
        if identity not in self._reference_coverage:
            person_dir = self.known_faces_dir / identity
            if person_dir.exists():
                self._sync_reference_manifest(person_dir)
        counts = {bucket: 0 for bucket in self._reference_buckets()}
        counts.update(self._reference_coverage.get(identity, {}))
        return counts

    def _counts_from_manifest(self, manifest: dict) -> dict[str, int]:
        counts = {bucket: 0 for bucket in self._reference_buckets()}
        for sample in manifest.get("samples", []):
            pose = sample.get("reference_pose")
            if pose in counts:
                counts[pose] += 1
        return counts

    def _reference_buckets(self) -> list[str]:
        buckets = [bucket.strip() for bucket in self.config.reference_pose_buckets.split(",") if bucket.strip()]
        return buckets or ["frontal", "left", "right", "up", "down", "close", "far"]

    def _auto_capture_allowed(self, identity: str, pose: str) -> bool:
        now = time.time()
        last = self._last_auto_capture_at.get((identity, pose), 0.0)
        if now - last < self.config.auto_reference_cooldown_seconds:
            return False
        person_dir = self.known_faces_dir / identity / "_auto"
        auto_count = len([
            path for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]) if person_dir.exists() else 0
        return auto_count < self.config.max_auto_reference_samples_per_identity

    def _predict(self, face: np.ndarray) -> tuple[str, float, float]:
        if self.method == "lbph" and self._recognizer is not None:
            label_id, distance = self._recognizer.predict(face)
            name = self._labels_by_id.get(int(label_id), self.config.unknown_label)
            if distance > self.config.lbph_threshold:
                return self.config.unknown_label, 0.0, float(distance)
            confidence = 1.0 - min(float(distance) / max(self.config.lbph_threshold, 1.0), 1.0)
            return name, confidence, float(distance)

        histogram = _face_histogram(face)
        best_name = self.config.unknown_label
        best_score = -1.0
        for name, known_histograms in self._histograms_by_label.items():
            score = max(float(cv2.compareHist(histogram, known, cv2.HISTCMP_CORREL)) for known in known_histograms)
            if score > best_score:
                best_name = name
                best_score = score

        if best_score < self.config.histogram_threshold:
            return self.config.unknown_label, 0.0, best_score
        confidence = min(max((best_score - self.config.histogram_threshold) / (1.0 - self.config.histogram_threshold), 0.0), 1.0)
        return best_name, confidence, best_score


def _normalized_face(frame: np.ndarray, bbox: list[int] | None, size: int) -> np.ndarray | None:
    if bbox is None:
        crop = frame
    else:
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.22)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_frame, x + w + pad_x)
        y2 = min(h_frame, y + h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def _face_crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]
    pad_x = int(w * 0.18)
    pad_y = int(h * 0.22)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_frame, x + w + pad_x)
    y2 = min(h_frame, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _reference_pose_for_bbox(frame: np.ndarray, bbox: list[int]) -> str:
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]
    cx = (x + w / 2.0) / max(frame_w, 1)
    cy = (y + h / 2.0) / max(frame_h, 1)
    face_area = (w * h) / max(frame_w * frame_h, 1)

    if face_area >= 0.13:
        return "close"
    if face_area <= 0.025:
        return "far"
    if cx < 0.38:
        return "left"
    if cx > 0.62:
        return "right"
    if cy < 0.38:
        return "up"
    if cy > 0.62:
        return "down"
    return "frontal"


def _reference_pose_from_path(image_path: Path) -> str:
    parts = {part.lower() for part in image_path.parts}
    for pose in ("frontal", "left", "right", "up", "down", "close", "far"):
        if pose in parts or pose in image_path.stem.lower():
            return pose
    return "frontal"


def _face_histogram(face: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([face], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def _read_manifest(person_dir: Path) -> dict:
    path = person_dir / REFERENCE_MANIFEST
    if not path.exists():
        return {"identity": person_dir.name, "samples": []}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"identity": person_dir.name, "samples": []}
    return data if isinstance(data, dict) else {"identity": person_dir.name, "samples": []}


def _write_manifest(person_dir: Path, manifest: dict) -> None:
    person_dir.mkdir(parents=True, exist_ok=True)
    path = person_dir / REFERENCE_MANIFEST
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
