from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from scout.types import Detection

from .face_config import FaceDetectionConfig

log = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, config: FaceDetectionConfig | None = None) -> None:
        self.config = config or FaceDetectionConfig()
        self.enabled = bool(self.config.enabled)
        self.cascade_path: Path | None = None
        self._classifier: cv2.CascadeClassifier | None = None

        if not self.enabled:
            return
        self.cascade_path = self._resolve_cascade_path(self.config.cascade_path)
        if self.cascade_path is None:
            log.warning("Face detection disabled. No OpenCV Haar cascade was found.")
            self.enabled = False
            return

        classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if classifier.empty():
            log.warning("Face detection disabled. Could not load cascade: %s", self.cascade_path)
            self.enabled = False
            return

        self._classifier = classifier
        log.info("Face detection enabled with cascade: %s", self.cascade_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self.enabled or self._classifier is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(self.config.min_size_px, self.config.min_size_px),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections: list[Detection] = []
        for x, y, w, h in faces[: self.config.max_faces]:
            detections.append(Detection(
                id=None,
                label="face",
                class_id=self.config.class_id,
                bbox=[int(x), int(y), int(w), int(h)],
                confidence=self.config.confidence,
                color="face",
                color_rgb=(255, 170, 80),
            ))
        return detections

    def detect_in_bbox(self, frame: np.ndarray, bbox: list[int]) -> list[Detection]:
        if not self.enabled or self._classifier is None:
            return []
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        if w <= 1 or h <= 1:
            return []
        roi = frame[y:y + h, x:x + w]
        detections = self.detect(roi)
        for det in detections:
            det.bbox[0] += x
            det.bbox[1] += y
        return detections

    @staticmethod
    def _resolve_cascade_path(configured_path: str | None) -> Path | None:
        candidates: list[Path] = []
        if configured_path:
            candidates.append(Path(configured_path).expanduser())

        cv2_data = getattr(cv2, "data", None)
        haar_dir = getattr(cv2_data, "haarcascades", None) if cv2_data is not None else None
        if haar_dir:
            candidates.extend([
                Path(haar_dir) / "haarcascade_frontalface_default.xml",
                Path(haar_dir) / "haarcascade_frontalface_alt2.xml",
            ])
        candidates.extend([
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml"),
            Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml"),
        ])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
