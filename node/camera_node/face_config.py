"""Face detection & recognition config.

Lives in camera_node because without a camera, face recognition is irrelevant.
Previously lived under scout/ — that was a leftover from where face code first
landed; scout is mobility/tracking, not vision.

Env-var namespace is FACE_* (system-wide concern, no longer SCOUT_-prefixed).
"""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class FaceDetectionConfig:
    enabled: bool = os.environ.get("FACE_DETECTION_ENABLED", "1") != "0"
    cascade_path: str | None = os.environ.get("FACE_DETECTION_CASCADE_PATH") or None
    interval_frames: int = int(os.environ.get("FACE_DETECTION_INTERVAL_FRAMES", "2"))
    class_id: int = int(os.environ.get("FACE_DETECTION_CLASS_ID", "10000"))
    confidence: float = float(os.environ.get("FACE_DETECTION_CONFIDENCE", "0.85"))
    scale_factor: float = float(os.environ.get("FACE_DETECTION_SCALE_FACTOR", "1.05"))
    min_neighbors: int = int(os.environ.get("FACE_DETECTION_MIN_NEIGHBORS", "3"))
    min_size_px: int = int(os.environ.get("FACE_DETECTION_MIN_SIZE_PX", "32"))
    max_faces: int = int(os.environ.get("FACE_DETECTION_MAX_FACES", "3"))
    person_upper_ratio: float = float(os.environ.get("FACE_DETECTION_PERSON_UPPER_RATIO", "0.55"))
    min_person_height_ratio: float = float(os.environ.get("FACE_DETECTION_MIN_PERSON_HEIGHT_RATIO", "0.08"))
    max_person_height_ratio: float = float(os.environ.get("FACE_DETECTION_MAX_PERSON_HEIGHT_RATIO", "0.50"))
    intro_min_seen_frames: int = int(os.environ.get("FACE_DETECTION_INTRO_MIN_SEEN_FRAMES", "2"))
    unknown_match_threshold: float = float(os.environ.get("FACE_DETECTION_UNKNOWN_MATCH_THRESHOLD", "0.32"))
    unknown_sample_interval_seconds: float = float(os.environ.get("FACE_DETECTION_UNKNOWN_SAMPLE_INTERVAL", "2.0"))
    unknown_max_samples: int = int(os.environ.get("FACE_DETECTION_UNKNOWN_MAX_SAMPLES", "24"))
    unknown_persist_seconds: float = float(os.environ.get("FACE_DETECTION_UNKNOWN_PERSIST_SECONDS", "8.0"))


# Backend-internal scaling constants. These are algorithm-specific raw cutoffs
# used only inside the LBPH/histogram confidence math — not user-facing tuning.
# The public knob is FACE_RECOGNITION_MIN_CONFIDENCE (0-1, unified space).
_LBPH_DISTANCE_SCALE: float = 72.0      # LBPH raw distance at which confidence = 0
_HISTOGRAM_SCORE_FLOOR: float = 0.62    # Histogram raw score at which confidence = 0


@dataclass
class FaceRecognitionConfig:
    enabled: bool = os.environ.get("FACE_RECOGNITION_ENABLED", "1") != "0"
    known_faces_dir: str = os.environ.get("FACE_KNOWN_FACES_DIR", "config/faces")
    interval_frames: int = int(os.environ.get("FACE_RECOGNITION_INTERVAL_FRAMES", "2"))
    image_size_px: int = int(os.environ.get("FACE_RECOGNITION_IMAGE_SIZE", "128"))
    # Unified 0-1 minimum confidence to accept a recognition. Was previously
    # SCOUT_FACE_LBPH_THRESHOLD (raw distance scale) + SCOUT_FACE_HISTOGRAM_THRESHOLD
    # (raw score scale). Both backends now emit confidence in unified 0-1 space
    # using the internal _LBPH_DISTANCE_SCALE / _HISTOGRAM_SCORE_FLOOR constants;
    # this knob gates rejection in that space. Default 0.0 preserves prior behavior.
    min_confidence: float = float(os.environ.get("FACE_RECOGNITION_MIN_CONFIDENCE", "0.0"))
    crop_training_faces: bool = os.environ.get("FACE_RECOGNITION_CROP_TRAINING", "1") != "0"
    unknown_label: str = os.environ.get("FACE_RECOGNITION_UNKNOWN_LABEL", "unknown")
    min_training_images_per_person: int = int(os.environ.get("FACE_RECOGNITION_MIN_TRAINING_IMAGES", "2"))
    reference_pose_buckets: str = os.environ.get("FACE_RECOGNITION_REFERENCE_POSES", "frontal,left,right,up,down,close,far")
    reference_samples_per_pose: int = int(os.environ.get("FACE_RECOGNITION_REFERENCE_SAMPLES_PER_POSE", "3"))
    auto_reference_capture_enabled: bool = os.environ.get("FACE_RECOGNITION_AUTO_REFERENCE_CAPTURE", "1") != "0"
    # Eager training-data-capture gate. Distinct from min_confidence by design:
    # training wants permissive (collect more samples), recognition acceptance
    # wants strict (don't claim identity unless confident).
    auto_reference_min_confidence: float = float(os.environ.get("FACE_RECOGNITION_AUTO_REFERENCE_MIN_CONFIDENCE", "0.35"))
    auto_reference_cooldown_seconds: float = float(os.environ.get("FACE_RECOGNITION_AUTO_REFERENCE_COOLDOWN", "20"))
    max_auto_reference_samples_per_identity: int = int(os.environ.get("FACE_RECOGNITION_MAX_AUTO_REFERENCE_SAMPLES", "80"))
