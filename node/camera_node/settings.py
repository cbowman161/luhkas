"""Camera-node runtime settings mutation helpers."""
from __future__ import annotations


def apply_settings(body: dict, vision_config, face_detector, face_recognizer) -> None:
    if "face_detection_enabled" in body and face_detector:
        face_detector.enabled = bool(body["face_detection_enabled"])
    if "face_recognition_enabled" in body and face_recognizer:
        enabled = bool(body["face_recognition_enabled"])
        face_recognizer.config.enabled = enabled
        if enabled:
            face_recognizer.reload_from_disk()
        else:
            face_recognizer.enabled = False
    if "auto_reference_capture_enabled" in body and face_recognizer:
        face_recognizer.config.auto_reference_capture_enabled = bool(body["auto_reference_capture_enabled"])
    if "auto_reference_min_confidence" in body and face_recognizer:
        face_recognizer.config.auto_reference_min_confidence = float(body["auto_reference_min_confidence"])

    if "pose_enabled" in body:
        vision_config.pose_enabled = bool(body["pose_enabled"])
    if "pose_filter_persons" in body:
        vision_config.pose_filter_persons = bool(body["pose_filter_persons"])
    if "pose_interval_frames" in body:
        vision_config.pose_interval_frames = int(body["pose_interval_frames"])
    if "pose_score_threshold" in body:
        vision_config.pose_score_threshold = float(body["pose_score_threshold"])
    if "jpeg_quality" in body:
        vision_config.jpeg_quality = int(body["jpeg_quality"])
