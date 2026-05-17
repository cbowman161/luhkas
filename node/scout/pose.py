from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import Detection
from .vision import bbox_iou


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

JOINT_PAIRS = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
]


@dataclass
class PoseDetection:
    bbox: list[int]
    score: float
    keypoints: list[tuple[float, float]]
    joint_scores: list[float]

    def to_json(self) -> dict:
        return {
            "bbox": [int(v) for v in self.bbox],
            "score": float(self.score),
            "keypoints": [
                {
                    "name": KEYPOINT_NAMES[index],
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "score": float(self.joint_scores[index]),
                }
                for index, point in enumerate(self.keypoints)
            ],
        }


class PoseEstimator:
    def __init__(self, hef_path: str, hailo_apps_dir: str, score_threshold: float, joint_threshold: float) -> None:
        import sys

        sys.path.insert(0, hailo_apps_dir)
        from hailo_apps.python.core.common.hailo_inference import HailoInfer
        from hailo_apps.python.standalone_apps.pose_estimation.pose_estimation_utils import PoseEstPostProcessing

        self.hef_path = str(Path(hef_path).expanduser())
        self.score_threshold = score_threshold
        self.joint_threshold = joint_threshold
        self.hailo = HailoInfer(self.hef_path, batch_size=1, output_type="FLOAT32")
        self.processor = PoseEstPostProcessing(
            max_detections=300,
            score_threshold=0.001,
            nms_iou_thresh=0.7,
            regression_length=15,
            strides=[8, 16, 32],
        )
        input_h, input_w, _ = self.hailo.get_input_shape()
        self.input_w = int(input_w)
        self.input_h = int(input_h)

    def run(self, frame: np.ndarray, inference_callback, output_queue) -> list[PoseDetection]:
        job = self.hailo.run([frame], inference_callback)
        job.wait(10000)
        result = output_queue.get(timeout=1.0)
        if isinstance(result, Exception):
            raise result
        return self.parse(result, frame.shape[1], frame.shape[0])

    def parse(self, raw: Any, frame_w: int, frame_h: int) -> list[PoseDetection]:
        if not isinstance(raw, dict):
            return []

        results = self.processor.post_process(raw, self.input_h, self.input_w, class_num=1)
        bboxes = results["bboxes"][0]
        scores = results["scores"][0]
        keypoints = results["keypoints"][0]
        joint_scores = results["joint_scores"][0]

        poses: list[PoseDetection] = []
        for bbox, score_arr, kpts, joints in zip(bboxes, scores, keypoints, joint_scores):
            score = float(np.asarray(score_arr).reshape(-1)[0])
            if score < self.score_threshold:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            x1 = max(0, min(frame_w - 1, x1))
            x2 = max(0, min(frame_w - 1, x2))
            y1 = max(0, min(frame_h - 1, y1))
            y2 = max(0, min(frame_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            points = [(float(point[0]), float(point[1])) for point in kpts.reshape(17, 2)]
            joint_values = [float(value) for value in np.asarray(joints).reshape(17)]
            poses.append(PoseDetection(
                bbox=[x1, y1, x2 - x1, y2 - y1],
                score=score,
                keypoints=points,
                joint_scores=joint_values,
            ))
        return poses

    def close(self) -> None:
        self.hailo.close()


def draw_poses(frame: np.ndarray, poses: list[PoseDetection], joint_threshold: float) -> np.ndarray:
    out = frame
    for pose in poses:
        x, y, w, h = pose.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 180), 1)
        for index, point in enumerate(pose.keypoints):
            if pose.joint_scores[index] < joint_threshold:
                continue
            cv2.circle(out, (int(point[0]), int(point[1])), 2, (255, 0, 180), -1)
        for a, b in JOINT_PAIRS:
            if pose.joint_scores[a] < joint_threshold or pose.joint_scores[b] < joint_threshold:
                continue
            pa = pose.keypoints[a]
            pb = pose.keypoints[b]
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 0, 180), 1)
    return out


def apply_pose_aim(target: Detection | None, poses: list[PoseDetection], joint_threshold: float) -> Detection | None:
    if target is None or target.label != "person" or not poses:
        return target

    # Face detection gives a more precise aim point than pose keypoints.
    if target.aim_source.startswith("face:"):
        return target

    pose = max(poses, key=lambda candidate: bbox_iou(target.bbox, candidate.bbox))
    if bbox_iou(target.bbox, pose.bbox) < 0.05:
        return target

    aim = _head_or_shoulders_aim(pose, joint_threshold)
    if aim is None:
        return target

    target.aim_x, target.aim_y = aim
    target.aim_source = "pose"
    return target


def _head_or_shoulders_aim(pose: PoseDetection, joint_threshold: float) -> tuple[float, float] | None:
    head_indexes = [0, 1, 2, 3, 4]
    visible_head = [
        pose.keypoints[index]
        for index in head_indexes
        if pose.joint_scores[index] >= joint_threshold
    ]
    if visible_head:
        return _average_points(visible_head)

    shoulder_indexes = [5, 6]
    visible_shoulders = [
        pose.keypoints[index]
        for index in shoulder_indexes
        if pose.joint_scores[index] >= joint_threshold
    ]
    if visible_shoulders:
        sx, sy = _average_points(visible_shoulders)
        return sx, sy - pose.bbox[3] * 0.12
    return None


def _average_points(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


