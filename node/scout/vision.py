from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 128, dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y = (height - new_h) // 2
    x = (width - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def parse_hailo_detections(
    results,
    frame: np.ndarray,
    labels: list[str],
    frame_w: int,
    frame_h: int,
    threshold: float,
) -> list[Detection]:
    detections: list[Detection] = []
    if not isinstance(results, list):
        return detections

    for class_id, class_dets in enumerate(results):
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        for det in class_dets:
            if len(det) < 5:
                continue
            y1, x1, y2, x2, score = [float(v) for v in det[:5]]
            if score < threshold:
                continue

            x = max(0, min(frame_w - 1, int(x1 * frame_w)))
            y = max(0, min(frame_h - 1, int(y1 * frame_h)))
            x_max = max(0, min(frame_w - 1, int(x2 * frame_w)))
            y_max = max(0, min(frame_h - 1, int(y2 * frame_h)))
            w = max(1, x_max - x)
            h = max(1, y_max - y)
            color_name, color_rgb = dominant_color(frame, [x, y, w, h])
            detections.append(Detection(
                id=None,
                label=label,
                class_id=class_id,
                bbox=[x, y, w, h],
                confidence=score,
                color=color_name,
                color_rgb=color_rgb,
            ))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def draw_detections(frame: np.ndarray, detections: list[Detection], target_id: int | None) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = (0, 140, 255) if det.predicted else (0, 220, 255) if det.id == target_id else tuple(int(c) for c in det.color_rgb)
        thickness = 1 if det.predicted else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = det.label
        if det.identity and det.identity != "unknown":
            label = f"{label}:{det.identity}"
        text = f"{label} {det.id} {det.confidence:.2f}"
        if det.predicted:
            text = f"{text} predicted"
        cv2.putText(out, text, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if det.id == target_id:
            ax, ay = det.aim_point
            cv2.circle(out, (int(ax), int(ay)), 5, color, -1)
    if target_id is not None:
        cv2.circle(out, (out.shape[1] // 2, out.shape[0] // 2), 5, (255, 0, 0), -1)
    return out


def id_color(class_id: int) -> tuple[int, int, int]:
    colors = [
        (80, 220, 80),
        (240, 120, 80),
        (80, 180, 255),
        (220, 120, 240),
        (240, 220, 80),
    ]
    return colors[class_id % len(colors)]


def is_face_label(label: str) -> bool:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return normalized in {"face", "face detect", "face detection"} or "face" in normalized


def center_inside(center: tuple[float, float], bbox: list[int]) -> bool:
    cx, cy = center
    x, y, w, h = bbox
    return x <= cx <= x + w and y <= cy <= y + h


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def dominant_color(frame: np.ndarray, bbox: list[int]) -> tuple[str, tuple[int, int, int]]:
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(w_frame - 1, x))
    y1 = max(0, min(h_frame - 1, y))
    x2 = max(0, min(w_frame, x + w))
    y2 = max(0, min(h_frame, y + h))
    if x2 <= x1 or y2 <= y1:
        return "unknown", (0, 255, 0)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown", (0, 255, 0)

    roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_bgr = np.mean(roi.reshape(-1, 3), axis=0).astype(int)
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

    avg_s = float(hsv[:, :, 1].mean())
    avg_v = float(hsv[:, :, 2].mean())
    if avg_s < 35:
        if avg_v < 55:
            return "black", avg_rgb
        if avg_v > 190:
            return "white", avg_rgb
        return "gray", avg_rgb

    hue = float(hsv[:, :, 0].mean())
    if hue < 10 or hue >= 170:
        name = "red"
    elif hue < 25:
        name = "orange"
    elif hue < 38:
        name = "yellow"
    elif hue < 85:
        name = "green"
    elif hue < 130:
        name = "blue"
    elif hue < 160:
        name = "purple"
    else:
        name = "red"
    return name, avg_rgb
