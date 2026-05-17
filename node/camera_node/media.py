"""Camera media persistence helpers for snapshots and buffered clips."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


StatusCallback = Callable[[str], None]


def save_snapshot(capture_dir: Path, frame: bytes | None, status: StatusCallback | None = None) -> dict:
    if frame is None:
        if status is not None:
            status("snapshot failed")
        return {"ok": False, "error": "no_frame_available"}
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    try:
        path.write_bytes(frame)
    except OSError as exc:
        if status is not None:
            status("snapshot failed")
        return {"ok": False, "error": str(exc)}
    if status is not None:
        status(f"snapshot {path.name}")
    return {
        "ok": True,
        "path": str(path),
        "filename": path.name,
        "content_type": "image/jpeg",
        "size": len(frame),
    }


def save_clip(
    capture_dir: Path,
    frame_history: Iterable[tuple[float, bytes]],
    seconds: float = 8.0,
    status: StatusCallback | None = None,
) -> dict:
    now = time.time()
    frames = [jpeg for ts, jpeg in frame_history if now - ts <= seconds]
    if len(frames) < 2:
        if status is not None:
            status("clip failed")
        return {"ok": False, "error": "not_enough_frames", "frame_count": len(frames), "duration": seconds}
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"clip-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    first = cv2.imdecode(np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        if status is not None:
            status("clip failed")
        return {"ok": False, "error": "decode_failed", "frame_count": len(frames), "duration": seconds}
    height, width = first.shape[:2]
    fps = max(6.0, min(20.0, len(frames) / seconds))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for jpeg in frames:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    if status is not None:
        status(f"clip {path.name}")
    return {
        "ok": True,
        "path": str(path),
        "filename": path.name,
        "duration": seconds,
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
    }
