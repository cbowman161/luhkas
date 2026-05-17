"""Camera-node Hailo inference setup helpers."""
from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)

DEFAULT_HEF_CANDIDATES = [
    "/usr/share/hailo-models/yolov8s_h8l.hef",
    "/usr/share/hailo-models/yolov6n_h8l.hef",
    "/usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef",
    "/usr/share/hailo-models/yolov8s_h8.hef",
    "/usr/share/hailo-models/yolov6n_h8.hef",
    "/usr/share/hailo-models/yolov8m_h10.hef",
    "/usr/share/hailo-models/yolov11m_h10.hef",
    "/usr/share/hailo-models/objects_h10.hef",
]

DEFAULT_POSE_HEF_CANDIDATES = [
    "/usr/share/hailo-models/yolov8m_pose_h10.hef",
    "/usr/share/hailo-models/yolov8s_pose_h10.hef",
    "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef",
    "/usr/share/hailo-models/yolov8s_pose_h8.hef",
]


def load_hailo(hef_path: str, hailo_apps_dir: str):
    sys.path.insert(0, hailo_apps_dir)
    from hailo_apps.python.core.common.hailo_inference import HailoInfer

    return HailoInfer(hef_path, batch_size=1)


def resolve_hailo(hef_path: str | None, hailo_apps_dir: str):
    candidates = [hef_path] if hef_path else DEFAULT_HEF_CANDIDATES
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        try:
            hailo = load_hailo(str(path), hailo_apps_dir)
            log.info("Using HEF: %s", path)
            return hailo, str(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            log.warning("Skipping incompatible HEF %s: %s", path, exc)
    detail = "\n".join(errors[-5:]) if errors else "No candidate HEF files were found."
    raise RuntimeError(f"Could not load a compatible HEF.\n{detail}")


def resolve_pose_estimator(args, vision_config, pose_estimator_cls):
    if args.no_pose or not vision_config.pose_enabled:
        return None

    candidates = [args.pose_hef_path] if args.pose_hef_path else DEFAULT_POSE_HEF_CANDIDATES
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        try:
            pose = pose_estimator_cls(
                str(path),
                args.hailo_apps_dir,
                score_threshold=vision_config.pose_score_threshold,
                joint_threshold=vision_config.pose_joint_threshold,
            )
            log.info("Using pose HEF: %s", path)
            return pose
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            log.warning("Skipping incompatible pose HEF %s: %s", path, exc)
    if errors:
        log.warning("Pose estimation disabled. Could not load compatible pose HEF: %s", errors[-1])
    else:
        log.warning("Pose estimation disabled. No pose HEF candidates found.")
    return None


def default_labels_path(hailo_apps_dir: str) -> str:
    return str(Path(hailo_apps_dir).expanduser() / "local_resources" / "coco.txt")


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def inference_callback(completion_info, bindings_list: list, output_queue: queue.Queue) -> None:
    if completion_info.exception:
        output_queue.put(completion_info.exception)
        return

    for bindings in bindings_list:
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                for name in bindings._output_names
            }
        output_queue.put(result)
