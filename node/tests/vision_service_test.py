#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
import unittest
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2
except ImportError as exc:
    raise unittest.SkipTest("cv2 is not installed; calibration service test is unavailable") from exc
import numpy as np

from scout.vault_memory import BrainMemoryClient
from scout.collision import CollisionGuard
from scout.config import VaultMemoryConfig, CollisionConfig, GuardConfig, MotionConfig, PersonMemoryConfig, SearchConfig, TrackingConfig, VisionConfig
from camera_node.face_config import FaceDetectionConfig, FaceRecognitionConfig
from camera_node.face_detection import FaceDetector
from camera_node.face_recognition import FaceRecognizer
from scout.motion import PanTiltController
from scout.person_memory import PersonMemoryStore
from scout.pose import PoseEstimator, apply_pose_aim, draw_poses
from scout.robot_client import RobotClient
from scout.search import SearchController
from scout.tracking import SimpleTracker
from scout.vision import bbox_iou, center_inside, draw_detections, is_face_label, letterbox, parse_hailo_detections


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("vision_service")

vision_config = VisionConfig()
face_config = FaceDetectionConfig()
recognition_config = FaceRecognitionConfig()
vault_memory_config = VaultMemoryConfig()
person_memory_config = PersonMemoryConfig()
tracking_config = TrackingConfig()
motion_config = MotionConfig()
search_config = SearchConfig()
guard_config = GuardConfig()
_collision_cfg = CollisionConfig()
collision_guard = CollisionGuard(
    enabled=_collision_cfg.enabled,
    height_threshold=_collision_cfg.height_threshold,
    center_zone_fraction=_collision_cfg.center_zone_fraction,
    skip_labels=frozenset(s.strip() for s in _collision_cfg.skip_labels.split(",") if s.strip()),
)

state_lock = threading.Lock()
recognition_lock = threading.Lock()
latest_jpeg: bytes | None = None
latest_frame: np.ndarray | None = None
latest_detections = []
latest_reference_captures = []
active_face_detector: FaceDetector | None = None
active_face_recognizer: FaceRecognizer | None = None
active_person_memory_store: PersonMemoryStore | None = None
active_vault_memory_client: BrainMemoryClient | None = None
active_controller = None  # set by run_vision(); used by /tracking endpoint
active_robot: RobotClient | None = None  # set by run_vision(); used by /pantilt endpoint
guard_mode_enabled: bool = False
_last_guard_alert_at: float = 0.0
_guard_alerted_this_target: bool = False
latest_meta = {
    "ok": False,
    "frame_id": 0,
    "frame_shape": None,
    "detections": [],
    "target_id": None,
    "robot_command_ok": None,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rover Hailo object detection and person tracking service.")
    parser.add_argument("--hef-path", default=None, help="Path to the Hailo HEF model. If omitted, a known local detector is selected.")
    parser.add_argument("--labels", default=None, help="Path to labels file, one label per line.")
    parser.add_argument("--camera-index", type=int, default=vision_config.camera_index)
    parser.add_argument("--host", default=vision_config.host)
    parser.add_argument("--port", type=int, default=vision_config.port)
    parser.add_argument("--robot-api-url", default=vision_config.robot_api_url)
    parser.add_argument("--no-motion", action="store_true", help="Run detection/video but do not command pan/tilt.")
    parser.add_argument("--hailo-apps-dir", default=vision_config.hailo_apps_dir)
    parser.add_argument("--pose-hef-path", default=vision_config.pose_hef_path)
    parser.add_argument("--no-pose", action="store_true", help="Disable optional pose estimation.")
    parser.add_argument("--no-face-detection", action="store_true", help="Disable OpenCV face detection enrichment.")
    parser.add_argument("--no-face-recognition", action="store_true", help="Disable known-person face recognition.")
    return parser.parse_args()


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


def resolve_pose_estimator(args: argparse.Namespace) -> PoseEstimator | None:
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
            pose = PoseEstimator(
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


# ── Calibration test: track chairs instead of people ──
TARGET_LABEL = "chair"  # COCO class 56


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


def run_vision(args: argparse.Namespace) -> None:
    labels_path = args.labels or default_labels_path(args.hailo_apps_dir)
    labels = load_labels(labels_path)
    tracker = SimpleTracker(tracking_config)
    controller = PanTiltController(motion_config)
    search_controller = SearchController(search_config)
    robot = RobotClient(args.robot_api_url)
    face_detector = FaceDetector(face_config)
    if args.no_face_detection:
        face_detector.enabled = False
    vault_memory_client = BrainMemoryClient(vault_memory_config)
    vault_memory_client.sync_face_references_if_due(force=True)
    recognizer_config = replace(recognition_config, enabled=False) if args.no_face_recognition else recognition_config
    if vault_memory_client.enabled:
        recognizer_config = replace(recognizer_config, known_faces_dir=str(vault_memory_client.face_cache_dir))
    face_recognizer = FaceRecognizer(recognizer_config, face_config)
    person_memory_store = PersonMemoryStore(person_memory_config, vault_memory_client)
    with recognition_lock:
        global active_face_detector, active_face_recognizer, active_person_memory_store, active_vault_memory_client
        active_face_detector = face_detector
        active_face_recognizer = face_recognizer
        active_person_memory_store = person_memory_store
        active_vault_memory_client = vault_memory_client

    if args.no_motion:
        controller.config.enabled = False
    global active_controller, active_robot
    global _last_guard_alert_at, _guard_alerted_this_target
    active_controller = controller
    active_robot = robot

    hailo, hef_path = resolve_hailo(args.hef_path, args.hailo_apps_dir)
    input_h, input_w, _ = hailo.get_input_shape()
    log.info("Loaded HEF input shape: %sx%s", input_w, input_h)
    pose_estimator = resolve_pose_estimator(args)
    pose_queue: queue.Queue = queue.Queue(maxsize=1)
    latest_poses = []

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vision_config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vision_config.camera_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    robot.pantilt(controller.center_command())
    output_queue: queue.Queue = queue.Queue(maxsize=1)
    frame_id = 0
    last_target_vx = 0.0

    try:
        while True:
            for _ in range(2):
                cap.grab()
            ok, frame = cap.retrieve()
            if not ok:
                time.sleep(0.05)
                continue

            frame_id += 1
            preprocessed = letterbox(frame, input_w, input_h)

            callback = partial(inference_callback, output_queue=output_queue)
            job = hailo.run([preprocessed], callback)
            job.wait(10000)

            try:
                result = output_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(result, Exception):
                raise result

            tracker.set_ego_motion(controller.ego_motion())
            detections = parse_hailo_detections(
                result,
                preprocessed,
                labels,
                frame_w=input_w,
                frame_h=input_h,
                threshold=tracking_config.score_threshold,
            )
            has_person = False  # TEST: face detection disabled for chair calibration
            has_target = any(d.label == TARGET_LABEL for d in detections)
            face_detections = []
            if has_person and frame_id % max(1, face_config.interval_frames) == 0:
                face_detections = face_detector.detect(preprocessed)
            all_detections = [*detections, *face_detections]
            reference_captures = []
            if has_person and frame_id % max(1, recognition_config.interval_frames) == 0:
                with recognition_lock:
                    all_detections = face_recognizer.recognize_faces(preprocessed, all_detections)
                    # Sync from brain when we see an unknown face rather than on a fixed timer.
                    has_unknown = any(
                        getattr(d, "identity", None) == face_recognizer.config.unknown_label
                        for d in all_detections
                    )
                    if has_unknown and vault_memory_client.sync_on_unknown_identity():
                        face_recognizer.reload_from_disk()
                        all_detections = face_recognizer.recognize_faces(preprocessed, all_detections)
                    reference_captures = face_recognizer.auto_capture_missing_references(preprocessed, all_detections)
                    for capture in reference_captures:
                        vault_memory_client.upload_face_reference(capture)
            attach_face_identities_to_people(all_detections)
            tracked = tracker.update(all_detections)
            tracker.hydrate_person_memory(person_memory_store)
            # TEST: select best chair target instead of person
            chair_objs = [
                obj for obj in tracked
                if obj.label == TARGET_LABEL
                and obj.confidence >= tracking_config.score_threshold
            ]
            if chair_objs:
                cx, cy = input_w / 2.0, input_h / 2.0
                target = min(chair_objs, key=lambda d: (
                    (d.bbox[0] + d.bbox[2]/2 - cx)**2 +
                    (d.bbox[1] + d.bbox[3]/2 - cy)**2
                ))
                # Set aim point to center of bounding box
                x, y, w, h = target.bbox
                target.aim_x = x + w / 2.0
                target.aim_y = y + h / 2.0
                target.aim_source = "bbox_center"
            else:
                target = None
            has_tracked_person = any(d.label == TARGET_LABEL for d in tracked)
            if not has_tracked_person:
                poses = []
                latest_poses = []
            else:
                poses = latest_poses
            if has_tracked_person and pose_estimator is not None and vision_config.pose_enabled and frame_id % max(1, vision_config.pose_interval_frames) == 0:
                try:
                    poses = pose_estimator.run(
                        preprocessed,
                        partial(inference_callback, output_queue=pose_queue),
                        pose_queue,
                    )
                    latest_poses = poses
                except Exception as exc:
                    log.warning("Pose inference failed: %s", exc)
                    poses = []
                    latest_poses = []
            if False and vision_config.pose_filter_persons and poses:  # TEST: disabled
                tracked = _filter_persons_by_pose(tracked, poses)
                if target is not None and all(det.id != target.id for det in tracked):
                    target = None
            # TEST: no pose aim for chair targets
            # target = apply_pose_aim(target, poses, vision_config.pose_joint_threshold)

            if target is not None:
                last_target_vx = getattr(target, "vx", 0.0)
                search_controller.on_target_acquired()
                if guard_mode_enabled:
                    now_t = time.time()
                    if not _guard_alerted_this_target and now_t - _last_guard_alert_at >= guard_config.alert_cooldown_seconds:
                        _last_guard_alert_at = now_t
                        _guard_alerted_this_target = True
                        threading.Thread(target=_fire_guard_alert, args=(target,), daemon=True).start()
            else:
                _guard_alerted_this_target = False
                if guard_mode_enabled or controller.config.enabled:
                    search_controller.on_target_lost(controller._estimated_pan, last_target_vx)

            command = controller.command_for_target(target, input_w, input_h)
            command_ok = None
            if command is not None:
                command_ok = dispatch_robot_command(robot, command)

            if target is None and controller.config.enabled and search_controller.phase != "idle":
                search_cmd = search_controller.search_command(controller._estimated_pan, controller._estimated_tilt)
                if search_cmd is not None:
                    robot.pantilt(search_cmd)
                    controller.notify_external_pantilt(search_cmd["pan"], search_cmd["tilt"])

            obstacle_detected = collision_guard.check(tracked, tracker.selected_target_id, input_w, input_h)
            if obstacle_detected:
                robot.move(0, 0)
            elif controller.config.follow_enabled:
                follow_cmd = controller.wheel_follow_command(target, input_w, input_h)
                if follow_cmd is not None:
                    robot.move(follow_cmd["x"], follow_cmd["z"])

            display_detections = tracked
            if target is not None and target.predicted and all(det.id != target.id for det in tracked):
                display_detections = [*tracked, target]

            annotated = draw_detections(preprocessed, display_detections, tracker.selected_target_id)
            if poses:
                annotated = draw_poses(annotated, poses, vision_config.pose_joint_threshold)
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), vision_config.jpeg_quality],
            )

            with state_lock:
                global latest_jpeg, latest_frame, latest_detections, latest_reference_captures, latest_meta
                if encode_ok:
                    latest_jpeg = encoded.tobytes()
                latest_frame = preprocessed.copy()
                latest_detections = [replace(det) for det in tracked]
                if reference_captures:
                    latest_reference_captures = reference_captures
                latest_meta = {
                    "ok": True,
                    "frame_id": frame_id,
                    "hef_path": hef_path,
                    "pose_hef_path": pose_estimator.hef_path if pose_estimator else None,
                    "labels_path": labels_path,
                    "frame_shape": [int(input_h), int(input_w), 3],
                    "detections": [det.to_json() for det in tracked],
                    "object_memory": [memory.to_json() for memory in tracker.memory],
                    "tracker": tracker.stats(),
                    "poses": [pose.to_json() for pose in poses],
                    "pose_enabled": vision_config.pose_enabled and bool(pose_estimator),
                    "pose_available": bool(pose_estimator),
                    "pose_filter_persons": vision_config.pose_filter_persons,
                    "pose_interval_frames": vision_config.pose_interval_frames,
                    "pose_score_threshold": vision_config.pose_score_threshold,
                    "jpeg_quality": vision_config.jpeg_quality,
                    "score_threshold": tracking_config.score_threshold,
                    "person_score_threshold": tracking_config.person_score_threshold,
                    "face_detection_enabled": bool(face_detector.enabled),
                    "face_interval_frames": face_config.interval_frames,
                    "face_recognition_enabled": bool(face_recognizer.enabled),
                    "face_recognition_method": face_recognizer.method,
                    "auto_reference_capture_enabled": bool(face_recognizer.config.auto_reference_capture_enabled),
                    "auto_reference_min_confidence": face_recognizer.config.auto_reference_min_confidence,
                    "follow_forward_speed": controller.config.follow_forward_speed,
                    "follow_steer_gain": controller.config.follow_steer_gain,
                    "follow_target_bbox_ratio": controller.config.follow_target_bbox_ratio,
                    "follow_deadzone_ratio": controller.config.follow_deadzone_ratio,
                    "close_target_bbox_ratio": controller.config.close_target_bbox_ratio,
                    "pan_invert": controller.config.pan_invert,
                    "tilt_invert": controller.config.tilt_invert,
                    "edge_reacquire_enabled": controller.config.edge_reacquire_enabled,
                    "wheel_enabled": controller.config.wheel_enabled,
                    "max_command": controller.config.max_command,
                    "min_command": controller.config.min_command,
                    "max_command_step": controller.config.max_command_step,
                    "command_interval_seconds": controller.config.command_interval_seconds,
                    "settle_enter_degrees": controller.config.settle_enter_degrees,
                    "settle_exit_degrees": controller.config.settle_exit_degrees,
                    "estimated_pan_min": controller.config.estimated_pan_min,
                    "estimated_pan_max": controller.config.estimated_pan_max,
                    "estimated_tilt_min": controller.config.estimated_tilt_min,
                    "estimated_tilt_max": controller.config.estimated_tilt_max,
                    "pan_estimate_scale": controller.config.pan_estimate_scale,
                    "tilt_estimate_scale": controller.config.tilt_estimate_scale,
                    "pan_limit_margin": controller.config.pan_limit_margin,
                    "known_faces_dir": str(face_recognizer.known_faces_dir),
                    "person_memory_enabled": bool(person_memory_store.config.enabled),
                    "people_dir": str(person_memory_store.people_dir),
                    "vault_memory": vault_memory_client.status(),
                    "reference_pose_coverage": face_recognizer.reference_pose_coverage(),
                    "latest_reference_captures": latest_reference_captures,
                    "learn_face_endpoint": "POST /learn_face?name=<your-name>",
                    "target_identity": tracking_config.target_identity,
                    "target_id": tracker.selected_target_id,
                    "target": target.to_json() if target else None,
                    "target_state": (
                        "predicted" if target and target.predicted
                        else "visible" if target
                        else search_controller.phase if search_controller.phase != "idle"
                        else "none"
                    ),
                    "search_phase": search_controller.phase,
                    "guard_mode": guard_mode_enabled,
                    "tracking_enabled": controller.config.enabled,
                    "follow_enabled": controller.config.follow_enabled,
                    "ego_motion": controller.ego_motion(),
                    "robot_command_ok": command_ok,
                    "robot_command": command,
                    "collision_blocked": obstacle_detected,
                    "collision_avoidance_enabled": collision_guard.enabled,
                    "collision_height_threshold": collision_guard.height_threshold,
                    "collision_center_zone_fraction": collision_guard.center_zone_fraction,
                }
    finally:
        cap.release()
        hailo.close()
        if pose_estimator is not None:
            pose_estimator.close()


def _fire_guard_alert(target) -> None:
    import urllib.request
    import urllib.error
    payload = {
        "event": "person_detected",
        "identity": getattr(target, "identity", None),
        "confidence": float(getattr(target, "confidence", 0.0)),
        "source": "rover",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        guard_config.brain_alert_url.rstrip("/") + "/guard/alert",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=guard_config.alert_timeout_seconds):
            pass
    except Exception as exc:
        log.warning("Guard alert failed: %s", exc)


def dispatch_robot_command(robot: RobotClient, command: dict) -> bool:
    if command.get("mode") == "edge_reacquire":
        pantilt_ok = robot.pantilt(command["pantilt"])
        move = command.get("move", {})
        move_ok = True
        if move:
            move_ok = robot.move(move.get("x", 0), move.get("z", 0))
            stop_after = float(command.get("move_stop_after", 0))
            if stop_after > 0:
                time.sleep(stop_after)
                move_ok = robot.move(0, 0) and move_ok
        return pantilt_ok and move_ok
    return robot.pantilt(command)


def _filter_persons_by_pose(detections: list, poses: list, iou_threshold: float = 0.05) -> list:
    result = []
    for det in detections:
        if det.label != "person":
            result.append(det)
            continue
        if any(bbox_iou(det.bbox, pose.bbox) >= iou_threshold for pose in poses):
            result.append(det)
    return result


def attach_face_identities_to_people(detections) -> None:
    faces = [det for det in detections if is_face_label(det.label) and det.identity and det.identity != "unknown"]
    if not faces:
        return
    for person in [det for det in detections if det.label == "person"]:
        inside = [face for face in faces if center_inside(face.center, person.bbox)]
        if not inside:
            continue
        face = max(inside, key=lambda det: det.identity_confidence or 0.0)
        person.copy_identity_from(face)


def _ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luhkas Vision</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;flex-shrink:0}
h1{font-size:.85rem;font-weight:700;letter-spacing:2px;color:#bbb}
#dot{font-size:1rem;color:#555;transition:color .3s}
#dot.live{color:#4c4;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
#hdr-right{margin-left:auto;font-size:.75rem;color:#666}
main{display:flex;flex:1;overflow:hidden}
#feed{flex:1;min-width:0;background:#000;display:flex;align-items:center;justify-content:center}
#feed img{max-width:100%;max-height:100%;object-fit:contain;display:block}
aside{width:290px;flex-shrink:0;overflow-y:auto;background:#161616;border-left:1px solid #252525;padding:10px;display:flex;flex-direction:column;gap:8px}
.mode-seg{display:flex;gap:4px}
.mode-btn{flex:1;font-size:.72rem;font-weight:700;padding:6px;border-radius:4px;border:1px solid #333;cursor:pointer;background:#1c1c1c;color:#555;transition:all .15s;letter-spacing:.5px}
.mode-btn.active{background:#163016;border-color:#2e6a2e;color:#6c6}
.dpad{display:grid;grid-template-areas:'. up . ''left ctr right''. dn . ';grid-template-columns:1fr 1fr 1fr;gap:5px;width:130px;margin:4px auto}
.dp{width:40px;height:40px;font-size:1rem;border-radius:5px;border:1px solid #333;background:#222;color:#888;cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none;-webkit-user-select:none;touch-action:none}
.dp:active,.dp.held{background:#2a2a2a;color:#eee}
.dp-up{grid-area:up}.dp-dn{grid-area:dn}.dp-left{grid-area:left}.dp-right{grid-area:right}
.dp-ctr{grid-area:ctr;background:#163016;border-color:#2e6a2e;color:#6c6;font-size:.7rem}
.card{background:#1c1c1c;border:1px solid #252525;border-radius:5px;padding:9px;display:flex;flex-direction:column;gap:7px}
.card-title{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#666;margin-bottom:1px}
.row{display:flex;align-items:center;justify-content:space-between;gap:6px}
.lbl{font-size:.75rem;color:#888}
.val{font-size:.75rem;font-weight:600;color:#ccc}
button{font-size:.7rem;padding:3px 9px;border-radius:3px;border:1px solid #333;cursor:pointer;background:#222;color:#999;transition:all .15s;min-width:42px}
button.on{background:#163016;border-color:#2e6a2e;color:#6c6}
button.off{background:#222;border-color:#333;color:#666}
.srow{display:flex;flex-direction:column;gap:2px}
.slbls{display:flex;justify-content:space-between;font-size:.7rem;color:#777}
input[type=range]{width:100%;height:4px;accent-color:#3a7;cursor:pointer;margin:2px 0}
input[type=text]{font-size:.72rem;padding:3px 6px;border-radius:3px;border:1px solid #333;background:#161616;color:#ccc;width:100%}
input[type=text]:focus{outline:none;border-color:#3a7}
.badge{font-size:.65rem;padding:2px 7px;border-radius:3px;font-weight:700;letter-spacing:.5px}
.badge.clear{background:#0e2a0e;color:#4a4;border:1px solid #1e4a1e}
.badge.blocked{background:#2a0e0e;color:#e44;border:1px solid #4a1e1e;animation:pulse .5s step-end infinite}
.det-item{display:flex;justify-content:space-between;align-items:center;font-size:.7rem;padding:3px 0;border-bottom:1px solid #1e1e1e}
.det-item:last-child{border:none}
.det-lbl{color:#bbb}.det-lbl.tgt{color:#6cf}
.det-conf{color:#555}
.none{font-size:.7rem;color:#444;font-style:italic}
.divider{border:none;border-top:1px solid #222;margin:2px 0}
</style>
</head>
<body>
<header>
  <span id="dot">●</span>
  <h1>LUHKAS VISION</h1>
  <span id="hdr-right">fr —</span>
</header>
<main>
  <div id="feed"><img src="/video_feed" alt="live feed"></div>
  <aside>

    <!-- MODE -->
    <div class="mode-seg">
      <button class="mode-btn active" id="btn-mode-tracking" onclick="setMode('tracking')">TRACKING</button>
      <button class="mode-btn" id="btn-mode-manual" onclick="setMode('manual')">MANUAL</button>
      <button class="mode-btn" id="btn-mode-guard" onclick="setMode('guard')">GUARD</button>
      <span id="guard-indicator" style="display:none;color:#f55;font-weight:700;font-size:.75rem;letter-spacing:1px;animation:pulse 1s ease-in-out infinite">&#9679; GUARDING</span>
    </div>

    <!-- MANUAL CAMERA CONTROL -->
    <div class="card" id="manual-card" style="display:none">
      <div class="card-title">Camera Control</div>
      <div class="dpad">
        <button class="dp dp-up" id="dp-up"
          onmousedown="startPT(0,1)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(0,1);event.preventDefault()" ontouchend="stopPT()">▲</button>
        <button class="dp dp-left" id="dp-left"
          onmousedown="startPT(-1,0)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(-1,0);event.preventDefault()" ontouchend="stopPT()">◀</button>
        <button class="dp dp-ctr" onclick="centerCamera()">⌖</button>
        <button class="dp dp-right" id="dp-right"
          onmousedown="startPT(1,0)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(1,0);event.preventDefault()" ontouchend="stopPT()">▶</button>
        <button class="dp dp-dn" id="dp-dn"
          onmousedown="startPT(0,-1)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(0,-1);event.preventDefault()" ontouchend="stopPT()">▼</button>
      </div>
      <div style="text-align:center;font-size:.65rem;color:#444;margin-top:2px">Arrow keys · ⌖ to center</div>
      <div class="srow" style="margin-top:4px">
        <div class="slbls"><span>Pan step</span><span id="pan-step-val">5</span></div>
        <input type="range" id="pan-step-sld" min="1" max="200" step="1" value="5"
          oninput="PAN_STEP=parseInt(this.value);q('pan-step-val').textContent=this.value">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt step</span><span id="tilt-step-val">5</span></div>
        <input type="range" id="tilt-step-sld" min="1" max="30" step="1" value="5"
          oninput="TILT_STEP=parseInt(this.value);q('tilt-step-val').textContent=this.value">
      </div>
    </div>

    <!-- TRACKING -->
    <div class="card">
      <div class="card-title">Tracking</div>
      <div class="row"><span class="lbl">Enabled</span><button id="btn-tracking_enabled" onclick="tog('tracking_enabled','/tracking','enabled')">—</button></div>
      <div class="row"><span class="lbl">Follow wheels</span><button id="btn-follow_enabled" onclick="tog('follow_enabled','/tracking','follow')">—</button></div>
      <div class="row"><span class="lbl">Wheel drive</span><button id="btn-wheel_enabled" onclick="setting('wheel_enabled')">—</button></div>
      <hr class="divider">
      <div class="srow">
        <div class="slbls"><span>Target identity</span></div>
        <input type="text" id="inp-identity" placeholder="name or blank for any"
          onchange="post('/tracking',{target_identity:this.value||null})">
      </div>
      <div class="srow">
        <div class="slbls"><span>Score threshold</span><span id="val-score_threshold">—</span></div>
        <input type="range" id="sld-score_threshold" min="0.10" max="0.90" step="0.05"
          oninput="sld(this,'score_threshold',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Person score threshold</span><span id="val-person_score_threshold">—</span></div>
        <input type="range" id="sld-person_score_threshold" min="0.10" max="0.90" step="0.05"
          oninput="sld(this,'person_score_threshold',2,'/settings')">
      </div>
    </div>

    <!-- FOLLOW TUNING -->
    <div class="card">
      <div class="card-title">Follow Tuning</div>
      <div class="srow">
        <div class="slbls"><span>Forward speed</span><span id="val-follow_forward_speed">—</span></div>
        <input type="range" id="sld-follow_forward_speed" min="100" max="800" step="50"
          oninput="sld(this,'follow_forward_speed',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Steer gain</span><span id="val-follow_steer_gain">—</span></div>
        <input type="range" id="sld-follow_steer_gain" min="0.5" max="10.0" step="0.5"
          oninput="sld(this,'follow_steer_gain',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Follow bbox ratio</span><span id="val-follow_target_bbox_ratio">—</span></div>
        <input type="range" id="sld-follow_target_bbox_ratio" min="0.10" max="0.60" step="0.02"
          oninput="sld(this,'follow_target_bbox_ratio',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Close bbox ratio</span><span id="val-close_target_bbox_ratio">—</span></div>
        <input type="range" id="sld-close_target_bbox_ratio" min="0.30" max="0.90" step="0.05"
          oninput="sld(this,'close_target_bbox_ratio',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Follow deadzone</span><span id="val-follow_deadzone_ratio">—</span></div>
        <input type="range" id="sld-follow_deadzone_ratio" min="0.01" max="0.20" step="0.01"
          oninput="sld(this,'follow_deadzone_ratio',2,'/settings')">
      </div>
    </div>

    <!-- PAN-TILT -->
    <div class="card">
      <div class="card-title">Pan-Tilt</div>
      <div class="row"><span class="lbl">Pan invert</span><button id="btn-pan_invert" onclick="setting('pan_invert')">—</button></div>
      <div class="row"><span class="lbl">Tilt invert</span><button id="btn-tilt_invert" onclick="setting('tilt_invert')">—</button></div>
      <div class="row"><span class="lbl">Edge reacquire</span><button id="btn-edge_reacquire_enabled" onclick="setting('edge_reacquire_enabled')">—</button></div>
      <hr class="divider">
      <div class="srow">
        <div class="slbls"><span>Max command</span><span id="val-max_command">—</span></div>
        <input type="range" id="sld-max_command" min="10" max="300" step="5"
          oninput="sld(this,'max_command',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Min command</span><span id="val-min_command">—</span></div>
        <input type="range" id="sld-min_command" min="1" max="30" step="1"
          oninput="sld(this,'min_command',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Max step (ramp rate)</span><span id="val-max_command_step">—</span></div>
        <input type="range" id="sld-max_command_step" min="1" max="100" step="1"
          oninput="sld(this,'max_command_step',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Command interval (s)</span><span id="val-command_interval_seconds">—</span></div>
        <input type="range" id="sld-command_interval_seconds" min="0.05" max="1.0" step="0.05"
          oninput="sld(this,'command_interval_seconds',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Settle enter (°)</span><span id="val-settle_enter_degrees">—</span></div>
        <input type="range" id="sld-settle_enter_degrees" min="0.5" max="20.0" step="0.5"
          oninput="sld(this,'settle_enter_degrees',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Settle exit (°)</span><span id="val-settle_exit_degrees">—</span></div>
        <input type="range" id="sld-settle_exit_degrees" min="0.5" max="25.0" step="0.5"
          oninput="sld(this,'settle_exit_degrees',1,'/settings')">
      </div>
      <hr class="divider">
      <div class="srow">
        <div class="slbls"><span>Pan min (°)</span><span id="val-estimated_pan_min">—</span></div>
        <input type="range" id="sld-estimated_pan_min" min="-180" max="0" step="5"
          oninput="sld(this,'estimated_pan_min',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan max (°)</span><span id="val-estimated_pan_max">—</span></div>
        <input type="range" id="sld-estimated_pan_max" min="0" max="180" step="5"
          oninput="sld(this,'estimated_pan_max',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt min (°)</span><span id="val-estimated_tilt_min">—</span></div>
        <input type="range" id="sld-estimated_tilt_min" min="-90" max="0" step="5"
          oninput="sld(this,'estimated_tilt_min',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt max (°)</span><span id="val-estimated_tilt_max">—</span></div>
        <input type="range" id="sld-estimated_tilt_max" min="0" max="90" step="5"
          oninput="sld(this,'estimated_tilt_max',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan scale</span><span id="val-pan_estimate_scale">—</span></div>
        <input type="range" id="sld-pan_estimate_scale" min="0.1" max="5.0" step="0.1"
          oninput="sld(this,'pan_estimate_scale',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt scale</span><span id="val-tilt_estimate_scale">—</span></div>
        <input type="range" id="sld-tilt_estimate_scale" min="0.1" max="5.0" step="0.1"
          oninput="sld(this,'tilt_estimate_scale',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan limit margin (°)</span><span id="val-pan_limit_margin">—</span></div>
        <input type="range" id="sld-pan_limit_margin" min="0" max="60" step="5"
          oninput="sld(this,'pan_limit_margin',0,'/settings')">
      </div>
    </div>

    <!-- COLLISION -->
    <div class="card">
      <div class="card-title">Collision Avoidance</div>
      <div class="row">
        <span class="lbl">Enabled</span>
        <button id="btn-collision_avoidance_enabled" onclick="tog('collision_avoidance_enabled','/collision','enabled')">—</button>
      </div>
      <div class="row"><span class="lbl">Status</span><span id="collision-badge" class="badge clear">CLEAR</span></div>
      <div class="srow">
        <div class="slbls"><span>Height threshold</span><span id="val-collision_height_threshold">—</span></div>
        <input type="range" id="sld-collision_height_threshold" min="0.10" max="0.80" step="0.05"
          oninput="sld(this,'collision_height_threshold',2,'/collision','height_threshold')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Center zone</span><span id="val-collision_center_zone_fraction">—</span></div>
        <input type="range" id="sld-collision_center_zone_fraction" min="0.30" max="1.00" step="0.05"
          oninput="sld(this,'collision_center_zone_fraction',2,'/collision','center_zone_fraction')">
      </div>
    </div>

    <!-- FACE -->
    <div class="card">
      <div class="card-title">Face</div>
      <div class="row"><span class="lbl">Detection</span><button id="btn-face_detection_enabled" onclick="setting('face_detection_enabled')">—</button></div>
      <div class="row"><span class="lbl">Recognition</span><button id="btn-face_recognition_enabled" onclick="setting('face_recognition_enabled')">—</button></div>
      <div class="row"><span class="lbl">Auto-capture refs</span><button id="btn-auto_reference_capture_enabled" onclick="setting('auto_reference_capture_enabled')">—</button></div>
      <div class="srow">
        <div class="slbls"><span>Auto-capture min conf</span><span id="val-auto_reference_min_confidence">—</span></div>
        <input type="range" id="sld-auto_reference_min_confidence" min="0.10" max="0.80" step="0.05"
          oninput="sld(this,'auto_reference_min_confidence',2,'/settings')">
      </div>
    </div>

    <!-- VISION -->
    <div class="card">
      <div class="card-title">Vision</div>
      <div class="row">
        <span class="lbl">Pose estimation</span>
        <button id="btn-pose_enabled" onclick="setting('pose_enabled')">—</button>
      </div>
      <div class="row">
        <span class="lbl">Filter ghosts by pose</span>
        <button id="btn-pose_filter_persons" onclick="setting('pose_filter_persons')">—</button>
      </div>
      <div class="srow">
        <div class="slbls"><span>Pose interval (frames)</span><span id="val-pose_interval_frames">—</span></div>
        <input type="range" id="sld-pose_interval_frames" min="1" max="10" step="1"
          oninput="sld(this,'pose_interval_frames',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pose score threshold</span><span id="val-pose_score_threshold">—</span></div>
        <input type="range" id="sld-pose_score_threshold" min="0.10" max="0.90" step="0.05"
          oninput="sld(this,'pose_score_threshold',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>JPEG quality</span><span id="val-jpeg_quality">—</span></div>
        <input type="range" id="sld-jpeg_quality" min="20" max="95" step="5"
          oninput="sld(this,'jpeg_quality',0,'/settings')">
      </div>
    </div>

    <!-- TARGET -->
    <div class="card">
      <div class="card-title">Target</div>
      <div class="row"><span class="lbl">State</span><span id="tgt-state" class="val">—</span></div>
      <div class="row"><span class="lbl">Identity</span><span id="tgt-identity" class="val">—</span></div>
      <div class="row"><span class="lbl">ID</span><span id="tgt-id" class="val">—</span></div>
    </div>

    <!-- DETECTIONS -->
    <div class="card">
      <div class="card-title">Detections <span id="det-count" style="color:#555;font-weight:400"></span></div>
      <div id="det-list"><span class="none">waiting…</span></div>
    </div>

  </aside>
</main>
<script>
var STATE = {};
var appMode = 'tracking';
var ptInterval = null;
var PAN_STEP = 5;
var TILT_STEP = 5;

function q(id){return document.getElementById(id)}

function post(url, body){
  fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

function setMode(m){
  appMode = m;
  q('btn-mode-tracking').className = 'mode-btn' + (m==='tracking'?' active':'');
  q('btn-mode-manual').className   = 'mode-btn' + (m==='manual'  ?' active':'');
  q('btn-mode-guard').className    = 'mode-btn' + (m==='guard'   ?' active':'');
  q('manual-card').style.display   = m==='manual' ? '' : 'none';
  q('guard-indicator').style.display = m==='guard' ? '' : 'none';
  if(m==='manual'){
    STATE['tracking_enabled'] = false;
    syncBtn('tracking_enabled', false);
    post('/tracking', {enabled: false});
    post('/guard', {enabled: false});
  } else if(m==='guard'){
    STATE['tracking_enabled'] = false;
    syncBtn('tracking_enabled', false);
    post('/tracking', {enabled: false});
    post('/guard', {enabled: true});
  } else {
    post('/guard', {enabled: false});
  }
}

function startPT(panDir, tiltDir){
  stopPT();
  function send(){post('/pantilt',{pan:panDir*PAN_STEP,tilt:tiltDir*TILT_STEP});}
  send();
  ptInterval = setInterval(send, 180);
}

function stopPT(){
  if(ptInterval){clearInterval(ptInterval);ptInterval=null;}
}

function centerCamera(){
  post('/pantilt',{center:true});
}

document.addEventListener('keydown', function(e){
  if(appMode!=='manual') return;
  if(e.target.tagName==='INPUT') return;
  if(e.repeat) return;
  var dirs={ArrowUp:[0,1],ArrowDown:[0,-1],ArrowLeft:[-1,0],ArrowRight:[1,0]};
  var d=dirs[e.key];
  if(!d) return;
  e.preventDefault();
  startPT(d[0],d[1]);
});

document.addEventListener('keyup', function(e){
  var keys=['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
  if(keys.indexOf(e.key)>=0) stopPT();
});

function setting(key){
  var cur = STATE[key];
  var next = !cur;
  STATE[key] = next;
  var b = q('btn-'+key);
  if(b){b.textContent=next?'ON':'OFF'; b.className=next?'on':'off';}
  var body = {}; body[key] = next;
  post('/settings', body);
}

function tog(stateKey, url, bodyKey){
  var cur = STATE[stateKey];
  var next = !cur;
  STATE[stateKey] = next;
  var b = q('btn-'+stateKey);
  if(b){b.textContent=next?'ON':'OFF'; b.className=next?'on':'off';}
  var body = {}; body[bodyKey] = next;
  post(url, body);
}

function sld(el, key, decimals, url, bodyKey){
  var v = parseFloat(el.value);
  var vq = q('val-'+key);
  if(vq) vq.textContent = v.toFixed(decimals);
  var k = bodyKey || key;
  var body = {}; body[k] = v;
  post(url, body);
}

function syncBtn(key, val){
  STATE[key] = !!val;
  var b = q('btn-'+key);
  if(b){b.textContent=val?'ON':'OFF'; b.className=val?'on':'off';}
}

function syncSld(key, val, decimals){
  var el = q('sld-'+key);
  var vq = q('val-'+key);
  if(el && document.activeElement !== el){
    el.value = val;
    if(vq) vq.textContent = parseFloat(val).toFixed(decimals||2);
  }
}

function syncText(id, val){
  var el = q(id);
  if(el && document.activeElement !== el) el.value = val || '';
}

function poll(){
  fetch('/meta').then(function(r){return r.json()}).then(function(d){
    q('dot').className = 'live';
    q('hdr-right').textContent = 'fr ' + d.frame_id;

    syncBtn('tracking_enabled', d.tracking_enabled);
    syncBtn('follow_enabled', d.follow_enabled);
    syncBtn('wheel_enabled', d.wheel_enabled);
    syncBtn('pan_invert', d.pan_invert);
    syncBtn('tilt_invert', d.tilt_invert);
    syncBtn('edge_reacquire_enabled', d.edge_reacquire_enabled);
    syncBtn('collision_avoidance_enabled', d.collision_avoidance_enabled);
    syncBtn('face_detection_enabled', d.face_detection_enabled);
    syncBtn('face_recognition_enabled', d.face_recognition_enabled);
    syncBtn('auto_reference_capture_enabled', d.auto_reference_capture_enabled);
    syncBtn('pose_enabled', d.pose_enabled);
    syncBtn('pose_filter_persons', d.pose_filter_persons);

    var blocked = !!d.collision_blocked;
    var badge = q('collision-badge');
    badge.textContent = blocked ? 'BLOCKED' : 'CLEAR';
    badge.className = 'badge ' + (blocked ? 'blocked' : 'clear');

    syncSld('score_threshold', d.score_threshold, 2);
    syncSld('person_score_threshold', d.person_score_threshold, 2);
    syncSld('follow_forward_speed', d.follow_forward_speed, 0);
    syncSld('follow_steer_gain', d.follow_steer_gain, 1);
    syncSld('follow_target_bbox_ratio', d.follow_target_bbox_ratio, 2);
    syncSld('close_target_bbox_ratio', d.close_target_bbox_ratio, 2);
    syncSld('follow_deadzone_ratio', d.follow_deadzone_ratio, 2);
    syncSld('max_command', d.max_command, 0);
    syncSld('min_command', d.min_command, 0);
    syncSld('max_command_step', d.max_command_step, 0);
    syncSld('command_interval_seconds', d.command_interval_seconds, 2);
    syncSld('settle_enter_degrees', d.settle_enter_degrees, 1);
    syncSld('settle_exit_degrees', d.settle_exit_degrees, 1);
    syncSld('estimated_pan_min', d.estimated_pan_min, 0);
    syncSld('estimated_pan_max', d.estimated_pan_max, 0);
    syncSld('estimated_tilt_min', d.estimated_tilt_min, 0);
    syncSld('estimated_tilt_max', d.estimated_tilt_max, 0);
    syncSld('pan_estimate_scale', d.pan_estimate_scale, 1);
    syncSld('tilt_estimate_scale', d.tilt_estimate_scale, 1);
    syncSld('pan_limit_margin', d.pan_limit_margin, 0);
    syncSld('collision_height_threshold', d.collision_height_threshold, 2);
    syncSld('collision_center_zone_fraction', d.collision_center_zone_fraction, 2);
    syncSld('auto_reference_min_confidence', d.auto_reference_min_confidence, 2);
    syncSld('pose_interval_frames', d.pose_interval_frames, 0);
    syncSld('pose_score_threshold', d.pose_score_threshold, 2);
    syncSld('jpeg_quality', d.jpeg_quality, 0);

    syncText('inp-identity', d.target_identity);

    q('tgt-state').textContent = d.target_state || 'none';
    q('tgt-identity').textContent = (d.target && d.target.identity) || '—';
    q('tgt-id').textContent = d.target_id != null ? d.target_id : '—';

    var dets = d.detections || [];
    q('det-count').textContent = '(' + dets.length + ')';
    q('det-list').innerHTML = dets.length ? dets.map(function(det){
      var isT = det.id === d.target_id;
      var lbl = (isT ? '▶ ' : '') + det.label + (det.identity ? ' · ' + det.identity : '');
      var conf = Math.round(det.confidence * 100) + '%';
      return '<div class="det-item"><span class="det-lbl' + (isT ? ' tgt' : '') + '">' + lbl +
             '</span><span class="det-conf">' + conf + '</span></div>';
    }).join('') : '<span class="none">none</span>';

  }).catch(function(){
    q('dot').className = '';
    q('hdr-right').textContent = 'disconnected';
  });
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/ui"):
            self._serve_ui()
        elif parsed.path == "/health":
            self._json({"ok": True, "meta": latest_meta})
        elif parsed.path == "/meta":
            self._json(latest_meta)
        elif parsed.path == "/reference_poses":
            self._reference_poses()
        elif parsed.path == "/video_feed":
            self._video_feed()
        elif parsed.path == "/snapshot":
            self._snapshot()
        elif parsed.path.startswith("/people/") and parsed.path.endswith("/memory"):
            self._get_person_memory(parsed)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/learn_face":
            self._learn_face(parsed)
        elif parsed.path == "/collision":
            self._set_collision()
        elif parsed.path == "/pantilt":
            self._handle_manual_pantilt()
        elif parsed.path == "/settings":
            self._set_settings()
        elif parsed.path == "/tracking":
            self._set_tracking(parsed)
        elif parsed.path == "/guard":
            self._set_guard()
        elif parsed.path.startswith("/people/") and parsed.path.endswith("/remember"):
            self._remember_person(parsed)
        elif parsed.path.startswith("/people/") and parsed.path.endswith("/preference"):
            self._set_person_preference(parsed)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        log.debug(fmt, *args)

    def _serve_ui(self) -> None:
        data = _ui_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_manual_pantilt(self) -> None:
        payload = self._read_json_body()
        robot = active_robot
        if robot is None:
            self._json({"ok": False, "error": "robot_not_ready"}, status=503)
            return
        ctrl = active_controller
        if payload.get("center"):
            if ctrl is not None:
                ctrl._estimated_pan = 0
                ctrl._estimated_tilt = 0
                ctrl._commanded_pan = 0.0
                ctrl._commanded_tilt = 0.0
            ok = robot.pantilt({"mode": "absolute", "pan": 0, "tilt": 0, "spd": 0, "acc": 0})
        else:
            pan_delta = int(payload.get("pan", 0))
            tilt_delta = int(payload.get("tilt", 0))
            if ctrl is not None:
                if ctrl.config.pan_invert:
                    pan_delta = -pan_delta
                if ctrl.config.tilt_invert:
                    tilt_delta = -tilt_delta
                new_pan = max(ctrl.config.estimated_pan_min,
                              min(ctrl.config.estimated_pan_max,
                                  ctrl._estimated_pan + pan_delta))
                new_tilt = max(ctrl.config.estimated_tilt_min,
                               min(ctrl.config.estimated_tilt_max,
                                   ctrl._estimated_tilt + tilt_delta))
                ctrl._estimated_pan = new_pan
                ctrl._estimated_tilt = new_tilt
                ok = robot.pantilt({"mode": "absolute", "pan": int(new_pan), "tilt": int(new_tilt), "spd": 0, "acc": 0})
            else:
                ok = False
        self._json({"ok": ok})

    def _set_settings(self) -> None:
        payload = self._read_json_body()
        updated: dict = {}

        def _apply(obj, key: str, typ):
            if key in payload:
                setattr(obj, key, typ(payload[key]))
                updated[key] = getattr(obj, key)

        _apply(vision_config, "pose_enabled", bool)
        _apply(vision_config, "pose_interval_frames", int)
        _apply(vision_config, "pose_score_threshold", float)
        _apply(vision_config, "pose_filter_persons", bool)
        _apply(vision_config, "jpeg_quality", int)

        _apply(tracking_config, "score_threshold", float)
        _apply(tracking_config, "person_score_threshold", float)

        _apply(face_config, "face_interval_frames", int)
        det = active_face_detector
        if det is not None and "face_detection_enabled" in payload:
            det.enabled = bool(payload["face_detection_enabled"])
            updated["face_detection_enabled"] = det.enabled

        rec = active_face_recognizer
        if rec is not None:
            if "face_recognition_enabled" in payload:
                rec.config.enabled = bool(payload["face_recognition_enabled"])
                updated["face_recognition_enabled"] = rec.config.enabled
            _apply(rec.config, "auto_reference_capture_enabled", bool)
            _apply(rec.config, "auto_reference_min_confidence", float)

        ctrl = active_controller
        if ctrl is not None:
            for key, typ in [
                ("follow_forward_speed", int),
                ("follow_steer_gain", float),
                ("follow_target_bbox_ratio", float),
                ("follow_deadzone_ratio", float),
                ("close_target_bbox_ratio", float),
                ("pan_invert", bool),
                ("tilt_invert", bool),
                ("edge_reacquire_enabled", bool),
                ("wheel_enabled", bool),
                ("max_command", int),
                ("min_command", int),
                ("max_command_step", int),
                ("command_interval_seconds", float),
                ("settle_enter_degrees", float),
                ("settle_exit_degrees", float),
                ("estimated_pan_min", int),
                ("estimated_pan_max", int),
                ("estimated_tilt_min", int),
                ("estimated_tilt_max", int),
                ("pan_estimate_scale", float),
                ("tilt_estimate_scale", float),
                ("pan_limit_margin", int),
            ]:
                _apply(ctrl.config, key, typ)

        self._json({"ok": True, "updated": updated})

    def _set_collision(self) -> None:
        payload = self._read_json_body()
        enabled = payload.get("enabled")
        height_threshold = payload.get("height_threshold")
        center_zone_fraction = payload.get("center_zone_fraction")
        if enabled is not None:
            collision_guard.enabled = bool(enabled)
        if height_threshold is not None:
            collision_guard.height_threshold = float(height_threshold)
        if center_zone_fraction is not None:
            collision_guard.center_zone_fraction = float(center_zone_fraction)
        self._json({
            "ok": True,
            "collision_avoidance_enabled": collision_guard.enabled,
            "collision_height_threshold": collision_guard.height_threshold,
            "collision_center_zone_fraction": collision_guard.center_zone_fraction,
        })

    def _set_tracking(self, parsed) -> None:
        payload = self._read_json_body()
        enabled = payload.get("enabled")
        follow = payload.get("follow")
        ctrl = active_controller
        if ctrl is None:
            self._json({"ok": False, "error": "controller_not_ready"}, status=503)
            return
        if enabled is not None:
            ctrl.config.enabled = bool(enabled)
        if "target_identity" in payload:
            tracking_config.target_identity = str(payload["target_identity"]) if payload["target_identity"] else None
        if follow is not None:
            ctrl.config.follow_enabled = bool(follow)
        self._json({
            "ok": True,
            "tracking_enabled": ctrl.config.enabled,
            "follow_enabled": ctrl.config.follow_enabled,
            "target_identity": tracking_config.target_identity,
        })

    def _learn_face(self, parsed) -> None:
        payload = self._read_json_body()
        params = parse_qs(parsed.query)
        name = (
            payload.get("name")
            or payload.get("identity")
            or _first_query_value(params, "name")
            or _first_query_value(params, "identity")
        )
        if not name:
            self._json({"ok": False, "error": "missing_name"}, status=400)
            return

        with state_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            detections = [replace(det) for det in latest_detections]

        if frame is None:
            self._json({"ok": False, "error": "no_frame_available"}, status=409)
            return

        with recognition_lock:
            recognizer = active_face_recognizer
            if recognizer is None:
                self._json({"ok": False, "error": "recognizer_not_ready"}, status=503)
                return
            result = recognizer.enroll_face(str(name), frame, detections)
            brain_client = active_vault_memory_client
            if brain_client is not None:
                brain_client.upload_face_reference(result)

        status = 200 if result.get("ok") else 409
        self._json(result, status=status)

    def _reference_poses(self) -> None:
        with recognition_lock:
            recognizer = active_face_recognizer
            if recognizer is None:
                self._json({"ok": False, "error": "recognizer_not_ready"}, status=503)
                return
            brain_client = active_vault_memory_client
            if brain_client is not None and brain_client.sync_face_references_if_due(force=True):
                recognizer.reload_from_disk()
            coverage = recognizer.reference_pose_coverage()
        self._json({"ok": True, "reference_pose_coverage": coverage})

    def _get_person_memory(self, parsed) -> None:
        identity = _person_identity_from_path(parsed.path, "memory")
        with recognition_lock:
            store = active_person_memory_store
            if store is None:
                self._json({"ok": False, "error": "person_memory_not_ready"}, status=503)
                return
            result = store.get_profile(identity)
        self._json(result, status=200 if result.get("ok") else 400)

    def _set_guard(self) -> None:
        global guard_mode_enabled, _guard_alerted_this_target
        payload = self._read_json_body()
        if "enabled" in payload:
            guard_mode_enabled = bool(payload["enabled"])
            if not guard_mode_enabled:
                _guard_alerted_this_target = False
        self._json({"ok": True, "guard_mode": guard_mode_enabled})

    def _remember_person(self, parsed) -> None:
        identity = _person_identity_from_path(parsed.path, "remember")
        payload = self._read_json_body()
        key = payload.get("key")
        value = payload.get("value")
        memory_type = payload.get("type", "fact")
        source = payload.get("source", "user")
        confidence = payload.get("confidence", 1.0)
        if not key:
            self._json({"ok": False, "error": "missing_key"}, status=400)
            return
        with recognition_lock:
            store = active_person_memory_store
            if store is None:
                self._json({"ok": False, "error": "person_memory_not_ready"}, status=503)
                return
            result = store.remember(identity, str(memory_type), str(key), value, str(source), float(confidence))
        self._json(result, status=200 if result.get("ok") else 400)

    def _set_person_preference(self, parsed) -> None:
        identity = _person_identity_from_path(parsed.path, "preference")
        payload = self._read_json_body()
        key = payload.get("key")
        value = payload.get("value")
        source = payload.get("source", "user")
        if not key:
            self._json({"ok": False, "error": "missing_key"}, status=400)
            return
        with recognition_lock:
            store = active_person_memory_store
            if store is None:
                self._json({"ok": False, "error": "person_memory_not_ready"}, status=503)
                return
            result = store.set_preference(identity, str(key), value, str(source))
        self._json(result, status=200 if result.get("ok") else 400)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video_feed(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with state_lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                break

    def _snapshot(self) -> None:
        with state_lock:
            frame = latest_jpeg
        if frame is None:
            self.send_error(503, "no frame available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)


def _first_query_value(params: dict, key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def _person_identity_from_path(path: str, suffix: str) -> str:
    prefix = "/people/"
    if not path.startswith(prefix) or not path.endswith(f"/{suffix}"):
        return ""
    return unquote(path[len(prefix): -len(f"/{suffix}")].strip("/"))


def main() -> None:
    args = parse_args()
    threading.Thread(target=run_vision, args=(args,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Vision service listening on http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
