#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_node.chat_log import ChatLog
from camera_node.faces import FaceRuntime
from camera_node.guard import dispatch_guard_alert as _dispatch_guard_alert
from camera_node.inference import default_labels_path
from camera_node.inference import inference_callback
from camera_node.inference import load_labels
from camera_node.inference import resolve_hailo
from camera_node.inference import resolve_pose_estimator as _resolve_pose_estimator
from camera_node.media import save_clip as _camera_save_clip
from camera_node.media import save_snapshot as _camera_save_snapshot
from camera_node.settings import apply_settings as _camera_apply_settings
from light_node.runtime import AutoLightRuntime
from luhkas_node.chat_context import build_presence_payload
from luhkas_node.ui import ui_html as _ui_html
from luhkas_node.wakeword import is_wakeword_only as _is_wakeword_only
from luhkas_node.wakeword import response as _wakeword_response
from pantilt_node.runtime import apply_settings as _pantilt_apply_settings
from pantilt_node.runtime import dispatch_robot_command
from pantilt_node.runtime import handle_manual_pantilt as _pantilt_handle_manual
from pantilt_node.runtime import set_tracking as _pantilt_set_tracking
from rover_node.gamepad import GamepadRuntime
from rover_node.runtime import apply_settings as _rover_apply_settings
from rover_node.runtime import handle_manual_move as _rover_handle_manual_move
from rover_node.runtime import set_collision as _rover_set_collision

try:
    from luhkas_node.local_commands import handle as _local_command_handle
    from luhkas_node.local_commands import capabilities as _local_command_capabilities
except ImportError:
    _local_command_handle = None
    _local_command_capabilities = None

import cv2
import numpy as np

from scout.vault_memory import BrainMemoryClient
from scout.collision import CollisionGuard
from scout.behavior import BehaviorState, BehaviorStateMachine, BehaviorConfig
from scout.config import VaultMemoryConfig, CollisionConfig, FaceDetectionConfig, FaceRecognitionConfig, GuardConfig, MotionConfig, PersonMemoryConfig, SearchConfig, TelemetryConfig, TrackingConfig, VisionConfig
from scout.face_detection import FaceDetector
from scout.face_recognition import FaceRecognizer
from scout.motion import PanTiltController
from scout.person_memory import PersonMemoryStore
from scout.pose import PoseEstimator, apply_pose_aim, draw_poses
from scout.robot_client import RobotClient
from scout.search import SearchController
from scout.tracking import SimpleTracker
from scout.vision import bbox_iou, draw_detections, is_face_label, letterbox, parse_hailo_detections


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("vision_service")

_VAULT_CHAT_URL = os.environ.get("VAULT_CHAT_URL", "http://10.10.1.1:7000").rstrip("/")

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
frame_history = deque(maxlen=360)
active_face_detector: FaceDetector | None = None
active_face_recognizer: FaceRecognizer | None = None
active_person_memory_store: PersonMemoryStore | None = None
active_vault_memory_client: BrainMemoryClient | None = None
active_controller = None  # set by run_vision(); used by /tracking endpoint
active_robot: RobotClient | None = None  # set by run_vision(); used by /pantilt endpoint
guard_mode_enabled: bool = False
guard_alert_last_sent: dict = {}
guard_alert_lock = threading.Lock()
guard_alerts_sent_count: int = 0
light_runtime = AutoLightRuntime(vision_config)
IDENTITY_PROMPT_TEXT = os.environ.get("SCOUT_IDENTITY_PROMPT_TEXT", "Who are you?")
IDENTITY_PROMPT_REPEAT_SECONDS = float(os.environ.get("SCOUT_IDENTITY_PROMPT_REPEAT_SECONDS", "45"))
IDENTITY_PROMPT_COMPLETE_GRACE_SECONDS = float(os.environ.get("SCOUT_IDENTITY_PROMPT_COMPLETE_GRACE_SECONDS", "20"))
gamepad_runtime: GamepadRuntime | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = Path(os.environ.get("SCOUT_CAPTURE_DIR", str(PROJECT_ROOT / "captures"))).expanduser()
CHAT_LOG_MAX = int(os.environ.get("SCOUT_CHAT_LOG_MAX", "0"))
CHAT_LOG_PATH = Path(os.environ.get("SCOUT_CHAT_LOG_PATH", str(CAPTURE_DIR / "chat_session.jsonl"))).expanduser()
if not CHAT_LOG_PATH.is_absolute():
    CHAT_LOG_PATH = PROJECT_ROOT / CHAT_LOG_PATH
UNKNOWN_FACE_DIR = Path(os.environ.get("SCOUT_UNKNOWN_FACE_DIR", str(PROJECT_ROOT / "config" / "unknown_faces"))).expanduser()
if not UNKNOWN_FACE_DIR.is_absolute():
    UNKNOWN_FACE_DIR = PROJECT_ROOT / UNKNOWN_FACE_DIR
chat_log = ChatLog(CHAT_LOG_PATH, CHAT_LOG_MAX)
face_runtime = FaceRuntime(
    face_config,
    recognition_config,
    UNKNOWN_FACE_DIR,
    lambda *args, **kwargs: _chat_log_add(*args, **kwargs),
    lambda: active_vault_memory_client,
)
face_runtime.configure_identity_prompt(IDENTITY_PROMPT_TEXT, IDENTITY_PROMPT_COMPLETE_GRACE_SECONDS)
latest_meta = {
    "ok": False,
    "frame_id": 0,
    "frame_shape": None,
    "detections": [],
    "target_id": None,
    "robot_command_ok": None,
}

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

def resolve_pose_estimator(args: argparse.Namespace) -> PoseEstimator | None:
    return _resolve_pose_estimator(args, vision_config, PoseEstimator)


def run_vision(args: argparse.Namespace) -> None:
    labels_path = args.labels or default_labels_path(args.hailo_apps_dir)
    labels = load_labels(labels_path)
    tracker = SimpleTracker(tracking_config)
    controller = PanTiltController(motion_config)
    search_controller = SearchController(search_config)
    behavior = BehaviorStateMachine(BehaviorConfig())
    robot = RobotClient(args.robot_api_url)
    cfg_telemetry = TelemetryConfig()
    if cfg_telemetry.enabled:
        threading.Thread(
            target=_telemetry_poll_loop,
            args=(args.robot_api_url, tracker, cfg_telemetry),
            daemon=True,
        ).start()
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
    global active_controller, active_robot, guard_alerts_sent_count, gamepad_runtime
    global latest_jpeg, latest_frame, latest_detections, latest_reference_captures, latest_meta
    active_controller = controller
    active_robot = robot
    if gamepad_runtime is None:
        gamepad_runtime = GamepadRuntime(
            lambda: active_robot,
            lambda: active_controller,
            _save_snapshot,
            _save_clip,
            _gamepad_toggle_light,
            _adjust_gamepad_light,
        )
        threading.Thread(target=gamepad_runtime.loop, daemon=True).start()

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
            light_level = light_runtime.update_auto(preprocessed, robot, vision_config)

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
            has_person = any(d.label == "person" for d in detections)
            face_detections = []
            if has_person and frame_id % max(1, face_config.interval_frames) == 0:
                face_detections = face_runtime.detect_faces_for_people(face_detector, preprocessed, detections)
                face_detections = face_runtime.filter_faces_inside_people(face_detections, detections)
            all_detections = [*detections, *face_detections]
            reference_captures = []
            if has_person and frame_id % max(1, recognition_config.interval_frames) == 0:
                with recognition_lock:
                    all_detections = face_recognizer.recognize_faces(preprocessed, all_detections)
                    all_detections = face_runtime.filter_faces_inside_people(all_detections, all_detections)
                    # Sync from brain when we see an unknown face rather than on a fixed timer.
                    has_unknown = any(
                        getattr(d, "identity", None) == face_recognizer.config.unknown_label
                        for d in all_detections
                    )
                    if has_unknown and vault_memory_client.sync_on_unknown_identity():
                        face_recognizer.reload_from_disk()
                        all_detections = face_recognizer.recognize_faces(preprocessed, all_detections)
                        all_detections = face_runtime.filter_faces_inside_people(all_detections, all_detections)
                    reference_captures = face_recognizer.auto_capture_missing_references(preprocessed, all_detections)
                    for capture in reference_captures:
                        vault_memory_client.upload_face_reference(capture)
            face_runtime.attach_face_identities_to_people(all_detections)
            tracked = tracker.update(all_detections)
            face_runtime.update_unknown_face_groups(preprocessed, tracked)
            tracker.hydrate_person_memory(person_memory_store)
            target = tracker.select_person_target(input_w, input_h)
            has_tracked_person = any(d.label == "person" for d in tracked)
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
            if vision_config.pose_filter_persons and poses:
                tracked = _filter_persons_by_pose(tracked, poses)
                if target is not None and all(det.id != target.id for det in tracked):
                    target = None
            target = apply_pose_aim(target, poses, vision_config.pose_joint_threshold)
            target, identity_queue = face_runtime.select_identity_prompt_target(
                tracker,
                target,
                tracked,
                frame_id,
                input_w,
                input_h,
                controller.config.enabled and not _gamepad_enabled(),
            )

            if guard_mode_enabled:
                _now = time.monotonic()
                with guard_alert_lock:
                    for _det in tracked:
                        if _det.label != "person":
                            continue
                        _key = _det.identity or f"unknown_{_det.id}"
                        if _now - guard_alert_last_sent.get(_key, 0.0) > guard_config.alert_cooldown_seconds:
                            guard_alert_last_sent[_key] = _now
                            guard_alerts_sent_count += 1
                            _snap = latest_jpeg
                            threading.Thread(
                                target=_dispatch_guard_alert,
                                args=(_det.identity, _det.confidence, _snap, guard_config),
                                daemon=True,
                            ).start()

            if target is not None:
                last_target_vx = getattr(target, "vx", 0.0)
                search_controller.on_target_acquired()
            else:
                if guard_mode_enabled or controller.config.enabled:
                    search_controller.on_target_lost(controller._estimated_pan, last_target_vx)

            obstacle_detected = (
                controller.config.wheel_enabled
                and collision_guard.check(tracked, tracker.selected_target_id, input_w, input_h)
            )

            bstate = behavior.update(
                target=target,
                tracking_enabled=controller.config.enabled,
                guard_enabled=guard_mode_enabled,
                collision_blocked=obstacle_detected,
                manual_enabled=_gamepad_enabled(),
                search_enabled=bool(search_controller.config.enabled),
            )

            command = None
            command_ok = None
            if bstate == BehaviorState.FOLLOWING:
                command = controller.command_for_target(target, input_w, input_h)
                if command is not None:
                    command_ok = dispatch_robot_command(robot, command)
                if controller.config.follow_enabled and not obstacle_detected:
                    follow_cmd = controller.wheel_follow_command(target, input_w, input_h)
                    if follow_cmd is not None:
                        robot.move(follow_cmd["x"], follow_cmd["z"])
                    else:
                        robot.move(0, 0)
                else:
                    robot.move(0, 0)

            elif bstate == BehaviorState.SEARCHING:
                search_cmd = search_controller.search_command(controller._estimated_pan, controller._estimated_tilt)
                if search_cmd is not None:
                    robot.pantilt(search_cmd)
                    controller.notify_external_pantilt(search_cmd["pan"], search_cmd["tilt"])
                robot.move(0, 0)

            elif bstate == BehaviorState.AVOIDING:
                robot.move(-200, 0)
                robot.pantilt(controller.center_command())

            elif bstate == BehaviorState.MANUAL:
                pass

            elif bstate in (BehaviorState.IDLE, BehaviorState.GUARDING):
                robot.move(0, 0)
                if behavior.time_in_state() < 0.1:
                    robot.pantilt(controller.center_command())

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
                if encode_ok:
                    latest_jpeg = encoded.tobytes()
                    frame_history.append((time.time(), latest_jpeg))
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
                    "target_label": tracking_config.target_label,
                    "person_score_threshold": tracking_config.person_score_threshold,
                    "face_detection_enabled": bool(face_detector.enabled),
                    "face_interval_frames": face_config.interval_frames,
                    "face_min_neighbors": face_config.min_neighbors,
                    "face_person_upper_ratio": face_config.person_upper_ratio,
                    "face_min_person_height_ratio": face_config.min_person_height_ratio,
                    "face_max_person_height_ratio": face_config.max_person_height_ratio,
                    "face_intro_min_seen_frames": face_config.intro_min_seen_frames,
                    "unknown_face_dir": str(UNKNOWN_FACE_DIR),
                    "unknown_face_groups": face_runtime.unknown_face_group_status(),
                    "face_recognition_enabled": bool(face_recognizer.enabled),
                    "face_recognition_method": face_recognizer.method,
                    "auto_reference_capture_enabled": bool(face_recognizer.config.auto_reference_capture_enabled),
                    "auto_reference_min_confidence": face_recognizer.config.auto_reference_min_confidence,
                    "follow_forward_speed": controller.config.follow_forward_speed,
                    "follow_steer_gain": controller.config.follow_steer_gain,
                    "follow_target_bbox_ratio": controller.config.follow_target_bbox_ratio,
                    "follow_deadzone_ratio": controller.config.follow_deadzone_ratio,
                    "close_target_bbox_ratio": controller.config.close_target_bbox_ratio,
                    **light_runtime.status(vision_config),
                    "ambient_light_level": light_level,
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
                    "identity_prompt": dict(face_runtime.latest_identity_prompt) if face_runtime.latest_identity_prompt else None,
                    "identity_prompt_queue": identity_queue,
                    "chat_log": _chat_log_snapshot(25),
                    "chat_log_endpoint": "GET /chat_log?limit=200",
                    "gamepad": _gamepad_status(),
                    "learn_face_endpoint": "POST /learn_face?name=<your-name>&face_id=<face-id>",
                    "target_identity": tracking_config.target_identity,
                    "target_id": tracker.selected_target_id,
                    "target": target.to_json() if target else None,
                    "target_state": (
                        "manual" if _gamepad_enabled()
                        else
                        "predicted" if target and target.predicted
                        else "visible" if target
                        else search_controller.phase if search_controller.config.enabled and search_controller.phase != "idle"
                        else "none"
                    ),
                    "search_phase": search_controller.phase if search_controller.config.enabled else "idle",
                    "search_movement_enabled": bool(search_controller.config.enabled),
                    "guard_mode": guard_mode_enabled,
                    "tracking_enabled": controller.config.enabled,
                    "follow_enabled": controller.config.follow_enabled,
                    "ego_motion": controller.ego_motion(),
                    "robot_command_ok": command_ok,
                    "robot_command": command,
                    "collision_blocked": obstacle_detected,
                    "collision_avoidance_enabled": collision_guard.enabled,
                    "behavior": {
                        "state": behavior.state_name,
                        "time_in_state": round(behavior.time_in_state(), 1),
                    },
                    "guard": {
                        "enabled": guard_mode_enabled,
                        "alerts_sent": guard_alerts_sent_count,
                        "last_alert_ts": max(guard_alert_last_sent.values(), default=0.0),
                    },
                    "collision_height_threshold": collision_guard.height_threshold,
                    "collision_center_zone_fraction": collision_guard.center_zone_fraction,
                }
    finally:
        cap.release()
        hailo.close()
        if pose_estimator is not None:
            pose_estimator.close()


def _chat_log_add(role: str, text: str, source: str = "chat", **meta) -> dict:
    return chat_log.add(role, text, source=source, **meta)


def _chat_log_snapshot(limit: int | None = None) -> list[dict]:
    return chat_log.snapshot(limit)


def _init_chat_log_file() -> None:
    chat_log.init_file()


def _gamepad_status() -> dict:
    return gamepad_runtime.status() if gamepad_runtime is not None else {
        "enabled": False,
        "connected": False,
        "device": None,
        "last_event": 0.0,
        "last_action": None,
        "axes": {},
        "buttons": {},
    }


def _gamepad_enabled() -> bool:
    return bool(gamepad_runtime and gamepad_runtime.enabled())


def _gamepad_toggle_light() -> None:
    robot = active_robot
    if robot is None:
        return None
    return light_runtime.toggle_manual(robot)


def _adjust_gamepad_light(delta: int) -> None:
    robot = active_robot
    if robot is None:
        return None
    return light_runtime.adjust_manual(delta, robot)


def _save_snapshot() -> dict:
    with state_lock:
        frame = latest_jpeg
    return _camera_save_snapshot(
        CAPTURE_DIR,
        frame,
        lambda action: gamepad_runtime and gamepad_runtime.set_status(True, gamepad_runtime.device(), action),
    )


def _save_clip(seconds: float = 8.0) -> dict:
    with state_lock:
        frames = list(frame_history)
    return _camera_save_clip(
        CAPTURE_DIR,
        frames,
        seconds,
        lambda action: gamepad_runtime and gamepad_runtime.set_status(True, gamepad_runtime.device(), action),
    )


def _set_manual_controller_enabled(enabled: bool) -> None:
    if gamepad_runtime is not None:
        gamepad_runtime.set_manual_enabled(enabled, search_config)


# Calibration procedure for gyro scales: hold the robot still to confirm gyro baseline
# is ~0, then command a known pan angle and measure how many pixels the frame shifted
# per gyro unit — that ratio is the scale. Leave SCOUT_EGO_MOTION_ENABLED=0 until
# calibrated.
def _telemetry_poll_loop(robot_api_url: str, tracker, cfg: TelemetryConfig) -> None:
    import requests as _requests
    while True:
        try:
            r = _requests.get(f"{robot_api_url}/telemetry", timeout=0.3)
            tel = r.json()
            pan_delta  = tel.get("gyro", {}).get("z", 0.0) * cfg.gyro_pan_scale
            tilt_delta = tel.get("gyro", {}).get("x", 0.0) * cfg.gyro_tilt_scale
            tracker.set_ego_motion({"pan": pan_delta, "tilt": tilt_delta})
        except Exception:
            pass
        time.sleep(cfg.poll_interval)


def _filter_persons_by_pose(detections: list, poses: list, iou_threshold: float = 0.05) -> list:
    result = []
    for det in detections:
        if det.label != "person":
            result.append(det)
            continue
        if any(bbox_iou(det.bbox, pose.bbox) >= iou_threshold for pose in poses):
            result.append(det)
    return result


def _latest_frame_detections() -> tuple[np.ndarray | None, list]:
    with state_lock:
        frame = None if latest_frame is None else latest_frame.copy()
        detections = [replace(det) for det in latest_detections]
    return frame, detections


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/ui"):
            self._serve_ui()
        elif parsed.path == "/health":
            self._json({"ok": True, "meta": latest_meta})
        elif parsed.path == "/capabilities":
            self._capabilities()
        elif parsed.path == "/meta":
            self._json(latest_meta)
        elif parsed.path == "/chat_log":
            params = parse_qs(parsed.query)
            try:
                limit = int((params.get("limit") or ["200"])[0])
            except (TypeError, ValueError):
                limit = 200
            self._json({"ok": True, "entries": _chat_log_snapshot(limit), "path": str(CHAT_LOG_PATH)})
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
        elif parsed.path == "/move":
            self._handle_manual_move()
        elif parsed.path == "/settings":
            self._set_settings()
        elif parsed.path == "/tracking":
            self._set_tracking(parsed)
        elif parsed.path == "/guard":
            self._set_guard()
        elif parsed.path == "/chat":
            self._chat()
        elif parsed.path == "/clip":
            self._clip()
        elif parsed.path == "/snapshot":
            self._save_snapshot()
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

    def _capabilities(self) -> None:
        local = (
            _local_command_capabilities()
            if _local_command_capabilities is not None
            else {"ok": False, "error": "local_commands_unavailable", "commands": []}
        )
        self._json({
            "ok": True,
            "node": "scout",
            "owner_node": "scout",
            "target_node": "scout",
            "scope": "scout_only",
            "description": "Capabilities exposed here are Scout-only and execute on Scout hardware/services.",
            "capabilities": [local],
            "commands": local.get("commands", []) if isinstance(local, dict) else [],
            "modules": local.get("module_status", {}) if isinstance(local, dict) else {},
        })

    def _video_feed(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with state_lock:
                    frame = latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                time.sleep(0.033)
        except Exception:
            pass

    def _snapshot(self) -> None:
        with state_lock:
            frame = latest_jpeg
        if frame is None:
            self.send_error(503, "No frame available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def _save_snapshot(self) -> None:
        result = _save_snapshot()
        self._json(result, status=200 if result.get("ok") else 409)

    def _clip(self) -> None:
        body = self._read_json()
        if body is None:
            return
        try:
            seconds = float(body.get("seconds", 8.0))
        except (TypeError, ValueError):
            seconds = 8.0
        seconds = max(1.0, min(30.0, seconds))
        result = _save_clip(seconds)
        self._json(result, status=200 if result.get("ok") else 409)

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
        parts = parsed.path.strip("/").split("/")
        person_id = unquote(parts[1]) if len(parts) >= 3 else None
        if not person_id:
            self.send_error(400)
            return
        with recognition_lock:
            store = active_person_memory_store
        if store is None:
            self._json({"ok": False, "error": "not_initialized"}, status=503)
            return
        memory = store.get_profile(person_id)
        self._json(memory if memory else {"ok": False, "error": "not_found"})

    def _learn_face(self, parsed) -> None:
        body = self._read_json()
        if body is None:
            return
        params = parse_qs(parsed.query)
        name = (
            body.get("name")
            or body.get("identity")
            or (params.get("name") or [""])[0]
            or (params.get("identity") or [""])[0]
        )
        name = str(name).strip()
        if not name:
            self._json({"ok": False, "error": "missing_name"}, status=400)
            return
        face_id_raw = (
            body.get("face_id")
            or body.get("faceId")
            or body.get("detection_id")
            or (params.get("face_id") or [""])[0]
            or (params.get("faceId") or [""])[0]
            or (params.get("detection_id") or [""])[0]
        )
        try:
            face_id = int(face_id_raw)
        except (TypeError, ValueError):
            face_id = None
        with state_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            detections = [replace(det) for det in latest_detections]
        if frame is None:
            self._json({"ok": False, "error": "no_frame_available"}, status=409)
            return
        face = face_runtime.target_face_detection(detections, face_id)
        if face is None:
            available = [det.id for det in detections if is_face_label(det.label) and det.id is not None]
            self._json({
                "ok": False,
                "error": "target_face_required",
                "available_face_ids": available,
            }, status=409)
            return
        with recognition_lock:
            recognizer = active_face_recognizer
            if recognizer is None:
                self._json({"ok": False, "error": "recognizer_not_ready"}, status=503)
                return
            result = recognizer.enroll_face(name, frame, face)
            if result.get("ok"):
                face_runtime.mark_identity_prompt_learned(face_id, str(result.get("identity") or name))
                promote_group_id = (
                    getattr(face, "vault_face_group_id", None)
                    or getattr(face, "face_group_id", None)
                )
                result["unknown_group_promotion"] = face_runtime.promote_unknown_face_group(
                    promote_group_id,
                    str(result.get("identity") or name),
                    recognizer,
                )
            brain_client = active_vault_memory_client
            if brain_client is not None:
                brain_client.upload_face_reference(result)

        status = 200 if result.get("ok") else 409
        self._json(result, status=status)

    def _set_collision(self) -> None:
        body = self._read_json()
        if body is None:
            return
        payload, status = _rover_set_collision(body, collision_guard)
        self._json(payload, status=status)

    def _handle_manual_pantilt(self) -> None:
        body = self._read_json()
        if body is None:
            return
        payload, status = _pantilt_handle_manual(body, active_robot, active_controller)
        self._json(payload, status=status)

    def _handle_manual_move(self) -> None:
        body = self._read_json()
        if body is None:
            return
        payload, status = _rover_handle_manual_move(body, active_robot)
        self._json(payload, status=status)

    def _set_tracking(self, parsed) -> None:
        body = self._read_json()
        if body is None:
            return
        payload, status = _pantilt_set_tracking(
            body,
            active_controller,
            tracking_config,
            lambda: _set_manual_controller_enabled(False),
        )
        self._json(payload, status=status)

    def _set_guard(self) -> None:
        global guard_mode_enabled, guard_alert_last_sent
        body = self._read_json()
        if body is None:
            return
        enabled = bool(body.get("enabled", False))
        guard_mode_enabled = enabled
        if not enabled:
            with guard_alert_lock:
                guard_alert_last_sent = {}
        self._json({"ok": True, "guard_enabled": guard_mode_enabled})

    def _chat(self) -> None:
        body = self._read_json()
        if body is None:
            return
        message = str(body.get("message", "")).strip()
        if not message:
            self.send_error(400, "message required")
            return
        _chat_log_add("user", message, source="chat_input")
        if _is_wakeword_only(message):
            response = _wakeword_response()
            _chat_log_add("assistant", response["message"], source="wakeword")
            self._json({"ok": True, "response": response})
            return
        identity_response = face_runtime.learn_identity_from_active_prompt(
            message,
            _latest_frame_detections,
            lambda: active_face_recognizer,
            recognition_lock,
        )
        if identity_response is not None:
            status = 200 if identity_response.get("ok") else 409
            response = identity_response.get("response") or {}
            text = response.get("message") or identity_response.get("error") or json.dumps(identity_response)
            _chat_log_add(
                "assistant" if identity_response.get("ok") else "error",
                text,
                source="identity_response",
                face_id=response.get("face_id"),
                skipped=response.get("skipped"),
                waiting_for_identity=response.get("waiting_for_identity"),
            )
            self._json(identity_response, status=status)
            return
        # Chat responses go through Vault so deterministic actions and generated
        # wording share the same ResponseComposer path. Node-local command
        # modules still advertise capabilities and execute when Vault routes
        # actions back to this service.
        try:
            import requests as _requests
            vault_payload = build_presence_payload(message, _chat_log_snapshot(), "scout")
            resp = _requests.post(
                f"{_VAULT_CHAT_URL}/presence/message",
                json=vault_payload,
                timeout=30,
            )
            payload = resp.json()
            response = payload.get("response") if isinstance(payload, dict) else None
            if response is None:
                response = payload
            _chat_log_add(
                "assistant",
                response.get("tts") or response.get("message") or json.dumps(response),
                source="vault_chat",
            )
            self._json({"ok": True, "response": response})
        except Exception as exc:
            _chat_log_add("error", str(exc), source="vault_chat")
            self._json({"ok": False, "error": str(exc)}, status=503)

    def _set_settings(self) -> None:
        body = self._read_json()
        if body is None:
            return
        _pantilt_apply_settings(body, active_controller, tracking_config, search_config)
        _rover_apply_settings(body, active_controller)
        if "manual_controller_enabled" in body:
            _set_manual_controller_enabled(bool(body["manual_controller_enabled"]))
        if "camera_light_auto_enabled" in body:
            light_runtime.set_auto_enabled(bool(body["camera_light_auto_enabled"]))
        if "camera_light_enabled" in body:
            light_runtime.set_enabled(bool(body["camera_light_enabled"]), active_robot)
        if "camera_light_brightness" in body:
            light_runtime.set_brightness(int(float(body["camera_light_brightness"])), active_robot)
        if "camera_light_trigger_threshold" in body:
            threshold = max(10.0, min(160.0, float(body["camera_light_trigger_threshold"])))
            vision_config.camera_light_low_threshold = threshold
            vision_config.camera_light_high_threshold = light_runtime.off_threshold(vision_config)

        _camera_apply_settings(body, vision_config, active_face_detector, active_face_recognizer)
        self._json({"ok": True})

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body or "{}")
        except (json.JSONDecodeError, Exception):
            self.send_error(400, "invalid JSON")
            return None

    def _json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    args = parse_args()
    _init_chat_log_file()
    vision_thread = threading.Thread(target=run_vision, args=(args,), daemon=True)
    vision_thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Vision service listening on http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
