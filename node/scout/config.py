from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class RobotApiConfig:
    host: str = os.environ.get("ROBOT_API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("ROBOT_API_PORT", "5001"))
    serial_port: str = os.environ.get("ROBOT_SERIAL_PORT", "/dev/ttyAMA0")
    baud_rate: int = int(os.environ.get("ROBOT_BAUD_RATE", "115200"))
    brain_url: str = os.environ.get("ROBOT_VAULT_URL", "http://10.10.1.1:7000")
    telemetry_log_enabled: bool = bool(int(os.environ.get("SCOUT_TELEMETRY_LOG_ENABLED", "1")))
    telemetry_db_path: str = os.environ.get("SCOUT_TELEMETRY_DB_PATH", "config/telemetry.db")


@dataclass
class VisionConfig:
    host: str = os.environ.get("SCOUT_VISION_HOST", "0.0.0.0")
    port: int = int(os.environ.get("SCOUT_VISION_PORT", "5000"))
    camera_index: int = int(os.environ.get("SCOUT_CAMERA_INDEX", "0"))
    camera_width: int = int(os.environ.get("SCOUT_CAMERA_WIDTH", "640"))
    camera_height: int = int(os.environ.get("SCOUT_CAMERA_HEIGHT", "480"))
    jpeg_quality: int = int(os.environ.get("SCOUT_JPEG_QUALITY", "65"))
    robot_api_url: str = os.environ.get("ROBOT_API_URL", "http://127.0.0.1:5001")
    hailo_apps_dir: str = os.environ.get("HAILO_APPS_DIR", "/home/luhkas/hailo-apps")
    pose_enabled: bool = os.environ.get("SCOUT_POSE_ENABLED", "1") != "0"
    pose_hef_path: str | None = os.environ.get("SCOUT_POSE_HEF_PATH") or None
    pose_interval_frames: int = int(os.environ.get("SCOUT_POSE_INTERVAL_FRAMES", "3"))
    pose_score_threshold: float = float(os.environ.get("SCOUT_POSE_SCORE_THRESHOLD", "0.45"))
    pose_joint_threshold: float = float(os.environ.get("SCOUT_POSE_JOINT_THRESHOLD", "0.45"))
    pose_filter_persons: bool = os.environ.get("SCOUT_POSE_FILTER_PERSONS", "0") != "0"
    camera_light_auto_enabled: bool = os.environ.get("SCOUT_CAMERA_LIGHT_AUTO_ENABLED", "1") != "0"
    camera_light_auto_brightness: int = int(os.environ.get("SCOUT_CAMERA_LIGHT_AUTO_BRIGHTNESS", "255"))
    camera_light_low_threshold: float = float(os.environ.get("SCOUT_CAMERA_LIGHT_LOW_THRESHOLD", "55"))
    camera_light_high_threshold: float = float(os.environ.get("SCOUT_CAMERA_LIGHT_HIGH_THRESHOLD", "85"))


@dataclass
class FaceDetectionConfig:
    enabled: bool = os.environ.get("SCOUT_FACE_DETECTION_ENABLED", "1") != "0"
    cascade_path: str | None = os.environ.get("SCOUT_FACE_CASCADE_PATH") or None
    interval_frames: int = int(os.environ.get("SCOUT_FACE_INTERVAL_FRAMES", "2"))
    class_id: int = int(os.environ.get("SCOUT_FACE_CLASS_ID", "10000"))
    confidence: float = float(os.environ.get("SCOUT_FACE_CONFIDENCE", "0.85"))
    scale_factor: float = float(os.environ.get("SCOUT_FACE_SCALE_FACTOR", "1.05"))
    min_neighbors: int = int(os.environ.get("SCOUT_FACE_MIN_NEIGHBORS", "3"))
    min_size_px: int = int(os.environ.get("SCOUT_FACE_MIN_SIZE_PX", "32"))
    max_faces: int = int(os.environ.get("SCOUT_FACE_MAX_FACES", "3"))
    person_upper_ratio: float = float(os.environ.get("SCOUT_FACE_PERSON_UPPER_RATIO", "0.55"))
    min_person_height_ratio: float = float(os.environ.get("SCOUT_FACE_MIN_PERSON_HEIGHT_RATIO", "0.08"))
    max_person_height_ratio: float = float(os.environ.get("SCOUT_FACE_MAX_PERSON_HEIGHT_RATIO", "0.50"))
    intro_min_seen_frames: int = int(os.environ.get("SCOUT_FACE_INTRO_MIN_SEEN_FRAMES", "2"))
    unknown_match_threshold: float = float(os.environ.get("SCOUT_FACE_UNKNOWN_MATCH_THRESHOLD", "0.32"))
    unknown_sample_interval_seconds: float = float(os.environ.get("SCOUT_FACE_UNKNOWN_SAMPLE_INTERVAL", "2.0"))
    unknown_max_samples: int = int(os.environ.get("SCOUT_FACE_UNKNOWN_MAX_SAMPLES", "24"))
    unknown_persist_seconds: float = float(os.environ.get("SCOUT_FACE_UNKNOWN_PERSIST_SECONDS", "8.0"))


@dataclass
class FaceRecognitionConfig:
    enabled: bool = os.environ.get("SCOUT_FACE_RECOGNITION_ENABLED", "1") != "0"
    known_faces_dir: str = os.environ.get("SCOUT_KNOWN_FACES_DIR", "config/faces")
    interval_frames: int = int(os.environ.get("SCOUT_FACE_RECOGNITION_INTERVAL_FRAMES", "2"))
    image_size_px: int = int(os.environ.get("SCOUT_FACE_RECOGNITION_IMAGE_SIZE", "128"))
    lbph_threshold: float = float(os.environ.get("SCOUT_FACE_LBPH_THRESHOLD", "72"))
    histogram_threshold: float = float(os.environ.get("SCOUT_FACE_HISTOGRAM_THRESHOLD", "0.62"))
    crop_training_faces: bool = os.environ.get("SCOUT_FACE_CROP_TRAINING", "1") != "0"
    unknown_label: str = os.environ.get("SCOUT_FACE_UNKNOWN_LABEL", "unknown")
    min_training_images_per_person: int = int(os.environ.get("SCOUT_FACE_MIN_TRAINING_IMAGES", "2"))
    reference_pose_buckets: str = os.environ.get("SCOUT_FACE_REFERENCE_POSES", "frontal,left,right,up,down,close,far")
    reference_samples_per_pose: int = int(os.environ.get("SCOUT_FACE_REFERENCE_SAMPLES_PER_POSE", "3"))
    auto_reference_capture_enabled: bool = os.environ.get("SCOUT_FACE_AUTO_REFERENCE_CAPTURE", "1") != "0"
    auto_reference_min_confidence: float = float(os.environ.get("SCOUT_FACE_AUTO_REFERENCE_MIN_CONFIDENCE", "0.35"))
    auto_reference_cooldown_seconds: float = float(os.environ.get("SCOUT_FACE_AUTO_REFERENCE_COOLDOWN", "20"))
    max_auto_reference_samples_per_identity: int = int(os.environ.get("SCOUT_FACE_MAX_AUTO_REFERENCE_SAMPLES", "80"))


@dataclass
class VaultMemoryConfig:
    enabled: bool = os.environ.get("SCOUT_VAULT_MEMORY_ENABLED", "0") != "0"
    url: str = os.environ.get("SCOUT_VAULT_MEMORY_URL", "")
    timeout_seconds: float = float(os.environ.get("SCOUT_VAULT_MEMORY_TIMEOUT", "1.5"))
    face_cache_dir: str = os.environ.get("SCOUT_VAULT_FACE_CACHE_DIR", "config/vault_faces")
    face_sync_interval_seconds: float = float(os.environ.get("SCOUT_VAULT_FACE_SYNC_INTERVAL", "300"))
    prefer_brain_person_memory: bool = os.environ.get("SCOUT_PREFER_VAULT_PERSON_MEMORY", "1") != "0"


@dataclass
class PersonMemoryConfig:
    enabled: bool = os.environ.get("SCOUT_PERSON_MEMORY_ENABLED", "1") != "0"
    people_dir: str = os.environ.get("SCOUT_PEOPLE_DIR", "config/people")
    summary_preference_keys: str = os.environ.get(
        "SCOUT_PERSON_MEMORY_SUMMARY_KEYS",
        "display_name,follow_distance,greeting,auto_follow_allowed,tracking_priority",
    )


@dataclass
class TrackingConfig:
    target_label: str = os.environ.get("SCOUT_TARGET_LABEL", "person")
    score_threshold: float = float(os.environ.get("SCOUT_SCORE_THRESHOLD", "0.45"))
    person_score_threshold: float = float(os.environ.get("SCOUT_PERSON_SCORE_THRESHOLD", "0.70"))
    object_ttl_seconds: float = float(os.environ.get("SCOUT_OBJECT_TTL_SECONDS", "3.5"))
    target_lost_grace_seconds: float = float(os.environ.get("SCOUT_TARGET_LOST_GRACE_SECONDS", "0.45"))
    target_reacquire_seconds: float = float(os.environ.get("SCOUT_TARGET_REACQUIRE_SECONDS", "4.0"))
    max_prediction_seconds: float = float(os.environ.get("SCOUT_MAX_PREDICTION_SECONDS", "2.0"))
    max_match_distance_px: float = float(os.environ.get("SCOUT_MAX_MATCH_DISTANCE_PX", "280"))
    min_match_iou: float = float(os.environ.get("SCOUT_MIN_MATCH_IOU", "0.05"))
    bbox_smoothing: float = float(os.environ.get("SCOUT_BBOX_SMOOTHING", "0.60"))
    velocity_smoothing: float = float(os.environ.get("SCOUT_VELOCITY_SMOOTHING", "0.35"))
    target_switch_margin: float = float(os.environ.get("SCOUT_TARGET_SWITCH_MARGIN", "0.25"))
    max_objects: int = int(os.environ.get("SCOUT_MAX_OBJECTS", "8"))
    memory_ttl_seconds: float = float(os.environ.get("SCOUT_MEMORY_TTL_SECONDS", "45"))
    max_memory_objects: int = int(os.environ.get("SCOUT_MAX_MEMORY_OBJECTS", "100"))
    memory_match_threshold: float = float(os.environ.get("SCOUT_MEMORY_MATCH_THRESHOLD", "0.62"))
    color_match_weight: float = float(os.environ.get("SCOUT_COLOR_MATCH_WEIGHT", "0.22"))
    pan_pixels_per_degree: float = float(os.environ.get("SCOUT_PAN_PIXELS_PER_DEGREE", "5.2"))
    tilt_pixels_per_degree: float = float(os.environ.get("SCOUT_TILT_PIXELS_PER_DEGREE", "4.4"))
    ego_motion_spatial_penalty: float = float(os.environ.get("SCOUT_EGO_MOTION_SPATIAL_PENALTY", "0.45"))
    frame_width: int = int(os.environ.get("SCOUT_TRACKING_FRAME_WIDTH", "640"))
    frame_height: int = int(os.environ.get("SCOUT_TRACKING_FRAME_HEIGHT", "640"))
    wide_angle_compensation: bool = os.environ.get("SCOUT_WIDE_ANGLE_COMPENSATION", "1") != "0"
    fisheye_strength: float = float(os.environ.get("SCOUT_FISHEYE_STRENGTH", "0.55"))
    target_torso_aim_ratio: float = float(os.environ.get("SCOUT_TARGET_TORSO_AIM", "0.10"))
    target_identity: str | None = os.environ.get("SCOUT_TARGET_IDENTITY") or None
    bytetracker_enabled: bool = os.environ.get("SCOUT_BYTETRACKER_ENABLED", "1") != "0"
    bytetracker_track_thresh: float = float(os.environ.get("SCOUT_BYTETRACKER_TRACK_THRESH", "0.1"))
    bytetracker_track_buffer: int = int(os.environ.get("SCOUT_BYTETRACKER_TRACK_BUFFER", "30"))
    bytetracker_match_thresh: float = float(os.environ.get("SCOUT_BYTETRACKER_MATCH_THRESH", "0.9"))
    bytetracker_min_box_area: float = float(os.environ.get("SCOUT_BYTETRACKER_MIN_BOX_AREA", "250"))


@dataclass
class MotionConfig:
    enabled: bool = os.environ.get("SCOUT_TRACKING_ENABLED", "1") != "0"
    wheel_enabled: bool = os.environ.get("SCOUT_WHEEL_ENABLED", "1") != "0"
    command_interval_seconds: float = float(os.environ.get("SCOUT_COMMAND_INTERVAL", "0.12"))
    deadzone_x: float = float(os.environ.get("SCOUT_DEADZONE_X", "0.032"))
    deadzone_y: float = float(os.environ.get("SCOUT_DEADZONE_Y", "0.065"))
    pan_gain: float = float(os.environ.get("SCOUT_PAN_GAIN", "145"))
    tilt_gain: float = float(os.environ.get("SCOUT_TILT_GAIN", "85"))
    camera_horizontal_fov_degrees: float = float(os.environ.get("SCOUT_CAMERA_HORIZONTAL_FOV", "120"))
    camera_vertical_fov_degrees: float = float(os.environ.get("SCOUT_CAMERA_VERTICAL_FOV", "90"))
    angular_pan_gain: float = float(os.environ.get("SCOUT_ANGULAR_PAN_GAIN", "1.7"))
    angular_tilt_gain: float = float(os.environ.get("SCOUT_ANGULAR_TILT_GAIN", "1.4"))
    settle_enter_degrees: float = float(os.environ.get("SCOUT_SETTLE_ENTER_DEGREES", "2.0"))
    settle_exit_degrees: float = float(os.environ.get("SCOUT_SETTLE_EXIT_DEGREES", "2.5"))
    close_target_bbox_ratio: float = float(os.environ.get("SCOUT_CLOSE_TARGET_BBOX_RATIO", "0.65"))
    close_target_command_scale: float = float(os.environ.get("SCOUT_CLOSE_TARGET_COMMAND_SCALE", "0.70"))
    close_target_settle_scale: float = float(os.environ.get("SCOUT_CLOSE_TARGET_SETTLE_SCALE", "1.35"))
    far_target_bbox_ratio: float = float(os.environ.get("SCOUT_FAR_TARGET_BBOX_RATIO", "0.18"))
    far_target_command_scale: float = float(os.environ.get("SCOUT_FAR_TARGET_COMMAND_SCALE", "0.85"))
    far_target_settle_scale: float = float(os.environ.get("SCOUT_FAR_TARGET_SETTLE_SCALE", "1.20"))
    pantilt_mode: str = os.environ.get("SCOUT_PANTILT_MODE", "absolute")
    absolute_pan_gain: float = float(os.environ.get("SCOUT_ABSOLUTE_PAN_GAIN", "28"))
    absolute_tilt_gain: float = float(os.environ.get("SCOUT_ABSOLUTE_TILT_GAIN", "22"))
    absolute_max_step: int = int(os.environ.get("SCOUT_ABSOLUTE_MAX_STEP", "8"))
    absolute_min_step: int = int(os.environ.get("SCOUT_ABSOLUTE_MIN_STEP", "1"))
    absolute_speed: int = int(os.environ.get("SCOUT_ABSOLUTE_PANTILT_SPEED", "0"))
    absolute_accel: int = int(os.environ.get("SCOUT_ABSOLUTE_PANTILT_ACCEL", "0"))
    absolute_distance_gain: float = float(os.environ.get("SCOUT_ABSOLUTE_DISTANCE_GAIN", "0.6"))
    absolute_distance_max_multiplier: float = float(os.environ.get("SCOUT_ABSOLUTE_DISTANCE_MAX_MULTIPLIER", "1.1"))
    min_command: int = int(os.environ.get("SCOUT_MIN_PANTILT_COMMAND", "7"))
    max_command: int = int(os.environ.get("SCOUT_MAX_PANTILT_COMMAND", "56"))
    max_command_step: int = int(os.environ.get("SCOUT_MAX_PANTILT_STEP", "14"))
    command_smoothing: float = float(os.environ.get("SCOUT_COMMAND_SMOOTHING", "0.48"))
    aim_smoothing: float = float(os.environ.get("SCOUT_AIM_SMOOTHING", "0.88"))
    aim_smoothing_x: float = float(os.environ.get("SCOUT_AIM_SMOOTHING_X", "0.88"))
    aim_smoothing_y: float = float(os.environ.get("SCOUT_AIM_SMOOTHING_Y", os.environ.get("SCOUT_AIM_SMOOTHING", "0.80")))
    error_smoothing: float = float(os.environ.get("SCOUT_ERROR_SMOOTHING", "0.88"))
    error_smoothing_x: float = float(os.environ.get("SCOUT_ERROR_SMOOTHING_X", "0.88"))
    error_smoothing_y: float = float(os.environ.get("SCOUT_ERROR_SMOOTHING_Y", os.environ.get("SCOUT_ERROR_SMOOTHING", "0.80")))
    pan_integral_gain: float = float(os.environ.get("SCOUT_PAN_INTEGRAL_GAIN", "0"))
    pan_integral_limit: int = int(os.environ.get("SCOUT_PAN_INTEGRAL_LIMIT", "0"))
    pan_integral_decay: float = float(os.environ.get("SCOUT_PAN_INTEGRAL_DECAY", "0.80"))
    sign_flip_deadband: int = int(os.environ.get("SCOUT_SIGN_FLIP_DEADBAND", "16"))
    reverse_settle_seconds: float = float(os.environ.get("SCOUT_REVERSE_SETTLE_SECONDS", "0.18"))
    target_velocity_lead_seconds: float = float(os.environ.get("SCOUT_TARGET_VELOCITY_LEAD", "0.06"))
    target_velocity_lead_x_seconds: float = float(os.environ.get("SCOUT_TARGET_VELOCITY_LEAD_X", "0.0"))
    target_velocity_lead_y_seconds: float = float(os.environ.get("SCOUT_TARGET_VELOCITY_LEAD_Y", os.environ.get("SCOUT_TARGET_VELOCITY_LEAD", "0.06")))
    pan_estimate_scale: float = float(os.environ.get("SCOUT_PAN_ESTIMATE_SCALE", "0.12"))
    tilt_estimate_scale: float = float(os.environ.get("SCOUT_TILT_ESTIMATE_SCALE", "0.10"))
    relative_speed_x: int = int(os.environ.get("SCOUT_PANTILT_SX", "800"))
    relative_speed_y: int = int(os.environ.get("SCOUT_PANTILT_SY", "450"))
    target_head_offset_ratio: float = float(os.environ.get("SCOUT_TARGET_HEAD_OFFSET", "0.15"))
    estimated_pan_min: int = int(os.environ.get("SCOUT_ESTIMATED_PAN_MIN", "-180"))
    estimated_pan_max: int = int(os.environ.get("SCOUT_ESTIMATED_PAN_MAX", "180"))
    estimated_tilt_min: int = int(os.environ.get("SCOUT_ESTIMATED_TILT_MIN", "-15"))
    estimated_tilt_max: int = int(os.environ.get("SCOUT_ESTIMATED_TILT_MAX", "70"))
    pan_limit_margin: int = int(os.environ.get("SCOUT_PAN_LIMIT_MARGIN", "12"))
    edge_reacquire_enabled: bool = os.environ.get("SCOUT_EDGE_REACQUIRE_ENABLED", "1") != "0"
    edge_reacquire_base_z: int = int(os.environ.get("SCOUT_EDGE_REACQUIRE_BASE_Z", "18"))
    edge_reacquire_base_pulse_seconds: float = float(os.environ.get("SCOUT_EDGE_REACQUIRE_BASE_PULSE", "0.22"))
    edge_reacquire_reset_pan: int = int(os.environ.get("SCOUT_EDGE_REACQUIRE_RESET_PAN", "55"))
    edge_reacquire_cooldown: float = float(os.environ.get("SCOUT_EDGE_REACQUIRE_COOLDOWN", "0.55"))
    follow_enabled: bool = os.environ.get("SCOUT_FOLLOW_ENABLED", "0") != "0"
    follow_target_bbox_ratio: float = float(os.environ.get("SCOUT_FOLLOW_TARGET_BBOX_RATIO", "0.28"))
    follow_deadzone_ratio: float = float(os.environ.get("SCOUT_FOLLOW_DEADZONE_RATIO", "0.06"))
    follow_forward_speed: int = int(os.environ.get("SCOUT_FOLLOW_FORWARD_SPEED", "400"))
    follow_steer_gain: float = float(os.environ.get("SCOUT_FOLLOW_STEER_GAIN", "4.5"))


@dataclass
class CollisionConfig:
    enabled: bool = os.environ.get("SCOUT_COLLISION_ENABLED", "1") != "0"
    height_threshold: float = float(os.environ.get("SCOUT_COLLISION_HEIGHT_THRESHOLD", "0.35"))
    center_zone_fraction: float = float(os.environ.get("SCOUT_COLLISION_CENTER_ZONE", "0.70"))
    skip_labels: str = os.environ.get("SCOUT_COLLISION_SKIP_LABELS", "")


@dataclass
class GuardConfig:
    brain_alert_url: str = os.environ.get("SCOUT_GUARD_VAULT_URL", "http://10.10.1.1:7000")
    alert_cooldown_seconds: float = float(os.environ.get("SCOUT_GUARD_ALERT_COOLDOWN", "60.0"))
    alert_timeout_seconds: float = float(os.environ.get("SCOUT_GUARD_ALERT_TIMEOUT", "3.0"))
    alert_url: str = os.environ.get("SCOUT_GUARD_ALERT_URL", "http://luhkas-vault.local:7000/alerts")
    snapshot_on_alert: bool = bool(int(os.environ.get("SCOUT_GUARD_SNAPSHOT", "1")))


@dataclass
class SearchConfig:
    enabled: bool = os.environ.get("SCOUT_SEARCH_ENABLED", "1") != "0"
    sweep_duration_seconds: float = float(os.environ.get("SCOUT_SEARCH_SWEEP_DURATION", "3.2"))
    sweep_pan_amount: int = int(os.environ.get("SCOUT_SEARCH_SWEEP_PAN_AMOUNT", "60"))
    scan_pan_positions: str = os.environ.get("SCOUT_SEARCH_SCAN_PANS", "-90,-60,0,0,60,90")
    scan_tilt_positions: str = os.environ.get("SCOUT_SEARCH_SCAN_TILTS", "0,20,0,20,0,20")
    scan_pan_period_seconds: float = float(os.environ.get("SCOUT_SEARCH_SCAN_PAN_PERIOD", "11.0"))
    scan_tilt_period_seconds: float = float(os.environ.get("SCOUT_SEARCH_SCAN_TILT_PERIOD", "18.0"))
    scan_command_interval_seconds: float = float(os.environ.get("SCOUT_SEARCH_SCAN_COMMAND_INTERVAL", "0.20"))
    command_interval_seconds: float = float(os.environ.get("SCOUT_SEARCH_COMMAND_INTERVAL", "0.20"))


@dataclass
class TelemetryConfig:
    enabled:         bool  = bool(int(os.environ.get("SCOUT_EGO_MOTION_ENABLED", "0")))
    gyro_pan_scale:  float = float(os.environ.get("SCOUT_GYRO_PAN_SCALE",  "0.0"))
    gyro_tilt_scale: float = float(os.environ.get("SCOUT_GYRO_TILT_SCALE", "0.0"))
    poll_interval:   float = float(os.environ.get("SCOUT_TELEMETRY_POLL_INTERVAL", "0.2"))
