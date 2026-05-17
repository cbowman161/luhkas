from dataclasses import dataclass, field
import time


@dataclass
class Detection:
    id: int | None
    label: str
    class_id: int
    bbox: list[int]
    confidence: float
    color: str = "unknown"
    color_rgb: tuple[int, int, int] = (0, 255, 0)
    memory_id: int | None = None
    tracker_id: int | None = None
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    seen_count: int = 1
    vx: float = 0.0
    vy: float = 0.0
    predicted: bool = False
    aim_x: float | None = None
    aim_y: float | None = None
    aim_source: str = "bbox"
    identity: str | None = None
    identity_confidence: float | None = None
    recognition_distance: float | None = None
    recognition_method: str | None = None
    reference_pose: str | None = None
    missing_reference_poses: list[str] = field(default_factory=list)
    face_group_id: str | None = None
    vault_face_group_id: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def aim_point(self) -> tuple[float, float]:
        if self.aim_x is not None and self.aim_y is not None:
            return self.aim_x, self.aim_y
        return self.center

    def copy_identity_from(self, face: "Detection") -> None:
        self.identity = face.identity
        self.identity_confidence = face.identity_confidence
        self.recognition_distance = face.recognition_distance
        self.recognition_method = face.recognition_method
        self.reference_pose = face.reference_pose
        self.missing_reference_poses = list(face.missing_reference_poses)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "class_id": self.class_id,
            "bbox": [int(v) for v in self.bbox],
            "confidence": float(self.confidence),
            "color": self.color,
            "color_rgb": [int(v) for v in self.color_rgb],
            "memory_id": self.memory_id,
            "tracker_id": self.tracker_id,
            "last_seen": self.last_seen,
            "seen_count": self.seen_count,
            "vx": float(self.vx),
            "vy": float(self.vy),
            "predicted": bool(self.predicted),
            "aim": {
                "x": self.aim_x,
                "y": self.aim_y,
                "source": self.aim_source,
            },
            "identity": self.identity,
            "identity_confidence": self.identity_confidence,
            "recognition_distance": self.recognition_distance,
            "recognition_method": self.recognition_method,
            "reference_pose": self.reference_pose,
            "missing_reference_poses": list(self.missing_reference_poses),
            "face_group_id": self.face_group_id,
            "vault_face_group_id": self.vault_face_group_id,
        }
