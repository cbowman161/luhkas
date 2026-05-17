from __future__ import annotations

from dataclasses import dataclass, field

from .types import Detection


@dataclass
class CollisionGuard:
    enabled: bool = True
    height_threshold: float = 0.35
    center_zone_fraction: float = 0.70
    skip_labels: frozenset[str] = field(default_factory=frozenset)

    def check(
        self,
        detections: list[Detection],
        target_id: int | None,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Return True if a close obstacle blocks forward motion."""
        if not self.enabled or not detections:
            return False

        margin = (1.0 - self.center_zone_fraction) / 2.0
        zone_left = frame_w * margin
        zone_right = frame_w * (1.0 - margin)
        min_height_px = frame_h * self.height_threshold

        for det in detections:
            if det.predicted:
                continue
            if det.id is not None and det.id == target_id:
                continue
            if det.label in self.skip_labels:
                continue
            x, y, w, h = det.bbox
            cx = x + w / 2.0
            if cx < zone_left or cx > zone_right:
                continue
            if h >= min_height_px:
                return True

        return False
