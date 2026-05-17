from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .config import TrackingConfig
from .types import Detection
from .vision import is_face_label, center_inside, bbox_iou

try:
    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
except Exception:
    BYTETracker = None


@dataclass
class ObjectMemory:
    memory_id: int
    object_id: int
    label: str
    class_id: int
    color: str
    color_rgb: tuple[int, int, int]
    bbox: list[int]
    first_seen: float
    last_seen: float
    seen_count: int = 1
    pan_at_last_seen: float = 0.0
    tilt_at_last_seen: float = 0.0
    identity: str | None = None
    identity_confidence: float | None = None
    recognition_distance: float | None = None
    recognition_method: str | None = None
    person_memory: dict | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    def to_json(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "object_id": self.object_id,
            "label": self.label,
            "class_id": self.class_id,
            "color": self.color,
            "color_rgb": [int(v) for v in self.color_rgb],
            "bbox": [int(v) for v in self.bbox],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "seen_count": self.seen_count,
            "pan_at_last_seen": self.pan_at_last_seen,
            "tilt_at_last_seen": self.tilt_at_last_seen,
            "identity": self.identity,
            "identity_confidence": self.identity_confidence,
            "recognition_distance": self.recognition_distance,
            "recognition_method": self.recognition_method,
            "person_memory": self.person_memory,
        }


class SimpleTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._objects: list[Detection] = []
        self._memory: list[ObjectMemory] = []
        self.created_at = time.time()
        self._next_id = 1
        self._next_memory_id = 1
        self.selected_target_id: int | None = None
        self.selected_memory_id: int | None = None
        self._last_selected_target: Detection | None = None
        self._last_selected_pan: float = 0.0
        self._last_selected_tilt: float = 0.0
        self._byte_trackers: dict[int, BYTETracker] = {}
        self._bytetracker_available = BYTETracker is not None
        self._ego_motion = {
            "estimated_pan": 0.0,
            "estimated_tilt": 0.0,
            "recent_base_turn": 0.0,
        }

    @property
    def objects(self) -> list[Detection]:
        return list(self._objects)

    @property
    def memory(self) -> list[ObjectMemory]:
        return list(self._memory)

    def stats(self) -> dict:
        return {
            "created_at": self.created_at,
            "uptime_seconds": time.time() - self.created_at,
            "next_id": self._next_id,
            "next_memory_id": self._next_memory_id,
            "active_objects": len(self._objects),
            "memory_objects": len(self._memory),
            "max_objects": self.config.max_objects,
            "bytetracker_enabled": bool(self.config.bytetracker_enabled),
            "bytetracker_available": bool(self._bytetracker_available),
            "bytetracker_classes": len(self._byte_trackers),
            "selected_memory_id": self.selected_memory_id,
        }

    def hydrate_person_memory(self, person_memory_store) -> None:
        for memory in self._memory:
            if memory.label != "person" or not memory.identity:
                continue
            if memory.person_memory is not None and memory.person_memory.get("identity") == memory.identity:
                continue
            memory.person_memory = person_memory_store.summary_for(memory.identity)

    def set_ego_motion(self, ego_motion: dict | None) -> None:
        if not ego_motion:
            return
        self._ego_motion["estimated_pan"] = float(ego_motion.get("estimated_pan", self._ego_motion["estimated_pan"]))
        self._ego_motion["estimated_tilt"] = float(ego_motion.get("estimated_tilt", self._ego_motion["estimated_tilt"]))
        self._ego_motion["recent_base_turn"] = float(ego_motion.get("recent_base_turn", 0.0))

    def update(self, detections: list[Detection]) -> list[Detection]:
        now = time.time()
        detections = self._apply_bytetracker(detections)
        self._prune_memory(now)
        self._objects = [
            obj for obj in self._objects
            if now - obj.last_seen <= self.config.object_ttl_seconds
        ]
        active_ids = {obj.id for obj in self._objects if obj.id is not None}

        unmatched = self._objects[:]
        tracked: list[Detection] = []

        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for det in ordered[: self.config.max_objects]:
            match = self._best_match(det, unmatched)
            if match is None:
                det.last_seen = now
                det.first_seen = now
                self._attach_memory(det, now, active_ids)
                if det.id is None:
                    det.id = self._take_id()
                tracked.append(det)
                self._objects.append(det)
                active_ids.add(det.id)
                continue

            old_center = match.center
            dt = max(now - match.last_seen, 1e-3)
            smoothed_bbox = _smooth_bbox(match.bbox, det.bbox, self.config.bbox_smoothing)
            match.label = det.label
            match.class_id = det.class_id
            match.bbox = smoothed_bbox
            match.confidence = det.confidence
            match.color = det.color
            match.color_rgb = det.color_rgb
            if det.identity is not None or det.recognition_method is not None:
                match.identity = det.identity
                match.identity_confidence = det.identity_confidence
                match.recognition_distance = det.recognition_distance
                match.recognition_method = det.recognition_method
                match.reference_pose = det.reference_pose
                match.missing_reference_poses = list(det.missing_reference_poses)
            if match.memory_id is None:
                self._attach_memory(match, now, active_ids)
            match.last_seen = now
            match.seen_count += 1
            new_center = match.center
            raw_vx = (new_center[0] - old_center[0]) / dt
            raw_vy = (new_center[1] - old_center[1]) / dt
            v_alpha = max(0.0, min(1.0, self.config.velocity_smoothing))
            match.vx = match.vx * (1.0 - v_alpha) + raw_vx * v_alpha
            match.vy = match.vy * (1.0 - v_alpha) + raw_vy * v_alpha
            self._update_memory(match, now)
            memory = self._memory_for_id(match.memory_id)
            if memory is not None:
                match.id = memory.object_id
                _apply_memory_identity(match, memory)
            unmatched.remove(match)
            tracked.append(match)

        self._objects = _dedupe_detections(self._objects)
        self._objects.sort(key=lambda d: d.last_seen, reverse=True)
        return _dedupe_detections(tracked)

    def _apply_bytetracker(self, detections: list[Detection]) -> list[Detection]:
        if not self.config.bytetracker_enabled or not self._bytetracker_available:
            return detections
        if not detections:
            for tracker in self._byte_trackers.values():
                tracker.update(np.empty((0, 5), dtype=float))
            return []

        tracked: list[Detection] = []
        for class_id in sorted({det.class_id for det in detections}):
            class_detections = [det for det in detections if det.class_id == class_id]
            tracker = self._byte_trackers.get(class_id)
            if tracker is None:
                tracker = BYTETracker(self._bytetracker_args())
                self._byte_trackers[class_id] = tracker

            tracker_input = np.array(
                [
                    [
                        det.bbox[0],
                        det.bbox[1],
                        det.bbox[0] + det.bbox[2],
                        det.bbox[1] + det.bbox[3],
                        det.confidence,
                    ]
                    for det in class_detections
                ],
                dtype=float,
            )
            tracks = tracker.update(tracker_input)
            used_detection_indexes: set[int] = set()

            for track in tracks:
                x1, y1, x2, y2 = [int(round(value)) for value in track.tlbr]
                bbox = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]
                best_index = _best_detection_index_for_bbox(bbox, class_detections, used_detection_indexes)
                if best_index is None:
                    continue
                used_detection_indexes.add(best_index)
                source = class_detections[best_index]
                tracked.append(replace(
                    source,
                    id=track.track_id,
                    tracker_id=track.track_id,
                    bbox=bbox,
                    confidence=float(track.score),
                ))

            for index, source in enumerate(class_detections):
                if index not in used_detection_indexes:
                    tracked.append(source)

        return tracked

    def _bytetracker_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            track_thresh=self.config.bytetracker_track_thresh,
            track_buffer=self.config.bytetracker_track_buffer,
            match_thresh=self.config.bytetracker_match_thresh,
            aspect_ratio_thresh=2.0,
            min_box_area=self.config.bytetracker_min_box_area,
            mot20=False,
        )

    def select_person_target(self, frame_w: int, frame_h: int) -> Detection | None:
        now = time.time()
        target_label = self.config.target_label
        target_score_threshold = (
            self.config.person_score_threshold
            if target_label == "person"
            else self.config.score_threshold
        )
        targets = [
            obj for obj in self._objects
            if obj.label == target_label and obj.confidence >= target_score_threshold
            and now - obj.last_seen <= self.config.target_lost_grace_seconds
        ]
        if not targets:
            predicted = self._predicted_selected_target(now, frame_w, frame_h)
            if predicted is not None:
                return predicted
            self.selected_target_id = None
            self.selected_memory_id = None
            return None

        if target_label == "person" and self.config.target_identity:
            preferred_targets = [
                target for target in targets
                if _identity_matches(self._best_face_for_person(target), self.config.target_identity)
            ]
            if preferred_targets:
                targets = preferred_targets

        current = next(
            (
                p for p in targets
                if p.id == self.selected_target_id
                or (self.selected_memory_id is not None and p.memory_id == self.selected_memory_id)
            ),
            None,
        )

        center = (frame_w / 2.0, frame_h / 2.0)
        best = max(targets, key=lambda d: _target_score(d, center, frame_w, frame_h))
        if current is None:
            self.selected_target_id = best.id
            self.selected_memory_id = best.memory_id
            return self._remember_selected(self._with_target_aim(best))

        current_score = _target_score(current, center, frame_w, frame_h)
        best_score = _target_score(best, center, frame_w, frame_h)
        if best.id != current.id and best_score > current_score + self.config.target_switch_margin:
            self.selected_target_id = best.id
            self.selected_memory_id = best.memory_id
            return self._remember_selected(self._with_target_aim(best))
        self.selected_target_id = current.id
        self.selected_memory_id = current.memory_id
        return self._remember_selected(self._with_target_aim(current))

    def _remember_selected(self, target: Detection) -> Detection:
        self._last_selected_target = replace(target)
        self._last_selected_pan = float(self._ego_motion.get("estimated_pan", 0.0))
        self._last_selected_tilt = float(self._ego_motion.get("estimated_tilt", 0.0))
        return target

    def _with_target_aim(self, target: Detection) -> Detection:
        if target.label == "person":
            return self._with_face_aim(target)
        x, y, w, h = target.bbox
        target.aim_x = x + w / 2.0
        target.aim_y = y + h / 2.0
        target.aim_source = "center"
        return target

    def _with_face_aim(self, person: Detection) -> Detection:
        if person.predicted:
            # Face detections are in current-frame screen coords; don't apply them
            # to a predicted (ego-compensated) bbox — the positions won't correspond.
            x, y, w, h = person.bbox
            person.aim_x = x + w / 2.0
            person.aim_y = y + h * self.config.target_torso_aim_ratio
            person.aim_source = "upper_body"
            return person

        face = self._best_face_for_person(person)
        if face is None:
            x, y, w, h = person.bbox
            person.aim_x = x + w / 2.0
            person.aim_y = y + h * self.config.target_torso_aim_ratio
            person.aim_source = "upper_body"
            return person

        person.aim_x, person.aim_y = face.center
        person.aim_source = f"face:{face.id}"
        if face.identity:
            person.copy_identity_from(face)
        return person

    def _best_face_for_person(self, person: Detection) -> Detection | None:
        faces = [obj for obj in self._objects if is_face_label(obj.label)]
        inside = [face for face in faces if center_inside(face.center, person.bbox)]
        if not inside:
            return None

        px, py, pw, ph = person.bbox
        head_band_bottom = py + ph * 0.55
        preferred = [face for face in inside if face.center[1] <= head_band_bottom]
        candidates = preferred or inside
        return max(candidates, key=lambda face: face.confidence)

    def _predicted_selected_target(self, now: float, frame_w: int, frame_h: int) -> Detection | None:
        if self.selected_target_id is None and self.selected_memory_id is None:
            return None

        target = next((obj for obj in self._objects if obj.id == self.selected_target_id), None)
        if target is None and self.selected_memory_id is not None:
            target = next((obj for obj in self._objects if obj.memory_id == self.selected_memory_id), None)

        # Use the last selected target as fallback, but apply ego-motion compensation
        # so the predicted screen position updates when the camera moves.
        if target is None and self._last_selected_target is not None:
            if (
                self._last_selected_target.label == self.config.target_label
                and (
                    self._last_selected_target.id == self.selected_target_id
                    or self._last_selected_target.memory_id == self.selected_memory_id
                )
            ):
                tmp_mem = ObjectMemory(
                    memory_id=self._last_selected_target.memory_id or 0,
                    object_id=self._last_selected_target.id or 0,
                    label=self._last_selected_target.label,
                    class_id=self._last_selected_target.class_id,
                    color=self._last_selected_target.color,
                    color_rgb=self._last_selected_target.color_rgb,
                    bbox=self._last_selected_target.bbox,
                    first_seen=self._last_selected_target.first_seen,
                    last_seen=self._last_selected_target.last_seen,
                    pan_at_last_seen=self._last_selected_pan,
                    tilt_at_last_seen=self._last_selected_tilt,
                )
                target = replace(
                    self._last_selected_target,
                    bbox=_ego_compensated_bbox(tmp_mem, self._ego_motion, self.config),
                )

        if target is None and self.selected_memory_id is not None:
            memory = self._memory_for_id(self.selected_memory_id)
            if memory is not None and memory.label == self.config.target_label:
                target = Detection(
                    id=memory.object_id,
                    label=memory.label,
                    class_id=memory.class_id,
                    bbox=_ego_compensated_bbox(memory, self._ego_motion, self.config),
                    confidence=0.0,
                    color=memory.color,
                    color_rgb=memory.color_rgb,
                    memory_id=memory.memory_id,
                    last_seen=memory.last_seen,
                    first_seen=memory.first_seen,
                    seen_count=memory.seen_count,
                    identity=memory.identity,
                    identity_confidence=memory.identity_confidence,
                    recognition_distance=memory.recognition_distance,
                    recognition_method=memory.recognition_method,
                    predicted=True,
                )

        if target is None or target.label != self.config.target_label:
            return None

        lost_for = now - target.last_seen
        if lost_for > self.config.target_reacquire_seconds:
            return None

        prediction_dt = min(lost_for, self.config.max_prediction_seconds)
        x, y, w, h = target.bbox
        predicted_x = int(round(x + target.vx * prediction_dt))
        predicted_y = int(round(y + target.vy * prediction_dt))

        # Allow the predicted center to move a bit outside the frame so the
        # pan/tilt controller still turns toward the side where the target left.
        margin_x = max(w, int(frame_w * 0.2))
        margin_y = max(h, int(frame_h * 0.2))
        predicted_x = max(-margin_x, min(frame_w + margin_x - w, predicted_x))
        predicted_y = max(-margin_y, min(frame_h + margin_y - h, predicted_y))

        predicted = replace(
            target,
            bbox=[predicted_x, predicted_y, w, h],
            predicted=True,
        )
        return self._with_target_aim(predicted)

    def _take_id(self) -> int:
        obj_id = self._next_id
        self._next_id += 1
        if self._next_id > 999:
            self._next_id = 1
        return obj_id

    def _take_memory_id(self) -> int:
        memory_id = self._next_memory_id
        self._next_memory_id += 1
        return memory_id

    def _attach_memory(self, det: Detection, now: float, active_ids: set[int] | None = None) -> None:
        memory = self._best_memory_match(det)
        if memory is None:
            object_id = self._take_id()
            memory = ObjectMemory(
                memory_id=self._take_memory_id(),
                object_id=object_id,
                label=det.label,
                class_id=det.class_id,
                color=det.color,
                color_rgb=det.color_rgb,
                bbox=det.bbox[:],
                first_seen=now,
                last_seen=now,
                pan_at_last_seen=self._ego_motion["estimated_pan"],
                tilt_at_last_seen=self._ego_motion["estimated_tilt"],
            )
            self._memory.append(memory)

        det.memory_id = memory.memory_id
        det.id = memory.object_id
        _apply_memory_identity(det, memory)
        self._merge_memory(memory, det, now)

    def _update_memory(self, det: Detection, now: float) -> None:
        if det.memory_id is None:
            self._attach_memory(det, now)
            return
        memory = next((m for m in self._memory if m.memory_id == det.memory_id), None)
        if memory is None:
            det.memory_id = None
            self._attach_memory(det, now)
            return
        self._merge_memory(memory, det, now)
        det.id = memory.object_id
        _apply_memory_identity(det, memory)

    def _merge_memory(self, memory: ObjectMemory, det: Detection, now: float) -> None:
        memory.label = det.label
        memory.class_id = det.class_id
        memory.color = det.color if det.color != "unknown" else memory.color
        memory.color_rgb = _smooth_rgb(memory.color_rgb, det.color_rgb, 0.25)
        memory.bbox = _smooth_bbox(memory.bbox, det.bbox, 0.22)
        memory.last_seen = now
        memory.pan_at_last_seen = self._ego_motion["estimated_pan"]
        memory.tilt_at_last_seen = self._ego_motion["estimated_tilt"]
        memory.seen_count += 1
        if det.identity and det.identity != "unknown":
            current_conf = memory.identity_confidence if memory.identity == det.identity else None
            incoming_conf = det.identity_confidence or 0.0
            if memory.identity is None or current_conf is None or incoming_conf >= current_conf * 0.82:
                memory.identity = det.identity
                memory.identity_confidence = det.identity_confidence
                memory.recognition_distance = det.recognition_distance
                memory.recognition_method = det.recognition_method

    def _best_memory_match(self, det: Detection) -> ObjectMemory | None:
        same_class = [m for m in self._memory if m.class_id == det.class_id]
        if not same_class:
            return None

        best = max(same_class, key=lambda memory: _memory_score(memory, det, self.config, self._ego_motion))
        score = _memory_score(best, det, self.config, self._ego_motion)
        if score < self.config.memory_match_threshold:
            return None
        return best

    def _memory_for_id(self, memory_id: int | None) -> ObjectMemory | None:
        if memory_id is None:
            return None
        return next((memory for memory in self._memory if memory.memory_id == memory_id), None)

    def _prune_memory(self, now: float) -> None:
        self._memory = [
            memory for memory in self._memory
            if now - memory.last_seen <= self.config.memory_ttl_seconds
        ]
        self._memory.sort(key=lambda memory: (memory.last_seen, memory.seen_count), reverse=True)
        self._memory = self._memory[: self.config.max_memory_objects]

    def _best_match(self, det: Detection, candidates: list[Detection]) -> Detection | None:
        same_class = [c for c in candidates if c.class_id == det.class_id]
        if not same_class:
            return None

        best = min(same_class, key=lambda c: _match_cost(c, det, self.config.max_match_distance_px))
        distance = _distance(_predicted_center(best), det.center)
        iou = bbox_iou(best.bbox, det.bbox)
        if distance > self.config.max_match_distance_px and iou < self.config.min_match_iou:
            return None
        return best


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _match_cost(existing: Detection, incoming: Detection, max_distance: float) -> float:
    distance_cost = min(_distance(_predicted_center(existing), incoming.center) / max(max_distance, 1), 2.0)
    iou_cost = 1.0 - bbox_iou(existing.bbox, incoming.bbox)
    size_cost = _size_delta(existing.bbox, incoming.bbox)
    age_bonus = min(existing.seen_count / 12.0, 1.0) * 0.08
    return 0.58 * distance_cost + 0.32 * iou_cost + 0.10 * size_cost - age_bonus


def _predicted_center(existing: Detection) -> tuple[float, float]:
    now = time.time()
    dt = min(max(now - existing.last_seen, 0.0), 0.6)
    cx, cy = existing.center
    return cx + existing.vx * dt, cy + existing.vy * dt


def _best_detection_index_for_bbox(
    bbox: list[int],
    detections: list[Detection],
    used_indexes: set[int],
) -> int | None:
    best_index = None
    best_score = -1.0
    for index, det in enumerate(detections):
        if index in used_indexes:
            continue
        score = bbox_iou(bbox, det.bbox)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _dedupe_detections(detections: list[Detection]) -> list[Detection]:
    by_id: dict[int, Detection] = {}
    without_id: list[Detection] = []
    for det in detections:
        if det.id is None:
            without_id.append(det)
            continue
        current = by_id.get(det.id)
        if current is None or (det.last_seen, det.confidence) > (current.last_seen, current.confidence):
            by_id[det.id] = det
    return list(by_id.values()) + without_id


def _size_delta(a: list[int], b: list[int]) -> float:
    _, _, aw, ah = a
    _, _, bw, bh = b
    area_a = max(aw * ah, 1)
    area_b = max(bw * bh, 1)
    return abs(area_a - area_b) / max(area_a, area_b)


def _smooth_bbox(old: list[int], new: list[int], alpha: float) -> list[int]:
    alpha = max(0.0, min(1.0, alpha))
    return [int(round(old_v * (1.0 - alpha) + new_v * alpha)) for old_v, new_v in zip(old, new)]


def _target_score(det: Detection, center: tuple[float, float], frame_w: int, frame_h: int) -> float:
    distance = _distance(det.center, center)
    max_distance = math.hypot(frame_w / 2.0, frame_h / 2.0)
    center_score = 1.0 - min(distance / max(max_distance, 1), 1.0)
    _, _, w, h = det.bbox
    area_score = min((w * h) / max(frame_w * frame_h, 1), 1.0)
    age_bonus = min(det.seen_count / 10.0, 1.0) * 0.08
    return 0.55 * det.confidence + 0.30 * center_score + 0.15 * area_score + age_bonus


def _identity_matches(face: Detection | None, target_identity: str) -> bool:
    if face is None or face.identity is None:
        return False
    return face.identity.casefold() == target_identity.casefold()


def _apply_memory_identity(det: Detection, memory: ObjectMemory) -> None:
    if not memory.identity:
        return
    det.identity = memory.identity
    det.identity_confidence = memory.identity_confidence
    det.recognition_distance = memory.recognition_distance
    det.recognition_method = memory.recognition_method


def _memory_score(memory: ObjectMemory, det: Detection, config: TrackingConfig, ego_motion: dict) -> float:
    compensated_bbox = _ego_compensated_bbox(memory, ego_motion, config)
    compensated_center = _bbox_center(compensated_bbox)
    distance = _distance(compensated_center, det.center)
    distance_score = 1.0 - min(distance / max(config.max_match_distance_px * 1.6, 1), 1.0)
    iou_score = bbox_iou(compensated_bbox, det.bbox)
    size_score = 1.0 - min(_size_delta(compensated_bbox, det.bbox), 1.0)
    color_score = _color_similarity(memory, det)
    stable_bonus = min(memory.seen_count / 20.0, 1.0) * 0.08
    color_weight = max(0.0, min(1.0, config.color_match_weight))
    ego_delta = abs(float(ego_motion.get("estimated_pan", 0.0)) - memory.pan_at_last_seen)
    ego_delta += 0.6 * abs(float(ego_motion.get("estimated_tilt", 0.0)) - memory.tilt_at_last_seen)
    ego_delta += 0.25 * abs(float(ego_motion.get("recent_base_turn", 0.0)))
    motion_factor = min(ego_delta / 60.0, 1.0)
    spatial_weight = max(0.25, 1.0 - color_weight - motion_factor * config.ego_motion_spatial_penalty)
    appearance_weight = min(0.75, color_weight + motion_factor * config.ego_motion_spatial_penalty)
    spatial_score = 0.45 * distance_score + 0.35 * iou_score + 0.20 * size_score
    return spatial_weight * spatial_score + appearance_weight * color_score + stable_bonus


def _color_similarity(memory: ObjectMemory, det: Detection) -> float:
    name_score = 1.0 if memory.color == det.color and det.color != "unknown" else 0.0
    rgb_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(memory.color_rgb, det.color_rgb)))
    rgb_score = 1.0 - min(rgb_distance / 441.7, 1.0)
    return 0.55 * name_score + 0.45 * rgb_score


def _smooth_rgb(old: tuple[int, int, int], new: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(int(round(a * (1.0 - alpha) + b * alpha)) for a, b in zip(old, new))


def _ego_compensated_bbox(memory: ObjectMemory, ego_motion: dict, config: TrackingConfig) -> list[int]:
    current_pan = float(ego_motion.get("estimated_pan", 0.0))
    current_tilt = float(ego_motion.get("estimated_tilt", 0.0))
    pan_delta = current_pan - memory.pan_at_last_seen
    tilt_delta = current_tilt - memory.tilt_at_last_seen

    x, y, w, h = memory.bbox
    if not config.wide_angle_compensation:
        dx = int(round(-pan_delta * config.pan_pixels_per_degree))
        dy = int(round(tilt_delta * config.tilt_pixels_per_degree))
        return [x + dx, y + dy, w, h]

    cx, cy = _bbox_center(memory.bbox)
    new_cx = _shift_wide_axis(
        coord=cx,
        size=config.frame_width,
        delta_degrees=-pan_delta,
        pixels_per_degree=config.pan_pixels_per_degree,
        strength=config.fisheye_strength,
    )
    new_cy = _shift_wide_axis(
        coord=cy,
        size=config.frame_height,
        delta_degrees=tilt_delta,
        pixels_per_degree=config.tilt_pixels_per_degree,
        strength=config.fisheye_strength,
    )
    return [int(round(new_cx - w / 2.0)), int(round(new_cy - h / 2.0)), w, h]


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _shift_wide_axis(
    coord: float,
    size: int,
    delta_degrees: float,
    pixels_per_degree: float,
    strength: float,
) -> float:
    if size <= 1:
        return coord

    strength = max(0.0, min(0.95, strength))
    half = size / 2.0
    normalized = max(-0.98, min(0.98, (coord - half) / half))

    angularish = math.atanh(normalized * strength) / strength if strength > 0 else normalized
    delta_norm = (delta_degrees * pixels_per_degree) / half
    shifted = angularish + delta_norm
    projected = math.tanh(shifted * strength) / strength if strength > 0 else shifted
    projected = max(-1.2, min(1.2, projected))
    return half + projected * half
