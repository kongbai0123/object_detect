from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .tracking import Track, compute_iou


class StableDoorState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CLOSED = "CLOSED"
    OPENING_CANDIDATE = "OPENING_CANDIDATE"
    OPEN = "OPEN"
    CLOSING_CANDIDATE = "CLOSING_CANDIDATE"


@dataclass(slots=True)
class EdgeStateMachineConfig:
    open_conf_threshold: float = 0.80
    close_conf_threshold: float = 0.55
    min_box_area_ratio: float = 0.010
    min_aspect_ratio: float = 0.20
    max_aspect_ratio: float = 2.50
    track_match_iou: float = 0.50
    max_center_shift_ratio: float = 0.15
    persist_window: int = 5
    persist_required: int = 3
    open_enter_frames: int = 3
    open_exit_frames: int = 5
    close_enter_frames: int = 3
    state_cooldown_ms: int = 2000
    state_timeout_ms: int = 3000
    roi_polygon: list[tuple[int, int]] | None = None


@dataclass(slots=True)
class CandidateObservation:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    timestamp_ms: int


@dataclass(slots=True)
class TrackStateMemory:
    state: StableDoorState = StableDoorState.UNKNOWN
    history: deque[CandidateObservation | None] = field(default_factory=lambda: deque(maxlen=5))
    stable_bbox: tuple[int, int, int, int] | None = None
    stable_confidence: float = 0.0
    stable_frames: int = 0
    cooldown_until_ms: int = 0
    last_seen_ms: int = 0
    last_reason: str = "init"


class EdgeDoorStateMachine:
    def __init__(self, config: EdgeStateMachineConfig | None = None) -> None:
        self.config = config if config is not None else EdgeStateMachineConfig()
        self._memories: dict[int, TrackStateMemory] = {}

    def update(self, track: Track, frame_shape: tuple[int, int, int], timestamp_ms: int) -> dict[str, Any]:
        memory = self._memories.setdefault(
            track.track_id,
            TrackStateMemory(history=deque(maxlen=self.config.persist_window)),
        )

        observation, reject_reason = self._build_observation(track, frame_shape, memory, timestamp_ms)
        memory.history.append(observation)
        memory.last_seen_ms = timestamp_ms

        if self._is_stale(memory, timestamp_ms):
            memory.state = StableDoorState.UNKNOWN
            memory.stable_frames = 0
            memory.last_reason = "state_timeout_unknown"

        cooldown_active = timestamp_ms < memory.cooldown_until_ms
        if not cooldown_active:
            self._advance_state(memory, observation, timestamp_ms)
        else:
            memory.last_reason = "cooldown_blocked"

        if observation is not None:
            memory.stable_bbox = observation.bbox
            memory.stable_confidence = observation.confidence

        return {
            "track_id": track.track_id,
            "state": memory.state.value,
            "confidence": round(self._aggregate_confidence(memory), 4),
            "stable_frames": int(memory.stable_frames),
            "bbox_xyxy": list(memory.stable_bbox) if memory.stable_bbox is not None else list(track.bbox),
            "timestamp_ms": int(timestamp_ms),
            "cooldown_active": cooldown_active,
            "source_class": observation.label if observation is not None else "none",
            "raw_class": track.cls_label,
            "raw_confidence": round(track.cls_conf, 4),
            "det_confidence": round(track.det_conf, 4),
            "reject_reason": reject_reason,
            "debug_reason": memory.last_reason,
        }

    def _build_observation(
        self,
        track: Track,
        frame_shape: tuple[int, int, int],
        memory: TrackStateMemory,
        timestamp_ms: int,
    ) -> tuple[CandidateObservation | None, str]:
        label = track.secondary_states.get("door_gap_state", track.cls_label).strip().lower()
        if label not in {"open", "close"}:
            return None, "unsupported_class"

        confidence = max(track.cls_conf, track.det_conf)
        if label == "open" and confidence < self.config.open_conf_threshold:
            return None, "below_open_threshold"
        if label == "close" and confidence < self.config.close_conf_threshold:
            return None, "below_close_threshold"

        frame_h, frame_w = frame_shape[:2]
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width <= 0 or height <= 0:
            return None, "invalid_bbox"

        area_ratio = (width * height) / max(frame_w * frame_h, 1)
        if area_ratio < self.config.min_box_area_ratio:
            return None, "small_box"

        aspect_ratio = width / max(height, 1)
        if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
            return None, "invalid_aspect_ratio"

        if self.config.roi_polygon and not _bbox_center_inside_polygon(bbox, self.config.roi_polygon):
            return None, "outside_roi"

        if memory.stable_bbox is not None:
            if compute_iou(memory.stable_bbox, bbox) < self.config.track_match_iou:
                shift_ratio = _center_shift_ratio(memory.stable_bbox, bbox, frame_w, frame_h)
                if shift_ratio > self.config.max_center_shift_ratio:
                    return None, "temporal_not_stable"

        return CandidateObservation(label=label, confidence=confidence, bbox=bbox, timestamp_ms=timestamp_ms), "accepted"

    def _advance_state(
        self,
        memory: TrackStateMemory,
        observation: CandidateObservation | None,
        timestamp_ms: int,
    ) -> None:
        open_confirmed = self._count_recent(memory, "open") >= self.config.persist_required
        close_confirmed = self._count_recent(memory, "close") >= self.config.persist_required
        sustained_open = self._tail_streak(memory, "open") >= self.config.open_enter_frames
        sustained_close = self._tail_streak(memory, "close") >= self.config.close_enter_frames
        open_absent = self._tail_absent(memory) >= self.config.open_exit_frames

        state = memory.state
        memory.last_reason = f"state={state.value}"

        if state == StableDoorState.UNKNOWN:
            if close_confirmed:
                memory.state = StableDoorState.CLOSED
                memory.cooldown_until_ms = timestamp_ms + self.config.state_cooldown_ms
                memory.stable_frames = self._tail_streak(memory, "close")
                memory.last_reason = "unknown_to_closed"
            return

        if state == StableDoorState.CLOSED:
            if open_confirmed:
                memory.state = StableDoorState.OPENING_CANDIDATE
                memory.stable_frames = self._tail_streak(memory, "open")
                memory.last_reason = "closed_to_opening_candidate"
            elif close_confirmed:
                memory.stable_frames = self._tail_streak(memory, "close")
            return

        if state == StableDoorState.OPENING_CANDIDATE:
            if sustained_open:
                memory.state = StableDoorState.OPEN
                memory.cooldown_until_ms = timestamp_ms + self.config.state_cooldown_ms
                memory.stable_frames = self._tail_streak(memory, "open")
                memory.last_reason = "opening_candidate_to_open"
            elif not open_confirmed:
                memory.state = StableDoorState.CLOSED
                memory.stable_frames = max(1, self._tail_streak(memory, "close"))
                memory.last_reason = "opening_candidate_reverted_to_closed"
            return

        if state == StableDoorState.OPEN:
            if sustained_close or (close_confirmed and open_absent):
                memory.state = StableDoorState.CLOSING_CANDIDATE
                memory.stable_frames = self._tail_streak(memory, "close")
                memory.last_reason = "open_to_closing_candidate"
            elif open_confirmed:
                memory.stable_frames = self._tail_streak(memory, "open")
            return

        if state == StableDoorState.CLOSING_CANDIDATE:
            if sustained_close:
                memory.state = StableDoorState.CLOSED
                memory.cooldown_until_ms = timestamp_ms + self.config.state_cooldown_ms
                memory.stable_frames = self._tail_streak(memory, "close")
                memory.last_reason = "closing_candidate_to_closed"
            elif open_confirmed:
                memory.state = StableDoorState.OPEN
                memory.stable_frames = self._tail_streak(memory, "open")
                memory.last_reason = "closing_candidate_reverted_to_open"

    def _aggregate_confidence(self, memory: TrackStateMemory) -> float:
        accepted = [item.confidence for item in memory.history if item is not None]
        if not accepted:
            return 0.0
        return sum(accepted[-self.config.persist_window :]) / min(len(accepted), self.config.persist_window)

    def _count_recent(self, memory: TrackStateMemory, label: str) -> int:
        return sum(1 for item in memory.history if item is not None and item.label == label)

    def _tail_streak(self, memory: TrackStateMemory, label: str) -> int:
        streak = 0
        for item in reversed(memory.history):
            if item is not None and item.label == label:
                streak += 1
            else:
                break
        return streak

    def _tail_absent(self, memory: TrackStateMemory) -> int:
        streak = 0
        for item in reversed(memory.history):
            if item is None:
                streak += 1
            else:
                break
        return streak

    def _is_stale(self, memory: TrackStateMemory, timestamp_ms: int) -> bool:
        return (timestamp_ms - memory.last_seen_ms) >= self.config.state_timeout_ms


def _bbox_center_inside_polygon(bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]]) -> bool:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return _point_in_polygon(cx, cy, polygon)


def _point_in_polygon(x: float, y: float, polygon: list[tuple[int, int]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return True
    for idx in range(n):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-6) + x1)
        if intersects:
            inside = not inside
    return inside


def _center_shift_ratio(
    prev_bbox: tuple[int, int, int, int],
    curr_bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    px = (prev_bbox[0] + prev_bbox[2]) / 2.0
    py = (prev_bbox[1] + prev_bbox[3]) / 2.0
    cx = (curr_bbox[0] + curr_bbox[2]) / 2.0
    cy = (curr_bbox[1] + curr_bbox[3]) / 2.0
    dx = (cx - px) / max(frame_w, 1)
    dy = (cy - py) / max(frame_h, 1)
    return (dx * dx + dy * dy) ** 0.5
