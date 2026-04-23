from __future__ import annotations

from dataclasses import dataclass, field


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    det_conf: float
    age: int = 1
    hits: int = 1
    missed: int = 0
    cls_label: str = "unknown"
    cls_conf: float = 0.0
    secondary_states: dict[str, str] = field(default_factory=dict)
    last_secondary_frame: int = -10**9
    last_seen_frame: int = 0

    def update_detection(self, bbox: tuple[int, int, int, int], det_conf: float, frame_index: int) -> None:
        self.bbox = bbox
        self.det_conf = det_conf
        self.hits += 1
        self.age += 1
        self.missed = 0
        self.last_seen_frame = frame_index


class TrackManager:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 12) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_track_id = 1
        self._tracks: list[Track] = []

    def update(self, detections: list[dict], frame_index: int) -> list[Track]:
        assigned_track_ids: set[int] = set()
        unmatched_detections: list[dict] = []

        for detection in detections:
            bbox = detection["bbox"]
            best_track = None
            best_iou = 0.0
            for track in self._tracks:
                if track.track_id in assigned_track_ids:
                    continue
                iou = compute_iou(track.bbox, bbox)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_track = track
                    best_iou = iou

            if best_track is None:
                unmatched_detections.append(detection)
                continue

            best_track.update_detection(bbox, detection["det_conf"], frame_index)
            assigned_track_ids.add(best_track.track_id)

        survivors: list[Track] = []
        for track in self._tracks:
            if track.track_id not in assigned_track_ids:
                track.age += 1
                track.missed += 1
            if track.missed <= self.max_missed:
                survivors.append(track)
        self._tracks = survivors

        for detection in unmatched_detections:
            self._tracks.append(
                Track(
                    track_id=self._next_track_id,
                    bbox=detection["bbox"],
                    det_conf=detection["det_conf"],
                    last_seen_frame=frame_index,
                )
            )
            self._next_track_id += 1

        return list(self._tracks)
