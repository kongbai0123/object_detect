from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
from ultralytics import YOLO

from .behavior import BehaviorAgent
from .state_machine import EdgeDoorStateMachine, EdgeStateMachineConfig
from .stabilization import SimpleStabilizer
from .tracking import Track, TrackManager


@dataclass(slots=True)
class BranchsRuntimeConfig:
    det_conf: float = 0.25
    det_iou: float = 0.45
    det_imgsz: int = 640
    cls_imgsz: int = 224
    track_iou_threshold: float = 0.3
    max_missed_frames: int = 12
    secondary_interval: int = 3
    max_secondary_targets_per_frame: int = 3
    min_transition_hits: int = 2
    half: bool = True
    enable_stabilization: bool = True
    stabilization_alpha: float = 0.8
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


class BranchsPipeline:
    def __init__(self, det_model: YOLO, cls_model: YOLO | None, config: BranchsRuntimeConfig) -> None:
        self.det_model = det_model
        self.cls_model = cls_model
        self.config = config
        self.track_manager = TrackManager(config.track_iou_threshold, config.max_missed_frames)
        self.behavior_agent = BehaviorAgent(config.min_transition_hits)
        self.edge_state_machine = EdgeDoorStateMachine(
            EdgeStateMachineConfig(
                open_conf_threshold=config.open_conf_threshold,
                close_conf_threshold=config.close_conf_threshold,
                min_box_area_ratio=config.min_box_area_ratio,
                min_aspect_ratio=config.min_aspect_ratio,
                max_aspect_ratio=config.max_aspect_ratio,
                track_match_iou=config.track_match_iou,
                max_center_shift_ratio=config.max_center_shift_ratio,
                persist_window=config.persist_window,
                persist_required=config.persist_required,
                open_enter_frames=config.open_enter_frames,
                open_exit_frames=config.open_exit_frames,
                close_enter_frames=config.close_enter_frames,
                state_cooldown_ms=config.state_cooldown_ms,
                state_timeout_ms=config.state_timeout_ms,
                roi_polygon=config.roi_polygon,
            )
        )
        self.stabilizer = (
            SimpleStabilizer(smoothing_alpha=config.stabilization_alpha) if config.enable_stabilization else None
        )
        self.frame_index = 0

    def process_frame(self, frame, timestamp_ms: int | None = None) -> dict[str, Any]:
        self.frame_index += 1
        perf: dict[str, float] = {}
        if timestamp_ms is None:
            timestamp_ms = self.frame_index * 33

        t0 = perf_counter()
        stable_frame = self.stabilizer.update(frame) if self.stabilizer is not None else frame
        perf["stabilization_ms"] = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        detections = self._run_detector(stable_frame)
        perf["detection_ms"] = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        tracks = self.track_manager.update(detections, self.frame_index)
        perf["tracking_ms"] = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        secondary_count = self._run_secondary(stable_frame, tracks)
        perf["secondary_ms"] = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        behavior_events = [self.behavior_agent.update(track) for track in tracks]
        edge_events = [
            self.edge_state_machine.update(track, stable_frame.shape, timestamp_ms)
            for track in tracks
        ]
        perf["behavior_ms"] = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        annotated = self._draw_overlay(stable_frame, tracks, behavior_events, edge_events, perf)
        perf["drawing_ms"] = (perf_counter() - t0) * 1000.0
        perf["secondary_targets"] = float(secondary_count)

        return {
            "frame": annotated,
            "tracks": tracks,
            "events": behavior_events,
            "edge_outputs": edge_events,
            "perf": perf,
            "mode": self.stabilizer.mode if self.stabilizer is not None else "disabled",
        }

    def _run_detector(self, frame) -> list[dict[str, Any]]:
        result = self.det_model.predict(
            frame,
            conf=self.config.det_conf,
            iou=self.config.det_iou,
            imgsz=self.config.det_imgsz,
            half=self.config.half,
            verbose=False,
        )[0]

        height, width = frame.shape[:2]
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "det_conf": float(box.conf[0].item()) if box.conf is not None else 0.0,
                }
            )
        return detections

    def _run_secondary(self, frame, tracks: list[Track]) -> int:
        if self.cls_model is None:
            return 0

        processed = 0
        prioritized = sorted(tracks, key=lambda track: (track.missed, -track.det_conf, track.track_id))
        for track in prioritized:
            if processed >= self.config.max_secondary_targets_per_frame:
                break

            should_run = (
                track.last_secondary_frame < 0
                or self.frame_index - track.last_secondary_frame >= self.config.secondary_interval
            )
            if not should_run:
                continue

            crop = self._crop(frame, track.bbox)
            if crop is None:
                continue

            label, cls_conf = self._classify(crop)
            track.cls_label = label
            track.cls_conf = cls_conf
            track.secondary_states = {
                "door_gap_state": label,
                "contact_state": "contact" if label == "close" else "separated" if label == "open" else "unknown",
            }
            track.last_secondary_frame = self.frame_index
            processed += 1

        return processed

    @staticmethod
    def _crop(frame, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _classify(self, crop) -> tuple[str, float]:
        result = self.cls_model.predict(crop, imgsz=self.config.cls_imgsz, half=self.config.half, verbose=False)[0]
        if result.probs is None:
            return "unknown", 0.0

        top_class_id = int(result.probs.top1)
        top_class_conf = float(result.probs.top1conf.item())
        label = str(result.names[top_class_id]).strip().lower()
        if label not in {"open", "close"}:
            label = "unknown"
        return label, top_class_conf

    def _draw_overlay(
        self,
        frame,
        tracks: list[Track],
        events: list[dict[str, object]],
        edge_outputs: list[dict[str, object]],
        perf: dict[str, float],
    ):
        vis = frame.copy()
        event_map = {int(event["track_id"]): event for event in events}
        edge_map = {int(event["track_id"]): event for event in edge_outputs}

        for track in tracks:
            color = (0, 190, 0) if track.cls_label == "open" else (0, 0, 220)
            if track.cls_label == "unknown":
                color = (255, 140, 0)

            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            event_name = str(event_map.get(track.track_id, {}).get("event_name", "idle"))
            edge_output = edge_map.get(track.track_id, {})
            stable_state = str(edge_output.get("state", "UNKNOWN"))
            stable_frames = int(edge_output.get("stable_frames", 0))
            cooldown = bool(edge_output.get("cooldown_active", False))
            label = (
                f"id={track.track_id} {track.cls_label} {track.cls_conf:.2f} "
                f"det={track.det_conf:.2f} {event_name} edge={stable_state} sf={stable_frames}"
            )
            self._draw_label(vis, label, x1, y1, color)
            if cooldown:
                cv2.putText(
                    vis,
                    "cooldown",
                    (x1, min(vis.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

        total_ms = sum(value for key, value in perf.items() if key.endswith("_ms"))
        fps = 1000.0 / max(total_ms, 1e-6)
        cv2.putText(vis, f"FPS: {fps:.1f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"Mode: {self.stabilizer.mode if self.stabilizer is not None else 'disabled'}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"sec budget: {self.config.max_secondary_targets_per_frame} det {perf['detection_ms']:.1f}ms sec {perf['secondary_ms']:.1f}ms",
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return vis

    @staticmethod
    def _draw_label(frame, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - th - baseline - 8)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + baseline + 8), color, -1)
        cv2.putText(frame, text, (x1 + 4, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
