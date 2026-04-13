from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from branchs_runtime import BranchsPipeline, BranchsRuntimeConfig


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "object_detect" / "outputs"
DEFAULT_STAGE1_MODEL = "yolov8n.pt"
DEFAULT_STAGE2_MODEL = ROOT / "object_detect" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def can_use_imshow() -> bool:
    try:
        cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test_window")
        return True
    except cv2.error:
        return False


def resolve_display_mode(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return can_use_imshow()


def default_output_path(source: int | str, suffix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"camera_{source}" if isinstance(source, int) else Path(source).stem
    return OUTPUT_DIR / f"{stem}_{stamp}{suffix}"


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_pipeline(args: argparse.Namespace) -> BranchsPipeline:
    det_model = YOLO(str(args.model1))
    cls_model = YOLO(str(args.model2))
    config = BranchsRuntimeConfig(
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        det_imgsz=args.det_imgsz,
        cls_imgsz=args.cls_imgsz,
        secondary_interval=args.secondary_interval,
        max_secondary_targets_per_frame=args.max_secondary_targets,
        min_transition_hits=args.min_transition_hits,
        half=not args.no_half,
        enable_stabilization=not args.no_stabilization,
        stabilization_alpha=args.stabilization_alpha,
        open_conf_threshold=args.open_conf,
        close_conf_threshold=args.close_conf,
        min_box_area_ratio=args.min_box_area_ratio,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        track_match_iou=args.track_match_iou,
        max_center_shift_ratio=args.max_center_shift_ratio,
        persist_window=args.persist_window,
        persist_required=args.persist_required,
        open_enter_frames=args.open_enter_frames,
        open_exit_frames=args.open_exit_frames,
        close_enter_frames=args.close_enter_frames,
        state_cooldown_ms=args.state_cooldown_ms,
        state_timeout_ms=args.state_timeout_ms,
    )
    return BranchsPipeline(det_model=det_model, cls_model=cls_model, config=config)


def run_stream(args: argparse.Namespace) -> None:
    source = parse_source(args.source)
    show_window = resolve_display_mode(args.display)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source}")

    pipeline = build_pipeline(args)
    output_path = Path(args.output) if args.output else None
    events_path = Path(args.events) if args.events else default_output_path(source, "_events.jsonl")
    states_path = Path(args.states) if args.states else default_output_path(source, "_states.jsonl")

    writer = None
    if output_path is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_id = 0
    t_start = perf_counter()
    while True:
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int((perf_counter() - t_start) * 1000.0)
        result = pipeline.process_frame(frame, timestamp_ms=timestamp_ms)
        frame_id += 1

        if writer is not None:
            writer.write(result["frame"])

        for event in result["events"]:
            append_jsonl(events_path, {"frame_id": frame_id, "timestamp_ms": timestamp_ms, **event})
        for state_output in result["edge_outputs"]:
            append_jsonl(states_path, {"frame_id": frame_id, **state_output})

        if show_window:
            cv2.imshow("object_detect edge runtime", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif frame_id % 30 == 0:
            print(
                f"frame={frame_id} mode={result['mode']} "
                f"tracks={len(result['tracks'])} edge_outputs={len(result['edge_outputs'])}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"completed frames={frame_id}")
    print(f"events={events_path}")
    print(f"states={states_path}")
    if output_path is not None:
        print(f"video={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="object_detect edge runtime with temporal state machine")
    parser.add_argument("--model1", type=str, default=DEFAULT_STAGE1_MODEL, help="stage-1 detector model")
    parser.add_argument("--model2", type=str, default=str(DEFAULT_STAGE2_MODEL), help="stage-2 classifier model")
    parser.add_argument("--source", type=str, default="0", help="camera index, image path, or video path")
    parser.add_argument("--display", type=str, default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--output", type=str, default="", help="optional annotated video path")
    parser.add_argument("--events", type=str, default="", help="JSONL for behavior events")
    parser.add_argument("--states", type=str, default="", help="JSONL for stable edge states")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 means full stream")

    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--secondary-interval", type=int, default=3)
    parser.add_argument("--max-secondary-targets", type=int, default=3)
    parser.add_argument("--min-transition-hits", type=int, default=2)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--no-stabilization", action="store_true")
    parser.add_argument("--stabilization-alpha", type=float, default=0.8)

    parser.add_argument("--open-conf", type=float, default=0.80)
    parser.add_argument("--close-conf", type=float, default=0.55)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.010)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.20)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.50)
    parser.add_argument("--track-match-iou", type=float, default=0.50)
    parser.add_argument("--max-center-shift-ratio", type=float, default=0.15)
    parser.add_argument("--persist-window", type=int, default=5)
    parser.add_argument("--persist-required", type=int, default=3)
    parser.add_argument("--open-enter-frames", type=int, default=3)
    parser.add_argument("--open-exit-frames", type=int, default=5)
    parser.add_argument("--close-enter-frames", type=int, default=3)
    parser.add_argument("--state-cooldown-ms", type=int, default=2000)
    parser.add_argument("--state-timeout-ms", type=int, default=3000)

    args = parser.parse_args()

    if args.model2 and not Path(args.model2).exists():
        raise FileNotFoundError(f"stage-2 model not found: {args.model2}")
    if args.model1 != DEFAULT_STAGE1_MODEL and not Path(args.model1).exists():
        raise FileNotFoundError(f"stage-1 model not found: {args.model1}")

    source = parse_source(args.source)
    if isinstance(source, str):
        path = Path(source)
        if path.exists() and path.suffix.lower() in IMAGE_EXTS:
            raise NotImplementedError("image mode is not implemented in this edge runtime entry; use video/camera input")

    run_stream(args)


if __name__ == "__main__":
    main()
