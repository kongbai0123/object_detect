from __future__ import annotations

import os
import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from pipeline_notice import print_pipeline_notice


def find_best_model() -> Path:
    """
    [Weights Discovery System] 動態搜尋當前系統最強大腦
    優先級: global_best.pt > latest_best.pt > 最新 exp*/weights/best.pt
    """
    runs_dir = ROOT / "data/7_experiments"
    
    # 策略 1: 冠軍模型 (經由 Promotion Gate 晉升)
    global_best = runs_dir / "weight/global_best.pt"
    if global_best.exists():
        return global_best
        
    # # 策略 2: 最新挑戰者模型
    # latest_challenger = runs_dir / "weight/latest_best.pt"
    # if latest_challenger.exists():
    #     return latest_challenger
        
    # # 策略 3: 遍歷所有 exp{num} 尋找最新修改時間的 best.pt
    # import glob
    # found = sorted(
    #     glob.glob(str(runs_dir / "exp*/weights/best.pt"), recursive=True), 
    #     key=os.path.getmtime
    # )
    # if found:
    #     return Path(found[-1])
        
    # # 策略 4: Fallback
    # return Path("yolov8n.pt")

DEFAULT_MODEL = find_best_model()
DEFAULT_OUTPUT_DIR = ROOT / "data/7_experiments/video_preds"
DEFAULT_VIDEO_DIR = ROOT / "data/1_raw/videos"
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MKV", ".AVI", ".MOV")


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def find_default_video_source() -> str | None:
    if not DEFAULT_VIDEO_DIR.exists():
        return None

    candidates = []
    for ext in VIDEO_EXTS:
        candidates.extend(DEFAULT_VIDEO_DIR.glob(f"*{ext}"))

    candidates = sorted({path.resolve() for path in candidates})
    if not candidates:
        return None
    return str(candidates[0])


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
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(source, int):
        return DEFAULT_OUTPUT_DIR / f"camera_pred_{stamp}{suffix}"
    return DEFAULT_OUTPUT_DIR / f"{Path(source).stem}_pred_{stamp}{suffix}"


def open_capture(source: int | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source}")
    return cap


def create_video_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def run_video_inference(
    model: YOLO,
    source: int | str,
    conf: float,
    iou: float,
    show_window: bool,
    output_path: Path,
) -> None:
    cap = open_capture(source)
    writer = create_video_writer(cap, output_path)
    window_name = "Video PT Test (press q to quit)"
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = perf_counter()
        result = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
        vis = result.plot()
        infer_fps = 1.0 / max(perf_counter() - t0, 1e-6)

        det_count = 0 if result.boxes is None else len(result.boxes)
        frame_count += 1

        cv2.putText(
            vis,
            f"Infer FPS: {infer_fps:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            f"Frame: {frame_count}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Detections: {det_count}",
            (12, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (120, 255, 120),
            2,
        )

        writer.write(vis)

        if show_window:
            cv2.imshow(window_name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif frame_count % 30 == 0:
            print(f"processed {frame_count} frames")

    cap.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"video inference complete, frames processed: {frame_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained YOLO .pt model on a video and save labeled output")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="trained YOLO .pt path")
    parser.add_argument("--source", type=str, default="", help="video path or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--display",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="show OpenCV window or not",
    )
    parser.add_argument("--output", type=str, default="", help="optional labeled video output path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    source_arg = args.source.strip()
    if not source_arg:
        default_video = find_default_video_source()
        if default_video is None:
            raise SystemExit(
                "No --source provided and no video found in data/1_raw/videos. "
                "Please pass --source <video_path>."
            )
        print(f"no --source provided, using first video found: {default_video}")
        source_arg = default_video

    source = parse_source(source_arg)
    show_window = resolve_display_mode(args.display)
    output_path = Path(args.output) if args.output else default_output_path(source, ".mp4")

    model = YOLO(str(model_path))
    run_video_inference(
        model=model,
        source=source,
        conf=args.conf,
        iou=args.iou,
        show_window=show_window,
        output_path=output_path,
    )

    print_pipeline_notice(
        output_paths=output_path,
        next_script="src/analyze_errors.py",
        notes=[
            "輸出影片已包含模型預測框與類別標籤。",
            "若你要測不同權重，可改 --model 指向 latest_best.pt 或其他實驗權重。",
        ],
    )


if __name__ == "__main__":
    main()
