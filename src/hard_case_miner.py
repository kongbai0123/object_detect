import argparse
from pathlib import Path
import cv2
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice


ROOT = Path(__file__).resolve().parent.parent


def find_latest_best_model():
    runs_dir = ROOT / "data/7_experiments"
    candidates = [p for p in runs_dir.rglob("best.pt") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return runs_dir / "weight/global_best.pt"


def iter_media_files(source_dir: Path):
    if source_dir.is_dir():
        patterns = ("*.mp4", "*.avi", "*.mkv", "*.jpg", "*.jpeg", "*.png")
        files = []
        for pattern in patterns:
            files.extend(sorted(source_dir.rglob(pattern)))
        return files
    return [source_dir]


def run_miner(model_path, source_dir, output_dir, sample_every=10):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("缺少 ultralytics，請先執行: pip install ultralytics")
        return

    model_path = Path(model_path)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    fp_dir = output_dir / "high_conf_review"
    unc_dir = output_dir / "low_conf_uncertain"

    fp_dir.mkdir(parents=True, exist_ok=True)
    unc_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"找不到模型權重: {model_path}")
        return

    media_files = iter_media_files(source_dir)
    if not media_files:
        print(f"在 {source_dir} 找不到可處理的影片或圖片")
        return

    print(f"載入模型: {model_path}")
    model = YOLO(str(model_path))

    print(f"開始進行 Hard Case Mining，共 {len(media_files)} 個媒體檔")

    processed_frames = 0
    saved_fp = 0
    saved_unc = 0

    for media in media_files:
        print(f"處理中: {media.name}")

        if media.suffix.lower() in {".mp4", ".avi", ".mkv"}:
            cap = cv2.VideoCapture(str(media))
            local_frame_idx = 0

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                local_frame_idx += 1
                processed_frames += 1

                if sample_every > 1 and local_frame_idx % sample_every != 0:
                    continue

                result = model.predict(frame, verbose=False)[0]
                boxes = result.boxes
                if len(boxes) == 0:
                    continue

                max_conf = float(boxes.conf.max())
                vis = result.plot()

                if max_conf > 0.6:
                    save_path = fp_dir / f"{media.stem}_f{local_frame_idx}.jpg"
                    cv2.imwrite(str(save_path), vis)
                    saved_fp += 1
                elif 0.2 < max_conf <= 0.6:
                    save_path = unc_dir / f"{media.stem}_f{local_frame_idx}.jpg"
                    cv2.imwrite(str(save_path), vis)
                    saved_unc += 1

            cap.release()
            continue

        processed_frames += 1
        frame = cv2.imread(str(media))
        if frame is None:
            continue

        result = model.predict(frame, verbose=False)[0]
        boxes = result.boxes
        if len(boxes) == 0:
            continue

        max_conf = float(boxes.conf.max())
        vis = result.plot()

        if max_conf > 0.6:
            save_path = fp_dir / media.name
            cv2.imwrite(str(save_path), vis)
            saved_fp += 1
        elif 0.2 < max_conf <= 0.6:
            save_path = unc_dir / media.name
            cv2.imwrite(str(save_path), vis)
            saved_unc += 1

    print("\nHard Case Mining 完成")
    print(f"處理幀數/圖片數: {processed_frames}")
    print(f"high_conf_review: {saved_fp} 張")
    print(f"low_conf_uncertain: {saved_unc} 張")
    print(f"輸出資料夾: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLO Hard Case Miner")
    parser.add_argument("--model", type=str, default=str(find_latest_best_model()), help="模型權重，預設自動使用最新 best.pt")
    parser.add_argument("--input", type=str, default=str(ROOT / "data/1_raw"), help="輸入影片、圖片或上層資料夾，會遞迴搜尋 1_raw 底下檔案")
    parser.add_argument("--output", type=str, default=str(ROOT / "data/8_hard_cases"), help="輸出 hard cases 的資料夾")
    parser.add_argument("--sample-every", type=int, default=10, help="影片每隔幾幀抽一張來做 mining")
    args = parser.parse_args()

    run_miner(args.model, args.input, args.output, sample_every=args.sample_every)
    print_pipeline_notice(
        output_paths=[str(Path(args.output) / "high_conf_review"), str(Path(args.output) / "low_conf_uncertain")],
        next_script="src/cvat_import.py",
        notes=[
            "high_conf_review 偏向高信心結果的人工複查，low_conf_uncertain 偏向低信心與邊界案例。",
            "確認 hard cases 後，建議送進 CVAT 補標，再回流到訓練資料。",
        ],
    )
