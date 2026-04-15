import argparse
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
# 預設指向 object_detect 目錄下的權重檔
DEFAULT_MODEL_PATH = ROOT / "object_detect" / "best.pt"

def main():
    parser = argparse.ArgumentParser(description="Simplified YOLO Object Detection")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()

    # 檢查模型路徑
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: 找不到模型檔案 {model_path}")
        return

    print(f"Loading model: {model_path}...")
    model = YOLO(str(model_path))

    # 開啟影像來源
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: 無法開啟影像來源 {source}")
        return

    window_name = f"YOLO Detection - {model_path.name} (Press 'q' to quit)"
    print(f"Starting detection on {source}. Inference size: {args.imgsz}")

    # === [側錄優化] 初始化暫存影片寫入器 ===
    temp_output = ROOT / "object_detect" / "_temp_detect.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_save = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps_save, (width, height))

    while True:
        t0 = perf_counter()
        
        ok, frame = cap.read()
        if not ok:
            print("串流結束或讀取失敗。")
            break

        # YOLO 單波推論 (不進行任何後續狀態機處理)
        # verbose=False 以維持終端機乾淨
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        
        # 使用官方 plot() 直接取得繪製好的偵測框影像 (不改動任何 Box 樣式)
        annotated_frame = results[0].plot()

        # 計算並顯示 FPS
        fps = 1.0 / max(perf_counter() - t0, 1e-6)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(window_name, annotated_frame)
        
        # 寫入暫存檔
        if writer.isOpened():
            writer.write(annotated_frame)

        # q 鍵跳出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer.isOpened():
        writer.release()
    cv2.destroyAllWindows()
    print("檢測程式已暫停。")

    # === [互動優化] 詢問是否儲存影片 ===
    save_choice = input("\n 推論結束。是否儲存預測影片至 outputs 目錄？(y/n): ").strip().lower()
    if save_choice == 'y':
        output_dir = ROOT / "object_detect" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video = output_dir / f"pred_{stamp}.mp4"
        import shutil
        shutil.move(str(temp_output), str(final_video))
        print(f" ✅ 影片已儲存: {final_video}")
    else:
        if temp_output.exists():
            os.remove(temp_output)
        print(" ❌ 已捨棄預測影片。")

    print("檢測程式已正式關閉。")

if __name__ == "__main__":
    main()