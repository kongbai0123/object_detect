import argparse
import time
import cv2
import os
from pathlib import Path
from datetime import datetime
from collections import deque, Counter
from ultralytics import YOLO
from anti_gravity.settings import settings
from anti_gravity.logger import logger

# =====================================================================
# 🛠️ 工業級即時推論參數 (Industrial Live Parameters)
# =====================================================================
THRESHOLDS = {
    0: 0.35,  # OPEN (高敏觸發)
    1: 0.60   # CLOSE (嚴謹監控)
}
MIN_BOX_AREA_RATIO = 0.02
MAX_BOX_AREA_RATIO = 0.75

SMOOTHING_WINDOW = 10      # 時序平滑窗口
CONFIRM_RATIO = 0.6        # 60% 判定閾值
# =====================================================================

class IndustrialDecisionEngine:
    """
    [Decision Engine] 將偵測信號轉化為穩定的工業狀態。
    """
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.stable_state = "CLOSE"
        self.colors = {"OPEN": (0, 0, 255), "CLOSE": (0, 255, 0)}

    def update(self, detections) -> str:
        current_frame_opinion = None
        if detections:
            # 優先權邏輯：Open 優先
            opens = [d for d in detections if d['cls'] == 0]
            current_frame_opinion = "OPEN" if opens else "CLOSE"
        
        if current_frame_opinion:
            self.history.append(current_frame_opinion)
        
        if self.history:
            counts = Counter(self.history)
            most_common, count = counts.most_common(1)[0]
            if count >= (SMOOTHING_WINDOW * CONFIRM_RATIO):
                self.stable_state = most_common
        return self.stable_state

def main():
    parser = argparse.ArgumentParser(description="Industrial Door Detection Engine")
    parser.add_argument("--model", type=str, default=str(settings.paths.models_promoted / "global_best.pt"), help="Path to YOLO model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file/stream")
    parser.add_argument("--imgsz", type=int, default=832, help="Inference image size")
    parser.add_argument("--save", action="store_true", help="Auto-save inference result")
    args = parser.parse_args()

    # 1. 載入資源
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"❌ 無法開啟影像來源: {source}")
        return

    # 2. 寫入器準備 (側錄功能)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_area = width * height
    temp_output = Path("_temp_detect.mp4")
    writer = cv2.VideoWriter(str(temp_output), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))

    engine = IndustrialDecisionEngine()
    logger.info(f"🚀 Engine Started: Source={source}, Model={Path(args.model).name}")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            
            # 3. 執行推論
            results = model.predict(frame, imgsz=args.imgsz, conf=0.2, classes=[0, 1], verbose=False)
            
            # 4. 數據過濾
            valid_detections = []
            for box in results[0].boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                if conf < THRESHOLDS.get(cls, 0.5): continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if MIN_BOX_AREA_RATIO * frame_area < area < MAX_BOX_AREA_RATIO * frame_area:
                    valid_detections.append({'cls': cls, 'conf': conf, 'xyxy': [x1, y1, x2, y2]})

            # 5. 決策更新
            state = engine.update(valid_detections)
            
            # 6. UI 繪製
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            # 左上角狀態欄
            cv2.putText(frame, f"FPS: {fps:.1f} | STATE: {state}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 繪製主目標框
            target_cls = 0 if state == "OPEN" else 1
            relevant_dets = [d for d in valid_detections if d['cls'] == target_cls]
            if relevant_dets:
                d = sorted(relevant_dets, key=lambda x: x['conf'], reverse=True)[0]
                x1, y1, x2, y2 = map(int, d['xyxy'])
                color = engine.colors[state]
                thickness = 4 if state == "OPEN" else 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"DOOR {state}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Antigravity Industrial Inference", frame)
            if writer.isOpened(): writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        logger.info("🛑 Engine Shutdown.")

    # 存檔處理
    if args.save or input("\n❓ 是否儲存推論影片？(y/n): ").lower() == 'y':
        output_dir = Path("object_detect/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video = output_dir / f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        import shutil
        shutil.move(str(temp_output), str(final_video))
        logger.info(f"✅ 影片已儲存: {final_video}")
    elif temp_output.exists():
        os.remove(temp_output)

if __name__ == "__main__":
    main()