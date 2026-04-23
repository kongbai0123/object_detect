import cv2
import torch
import time
import argparse
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO
from anti_gravity.settings import settings
from anti_gravity.logger import logger

# =====================================================================
# 🛠️ 雙軌決策參數 (Industrial Decision Parameters)
# =====================================================================
THRESHOLDS = {
    0: 0.30,  # OPEN (較靈敏，確保 Recall)
    1: 0.60   # CLOSE (較嚴格，確保 Precision)
}
# 偵測框的面積與畫面占比
MIN_BOX_AREA_RATIO = 0.02
MAX_BOX_AREA_RATIO = 0.75

SMOOTHING_WINDOW = 10      # 投票窗口
CONFIRM_RATIO = 0.6        # 60% 一致性即切換狀態
# =====================================================================

class IndustrialDecisionEngine:
    """
    [Decision Engine] 雙軌判定，Open 具有優先決策權。
    """
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.stable_state = "CLOSE"
        self.colors = {"OPEN": (0, 0, 255), "CLOSE": (0, 255, 0)}

    def update(self, detections) -> str:
        current_frame_opinion = None
        
        if detections:
            # 優先權邏輯：只要偵測到合格的 OPEN，本影格傾向於 OPEN
            opens = [d for d in detections if d['cls'] == 0]
            if opens:
                current_frame_opinion = "OPEN"
            else:
                current_frame_opinion = "CLOSE"
        
        if current_frame_opinion:
            self.history.append(current_frame_opinion)
        
        # 投票決策
        if self.history:
            counts = Counter(self.history)
            most_common, count = counts.most_common(1)[0]
            if count >= (SMOOTHING_WINDOW * CONFIRM_RATIO):
                self.stable_state = most_common
                
        return self.stable_state

def process_video_industrial(model_path, video_path, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_area = width * height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    engine = IndustrialDecisionEngine()
    
    logger.info(f"Industrial Dual-Track Detection: {video_path.name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 執行雙類別偵測
        results = model.predict(frame, conf=0.2, classes=[0, 1], verbose=False)
        
        valid_detections = []
        best_box = None
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            # --- [過濾 1: 專屬閾值] ---
            if conf < THRESHOLDS.get(cls, 0.5): continue
            
            # --- [過濾 2: 幾何約束] ---
            x1, y1, x2, y2 = xyxy
            area = (x2 - x1) * (y2 - y1)
            if not (MIN_BOX_AREA_RATIO * frame_area < area < MAX_BOX_AREA_RATIO * frame_area):
                continue
            
            valid_detections.append({'cls': cls, 'conf': conf, 'xyxy': xyxy})

        # 2. 引擎決策
        state = engine.update(valid_detections)
        
        # 3. 渲染 UI
        # A. 左上角效能資訊
        cv2.putText(frame, f"FPS: {fps:.1f} | STATE: {state}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # B. 繪製框體與標籤
        # 優先找與當前 stable_state 一致的框來畫
        target_cls = 0 if state == "OPEN" else 1
        relevant_dets = [d for d in valid_detections if d['cls'] == target_cls]
        
        if relevant_dets:
            # 取信心度最高的一個
            d = sorted(relevant_dets, key=lambda x: x['conf'], reverse=True)[0]
            x1, y1, x2, y2 = map(int, d['xyxy'])
            color = engine.colors[state]
            thickness = 4 if state == "OPEN" else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"DOOR {state}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Done! Industrial report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(settings.paths.experiments / "incremental/exp_incremental_auto_iter_all_0423_1405/weights/best.pt"))
    parser.add_argument("--source", type=str, default=str(settings.paths.storage / "assets/videos"))
    args = parser.parse_args()
    
    m_path = Path(args.model)
    s_path = Path(args.source)
    out_dir = settings.paths.storage / "artifacts/evaluations/videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
    target_videos = [f for f in s_path.iterdir() if f.suffix in VIDEO_EXTS] if s_path.is_dir() else [s_path]
    
    for v_path in target_videos:
        o_path = out_dir / f"{v_path.stem}_industrial_eval.mp4"
        process_video_industrial(m_path, v_path, o_path)
