import cv2
import time
import csv
import psutil
import os
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO

from pathlib import Path

# ==========================================
# 1. 系統與推論參數設定
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "data/7_experiments/weight/global_best.pt")

IMGSZ = 832
PREDICT_CONF = 0.25      
CONF_OPEN = 0.80
CONF_CLOSE = 0.85
IOU_THRES = 0.60

# ROI 設定 (硬式矩形: [x1, y1, x2, y2])
ROI_RECT = [100, 100, 1180, 700]  

# 穩定化與 Alert 參數 (現場校正起始值)
OPEN_CONFIRM_FRAMES = 2
CLOSE_CONFIRM_FRAMES = 5
OPEN_MIN_AREA = 0.05       
OPEN_ALERT_THRES = 0.10    
RELEASE_AREA_THRES = 0.08  
ALERT_ON_FRAMES = 3
ALERT_OFF_FRAMES = 5

# 系統延遲監控雙門檻 (ms)
LAG_WARN_MS = 50
LAG_CRIT_MS = 100

class AsymmetricStabilizer:
    """處理非對稱多幀投票與 Alert 緩衝邏輯"""
    def __init__(self):
        self.state_history = deque(maxlen=max(OPEN_CONFIRM_FRAMES, CLOSE_CONFIRM_FRAMES))
        self.current_state = "UNKNOWN"
        self.alert_on_counter = 0
        self.alert_off_counter = 0
        self.alert_active = False

    def update_state(self, candidates):
        open_candidates = [c for c in candidates if c['class'] == 'open']
        close_candidates = [c for c in candidates if c['class'] == 'close']
        
        valid_dangerous_opens = [c for c in open_candidates if c['area_ratio'] >= OPEN_MIN_AREA]
        
        current_frame_intent = "UNKNOWN"
        max_area_ratio = 0.0

        if valid_dangerous_opens:
            current_frame_intent = "OPEN"
            max_area_ratio = max([c['area_ratio'] for c in valid_dangerous_opens])
        elif close_candidates:
            current_frame_intent = "CLOSE"
            max_area_ratio = max([c['area_ratio'] for c in close_candidates])

        if current_frame_intent in ["OPEN", "CLOSE"]:
            self.state_history.append(current_frame_intent)

        history_list = list(self.state_history)
        if history_list[-OPEN_CONFIRM_FRAMES:].count("OPEN") == OPEN_CONFIRM_FRAMES:
            self.current_state = "OPEN"
        elif history_list[-CLOSE_CONFIRM_FRAMES:].count("CLOSE") == CLOSE_CONFIRM_FRAMES:
            self.current_state = "CLOSE"

        if self.current_state == "OPEN" and max_area_ratio >= OPEN_ALERT_THRES:
            self.alert_on_counter += 1
            self.alert_off_counter = 0
            if self.alert_on_counter >= ALERT_ON_FRAMES:
                self.alert_active = True
        elif self.current_state != "OPEN" or max_area_ratio < RELEASE_AREA_THRES:
            self.alert_off_counter += 1
            self.alert_on_counter = 0
            if self.alert_off_counter >= ALERT_OFF_FRAMES:
                self.alert_active = False

        counts = {
            'open_cands': len(open_candidates),
            'close_cands': len(close_candidates),
            'valid_danger_opens': len(valid_dangerous_opens)
        }
        return self.current_state, self.alert_active, max_area_ratio, counts

def check_roi(box_xyxy, roi):
    cx = (box_xyxy[0] + box_xyxy[2]) / 2
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    return (roi[0] <= cx <= roi[2]) and (roi[1] <= cy <= roi[3])

def main():
    # 解析 CLI 參數
    parser = argparse.ArgumentParser(description="PC Edge Testing Pipeline v1.1")
    parser.add_argument('--source', type=str, default='0', help='Camera index (e.g., 0) or path to mp4 file')
    args = parser.parse_args()
    
    video_source = int(args.source) if args.source.isdigit() else args.source

    print("--- 系統初始化 ---")
    model = YOLO(MODEL_PATH)
    
    # [修正 4] 類別硬檢查 (Assert)
    expected_names = {0: 'open', 1: 'close'}
    if model.names != expected_names:
        raise RuntimeError(f"[致命錯誤] 模型類別映射不符！\n預期: {expected_names}\n實際: {model.names}\n請檢查 weights 檔案！")
    print(f"類別映射檢查通過: {model.names}")

    cap = cv2.VideoCapture(video_source) 
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("--- 嘗試鎖定相機曝光 ---")
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cap.set(cv2.CAP_PROP_EXPOSURE, -5) 
        time.sleep(1)
        print(f"實際 AUTO_EXPOSURE: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        print(f"實際 EXPOSURE: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"實際 GAIN: {cap.get(cv2.CAP_PROP_GAIN)}")
    
    print("------------------")

    stabilizer = AsymmetricStabilizer()
    process = psutil.Process(os.getpid())
    
    os.makedirs('logs', exist_ok=True)
    run_id = int(time.time())
    
    frame_log_file = open(f"logs/frame_log_{run_id}.csv", "w", newline="")
    frame_writer = csv.writer(frame_log_file)
    frame_writer.writerow(["Frame_ID", "Timestamp", "Stable_State", "Alert", "Max_Area_Ratio", 
                           "Open_Cands", "Close_Cands", "Danger_Opens",
                           "FPS", "Capture_ms", "Infer_ms", "Logic_ms", "Render_ms", "E2E_ms", "RAM_MB", "Lag_Status"])

    det_log_file = open(f"logs/detection_log_{run_id}.csv", "w", newline="")
    det_writer = csv.writer(det_log_file)
    det_writer.writerow(["Frame_ID", "Det_ID", "Class_Raw", "Conf_Raw", "x1", "y1", "x2", "y2", 
                         "Area_Ratio", "ROI_Pass", "Filtered_Pass"])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f"logs/output_{run_id}.mp4", fourcc, 20.0, (1280, 720))

    frame_id = 0
    lag_warn_count = 0
    lag_crit_count = 0

    while cap.isOpened():
        t_frame_start = time.time()

        # [1. Capture Profiling]
        ret, frame = cap.read()
        if not ret: break
        t_cap_end = time.time()
        capture_ms = (t_cap_end - t_frame_start) * 1000
        
        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w 

        # [2. Inference Profiling]
        results = model.predict(frame, imgsz=IMGSZ, conf=PREDICT_CONF, iou=IOU_THRES, verbose=False)
        t_infer_end = time.time()
        infer_ms = (t_infer_end - t_cap_end) * 1000

        # [3. Logic Profiling]
        filtered_candidates = []
        det_id = 0

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                area_ratio = float(box_area / img_area)

                roi_pass = check_roi([x1, y1, x2, y2], ROI_RECT)
                conf_pass = (label == 'open' and conf >= CONF_OPEN) or (label == 'close' and conf >= CONF_CLOSE)
                filtered_pass = roi_pass and conf_pass

                det_writer.writerow([frame_id, det_id, label, f"{conf:.3f}", x1, y1, x2, y2, 
                                     f"{area_ratio:.3f}", roi_pass, filtered_pass])
                det_id += 1

                if filtered_pass:
                    filtered_candidates.append({
                        'class': label, 'conf': conf, 'box': [x1, y1, x2, y2], 'area_ratio': area_ratio
                    })

        stable_state, alert_active, max_area, counts = stabilizer.update_state(filtered_candidates)
        t_logic_end = time.time()
        logic_ms = (t_logic_end - t_infer_end) * 1000

        # [4. Render Profiling]
        cv2.rectangle(frame, (ROI_RECT[0], ROI_RECT[1]), (ROI_RECT[2], ROI_RECT[3]), (255, 255, 0), 2)
        for c in filtered_candidates:
            color = (0, 0, 255) if c['class'] == 'open' else (0, 255, 0)
            bx1, by1, bx2, by2 = c['box']
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
            cv2.putText(frame, f"{c['class']} {c['conf']:.2f} A:{c['area_ratio']:.2f}", 
                        (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(frame, (10, 10), (550, 200), (0, 0, 0), -1)
        cv2.putText(frame, f"State: {stable_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if stable_state == "OPEN" else (0, 255, 0), 3)
        cv2.putText(frame, f"Alert: {'ON' if alert_active else 'OFF'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if alert_active else (255, 255, 255), 3)

        if alert_active:
            cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 255), 10)

        # 預計算 Render 時間與總 E2E (為了將數據印在當前畫面上)
        t_pre_render_end = time.time()
        render_ms_est = (t_pre_render_end - t_logic_end) * 1000
        e2e_ms = capture_ms + infer_ms + logic_ms + render_ms_est
        
        fps = 1000 / e2e_ms if e2e_ms > 0 else 0
        ram_mb = process.memory_info().rss / (1024 * 1024)
        
        # [雙門檻 Lag 監控]
        lag_status = "OK"
        if e2e_ms > LAG_CRIT_MS:
            lag_status = "CRIT"
            lag_crit_count += 1
        elif e2e_ms > LAG_WARN_MS:
            lag_status = "WARN"
            lag_warn_count += 1

        lag_color = (0, 255, 0) if lag_status == "OK" else (0, 255, 255) if lag_status == "WARN" else (0, 0, 255)

        cv2.putText(frame, f"Cap:{capture_ms:.0f}ms Inf:{infer_ms:.0f}ms Log:{logic_ms:.0f}ms Ren:{render_ms_est:.0f}ms", 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"E2E:{e2e_ms:.0f}ms FPS:{fps:.1f} RAM:{ram_mb:.0f}MB", 
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Lag [{lag_status}] - W:{lag_warn_count} C:{lag_crit_count}", 
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lag_color, 1)

        # 寫入 Frame Log (包含精準的 Timing)
        frame_writer.writerow([frame_id, time.time(), stable_state, alert_active, f"{max_area:.3f}",
                               counts['open_cands'], counts['close_cands'], counts['valid_danger_opens'],
                               f"{fps:.1f}", f"{capture_ms:.1f}", f"{infer_ms:.1f}", f"{logic_ms:.1f}", f"{render_ms_est:.1f}", 
                               f"{e2e_ms:.1f}", f"{ram_mb:.1f}", lag_status])

        if frame_id % 50 == 0:
            frame_log_file.flush()
            det_log_file.flush()
            os.fsync(frame_log_file.fileno())
            os.fsync(det_log_file.fileno())

        # 最終繪圖與錄影
        out_video.write(frame)
        cv2.imshow('PC Edge Testing Pipeline v1.1', frame) 

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    frame_log_file.close()
    det_log_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()