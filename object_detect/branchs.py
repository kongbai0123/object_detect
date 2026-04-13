from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

def calc_iou(box1: np.ndarray | list | tuple, box2: np.ndarray | list | tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    if interArea == 0: return 0.0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea)

# 無論從哪個目錄執行，路徑永遠相對於此腳本所在位置
ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =====================================================================
# [設定] 兩階段架構之預設模型設定
# =====================================================================
# 【Stage 1】 預設通用物體追蹤模型 (負責先找出車輛 ROI，交給內建 ByteTrack 免去手刻之苦)
DEFAULT_STAGE1_MODEL = "yolov8n.pt"  # 確保你根目錄或環境會自動下載此通用權重
# 【Stage 2】 自訓練的 3 公尺預警判定網路 (負責對 ROI 裡面找 door_opening)
DEFAULT_STAGE2_MODEL = ROOT / "object_detect/best.pt"  

DEFAULT_OUTPUT_DIR = ROOT / "object_detect/outputs"

# COCO 類別中與車門預測相關的載具 ID (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = {2, 5, 7}


# =====================================================================
# RuntimeConfig — 執行期可即時調變的參數控制中心
# 鍵盤熱鍵映射表：
#   c  — 切換 CLAHE 補償 ON/OFF
#   d  — 切換 debug_illu 比對圖 ON/OFF
#   i  — 切換 控制面板顯示 ON/OFF
#   +  — dark_threshold +5
#   -  — dark_threshold -5
#   2  — Stage 2 預警間隔 1 → 2 → 3 幀 循環
#   q  — 離開
# =====================================================================
@dataclass
class RuntimeConfig:
    # --- 光照補償控制 ---
    enable_clahe: bool = True           # 是否啟用 CLAHE 自適應補償
    dark_threshold: int = 85            # mean(L) 低於此即嗚動 CLAHE (節圍 15~150)

    # --- Stage 2 觸發策略 ---
    conf2: float = 0.20                 # Stage 2 信心度門檻 (可動態調)
    stage2_interval: int = 1            # [相容保留] 面板上熱鍵切換用
    min_roi_px: int = 40                # ROI 寬或高小於此 px 則跳過 Stage 2
    max_speed_px: float = 15.0          # 速度 < max_speed=MEDIUM, <3=HIGH, >=max=LOW
    grace_frames: int = 20              # 追蹤丟失可容忍的最大幀數

    # --- 展示控制 ---
    show_dashboard: bool = True         # 是否在畫面上顯示參數控制面板
    enable_debug_illu: bool = False     # 是否儲存 debug_illu 比對圖 (不說全面支援)

    # --- 統計 (由引擎寫入, 使用者不直接修改) ---
    stat_vehicles: int = 0
    curr_high: int = 0                  # 當幀 HIGH 優先級數量
    curr_medium: int = 0                # 當幀 MEDIUM 優先級數量
    curr_low: int = 0                   # 當幀 LOW 優先級數量
    
    stat_s2_batches: int = 0            # 累積 Stage 2 呼叫次數 (Batch基準)
    stat_s2_rois: int = 0               # 累積 Stage 2 處理的總 ROI 車數
    stat_clahe_calls: int = 0
    stat_alarms: int = 0
    stat_frame: int = 0



# =====================================================================
# Module 1: ROI Illumination Assessment & Compensation (光照評估與補償)
# =====================================================================
def assess_and_enhance_roi(
    roi_img: np.ndarray,
    config: "RuntimeConfig",
    debug_dir: Path | None = None,
    frame_id: int = 0,
    track_id: int = 0
) -> tuple[np.ndarray, str, float]:
    """
    動態評估 ROI 亮度並補償。所有策略切換按照 config 即時變動。
    """
    if roi_img.size == 0:
        return roi_img, "invalid", 0.0

    try:
        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    except Exception:
        return roi_img, "color_err", 0.0
    
    l_channel, a, b = cv2.split(lab)
    mean_l = float(np.mean(l_channel))
    std_l  = float(np.std(l_channel))
    
    # 策略 1：極限暗區 SNR 崩潰
    if mean_l < 15 and std_l < 10:
        _save_debug_comparison(roi_img, roi_img, "low_confidence_dark", mean_l, debug_dir if config.enable_debug_illu else None, frame_id, track_id)
        return roi_img, "low_confidence_dark", mean_l
        
    # 策略 2：強光過曝
    if mean_l > 220:
        _save_debug_comparison(roi_img, roi_img, "glare_ignored", mean_l, debug_dir if config.enable_debug_illu else None, frame_id, track_id)
        return roi_img, "glare_ignored", mean_l
        
    # 策略 3：低光源 — 但因應 RuntimeConfig.enable_clahe 開關
    if config.enable_clahe and mean_l < config.dark_threshold:
        config.stat_clahe_calls += 1
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        _save_debug_comparison(roi_img, enhanced_bgr, "clahe_enhanced", mean_l, debug_dir if config.enable_debug_illu else None, frame_id, track_id)
        return enhanced_bgr, "clahe_enhanced", mean_l
        
    _save_debug_comparison(roi_img, roi_img, "normal", mean_l, debug_dir if config.enable_debug_illu else None, frame_id, track_id)
    return roi_img, "normal", mean_l



def _save_debug_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    strategy: str,
    mean_l: float,
    debug_dir: Path | None,
    frame_id: int,
    track_id: int
) -> None:
    """
    儲存「原圖 | 補償後 | 差異圖」三列式比對圖，供人工分析光照補償效果使用。
    只有當 --debug-illu 標誌啟用時才儲存，不會影響主程序 FPS。
    """
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    H = max(original.shape[0], 60)
    W = max(original.shape[1], 60)
    
    orig_r = cv2.resize(original, (W, H))
    enh_r = cv2.resize(enhanced, (W, H))
    
    # 差異圖：明顯化兩者變化包括車門緣細節差異
    diff = cv2.absdiff(orig_r, enh_r)
    diff_amplified = cv2.convertScaleAbs(diff, alpha=4.0)  # 放大 4倍讓差異更明顯
    diff_color = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)

    canvas = np.hstack([orig_r, enh_r, diff_color])
    
    # 加上文字說明列
    label_row = np.zeros((22, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_row, f"ORIG  mean_L={mean_l:.1f}", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,255), 1)
    cv2.putText(label_row, f"ENHANCED [{strategy}]", (W + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,255,100), 1)
    cv2.putText(label_row, f"DIFF x4", (W * 2 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,180,80), 1)
    canvas = np.vstack([label_row, canvas])
    
    fname = debug_dir / f"f{frame_id:05d}_car{track_id}_{strategy}.jpg"
    cv2.imwrite(str(fname), canvas)



# =====================================================================
# Module 2: The Two-Stage Early Warning Engine (預警決策引擎)
# =====================================================================
class EarlyWarningEngine:
    """
    以任務為導向的雙軌引擎：
    1. 用 Yolov8 原生權重在廣角原圖上「追蹤」車輛
    2. 對局部車輛實施「光線自適應強化」
    3. 對強化後的局部畫面用自訂權重研判「車門預警狀態」
    4. 實施時間濾波 (Temporal Voting) 封殺雜訊誤觸
    """
    def __init__(
        self,
        stage1_model: str,
        stage2_model: str,
        conf1: float = 0.3,
        iou2: float = 0.45,
        imgsz: int = 720,
        debug_illu_dir: Path | None = None,
        config: RuntimeConfig | None = None
    ):
        self.model1 = YOLO(str(stage1_model))
        self.model2 = YOLO(str(stage2_model))
        self.conf1 = conf1
        self.iou2 = iou2
        self.imgsz = imgsz
        self.debug_illu_dir = debug_illu_dir
        # config 是對外共享的 mutable 物件，主迴圈閇盤修改即影響下一幀
        self.config = config if config is not None else RuntimeConfig()
        
        self.track_memory = {}
        self.events = []
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> tuple[list[dict], list[dict]]:
        self.frame_count += 1
        cfg = self.config  # 對外共享的 RuntimeConfig
        cfg.stat_frame = self.frame_count

        # [Stage 1] 追蹤全景載具 (用原圖，保持追蹤器特徵穩定，避免補償後 ID 跳號)
        results1 = self.model1.track(
            frame, classes=list(VEHICLE_CLASSES), persist=True, verbose=False,
            conf=self.conf1, imgsz=self.imgsz
        )
        
        current_tracks = []
        new_events = []
        
        if results1 and results1[0].boxes and results1[0].boxes.id is not None:
            boxes1 = results1[0].boxes.xyxy.cpu().numpy().astype(int)
            ids1   = results1[0].boxes.id.cpu().numpy().astype(int)
            cfg.stat_vehicles = len(ids1)
            
            # --- 1. Grace Period & ID Inheritance ---
            active_tids = set(ids1)
            known_tids = set(self.track_memory.keys())
            new_tids = active_tids - known_tids
            lost_tids = known_tids - active_tids

            for idx, tid in enumerate(ids1):
                if tid in new_tids:
                    b1 = boxes1[idx]
                    best_iou, best_ltid = 0.0, None
                    for ltid in list(lost_tids):
                        lmem = self.track_memory[ltid]
                        if "last_bbox" in lmem:
                            iou = calc_iou(b1, lmem["last_bbox"])
                            if iou > 0.5 and iou > best_iou:
                                best_iou = iou
                                best_ltid = ltid
                    if best_ltid is not None:
                        self.track_memory[tid] = self.track_memory.pop(best_ltid)
                        lost_tids.remove(best_ltid)

            # --- 2. Gather ROIs for Batch Inference ---
            valid_targets = []
            roi_batch = []
            roi_meta = []
            
            cfg.curr_high = 0
            cfg.curr_medium = 0
            cfg.curr_low = 0
            
            for bbox, tid in zip(boxes1, ids1):
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
                if tid not in self.track_memory:
                    self.track_memory[tid] = {
                        "door_history": deque(maxlen=5),
                        "alarm_triggered": False,
                        "closed_streak": 0,
                        "illumination": "normal",
                        "age": 0,
                        "speed": 0.0,
                        "last_center": (cx, cy),
                        "last_seen_frame": self.frame_count,
                        "last_bbox": bbox,
                        "last_door_state": "door_closed",
                        "last_door_box": None
                    }
                mem = self.track_memory[tid]
                mem["age"] += 1
                mem["last_seen_frame"] = self.frame_count
                mem["last_bbox"] = bbox
                
                # Speed Gating 平滑測速
                px, py = mem["last_center"]
                speed = np.sqrt((cx - px)**2 + (cy - py)**2)
                mem["speed"] = 0.8 * mem["speed"] + 0.2 * speed
                mem["last_center"] = (cx, cy)
                
                # 決定 Priority
                priority = "LOW"
                if mem["age"] < 5 or mem["speed"] < 3.0:
                    priority = "HIGH"
                    cfg.curr_high += 1
                elif mem["speed"] < cfg.max_speed_px:
                    priority = "MEDIUM"
                    cfg.curr_medium += 1
                else:
                    cfg.curr_low += 1
                
                # ROI 小於閾值直接判定 closed
                if x2 - x1 < cfg.min_roi_px or y2 - y1 < cfg.min_roi_px:
                    mem["last_door_state"] = "door_closed"
                    mem["last_door_box"] = None
                    valid_targets.append((tid, x1, y1, x2, y2, "normal", 0.0, False))
                    continue
                
                # [光照評估與 ROI 補償]
                roi = frame[y1:y2, x1:x2]
                enhanced_roi, illu_status, mean_y = assess_and_enhance_roi(
                    roi,
                    config=cfg,
                    debug_dir=self.debug_illu_dir,
                    frame_id=self.frame_count,
                    track_id=int(tid)
                )
                
                # Priority Scan 決定是否進行 Stage 2 預測
                should_run_s2 = False
                if illu_status != "low_confidence_dark":
                    if priority == "HIGH":
                        should_run_s2 = True
                    elif priority == "MEDIUM" and self.frame_count % cfg.stage2_interval == 0:
                        should_run_s2 = True
                    elif priority == "LOW" and self.frame_count % 5 == 0:
                        should_run_s2 = True
                        
                valid_targets.append((tid, x1, y1, x2, y2, illu_status, mean_y, should_run_s2))
                
                if should_run_s2:
                    roi_batch.append(enhanced_roi)
                    roi_meta.append((tid, x1, y1))

            # --- 3. Stage 2 Batch Inference ---
            tid_to_res2 = {}
            if roi_batch:
                cfg.stat_s2_batches += 1
                cfg.stat_s2_rois += len(roi_batch)
                
                # 傳入 List[np.ndarray]，YOLO 會自動 resize & pad 進行 batch 計算
                res2_list = self.model2.predict(
                    roi_batch,
                    conf=cfg.conf2,   # ✅ 修正：使用 RuntimeConfig 內的動態信心度
                    iou=self.iou2,
                    imgsz=self.imgsz,
                    verbose=False
                )
                for res_idx, res2 in enumerate(res2_list):
                    tid_to_res2[roi_meta[res_idx][0]] = res2
            
            # --- 4. Process Target Results ---
            for targ in valid_targets:
                tid, x1, y1, x2, y2, illu_status, mean_y, did_run_s2 = targ
                mem = self.track_memory[tid]
                door_state = mem["last_door_state"]
                door_box_global = mem["last_door_box"]
                
                if did_run_s2:
                    door_state = "door_closed"
                    door_box_global = None
                    if tid in tid_to_res2:
                        res2 = tid_to_res2[tid]
                        highest_conf = 0.0
                        if res2 and len(res2.boxes) > 0:
                            boxes2 = res2.boxes.xyxy.cpu().numpy().astype(int)
                            confs2 = res2.boxes.conf.cpu().numpy()
                            clss2  = res2.boxes.cls.cpu().numpy().astype(int)
                            names2 = self.model2.names
                            
                            for b2, c2, cls2 in zip(boxes2, confs2, clss2):
                                state_name = names2[cls2]
                                gx1, gy1 = x1 + b2[0], y1 + b2[1]
                                gx2, gy2 = x1 + b2[2], y1 + b2[3]
                                if state_name in ["door_opening", "door_open"] and c2 > highest_conf:
                                    highest_conf = c2
                                    door_state   = state_name
                                    door_box_global = (gx1, gy1, gx2, gy2)
                    
                    mem["last_door_state"] = door_state
                    mem["last_door_box"] = door_box_global

                
                # [Temporal Voting 時間序列防抖]
                mem["illumination"] = illu_status
                mem["door_history"].append(door_state)

                
                # ✅ 修正：alarm 重置機制
                # 若連續 5 幀都是 door_closed，代表這輛車的門已經確實關上，重置允許二次警報
                if door_state == "door_closed":
                    mem["closed_streak"] += 1
                    if mem["closed_streak"] >= 5 and mem["alarm_triggered"]:
                        mem["alarm_triggered"] = False
                        mem["closed_streak"] = 0
                else:
                    mem["closed_streak"] = 0
                
                opening_count = sum(1 for s in mem["door_history"] if s == "door_opening")
                open_count = sum(1 for s in mem["door_history"] if s == "door_open")
                
                current_alarm = False
                event_type = None
                
                if opening_count >= 3:
                    current_alarm = True
                    event_type = "EARLY_WARNING_OPENING"
                elif open_count >= 3:
                    current_alarm = True
                    event_type = "DANGER_DOOR_OPEN"
                    
                if current_alarm and not mem["alarm_triggered"]:
                    mem["alarm_triggered"] = True
                    cfg.stat_alarms += 1
                    evt = {
                        "event": event_type,
                        "track_id": int(tid),
                        "time": float(timestamp),
                        "illumination_status": illu_status,
                        "luma": mean_y
                    }
                    new_events.append(evt)
                    self.events.append(evt)
                    suffix = " 💡(夜間補償介入)" if illu_status == "clahe_enhanced" else f" (光照: {illu_status})"
                    print(f"[{timestamp:.1f}s] 🚨 {event_type} 車輛 ID: {tid}{suffix}")
                
                current_tracks.append({
                    "track_id": int(tid),
                    "bbox": (x1, y1, x2, y2),
                    "door_box": door_box_global,
                    "door_state": "ALARM" if current_alarm else door_state,
                    "illumination": illu_status,
                    "avg_luma": mean_y
                })
                
        # Garbage Collection: 基於 Grace Period 的緩存清理
        for tid in list(self.track_memory.keys()):
            if self.frame_count - self.track_memory[tid].get("last_seen_frame", self.frame_count) > cfg.grace_frames:
                del self.track_memory[tid]

        return current_tracks, new_events



# =====================================================================
# Phase 5: Visualization & Output Sink
# =====================================================================
def get_color(idx: int) -> tuple[int, int, int]:
    np.random.seed(idx)
    return tuple(int(x) for x in np.random.randint(50, 255, 3))

def draw_tracking_and_events(frame: np.ndarray, tracks: list[dict], fps: float, config: RuntimeConfig | None = None) -> None:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = t["track_id"]
        d_state = t["door_state"]
        illu = t["illumination"]
        
        color = get_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        illu_tag = "[CLAHE]" if illu == "clahe_enhanced" else ("[DARK]" if illu == "low_confidence_dark" else "")
        label = f"Car {tid} {illu_tag}"
        cv2.putText(frame, label, (x1 + 2, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if d_state == "ALARM":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "!!! DOORS APART !!!", (x1, max(0, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if t["door_box"] is not None:
                dx1, dy1, dx2, dy2 = t["door_box"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 165, 255), 2)

    # FPS 高對比渲染
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Dashboard 控制面板
    if config is not None and config.show_dashboard:
        _draw_dashboard(frame, fps, config)


def _draw_dashboard(frame: np.ndarray, fps: float, cfg: RuntimeConfig) -> None:
    """
    在左下角繪制半透明控制面板。
    分為三区塊：參數現況 / 統計資訊 / 熱鍵說明。
    """
    H, W = frame.shape[:2]
    PX = 14          # 每列高度
    PAD = 6          # 小內邊距
    COL_W = 310      # 面板寬度

    lines = [
        # ── 區塊 1：參數當前狀態 ──
        ("PARAMS",           (180, 180, 180), True),
        (f"CLAHE   : {'ON  '+chr(8592)+' [c] to OFF' if cfg.enable_clahe else 'OFF '+chr(8592)+' [c] to ON'}",
                             (100, 255, 100) if cfg.enable_clahe else (100, 100, 255), False),
        (f"dark_thr: {cfg.dark_threshold:3d}  (+/-  to adjust)",   (200, 200, 255), False),
        (f"conf2   : {cfg.conf2:.2f} [w/s to adjust]",               (200, 200, 255), False),
        (f"Stage2  : priority ON (scan intvl {cfg.stage2_interval})",                 (200, 200, 255), False),
        (f"min_roi : {cfg.min_roi_px} px",                                          (200, 200, 255), False),
        (f"debug   : {'ON ' if cfg.enable_debug_illu else 'OFF'}  [d] toggle",      (200, 255, 200) if cfg.enable_debug_illu else (150, 150, 150), False),
        # ── 區塊 2：統計 ──
        ("STATS (Current Frame & Totals)",            (180, 180, 180), True),
        (f"Cars: [H:{cfg.curr_high} | M:{cfg.curr_medium} | L:{cfg.curr_low}] / Total: {cfg.stat_vehicles}", (255, 220, 100), False),
        (f"S2 batches: {cfg.stat_s2_batches:,} | S2 ROIs: {cfg.stat_s2_rois:,}",    (255, 220, 100), False),
        (f"CLAHE ct : {cfg.stat_clahe_calls:,}",     (255, 220, 100), False),
        (f"Alarms   : {cfg.stat_alarms}",             (100, 80, 255) if cfg.stat_alarms > 0 else (255, 220, 100), False),
        # ── 區塊 3：熱鍵提示 ──
        ("KEYS",             (180, 180, 180), True),
        ("[c] CLAHE  [d] debug  [w/s] conf2",         (160, 160, 160), False),
        ("[+]/[-] dark_thr  [i] hide [q] quit",       (160, 160, 160), False),
    ]

    box_h = PX * (len(lines) + 1) + PAD * 2
    x0, y0 = 8, H - box_h - 8

    # 半透明黑底
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + COL_W, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # 框線
    cv2.rectangle(frame, (x0, y0), (x0 + COL_W, y0 + box_h), (60, 60, 60), 1)

    ty = y0 + PAD + PX
    for text, color, is_header in lines:
        if is_header:
            cv2.putText(frame, f"▶ {text}", (x0 + PAD, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 100), 1)
        else:
            cv2.putText(frame, text, (x0 + PAD + 6, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1)
        ty += PX



def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


# =====================================================================
# Main Execution Flow
# =====================================================================
def run_stream_inference(
    stage1_model_path: str,
    stage2_model_path: str,
    source: int | str,
    conf: float,
    iou: float,
    imgsz: int,
    show_window: bool,
    output_path: Path | None,
    events_path: Path | None,
    debug_illu_dir: Path | None = None,
    config: RuntimeConfig | None = None
) -> None:
    if config is None:
        config = RuntimeConfig()
    engine = EarlyWarningEngine(
        stage1_model_path, stage2_model_path,
        conf1=0.3, iou2=iou, imgsz=imgsz,
        debug_illu_dir=debug_illu_dir,
        config=config
    )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟輸入來源: {source}")
    
    window_name = "Two-Stage 3M Early Warning (press q to quit)"
    frame_count = 0
    t_start = perf_counter()

    print("\n💡 已正式移除影片 (mp4) 輸出功能以消弭 I/O 延遲，專注於邊緣推論效能。\n")
    print("熱鍵控制: [c]切換CLAHE  [d]切換debug  [w/s]調整conf2  [+/-]調整dark_thr  [q]離開\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = perf_counter() - t_start
        t0 = perf_counter()
        tracks, new_events = engine.process_frame(frame, timestamp)
        fps = 1.0 / max(perf_counter() - t0, 1e-6)
        
        if events_path is not None and new_events:
            for evt in new_events:
                append_jsonl(events_path, evt)

        draw_frame = frame.copy()
        draw_tracking_and_events(draw_frame, tracks, fps, config=config)

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"[{timestamp:.1f}s] 幀: {frame_count} | 車(H:{config.curr_high}|M:{config.curr_medium}|L:{config.curr_low}) | "
                  f"S2 batches: {config.stat_s2_batches} | S2 ROIs: {config.stat_s2_rois} | 警報: {config.stat_alarms}")

        if show_window:
            cv2.imshow(window_name, draw_frame)
            key = cv2.waitKey(1) & 0xFF
            # ====== 熱鍵映射：直接修改 RuntimeConfig ，下一幀即生效 ======
            if   key == ord("q"):  break
            elif key == ord("c"):  config.enable_clahe = not config.enable_clahe;  print(f"CLAHE -> {'ON' if config.enable_clahe else 'OFF'}")
            elif key == ord("d"):  config.enable_debug_illu = not config.enable_debug_illu; print(f"debug_illu -> {'ON' if config.enable_debug_illu else 'OFF'}")
            elif key == ord("i"):  config.show_dashboard = not config.show_dashboard
            elif key == ord("+") or key == ord("="): config.dark_threshold = min(200, config.dark_threshold + 5); print(f"dark_thr -> {config.dark_threshold}")
            elif key == ord("-"):  config.dark_threshold = max(15,  config.dark_threshold - 5); print(f"dark_thr -> {config.dark_threshold}")
            elif key == ord("2"):  config.stage2_interval = (config.stage2_interval % 3) + 1; print(f"Stage2 interval -> {config.stage2_interval}")
            elif key == ord("w"):  config.conf2 = min(1.0, config.conf2 + 0.05); print(f"conf2 -> {config.conf2:.2f}")
            elif key == ord("s"):  config.conf2 = max(0.05, config.conf2 - 0.05); print(f"conf2 -> {config.conf2:.2f}")
        
    cap.release()
    if show_window: cv2.destroyAllWindows()
    print(f"推論結束，共處理: {frame_count} 幀 | 觸發警報: {len(engine.events)} 次。")



def run_image_inference(
    stage1_model_path: str,
    stage2_model_path: str,
    image_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    show_window: bool,
    output_path: Path | None,
    events_path: Path | None,
    debug_illu_dir: Path | None = None,
    config: RuntimeConfig | None = None
) -> None:
    if config is None:
        config = RuntimeConfig()
    engine = EarlyWarningEngine(
        stage1_model_path, stage2_model_path,
        conf1=0.3, iou2=iou, imgsz=imgsz,
        debug_illu_dir=debug_illu_dir,
        config=config
    )
    
    img = cv2.imread(str(image_path))
    if img is None: raise RuntimeError(f"無法讀取圖片: {image_path}")

    tracks, new_events = engine.process_frame(img, 0.0)
    
    if events_path is not None and new_events:
        for evt in new_events:
            append_jsonl(events_path, evt)
            
    draw_frame = img.copy()
    draw_tracking_and_events(draw_frame, tracks, fps=0.0, config=config)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), draw_frame)
        print(f"圖片檢測完稿輸出至: {output_path}")

    if show_window:
        cv2.imshow("Detection", draw_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def default_output_path(source: int | str, suffix: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"camera_{stamp}{suffix}" if isinstance(source, int) else f"{Path(source).stem}_pred_{stamp}{suffix}"
    return DEFAULT_OUTPUT_DIR / name

def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source

def resolve_display_mode(mode: str) -> bool:
    if mode == "always": return True
    if mode == "never": return False
    try:
        cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test_window")
        return True
    except cv2.error:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Stage Vision Event Pipeline with Auto-Illumination (Plan C)")
    parser.add_argument("--model1", type=str, default=str(DEFAULT_STAGE1_MODEL), help="Stage 1 通用追蹤模型 (預設找出車身)")
    parser.add_argument("--model2", type=str, default=str(DEFAULT_STAGE2_MODEL), help="Stage 2 開門預警模型")
    parser.add_argument("--source", type=str, default="0", help="鏡頭代號或影片路徑")
    parser.add_argument("--conf", type=float, default=0.20, help="Stage 2 開門判定信心度")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="推論輸入尺寸")
    parser.add_argument("--display", type=str, default="always", choices=["auto", "always", "never"],
                        help="視窗顯示模式: always(預設顯示)/auto(自動偵測)/never(不顯示)")
    parser.add_argument("--output", type=str, default="", help="儲存視覺化的路徑")
    parser.add_argument("--events", type=str, default="", help="JSONL 事件日誌檔")
    parser.add_argument(
        "--debug-illu", action="store_true",
        help="啟用光照比對分析模式：每個偵測到的車輛 ROI 都會儲存『原圖|補償後|差異圖』三排式比對影像至 outputs/debug_illu/ 供人工驗證"
    )
    args = parser.parse_args()

    s1_path = Path(args.model1)
    if args.model1 != str(DEFAULT_STAGE1_MODEL) and not s1_path.exists():
        print(f"Warning: {s1_path} not found. Will let framework try to load it via internet.")
        
    s2_path = Path(args.model2)
    if not s2_path.exists():
        raise FileNotFoundError(f"找不到核心門特徵模型檔: {s2_path}")

    source = parse_source(args.source)
    show_window = resolve_display_mode(args.display)
    if args.display == "auto" and not show_window:
        print("偵測到目前 OpenCV 不支援 imshow，已自動切換成無視窗模式，輸出將儲存至檔案。")

    explicit_output = Path(args.output) if args.output else None
    explicit_events = Path(args.events) if args.events else None
    # ✅ debug_illu：啟用時把比對圖存入 outputs/debug_illu/
    debug_illu_dir = (DEFAULT_OUTPUT_DIR / "debug_illu") if args.debug_illu else None
    if debug_illu_dir:
        print(f"💡 光照比對模式已啟用，比對圖將儲存於: {debug_illu_dir}")

    # 初始化 RuntimeConfig — argparse 給初始值，執行中可由熱鍵即時改變
    cfg = RuntimeConfig(
        conf2=args.conf,
        enable_clahe=True,
        dark_threshold=85,
        stage2_interval=1,
        min_roi_px=40,
        max_speed_px=15.0,
        grace_frames=20,
        show_dashboard=True,
        enable_debug_illu=(debug_illu_dir is not None),
    )
    if not show_window:
        print("💡 由於 OpenCV imshow 停用且影片寫出器已被拔除，事件紀錄請前往 _events.jsonl 查看。")

    # Handle image mode vs stream mode
    if isinstance(source, str):
        src_path = Path(source)
        if src_path.exists() and src_path.suffix.lower() in IMAGE_EXTS:
            out_img = explicit_output or default_output_path(source, ".jpg")
            out_evt = explicit_events or default_output_path(source, "_events.jsonl")
            run_image_inference(args.model1, args.model2, src_path, args.conf, args.iou, args.imgsz, show_window, out_img, out_evt, debug_illu_dir, cfg)
            return

    # Stream 模式不再輸出 mp4 影片，只輸出事件
    out_evt = explicit_events or default_output_path(source, "_events.jsonl")
    run_stream_inference(args.model1, args.model2, source, args.conf, args.iou, args.imgsz, show_window, None, out_evt, debug_illu_dir, cfg)


if __name__ == "__main__":
    main()
