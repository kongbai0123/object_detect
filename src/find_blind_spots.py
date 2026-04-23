import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

# ==============================================================
# Configuration
# ==============================================================
MODEL_PATH = r"C:\antigravity\storage\artifacts\experiments\specialized\exp_specialized_auto_iter_videos_0422_1402\weights\best.pt"
VAL_IMAGES_DIR = Path(r"C:\antigravity\storage\assets\validation\frozen_v1\images")
VAL_LABELS_DIR = Path(r"C:\antigravity\storage\assets\validation\frozen_v1\labels")
OUTPUT_DIR = Path(r"C:\antigravity\storage\artifacts\evaluations\fn_mining")
OUTPUT_JSON = OUTPUT_DIR / "strict_missed_open_list.json"
OUTPUT_IMG_DIR = OUTPUT_DIR / "strict_missed_images"

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    # intersection
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    # union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def yolo2xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]

def main():
    print(f"🚀 啟動嚴格版盲區挖掘 (Strict IoU FN Mining) ...")
    if not os.path.exists(MODEL_PATH): return print("❌ 找不到模型")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    strict_missed = []
    
    total_gt_open = 0
    total_missed_open = 0
    
    for img_path in VAL_IMAGES_DIR.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']: continue
        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists(): continue
            
        # 1. 讀取預測結果 (獲取原始圖片大小)
        results = model.predict(source=str(img_path), conf=0.01, verbose=False)
        pred = results[0]
        img_h, img_w = pred.orig_shape
        
        # 解析預測框 (x1, y1, x2, y2)
        pred_open_boxes = []
        for i, cls in enumerate(pred.boxes.cls):
            if int(cls) == 0:  # 0 is Open
                pred_open_boxes.append(pred.boxes.xyxy[i].cpu().numpy())
                
        # 2. 讀取 GT 框並逐一比對
        gt_opens_in_img = 0
        img_missed_count = 0
        
        with open(label_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    total_gt_open += 1
                    gt_opens_in_img += 1
                    
                    gt_x, gt_y, gt_w, gt_h = map(float, parts[1:5])
                    gt_box = yolo2xyxy(gt_x, gt_y, gt_w, gt_h, img_w, img_h)
                    
                    # 檢查是否有任何一個 Open 預測框的 IoU > 0.45
                    is_matched = False
                    for p_box in pred_open_boxes:
                        iou = box_iou(gt_box, p_box)
                        if iou > 0.45: # 稍微放寬一點容忍度
                            is_matched = True
                            break
                            
                    if not is_matched:
                        img_missed_count += 1
                        total_missed_open += 1
                        print(f"⚠️ 漏抓 GT (IoU < 0.45): {img_path.name} (第 {line_idx+1} 個框)")
                        
        if img_missed_count > 0:
            strict_missed.append({
                "image_name": img_path.name,
                "missed_count": img_missed_count,
                "total_gt_in_img": gt_opens_in_img
            })
            dest_img = OUTPUT_IMG_DIR / f"StrictFN_{img_path.name}"
            shutil.copy2(img_path, dest_img)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(strict_missed, f, indent=4)

    print("\n" + "="*50)
    print(f"🎯 嚴格盲區挖掘完成！")
    print(f"資料集中總共的 GT Open 實體數量: {total_gt_open}")
    print(f"真實漏抓 (Strict FN) 數量: {total_missed_open}")
    print(f"牽涉的圖片數量: {len(strict_missed)}")
    print(f"名單已儲存至: {OUTPUT_JSON}")
    print("="*50)

if __name__ == '__main__':
    main()
