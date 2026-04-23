import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

# ==============================================================
# Configuration
# ==============================================================
# 本次驗證基底為 0422_1402 模型
MODEL_PATH = r"C:\antigravity\storage\artifacts\experiments\specialized\exp_specialized_auto_iter_videos_0422_1435\weights\best.pt"
VAL_IMAGES_DIR = Path(r"C:\antigravity\storage\assets\validation\frozen_v1\images")
VAL_LABELS_DIR = Path(r"C:\antigravity\storage\assets\validation\frozen_v1\labels")

OUTPUT_DIR = Path(r"C:\antigravity\storage\artifacts\evaluations\fn_mining_v2")
OUTPUT_JSON = OUTPUT_DIR / "fn_classification.json"

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def yolo2xyxy(x, y, w, h, img_w, img_h):
    x1, y1 = (x - w/2) * img_w, (y - h/2) * img_h
    x2, y2 = (x + w/2) * img_w, (y + h/2) * img_h
    return [x1, y1, x2, y2]

def main():
    print(f"🚀 啟動盲區細粒度分類探勘 (FN Classification v2) ...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 建立各 Type 的子資料夾存放圖片
    for type_name in ["Type_A", "Type_B", "Type_C", "Type_D"]:
        (OUTPUT_DIR / type_name).mkdir(exist_ok=True)

    model = YOLO(MODEL_PATH)
    
    results_data = {
        "Type_A": [], # 完全無框
        "Type_B": [], # 定位差
        "Type_C": [], # 分類錯
        "Type_D": [], # Conf抑制
    }
    
    stats = {"TP": 0, "Type_A": 0, "Type_B": 0, "Type_C": 0, "Type_D": 0}
    
    for img_path in VAL_IMAGES_DIR.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']: continue
        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists(): continue
            
        # 1. 執行推論 (conf=0.01 探底)
        results = model.predict(source=str(img_path), conf=0.01, verbose=False)
        pred = results[0]
        img_h, img_w = pred.orig_shape
        
        # 整理所有預測框
        pred_boxes = []
        for i, cls_id in enumerate(pred.boxes.cls):
            pred_boxes.append({
                "xyxy": pred.boxes.xyxy[i].cpu().numpy(),
                "cls": int(cls_id),
                "conf": float(pred.boxes.conf[i])
            })
            
        # 2. 讀取 GT 框並逐一比對
        with open(label_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0: # 只看 GT Open
                    gt_box = yolo2xyxy(*map(float, parts[1:5]), img_w, img_h)
                    
                    max_open_iou = 0.0
                    max_open_conf = 0.0
                    max_any_iou = 0.0
                    max_any_cls = -1
                    
                    for p in pred_boxes:
                        iou = box_iou(gt_box, p["xyxy"])
                        
                        if iou > max_any_iou:
                            max_any_iou = iou
                            max_any_cls = p["cls"]
                            
                        if p["cls"] == 0: # 如果是 Open
                            if iou > max_open_iou:
                                max_open_iou = iou
                                max_open_conf = p["conf"]

                    # ==========================================
                    # 決策樹 (Decision Tree for FN types)
                    # ==========================================
                    status = ""
                    
                    if max_open_iou >= 0.45:
                        if max_open_conf >= 0.25:
                            status = "TP"
                        else:
                            status = "Type_D" # 有框、準確、但被 Conf 吃掉
                    else:
                        # 最高 Open IoU < 0.45
                        if max_any_iou >= 0.45:
                            if max_any_cls == 1:
                                status = "Type_C" # 被分類成 Close
                            else:
                                status = "Type_C" # 其他分類錯誤 (防呆)
                        elif max_any_iou > 0.1:
                            status = "Type_B" # 有框，但 IoU 介於 0.1 ~ 0.45 之間 (定位不準)
                        else:
                            status = "Type_A" # 完全無框 (IoU < 0.1)
                            
                    stats[status] += 1
                    
                    if status != "TP":
                        record = {
                            "image_name": img_path.name,
                            "gt_idx": line_idx + 1,
                            "max_open_iou": round(float(max_open_iou), 4),
                            "max_open_conf": round(float(max_open_conf), 4),
                            "max_any_iou": round(float(max_any_iou), 4),
                            "max_any_cls": int(max_any_cls)
                        }
                        results_data[status].append(record)
                        
                        # 複製圖片到對應資料夾
                        dest_img = OUTPUT_DIR / status / f"{status}_{img_path.name}"
                        if not dest_img.exists():
                            shutil.copy2(img_path, dest_img)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)

    print("\n" + "="*50)
    print(f"🎯 盲區分類挖掘完成！")
    print(f"總 GT Open 實體數: {sum(stats.values())}")
    print(f"✅ True Positive (正常抓到): {stats['TP']}")
    print(f"❌ Type A (完全無框 - Backbone問題): {stats['Type_A']}")
    print(f"❌ Type B (定位不準 - Regression問題): {stats['Type_B']}")
    print(f"❌ Type C (被判成Close - Classification問題): {stats['Type_C']}")
    print(f"❌ Type D (Conf太低 - Threshold問題): {stats['Type_D']}")
    print(f"名單已儲存至: {OUTPUT_JSON}")
    print("="*50)

if __name__ == '__main__':
    main()
