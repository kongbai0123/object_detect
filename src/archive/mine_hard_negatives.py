import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def mine_hard_negatives(
    model_path='weight/best.pt',
    input_dir='data/1_raw/door_opening_frames',
    output_base='data/2_filtered/hard_negatives',
    conf_close_high=0.35,
    conf_open_high=0.25,
    conf_uncertain=(0.15, 0.35),
    imgsz=768,
    max_samples=500
):
    """
    分層困難樣本挖掘 (Stratified Hard Negative Mining)
    邏輯依據：用戶專家回饋 v0.6 版決策
    """
    # 建立目錄架構
    buckets = ['fp_close_high', 'fp_open_high', 'uncertain_review']
    for b in buckets:
        os.makedirs(os.path.join(output_base, b), exist_ok=True)
        os.makedirs(os.path.join(output_base, b, 'overlay'), exist_ok=True)

    print(f"🚀 啟動獵鬼行動... 使用模型: {model_path}")
    model = YOLO(model_path)
    
    # 取得輸入清單
    image_files = list(Path(input_dir).rglob('*.jpg')) + list(Path(input_dir).rglob('*.png'))
    print(f"📦 掃描到 {len(image_files)} 張原始影像，預計挖掘上限 {max_samples} 張...")

    count_map = {b: 0 for b in buckets}
    
    for img_path in tqdm(image_files):
        if sum(count_map.values()) >= max_samples:
            break
            
        results = model.predict(str(img_path), conf=0.15, imgsz=imgsz, verbose=False)[0]
        
        if len(results.boxes) == 0:
            continue
            
        # 取得最高信心度的框
        best_box = results.boxes[results.boxes.conf.argmax()]
        max_conf = float(best_box.conf[0])
        cls_idx = int(best_box.cls[0])
        cls_name = results.names[cls_idx]
        
        target_bucket = None
        
        # --- 分層邏輯 ---
        
        # 1. fp_close_high (最重災區)
        if cls_name == 'close' and max_conf >= conf_close_high:
            target_bucket = 'fp_close_high'
            
        # 2. fp_open_high
        elif cls_name == 'open' and max_conf >= conf_open_high:
            target_bucket = 'fp_open_high'
            
        # 3. uncertain_review (信心度模糊地帶)
        elif conf_uncertain[0] <= max_conf < conf_uncertain[1]:
            target_bucket = 'uncertain_review'
            
        # --- 執行存檔 ---
        if target_bucket and count_map[target_bucket] < 200:
            count_map[target_bucket] += 1
            filename = img_path.name
            
            # 存原圖
            shutil.copy2(img_path, os.path.join(output_base, target_bucket, filename))
            
            # 存可視化圖 (Overlay) 供人工審查
            res_plotted = results.plot()
            cv2.imwrite(os.path.join(output_base, target_bucket, 'overlay', filename), res_plotted)

    print("\n" + "="*30)
    print("📈 挖掘成果摘要 (Phase A):")
    for b, count in count_map.items():
        print(f" - {b}: {count} 張")
    print("="*30)
    print(f"請至 {output_base} 進行人工審查，確認為純背景後再回流至 3_processed。")

if __name__ == '__main__':
    mine_hard_negatives()
