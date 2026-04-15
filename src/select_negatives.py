import os
import argparse
import shutil
from tqdm import tqdm
from ultralytics import YOLO

def select_negatives(model_path, img_dir, output_dir, conf_threshold=0.1):
    """
    使用 YOLO 模型篩選沒有偵測到任何目標的圖片，並為其生成空標註檔 (.txt)。
    
    Args:
        model_path (str): 模型路徑 (建議使用 global_best.pt)
        img_dir (str): 待篩選的圖片資料夾
        output_dir (str): 輸出的標註檔資料夾 (產生的 .txt 會放在這)
        conf_threshold (float): 偵測門檻。設低一點 (如 0.1) 可以確保負樣本更乾淨。
    """
    # 1. 載入標準模型 (COCO 80 類)
    # 如果本地沒有 yolov8n.pt，它會自動從官網下載
    model = YOLO(model_path)
    
    # COCO 資料集中與「交通工具」相關的類別 ID:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    vehicle_classes = [2, 5, 7]
    
    # 2. 準備輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 獲取圖片列表
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"[*] 開始處理 {len(images)} 張圖片...")
    print(f"[*] 使用官網預訓練模型: {model_path}")
    print(f"[*] 篩選目標：排除包含 Car, Truck, Bus, Motorcycle 的圖片")
    
    negative_count = 0
    
    # 4. 批次推理
    for img_name in tqdm(images):
        img_path = os.path.join(img_dir, img_name)
        
        # 執行推理
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # 檢查是否有偵測到任何交通工具
        found_vehicle = False
        if len(results[0].boxes) > 0:
            # 取得所有偵測到的類別 ID
            detected_classes = results[0].boxes.cls.cpu().tolist()
            # 如果偵測到的類別中有任何一個屬於交通工具
            if any(cls_id in vehicle_classes for cls_id in detected_classes):
                found_vehicle = True
        
        # 如果「沒有」偵測到車輛，則判定為負樣本
        if not found_vehicle:
            # A. 建立同名的空 .txt 檔案
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_name)
            
            with open(txt_path, 'w') as f:
                pass  # 檔案留空
            
            # B. 移動圖片到目標資料夾
            dest_img_path = os.path.join(output_dir, img_name)
            try:
                shutil.move(img_path, dest_img_path)
                negative_count += 1
            except Exception as e:
                print(f"移動檔案 {img_name} 時出錯: {e}")
            
    print(f"\n[+] 處理完成！")
    print(f"[+] 在 {len(images)} 張圖中，共搬移了 {negative_count} 張「無車輛」的負樣本圖片。")
    print(f"[+] 目的地 (圖片與空 .txt): {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 負樣本(背景)自動篩選工具 - 使用 COCO 模型")
    
    # 使用官網預訓練模型
    default_model = "yolov8n.pt"
    
    parser.add_argument("--model", type=str, default=default_model, help="模型路徑 (預設 yolov8n.pt)")
    parser.add_argument("--input", type=str, default="C:/antigravity/data/1_raw", help="輸入圖片資料夾路徑")
    parser.add_argument("--output", type=str, default="C:/antigravity/data/1_raw/labels", help="輸出空標註檔的資料夾路徑")
    parser.add_argument("--conf", type=float, default=0.25, help="判定為有車的信心門檻 (預設 0.25)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"錯誤：找不到輸入資料夾 {args.input}")
    else:
        select_negatives(args.model, args.input, args.output, args.conf)
