import os
import shutil
import subprocess
import sys
from pathlib import Path
import argparse
import random
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
    import albumentations as A
    from tqdm import tqdm
except ImportError:
    print("尚缺 albumentations 套件，正在自動補齊套件庫...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations", "tqdm", "--user"])
    import cv2
    import albumentations as A
    from tqdm import tqdm

def load_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    bboxes.append(bbox)
                    class_labels.append(cls_id)
    return bboxes, class_labels

def save_yolo_labels(output_path, bboxes, class_labels):
    with open(output_path, "w", encoding="utf-8") as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            cx = max(0.000001, min(0.999999, bbox[0]))
            cy = max(0.000001, min(0.999999, bbox[1]))
            w = max(0.000001, min(0.999999, bbox[2]))
            h = max(0.000001, min(0.999999, bbox[3]))
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def get_profiles():
    """
    實作 0.7.1 專家規範：條件式分級增強模式
    """
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=32, min_visibility=0.3)
    
    # --- Profile 1: Open / Robust (補強遠距、夜間、雨天，保邊界) ---
    open_pipes = {
        "far_scale": A.Compose([
            A.Affine(scale=(0.3, 0.7), rotate=0, shear=0, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.4), # 模擬行車記錄器抖動
            A.ImageCompression(quality_lower=60, quality_upper=90, p=0.4)
        ], bbox_params=bbox_params),
        "night": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.2), contrast_limit=(0.1, 0.4), p=1.0),
            A.CLAHE(clip_limit=2.0, p=0.4)
        ], bbox_params=bbox_params),
        "weather": A.Compose([
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.4),
            A.ImageCompression(p=0.5)
        ], bbox_params=bbox_params),
        "geo_light": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=3, p=1.0)
        ], bbox_params=bbox_params)
    }

    # --- Profile 2: Close / Conservative (防禦性控制，避免背景同化) ---
    close_pipes = {
        "light_shift": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(sat_shift_limit=20, val_shift_limit=20, p=0.5)
        ], bbox_params=bbox_params),
        "blur_comp": A.Compose([
            A.MotionBlur(blur_limit=(3, 5), p=0.5), # 調降強度以保留 Close 邊界
            A.ImageCompression(quality_lower=70, quality_upper=90, p=0.5)
        ], bbox_params=bbox_params)
    }

    # --- Profile 3: Background / Hygiene (純背景/鬼圖，保持真實拒判) ---
    bg_pipes = {
        "bg_noise": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ISONoise(intensity=(0.1, 0.2), p=0.4)
        ], bbox_params=bbox_params)
    }

    return {"open": open_pipes, "close": close_pipes, "bg": bg_pipes}

def run_augmentation(input_dir, output_dir, multiplier=3):
    input_path = Path(input_dir)
    input_imgs = input_path / "images" if (input_path / "images").exists() else input_path
    input_lbls = input_path / "labels" if (input_path / "labels").exists() else input_path
    
    out_imgs = Path(output_dir) / "images"
    out_lbls = Path(output_dir) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    
    profiles = get_profiles()
    img_files = list(input_imgs.glob("*.jpg")) + list(input_imgs.glob("*.png"))
    
    print(f"🚀 [0.7.1 Class-Aware Aug] 啟動條件式增強系統...")
    
    total_generated = 0
    
    for img_path in tqdm(img_files, desc="[分類增強]"):
        lbl_path = input_lbls / f"{img_path.stem}.txt"
        bboxes, labels = load_yolo_labels(lbl_path)
        
        # 1. 衛生修補：無論如何先複製原圖與標籤
        dest_img = out_imgs / img_path.name
        dest_lbl = out_lbls / lbl_path.name
        shutil.copy2(img_path, dest_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dest_lbl)
        else:
            with open(dest_lbl, 'w') as f: pass

        # 2. 類別判定 (Decision Engine)
        if not bboxes or not labels:
            profile_name = "bg"
        elif 0 in labels: # 含有 Open (Class 0)
            profile_name = "open"
        else: # 僅有 Close
            profile_name = "close"
            
        active_pipes = profiles[profile_name]
        pipe_names = list(active_pipes.keys())
        
        # 3. 執行增強
        image = cv2.imread(str(img_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 對每張圖根據 multiplier 進行膨脹
        for i in range(multiplier - 1):
            p_name = random.choice(pipe_names)
            transform = active_pipes[p_name]
            
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                trans_img = transformed['image']
                trans_bboxes = transformed['bboxes']
                trans_labels = transformed['class_labels']
                
                # 背景圖即使沒有 bbox 也可以存，但有標籤的圖若 bbox 被切光則跳過
                if bboxes and not trans_bboxes:
                    continue
                    
                aug_suffix = f"{img_path.stem}_{profile_name}_aug_{p_name}_{i}"
                new_img_path = out_imgs / f"{aug_suffix}.jpg"
                new_lbl_path = out_lbls / f"{aug_suffix}.txt"
                
                cv2.imwrite(str(new_img_path), cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR))
                save_yolo_labels(new_lbl_path, trans_bboxes, trans_labels)
                total_generated += 1
            except:
                continue

    print(f"\n✅ 條件式增強完成！共生成 {total_generated} 張變異樣本。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ROOT / "data/6_augmented/train_src"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data/6_augmented/train"))
    parser.add_argument("--multiplier", type=int, default=3)
    args = parser.parse_args()
    
    run_augmentation(args.input, args.output, args.multiplier)
    print_pipeline_notice(
        output_paths=args.output,
        next_script="src/train.py",
        notes=["已依照 Open/Close/BG 實施分級增強。", "下一步請執行 train.py --incremental。"]
    )
