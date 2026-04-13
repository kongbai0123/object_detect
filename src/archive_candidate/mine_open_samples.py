import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def boost_open_category(
    model_path='weight/best.pt',
    input_dir='data/1_raw',
    out_img_dir='data/3_processed/images',
    out_lbl_dir='data/3_processed/labels',
    target_count=350,
    conf_thresh=0.45
):
    print(f"🚀 [Phase D] 補強 Open 類別，目標新增 {target_count} 張...")
    model = YOLO(model_path)
    image_files = list(Path(input_dir).rglob('*.jpg')) + list(Path(input_dir).rglob('*.png'))
    
    current_count = 0
    for img_path in image_files:
        if current_count >= target_count:
            break
        
        # 避免重複抽樣
        if (Path(out_img_dir) / img_path.name).exists() or \
           (Path(out_img_dir) / f"miner_open_{img_path.name}").exists() or \
           (Path(out_img_dir) / f"boost_open_{img_path.name}").exists():
            continue
            
        results = model.predict(str(img_path), conf=conf_thresh, imgsz=768, verbose=False)[0]
        has_open = False
        labels = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            if cls_name == 'open':
                b = box.xywhn[0].tolist()
                w, h = b[2], b[3]
                
                # Distance-Aware Filter: Ignore objects smaller than 0.5% of the frame
                if (w * h) < 0.005:
                    continue
                    
                has_open = True
                labels.append(f"{int(box.cls[0])} {b[0]:.6f} {b[1]:.6f} {w:.6f} {h:.6f}")
                
        if has_open:
            new_name = f"boost_open_{img_path.name}"
            shutil.copy2(img_path, Path(out_img_dir) / new_name)
            with open(Path(out_lbl_dir) / f"{Path(new_name).stem}.txt", 'w') as f:
                f.write("\n".join(labels))
            current_count += 1
            if current_count % 50 == 0:
                print(f"  -> 已採集 {current_count} 張 open...")
                
    print(f"✅ 補強完成！成功發掘 {current_count} 張新的 Open 金標。")

if __name__ == '__main__':
    boost_open_category()
