import os
import shutil
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

def run_categorized_mining(
    model_path=str(ROOT / "data/7_experiments/weight/latest_best.pt"),
    input_dir=str(ROOT / "data/1_raw"),
    output_base=str(ROOT / "data/2_filtered/open_mining"),
    target_count=500,
    min_conf=0.1,
    sample_every=15
):
    print(f"🚀 [0.7.2 Mining] 啟動分類採礦...")
    print(f"  -> 使用模型: {model_path}")
    print(f"  -> 掃描目錄: {input_dir}")
    
    if not Path(model_path).exists():
        model_path = str(ROOT / "data/7_experiments/weight/global_best.pt")
        if not Path(model_path).exists():
            print("❌ 找不到可用模型權重")
            return

    model = YOLO(model_path)
    buckets = {
        "interactive_open": Path(output_base) / "interactive_open",
        "uncertain_review": Path(output_base) / "uncertain_review",
        "slight_open": Path(output_base) / "slight_open",
        "edge_visible": Path(output_base) / "edge_visible" 
    }
    for p in buckets.values():
        p.mkdir(parents=True, exist_ok=True)

    # 支援圖片與影片
    media_files = list(Path(input_dir).rglob('*.jpg')) + \
                  list(Path(input_dir).rglob('*.png')) + \
                  list(Path(input_dir).rglob('*.mp4')) + \
                  list(Path(input_dir).rglob('*.avi'))
    
    current_count = 0
    processed_files = 0
    
    existing_imgs = set()
    if (ROOT / "data/3_processed/images").exists():
        existing_imgs = {p.name for p in (ROOT / "data/3_processed/images").glob('*')}

    for media_path in media_files:
        if current_count >= target_count:
            break
            
        processed_files += 1
        print(f"  -> 正在處理: {media_path.name}")

        if media_path.suffix.lower() in ['.mp4', '.avi']:
            cap = cv2.VideoCapture(str(media_path))
            frame_idx = 0
            while cap.isOpened() and current_count < target_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                if frame_idx % sample_every != 0:
                    continue
                
                # 預測
                results = model.predict(frame, conf=min_conf, imgsz=768, verbose=False)[0]
                open_boxes = [box for box in results.boxes if results.names[int(box.cls[0])] == 'open']
                
                if open_boxes:
                    max_open_conf = max([float(box.conf[0]) for box in open_boxes])
                    
                    # 分桶
                    if max_open_conf >= 0.6:
                        target_bucket = buckets["interactive_open"]
                        prefix = "high"
                    elif 0.3 < max_open_conf < 0.6:
                        target_bucket = buckets["uncertain_review"]
                        prefix = "mid"
                    elif 0.1 < max_open_conf <= 0.3:
                        target_bucket = buckets["slight_open"]
                        prefix = "low"
                    else:
                        target_bucket = buckets["edge_visible"]
                        prefix = "edge"

                    new_name = f"m072_{prefix}_{media_path.stem}_f{frame_idx}.jpg"
                    cv2.imwrite(str(target_bucket / new_name), frame)
                    
                    # 預覽圖
                    vis_img = results.plot()
                    cv2.imwrite(str(target_bucket / f"vis_{new_name}"), vis_img)
                    
                    # 標籤 (yolo 格式)
                    labels = []
                    for box in open_boxes:
                        b = box.xywhn[0].tolist()
                        labels.append(f"{int(box.cls[0])} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                    with open(target_bucket / f"{Path(new_name).stem}.txt", 'w') as f:
                        f.write("\n".join(labels))

                    current_count += 1
                    if current_count % 20 == 0:
                        print(f"    [Progress] 已挖掘 {current_count} 個 Open 樣本...")
            cap.release()
        else:
            # 圖片處理
            if any(media_path.name in s for s in existing_imgs):
                continue
            
            results = model.predict(str(media_path), conf=min_conf, imgsz=768, verbose=False)[0]
            open_boxes = [box for box in results.boxes if results.names[int(box.cls[0])] == 'open']
            
            if open_boxes:
                # ... 圖片分桶邏輯 (略，與影片相似) ...
                # 為了節省篇幅與一致性，圖片也走一樣的分流
                max_open_conf = max([float(box.conf[0]) for box in open_boxes])
                if max_open_conf >= 0.6: target_bucket = buckets["interactive_open"]; prefix="high"
                elif 0.3 < max_open_conf < 0.6: target_bucket = buckets["uncertain_review"]; prefix="mid"
                else: target_bucket = buckets["slight_open"]; prefix="low"
                
                new_name = f"m072_{prefix}_{media_path.name}"
                shutil.copy2(media_path, target_bucket / new_name)
                vis_img = results.plot()
                cv2.imwrite(str(target_bucket / f"vis_{new_name}"), vis_img)
                
                labels = [f"{int(box.cls[0])} {box.xywhn[0][0]:.6f} {box.xywhn[0][1]:.6f} {box.xywhn[0][2]:.6f} {box.xywhn[0][3]:.6f}" for box in open_boxes]
                with open(target_bucket / f"{Path(new_name).stem}.txt", 'w') as f:
                    f.write("\n".join(labels))
                current_count += 1

    print(f"✅ 挖掘任務結束。共挖掘: {current_count} 張候選圖。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--input", type=str, default=str(ROOT / "data/1_raw"))
    parser.add_argument("--model", type=str, default=str(ROOT / "data/7_experiments/weight/latest_best.pt"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data/2_filtered/open_mining"))
    parser.add_argument("--min_conf", type=float, default=0.1)
    parser.add_argument("--sample-every", type=int, default=15)
    args = parser.parse_args()
    
    run_categorized_mining(
        model_path=args.model,
        input_dir=args.input,
        output_base=args.output,
        target_count=args.count,
        min_conf=args.min_conf,
        sample_every=args.sample_every
    )
