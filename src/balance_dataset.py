import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
from pipeline_notice import print_pipeline_notice
import argparse

ROOT = Path(__file__).resolve().parent.parent

def extract_scene_key(filename):
    name = Path(filename).stem
    m1 = re.match(r"(.*?)-[a-zA-Z0-9]+$", name)
    if m1: return m1.group(1)
    m2 = re.match(r"(.*?)_\d+$", name)
    if m2: return m2.group(1)
    return name

def analyze_image(lbl_file):
    has_open = False
    has_close = False
    open_boxes = 0
    close_boxes = 0

    if lbl_file.exists():
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == '0':
                    has_open = True
                    open_boxes += 1
                elif parts[0] == '1':
                    has_close = True
                    close_boxes += 1
                    
    return has_open, has_close, open_boxes, close_boxes

def balance_dataset(input_dir, output_dir, target_ratio=2.0):
    in_img_dir = Path(input_dir) / "images"
    in_lbl_dir = Path(input_dir) / "labels"
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(in_img_dir.glob("*.jpg")) + list(in_img_dir.glob("*.png"))
    
    kept_images = []
    
    # Categorize
    hard_negatives = []
    open_positives = []
    pure_close_groups = defaultdict(list)
    
    total_open_boxes_before = 0
    total_close_boxes_before = 0
    
    for img in images:
        lbl = in_lbl_dir / f"{img.stem}.txt"
        has_open, has_close, o_cnt, c_cnt = analyze_image(lbl)
        total_open_boxes_before += o_cnt
        total_close_boxes_before += c_cnt
        
        if not has_open and not has_close:
            hard_negatives.append(img)
            kept_images.append(img)
        elif has_open:
            open_positives.append(img)
            kept_images.append(img)
        else: # pure close
            key = extract_scene_key(img.name)
            pure_close_groups[key].append(img)
            
    print(f" 分析完畢！")
    print(f"原始狀態 -> Open Box: {total_open_boxes_before}, Close Box: {total_close_boxes_before}")
    print(f"Hard negative 圖: {len(hard_negatives)} (全保留)")
    print(f"含 Open 的圖: {len(open_positives)} (全保留)")
    print(f"純 Close 的 Scene Groups: {len(pure_close_groups)}")
    
    # 目標 close 箱數
    target_close_boxes = int(total_open_boxes_before * target_ratio)
    
    current_close_boxes = 0
    # 我們目前 kept 裡面，只有 open_positives 可能含有 close_box
    for img in open_positives:
        lbl = in_lbl_dir / f"{img.stem}.txt"
        _, _, _, c_cnt = analyze_image(lbl)
        current_close_boxes += c_cnt
        
    allowed_pure_close_boxes = max(0, target_close_boxes - current_close_boxes)
    print(f"目標將總 Close 控制在 {target_close_boxes} (尚需 {allowed_pure_close_boxes} 箱來自純 close 圖)")
    
    # 執行 Scene-aware Sub-sampling
    # 對所有 pure close 群集進行 stride 下採樣，直到我們滿足要求
    dropped_close_imgs = 0
    for key, group in pure_close_groups.items():
        # 對群內的圖片名稱排序，確保是時間連續幀
        group.sort(key=lambda x: x.name)
        
        # 決定這個 group 的 stride
        # 如果組內只有不到 3 張，全留
        if len(group) <= 2:
            kept_images.extend(group)
            continue
            
        # 否則我們至少 stride=3 (保留 1/3)
        stride = 3
        # 從中間抽
        sampled_group = group[::stride]
        kept_images.extend(sampled_group)
        dropped_close_imgs += (len(group) - len(sampled_group))
        
    # 計算平衡後的總箱數
    final_open = 0
    final_close = 0
    for img in kept_images:
        lbl = in_lbl_dir / f"{img.stem}.txt"
        _, _, o_cnt, c_cnt = analyze_image(lbl)
        final_open += o_cnt
        final_close += c_cnt
        
        # Copy to output
        shutil.copy2(img, out_img_dir / img.name)
        if lbl.exists():
            shutil.copy2(lbl, out_lbl_dir / lbl.name)

    print("\n✅ 平衡完成！")
    print(f"丟棄了 {dropped_close_imgs} 張高度重複的純 Close 背景圖")
    print(f"最終比例 -> Open Box: {final_open}, Close Box: {final_close} (比例 1 : {final_close/final_open if final_open else 0:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ROOT / "data/6_augmented/train_src"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data/6_augmented/train_src_balanced"))
    parser.add_argument("--ratio", type=float, default=2.0, help="Target Close:Open ratio")
    args = parser.parse_args()
    
    balance_dataset(args.input, args.output, args.ratio)
    print_pipeline_notice(
        output_paths=args.output,
        next_script="src/augment_dataset.py",
        notes=[
            "已完成 Close 降採樣。請使用 train_src_balanced 傳給下一步 augment_dataset.py 再進行訓練。"
        ]
    )
