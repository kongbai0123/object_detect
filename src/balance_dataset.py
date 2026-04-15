import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
import argparse
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

ROOT = Path(__file__).resolve().parent.parent

def extract_scene_key(filename):
    name = Path(filename).stem
    m1 = re.match(r"(.*?)-[a-zA-Z0-9]+$", name)
    if m1: return m1.group(1)
    m2 = re.match(r"(.*?)_\d+$", name)
    if m2: return m2.group(1)
    return name

def analyze_image_boxes(lbl_file):
    """
    Returns lists of box areas for open and close classes.
    """
    open_areas = []
    close_areas = []

    if lbl_file.exists():
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 5: continue
                # class cx cy w h
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                
                if cls_id == 0:
                    open_areas.append(area)
                elif cls_id == 1:
                    close_areas.append(area)
                    
    return open_areas, close_areas

def balance_dataset(input_dir, output_dir, target_ratio=2.0, open_hard_threshold=0.05):
    in_img_dir = Path(input_dir) / "images"
    in_lbl_dir = Path(input_dir) / "labels"
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(in_img_dir.glob("*.jpg")) + list(in_img_dir.glob("*.png"))
    
    kept_images = []
    
    # Categories
    hard_negatives = []
    mixed_boundary = []
    open_only = []
    close_only = []
    open_hard_count = 0
    
    total_open_boxes_before = 0
    total_close_boxes_before = 0
    
    # 紀錄各圖片的 statistics 供後續排序使用
    img_stats = {}

    for img in images:
        lbl = in_lbl_dir / f"{img.stem}.txt"
        open_areas, close_areas = analyze_image_boxes(lbl)
        
        o_cnt = len(open_areas)
        c_cnt = len(close_areas)
        
        total_open_boxes_before += o_cnt
        total_close_boxes_before += c_cnt
        
        has_open = o_cnt > 0
        has_close = c_cnt > 0
        
        img_stats[img] = {
            'open_cnt': o_cnt,
            'close_cnt': c_cnt,
            'avg_open_area': sum(open_areas)/o_cnt if o_cnt else 0,
            'avg_close_area': sum(close_areas)/c_cnt if c_cnt else 0,
        }
        
        if has_open and any(a < open_hard_threshold for a in open_areas):
            open_hard_count += 1
            
        if not has_open and not has_close:
            hard_negatives.append(img)
            kept_images.append(img)
        elif has_open and has_close:
            mixed_boundary.append(img)
            kept_images.append(img)
        elif has_open:
            open_only.append(img)
            kept_images.append(img)
        else: # only close
            close_only.append(img)
            
    print(f"📦 [第一階段] 分析完畢！")
    print(f"原始狀態 -> 總 Open Box: {total_open_boxes_before}, 總 Close Box: {total_close_boxes_before}")
    print(f"--- 錯誤型態 (Error-aware) 分群結果 ---")
    print(f"🔹 Hard negatives (純背景無框，全留): {len(hard_negatives)} 張")
    print(f"🔹 Mixed boundary (同框包含開與關，最強邊界資料，全留): {len(mixed_boundary)} 張")
    print(f"🔹 Open_only (全留): {len(open_only)} 張")
    print(f"   => 其中包含遠距/小目標 Open (w*h < {open_hard_threshold}) 的圖片數: {open_hard_count} 張")
    print(f"🔹 Close_only (等待 Box-level 精準降採樣): {len(close_only)} 張")
    print(f"----------------------------------------")
    
    # 精算目標 Box
    target_close_boxes = int(total_open_boxes_before * target_ratio)
    
    # 扣掉已經全留的 close boxes (來自 mixed 框)
    current_close_boxes = 0
    for img in kept_images:
        current_close_boxes += img_stats[img]['close_cnt']
        
    required_pure_close_boxes = max(0, target_close_boxes - current_close_boxes)
    print(f"🎯 目標將總 Close 框數控制在約 {target_close_boxes} 箱 (Mixed 內已含 {current_close_boxes}，尚需抽取 {required_pure_close_boxes} 箱來自純 Close 圖)")
    
    # 執行 Priority Queue 抽樣
    # 我們根據 close 框的難度進行打分：框越小 (avg_close_area 越小)、或是單圖多框 (過濾密集物) 的我們認為越值得保留
    # Score 算法：以 1.0 / (avg_area + 0.05) 為主, 框越小分數越高。若有一圖多框則額外加權。
    
    close_only_scored = []
    for img in close_only:
        stats = img_stats[img]
        avg_area = stats['avg_close_area']
        cnt = stats['close_cnt']
        # 難度分數：小面積越高分。多個近接物體也算是難例，給予加分
        score = (1.0 / (avg_area + 0.05)) * (min(cnt, 3))
        close_only_scored.append((score, img, cnt))
        
    # 高分 (最難) 優先
    close_only_scored.sort(key=lambda x: x[0], reverse=True)
    
    sampled_close_imgs = 0
    added_close_boxes = 0
    
    for score, img, cnt in close_only_scored:
        if added_close_boxes >= required_pure_close_boxes and required_pure_close_boxes > 0:
            break
        kept_images.append(img)
        added_close_boxes += cnt
        sampled_close_imgs += 1
        
    dropped_close_imgs = len(close_only) - sampled_close_imgs
        
    # 計算平衡後的最終狀態
    final_open = 0
    final_close = 0
    for img in kept_images:
        final_open += img_stats[img]['open_cnt']
        final_close += img_stats[img]['close_cnt']
        
        # Copy to output
        out_img_path = out_img_dir / img.name
        if not out_img_path.exists() or os.stat(img).st_mtime > os.stat(out_img_path).st_mtime:
             shutil.copy2(img, out_img_path)
             
        lbl = in_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            out_lbl_path = out_lbl_dir / lbl.name
            if not out_lbl_path.exists() or os.stat(lbl).st_mtime > os.stat(out_lbl_path).st_mtime:
                shutil.copy2(lbl, out_lbl_path)

    print("\n✅ 平衡完成！")
    print(f"從 {len(close_only)} 張極為普通的純 Close 背景圖中，精準抽取了最困難的 {sampled_close_imgs} 張進入訓練集。")
    print(f"丟棄了 {dropped_close_imgs} 張低資訊價值的純 Close 圖。")
    print(f"最終 Box 數 -> Open Box: {final_open}, Close Box: {final_close} (比例 1 : {final_close/final_open if final_open else 0:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ROOT / "data/6_augmented/train_src"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data/6_augmented/train_src_balanced"))
    parser.add_argument("--ratio", type=float, default=2.0, help="Target Close:Open box ratio")
    parser.add_argument("--hard_th", type=float, default=0.05, help="Area ratio threshold to define open_hard")
    args = parser.parse_args()
    
    balance_dataset(args.input, args.output, args.ratio, args.hard_th)
    print_pipeline_notice(
        output_paths=args.output,
        next_script="src/augment_dataset.py",
        notes=[
            "🎯 已完成 Box-level 精準難度抽樣降載！",
            "請將 train_src_balanced 接力傳給下一步 augment_dataset.py 再進行訓練。"
        ]
    )
