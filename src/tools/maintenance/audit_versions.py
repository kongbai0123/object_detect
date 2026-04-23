import os
from pathlib import Path
from collections import Counter

def analyze_all_versions(versions_root):
    root = Path(versions_root)
    versions = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print(f"{'Version':<15} | {'Images':<8} | {'BG%':<8} | {'Total Box':<10} | {'Open':<8} | {'Close':<8} | {'Ratio':<8}")
    print("-" * 80)
    
    global_stats = Counter()
    total_images = 0

    for v in versions:
        img_dir = v / "images"
        lbl_dir = v / "labels"
        
        if not lbl_dir.exists(): continue
        
        lbl_files = list(lbl_dir.glob("*.txt"))
        img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        total_images += img_count
        
        v_open = 0
        v_close = 0
        v_bg = 0
        
        for lbl in lbl_files:
            with open(lbl, 'r') as f:
                lines = f.readlines()
                if not lines:
                    v_bg += 1
                    continue
                
                for line in lines:
                    cls_id = line.split()[0]
                    if cls_id == '0': v_open += 1
                    elif cls_id == '1': v_close += 1
        
        total_v_boxes = v_open + v_close
        ratio = round(v_open / v_close, 3) if v_close > 0 else 0
        bg_p = round((v_bg / len(lbl_files)) * 100, 1) if lbl_files else 0
        
        print(f"{v.name:<15} | {img_count:<8} | {bg_p:<7}% | {total_v_boxes:<10} | {v_open:<8} | {v_close:<8} | {ratio:<8}")
        
        global_stats['open'] += v_open
        global_stats['close'] += v_close
        global_stats['bg'] += v_bg

    print("-" * 80)
    total_boxes = global_stats['open'] + global_stats['close']
    g_ratio = round(global_stats['open'] / global_stats['close'], 3)
    g_bg_p = round((global_stats['bg'] / total_images) * 100, 1)
    print(f"{'GLOBAL':<15} | {total_images:<8} | {g_bg_p:<7}% | {total_boxes:<10} | {global_stats['open']:<8} | {global_stats['close']:<8} | {g_ratio:<8}")

if __name__ == "__main__":
    analyze_all_versions(r"C:\antigravity\storage\assets\goldenset\versions")
