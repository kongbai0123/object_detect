import os
import json
from pathlib import Path

def analyze_labels(version_dir):
    lbl_dir = version_dir / 'labels'
    if not lbl_dir.exists(): return None
    
    stats = {'total_imgs': 0, 'empty_imgs': 0, 'boxes': 0, 'open': 0, 'close': 0, 'avg_w': 0, 'avg_h': 0}
    w_sum = 0
    h_sum = 0
    
    for lbl in lbl_dir.glob('*.txt'):
        stats['total_imgs'] += 1
        with open(lbl, 'r') as f:
            lines = f.readlines()
            if not lines:
                stats['empty_imgs'] += 1
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    stats['boxes'] += 1
                    cls = int(parts[0])
                    if cls == 0: stats['open'] += 1
                    else: stats['close'] += 1
                    w_sum += float(parts[3])
                    h_sum += float(parts[4])
                    
    if stats['boxes'] > 0:
        stats['avg_w'] = w_sum / stats['boxes']
        stats['avg_h'] = h_sum / stats['boxes']
        
    return stats

base = Path(r'C:\antigravity\storage\assets\goldenset\versions')
for v in ['1_img', '2_img', '3_img', '4_img', '5_img']:
    s = analyze_labels(base / v)
    if s:
        print(f'--- {v} ---')
        print(f'Images: {s["total_imgs"]}, Empty(Bg): {s["empty_imgs"]} ({s["empty_imgs"]/s["total_imgs"]*100:.1f}%)')
        if s["boxes"] > 0:
            print(f'Boxes: {s["boxes"]}, Open: {s["open"]} ({s["open"]/s["boxes"]*100:.1f}%), Close: {s["close"]} ({s["close"]/s["boxes"]*100:.1f}%)')
            print(f'Avg BBox (W x H): {s["avg_w"]:.3f} x {s["avg_h"]:.3f}')
