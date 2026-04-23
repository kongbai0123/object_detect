import os
from pathlib import Path

def fix_polygons_to_bboxes(base_dir):
    base_path = Path(base_dir)
    fixed_count = 0
    total_scanned = 0
    
    # 掃描所有 txt 檔
    for txt_file in base_path.rglob("*.txt"):
        if txt_file.name == "classes.txt":
            continue
            
        total_scanned += 1
        needs_fix = False
        fixed_lines = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split()
                if not p:
                    continue
                    
                cls_id = p[0]
                
                # 如果是多邊形 (數值大於 5 個)
                if len(p) > 5:
                    needs_fix = True
                    coords = [float(x) for x in p[1:]]
                    min_x, max_x = min(coords[0::2]), max(coords[0::2])
                    min_y, max_y = min(coords[1::2]), max(coords[1::2])
                    w, h = max_x - min_x, max_y - min_y
                    xc, yc = (min_x + max_x) / 2, (min_y + max_y) / 2
                    
                    fixed_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                else:
                    fixed_lines.append(line.strip())
                    
        # 如果發現有問題的行，就覆寫該檔案
        if needs_fix:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(fixed_lines) + "\n")
            fixed_count += 1

    return total_scanned, fixed_count

if __name__ == "__main__":
    target_dir = r"C:\antigravity\storage\assets\goldenset\versions"
    print(f"開始掃描目錄: {target_dir}")
    scanned, fixed = fix_polygons_to_bboxes(target_dir)
    print(f"掃描完畢！總共檢查了 {scanned} 個標籤檔。")
    print(f"成功修復了 {fixed} 個包含多邊形的異常檔案！")
