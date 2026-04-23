import os
import shutil
import glob
from pathlib import Path

def recover_labels():
    frozen_labels_dir = Path("storage/assets/validation/frozen_v1/labels")
    golden_base_dir = Path("storage/assets/goldenset/versions")
    
    # 收集所有的 label 資料夾
    all_golden_label_dirs = list(golden_base_dir.glob("*/labels"))

    count = 0
    for txt_file in frozen_labels_dir.glob("*.txt"):
        if txt_file.stat().st_size == 0:
            found = False
            for golden_labels_dir in all_golden_label_dirs:
                golden_file = golden_labels_dir / txt_file.name
                if golden_file.exists() and golden_file.stat().st_size > 0:
                    with open(golden_file, 'r', encoding='utf-8-sig') as f:
                        content = f.read()
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    count += 1
                    found = True
                    break
            
            if not found:
                print(f"找不到原始標籤: {txt_file.name}")
    
    print(f"成功修復了 {count} 個標籤檔！")

if __name__ == "__main__":
    recover_labels()
