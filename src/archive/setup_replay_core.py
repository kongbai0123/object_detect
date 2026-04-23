import os
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def add_to_replay_core(category, source_img_dir, source_lbl_dir, file_list):
    """
    將指定清單的圖片與標籤加入 Replay Core
    """
    target_dir = ROOT / "data/9_replay_core" / category
    target_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for file_name in file_list:
        img_src = Path(source_img_dir) / file_name
        lbl_src = Path(source_lbl_dir) / f"{img_src.stem}.txt"
        
        if not img_src.exists():
            print(f"⚠️ 找不到圖檔: {img_src}")
            continue
            
        new_name = f"replay_{category}_{img_src.name}"
        shutil.copy2(img_src, target_dir / new_name)
        
        if lbl_src.exists():
            shutil.copy2(lbl_src, target_dir / f"{Path(new_name).stem}.txt")
        else:
            # 如果是 ghost 且沒標籤，建立空標籤
            if category == 'ghost':
                with open(target_dir / f"{Path(new_name).stem}.txt", 'w') as f:
                    pass
            else:
                print(f"⚠️ 找不到標籤檔: {lbl_src}")
        
        count += 1
    
    print(f"✅ 已成功將 {count} 張樣本加入 Replay Core [{category}]。")

def init_replay_folders():
    for cat in ['open', 'close', 'ghost']:
        (ROOT / "data/9_replay_core" / cat).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=['init', 'add'], default='init')
    parser.add_argument("--category", type=str, choices=['open', 'close', 'ghost'])
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--lbl_dir", type=str)
    parser.add_argument("--files", nargs='+', help="檔案名稱列表")
    
    args = parser.parse_args()
    
    if args.action == 'init':
        init_replay_folders()
    elif args.action == 'add':
        if not all([args.category, args.img_dir, args.lbl_dir, args.files]):
            print("❌ 'add' 模式需要 --category, --img_dir, --lbl_dir 與 --files")
        else:
            add_to_replay_core(args.category, args.img_dir, args.lbl_dir, args.files)
