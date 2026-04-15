import os
import shutil
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

ROOT = Path(__file__).resolve().parent.parent

def extract_scene_key(filename):
    """
    從檔名萃取 scene key，避免同一影片的相鄰幀被切分到不同 set。
    例如：
    - BDD100k 格式: '00054602-3bf57337.jpg' -> '00054602'
    - 連續幀格式: 'video1_frame001.jpg' -> 'video1_frame'
    - 其他: fallback 到完整檔名 (獨立場景)
    """
    name = Path(filename).stem
    
    # 規則 1: 破折號分隔 (如 BDD)
    m1 = re.match(r"(.*?)-[a-zA-Z0-9]+$", name)
    if m1: return m1.group(1)
        
    # 規則 2: 底線加數字結尾 (如連續截圖)
    m2 = re.match(r"(.*?)_\d+$", name)
    if m2: return m2.group(1)
        
    return name # 獨立場景

def count_labels(lbl_dir, img_list):
    open_cnt, close_cnt = 0, 0
    for img in img_list:
        lbl_file = lbl_dir / f"{img.stem}.txt"
        if lbl_file.exists():
            with open(lbl_file, 'r') as f:
                content = f.read().split()
                if not content: continue
                # 如果有多個 bounding box
                lines = f.read().split('\n')
                lines = [c for c in content if len(c) > 0]
                # 重新準確計算
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        if parts[0] == '0': open_cnt += 1
                        elif parts[0] == '1': close_cnt += 1
    return open_cnt, close_cnt

def main():
    parser = argparse.ArgumentParser(description="Scene-Aware Split dataset into Train and Val")
    parser.add_argument("--input", type=str, default=str(ROOT / "data/3_processed"))
    parser.add_argument("--out_train", type=str, default=str(ROOT / "data/6_augmented/train_src"))
    parser.add_argument("--out_val", type=str, default=str(ROOT / "data/6_augmented/val"))
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empty-labels", action="store_true", default=True, help="是否為無標籤背景產生空的 txt")
    parser.add_argument("--freeze_val", type=str, default=str(ROOT / "data/val_frozen_v1"),
                        help="若此路徑存在，則跳過 val 切分，直接同步使用凍結 val 集")
    args = parser.parse_args()

    random.seed(args.seed)
    
    input_path = Path(args.input)
    if (input_path / "images").exists():
        img_dir = input_path / "images"
    elif (input_path / "image").exists():
        img_dir = input_path / "image"
    else:
        img_dir = input_path
    lbl_dir = input_path / "labels" if (input_path / "labels").exists() else input_path
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"錯誤：沒有在 {img_dir} 找到圖片")
        return
        
    # 1. 根據 Scene Key 分群
    groups = defaultdict(list)
    for img in images:
        key = extract_scene_key(img.name)
        groups[key].append(img)
        
    group_keys = list(groups.keys())
    
    # 若所有圖片被歸入同一 scene key，則退回以檔名為 key（獨立場景模式）
    if len(group_keys) <= 1 and len(images) > 1:
        groups = defaultdict(list)
        for img in images:
            groups[img.stem].append(img)
        group_keys = list(groups.keys())
    
    random.shuffle(group_keys)
    
    # 2. Scene-Aware 切分
    split_idx = max(1, int(len(group_keys) * args.split))
    train_keys = group_keys[:split_idx]
    val_keys = group_keys[split_idx:]
    
    train_imgs = []
    for k in train_keys: train_imgs.extend(groups[k])
        
    val_imgs = []
    for k in val_keys: val_imgs.extend(groups[k])

    # --- Frozen Val 支援 ---
    frozen_val_path = Path(args.freeze_val)
    use_frozen_val = frozen_val_path.exists() and any(frozen_val_path.rglob("*.jpg"))
    if use_frozen_val:
        print(f"[Frozen Val] 偵測到 {frozen_val_path}，略過 val 切分，使用凍結 val 集。")
        # 將原本切出的 val 歸還給 train
        train_imgs.extend(val_imgs)
        val_imgs = []
        # 同步凍結 val 至 out_val
        import shutil as _shutil
        out_val_path = Path(args.out_val)
        for subdir in ["images", "labels"]:
            (out_val_path / subdir).mkdir(parents=True, exist_ok=True)
            for f in (frozen_val_path / subdir).glob("*"):
                _shutil.copy2(f, out_val_path / subdir / f.name)
        print(f"[Frozen Val] 已同步凍結 val 至 {args.out_val} ({len(list((frozen_val_path/'images').glob('*')))} 張)")
        
    # 建立輸出庫
    for d in [args.out_train, args.out_val]:
        for subdir in ["images", "labels"]:
            (Path(d) / subdir).mkdir(parents=True, exist_ok=True)
            
    def copy_split(img_list, keys_list, out_dir, name):
        out_path = Path(out_dir)
        for img in img_list:
            lbl_name = img.stem + ".txt"
            lbl_file = lbl_dir / lbl_name
            
            shutil.copy2(img, out_path / "images" / img.name)
            
            if lbl_file.exists():
                shutil.copy2(lbl_file, out_path / "labels" / lbl_name)
            elif args.empty_labels:
                # 衛生修補: 生成空白標籤供 Ultralytics 正確視為背景
                with open(out_path / "labels" / lbl_name, 'w') as f:
                    pass
                    
        o_cnt, c_cnt = count_labels(lbl_dir, img_list)
        bg_cnt = len([img for img in img_list if not (lbl_dir / f"{img.stem}.txt").exists() or os.path.getsize(lbl_dir / f"{img.stem}.txt") == 0])
        
        print(f"[{name}]")
        print(f"  - Scenes (Groups): {len(keys_list)}")
        print(f"  - Images: {len(img_list)}")
        print(f"  - Open box: {o_cnt}, Close box: {c_cnt}, Pure BG images: {bg_cnt}")
        print("-" * 30)

    print(f"\n 開始進行 [Scene-Aware Split]...")
    copy_split(train_imgs, train_keys, args.out_train, "Train 訓練集")
    copy_split(val_imgs, val_keys, args.out_val, "Val 基準驗證集 (val_frozen)")
    
    # 強制注入 Replay Core 到 Train
    replay_dir = ROOT / "data/9_replay_core"
    if replay_dir.exists():
        print("\n [Replay Core] 啟動注入保護...")
        replay_imgs = list(replay_dir.rglob("*.jpg")) + list(replay_dir.rglob("*.png"))
        if replay_imgs:
            out_train_img = Path(args.out_train) / "images"
            out_train_lbl = Path(args.out_train) / "labels"
            r_cnt = 0
            for r_img in replay_imgs:
                r_lbl = r_img.parent / f"{r_img.stem}.txt"
                shutil.copy2(r_img, out_train_img / r_img.name)
                if r_lbl.exists():
                    shutil.copy2(r_lbl, out_train_lbl / r_lbl.name)
                elif args.empty_labels:
                    with open(out_train_lbl / f"{r_img.stem}.txt", 'w') as f:
                        pass
                r_cnt += 1
            print(f"✅ 成功將 {r_cnt} 張 Replay Core 定海神針強制植入訓練集！")
            
    print("\n切分完成！\n")

if __name__ == "__main__":
    main()
    print_pipeline_notice(
        output_paths=[str(ROOT / "data/6_augmented/train_src"), str(ROOT / "data/6_augmented/val")],
        next_script="src/balance_dataset.py",
        notes=[
            "已啟用 Scene-Aware Split 預防場景洩露。",
            "若進行 0.6.1 訓練，請先執行 balance_dataset.py 進行 close 降採樣。",
        ],
    )
