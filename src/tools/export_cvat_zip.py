import os
import shutil
import zipfile
import yaml
from pathlib import Path
import argparse

def export_for_cvat(source_dir, output_zip):
    """
    將 YOLO 格式資料夾打包成 CVAT YOLO 1.1 匯入格式
    source_dir: 包含 images/ 和 labels/ 的資料夾
    output_zip: 輸出的 zip 檔案路徑
    """
    source_path = Path(source_dir)
    # 支持多種可能的子資料夾名稱
    img_dir = source_path / "images"
    if not img_dir.exists(): img_dir = source_path / "image"
    
    lbl_dir = source_path / "labels"
    if not lbl_dir.exists(): lbl_dir = source_path / "label"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"錯誤：來源路徑 {source_path} 內部找不到 images/ 或 labels/ 子資料夾")
        return

    # 1. 讀取類別名稱 (從 dataset.yaml)
    yaml_path = Path("data/dataset.yaml")
    if not yaml_path.exists():
        print("警告：未找到 data/dataset.yaml，將使用預設 [open, close] 類別")
        class_names = ["open", "close"]
    else:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        class_names = [data_cfg['names'][i] for i in range(len(data_cfg['names']))]
    
    # 2. 建立臨時工作目錄
    temp_dir = Path("temp_cvat_export")
    data_subdir = temp_dir / "data"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    os.makedirs(data_subdir, exist_ok=True)

    # 3. 準備 obj.names
    names_file = temp_dir / "obj.names"
    with open(names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # 4. 準備 obj.data
    data_file = temp_dir / "obj.data"
    with open(data_file, 'w') as f:
        f.write(f"classes = {len(class_names)}\n")
        f.write("train = data/train.txt\n")
        f.write("names = data/obj.names\n")
        f.write("backup = backup/\n")

    # 5. 複製圖片與標籤，並建立 train.txt
    train_txt = temp_dir / "train.txt"
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    count = 0
    
    with open(train_txt, 'w') as f_list:
        for img_file in img_dir.glob("*"):
            if img_file.suffix.lower() in valid_extensions:
                # 找出對應的標籤
                label_file = lbl_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    # 複製到 data/ 子目錄
                    shutil.copy(img_file, data_subdir / img_file.name)
                    shutil.copy(label_file, data_subdir / label_file.name)
                    # 寫入清單 (路徑需為 data/filename)
                    f_list.write(f"data/{img_file.name}\n")
                    count += 1

    # 6. 打包成 ZIP (遵守 CVAT YOLO 1.1 結構)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(data_file, "obj.data")
        z.write(names_file, "obj.names")
        z.write(train_txt, "train.txt")
        # 同時將 names 放入 data 內部以防萬一
        z.write(names_file, "data/obj.names")
        # 遍歷 data/ 子目錄內所有圖標
        for file in data_subdir.glob("*"):
            z.write(file, f"data/{file.name}")

    # 7. 清理
    shutil.rmtree(temp_dir)
    print("==========================================")
    print(f"DONE: Success! Processed {count} samples.")
    print(f"Source: {source_path}")
    print(f"Output ZIP: {output_zip}")
    print("CVAT Info: Upload as 'YOLO 1.1' in Annotations.")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack YOLO dataset for CVAT 1.1 import")
    # 將所需參數設為可選，並提供預設值
    parser.add_argument("--input", "-i", default="C:/antigravity/origin_img/door_open", help="Input directory")
    parser.add_argument("--output", "-o", default="cvat_export.zip", help="Output ZIP filename")
    
    args = parser.parse_args()
    
    # 執行轉換
    export_for_cvat(args.input, args.output)
