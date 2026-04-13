import os
import shutil
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from pipeline_notice import print_pipeline_notice

# 無論從哪個目錄執行，路徑永遠相對於此腳本所在位置
ROOT = Path(__file__).resolve().parent.parent

# 自動偵測並安裝 fiftyone 套件
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("尚未安裝 fiftyone。正在自動安裝...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fiftyone"])
    import fiftyone as fo
    import fiftyone.zoo as foz


def download_and_export_openimages(max_samples: int = 5000):
    output_base = ROOT / "data/4_external/openimages"  # 【輸出】Open Images 下載完成的圖片與標籤
    img_dir = output_base / "images"
    lbl_dir = output_base / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"開始下載 Open Images V7 子集 (Car, Person) - 上限 {max_samples} 張...")
    # 透過 Zoo 僅下載包含我們需要的類別與資訊
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Car", "Person"],
        max_samples=max_samples,
        dataset_name="openimages-car-person-custom"
    )

    # 嚴格定義 YOLO class mapping
    class_mapping = {
        "Car": 0,
        "Person": 1
    }

    print(f"開始匯出至 {output_base}")
    print(f" 影像將儲存於: {img_dir}")
    print(f" 標籤將儲存於: {lbl_dir} (0: car, 1: person)")
    
    exported_count = 0
    total_samples = len(dataset)
    
    for sample in tqdm(dataset, desc="[轉換/匯出 YOLO 格式]", unit="張"):
        # Open Images V7 在 fiftyone 裡的預設檢測框欄位名是 "ground_truth"
        if not sample.has_field("ground_truth") or sample.ground_truth is None:
            continue
            
        src_img_path = Path(sample.filepath)
        dest_img_path = img_dir / src_img_path.name
        
        yolo_lines = []
        for det in sample.ground_truth.detections:
            if det.label in class_mapping:
                cls_id = class_mapping[det.label]
                # FiftyOne bbox 的原生格式是 [top-left-x, top-left-y, width, height] (相對於圖片比例 0~1)
                x, y, w, h = det.bounding_box
                
                # 轉換為 YOLO 格式所需的 [center-x, center-y, width, height]
                cx = x + w / 2.0
                cy = y + h / 2.0
                
                # 確保邊界值合法
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # 僅複製真的包含 Car 或 Person 標籤的影像
        if yolo_lines:
            if not dest_img_path.exists():
                shutil.copy2(src_img_path, dest_img_path)
            
            lbl_path = lbl_dir / f"{src_img_path.stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")
            
            exported_count += 1

    print(f"\n匯出完成！成功轉換並儲存 {exported_count} 張圖片及其 YOLO 格式標籤。")


if __name__ == "__main__":
    download_and_export_openimages(max_samples=3000)
    print_pipeline_notice(
        output_paths=[str(ROOT / "data/4_external/openimages/images"), str(ROOT / "data/4_external/openimages/labels")],
        next_script="src/tools/parse_bdd10k.py",
        notes=[
            "這是外部資料來源之一，可和其他來源一起整理後再進入正式訓練流程。",
            "若只使用 Open Images，也可以在人工檢查後自行接到 split_dataset.py。",
        ],
    )
