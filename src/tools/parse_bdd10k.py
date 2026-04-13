import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
from pipeline_notice import print_pipeline_notice

# 無論從哪個目錄執行，路徑永遠相對於此腳本所在位置
ROOT = Path(__file__).resolve().parent.parent

def parse_bdd10k(
    json_path: str,
    img_src_dir: str,
    output_dir: str,
    max_samples: int = 10000
):
    json_path = Path(json_path)
    img_src_dir = Path(img_src_dir)
    output_base = Path(output_dir)
    
    img_out_dir = output_base / "images"
    lbl_out_dir = output_base / "labels"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    if not json_path.exists():
        print(f"錯誤：找不到標籤檔 {json_path}")
        print("請確認您下載的 BDD100K 標籤檔路徑是否正確。")
        return
        
    print(f"開始載入 BDD 標籤 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        bdd_data = json.load(f)
        
    print(f"總共讀取到 {len(bdd_data)} 筆影像標註，將為您擷取前 {max_samples} 張有效資料。")
    
    # YOLO 標籤映射
    class_mapping = {
        "car": 0,
        "person": 1
    }
    
    exported_count = 0
    missing_images = 0
    
    for item in tqdm(bdd_data, desc="[解析 BDD 轉 YOLO]", unit="張"):
        if exported_count >= max_samples:
            break
            
        img_name = item.get("name")
        labels = item.get("labels", [])
        
        # 篩選我們需要的 YOLO BBox: 
        yolo_lines = []
        for lbl in labels:
            cat = lbl.get("category", "").lower()
            if cat in class_mapping and "box2d" in lbl:
                cls_id = class_mapping[cat]
                box = lbl["box2d"]
                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])
                
                # BDD100K 的標準影像大小是 1280x720，但為防萬一我們動態讀取
                # 若速度太慢可以取消 Image.open 註解，直接寫死 dw = 1/1280.0, dh = 1/720.0
                # cx, cy, w, h
                bw = x2 - x1
                bh = y2 - y1
                bcx = x1 + bw / 2.0
                bcy = y1 + bh / 2.0
                
                # 保留特徵供稍後歸一化
                yolo_lines.append((cls_id, bcx, bcy, bw, bh))

        if not yolo_lines:
            continue
            
        src_img_path = img_src_dir / img_name
        dest_img_path = img_out_dir / img_name
        dest_lbl_path = lbl_out_dir / f"{Path(img_name).stem}.txt"
        
        if not src_img_path.exists():
            missing_images += 1
            continue
            
        # 取得實際影像長寬來做正規化 (YOLO 格式要求) 
        # (這裡利用 PIL 的 lazy loading 只讀 metadata，速度極快)
        try:
            with Image.open(src_img_path) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1280.0, 720.0 # 預設 Fallback

        # 寫入 YOLO 標籤
        normalized_lines = []
        for cls_id, bcx, bcy, bw, bh in yolo_lines:
            nx = bcx / img_w
            ny = bcy / img_h
            nw = bw / img_w
            nh = bh / img_h
            
            # 防呆：確保在 0~1 之間
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            
            normalized_lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            
        with open(dest_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized_lines) + "\n")
            
        # 拷貝圖片
        if not dest_img_path.exists():
            shutil.copy2(src_img_path, dest_img_path)
            
        exported_count += 1
        
    print(f"\n匯出完成！成功從 BDD10K 抽取並存到 {output_dir}")
    print(f"總計匯出: {exported_count} 張圖片及標籤。")
    if missing_images > 0:
        print(f"警告：有 {missing_images} 張圖片在 JSON 中有標註，但在您提供的圖片資料夾 {img_src_dir} 中找不到實體檔案。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD10K to YOLO Parser (Only Car/Person)")
    parser.add_argument("--json",        type=str,                                                                     # 【輸入】BDD100K 標注 JSON
                        default=str(ROOT / "data/4_external/bdd100k/labels/bdd100k_labels_images_train.json"),
                        help="BDD100K 標注 JSON 路徑")
    parser.add_argument("--img_dir",     type=str,                                                                     # 【輸入】BDD100K 原始圖片
                        default=str(ROOT / "data/4_external/bdd100k/images/100k/train"),
                        help="BDD100K 圖片目錄")
    parser.add_argument("--output",      type=str, default=str(ROOT / "data/4_external/bdd10k"),                       # 【輸出】轉換完成的 YOLO 格式標注
                        help="輸出目錄")
    parser.add_argument("--max_samples", type=int, default=10000)
    
    args = parser.parse_args()
    
    parse_bdd10k(
        json_path=args.json,
        img_src_dir=args.img_dir,
        output_dir=args.output,
        max_samples=args.max_samples
    )
    print_pipeline_notice(
        output_paths=args.output,
        next_script="src/split_dataset.py",
        notes=[
            "BDD10K 解析後會轉成 YOLO 格式，可與其他來源一起整理。",
            "若此資料夾要進入正式流程，請先確認類別映射與影像品質是否符合專案定義。",
        ],
    )
