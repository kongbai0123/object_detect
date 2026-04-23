import os
from datetime import datetime
import argparse
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
import glob as _glob
import zipfile
import yaml
import sys
from pathlib import Path
import json
import shutil
from PIL import Image
from tqdm import tqdm
from anti_gravity.settings import settings
from anti_gravity.pipeline_notice import print_pipeline_notice

def find_best_local_model():
    """
    深度搜尋 experiments 目錄下的最強模型
    優先級：global_best.pt > 最新增量權重 > 最新基礎權重 > yolov8s.pt
    """
    # 1. 優先找晉升後的全局最強模型
    global_best = settings.paths.models_promoted / "global_best.pt"
    if global_best.exists():
        return str(global_best)
    
    # 2. 深度搜尋 experiments 目錄下的所有 best.pt
    exp_dir = settings.paths.experiments
    if exp_dir.exists():
        # 遞迴搜尋所有模式 (rebuild, incremental) 下的 best.pt
        all_weights = list(exp_dir.rglob("best.pt"))
        if all_weights:
            # 優先找修改時間最新的
            latest_weight = max(all_weights, key=os.path.getmtime)
            return str(latest_weight)
            
    return "yolov8s.pt"

def run_auto_annotation_refined(data_dir, det_model='yolov8s.pt', sam_model='sam2_t.pt', 
                               conf=0.6, iou=0.5, imgsz=768, min_box_area_px=144, min_box_dim_px=8):
    """
    優化版自動標註：支援遞迴目錄，使用 tqdm 刷新進度
    """
    data_path = Path(data_dir)
    
    # 建立輸出目錄
    final_out_dir = settings.paths.auto_ann / "current"
    img_out_dir = final_out_dir / 'images'
    lbl_out_dir = final_out_dir / 'labels'
    for d in [img_out_dir, lbl_out_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 搜尋影像 (支援大小寫)
    img_mapping = {}
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP']:
        for p in data_path.rglob(f'*{ext}'):
            img_mapping[p.stem] = p
            
    if not img_mapping:
        print(f"❌ 錯誤: 在 {data_dir} 找不到任何影像。")
        return None

    # === [關鍵修復] 處理遞迴目錄：將影像扁平化到臨時工作區 ===
    # 因為 auto_annotate 不支援遞迴搜尋子目錄
    flat_work_dir = settings.paths.auto_ann / f"temp_flat_{datetime.now().strftime('%H%M%S')}"
    if flat_work_dir.exists(): shutil.rmtree(flat_work_dir)
    flat_work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 正在扁平化影像至工作區...")
    for stem, p in img_mapping.items():
        shutil.copy2(p, flat_work_dir / p.name)

    print(f"📦 正在準備標註引擎 (Model: {os.path.basename(det_model)})...")
    
    print(f"🚀 開始自動標註 ({len(img_mapping)} 張影像):")
    
    # 使用 tqdm 建立進度條
    pbar = tqdm(total=len(img_mapping), desc="Annotating", unit="img")
    
    # 執行 auto_annotate
    try:
        # --- [強效靜音] ---
        import logging
        for logger_name in ['ultralytics', 'yolo', 'sam']:
            l = logging.getLogger(logger_name)
            l.setLevel(logging.ERROR)
            l.propagate = False
            
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        auto_annotate(
            data=str(flat_work_dir), 
            det_model=det_model, 
            sam_model=sam_model, 
            conf=conf, 
            iou=iou, 
            imgsz=imgsz
        )
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"❌ 標註過程發生錯誤: {e}")
    finally:
        pbar.update(len(img_mapping))
        pbar.close()

    # === 後處理與治理 ===
    stats = {"open": 0, "close": 0, "filtered": 0, "empty": 0}
    
    # Ultralytics 預設會產生在 {data_dir}_auto_annotate_labels
    temp_labels = Path(str(flat_work_dir) + "_auto_annotate_labels")
    
    # 處理標籤
    label_files = list(temp_labels.glob("*.txt")) if temp_labels.exists() else []
    for lbl_path in label_files:
        stem = lbl_path.stem
        img_p = img_mapping.get(stem)
        if not img_p: continue
        
        # 讀取影像大小
        with Image.open(img_p) as im:
            iw, ih = im.size
            
        valid_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                if not p: continue
                cls_id = int(p[0])
                
                # 將多邊形轉換為標準 BBox，或保持 BBox 原樣
                if len(p) > 5: # SAM 產生的多邊形 (Polygon)
                    coords = [float(x) for x in p[1:]]
                    min_x, max_x = min(coords[0::2]), max(coords[0::2])
                    min_y, max_y = min(coords[1::2]), max(coords[1::2])
                    
                    # 限制在 frame 內 (0.0 到 1.0 之間)
                    min_x, max_x = max(0.0, min_x), min(1.0, max_x)
                    min_y, max_y = max(0.0, min_y), min(1.0, max_y)
                    
                    w, h = max_x - min_x, max_y - min_y
                    xc, yc = (min_x + max_x) / 2, (min_y + max_y) / 2
                    
                    # 轉換為標準 YOLO 偵測框格式
                    out_line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                else: # 已經是 YOLO BBox
                    _, xc, yc, w, h = map(float, p)
                    
                    # 限制原有 YOLO BBox 也在 frame 內
                    min_x, max_x = max(0.0, xc - w/2), min(1.0, xc + w/2)
                    min_y, max_y = max(0.0, yc - h/2), min(1.0, yc + h/2)
                    w, h = max_x - min_x, max_y - min_y
                    xc, yc = (min_x + max_x) / 2, (min_y + max_y) / 2
                    
                    out_line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                
                # 執行噪點過濾
                if (w * iw * h * ih < min_box_area_px) or min(w * iw, h * ih) < min_box_dim_px:
                    stats["filtered"] += 1
                    continue
                
                valid_lines.append(out_line)
                if cls_id == 0: stats["open"] += 1
                else: stats["close"] += 1
        
        # 存檔
        (lbl_out_dir / lbl_path.name).write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))

    # 複製影像並補全
    for stem, p in img_mapping.items():
        shutil.copy2(p, img_out_dir / p.name)
        target_lbl = lbl_out_dir / f"{stem}.txt"
        if not target_lbl.exists():
            target_lbl.write_text("")
            stats["empty"] += 1

    # 清理臨時目錄
    if temp_labels.exists(): shutil.rmtree(temp_labels)
    if flat_work_dir.exists(): shutil.rmtree(flat_work_dir)
    
    # 產出摘要
    print(f"\n✅ 標註完成！ 總影像: {len(img_mapping)} | 偵測框: {stats['open']+stats['close']} | 空標籤: {stats['empty']} | 已過濾噪點: {stats['filtered']}")
    return final_out_dir

def create_cvat_package(img_dir, lbl_dir):
    lbl_path = Path(lbl_dir)
    archive_subdir = settings.paths.artifacts / "exports"
    archive_subdir.mkdir(parents=True, exist_ok=True)
    zip_name = f"cvat_auto_ann_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    zip_out = archive_subdir / zip_name
    
    class_names = ["open", "close"]
    txt_files = [f for f in lbl_path.glob("*.txt") if f.name != "classes.txt"]
    if not txt_files: return None

    with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", "\n".join(class_names) + "\n")
        zf.writestr("obj.data", f"classes = 2\ntrain = train.txt\nnames = obj.names\n")
        
        train_list = []
        img_src_dir = Path(img_dir)
        for txt in txt_files:
            img_p = next((img_src_dir / f"{txt.stem}{ext}" for ext in ['.jpg', '.png', '.jpeg', '.webp'] if (img_src_dir / f"{txt.stem}{ext}").exists()), None)
            if not img_p: continue
            zf.write(str(img_p), f"obj_train_data/{img_p.name}")
            train_list.append(f"obj_train_data/{img_p.name}")
            # 轉換多邊形為 BBox (CVAT 需求)
            lines = []
            with open(txt, 'r') as tf:
                for line in tf:
                    p = line.strip().split()
                    if len(p) > 5:
                        coords = [float(x) for x in p[1:]]
                        min_x, max_x, min_y, max_y = min(coords[0::2]), max(coords[0::2]), min(coords[1::2]), max(coords[1::2])
                        lines.append(f"{p[0]} {(min_x+max_x)/2:.6f} {(min_y+max_y)/2:.6f} {max_x-min_x:.6f} {max_y-min_y:.6f}")
                    else:
                        lines.append(line.strip())
            zf.writestr(f"obj_train_data/{txt.name}", "\n".join(lines) + "\n")
        zf.writestr("train.txt", "\n".join(train_list) + "\n")
    return zip_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO+SAM 自動標註工具 (MLOps 刷新優化版)')
    parser.add_argument('--config', type=str, default=str(settings.paths.configs / 'pipeline.yaml'))
    parser.add_argument('--data', type=str, default=str(settings.paths.raw)) # load data path is in settings.py
    parser.add_argument('--det', type=str, default=None, help='推理模型路徑')
    parser.add_argument('--package', action='store_true', help='完成後自動執行 ZIP 封裝')
    
    args = parser.parse_args()
    
    # 0. 讀取 YAML 配置
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f).get('autolabel', {})
    
    # 1. 決定最終參數 (優先權: 指令列 > YAML > 自動偵測)
    raw_det = args.det or cfg.get('det_model')
    final_det = None

    if raw_det:
        # --- 智慧路徑轉換邏輯 ---
        p = Path(raw_det)
        base_exp = settings.paths.experiments
        
        # 定義可能的搜尋路徑組合
        paths_to_check = []
        if p.is_absolute():
            paths_to_check.extend([
                p,
                p / "weights" / "best.pt"
            ])
        else:
            paths_to_check.extend([
                base_exp / raw_det,
                base_exp / raw_det / "weights" / "best.pt",
                Path(raw_det),
                Path(raw_det) / "weights" / "best.pt"
            ])
        
        for check_p in paths_to_check:
            if check_p.exists() and check_p.is_file():
                final_det = str(check_p)
                break
        
        # 如果最後還是沒找到實體檔案，就保持原樣 (交給 YOLO 報錯)
        if not final_det:
            final_det = raw_det
    else:
        final_det = find_best_local_model()

    final_conf = cfg.get('conf', 0.6)
    final_imgsz = cfg.get('imgsz', 768)
    final_sam = cfg.get('sam_model', 'sam2_t.pt')

    # 2. 顯示啟動橫幅
    print("="*60)
    print(f"🚀 啟動 MLOps 自動標註流水線 (智慧路徑模式)")
    print(f"   【檢測模型】: {final_det}")
    print(f"   【影像路徑】: {args.data}")
    print(f"   【推理參數】: imgsz={final_imgsz}, conf={final_conf}")
    print("="*60)

    out_dir = run_auto_annotation_refined(
        data_dir=args.data,
        det_model=final_det,
        sam_model=final_sam,
        conf=final_conf,
        imgsz=final_imgsz
    )
    
    if out_dir:
        notes = [f"標註結果: {out_dir}"]
        if args.package:
            zip_p = create_cvat_package(out_dir / "images", out_dir / "labels")
            notes.append(f"📦 已封裝 CVAT 包: {zip_p}")
        
        print_pipeline_notice(output_paths=[str(out_dir)], next_script="src/analyze_errors.py", notes=notes)
