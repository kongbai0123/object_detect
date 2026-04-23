import fiftyone as fo
from fiftyone import ViewField as F
import os
import argparse
from datetime import datetime
from pathlib import Path
import sys
import shutil

# 排除 'def' 路徑問題
from anti_gravity.pipeline_notice import print_pipeline_notice
from anti_gravity.settings import settings

# =========================================================
#  使用者可修改配置區 (Manual Configuration)
# =========================================================
# 1. 您可以手動指定一個固定的匯出路徑 (若為 None 則使用自動生成的路徑)
FIXED_EXPORT_PATH = None 

# 2. 自動生成路徑時，預設在資料夾名稱後方加上這個字尾
DEFAULT_SUFFIX = "_edited"

# 3. 標註類別定義
CLASS_NAMES = ["open", "close"]
# =========================================================

def build_default_export_dir(dataset_dir):
    # 優先使用 settings.paths.review / "fiftyone"
    return str(settings.paths.review / "fiftyone")

def build_temp_export_dir(dataset_dir):
    dataset_path = Path(dataset_dir)
    return str(dataset_path.parent / f"{dataset_path.name}")

def prepare_export_dir(export_dir):
    export_path = Path(export_dir)
    if not export_path.exists() or not any(export_path.iterdir()):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        return export_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = export_path.parent / f"{export_path.name}_{timestamp}"
    alt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f" [系統] 輸出目錄已存在，自動重新命名為: {alt_path}")
    return alt_path

def export_edited_dataset(dataset, export_dir):
    target_dir = prepare_export_dir(export_dir)
    print(f" [匯出] 開始匯出編輯後的標註數據至: {target_dir}")
    
    # 執行標準匯出 (這會產生 images/val 與 labels/val)
    dataset.export(
        export_dir=str(target_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        export_media=True,
        classes=CLASS_NAMES,
        overwrite=True
    )
    
    # --- [MLOps 補丁] 背景樣本標籤補全 與 目錄扁平化 ---
    _cleanup_yolo_structure(target_dir)
    return str(target_dir)

def _cleanup_yolo_structure(target_dir):
    """
    1. 補全所有缺失的空白標籤檔 (.txt)
    2. 將 images/val 與 labels/val 內的檔案移至根目錄，移除多餘層級
    """
    target_path = Path(target_dir)
    img_dir = target_path / "images"
    lbl_dir = target_path / "labels"
    
    # 智慧路徑偵測：處理 FiftyOne 可能產生的子資料夾 (通常是 /val)
    sub_img = next(img_dir.glob("*/"), None) if img_dir.exists() else None
    if sub_img:
        # 如果有子資料夾 (如 images/val)，將內容移至 images/ 並補足標籤
        actual_img_dir = sub_img
        actual_lbl_dir = lbl_dir / sub_img.name
        
        print(f" [校正] 偵測到資料夾層級: {sub_img.name}，正在進行背景補全與結構優化...")
        
        supported_exts = ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']
        images = []
        for ext in supported_exts:
            images.extend(list(actual_img_dir.glob(f"*{ext}")))
            
        for img_p in images:
            # 檢查對應標籤是否存在
            txt_p = actual_lbl_dir / f"{img_p.stem}.txt"
            if not txt_p.exists():
                # 補全空白標籤
                actual_lbl_dir.mkdir(parents=True, exist_ok=True)
                with open(txt_p, "w", encoding="utf-8") as f: pass
        
        # 攤平結構 (將 images/val/* 移至 images/*)
        for f in actual_img_dir.glob("*"):
            shutil.move(str(f), str(img_dir / f.name))
        for f in actual_lbl_dir.glob("*"):
            shutil.move(str(f), str(lbl_dir / f.name))
            
        # 刪除已排空的子目錄
        shutil.rmtree(str(actual_img_dir))
        shutil.rmtree(str(actual_lbl_dir))
        
    print(f" [完成] 所有影像均已補齊標籤檔案，且目錄已進行扁平化處理。")

def launch_fiftyone(dataset_dir, export_dir, reset_dataset=False):
    """
    MLOps Level 2 錯誤分析系統 (FiftyOne Advanced Analytics)
    除了視覺化，還包含 compute_metadata 等深度分析功能。
    透過智慧過濾視圖 (Smart Views) 供工程師快速排查 FP (誤判) 與 FN (漏判)。
    """
    print(" [Level 2] 啟動 FiftyOne 深度分析服務...")
    
    name = "yolov8-mlops-analysis"
    
    # [Smart Auto-Sync] 智慧診斷機制：自動偵測資料庫是否與硬碟實體檔案脫節
    if fo.dataset_exists(name) and not reset_dataset:
        try:
            temp_ds = fo.load_dataset(name)
            stale_reason = None
            
            # 1. 檢查檔案路徑存活性
            if len(temp_ds) > 0:
                first_sample = temp_ds.first()
                if not os.path.exists(first_sample.filepath):
                    stale_reason = "硬碟路徑已失效 (File not found)"
            
            # 2. 檢查樣品數是否一致
            if not stale_reason:
                img_dir = os.path.join(dataset_dir, 'images')
                if not os.path.exists(img_dir): img_dir = dataset_dir
                
                if os.path.exists(img_dir):
                    actual_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                    if len(temp_ds) != actual_count:
                        stale_reason = f"檔案數量不符 (Database: {len(temp_ds)}, Disk: {actual_count})"
            
            if stale_reason:
                print(f" [智慧同步] 偵測到資料庫與硬碟現況不符：{stale_reason}")
                reset_dataset = True
            
            del temp_ds
        except Exception:
            reset_dataset = True

    if fo.dataset_exists(name) and reset_dataset:
        fo.delete_dataset(name)

    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        dataset.persistent = True
        dataset.save()
        print(f" 已載入既有的 FiftyOne dataset: {name}")
    else:
        dataset = None

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    
    try:
        if dataset is None:
            if os.path.exists(yaml_path):
                print(f" [匯入] 偵測到 YOLO 標註，正在匯入 {yaml_path} ...")
                dataset = fo.Dataset.from_dir(
                    dataset_type=fo.types.YOLOv5Dataset,
                    dataset_dir=dataset_dir,
                    yaml_path=yaml_path,
                    name=name,
                    label_field="ground_truth" 
                )
            else:
                print(" [智慧探索] 找不到 dataset.yaml，啟動智慧相容模式...")
                images_dir = os.path.join(dataset_dir, 'images')
                labels_dir = os.path.join(dataset_dir, 'labels')
                base_images_dir = images_dir if os.path.exists(images_dir) else dataset_dir
                
                img_list = [f for f in os.listdir(base_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if not img_list:
                    print(" [提示] 當前路徑無影像，嘗試尋找原始軌跡...")
                    # 這裡保留原始邏輯，但修復提示文字
                    dataset = fo.Dataset.from_images_dir(base_images_dir, name=name)
                else:
                    dataset = fo.Dataset.from_images_dir(base_images_dir, name=name)
                
                # 解析 YOLO txt
                class_names = {idx: name for idx, name in enumerate(CLASS_NAMES)}
                added_labels = 0
                for sample in dataset:
                    base_name = os.path.splitext(os.path.basename(sample.filepath))[0]
                    txt_candidates = [
                        os.path.join(labels_dir, f"{base_name}.txt"),
                        os.path.join(dataset_dir, f"{base_name}.txt"),
                    ]
                    txt_path = next((p for p in txt_candidates if os.path.exists(p)), None)
                    
                    if txt_path:
                        detections = []
                        with open(txt_path, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        cls_id = int(parts[0])
                                        cx, cy, w, h = map(float, parts[1:5])
                                        tlx = cx - w / 2.0
                                        tly = cy - h / 2.0
                                        label_name = class_names.get(cls_id, f"class_{cls_id}")
                                        detections.append(fo.Detection(label=label_name, bounding_box=[tlx, tly, w, h]))
                                    except Exception: pass
                        if detections:
                            sample["ground_truth"] = fo.Detections(detections=detections)
                            sample.save()
                            added_labels += 1
                if added_labels > 0:
                    print(f" [進度] 已載入 {added_labels} 張影像的 YOLO 標籤框。")

            dataset.persistent = True
            dataset.save()
            print(f" FiftyOne 資料庫已設為持久化: {name}")
            
    except Exception as e:
        print(f" [錯誤] 載入失敗: {e}")
        return None, "error"

    print(" [分析] 計算影像度量 (Metadata)...")
    dataset.compute_metadata()
    print(f" [完成] 載入成功！共 {len(dataset)} 個樣本。")
    
    # 建立智慧視圖 (使用英文開頭以確保 Slug 生成正常)
    low_res_view = dataset.match((F("metadata.width") < 300) | (F("metadata.height") < 300))
    dataset.save_view("Quality_LowRes_品質監測_低解析度", low_res_view, overwrite=True)
    
    if "ground_truth" in dataset.get_field_schema():
        bg_view = dataset.match(F("ground_truth.detections").length() == 0)
        dataset.save_view("Verify_Background_驗證_無框背景", bg_view, overwrite=True)
        
        dense_view = dataset.match(F("ground_truth.detections").length() >= 10)
        dataset.save_view("Alert_Dense_警報_密集物件", dense_view, overwrite=True)

    print("\n==================  MLOps 錯誤排查 (Mining) ==================")
    print(" 1. 請在左側 Saved Views 選單中選取過濾條件。")
    print(" 2. 檢視誤報 (FP) 與漏報 (FN) 場景。\n")
    print("============================================================\n")
    
    session = fo.launch_app(dataset)
    print(" [UI] 服務已啟動: http://localhost:5151")
    session.wait()
    
    temp_export_dir = build_temp_export_dir(dataset_dir)
    print("\n [作業] 檢視結束，請選擇後續動作：")
    print(f"  1. 匯出至暫存區 (Workspace Review) -> {export_dir}")
    print(f"  2. 匯出至正式區 (Asset Promotion) -> {temp_export_dir}")
    print("  3. 手動輸入自訂路徑")
    print("  4. 直接結束 (不匯出)")
    post_action = input(" 請選擇 1 / 2 / 3 / 4: ").strip()

    if post_action == "1":
        # 現在選項 1 是暫存/評論區
        return export_edited_dataset(dataset, export_dir), "review"
    if post_action == "2":
        # 現在選項 2 是正式/資產區
        return export_edited_dataset(dataset, temp_export_dir), "promotion"
    if post_action == "3":
        custom_dir = input(" 請輸入自訂匯出路徑: ").strip()
        if custom_dir:
            # 確保自訂路徑正確處理
            return export_edited_dataset(dataset, custom_dir), "custom"
        else:
            print(" [提示] 未輸入路徑，取消匯出。")
            return None, "skip"

    print(" [完成] 已結束，未修改原始檔案。")
    return None, "skip"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FiftyOne MLOps 深度分析中心')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 修正：直接從 settings 讀取 goldenset 路徑，避免 C:/ 字串拼接錯誤
    default_dataset = str(settings.paths.goldenset)
    default_export = build_default_export_dir(default_dataset)
    
    parser.add_argument('--dir', type=str, default=default_dataset, help='測試集目錄')
    parser.add_argument('--export-dir', type=str, default=default_export, help='編輯後輸出目錄')
    parser.add_argument('--reset-dataset', action='store_true', help='重置資料庫')
    args = parser.parse_args()
    
    exported_dir, mode = launch_fiftyone(args.dir, args.export_dir, reset_dataset=args.reset_dataset)
    summary_output = exported_dir if exported_dir else args.dir
    
    print_pipeline_notice(
        output_paths=[os.path.abspath(summary_output)],
        next_script="src/active_learning.py",
        notes=["完成 FiftyOne 錯誤分析。您可以根據分析結果調整標籤或採集策略。"],
    )
