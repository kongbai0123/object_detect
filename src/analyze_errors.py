import fiftyone as fo
from fiftyone import ViewField as F
import os
import argparse
from datetime import datetime
from pathlib import Path
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

CLASS_NAMES = ["open", "close"]


def build_default_export_dir(dataset_dir):
    dataset_path = Path(dataset_dir)
    return str(dataset_path.parent / f"{dataset_path.name}_edited")


def build_temp_export_dir(dataset_dir):
    dataset_path = Path(dataset_dir)
    return str(dataset_path.parent / f"{dataset_path.name}_temp")


def prepare_export_dir(export_dir):
    export_path = Path(export_dir)
    if not export_path.exists() or not any(export_path.iterdir()):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        return export_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = export_path.parent / f"{export_path.name}_{timestamp}"
    alt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f" 匯出目錄已存在，改存到: {alt_path}")
    return alt_path


def export_edited_dataset(dataset, export_dir):
    target_dir = prepare_export_dir(export_dir)
    print(f" 正在匯出編輯後資料集到: {target_dir}")
    dataset.export(
        export_dir=str(target_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        export_media=True,
        classes=CLASS_NAMES,
        overwrite=True,
    )
    return str(target_dir)


def launch_fiftyone(dataset_dir, export_dir, reset_dataset=False):
    """
    MLOps Level 2 五一勘錯分析系統 (FiftyOne Advanced Analytics)
    不止於視覺化，深度結合 compute_metadata 等操作，
    直接提供聰明檢視 (Smart Views)供工程師一鍵鎖定 FP (誤標) / FN (漏抓)
    """
    print(" [Level 2] 啟動 FiftyOne 深度聚類與勘錯分析...")
    
    name = "yolov8-mlops-analysis"
    
    # [Smart Auto-Sync] 智慧診斷機制：自動偵測資料庫是否與硬碟脫鉤
    if fo.dataset_exists(name) and not reset_dataset:
        try:
            temp_ds = fo.load_dataset(name)
            stale_reason = None
            
            # 1. 檢查路徑存活性 (抽查首張圖片)
            if len(temp_ds) > 0:
                first_sample = temp_ds.first()
                if not os.path.exists(first_sample.filepath):
                    stale_reason = "硬碟路徑已變動 (File not found)"
            
            # 2. 檢查樣本數量一致性
            if not stale_reason:
                img_dir = os.path.join(dataset_dir, 'images')
                if not os.path.exists(img_dir):
                    img_dir = dataset_dir
                
                if os.path.exists(img_dir):
                    actual_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                    if len(temp_ds) != actual_count:
                        stale_reason = f"樣本數量不符 (Database: {len(temp_ds)}, Disk: {actual_count})"
            
            if stale_reason:
                print(f" 🚩 偵測到資料庫與硬碟現狀不符：{stale_reason}")
                print(" 🔄 正在自動重置並同步最新的 3_processed 黃金資料...")
                reset_dataset = True
            
            # 釋放暫存物件
            del temp_ds
        except Exception:
            reset_dataset = True

    if fo.dataset_exists(name) and reset_dataset:
        fo.delete_dataset(name)

    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        dataset.persistent = True
        dataset.save()
        print(f" 已載入既有 FiftyOne dataset: {name}")
    else:
        dataset = None

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    
    try:
        if dataset is None:
            if os.path.exists(yaml_path):
                print(f" 探測到 YOLO 格式標註，正在匯入 {yaml_path} ...")
                dataset = fo.Dataset.from_dir(
                    dataset_type=fo.types.YOLOv5Dataset,
                    dataset_dir=dataset_dir,
                    yaml_path=yaml_path,
                    name=name,
                    label_field="ground_truth" # 設定標籤對應欄位
                )
            else:
                print(" 找不到 dataset.yaml，啟動 [智慧相容掃描模式]，支援未整理的扁平資料夾（如 auto_ann）...")
                images_dir = os.path.join(dataset_dir, 'images')
                labels_dir = os.path.join(dataset_dir, 'labels')
                
                # 若 images/ 子目錄不存在，回退到根目錄
                base_images_dir = images_dir if os.path.exists(images_dir) else dataset_dir
                
                # [Smart Discovery] 如果目錄裡沒圖片但有標籤，改從標籤出發搜尋影像
                img_list = [f for f in os.listdir(base_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if not img_list:
                    print(" [提示] 目錄內無影像，嘗試從標籤檔反向追蹤原始影像池...")
                    txt_list = [f for f in os.listdir(dataset_dir) if f.endswith('.txt') and f != 'classes.txt']
                    if not txt_list and os.path.exists(labels_dir):
                        txt_list = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                    
                    if txt_list:
                        potential_pools = [
                            os.path.join(dataset_dir, "../2_filtered"),
                            os.path.join(dataset_dir, "../1_raw/door_opening_frames"),
                        ]
                        samples = []
                        for txt_file in txt_list:
                            base_name = os.path.splitext(txt_file)[0]
                            found_img = None
                            for pool in potential_pools:
                                if os.path.exists(pool):
                                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
                                        cand = os.path.join(pool, base_name + ext)
                                        if os.path.exists(cand):
                                            found_img = cand
                                            break
                                if found_img: break
                            if found_img:
                                samples.append(fo.Sample(filepath=found_img))
                        
                        dataset = fo.Dataset(name)
                        dataset.add_samples(samples)
                    else:
                        print(" [錯誤] 找不到任何影像或標籤檔，請確認路徑。")
                        return None, "error"
                else:
                    dataset = fo.Dataset.from_images_dir(images_dir, name=name)
                
                # 手動解析 YOLO txt 並加入 Bbox
                class_names = {idx: name for idx, name in enumerate(CLASS_NAMES)}
                added_labels = 0
                for sample in dataset:
                    base_name = os.path.splitext(os.path.basename(sample.filepath))[0]
                    txt_candidates = [
                        os.path.join(labels_dir, f"{base_name}.txt"),
                        os.path.join(dataset_dir, f"{base_name}.txt"),
                    ]
                    txt_path = next((p for p in txt_candidates if os.path.exists(p)), None)
                    
                    # --- [Smart Image Discovery] 智慧影像偵測 ---
                    # 如果 dataset_dir 裡面只有 .txt，自動去生肉池尋找對應的 .jpg
                    if not os.path.exists(sample.filepath):
                        potential_pools = [
                            os.path.join(dataset_dir, "../2_filtered"),
                            os.path.join(dataset_dir, "../1_raw/door_opening_frames"),
                        ]
                        for pool in potential_pools:
                            if os.path.exists(pool):
                                for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
                                    cand = os.path.join(pool, base_name + ext)
                                    if os.path.exists(cand):
                                        sample.filepath = cand
                                        break
                            if os.path.exists(sample.filepath): break

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
                                    except Exception:
                                        pass
                        if detections:
                            sample["ground_truth"] = fo.Detections(detections=detections)
                            sample.save()
                            added_labels += 1
                if added_labels > 0:
                    print(f" 成功為 {added_labels} 張圖片掛載 YOLO 標籤框！")

            dataset.persistent = True
            dataset.save()
            print(f" FiftyOne dataset 已設為 persistent: {name}")
            
    except Exception as e:
        print(f" 載入失敗，請確認資料結構: {e}")
        return None, "error"

    print(" 正在計算影像 Metadata (解析檔案大小長寬比例，嚴格防範異常資料)...")
    dataset.compute_metadata()
    print(f" 成功載入！資料庫內含 {len(dataset)} 個分析樣本")
    
    # =========================================================
    #  [Level 2 核心升級] 自動產生聰明過濾視圖 (Smart Views)
    # 這能讓你在 UI 直接點選，一秒過濾出最可能有瑕疵的 Corner Cases 圖片
    # =========================================================
    
    # 1. 揪出品質不過關的地雷圖片 (解析度嚴重過低的相片通常會破壞學習)
    low_res_view = dataset.match((F("metadata.width") < 300) | (F("metadata.height") < 300))
    dataset.save_view(" 地雷過濾：長寬小於 300px 的劣質影像", low_res_view, overwrite=True)
    
    if "ground_truth" in dataset.get_field_schema():
        # 2. 空標籤分析：這可以是用來驗證你混進去的背景學習圖是否都正確呈現為空！
        bg_view = dataset.match(F("ground_truth.detections").length() == 0)
        dataset.save_view(" 分析：無目標物件之背景圖 (Negative Samples)", bg_view, overwrite=True)
        
        # 3. 超密集場景：超過 10 個框在同一張照片這裡是最最最常發生漏標 (FN) 或誤標的地方
        dense_view = dataset.match(F("ground_truth.detections").length() >= 10)
        dataset.save_view(" 易錯分析：極度密集擁擠的場景 (超過10個物件)", dense_view, overwrite=True)

    print("\n==================  MLOps 工業級 FP / FN Mining (誤判抓補) ==================")
    print("如果您已經有了最佳模型 (best.pt)，真正的玩法是呼叫預測並進行 evaluate()：")
    print("  1. 給每張圖新增 'predictions' 欄位填入推論信心度與 Bbox")
    print("  2. dataset.evaluate_detections('predictions', gt_field='ground_truth', compute_mAP=True)")
    print("  3. 在介面上使用 dataset.match(F('false_positives') > 0) 一秒抓出把背景看錯的嚴重暇疵圖！\n")
    print("==============================================================================\n")
    
    # 啟動應用程式
    session = fo.launch_app(dataset)
    print(" 分析中心服務啟動！請開啟瀏覽器至: http://localhost:5151 進行檢視")
    print(" 在左側的Saved Views下拉選單中，可以直接切換我剛剛為您建立的過濾條件")
    session.wait()
    temp_export_dir = build_temp_export_dir(dataset_dir)

    print("\n離開 FiftyOne 後的處理方式：")
    print(f"  1. 暫存 -> 匯出到 {temp_export_dir}")
    print(f"  2. 匯出 -> 匯出到 {export_dir}")
    print("  3. 其他 -> 自訂路徑，或直接 Enter 略過")
    post_action = input("請選擇 1 / 2 / 3: ").strip()

    if post_action == "1":
        return export_edited_dataset(dataset, temp_export_dir), "temp"

    if post_action == "2":
        return export_edited_dataset(dataset, export_dir), "export"

    if post_action == "3":
        custom_dir = input("請輸入自訂匯出資料夾，直接 Enter 代表略過: ").strip()
        if custom_dir:
            return export_edited_dataset(dataset, custom_dir), "custom"

    print(" 已略過匯出，原始資料夾不會被修改。")
    return None, "skip"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FiftyOne MLOps 深度分析中心')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 預設指向自動標註輸出的目錄，以便進行快節奏的成果分析與過濾
    default_dataset = os.path.normpath(os.path.join(script_dir, '../data/5_auto_ann'))
    default_export = build_default_export_dir(default_dataset)
    
    parser.add_argument('--dir', type=str, default=default_dataset, help='包含 images 或 YOLO txt 標籤的目錄 (預設指向 5_auto_ann)')
    parser.add_argument('--export-dir', type=str, default=default_export, help='關閉 FiftyOne 後，將編輯後標註匯出的資料夾')
    parser.add_argument('--reset-dataset', action='store_true', help='重建 FiftyOne dataset，而不是沿用既有修改')
    args = parser.parse_args()
    
    exported_dir, export_mode = launch_fiftyone(args.dir, args.export_dir, reset_dataset=args.reset_dataset)
    summary_output = exported_dir if exported_dir else args.dir
    summary_notes = [
        "若在 FiftyOne 關閉後選擇暫存或匯出，ground_truth 會另存成新的 YOLO 資料集，不會覆蓋原始 val_frozen。",
        "確認 edited 資料集內容後，下一步可做 active learning、重新切分，或直接回訓。",
    ]
    if export_mode == "temp":
        summary_notes[0] = "這次已將 FiftyOne 修改暫存成新的 YOLO 資料集，原始 val_frozen 沒有被覆蓋。"
    elif export_mode == "custom":
        summary_notes[0] = "這次已將 FiftyOne 修改匯出到自訂資料夾，原始 val_frozen 沒有被覆蓋。"
    elif not exported_dir:
        summary_notes[0] = "這次未執行匯出，FiftyOne 內的修改沒有寫回原始資料夾。"

    next_script = "src/cvat_import.py" if exported_dir else "src/active_learning.py"

    if exported_dir:
        summary_notes.append(
            f"若要將這次 FiftyOne 校閱結果同步回黃金池，請執行 python src/cvat_import.py，選擇 Mode 2，並輸入：{exported_dir}"
        )
        summary_notes.append(
            "建議確認 edited 資料夾內容無誤後再執行合併，以維持 3_processed 的純淨度。"
        )

    print_pipeline_notice(
        output_paths=summary_output,
        next_script=next_script,
        notes=summary_notes,
    )
