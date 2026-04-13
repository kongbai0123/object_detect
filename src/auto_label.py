import os
import argparse
# Ultralytics 內建支援 YOLO + SAM 自動標註功能
from ultralytics.data.annotator import auto_annotate
import glob as _glob
import zipfile
import yaml
from pathlib import Path
from datetime import datetime
from pipeline_notice import print_pipeline_notice
def run_auto_annotation(data_dir, det_model='yolov8x.pt', sam_model='mobile_sam.pt', conf=0.6, iou=0.5):
    """
    半自動標註引擎：
    1. 使用大型的 YOLO 預訓練模型 (或特定目標的模型) 找出物件預選區域
    2. 信心度與 NMS 過濾器 
    3. 使用 SAM (Segment Anything Model) 或 MobileSAM 在預選區內拉出極為精準的 Bounding Box / Mask
    """
    print(" [半自動化流程 1/3]: 開始執行 SAM + YOLO 自動標註...")
    print(f" 影像來源: {data_dir}")
    print(f" 預選框生成 (YOLO): {det_model}")
    print(f" 精細化分割 (SAM): {sam_model}")
    print(f" 過濾閘門: Confidence Threshold = {conf}, NMS IoU = {iou} (防 Dataset Collapse)")
    
    # 傳入 conf 與 iou 以免 YOLO 初步抓取時產生過多破碎/堆疊垃圾框
    auto_annotate(data=data_dir, det_model=det_model, sam_model=sam_model, conf=conf, iou=iou)
    
    import shutil
    auto_out_dir = data_dir + '_auto_annotate_labels'
    # 修正：直接定位到專案根目錄下的 data/4_auto_ann
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_out_dir = os.path.normpath(os.path.join(script_dir, '../data/5_auto_ann'))
    
    if os.path.exists(auto_out_dir):
        if os.path.exists(final_out_dir):
            shutil.rmtree(final_out_dir)
        shutil.move(auto_out_dir, final_out_dir)
    
    print("\n 自動標註程序已完成！")
    print(f" 標籤檔已經移動至指定的存放目錄: {final_out_dir}")
    return final_out_dir

def create_cvat_package(img_dir, lbl_dir, zip_name=None):
    """
    將標籤打包為 CVAT 專用的 Darknet 格式 (YOLO 1.1)
    與 hard.zip 範例 1:1 對位 (僅上傳標籤用)
    """
    img_path = Path(img_dir)
    lbl_path = Path(lbl_dir)
    
    # 專案根目錄 C:\antigravity\
    root_dir = Path(__file__).resolve().parent.parent
    
    # 建構歸檔目錄 5_auto_ann_zip
    archive_subdir = root_dir / "data/5_auto_ann_zip"
    archive_subdir.mkdir(parents=True, exist_ok=True)
    
    # === 1. 執行舊有壓縮檔的自動歸檔 ===
    import shutil
    old_zips = root_dir.glob("cvat_auto_ann_*.zip")
    for oz in old_zips:
        dest = archive_subdir / oz.name
        try:
            shutil.move(str(oz), str(dest))
            print(f" [歸檔管理] 已將先前的封裝包移至: {dest}")
        except Exception as e:
            print(f" [歸檔管理] 警告: 搬移 {oz.name} 失敗 ({e})")
    
    # === 2. 建立新的 ZIP 檔至根目錄 ===
    if zip_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        zip_name = f"cvat_auto_ann_{stamp}.zip"
    else:
        zip_name = os.path.basename(zip_name)
    
    zip_out = root_dir / zip_name
    print(f"\n [打包程序] 正在建立 Darknet 標籤包: {zip_out}")
    
    # === 3. 讀取 dataset.yaml 獲取類別定義 ===
    dataset_yaml = lbl_path.parent / "dataset.yaml"
    class_names = ["open", "close"] # Fallback
    if dataset_yaml.exists():
        try:
            with open(dataset_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                names_dict = data.get('names', {})
                if names_dict:
                    sorted_keys = sorted(names_dict.keys(), key=lambda x: int(x))
                    class_names = [names_dict[i] for i in sorted_keys]
        except Exception:
            pass

    # === 4. 蒐集目前生成的標籤檔 ===
    txt_files = list(lbl_path.glob("*.txt"))
    if not txt_files:
        print(" 警告: 找不到任何標籤檔，取消打包。")
        return None

    # === 5. 執行 CVAT 相容的 Darknet 格式封裝 ===
    count = 0
    with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
        # A. 生成 obj.names (純類別名，無 ID)
        zf.writestr("obj.names", "\n".join(class_names) + "\n")
        
        # B. 生成 obj.data
        obj_data = f"classes = {len(class_names)}\ntrain = train.txt\nnames = obj.names\n"
        zf.writestr("obj.data", obj_data)
        
        # C. 生成 train.txt 並將檔案壓入 obj_train_data/ (第一層)
        train_list = []
        for txt in txt_files:
            img_base = txt.stem
            
            # 從 img_path 尋找實際的副檔名 (若找不到，預設為 .jpg)
            ext = ".jpg"
            for test_ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
                if (img_path / f"{img_base}{test_ext}").exists():
                    ext = test_ext
                    break
                    
            # 必須為 obj_train_data/xxx.ext，不可含 data/，因為這是在 ZIP 內的路徑關係
            train_list.append(f"obj_train_data/{img_base}{ext}")
            
            # --- 核心修復：動態攔截並降維 Polygon 為 Tight Bounding Box ---
            converted_lines = []
            try:
                with open(txt, 'r', encoding='utf-8') as tf:
                    for line in tf:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # 如果超過 5 個元素，代表是 Segmentation Polygon (class x1 y1 x2 y2...)
                        if len(parts) > 5:
                            class_id = parts[0]
                            try:
                                coords = [float(x) for x in parts[1:]]
                                xs = coords[0::2]
                                ys = coords[1::2]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                
                                # 轉回標準中心點座標 (cx, cy, w, h)
                                cx = (min_x + max_x) / 2.0
                                cy = (min_y + max_y) / 2.0
                                w = max_x - min_x
                                h = max_y - min_y
                                converted_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            except ValueError:
                                # 若解析失敗，原樣保留
                                converted_lines.append(line.strip())
                        else:
                            # 已經是標準 5 元素的 Bounding Box，直接放行
                            converted_lines.append(line.strip())
            except Exception as e:
                print(f" 警告: 處理標籤 {txt.name} 失敗 ({e})")
                continue
            
            # 將轉換後的「純淨邊界框內容」寫入 ZIP
            zf.writestr(f"obj_train_data/{txt.name}", "\n".join(converted_lines) + "\n")
            count += 1
            
        zf.writestr("train.txt", "\n".join(train_list) + "\n")
    
    print(f" 打包完成！共封裝 {count} 份標籤，結構 100% 符合 CVAT YOLO 1.1 規範。")
    print(f" 👉 封裝包已儲存於專案根目錄: {zip_out}")
    return zip_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO + SAM 半自動標註腳本')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 預設指向抽幀後的生肉圖片目錄
    # 預設指向經過 CLIP 語意粗篩後的資料夾 (open/ 類別)
    default_data = os.path.normpath(os.path.join(script_dir, '../data/2_filtered/open'))
    
    # 自動尋找 data/7_experiments/ 底下最新的 best.pt (包含 exp{num}/weights/best.pt) 作為預設大腦
    
    # 優先搜尋 global_best.pt，若無則搜尋 7_experiments 底下所有 exp{num}/weights/best.pt
    runs_dir = os.path.normpath(os.path.join(script_dir, '../data/7_experiments'))

    global_best = os.path.join(runs_dir, 'weight', 'global_best.pt')
    if os.path.exists(global_best):
        default_model = global_best
    else:
        import glob as _glob
        found = _glob.glob(os.path.join(runs_dir, 'exp*', 'weights', 'best.pt'), recursive=True)
        # 過濾掉分類器 (cls_*) 的權重，並按修改時間排序
        det_cands = [p for p in found if 'cls_' not in p.replace('\\', '/')]
        det_cands.sort(key=os.path.getmtime)
        default_model = det_cands[-1] if det_cands else 'yolov8x.pt'
    # 過濾掉分類器 (cls_*) 的權重，否則 auto_annotate 會因為沒看到 boxes 而報錯
    
    parser.add_argument('--data', type=str, default=default_data, help='需要自動標註的未標籤生肉圖庫')
    # 改為預設使用您自己練出來的 best.pt
    parser.add_argument('--det', type=str, default=default_model, help='負責提供 Bounding Box 的目標偵測模型')
    # 使用 MobileSAM 或是 sam_b.pt 皆可
    parser.add_argument('--sam', type=str, default='mobile_sam.pt', help='負責分割物體輪廓以確保 box 吸附真實邊緣的 SAM 模型')
    
    # [Dataset Feedback Collapse 防護] 強制過濾器
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence Threshold：過濾沒把握的預選框 (建議 0.5~0.6)')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU Threshold：去除重疊與抖動框 (建議 0.45~0.5)')
    
    args = parser.parse_args()
    
    # 防呆機制：檢查資料夾是否存在且有圖片
    valid_images = []
    if os.path.exists(args.data):
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            valid_images.extend(_glob.glob(os.path.join(args.data, f'*{ext}')))
            
    if not valid_images:
        print(f" 錯誤: '{args.data}' 裡面沒有找到任何圖片！")
        print(" 請確認您的未標註生肉圖片是放在哪裡，並使用 --data 指定該資料夾，或把圖放回預設路徑")
        exit(1)
    
    out_ann_dir = run_auto_annotation(
        data_dir=args.data, 
        det_model=args.det, 
        sam_model=args.sam,
        conf=args.conf,
        iou=args.iou
    )
    
    print("\n" + "="*45)
    print(" [互動選項] 自動標註後續動作")
    print("="*45)
    print(" [1] 立即打包資料 (建立 CVAT 一鍵匯入 ZIP)")
    print(" [2] 結束不執行")
    
    choice = input("\n 請選擇執行代碼 (1 或 2): ").strip()
    
    notes = [
        "此步驟輸出的是自動標註結果，建議人工複核後再納入正式資料集。",
        "若框品質不穩，優先檢查 --det 權重與來源影像品質。",
    ]
    
    if choice == '1':
        zip_file = create_cvat_package(args.data, out_ann_dir)
        if zip_file:
            notes.append(f"📦 已打包完成：{zip_file}")
            notes.append("💡 下一步：執行 python src/cvat_import.py [Mode 1] 啟動伺服器，並手動匯入此 ZIP。")
    
    print_pipeline_notice(
        output_paths=out_ann_dir,
        next_script="src/cvat_import.py",
        notes=notes,
    )
