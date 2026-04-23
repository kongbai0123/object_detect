import os
from pathlib import Path
import yaml
from anti_gravity.logger import logger

def validate_dataset(yaml_path: str):
    """
    資料集健康度檢查 (Dataset Sanity Check)
    確保在 YOLO 開始訓練前，攔截所有常見的資料層級錯誤。
    """
    logger.info(f"🔍 [Sanity Check] 開始檢查資料集完整性: {yaml_path}")
    
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise RuntimeError(f"找不到 dataset.yaml: {yaml_path}")
        
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    base_path = Path(data.get('path', str(yaml_file.parent)))
    nc = data.get('nc', 0)
    
    if nc == 0:
        raise RuntimeError("資料集設定檔缺少 'nc' (類別數量)")
        
    # 檢查 train / val 是否完全重疊
    train_p = data.get('train', '')
    val_p = data.get('val', '')
    if train_p and val_p and train_p == val_p:
        logger.error("🚨 [嚴重警告] train 與 val 指向完全相同的資料夾！這會導致 Validation Leakage！")
        # 這裡不硬性阻擋(以防有人故意為之)，但強烈警告
        
    for split in ['train', 'val']:
        split_path = data.get(split)
        if not split_path:
            continue
            
        if os.path.isabs(split_path):
            img_dir = Path(split_path)
        else:
            img_dir = base_path / split_path
            
        if not img_dir.exists():
            raise RuntimeError(f"[{split}] 找不到圖片資料夾: {img_dir}")
            
        # 推導 labels 資料夾路徑
        lbl_dir = Path(str(img_dir).replace('images', 'labels'))
        if not lbl_dir.exists():
            raise RuntimeError(f"[{split}] 找不到對應的標籤資料夾: {lbl_dir}")
            
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]
        labels = list(lbl_dir.glob('*.txt'))
        
        logger.info(f"  - {split.upper()}: 找到 {len(images)} 張圖片, {len(labels)} 個標籤檔")
        
        if len(labels) == 0:
            raise RuntimeError(f"[{split}] 標籤檔數量為 0！(這是導致 mAP=0 的主因)")
            
        empty_count = 0
        for lbl_file in labels:
            if lbl_file.stat().st_size == 0:
                empty_count += 1
                
        if empty_count > 0:
            empty_ratio = empty_count / len(labels)
            logger.info(f"    * 發現 {empty_count} 個空標籤檔 (佔 {empty_ratio:.1%})，將視為背景樣本 (Background)。")
            if empty_count == len(labels):
                raise RuntimeError(f"[{split}] 嚴重錯誤：所有標籤檔皆為空！這將導致模型無法學習任何特徵或 mAP=0。")
                
        # 抽樣檢查前 10 個「非空」標籤檔 (檢查 BOM, class 越界, bbox 格式)
        non_empty_labels = [f for f in labels if f.stat().st_size > 0]
        for lbl_file in non_empty_labels[:10]:
            with open(lbl_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('\ufeff'):
                    raise RuntimeError(f"[{split}] 標籤檔包含 UTF-8 BOM 亂碼: {lbl_file.name}")
                
                for line in content.splitlines():
                    parts = line.strip().split()
                    if not parts: continue
                    if len(parts) != 5:
                        raise RuntimeError(f"[{split}] BBox 欄位數量不為 5: {lbl_file.name} -> {line}")
                        
                    cls_id = int(float(parts[0]))
                    if cls_id < 0 or cls_id >= nc:
                        raise RuntimeError(f"[{split}] 類別索引 {cls_id} 越界 (應該在 0~{nc-1}): {lbl_file.name}")
                        
                    for coord in parts[1:]:
                        val_coord = float(coord)
                        if val_coord < 0.0 or val_coord > 1.0:
                            raise RuntimeError(f"[{split}] 座標 {val_coord} 超出 [0, 1] 範圍: {lbl_file.name}")
                            
    logger.info("✅ [Sanity Check] 資料集檢查通過！格式合法，允許包含背景樣本。")
    return True

def purge_empty_labels(yaml_path: str):
    """
    資料集淨化：從訓練集中物理刪除所有空標籤及其對應的圖片。
    """
    logger.info(f"🧹 [Dataset Purge] 開始清理空標籤樣本: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    base_path = Path(data.get('path', ''))
    train_path = data.get('train')
    
    if not train_path:
        logger.warning(" [Purge] 找不到 train 路徑，跳過。")
        return
        
    if os.path.isabs(train_path):
        img_dir = Path(train_path)
    else:
        img_dir = base_path / train_path
        
    lbl_dir = Path(str(img_dir).replace('images', 'labels'))
    if not lbl_dir.exists():
        return

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    labels = list(lbl_dir.glob('*.txt'))
    
    purge_count = 0
    for lbl_file in labels:
        if lbl_file.stat().st_size == 0:
            # 尋找對應的圖片
            stem = lbl_file.stem
            for ext in img_exts:
                img_file = img_dir / f"{stem}{ext}"
                if img_file.exists():
                    try:
                        os.remove(img_file)
                        os.remove(lbl_file)
                        purge_count += 1
                        break
                    except Exception as e:
                        logger.error(f"無法刪除檔案 {stem}: {e}")
    
    if purge_count > 0:
        logger.info(f"✨ [Purge] 成功剔除了 {purge_count} 個背景樣本！現在訓練集僅包含有標註的圖片。")
    else:
        logger.info(" [Purge] 檢查完畢，沒有發現需要剔除的空標籤。")
