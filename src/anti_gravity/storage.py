import os
import shutil
from pathlib import Path
from typing import List, Set
from anti_gravity.entities import ImageMetadata, BoxInfo, ClassID
from anti_gravity.settings import settings
from anti_gravity.logger import logger
from tqdm import tqdm

class DatasetStorage:
    """
    🟢 基礎設施層：檔案系統實作
    🔴 SRP：只負責與硬碟 I/O 與資料格式解析相關的任務。
    """

    def __init__(self, min_area: float = 0.00005, min_dim: float = 0.008):
        self.min_area = min_area
        self.min_dim = min_dim

    def parse_yolo_label(self, label_path: Path) -> List[BoxInfo]:
        """
        解析 YOLO 格式標籤。
        """
        boxes = []
        if not label_path.exists():
            return boxes

        try:
            with open(label_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(parts[0])
                    # Robust Clipping with safe margin for floating point precision
                    eps = 1e-6
                    cx, cy, w, h = [max(eps, min(1.0 - eps, float(x))) for x in parts[1:5]]
                    
                    area = w * h
                    if area < self.min_area or min(w, h) < self.min_dim:
                        continue
                    
                    is_near_edge = cx < 0.1 or cx > 0.9 or cy < 0.1 or cy > 0.9
                    
                    boxes.append(BoxInfo(
                        cls_id=ClassID(cls_id),
                        area=area,
                        cx=cx, cy=cy, w=w, h=h,
                        is_near_edge=is_near_edge
                    ))
        except Exception as e:
            logger.error(f"[Storage] Label parse error {label_path}: {e}")
            
        return boxes

    def scan_directories(self, input_dirs: List[Path], error_stems: Set[str] = None) -> List[ImageMetadata]:
        """
        遞迴掃描多個目錄並轉換為 Domain Entities。
        """
        from anti_gravity.utils import extract_scene_key
        error_stems = error_stems or set()
        metadata_list = []
        
        # 🟢 遞迴尋找所有包含 'images' 子目錄的資料夾
        valid_img_dirs = []
        for d in input_dirs:
            if not d.exists(): continue
            if (d / "images").exists():
                valid_img_dirs.append(d / "images")
            else:
                # 如果主目錄沒有 images，則向下搜尋所有名為 images 的子資料夾
                valid_img_dirs.extend(list(d.rglob("images")))
        
        for img_dir in valid_img_dirs:
            # 掃描目前的影像目錄 (使用 tqdm 並在完成後清除)
            all_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            for img_path in tqdm(all_imgs, desc=f"[Storage] Scanning {img_dir.parent.name}", leave=False):
                # 智能標籤定位：對齊 images 同級的 labels
                lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                
                boxes = self.parse_yolo_label(lbl_path)
                
                metadata_list.append(ImageMetadata(
                    path=img_path,
                    label_path=lbl_path if lbl_path.exists() else None,
                    scene=extract_scene_key(img_path.name),
                    boxes=boxes,
                    is_error=img_path.stem in error_stems
                ))
        
        logger.info(f"[Storage] Scan completed. Loaded {len(metadata_list)} image records.")
        return metadata_list

    def deploy_dataset(self, selected_metadata: List[ImageMetadata], output_dir: Path):
        """
        執行物理搬移 (Deployment)。
        """
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"
        
        # 🟣 穩定性：Idempotency
        for d in [out_img, out_lbl]:
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Storage] Deploying balanced dataset to: {output_dir}")
        
        for meta in tqdm(selected_metadata, desc="[Storage] Deploying", leave=False):
            # 複製影像
            shutil.copy2(meta.path, out_img / meta.path.name)
            
            # 複製標籤 (或生成空白以防背景圖洩露)
            if meta.label_path and meta.label_path.exists():
                shutil.copy2(meta.label_path, out_lbl / meta.label_path.name)
            else:
                (out_lbl / f"{meta.stem}.txt").touch() # Empty for background

        logger.info(f"[Storage] Deployment finished. Total: {len(selected_metadata)} images.")
