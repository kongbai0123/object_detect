import random
from typing import List, Dict, Any, Tuple
import albumentations as A
import cv2
import numpy as np
from anti_gravity.entities import ImageMetadata, ClassID, BoxInfo

class Augmenter:
    """
    🟡 應用邏輯層 (Use Case)：離線增強核心
    封裝多套 Albumentations 變換策略。
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        # 🟢 加入防護校驗，並允許微幅溢出後的自動裁剪
        self.bbox_params = A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'], 
            min_visibility=0.3
        )
        self._init_profiles()

    def _init_profiles(self):
        """
        🔵 策略模式：定義不同情境的增強方案。
        """
        # 修正 ImageCompression 的參數警告
        self.profiles = {
            "mixed_boundary": [
                A.Compose([
                    A.HorizontalFlip(p=0.5), 
                    A.RandomBrightnessContrast(p=0.5)
                ], bbox_params=self.bbox_params)
            ],
            "small_object_open": [
                A.Compose([
                    A.Affine(scale=(0.8, 1.2), p=1.0), 
                    A.MotionBlur(p=0.3)
                ], bbox_params=self.bbox_params)
            ],
            "open_normal": [
                A.Compose([
                    A.Affine(scale=(0.8, 1.1), p=0.7), 
                    A.RandomBrightnessContrast(p=0.5)
                ], bbox_params=self.bbox_params)
            ],
            "close_fp_like": [
                A.Compose([
                    A.Blur(blur_limit=3, p=1.0),
                    A.RandomGamma(p=0.5)
                ], bbox_params=self.bbox_params)
            ],
            "hard_negative": [
                A.Compose([
                    A.ISONoise(p=1.0), 
                    A.GaussianBlur(p=0.5)
                ], bbox_params=self.bbox_params)
            ],
            "augment_fn_open": [
                A.Compose([
                    A.Affine(scale=(0.5, 1.5), translate_percent=0.2, rotate=(-10, 10), p=1.0),
                    A.RandomBrightnessContrast(p=0.5)
                ], bbox_params=self.bbox_params)
            ]
        }

    def select_profile(self, img: ImageMetadata) -> Tuple[str, int]:
        has_open = img.open_cnt > 0
        has_close = img.close_cnt > 0
        
        # 預設為純背景圖
        profile = "hard_negative"
        multiplier = 1

        if has_open and getattr(img, 'is_type_a', False):
            # Stage 5.2 特權增強：針對 Type A (完全無框) 的強效幾何變換
            profile, multiplier = "augment_fn_open", 5
        elif has_open and has_close:
            # 特權增強：有開有關，改為 5 倍繁殖 (原本 15 倍太過火)
            profile, multiplier = "mixed_boundary", 5
        elif has_open:
            # 特權增強：純開門，改為 5 倍繁殖
            min_area = min([b.area for b in img.boxes if b.cls_id == ClassID.OPEN])
            if min_area < self.settings.get('open_hard_threshold', 0.05):
                profile, multiplier = "small_object_open", 5
            else:
                profile, multiplier = "open_normal", 5
        elif has_close:
            # 剝奪特權：純關門，樣本已過剩，強制 1 倍 (原圖直出，不產生額外增強)
            profile, multiplier = "close_fp_like", 1

        return profile, multiplier

    def apply(self, image_np: np.ndarray, metadata: ImageMetadata, profile_name: str):
        if profile_name not in self.profiles:
            return None
            
        transform = random.choice(self.profiles[profile_name])
        
        # 🟣 穩定性強化：手動校準 YOLO 座標，防止浮點數溢出導致 albumentations 崩潰
        bboxes = []
        for b in metadata.boxes:
            # 將 YOLO 轉回 min/max 進行強制限制
            x_min = max(0.0001, b.cx - b.w / 2)
            y_min = max(0.0001, b.cy - b.h / 2)
            x_max = min(0.9999, b.cx + b.w / 2)
            y_max = min(0.9999, b.cy + b.h / 2)
            
            # 再轉回 YOLO 格式
            w_new = x_max - x_min
            h_new = y_max - y_min
            cx_new = x_min + w_new / 2
            cy_new = y_min + h_new / 2
            bboxes.append([cx_new, cy_new, w_new, h_new])
            
        labels = [int(b.cls_id) for b in metadata.boxes]
        
        try:
            transformed = transform(image=image_np, bboxes=bboxes, class_labels=labels)
            return transformed
        except Exception as e:
            from anti_gravity.infrastructure.logger import logger
            logger.error(f"[Augmenter] Transformation failed for {metadata.path}: {e}")
            return None
