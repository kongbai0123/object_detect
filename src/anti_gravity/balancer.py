import random
from typing import List, Dict, Optional
from collections import defaultdict
from anti_gravity.entities import ImageMetadata, ClassID, BoxInfo

class Balancer:
    """
    🟡 應用邏輯層 (Use Case)：抽樣平衡核心
    🟢 可測試性：Input/Output Separation (不涉及 I/O)
    """

    def __init__(
        self,
        target_ratio: float = 2.0,
        scene_cap: int = 2,
        max_bg_ratio: float = 0.2,
        error_bonus: float = 10.0,
        edge_penalty: float = -0.6
    ):
        self.target_ratio = target_ratio
        self.scene_cap = scene_cap
        self.max_bg_ratio = max_bg_ratio
        self.error_bonus = error_bonus
        self.edge_penalty = edge_penalty

    def calculate_score(self, img: ImageMetadata) -> float:
        """
        🔵 可擴充性：Strategy Pattern (評分機制)
        計算影像的優先權權重。
        """
        score = 10.0
        
        # 1. 錯誤清單加成 (Traceability)
        if img.is_error:
            score += self.error_bonus
            
        # 2. 邊緣懲罰 (Preventing boundary bias)
        if any(b.is_near_edge for b in img.boxes if b.cls_id == ClassID.CLOSE):
            score += self.edge_penalty
            
        return max(0.1, score)

    def run(self, input_metadata: List[ImageMetadata]) -> List[ImageMetadata]:
        """
        執行平衡演算法。
        """
        # --- 1. 資料分流 (Flow Isolation) ---
        kept_images = [] # 包含 Open 的全留 (或 Mixed)
        close_only = []
        hard_negatives = []
        
        scene_counts = defaultdict(int)
        
        # 載入 Strict FN 名單 (Stage 5.2 - 只抓 Type A)
        import json
        from pathlib import Path
        from anti_gravity.settings import settings
        fn_list_path = settings.paths.artifacts / "evaluations/fn_mining_v2/fn_classification.json"
        fn_images = set()
        if fn_list_path.exists():
            try:
                with open(fn_list_path, 'r', encoding='utf-8') as f:
                    fn_data = json.load(f)
                    for item in fn_data.get("Type_A", []):
                        fn_images.add(item['image_name'])
            except Exception as e:
                print(f"無法載入 FN 名單: {e}")

        for img in input_metadata:
            o_cnt = img.open_cnt
            c_cnt = img.close_cnt
            img_name = img.path.name
            
            if o_cnt > 0:
                # 階段二：定向增樣 (Type A FN x4, Hard Open x3, Normal Open x1.5)
                if img_name in fn_images:
                    kept_images.extend([img] * 4)
                    scene_counts[img.scene] += 4
                    img.is_type_a = True # 標記為 Type A 給 augmenter 使用
                elif img.is_error:
                    kept_images.extend([img] * 3)
                    scene_counts[img.scene] += 3
                else:
                    kept_images.append(img)
                    scene_counts[img.scene] += 1
                    if random.random() < 0.5:
                        kept_images.append(img)
                        scene_counts[img.scene] += 1
            elif c_cnt > 0:
                close_only.append(img)
            else:
                hard_negatives.append(img)

        # --- 2. 核心計算：目標 Close Box 數 ---
        total_open_boxes = sum(img.open_cnt for img in kept_images)
        target_close_boxes = total_open_boxes * self.target_ratio
        current_close_boxes = sum(img.close_cnt for img in kept_images)
        
        needed_close_boxes = target_close_boxes - current_close_boxes
        
        # --- 3. 抽樣 Close-only 樣本 ---
        sampled_close = []
        if needed_close_boxes > 0 and close_only:
            # 根據評分排序
            scored_close = [(self.calculate_score(img), img) for img in close_only]
            # 隨機加權排序 (引入隨機性避免過擬合)
            scored_close.sort(key=lambda x: x[0] * random.random(), reverse=True)
            
            for score, img in scored_close:
                if needed_close_boxes <= 0:
                    break
                
                # 場景治理 (Scene Cap)
                if scene_counts[img.scene] >= self.scene_cap:
                    continue
                    
                sampled_close.append(img)
                scene_counts[img.scene] += 1
                needed_close_boxes -= img.close_cnt

        # --- 4. 抽樣背景樣本 (Max BG Ratio) ---
        sampled_bg = []
        bg_cap = int(len(kept_images + sampled_close) * self.max_bg_ratio)
        if hard_negatives and bg_cap > 0:
            random.shuffle(hard_negatives)
            sampled_bg = hard_negatives[:bg_cap]

        return kept_images + sampled_close + sampled_bg
