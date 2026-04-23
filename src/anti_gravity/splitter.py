import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from anti_gravity.entities import ImageMetadata
from anti_gravity.logger import logger

class Splitter:
    """
    🟡 應用邏輯層 (Use Case)：切分核心器
    🔴 SRP：專注於如何公平、科學地將數據分配到過不同 Set。
    """
    def __init__(self, split_ratio: float = 0.8, seed: int = 42):
        self.split_ratio = split_ratio
        self.seed = seed
        random.seed(seed)

    def extract_scene_key(self, filename: str) -> str:
        from anti_gravity.utils import extract_scene_key
        return extract_scene_key(filename)

    def perform_split(self, metadata_list: List[ImageMetadata]) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        """
        執行場景感知切分。
        """
        # 1. 根據 Scene Key 分群
        groups = defaultdict(list)
        for meta in metadata_list:
            groups[meta.scene].append(meta)
            
        group_keys = list(groups.keys())
        
        # 🛡️ 衛生檢查：若場景數過少，退化為獨立影像模式
        if len(group_keys) <= 1 and len(metadata_list) > 1:
            logger.warning("⚠️ 場景過於單一，退化為逐張切分模式 (Independent Sample Mode)")
            random.shuffle(metadata_list)
            split_idx = int(len(metadata_list) * self.split_ratio)
            return metadata_list[:split_idx], metadata_list[split_idx:]

        random.shuffle(group_keys)
        
        # 2. 按比例切分場景
        split_idx = max(1, int(len(group_keys) * self.split_ratio))
        train_keys = group_keys[:split_idx]
        val_keys = group_keys[split_idx:]
        
        train_set = []
        for k in train_keys: train_set.extend(groups[k])
            
        val_set = []
        for k in val_keys: val_set.extend(groups[k])
        
        logger.info(f"Split completed: Train {len(train_set)} images ({len(train_keys)} scenes), Val {len(val_set)} images ({len(val_keys)} scenes)")
        return train_set, val_set
