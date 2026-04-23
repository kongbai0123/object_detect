from typing import List, Optional
from pathlib import Path
from anti_gravity.settings import settings
from anti_gravity.logger import logger
from anti_gravity.storage import DatasetStorage
from anti_gravity.balancer import Balancer
from anti_gravity.entities import ImageMetadata

class BalanceService:
    """
    🟡 應用邏輯層 (Use Case)：平衡服務協調者
    負責組裝基礎設施與領域邏輯，完成端到端的平衡任務。
    """
    def __init__(self, storage: Optional[DatasetStorage] = None, balancer: Optional[Balancer] = None):
        # 如果沒有傳入，則使用預設配置初始化 (Dependency Injection)
        self.storage = storage or DatasetStorage(
            min_area=settings.balance.min_valid_area,
            min_dim=settings.balance.min_valid_dim
        )
        self.balancer = balancer or Balancer(
            target_ratio=settings.balance.target_ratio,
            scene_cap=settings.balance.scene_cap_close_only,
            max_bg_ratio=settings.balance.max_bg_ratio,
            error_bonus=settings.balance.error_bonus,
            edge_penalty=settings.balance.edge_penalty
        )

    def execute(self, input_raw: str, output_path: Optional[Path] = None):
        """
        執行平衡工作流。
        """
        logger.info(f"[BalanceService] Starting balance task, input: {input_raw}")
        
        # 1. 智慧路徑解析
        input_paths = self._resolve_input_paths(input_raw)
        output_dir = output_path or settings.paths.balance / "current"

        # 2. 數據掃描與轉換
        metadata_list = self.storage.scan_directories(input_paths)
        if not metadata_list:
            logger.error("[BalanceService] No valid data found, terminating.")
            return
            
        # 3. 領域邏輯運算 (平衡抽樣)
        selected_metadata = self.balancer.run(metadata_list)
        
        # 4. 基礎設施執行 (物理搬移)
        self.storage.deploy_dataset(selected_metadata, output_dir)
        
        logger.info(f"[BalanceService] Task completed successfully. Output: {output_dir}")
        print("")
        print(f"[完成] 平衡結果已輸出至:")
        print(f"       {output_dir}")
        print("")
        print("[下一步] 執行資料增強:")
        print(f"         python augment_dataset.py --input workspace/balance/current")
        return selected_metadata

    def _resolve_input_paths(self, input_raw: str) -> List[Path]:
        """
        實作之前的「5 / 6」簡寫邏輯與空資料夾自動檢查。
        """
        def has_data(p: Path):
            return (p / "images").exists() and any((p / "images").glob("*"))

        # 智慧簡寫對應新架構
        if input_raw == "5": # Auto-labeled
            candidates = [
                settings.paths.auto_ann / "reviewed",
                settings.paths.auto_ann / "current"
            ]
            for c in candidates:
                if has_data(c): return [c]
        elif input_raw == "split": # From split
            c = settings.paths.split / "current/train_src"
            if has_data(c): return [c]
            
        # 一般路徑處理
        path = Path(input_raw)
        if has_data(path):
            return [path]
            
        # 備援搜尋
        logger.warning(f"[BalanceService] Path {input_raw} has no data, searching fallbacks...")
        fallbacks = [
            settings.paths.split / "current/train_src",
            settings.paths.auto_ann / "reviewed",
            settings.paths.auto_ann / "current"
        ]
        for c in fallbacks:
            if has_data(c): 
                logger.info(f"[BalanceService] Found usable source: {c}")
                return [c]
                

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="5")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    BalanceService().execute(
        input_raw=args.input, 
        output_path=Path(args.output) if args.output else None
    )
