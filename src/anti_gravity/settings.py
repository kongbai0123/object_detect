from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 🔍 第一層級：專案根路徑鎖定
ROOT = Path(__file__).resolve().parent.parent.parent

class PathConfig(BaseSettings):
    """
    🔴 核心設計：Layered Architecture
    管理所有實體路徑，確保 Single Source of Truth。
    """
    root: Path = ROOT
    storage: Path = ROOT / "storage"
    
    # 1. Sources - 來源層
    sources: Path = ROOT / "storage/sources"
    raw: Path = ROOT / "storage/sources/raw"
    imported: Path = ROOT / "storage/sources/imported"
    
    # 2. Assets - 資產層 (驗證集)
    assets: Path = ROOT / "storage/assets"
    goldenset: Path = ROOT / "storage/assets/goldenset/versions/fif"   # split 、analyze_error、labeling的輸入
    # goldenset: Path = ROOT / "storage/sources/raw"
    validation: Path = ROOT / "storage/assets/validation"
    val_frozen: Path = ROOT / "storage/assets/validation/frozen_v1"
    replay_core: Path = ROOT / "storage/assets/replay_core"

    # 3. Workspace - 工作流層 (暫存 / 回合資料)
    workspace: Path = ROOT / "storage/workspace"
    auto_ann: Path = ROOT / "storage/workspace/auto_annotation"
    split: Path = ROOT / "storage/workspace/split"
    balance: Path = ROOT / "storage/workspace/balance"
    augment: Path = ROOT / "storage/workspace/augment"
    mining: Path = ROOT / "storage/workspace/mining"
    review: Path = ROOT / "storage/workspace/review"
    rounds: Path = ROOT / "storage/workspace/rounds"
    
    # 4. Artifacts - 產出層 (模型 / 實驗)
    artifacts: Path = ROOT / "storage/artifacts"
    experiments: Path = ROOT / "storage/artifacts/experiments"
    models: Path = ROOT / "storage/artifacts/models"
    models_foundation: Path = ROOT / "storage/artifacts/models/foundation"
    models_baselines: Path = ROOT / "storage/artifacts/models/baselines"
    models_incremental: Path = ROOT / "storage/artifacts/models/incremental"
    models_promoted: Path = ROOT / "storage/artifacts/models/promoted"
    models_registry: Path = ROOT / "storage/artifacts/models/registry"
    evaluations: Path = ROOT / "storage/artifacts/evaluations"
    storage_logs: Path = ROOT / "storage/artifacts/logs"
    
    # 配置 train.yaml 的資訊取得管道
    configs: Path = ROOT / "configs"
    pipeline_yaml: Path = ROOT / "configs/pipeline.yaml"

    def validate_paths(self):
        """
        🟣 穩定性：Fail Fast
        檢查必要路徑是否存在。
        """
        required = [self.storage, self.configs, self.pipeline_yaml]
        for p in required:
            if not p.exists():
                raise FileNotFoundError(f"❌ [PathConfig] 必要路徑不存在: {p}")
        return True

class AutoLabelSettings(BaseSettings):
    det_model: str = str(ROOT / "storage/artifacts/models/yolov8s.pt")
    sam_model: str = str(ROOT / "storage/artifacts/models/mobile_sam.pt")
    conf: float = 0.5
    iou: float = 0.45
    imgsz: int = 1024
    min_box_area_px: int = 200
    min_box_dim_px: int = 10
    target_classes: List[int] = [0, 1]

class BalanceSettings(BaseSettings):
    target_ratio: float = 2.0
    max_bg_ratio: float = 0.2
    min_valid_area: float = 0.00005
    min_valid_dim: float = 0.008
    error_bonus: float = 10.0
    edge_penalty: float = -0.6
    scene_cap_close_only: int = 2

class AugmentSettings(BaseSettings):
    base_multiplier: int = 3
    close_fp_like_multiplier: int = 2
    hard_negative_multiplier: int = 4
    open_hard_threshold: float = 0.05

class TrainSettings(BaseSettings):
    weights: str = "yolov8s.pt"
    patience: int = 10
    epochs: int = 40
    eval_conf: float = 0.25
    imgsz: int = 640

class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = Field(default_factory=lambda: ROOT / "storage/artifacts/logs/pipeline.log")

class Settings(BaseSettings):
    """
    🟠 可維護性：Modularization
    全域配置聚合器。
    """
    paths: PathConfig = PathConfig()
    autolabel: AutoLabelSettings = AutoLabelSettings()
    balance: BalanceSettings = BalanceSettings()
    augment: AugmentSettings = AugmentSettings()
    train: TrainSettings = TrainSettings()
    logging: LoggingSettings = LoggingSettings()

    model_config = SettingsConfigDict(
        env_prefix="ANTIGRAVITY_",
        env_nested_delimiter="__",
        extra="ignore"
    )

# 🚀 導出為單例物件
settings = Settings()
