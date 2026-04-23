from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ClassID(IntEnum):
    """
    🔴 核心設計：Domain Entity
    類別標籤的單一來源。
    """
    OPEN = 0
    CLOSE = 1

class BoxInfo(BaseModel):
    """
    標籤框實體。
    """
    cls_id: ClassID
    area: float
    cx: float
    cy: float
    w: float
    h: float
    is_near_edge: bool = False

class ImageMetadata(BaseModel):
    """
    影像元資料實體，封裝影像統計資訊與決策標籤。
    """
    path: Path
    label_path: Optional[Path] = None
    scene: str
    boxes: List[BoxInfo] = Field(default_factory=list)
    is_error: bool = False
    is_type_a: bool = False
    
    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def open_cnt(self) -> int:
        return len([b for b in self.boxes if b.cls_id == ClassID.OPEN])
        
    @property
    def close_cnt(self) -> int:
        return len([b for b in self.boxes if b.cls_id == ClassID.CLOSE])
    
    @property
    def is_background(self) -> bool:
        return len(self.boxes) == 0

class BalanceManifest(BaseModel):
    """
    平衡結果清單，用於持久化與追蹤 (Traceability)。
    """
    timestamp: str
    input_sources: List[str]
    output_dir: str
    parameters: Dict[str, Any]
    stats: Dict[str, Any]
    sampled_stems: List[str]
