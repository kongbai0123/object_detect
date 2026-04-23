import argparse
import cv2
import shutil
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

# ============================================================================
# System Setup & Paths
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from ultralytics import YOLO
from anti_gravity.pipeline_notice import print_pipeline_notice
from auto_label import create_cvat_package
from anti_gravity.settings import settings

# ============================================================================
# 1. Domain Entities & Configuration (Single Source of Truth, Immutable)
# ============================================================================

@dataclass(frozen=True)
class MiningBucket:
    """定義採樣策略的唯讀領域物件 (Immutable data)"""
    name: str
    min_conf: float
    max_conf: float
    description: str

@dataclass(frozen=True)
class MinerConfig:
    """核心配置 (Configuration over hardcode)"""
    model_path: Path
    input_dir: Path
    output_dir: Path
    sample_every_n_frames: int
    buckets: tuple  # Tuple for immutability

# ============================================================================
# 2. Interfaces (Abstraction First, ISP)
# ============================================================================

class IDetector(ABC):
    @abstractmethod
    def predict(self, image) -> List: pass

class IStorage(ABC):
    @abstractmethod
    def save_sample(self, bucket_name: str, source_name: str, image, result) -> None: pass
    @abstractmethod
    def get_saved_buckets(self) -> List[Path]: pass

# ============================================================================
# 3. Infrastructure Implementations (Encapsulation, Defensive programming)
# ============================================================================

class YOLOv8Detector(IDetector):
    """YOLO 推理實作細節的封裝 (Plug-in architecture)"""
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型權重不存在: {model_path}") # Fail fast
        self.model = YOLO(model_path)
        
    def predict(self, image) -> List:
        return self.model.predict(image, verbose=False)

class YOLOFormatStorage(IStorage):
    """處理標準化儲存與目錄結構 (State isolation, Idempotency)"""
    def __init__(self, base_output_dir: Path):
        self.base_dir = Path(base_output_dir)
        self.saved_buckets = set()

    def _ensure_dirs(self, bucket_name: str) -> Path:
        bucket_dir = self.base_dir / bucket_name
        if bucket_name not in self.saved_buckets:
            (bucket_dir / "images").mkdir(parents=True, exist_ok=True)
            (bucket_dir / "labels").mkdir(parents=True, exist_ok=True)
            (bucket_dir / "previews").mkdir(parents=True, exist_ok=True)
            self.saved_buckets.add(bucket_name)
        return bucket_dir

    def save_sample(self, bucket_name: str, source_name: str, image, result) -> None:
        bucket_dir = self._ensure_dirs(bucket_name)
        stem = Path(source_name).stem
        
        cv2.imwrite(str(bucket_dir / "images" / f"{stem}.jpg"), image)
        cv2.imwrite(str(bucket_dir / "previews" / f"{stem}.jpg"), result.plot())
        
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            with open(bucket_dir / "labels" / f"{stem}.txt", "w") as f:
                for box in boxes:
                    cls = int(box.cls[0])
                    xywh = box.xywhn[0].tolist()
                    f.write(f"{cls} {' '.join([f'{x:.6f}' for x in xywh])}\n")
                    
    def get_saved_buckets(self) -> List[Path]:
        return [self.base_dir / b for b in self.saved_buckets]

# ============================================================================
# 4. Use Case / Application Service (SRP, Explicit Data Flow)
# ============================================================================

class MiningCoordinator:
    """
    挖掘協調器：組合基礎設施與領域邏輯 (Dependency Injection)
    不依賴具體的 YOLO 或 FileSystem，只依賴 Interface。
    """
    def __init__(self, config: MinerConfig, detector: IDetector, storage: IStorage):
        self.config = config
        self.detector = detector
        self.storage = storage
        self.stats = {b.name: 0 for b in config.buckets}

    def _evaluate_strategy(self, result) -> Optional[str]:
        """策略模式判斷：根據領域規則 (buckets) 將結果分流 (Strategy pattern)"""
        boxes = result.boxes
        if len(boxes) == 0:
            return None
            
        max_conf = float(boxes.conf.max())
        for bucket in self.config.buckets:
            # 判斷落在哪個信心度區間
            if bucket.min_conf <= max_conf <= bucket.max_conf:
                return bucket.name
        return None

    def execute(self) -> Dict[str, int]:
        """執行核心作業流"""
        media_files = self._gather_media(self.config.input_dir)
        if not media_files:
            print(f" [跳過] 找不到任何可處理的媒體: {self.config.input_dir}")
            return self.stats

        pbar_files = tqdm(media_files, desc="挖掘進度", unit="file")
        
        for media_path in pbar_files:
            pbar_files.set_description(f"處理: {media_path.name}")
            
            if media_path.suffix.lower() in {".mp4", ".avi", ".mkv"}:
                self._process_video(media_path)
            else:
                self._process_image(media_path)
                
            pbar_files.set_postfix(**self.stats)
            
        return self.stats

    def _gather_media(self, source_dir: Path) -> List[Path]:
        if not source_dir.is_dir():
            return [source_dir]
        patterns = ("*.mp4", "*.avi", "*.mkv", "*.jpg", "*.jpeg", "*.png", "*.webp")
        files = []
        for pattern in patterns:
            files.extend(source_dir.rglob(pattern))
        return sorted(files)

    def _process_video(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar_video = tqdm(total=total_frames, desc=" -> 影格", leave=False, unit="frame")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            pbar_video.update(1)
            
            if frame_idx % self.config.sample_every_n_frames != 0:
                continue
                
            result = self.detector.predict(frame)[0]
            bucket_name = self._evaluate_strategy(result)
            
            if bucket_name:
                source_name = f"{video_path.stem}_f{frame_idx}.jpg"
                self.storage.save_sample(bucket_name, source_name, frame, result)
                self.stats[bucket_name] += 1
                
        pbar_video.close()
        cap.release()

    def _process_image(self, image_path: Path):
        frame = cv2.imread(str(image_path))
        if frame is None: return
        
        result = self.detector.predict(frame)[0]
        bucket_name = self._evaluate_strategy(result)
        
        if bucket_name:
            self.storage.save_sample(bucket_name, image_path.name, frame, result)
            self.stats[bucket_name] += 1

# ============================================================================
# 5. Presentation / CLI (Loose coupling, Fail Fast)
# ============================================================================

def resolve_model_path(user_input: str) -> Path:
    """智慧模型路徑解析 (Convention over configuration, Fail fast)"""
    if user_input and user_input.strip():
        p = Path(user_input)
        if p.exists() and p.is_file(): 
            return p
    
    fallback_1 = settings.paths.models / "latest/latest_best.pt"
    fallback_2 = settings.paths.models / "promoted/yolov8s.pt"
    
    if fallback_1.exists(): return fallback_1
    if fallback_2.exists(): return fallback_2
    
    raise FileNotFoundError("無法解析模型路徑，請確認路徑或備援模型是否存在。")

def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Miner (MLOps 企業級架構)")
    parser.add_argument("--mode", type=str, choices=["active", "hardcase", "both"], default="both",
                        help="採樣策略：active(不確定性), hardcase(高低信心區段), both(全開)")
    parser.add_argument("--model", type=str, default="", help="YOLO 權重路徑")
    parser.add_argument("--input", type=str, default=str(settings.paths.raw), help="輸入媒體來源目錄")
    parser.add_argument("--output", type=str, default=str(settings.paths.mining / "current"), help="輸出結果存放目錄")
    parser.add_argument("--sample-every", type=int, default=10, help="影片抽樣幀數間隔")
    args = parser.parse_args()

    # 組合採樣策略 (Strategy / Composition over inheritance)
    bucket_list = []
    
    if args.mode in ["hardcase", "both"]:
        bucket_list.append(MiningBucket(name="high_conf_fp", min_conf=0.61, max_conf=1.00, description="高信心度誤報區 (Hard Case)"))
    
    if args.mode in ["active", "both"]:
        bucket_list.append(MiningBucket(name="low_conf_uncertain", min_conf=0.05, max_conf=0.60, description="低信心度不確定區 (Active Learning)"))
    
    if args.mode == "hardcase":
        # 如果是純 Hard Case 模式，通常也需要抓取低信心度來補強
        bucket_list.append(MiningBucket(name="low_conf_uncertain", min_conf=0.05, max_conf=0.60, description="低信心度不確定區 (Hard Case)"))

    # 初始化不可變配置 (Immutable config)
    config = MinerConfig(
        model_path=resolve_model_path(args.model),
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sample_every_n_frames=args.sample_every,
        buckets=tuple(bucket_list)
    )

    print("\n" + "="*50)
    print(f" 🚀 [Miner] 啟動統一數據挖掘引擎 (目前模式: {args.mode.upper()})")
    print(f" 📦 來源: {config.input_dir}")
    print(f" 🧠 模型: {config.model_path}")
    print(" ⚙️ 啟用的策略 (Buckets):")
    for b in config.buckets:
        print(f"    - [{b.name}] : 信心度 {b.min_conf} ~ {b.max_conf} ({b.description})")
    print("="*50 + "\n")
    
    # 依賴注入 (Dependency Injection)
    detector = YOLOv8Detector(str(config.model_path))
    storage = YOLOFormatStorage(config.output_dir)
    coordinator = MiningCoordinator(config, detector, storage)
    
    # 執行流程
    stats = coordinator.execute()
    
    # 總結與封裝 (Post-processing)
    print("\n" + "="*50)
    print(" 📊 [挖掘完成] 總結報告")
    print("="*50)
    total_mined = 0
    for k, v in stats.items():
        print(f"  - {k}: {v} 張")
        total_mined += v
    print("="*50)

    saved_dirs = storage.get_saved_buckets()
    zips = []
    if total_mined > 0:
        for d in saved_dirs:
            if any((d / "images").glob("*.jpg")):
                print(f" [封裝] 正在建立 {d.name} 標籤包...")
                z = create_cvat_package(str(d / "images"), str(d / "labels"), zip_name=f"miner_{d.name}.zip")
                if z: zips.append(str(z))

    print_pipeline_notice(
        output_paths=[str(d) for d in saved_dirs] + zips,
        next_script="CVAT Import",
        notes=[
            "✅ 已自動導出 YOLO 標籤檔案 (.txt) 與 預覽圖。",
            f"📦 已建立可匯入 CVAT 的壓縮檔: {', '.join(zips)}" if zips else "未發現足夠樣本進行封裝。"
        ]
    )

if __name__ == "__main__":
    main()
