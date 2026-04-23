import re
from pathlib import Path

def extract_scene_key(filename: str) -> str:
    """
    [Domain Utility] 場景識別工具
    針對長序列影片資料（如 Video Project），引入 Segment 分段邏輯，
    防止過大場景導致 Scene-aware split 退化。
    """
    name = Path(filename).stem
    
    # 規則 1: 處理 Video Project 類型的分段 (每 30 幀一段)
    # 範例: "10_Video Project_000035" -> "10_Video Project_seg01"
    video_m = re.match(r"(.*?_Video Project)_(\d+)$", name)
    if video_m:
        base = video_m.group(1)
        frame_idx = int(video_m.group(2))
        segment = frame_idx // 30  # 每 30 張圖切分一個場景 Bucket
        return f"{base}_seg{segment:03d}"
    
    # 規則 2: 處理破折號分隔 (例如 N-12345-001 -> N-12345)
    m1 = re.match(r"(.*?)-[a-zA-Z0-9]+$", name)
    if m1: return m1.group(1)
    
    # 規則 3: 處理底線數字結尾 (例如 image_001 -> image)
    m2 = re.match(r"(.*?)_\d+$", name)
    if m2: return m2.group(1)
    
    return name
