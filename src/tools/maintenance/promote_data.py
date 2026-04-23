import shutil
from pathlib import Path
import os

src_base = Path(r"C:\antigravity\storage\sources\raw")
dst_base = Path(r"C:\antigravity\storage\assets\goldenset\versions\5_new_videos")

dst_base.mkdir(parents=True, exist_ok=True)

for folder in ["images", "labels"]:
    src = src_base / folder
    dst = dst_base / folder
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"Moved {src} to {dst}")
    else:
        print(f"Source {src} not found")
