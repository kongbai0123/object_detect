import os
from pathlib import Path

target = Path(r'C:\antigravity\storage')
for p in target.rglob('*.mp4'):
    print(p)
