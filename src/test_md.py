import json
from anti_gravity.settings import settings
from datetime import datetime
import os

with open(settings.paths.experiments / "experiments_history.json", 'r', encoding='utf-8') as f:
    history = json.load(f)
    
record = history[-1]

md_file = settings.paths.experiments / "training_history.md"

try:
    ts = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
    name = os.path.basename(record["save_dir"])
    ds = record["dataset"]
    map50 = record["metrics"]["mAP50"]
    
    status = "未達晉升標準"
    
    # 簡潔的一行紀錄，包含動態 status
    new_line = f"| - | **{name}** | {ts} | {ds} | {map50:.4f} | {status} |\n"
    
    print(new_line)
except Exception as e:
    print(f"Failed: {e}")
