import json
import os
from pathlib import Path
from datetime import datetime

history_json = r'C:\antigravity\storage\artifacts\experiments\experiments_history.json'
md_file = r'C:\antigravity\storage\artifacts\experiments\training_history.md'

with open(history_json, 'r', encoding='utf-8') as f:
    history = json.load(f)

history.sort(key=lambda x: x['timestamp'], reverse=True)

lines = [
    "# 🚀 訓練實驗歷程 (Training History)\n",
    "\n",
    "| Order | Experiment Name | Date & Time | Dataset | mAP50 | Notes |\n",
    "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
]

for record in history:
    ts = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
    name = os.path.basename(record["save_dir"])
    ds = record["dataset"]
    mAP50 = record["metrics"]["mAP50"]
    
    # 預設標記，如果 mAP 太低則標記為 X (異常)
    status = "已存檔"
    if mAP50 < 0.1:
        status = "⚠️ 異常 (mAP過低)"
        
    lines.append(f"| - | **{name}** | {ts} | {ds} | {mAP50:.4f} | {status} |\n")

with open(md_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)
