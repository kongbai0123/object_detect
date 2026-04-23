import json
import os
from pathlib import Path
from datetime import datetime

history_json = r'C:\antigravity\storage\artifacts\experiments\experiments_history.json'
md_file = r'C:\antigravity\storage\artifacts\experiments\training_history.md'

if not os.path.exists(history_json):
    print("History JSON not found.")
    exit(1)

with open(history_json, 'r', encoding='utf-8') as f:
    history = json.load(f)

# 按時間倒序排列
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
    
    # 這裡目前無法從 JSON 取得 "reason"，所以先填 "已存檔" 
    # 除非我們從 Markdown 讀回舊的 Notes 或是在 JSON 增加欄位
    # 但為了快速更新，我們先列出所有實驗
    lines.append(f"| - | **{name}** | {ts} | {ds} | {mAP50:.4f} | 已存檔 |\n")

with open(md_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Updated {md_file} with {len(history)} records.")
