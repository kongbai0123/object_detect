import os
import json
import yaml
import sys
from pathlib import Path

def validate_balance(manifest_path, config_path=None):
    print("--- [Gate B: Balance & Governance Gate] ---")
    
    if not os.path.exists(manifest_path):
        print(f" ❌ Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    stats = manifest.get("stats", {})
    final_ratio = stats.get("final_ratio", 0)
    target_ratio = manifest.get("config", {}).get("target_ratio", 2.0)
    
    # --- 1. 比例偏離檢查 (Ratio Guard) ---
    max_drift = 0.20 # 允許 20% 偏離
    drift = abs(final_ratio - target_ratio) / target_ratio if target_ratio > 0 else 0
    
    print(f" 📊 Target Ratio: {target_ratio}")
    print(f" 📊 Final Ratio: {final_ratio} (Drift: {drift:.2%})")
    
    if drift > max_drift:
        print(f" 🚨 [HARD VETO] 數據平衡比例偏離過大 (>{max_drift:.0%})！")
        print(f" 💡 建議：檢查原始數據中 Open/Close 樣本是否極度不均，或 target_ratio 設定過於激進。")
        sys.exit(1)
        
    # --- 2. 樣本多樣性檢查 (基本) ---
    close_only_sampled = stats.get("close_only_sampled", 0)
    if close_only_sampled == 0 and stats.get("close_boxes", 0) > 0:
        print(" ⚠️ [WARNING] 沒有從 Close_only 群體中抽樣到任何樣本，場景多樣性可能受損。")

    print("\n ✅ [PASS] Balance 階段通過。樣本比例與場景治理符合門禁標準。")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_manifest = os.path.normpath(os.path.join(script_dir, '../data/6_balanced/balance_manifest.json'))
    default_config = os.path.normpath(os.path.join(script_dir, '../configs/pipeline.yaml'))
    
    validate_balance(default_manifest, default_config)
