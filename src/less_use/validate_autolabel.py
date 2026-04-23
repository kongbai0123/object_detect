import os
import json
import yaml
import sys
from pathlib import Path

def validate_autolabel(manifest_path, config_path=None):
    print("--- [Gate A: Auto-Label Quality Gate] ---")
    
    if not os.path.exists(manifest_path):
        print(f" ❌ Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    stats = manifest.get("stats", {})
    total = stats.get("total_images", 0)
    fp_count = stats.get("suspect_fp_count", 0)
    
    if total == 0:
        print(" ❌ Error: Total images in manifest is 0.")
        sys.exit(1)
        
    fp_ratio = fp_count / total
    
    # --- 門檻設定 (預設或從 config 讀取) ---
    max_fp_ratio = 0.05
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_cfg = yaml.safe_load(f)
            # 可以在 pipeline.yaml 擴充 gates 區塊
            max_fp_ratio = full_cfg.get('gates', {}).get('max_autolabel_fp_ratio', 0.05)

    print(f" 📊 Suspect FP Count: {fp_count}")
    print(f" 📊 Suspect FP Ratio: {fp_ratio:.2%} (Threshold: {max_fp_ratio:.2%})")
    
    if fp_ratio > max_fp_ratio:
        print(f"\n 🚨 [HARD VETO] 標註品質不合格！背景誤報比例 ({fp_ratio:.2%}) 超過門檻。")
        print(" 💡 建議：檢查標註模型 (det_model) 是否權重過擬合，或生肉圖中包含過多雜訊。")
        print(" ❌ Pipeline 終止。")
        sys.exit(1)
    
    print("\n ✅ [PASS] Auto-Label 階段通過。數據品質符合門禁標準。")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_manifest = os.path.normpath(os.path.join(script_dir, '../data/5_auto_ann/auto_label_manifest.json'))
    default_config = os.path.normpath(os.path.join(script_dir, '../configs/pipeline.yaml'))
    
    validate_autolabel(default_manifest, default_config)
