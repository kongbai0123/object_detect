import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Delay import to keep startup fast
def get_yolo_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

def evaluate_ghosts(model_path, ghost_dir, out_dir, historical_model_path=None):
    print(f"--- [Ghost Evaluator 2.0] ---")
    print(f"Target Model: {model_path}")
    
    model = get_yolo_model(model_path)
    ghost_path = Path(ghost_dir)
    images = list(ghost_path.glob("*.jpg")) + list(ghost_path.glob("*.png"))
    
    if not images:
        print("Error: No ghost images found for testing.")
        return None
        
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    review_dir = out_path / "ghost_fp_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = [0.1, 0.25, 0.4]
    
    def run_inference(m):
        fp_data = {t: {"any": 0, "close": 0, "open": 0} for t in thresholds}
        max_confs = []
        for img_path in images:
            res = m.predict(str(img_path), verbose=False)[0]
            img_max_conf = 0.0
            for box in res.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0]) # 0: open, 1: close
                img_max_conf = max(img_max_conf, conf)
                for t in thresholds:
                    if conf >= t:
                        if cls == 1: fp_data[t]["close"] += 1
                        else: fp_data[t]["open"] += 1
            
            max_confs.append(img_max_conf)
            for t in thresholds:
                if img_max_conf >= t:
                    fp_data[t]["any"] += 1
            
            # Save visual check for target model only
            if m == model and img_max_conf >= 0.1:
                vis = res.plot()
                cv2.imwrite(str(review_dir / f"fp_{img_path.name}"), vis)
                
        stats = {
            "num_images": len(images),
            "mean_max_conf": float(np.mean(max_confs)),
            "95th_conf": float(np.percentile(max_confs, 95)),
            "max_conf": float(np.max(max_confs)) if max_confs else 0.0
        }
        for t in thresholds:
            stats[f"any@{t}"] = fp_data[t]["any"]
            stats[f"close@{t}"] = fp_data[t]["close"]
            stats[f"open@{t}"] = fp_data[t]["open"]
        return stats

    target_stats = run_inference(model)
    
    # 產出 JSON
    with open(out_path / "ghost_eval.json", "w") as f:
        json.dump(target_stats, f, indent=4)
        
    # 比對邏輯
    comparison_md = [f"# Ghost Evaluation Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    comparison_md.append(f"- **Target Model**: `{Path(model_path).name}`")
    
    if historical_model_path and os.path.exists(historical_model_path):
        print(f"Comparing against historical: {historical_model_path}")
        hist_model = get_yolo_model(historical_model_path)
        hist_stats = run_inference(hist_model)
        
        comparison_md.append(f"- **Baseline Model**: `{Path(historical_model_path).name}`")
        comparison_md.append("\n## 📊 Comparison Strategy")
        comparison_md.append("| Metric | Baseline | Target | Status |")
        comparison_md.append("| :--- | :--- | :--- | :--- |")
        
        for k in target_stats.keys():
            if k in hist_stats:
                v_target = target_stats[k]
                v_hist = hist_stats[k]
                status = "✅"
                if "any@" in k or "close@" in k:
                    if v_target > v_hist: status = "🚨" # 誤報上升
                    elif v_target < v_hist: status = "💎" # 進步
                comparison_md.append(f"| {k} | {v_hist:.3f} | {v_target:.3f} | {status} |")
    else:
        comparison_md.append("\n## 📊 Target Model Statistics")
        comparison_md.append("| Metric | Value |")
        comparison_md.append("| :--- | :--- |")
        for k, v in target_stats.items():
            comparison_md.append(f"| {k} | {v:.3f} |")

    with open(out_path / "ghost_comparison.md", "w", encoding='utf-8') as f:
        f.write("\n".join(comparison_md))
        
    print(f"Evaluation finished. Result saved to: {out_path}")
    return target_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="")
    parser.add_argument("--input", type=str, default="data/2_filtered/hard_negatives/fp_close_high")
    parser.add_argument("--output", type=str, default="data/8_ghost_evals/latest")
    args = parser.parse_args()
    
    evaluate_ghosts(args.model, args.input, args.output, historical_model_path=args.baseline)
