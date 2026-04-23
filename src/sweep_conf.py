import os
import json
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    # Path configuration
    model_path = r"C:\antigravity\storage\artifacts\experiments\specialized\exp_specialized_auto_iter_videos_0422_1435\weights\best.pt"
    data_yaml = r"C:\antigravity\storage\workspace\augment\current\dataset.yaml"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)

    model = YOLO(model_path)

    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.50]
    results_list = []

    print(f"{'Conf':<6} | {'mAP50':<8} | {'Prec':<8} | {'Recall':<8} | {'OpenRec':<8}")
    print("-" * 50)

    for conf in thresholds:
        metrics = model.val(data=data_yaml, conf=conf, plots=False, save=False, verbose=False, workers=0)
        
        mp, mr, map50, map50_95 = metrics.box.mean_results()
        
        # Extract Open Recall (Class 0)
        open_recall = 0.0
        try:
            classes = list(metrics.box.ap_class_index)
            if 0 in classes:
                idx = classes.index(0)
                open_recall = metrics.box.R[idx]
        except:
            open_recall = mr

        print(f"{conf:<6.2f} | {map50:<8.4f} | {mp:<8.4f} | {mr:<8.4f} | {open_recall:<8.4f}")
        
        results_list.append({
            "conf": conf,
            "mAP50": map50,
            "precision": mp,
            "recall": mr,
            "open_recall": open_recall
        })

    # Save results to a temporary JSON
    output_path = Path(model_path).parent / "conf_sweep.json"
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f"\nResults saved to: {output_path}")
