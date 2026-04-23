import os
import sys
import argparse
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from pathlib import Path

# Add src to pythonpath
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from anti_gravity.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for YOLO validation")
    parser.add_argument('--model', type=str, default=str(settings.paths.models_promoted / "global_best.pt"), help='Path to model weights')
    parser.add_argument('--data', type=str, default=str(settings.paths.augment / "current/dataset.yaml"), help='Path to dataset.yaml')
    parser.add_argument('--confs', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], help='Confidence thresholds to sweep')
    args = parser.parse_args()

    model_path = args.model
    data_path = args.data
    conf_values = args.confs

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    results_list = []
    
    print("\nStarting Threshold Sweep...")
    print("="*60)
    
    # We will assume class 0 is Open and class 1 is Close, but we will dynamically check if possible.
    # YOLO validation object returns metrics.box with ap_class_index, R, P, F1
    for conf in conf_values:
        print(f"\n--- Evaluating at conf={conf} ---")
        # Run validation. Set save_json=False, plots=False to speed up.
        metrics = model.val(data=data_path, conf=conf, save_json=False, plots=False, verbose=False)
        
        box_metrics = metrics.box
        
        # Get overall metrics
        map50 = box_metrics.map50
        
        # Get class-wise metrics
        class_indices = box_metrics.ap_class_index
        class_names = metrics.names
        
        recalls = box_metrics.r
        precisions = box_metrics.p
        f1_scores = box_metrics.f1
        
        # Find 'open' class metrics
        open_idx = -1
        for idx, cls_id in enumerate(class_indices):
            if class_names[cls_id].lower() == 'open':
                open_idx = idx
                break
                
        # If not named 'open', assume 0 is open
        if open_idx == -1 and len(class_indices) > 0:
            if 0 in class_indices:
                open_idx = list(class_indices).index(0)
            else:
                open_idx = 0
                
        open_r = recalls[open_idx] if open_idx < len(recalls) else 0.0
        open_p = precisions[open_idx] if open_idx < len(precisions) else 0.0
        open_f1 = f1_scores[open_idx] if len(f1_scores.shape) > 0 and open_idx < len(f1_scores) else (2 * open_p * open_r / (open_p + open_r + 1e-16))
        
        # Calculate overall F1 from mean Precision/Recall
        mean_p = box_metrics.mp
        mean_r = box_metrics.mr
        overall_f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-16)
        
        # Look at Confusion Matrix to find Close->Open (False Positive for Open)
        # confusion_matrix.matrix is typically shape (num_classes + 1, num_classes + 1)
        # rows are true, cols are predicted. Last col is background FN. Last row is background FP.
        # Assuming class 0 = Open, class 1 = Close
        # Close -> Open is True=1, Pred=0
        close_to_open_fps = 0
        conf_mat = metrics.confusion_matrix.matrix
        if conf_mat.shape[0] >= 2 and conf_mat.shape[1] >= 2:
            # Assuming row 1 is Close, col 0 is Open prediction
            # Let's dynamically find it if names match
            open_c_id = 0
            close_c_id = 1
            for k, v in class_names.items():
                if v.lower() == 'open': open_c_id = k
                if v.lower() == 'close': close_c_id = k
                
            close_to_open_fps = int(conf_mat[close_c_id, open_c_id])
            
        
        results_list.append({
            'Conf': conf,
            'Open_Recall': open_r,
            'Open_Precision': open_p,
            'Open_F1': open_f1,
            'Overall_F1': overall_f1,
            'Close->Open_FPs': close_to_open_fps,
            'mAP50': map50
        })
        
        print(f"Conf: {conf:.2f} | Open Recall: {open_r:.4f} | Open Precision: {open_p:.4f} | Close->Open FP: {close_to_open_fps}")

    print("\n" + "="*60)
    print("Sweep Results Summary")
    print("="*60)
    df = pd.DataFrame(results_list)
    print(df.to_string(index=False))
    
    out_csv = settings.paths.evaluations / f"threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    from datetime import datetime
    main()
