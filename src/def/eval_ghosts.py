import os
import json
import argparse
from ultralytics import YOLO
from pathlib import Path

def eval_ghosts(model_path, background_dir, output_dir):
    """
    Ghost Veto: 測試模型在純背景圖上的誤報率。
    """
    model = YOLO(model_path)
    results = model.predict(source=background_dir, conf=0.25, save=False, verbose=False)
    
    total_imgs = len(results)
    fp_count = 0
    
    for r in results:
        if len(r.boxes) > 0:
            fp_count += 1
            
    stats = {
        "total_images": total_imgs,
        "any@0.25": fp_count,
        "fp_rate": round(fp_count / total_imgs, 4) if total_imgs > 0 else 0
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ghost_eval.json"), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Ghost Eval Finished: {fp_count}/{total_imgs} images had FPs.")
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None)
    args = parser.parse_args()
    
    # 預設背景圖路徑
    from anti_gravity.settings import settings
    bg_dir = str(settings.paths.storage / "assets/goldenset/versions/1_img/images")
    
    eval_ghosts(args.model, bg_dir, args.output)
