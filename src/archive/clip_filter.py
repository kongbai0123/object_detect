import os
import shutil
import argparse
import sys
from pathlib import Path

# 引入 MLOps 通知模組
from anti_gravity.pipeline_notice import print_pipeline_notice

ROOT = Path(__file__).resolve().parent.parent

# 使用延遲載入以加速 --help 指令的回傳速度
def lazy_import():
    global torch, Image, CLIPProcessor, CLIPModel, tqdm
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm

def run_clip_filter(img_dir, output_pos, output_neg, threshold=0.3):
    """
    [MLOps 數據治理] 基於 CLIP 的 Zero-Shot 資料篩選器
    自動將抽幀後的影格過濾為「門開啟事件」與「硬負樣本」或其他背景。
    """
    img_dir = Path(img_dir)
    output_pos = Path(output_pos)
    output_neg = Path(output_neg)
    
    output_pos.mkdir(parents=True, exist_ok=True)
    output_neg.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f" [跳過] 在 {img_dir} 中找不到任何影格！請先執行 video2frames.py。")
        return
        
    print(" [啟動] 正在載入 PyTorch 與 CLIP 模型 (openai/clip-vit-base-patch32)...")
    lazy_import()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [系統] 計算硬體偵測：{device.upper()}")
    
    # 載入預訓練模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # [核心策略] 定義語意標籤
    texts = [
        "a person opening a car door",                     # [0] 正樣本：門正在開啟
        "cyclist riding past parked cars with closed doors", # [1] 硬負樣本：路邊違標或腳踏車 (容易誤判)
        "empty street or normal traffic background"        # [2] 背景雜訊庫
    ]
    
    kept_pos = 0
    kept_neg = 0
    
    print(f" [分析] 準備針對 {len(images)} 張影像進行語意比對...")
    print(f" [門檻] 置信度預設大於 {threshold:.2f} 則判定為目標樣本。")
    
    for img_path in tqdm(images, desc="[CLIP 語意濾網]"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 取得 Softmax 機率
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        open_prob = probs[0]
        close_prob = probs[1]
        
        # 執行篩選邏輯
        if open_prob > threshold:
            dest = output_pos / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            kept_pos += 1
        elif close_prob > 0.4:  # 如果模型有極大把握認定為路邊停車或雜訊
            dest = output_neg / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            kept_neg += 1
            
    if len(images) > 0:
        compression = 100.0 * (1 - (kept_pos + kept_neg) / len(images))
        print("\n" + "="*45)
        print(" [結案] CLIP 語意自動化過濾總結")
        print("="*45)
        print(f" 總輸入影像數: {len(images)} 張")
        print(f" 正向樣本 (Open): {kept_pos} 張 -> {output_pos}")
        print(f" 硬負樣本 (Neg): {kept_neg} 張 -> {output_neg}")
        print(f" 垃圾影像過濾率: {compression:.2f}%")
        print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基於 OpenAI CLIP 的 Zero-Shot 資料篩選工具")
    parser.add_argument("--input", type=str, default=str(ROOT / "data/1_raw/door_opening_frames"), help="待過濾影格目錄")
    parser.add_argument("--output_pos", type=str, default=str(ROOT / "data/2_filtered/open"), help="正樣本輸出目錄")
    parser.add_argument("--output_neg", type=str, default=str(ROOT / "data/2_filtered/close"), help="硬負樣本輸出目錄")
    parser.add_argument("--thresh", type=float, default=0.3, help="正樣本置信度門檻 (建議 0.25 - 0.35)")
    args = parser.parse_args()
    
    run_clip_filter(args.input, args.output_pos, args.output_neg, args.thresh)
    
    print_pipeline_notice(
        output_paths=[os.path.abspath(args.output_pos), os.path.abspath(args.output_neg)],
        next_script="src/auto_label.py",
        notes=[
            "CLIP 已初步幫您分離出高機率的門切事件與難題樣本。",
            "若篩選結果不如預期，可微調 --thresh 參數再次執行。",
        ],
    )
