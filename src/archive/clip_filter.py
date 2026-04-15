import os
import shutil
import argparse
import sys
from pathlib import Path

from pipeline_notice import print_pipeline_notice
# 無論從哪個目錄執行，路徑永遠相對於此腳本所在位置
ROOT = Path(__file__).resolve().parent.parent

# 延遲載入以加速 `--help` 等命令列查詢速度
def lazy_import():
    global torch, Image, CLIPProcessor, CLIPModel, tqdm
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm

def run_clip_filter(img_dir, output_pos, output_neg, threshold=0.3):
    img_dir = Path(img_dir)
    output_pos = Path(output_pos)
    output_neg = Path(output_neg)
    
    output_pos.mkdir(parents=True, exist_ok=True)
    output_neg.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"目錄 {img_dir} 中找不到任何圖片！請先用 video2frames.py 將影片抽出來。")
        return
        
    print("正在載入 PyTorch 與 CLIP (openai/clip-vit-base-patch32) 模型...")
    lazy_import()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"啟動硬體環境：{device.upper()}")
    
    # 掛載 CLIP Model (預訓練的神明)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 策略精華：利用三個類別做 Softmax，分離出「正樣本」、「負樣本」與「垃圾背景」
    texts = [
        "a person opening a car door",                     # [0] 正樣本：真·開門
        "cyclist riding past parked cars with closed doors", # [1] 負樣本：路過關著的車門 (Hard Negative)
        "empty street or normal traffic background"        # [2] 垃圾背景：直接丟棄
    ]
    
    kept_pos = 0
    kept_neg = 0
    
    print(f"準備從 {len(images)} 張生肉中提煉出真實的「人與開門」以及「高價值負樣本」...")
    print(f"Zero-Shot正向信心閥值: > {threshold:.2f}")
    
    for img_path in tqdm(images, desc="[雙向分析濾網]"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 影像與文字分數配對 -> 轉機率
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        open_prob = probs[0]
        close_prob = probs[1]
        
        # 狀態檢出分離器
        if open_prob > threshold:
            dest = output_pos / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            kept_pos += 1
        elif close_prob > 0.4:  # 如果模型很高機率覺得這是「騎士安全路過」
            dest = output_neg / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            kept_neg += 1
            
    if len(images) > 0:
        compression = 100.0 * (1 - (kept_pos + kept_neg) / len(images))
        print("\n" + "="*45)
        print(" CLIP 正負樣本分離報告 ")
        print("="*45)
        print(f" 原始解壓縮影格 : {len(images)} 張")
        print(f" [Pos] 擷取開門: {kept_pos} 張 -> {output_pos}")
        print(f" [Neg] 擷取安全通過: {kept_neg} 張 -> {output_neg}")
        print(f" 垃圾雜訊剔除率 : {compression:.2f}%")
        print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Data Filter using OpenAI CLIP")
    parser.add_argument("--input",      type=str, default=str(ROOT / "data/1_raw/door_opening_frames"),  # 【輸入】從影片抄出的原始未篩影格
                        help="原始影格目錄")
    parser.add_argument("--output_pos", type=str, default=str(ROOT / "data/2_filtered/open"),    # 【輸出】CLIP 確認為「開門」的正樣本
                        help="正樣本輸出目錄")
    parser.add_argument("--output_neg", type=str, default=str(ROOT / "data/2_filtered/close"),     # 【輸出】CLIP 確認為「關門路過」的負樣本
                        help="負樣本輸出目錄")
    parser.add_argument("--thresh",     type=float, default=0.3, help="Positive Confidence threshold")
    args = parser.parse_args()
    
    run_clip_filter(args.input, args.output_pos, args.output_neg, args.thresh)
    print_pipeline_notice(
        output_paths=[args.output_pos, args.output_neg],
        next_script="src/auto_label.py",
        notes=[
            "open 資料夾保留較像開門事件的影像，close 資料夾保留 hard negative。",
            "若過濾比例不理想，可調整 --thresh 後重跑。",
        ],
    )
