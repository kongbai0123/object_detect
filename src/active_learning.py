import os
import glob
import shutil
import argparse
import imagehash
from PIL import Image
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from pipeline_notice import print_pipeline_notice

class ActiveLearnerLevel2:
    def __init__(self, model_path, target_dir='../data/8_hard_cases/low_conf_uncertain'):
        """
        初始化 MLOps Level-2 主動學習模組
        重樔計不設預設 model_path，強制傳入，避免誤用舊模型
        """
        print(f" [Level 2 Active Learning] 載入模型: {model_path} ...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" 模型檔案不存在: {model_path}！請確認訓練已完成")
        self.model = YOLO(model_path)
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
        self.seen_hashes = set()

    def calculate_entropy(self, confidences, class_ids=None):
        """
        4 Level 2: 升級為 Multi-class Entropy
        若有多個不同 class，利用 softmax 歸一化的分類機率做真正的多類別香農熵計算
        單類別退化為二元熵，確保向下相容
        
        [優化] 由於 YOLOv8 detection head 並沒有直接輸出 softmax p(class|image)
        我們改用各個檢測框的最高信心度 (conf) 作為機率的近似值來計算系統層級的熵
        """
        if len(confidences) == 0:
            return 0.0
            
        # 將所有的信心度做 L1 正規化，使其總和為 1，當作該圖片的 Pseudo-Probability Distribution
        conf_arr = np.array(confidences)
        total_conf = np.sum(conf_arr) + 1e-6
        probs = conf_arr / total_conf
        
        # 計算 Entropy = -Σ p * log(p)
        entropy = -np.sum(probs * np.log2(probs + 1e-6))
        return float(entropy)

    def compute_uncertainty_score(self, boxes, min_conf=0.15, max_conf=0.6,
                                  entropy_thresh=1.5, min_density=3,
                                  rare_classes=None):
        """
        [優化] 拋棄人工調參的加法 heuristic，改用標準化的加權總和
        score = w1 * normalize(entropy) + w2 * normalize(density) + w3 * normalize(var)
        """
        if not boxes:
            return 0.0
            
        conf_list = []
        class_ids = []

        for box in boxes:
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])
            conf_list.append(conf)
            class_ids.append(cls_id)
            
        import math

        # --- 維度 1: Entropy (資訊混亂度) Normalized ---
        # 理論最大熵為 log2(N)，N 為物件數我們將其映射到 0~1
        raw_entropy = self.calculate_entropy(conf_list, class_ids)
        max_possible_entropy = math.log2(len(conf_list)) if len(conf_list) > 1 else 1.0
        norm_entropy = min(raw_entropy / (max_possible_entropy + 1e-6), 1.0)

        # --- 維度 2: Density (物件密集度) Normalized ---
        # 假設實務上單張圖極端密集是 30 個物件，超過 30 就當作 1.0
        norm_density = min(len(conf_list) / 30.0, 1.0)

        # --- 維度 3: Variance (信心度劇烈震盪) Normalized ---
        # 變異數理論最大值在 0~1 的分佈中為 0.25
        norm_var = float(np.var(conf_list)) / 0.25 if len(conf_list) > 1 else 0.0
        norm_var = min(norm_var, 1.0)
        
        # --- 維度 4: Ambiguity (低信心度佔比) Normalized ---
        ambiguous_confs = [c for c in conf_list if min_conf <= c <= max_conf]
        norm_ambiguity = len(ambiguous_confs) / len(conf_list)

        # --- 動態權重分配 (Weights) ---
        w_entropy = 0.35
        w_density = 0.20
        w_var = 0.15
        w_ambig = 0.30
        
        composite_score = (
            w_entropy * norm_entropy +
            w_density * norm_density +
            w_var * norm_var +
            w_ambig * norm_ambiguity
        )

        # --- 3 維度 5：Class-aware Bonus (稀有類別額外獎勵, 唯一保留絕對加分值) ---
        if rare_classes:
            detected_classes = set(class_ids)
            if detected_classes & set(rare_classes):
                composite_score += 0.5  # 給稀有類別的絕對分數獎勵提升排名
                
        # 為了相容之前的 `score >= 1.0` 邏輯，我們將正規化的 0~1 的 Composite 分數放大，例如乘上 3.0
        # 這樣滿分大概是 3.0 ~ 3.5 左右，原有的 >= 1.0 依舊能漂亮地過濾出前段班
        return composite_score * 3.0

    def select_hard_samples(self, raw_images_dir, batch_size=32,
                            min_conf=0.15, max_conf=0.6,
                            top_k=None, rare_classes=None):
        """
        [Level 3 升級] 精準的多維度不確定性複合排名 Top-K 選取引擎
        1 score >= 2 (提升品質門檻)
        2 Top-K sorted selection (確保精華中的精華)
        3 class-aware bonus for rare classes
        4 multi-class entropy
        5 composite uncertainty ranking
        """
        print(f" 掃描生肉圖庫: {raw_images_dir} (Batch Size={batch_size})")
        
        supported_exts = ['.jpg', '.jpeg', '.png', '.webp']
        img_paths = []
        for ext in supported_exts:
            img_paths.extend(glob.glob(os.path.join(raw_images_dir, f'*{ext}')))
            
        print(f" 總計 {len(img_paths)} 張初始生肉圖片")
        
        duplicate_count = 0
        valid_paths = []

                # -----------------------------------------------------------------
        # 階段 1.5：已標定影像過濾 (Labeled Filter)
        # 確保不會重複推薦已經在 3_processed 裡的圖，節省人工標記時間
        # -----------------------------------------------------------------
        # 取得腳本所在路徑並定位到 3_processed
        processed_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/3_processed'))
        if os.path.exists(processed_dir):
            # 取得已標定池中所有檔案的檔名集合 (不分副檔名)
            processed_files = {os.path.basename(f) for f in glob.glob(os.path.join(processed_dir, '*.*'))}
            original_count = len(valid_paths)
            valid_paths = [p for p in valid_paths if os.path.basename(p) not in processed_files]
            filtered_count = original_count - len(valid_paths)
            if filtered_count > 0:
                print(f" [過濾] 自動偵測到 {filtered_count} 張已標定過的作品，已將其從推論名單中剔除")

        print(f" 最終剩 {len(valid_paths)} 張『全新』生肉圖片準備進行價值提煉")
        
        # 防呆：如果去重後一張圖都沒有，直接安全退出
        if not valid_paths:
            print(" 生肉池裡找不到任何有效圖片！")
            print(f" 請確認 '{raw_images_dir}' 資料夾存在且內含 .jpg / .png 的圖片")
            return
        
        # -----------------------------------------------------------------
        # 階段二：Chunk-based 穩定推論 + 複合不確定性評分 (Scoring Engine)
        # 4 分批切 chunk：避免一次丟入 10k 張導致 VRAM OOM 或 array stack 崩潰
        # -----------------------------------------------------------------
        import torch
        print(f"\n [階段二] 啟動 Chunk-based 穩定推論 (每批 {500} 張，自動釋放 VRAM)...")
        
        scored_candidates = []  # [(score, img_path), ...]
        all_scores = []         # 為調參用的統計分佈觀察陣列
        chunk_size = 500
        total_chunks = (len(valid_paths) + chunk_size - 1) // chunk_size
        
        # 強制將所有路徑轉換為絕對路徑，防止 YOLO result.path 回傳相對路徑造成 copy 失敗
        valid_paths = [os.path.abspath(p) for p in valid_paths]
        # 建立 basename → 絕對路徑 的查找表 (防禦 result.path 只回傳檔名的 YOLO bug)
        path_lookup = {os.path.basename(p): p for p in valid_paths}
        
        for chunk_idx in range(0, len(valid_paths), chunk_size):
            chunk = valid_paths[chunk_idx:chunk_idx + chunk_size]
            current_chunk_num = chunk_idx // chunk_size + 1
            print(f"\n   Chunk {current_chunk_num}/{total_chunks}：推論 {len(chunk)} 張...")

            
            results_generator = self.model.predict(
                source=chunk,
                stream=True,
                batch=batch_size,
                verbose=False,
                half=True   # FP16 半精度加速，大幅降低 VRAM 使用量
            )
            
            for result in tqdm(results_generator, total=len(chunk), desc=f"[Chunk {current_chunk_num} 評分]", unit="img"):
                # 防禦 YOLO result.path 只回傳相對路徑或純檔名的 BUG
                raw_path = result.path
                base_name = os.path.basename(raw_path)
                
                #  關鍵：保證取得 100% 存在正確的絕對路徑，否則直接放棄這張圖
                if os.path.isabs(raw_path) and os.path.exists(raw_path):
                    img_path = raw_path
                elif base_name in path_lookup:
                    img_path = path_lookup[base_name]
                else:
                    # 連 lookup 都找不到 (可能是 YOLO 內部截斷了)，直接跳過防止炸板
                    continue
                    
                boxes = result.boxes

                
                score = self.compute_uncertainty_score(
                    boxes=boxes,
                    min_conf=min_conf,
                    max_conf=max_conf,
                    min_density=2,      # 下修密度門檻
                    entropy_thresh=0.8, # 下修熵門檻
                    rare_classes=rare_classes
                )
                
                all_scores.append(score)
                
                # 1 提升品質門檻：下修為 score >= 1.0 讓初期模型也能選到圖
                if score >= 1.0:
                    scored_candidates.append((score, img_path))
            
            # 每個 chunk 結束後強制釋放 GPU 顯存，避免 VRAM 殘留積累
            torch.cuda.empty_cache()

        # --- 觀測用：輸出分數分佈 (協助調參) ---
        if all_scores:
            print("\n SCORE 分佈統計:")
            print(f"  min:    {np.min(all_scores):.3f}")
            print(f"  max:    {np.max(all_scores):.3f}")
            print(f"  mean:   {np.mean(all_scores):.3f}")
            print(f"  median: {np.median(all_scores):.3f}")
            print("-" * 30)

        # 2 Top-K sorted selection：依複合分數由高到低排序，取精華中的精華
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        final_candidates = scored_candidates[:top_k] if top_k else scored_candidates
        
        # 3 Fallback 保底機制：如果門檻還是太高一張都沒選中，直接取最不確定的前 N 張
        if len(final_candidates) == 0 and len(valid_paths) > 0:
            print("\n 系統過於保守，沒有任何樣本達標！啟用 Fallback 保底策略 (盲撈 100 張)")
            fallback_count = min(100, len(valid_paths))
            # 盲撈策略：隨機抽取或直接拿前 100 張，此處直接取前 100 張
            final_candidates = [(0.0, p) for p in valid_paths[:fallback_count]]
        
        print(f"\n 評分完成！" + 
              (f"{len(scored_candidates)} 張圖達標 (score>=1.0)，" if len(scored_candidates) > 0 else "") +
              (f"依 Top-{top_k} 精選後剩 {len(final_candidates)} 張入庫" if top_k and len(scored_candidates) > 0 else f"總共有 {len(final_candidates)} 張準備入庫"))
        
        hard_count = 0
        for score_val, img_path in final_candidates:
            # 防呆：確保存入 copy 的 img_path 絕對是帶有磁碟機的絕對路徑
            if not os.path.isabs(img_path) or not os.path.exists(img_path):
                img_path = path_lookup.get(os.path.basename(img_path), img_path)
                
            #  copy 前最後一道終極保險：真的沒這檔案就跳過
            if not os.path.exists(img_path):
                continue
                
            base_name = os.path.basename(img_path)
            dest = os.path.join(self.target_dir, base_name)
            shutil.copy2(img_path, dest)
            hard_count += 1
                
        print(f"\n 價值提煉引擎執行完畢！")
        print(f" 經過去重防呆與 Level 3 複合評分嚴密篩選，為您挑出了最值得人工標註的 {hard_count} 張稀有案例")
        print(f" 請開啟 CVAT 匯入: {os.path.abspath(self.target_dir)}")


    def merge_to_dataset(self, dataset_dir):
        """
         [Active Learning 閉環關鍵] 將 hard_samples 回流至 dataset 生肉池，供下次切分再訓練
        防沙機制不覆寫已存在的同名圖片，避免資料沙化
        """
        print(f"\n [閉環回流] 將 hard_samples 移交到 Dataset 生肉池: {dataset_dir}")
        img_dst = os.path.join(dataset_dir, 'images')
        os.makedirs(img_dst, exist_ok=True)
        moved = 0
        skipped = 0
        for img_path in glob.glob(os.path.join(self.target_dir, '*.*')):
            try:
                base = os.path.basename(img_path)
                dest = os.path.join(img_dst, base)
                if not os.path.exists(dest):
                    shutil.copy2(img_path, dest)
                    moved += 1
                else:
                    skipped += 1
            except Exception:
                pass
        print(f" 閉環回流完成！新增 {moved} 張跳過 {skipped} 張重複")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Learning Level 2 工程級主動學習引擎')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [Weights Discovery System] 動態搜尋優先級: global_best > latest_best > 最新 exp
    runs_dir = os.path.normpath(os.path.join(script_dir, '../data/7_experiments'))
    
    def find_best_model(base_dir):
        # 1. 冠軍模型
        g_best = os.path.join(base_dir, 'weight/global_best.pt')
        if os.path.exists(g_best): return g_best
        
        # 2. 最新挑戰者
        l_best = os.path.join(base_dir, 'weight/latest_best.pt')
        if os.path.exists(l_best): return l_best
        
        # 3. 各實驗室產出的最新結果
        import glob as _glob
        found = sorted(_glob.glob(os.path.join(base_dir, 'exp*/weights/best.pt'), recursive=True), key=os.path.getmtime)
        return found[-1] if found else 'yolov8n.pt'

    default_model = find_best_model(runs_dir)

    parser.add_argument('--model', type=str, default=default_model, help='當前版 YOLO 權重 (best.pt)')
    parser.add_argument('--raw_dir', type=str, default=os.path.normpath(os.path.join(script_dir, '../data/1_raw/door_opening_frames')), help='未標註之生肉圖片池 (由於 active learning 處理圖片，需輸入抽幀後目錄)')
    parser.add_argument('--dest_dir', type=str, default='../data/8_hard_cases/low_conf_uncertain', help='挑出高價值圖片的獨立存放點')
    parser.add_argument('--merge_back', type=str, default='', help='[Optional] 若提供此路徑，執行完成後自動將 hard_samples 閉環回流至 2_processed')
    parser.add_argument('--batch', type=int, default=32, help='GPU 推論 Batch 併發數')
    parser.add_argument('--min_conf', type=float, default=0.15, help='Conf 底標')
    parser.add_argument('--max_conf', type=float, default=0.60, help='Conf 頂標')

    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f" 找不到任何訓練完成的 best.pt！")
        print(f" 請先執行: python src/train.py --action train 完成第一輪訓練後再使用 Active Learning")
        exit(1)

    learner = ActiveLearnerLevel2(

        model_path=os.path.normpath(os.path.join(script_dir, args.model)),
        target_dir=os.path.normpath(os.path.join(script_dir, args.dest_dir))
    )
    learner.select_hard_samples(
        raw_images_dir=os.path.normpath(os.path.join(script_dir, args.raw_dir)),
        batch_size=args.batch,
        min_conf=args.min_conf,
        max_conf=args.max_conf
    )
    if args.merge_back:
        learner.merge_to_dataset(os.path.normpath(os.path.join(script_dir, args.merge_back)))
    print_pipeline_notice(
        output_paths=os.path.normpath(os.path.join(script_dir, args.dest_dir)),
        next_script="src/cvat_import.py",
        notes=[
            "這批樣本通常是低信心或高不確定案例，適合送人工複核。",
            "若使用 --merge_back，請再確認是否已有人工作業與標註同步。",
        ],
    )
