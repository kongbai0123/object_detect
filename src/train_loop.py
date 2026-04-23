import os
import shutil
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from anti_gravity.settings import settings

import argparse

class TrainingOrchestrator:
    def __init__(self, start_version=None, init_weights=None, force_mode=None):
        self.root = Path(__file__).resolve().parent.parent
        self.force_mode = force_mode
        self.history_file = settings.paths.experiments / "global_best_info.json"
        
        # 動態掃描 goldenset/versions 下的所有資料夾 (確保有順序)
        versions_dir = settings.paths.assets / "goldenset" / "versions"
        if versions_dir.exists():
            # 過濾出有包含 images 或 labels 的真實資料夾
            all_versions = sorted([d.name for d in versions_dir.iterdir() if d.is_dir()])
        else:
            all_versions = ["1_img", "2_img", "3_img", "4_img"]
        
        # 處理自訂起始點
        if start_version == "all":
            self.versions = ["all"]
            self.is_resuming = True
            print("[INFO] Full Merge Mode activated. Integrating all versions.")
        elif start_version and start_version in all_versions:
            start_idx = all_versions.index(start_version)
            self.versions = all_versions[start_idx:]
            self.is_resuming = True # 標記為接續模式
        elif start_version:
            # 防呆：如果輸入了自訂名稱但不在列表中，當作單次任務處理
            self.versions = [start_version]
            self.is_resuming = True
        else:
            self.versions = all_versions
            self.is_resuming = False
            
        self.current_best_map = 0.0
        
        # 處理自訂初始權重
        if init_weights:
            self.current_weights = init_weights
        elif self.is_resuming and (settings.paths.models_promoted / "global_best.pt").exists():
            # 如果是接續執行，且有冠軍模型，自動繼承
            self.current_weights = str(settings.paths.models_promoted / "global_best.pt")
        else:
            self.current_weights = "yolov8s.pt"
            
        # 如果使用 yolov8s.pt，代表是從頭訓練，強制將歷史成績歸零
        if self.current_weights == "yolov8s.pt":
            self.is_resuming = False

    def get_last_map(self):
        """從全局紀錄中讀取冠軍的 mAP50"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("metrics", {}).get("mAP50", 0.0)
        return 0.0
        
    def get_latest_experiment_map(self):
        """從 experiments_history.json 讀取剛跑完的那一次實驗的 mAP"""
        hist_file = settings.paths.experiments / "experiments_history.json"
        if hist_file.exists():
            with open(hist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:
                    return data[-1].get("metrics", {}).get("mAP50", 0.0)
        return 0.0

    def run_step(self, script_name, args=[]):
        """執行子腳本並監控狀態"""
        script_path = self.root / "src" / script_name
        cmd = [sys.executable, str(script_path)] + args
        print(f"\n[LOOP] 正在執行: {' '.join(cmd)}")
        
        # 注入 PYTHONPATH 確保模組可用
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"腳本 {script_name} 執行失敗，回傳值: {result.returncode}")

    def clean_workspace(self):
        """
        [HARD RECOVERY] 強制清理工作空間的暫存區與 YOLO 快取。
        防止舊資料殘留 (Ghost data) 或失效快取 (.cache) 導致訓練崩潰。
        """
        print("\n--- [Step 0: Workspace Cleaning] ---")
        
        # 1. 清除暫存工作目錄
        targets = [
            settings.paths.workspace / "split",
            settings.paths.workspace / "balance",
            settings.paths.workspace / "augment"
        ]
        
        for t in targets:
            current_dir = t / "current"
            if current_dir.exists():
                print(f"[*] 清理目錄: {current_dir}")
                shutil.rmtree(current_dir)
                current_dir.mkdir(parents=True, exist_ok=True)

        # 2. 深度搜尋並清除所有 YOLO 快取 (.cache)
        # 涵蓋整個 storage 目錄，包括 workspace 和 assets/validation
        print("[*] 正在清除系統中的所有 .cache 文件...")
        search_roots = [settings.paths.workspace, settings.paths.assets]
        cache_count = 0
        for root in search_roots:
            if not root.exists(): continue
            for cache_file in root.rglob("*.cache"):
                try:
                    cache_file.unlink()
                    cache_count += 1
                except Exception as e:
                    print(f"無法刪除快取 {cache_file}: {e}")
        
        if cache_count > 0:
            print(f"✅ 已成功清除 {cache_count} 個 .cache 文件。")
        else:
            print("✅ 未發現殘留快取。")

    def start_loop(self):
        print("="*60)
        print("[START] Starting MLOps Iterative Training Pipeline")
        print(f"   Start version: {self.versions[0]}")
        print(f"   Initial weights: {self.current_weights}")
        print("="*60)

        # 讀取初始基準 mAP (如果不是從頭開始，或者有歷史紀錄)
        self.current_best_map = self.get_last_map() if self.is_resuming else 0.0
        print(f"[*] 當前系統基準 mAP50: {self.current_best_map:.4f}")

        for i, version in enumerate(self.versions):
            print(f"\n\n### [Iteration Stage {i+1}/{len(self.versions)}]: Target Dataset {version} ###")
            
            # 檢查資料夾是否存在 (虛擬版本 "all" 跳過此檢查)
            if version != "all":
                v_path = settings.paths.assets / "goldenset/versions" / version
                if not v_path.exists():
                    print(f"[WARN] Version folder not found: {version}, stopping iteration.")
                    break

            try:
                # 0. 衛生檢查
                self.clean_workspace()

                # 1. 準備階段: 指定資料來源
                print(f"--- [Step 1: Data Preparation ({version})] ---")
                
                if version == "all":
                    # 整合模式：直接傳遞 "all" 字串觸發 Split-then-Merge 邏輯
                    self.run_step("split_dataset.py", ["--input", "all"])
                else:
                    # 單一版本模式
                    v_path = settings.paths.assets / "goldenset/versions" / version
                    self.run_step("split_dataset.py", ["--input", str(v_path)])
                
                # 必須指定輸入為 split 的結果，否則它會預設去抓自動標註 (auto_ann) 的暫存區！
                self.run_step("balance_dataset.py", ["--input", "split"])
                
                self.run_step("augment_dataset.py")

                # === 🤖 決策大腦 (Adaptive Mode Selection) ===
                if self.force_mode:
                    current_mode = self.force_mode
                else:
                    # 決策 1：第一輪 (i==0) 強制打地基 (Rebuild)；後續改用溫和吸收 (Incremental)
                    current_mode = "rebuild" if i == 0 else "incremental"
                
                print(f"--- [Step 2: Training (Mode: {current_mode.upper()})] ---")
                # 訓練階段：將自動讀取 train_base.yaml 中的最新設定 (epochs: 60, imgsz: 832)
                train_args = [
                    "--weights", self.current_weights,
                    "--mode", self.force_mode or "incremental",
                    "--task", f"auto_iter_{version}"
                ]
                self.run_step("train.py", train_args)

                # 3. 效能驗證
                time.sleep(2)
                # 取得「剛剛訓練完」的最新一次實驗成績
                new_map = self.get_latest_experiment_map()
                print(f"[RESULT] Old mAP={self.current_best_map:.4f} -> New mAP={new_map:.4f}")

                if new_map > self.current_best_map:
                    print(f"[PROMOTED] Performance improved! Absorbed {version}.")
                    self.current_best_map = new_map
                    self.current_weights = str(settings.paths.models_promoted / "global_best.pt")
                else:
                    print(f"[WARN] {current_mode} mode did not improve. (mAP {new_map:.4f} <= {self.current_best_map:.4f})")
                    
                    # Decision 2: If Incremental fails, fallback to Rebuild
                    if current_mode == "incremental":
                        print("[FALLBACK] Switching to 'rebuild' mode for retry...")
                        self.run_step("train.py", [
                            "--mode", "rebuild",
                            "--weights", self.current_weights,
                            "--task", f"auto_iter_{version}_retry"
                        ])
                        
                        time.sleep(2)
                        retry_map = self.get_latest_experiment_map()
                        print(f"[RETRY RESULT] Old mAP={self.current_best_map:.4f} -> New mAP={retry_map:.4f}")
                        
                        if retry_map > self.current_best_map:
                            print(f"[PROMOTED] Fallback succeeded! Model adapted to new distribution.")
                            self.current_best_map = retry_map
                            self.current_weights = str(settings.paths.models_promoted / "global_best.pt")
                            continue
                            
                    # Decision 3: Both strategies failed
                    print(f"[STOP] Fallback also failed to improve performance.")
                    print(f"[ANALYSIS] Model could not improve after adding {version}. Check labels with analyze_errors.py.")
                    break

            except Exception as e:
                print(f"[ERROR] Iteration interrupted: {e}")
                break

        print("\n" + "="*60)
        print("[DONE] Iterative Training Orchestrator finished.")
        print(f"   Final best mAP: {self.current_best_map:.4f}")
        print("="*60)

def interactive_setup():
    print("\n" + "="*60)
    print("[MLOps] Training Orchestrator - Interactive Setup")
    print("="*60)
    print("Select training strategy:")
    print("  1. Train from scratch (yolov8s.pt)")
    print("  2. Resume from existing model (global_best.pt)")
    print("  3. Custom start folder and weights")
    
    choice = input("\n>> Enter option [1/2/3]: ").strip()
    
    start_version = "1_img"
    init_weights = "yolov8s.pt"
    mode = None
    
    if choice == "1":
        init_weights = "yolov8s.pt"
        print("[INFO] Starting from scratch, iterating from 1_img.")
        
    elif choice == "2":
        print("\nVersion folders: 1_img, 2_img, 3_img, 4_img, 5_img, temp, temp2")
        start_version = input(">> 1. Enter start version (e.g. 2_img): ").strip()
        if not start_version: start_version = "1_img"
        
        exp_name = input(">> 2. Enter experiment folder name (e.g. exp_rebuild_general_0421_0935): ").strip()
        
        if exp_name:
            base_exp = settings.paths.experiments
            found = False
            
            if exp_name.endswith(".pt"):
                paths_to_try = [base_exp / exp_name, Path(exp_name)]
                for p in paths_to_try:
                    if p.exists():
                        init_weights = str(p)
                        found = True
                        break
            
            if not found:
                for p in base_exp.rglob(exp_name):
                    if p.is_dir():
                        best_pt = p / "weights" / "best.pt"
                        if best_pt.exists():
                            init_weights = str(best_pt)
                            found = True
                            print(f"[OK] Found weights: {init_weights}")
                            break
            
            if not found:
                print(f"[WARN] Could not find best.pt for '{exp_name}', falling back to yolov8s.pt")
                init_weights = "yolov8s.pt"
        else:
            print("[WARN] No model name entered, using yolov8s.pt")
            init_weights = "yolov8s.pt"
        
    elif choice == "3":
        start_version = input(">> Enter start folder name (default: videos): ").strip()
        init_weights = input(">> Enter absolute path to .pt weights (default: yolov8s.pt): ").strip()
        
        print("\nAvailable training modes:")
        print("  - rebuild: High LR, good for foundation building")
        print("  - incremental: Medium LR, good for gentle absorption")
        print("  - specialized: Low LR, no mosaic, for hard sample targeting")
        mode_input = input(">> Enter mode [rebuild/incremental/specialized] (default: specialized): ").strip().lower()
        
        if not start_version: start_version = "videos"
        if not init_weights: init_weights = str(settings.paths.models_promoted / "global_best.pt")
        mode = mode_input if mode_input in ['rebuild', 'incremental', 'specialized'] else 'specialized'
        
    else:
        print("[WARN] Invalid option, defaulting to scratch training (yolov8s.pt, 1_img).")
        init_weights = "yolov8s.pt"
        
    return start_version, init_weights, mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Iterative Training Orchestrator")
    parser.add_argument("--interactive", action="store_true", default=True, help="使用互動式選單")
    parser.add_argument("--start", "--start_from", dest="start_from", type=str, default=None, help="Start version folder name")
    parser.add_argument("--weights", type=str, default=None, help="Force initial weights path")
    parser.add_argument("--mode", type=str, default=None, choices=['rebuild', 'incremental', 'specialized'], help="Force training mode")
    args = parser.parse_args()

    # 如果使用者有在指令後加上參數，就跳過互動選單
    if args.start_from or args.weights or args.mode:
        start_ver = args.start_from
        init_w = args.weights
        mode = args.mode
    else:
        start_ver, init_w, mode = interactive_setup()

    orchestrator = TrainingOrchestrator(start_version=start_ver, init_weights=init_w, force_mode=mode)
    orchestrator.start_loop()
