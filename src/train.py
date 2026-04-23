import sys
import os
import json
import yaml
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from anti_gravity.pipeline_notice import print_pipeline_notice
from anti_gravity.settings import settings

# =====================================================================
# train_base 訓練參數 >> C:\antigravity\src\train.py
# dataset_validator 資料集健康度檢查 >> C:\antigravity\src\anti_gravity\dataset_validator.py
# =====================================================================
# 1 Logging System
# =====================================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("YOLOv8Trainer")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

# =====================================================================
# 2 Experiment & Governance Tracker (v1.2)
# =====================================================================
class ExperimentTracker:
    def __init__(self, history_file=None, logger=None):
        self.history_file = history_file or settings.paths.experiments / "experiments_history.json"
        self.logger = logger
        self.global_best_info_file = settings.paths.experiments / "global_best_info.json"
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
    def check_promotion_gate(self, metrics_dict, new_weights_path, train_mode="rebuild", task_tag="general"):
        """
        三層式晉升門檻 (Promotion Gates) v1.2
        1. Common Gate: Ghost Veto & Lineage Check
        2. Mode Gate: Rebuild (Baseline) vs Incremental (Patch)
        3. Task Gate: 特定任務指標 (如 open_fn_repair)
        """
        import shutil
        import subprocess
        
        current_fitness = 0.1 * metrics_dict['mAP50'] + 0.9 * metrics_dict.get('mAP50_95', 0.0)
        current_recall = metrics_dict['recall']
        current_open_recall = metrics_dict.get('open_recall', current_recall)
        
        # 決定檔案命名與存放路徑
        if train_mode == "rebuild":
            dest_dir = settings.paths.models_baselines
            weight_name = f"door_base_{datetime.now().strftime('%Y%m%d_%H%M')}.pt"
        else:
            dest_dir = settings.paths.models_incremental
            weight_name = f"door_inc_{task_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.pt"
            
        os.makedirs(dest_dir, exist_ok=True)
        final_dest_path = dest_dir / weight_name
        
        global_best_path = settings.paths.models_promoted / "global_best.pt"
        os.makedirs(settings.paths.models_promoted, exist_ok=True)
        
        is_promoted = False
        reason = ""
        ghost_stats = {}
        
        # --- [Common Gate] Ghost 背景拒判測試 (SoC: 由 Tracker 驅動) ---
        ghost_pass = True
        ghost_eval_dir = settings.paths.evaluations / "ghost/latest_gate"
        hist_ghost_json = settings.paths.evaluations / "ghost/global_best/ghost_eval.json"
        
        try:
            eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'def', 'eval_ghosts.py')
            if os.path.exists(eval_script):
                cmd = [sys.executable, eval_script, "--model", str(new_weights_path), "--output", str(ghost_eval_dir)]
                if global_best_path.exists():
                    cmd += ["--baseline", str(global_best_path)]
                
                self.logger.info(f" [Governance] 啟動 Ghost 背景測試 (Mode: {train_mode}, Task: {task_tag})...")
                
                # 修正：注入 PYTHONPATH 確保子程序能找到 anti_gravity 套件
                import copy
                env = copy.deepcopy(os.environ)
                src_root = str(Path(__file__).parent.absolute()) # 指向 src/
                env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
                
                subprocess.check_call(cmd, env=env)
                
                with open(os.path.join(ghost_eval_dir, 'ghost_eval.json'), 'r') as f:
                    ghost_stats = json.load(f)
                
                if os.path.exists(hist_ghost_json):
                    with open(hist_ghost_json, 'r') as f:
                        old_ghost_stats = json.load(f)
                    # Veto 門檻：總誤報不應大幅上升
                    if ghost_stats['any@0.25'] > old_ghost_stats['any@0.25'] + 2:
                        ghost_pass = False
                        reason = f"Ghost Veto: 誤報劣化 ({old_ghost_stats['any@0.25']} -> {ghost_stats['any@0.25']})"
            else:
                self.logger.warning(" [Governance] 找不到 eval_ghosts.py，跳過 Ghost 檢查。")
        except Exception as e:
            self.logger.warning(f" [Governance] Ghost 評估過程出錯: {e}")

        # --- [Mode & Task Gates] 判定 ---
        reason = "未達晉升標準"
        if not os.path.exists(self.global_best_info_file):
            is_promoted = True
            reason = "初代基準建立。"
        else:
            with open(self.global_best_info_file, 'r', encoding='utf-8') as f:
                hist = json.load(f)
            h_fitness = hist.get("fitness", 0.0)
            h_open_recall = hist.get("metrics", {}).get("open_recall", 0.0)

            if not ghost_pass:
                is_promoted = False # Ghost 一票否決
            else:
                if train_mode == "rebuild":
                    # Rebuild 模式：Fitness 不退步即晉升，提升 2% 視為強晉升
                    if current_fitness >= h_fitness:
                        is_promoted = True
                        reason = f"Rebuild 成功: Fitness穩定或提升 ({current_fitness:.4f} >= {h_fitness:.4f})"
                    else:
                        is_promoted = False
                        reason = f"Rebuild 失敗: 未達舊標準 ({current_fitness:.4f} < {h_fitness:.4f})"
                else:
                    # Incremental 模式：Recall 保護機制 (核心 Open Recall 不退步過多)
                    if current_open_recall < (h_open_recall - 0.03):
                        is_promoted = False
                        reason = f"增量保護: Open Recall 退步過多 ({h_open_recall:.4f} -> {current_open_recall:.4f})"
                    else:
                        is_promoted = True
                        reason = "增量成功: 通過 Recall 安全防線。"

        # --- 執行晉升與 Registry 註冊 ---
        if is_promoted:
            try:
                shutil.copy2(new_weights_path, global_best_path)
                shutil.copy2(new_weights_path, final_dest_path)
                
                # 更新 Registry (Model Identity)
                self.save_to_registry(weight_name, {
                    "train_mode": train_mode,
                    "task_tag": task_tag,
                    "metrics": metrics_dict,
                    "ghost_stats": ghost_stats,
                    "parent_model": hist.get("model_name", "official") if 'hist' in locals() else "official"
                })

                with open(self.global_best_info_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "promoted_at": datetime.now().isoformat(),
                        "model_name": weight_name,
                        "fitness": round(current_fitness, 4),
                        "metrics": metrics_dict,
                        "ghost_metrics": ghost_stats,
                        "reason": reason
                    }, f, indent=4)
                self.logger.info(f"🏆 模型晉升成功！權重已保存至: {weight_name}")
            except Exception as e:
                self.logger.error(f"晉升寫入時發生錯誤: {e}")
        
        return is_promoted, reason

    def save_to_registry(self, weight_name, metadata):
        """實作 Model Registry JSON 註冊"""
        reg_path = settings.paths.models_registry / f"{Path(weight_name).stem}.json"
        metadata["registered_at"] = datetime.now().isoformat()
        metadata["lineage"] = {
            "dataset_version": getattr(self, 'ds_version', 'unknown'),
            "dataset_hash": getattr(self, 'ds_hash', 'unknown'),
            "config_hash": getattr(self, 'cfg_hash', 'unknown')
        }
        with open(reg_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        self.logger.info(f"📄 模型註冊表已更新: {reg_path.name}")

    def log_experiment(self, dataset_info, model_weights, metrics, save_dir):
        """紀錄實驗歷程並保存 Hash 資訊供血統追蹤"""
        self.ds_version = dataset_info.get('version')
        self.ds_hash = dataset_info.get('manifest_hash')
        self.cfg_hash = dataset_info.get('config_hash')
        
        mp, mr, map50, map50_95 = metrics.box.mean_results() if hasattr(metrics.box, 'mean_results') else (
            getattr(metrics.box, 'mp', 0.0), getattr(metrics.box, 'mr', 0.0), metrics.box.map50, 0.0
        )
        
        # 嘗試擷取真實的 class-wise recall (假設 open 為類別 0)
        open_recall = mr
        try:
            if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'R'):
                classes = list(metrics.box.ap_class_index)
                if 0 in classes:
                    idx = classes.index(0)
                    open_recall = metrics.box.R[idx]
        except Exception as e:
            self.logger.warning(f"無法擷取 class-wise recall: {e}")
        
        metrics_dict = {
            "mAP50": round(map50, 4),
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "open_recall": round(open_recall, 4)
        }
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.ds_version,
            "dataset_hash": self.ds_hash,
            "base_model": model_weights,
            "metrics": metrics_dict,
            "save_dir": save_dir
        }
        
        try:
            with open(self.history_file, 'r+', encoding='utf-8') as f:
                history = json.load(f)
                history.append(record)
                f.seek(0); json.dump(history, f, indent=4)
        except Exception:
            pass
            
        # --- 改為在外部呼叫，以便帶入晉升狀態 ---
        # self._update_markdown_report(record)
            
        return metrics_dict, len(history) - 1, record

    def update_markdown_report(self, record, status="已存檔"):
        """自動維護 data/7_experiments/training_history.md (表格式)"""
        md_file = settings.paths.experiments / "training_history.md"
        os.makedirs(os.path.dirname(md_file), exist_ok=True)
        
        if not os.path.exists(md_file):
            header = "# 🚀 訓練實驗歷程 (Training History)\n\n"
            header += "| 序號 | 實驗名稱 | 日期時間 | 資料集版本 | mAP50 | 狀態 |\n"
            header += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(header)
        
        try:
            ts = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
            name = os.path.basename(record["save_dir"])
            ds = record["dataset"]
            map50 = record["metrics"]["mAP50"]
            
            # 簡潔的一行紀錄，包含動態 status
            new_line = f"| - | **{name}** | {ts} | {ds} | {map50:.4f} | {status} |\n"
            
            with open(md_file, 'a', encoding='utf-8') as f:
                f.write(new_line)
        except Exception as e:
            import traceback
            self.logger.warning(f"更新 Markdown 報表失敗: {e}")
            traceback.print_exc()

# =====================================================================
# 3 Hyperparameter Management
# =====================================================================
class HyperparameterConfig:
    def __init__(self, config_path, mode, logger):
        self.config_path = config_path
        self.mode = mode
        self.logger = logger
        self.config = self._load_from_yaml()
        # 機驗資訊：Config Hash
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:12]

    def _load_from_yaml(self):
        """從 YAML 中讀取對應模式的參數"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"找不到訓練設定檔: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)
            
        if self.mode not in all_configs:
            raise KeyError(f"設定檔中找不到對應模式 '{self.mode}' 的參數區塊。")
            
        cfg = all_configs[self.mode]
        return cfg

# =====================================================================
# 4 YOLO Trainer
# =====================================================================
class YOLOv8Trainer:
    def __init__(self, model_weights='yolov8s.pt', logger=None):
        self.logger = logger
        self.model = YOLO(model_weights)
        if self.logger: self.logger.info(f" 模型載入: {model_weights}")

    def train(self, data_yaml, hyper_params, project=None, name='exp'):
        if project is None: project = settings.paths.experiments
        try:
            self.logger.info(f"🚀 訓練啟動！您可以開啟另一個終端機輸入 'tensorboard --logdir {project}' 來監看曲線。")
            results = self.model.train(
                data=data_yaml, 
                project=project, 
                name=name, 
                exist_ok=True,    # 允許覆寫同名資料夾，避免產生大量 exp1, exp2
                visualize=True,   # 儲存特徵圖
                plots=True,       # 產生訓練圖表
                **hyper_params
            )
            best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
            
            # 備份至 latest 供基礎使用
            latest_dir = settings.paths.models / "latest"
            os.makedirs(latest_dir, exist_ok=True)
            import shutil
            shutil.copy2(best_weights, latest_dir / "latest_best.pt")
            
            return best_weights
        except Exception as e:
            raise RuntimeError(f"訓練中斷: {e}")

    def evaluate(self, weights_path, data_yaml):
        eval_model = YOLO(weights_path)
        metrics = eval_model.val(data=data_yaml, conf=0.25)
        return metrics

# =====================================================================
# 5 Orchestrator
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLOps Level 3 Governance Trainer')
    parser.add_argument('--action', type=str, default='train', choices=['train', 'tune'])
    parser.add_argument('--data', type=str, default=str(settings.paths.augment / "current/dataset.yaml"))
    parser.add_argument('--config', type=str, default=str(settings.paths.configs / "experiments/train_base.yaml"))
    # pt_load = str(settings.paths.experiments / "exp_rebuild_general_0421_0902/weights/best.pt")
    pt_load = str(settings.paths.experiments / "incremental/exp_incremental_general_0421_1002/weights/best.pt")
    # pt_load = str(settings.paths.artifacts / "yolov8s.pt")
    parser.add_argument('--weights', type=str, default=pt_load)
    # Rebuild: 適合「打地基」、Incremental: 適合「餵新圖」、Specialized: 適合「修難樣本」
    parser.add_argument('--mode', type=str, default='specialized', choices=['rebuild', 'incremental', 'specialized'])
    parser.add_argument('--task', type=str, default='general', help='例如: open_fn_repair, close_fp_suppression')
    parser.add_argument('--dataset_version', type=str, default='v0.8.2_auto')
    parser.add_argument('--purge', action='store_true', help='是否在訓練前自動剔除空標籤樣本(背景圖)')
    
    args = parser.parse_args()
    logger = setup_logger(settings.paths.storage_logs)
    
    # 啟動追蹤器與配置管理
    tracker = ExperimentTracker(logger=logger)
    config_mgr = HyperparameterConfig(args.config, args.mode, logger)
    
    # [模式日誌標記]
    logger.info("="*60)
    logger.info(f"🚀 啟動 MLOps 訓練流水線")
    logger.info(f"   【當前模式】: {args.mode.upper()}")
    logger.info(f"   【任務標籤】: {args.task}")
    logger.info(f"   【資料版本】: {args.dataset_version}")
    logger.info(f"   【初始權重】: {os.path.basename(args.weights)}")
    logger.info(f"   【參數摘要】: LR={config_mgr.config.get('lr0')}, Epochs={config_mgr.config.get('epochs')}")
    logger.info("="*60)
    
    # 載入模型並訓練
    trainer = YOLOv8Trainer(model_weights=args.weights, logger=logger)
    
    if args.action == 'train':
        # [新增] 訓練前強制執行 Dataset Sanity Check
        from anti_gravity.dataset_validator import validate_dataset, purge_empty_labels
        try:
            validate_dataset(args.data)
            
            # [新增] 自動清理 YOLO 快取，確保資料同步
            data_dir = Path(args.data).parent
            for cache_file in data_dir.rglob("*.cache"):
                try:
                    os.remove(cache_file)
                    logger.info(f"🧹 已清理舊的快取檔案: {cache_file.name}")
                except Exception: pass

            # 如果使用者啟動了 --purge，則執行剔除空標籤
            if args.purge:
                purge_empty_labels(args.data)
        except RuntimeError as e:
            logger.error(f"❌ 終止訓練 (Dataset Sanity Check Failed): {e}")
            logger.error("🚫 請先修復資料集問題再重新啟動！")
            sys.exit(1)

        exp_name = f"exp_{args.mode}_{args.task}_{datetime.now().strftime('%m%d_%H%M')}"
        best_pt = trainer.train(
            data_yaml=args.data, 
            hyper_params=config_mgr.config, 
            project=settings.paths.experiments / args.mode,
            name=exp_name
        )
        metrics = trainer.evaluate(best_pt, data_yaml=args.data)

        # [新增] Fail-Fast 防護: 驗證集失效直接報錯
        map50 = getattr(metrics.box, 'map50', 0.0) if hasattr(metrics, 'box') else 0.0
        if map50 == 0.0:
            logger.error("❌ [致命錯誤] 驗證集 mAP50 為 0.0000！資料/標籤極可能已損壞或失效。")
            logger.error("🚫 模型不會被晉升，請檢查儲存路徑與 YAML 設定。")
            sys.exit(1)

        # 擷取血統資訊: 真實讀取 dataset.yaml 的內容進行 Hash，而非僅 Hash 路徑字串
        try:
            with open(args.data, 'r', encoding='utf-8') as f:
                ds_hash = hashlib.sha256(f.read().encode()).hexdigest()[:12]
        except Exception:
            ds_hash = hashlib.sha256(str(args.data).encode()).hexdigest()[:12]
            
        metrics_dict, record_idx, record = tracker.log_experiment(
            dataset_info={
                "version": args.dataset_version,
                "manifest_hash": ds_hash,
                "config_hash": config_mgr.config_hash
            },
            model_weights=args.weights,
            metrics=metrics,
            save_dir=os.path.dirname(os.path.dirname(best_pt))
        )

        # 執行治理門檻判定
        is_promoted, reason = tracker.check_promotion_gate(metrics_dict, best_pt, train_mode=args.mode, task_tag=args.task)
        if is_promoted:
            logger.info(" [MLOps] 模型已完成晉升與身分註冊。")
        else:
            logger.warning(" [MLOps] 模型未通過治理門檻，僅保留實驗結果。")
            
        # 最後將帶有真實原因的記錄寫入 Markdown
        tracker.update_markdown_report(record, status=reason)

    print_pipeline_notice(
        output_paths=[str(settings.paths.models_promoted / "global_best.pt")],
        next_script="src/analyze_errors.py",
        notes=[f"當前模式: {args.mode}", f"任務標籤: {args.task}"]
    )
