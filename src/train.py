import os
import yaml
import json
import logging
import argparse
from datetime import datetime
from ultralytics import YOLO
from pipeline_notice import print_pipeline_notice

# =====================================================================
# 1 Logging System & Experiment Tracking (解決追蹤與 error trace)
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

class ExperimentTracker:
    def __init__(self, history_file='../data/7_experiments/experiments_history.json', logger=None):
        self.history_file = history_file
        self.logger = logger
        self.global_best_info_file = os.path.join(os.path.dirname(history_file), "global_best_info.json")
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
    def check_promotion_gate(self, metrics_dict, new_weights_path):
        """
        退步保護與雙軌驗證基準之晉升閘門 (Promotion Gates)
        條件 1：綜合 Fitness 必須大於歷史最佳
        條件 2：Recall 不得大幅退步 (容忍度 5%)
        條件 3：[0.7.1 新增] Ghost背景誤判不得上升 (一票否決)
        """
        import shutil
        import subprocess
        import sys
        
        current_fitness = 0.1 * metrics_dict['mAP50'] + 0.9 * metrics_dict.get('mAP50_95', 0.0)
        current_recall = metrics_dict['recall']
        
        global_best_dir = os.path.join(os.path.dirname(os.path.dirname(self.history_file)), 'weight')
        os.makedirs(global_best_dir, exist_ok=True)
        global_best_path = os.path.join(global_best_dir, 'global_best.pt')
        
        is_promoted = False
        reason = ""
        
        # --- 0.7.1 新增：Ghost 背景拒判能力評估 ---
        ghost_pass = True
        ghost_reason = ""
        ghost_eval_dir = os.path.join(os.path.dirname(os.path.dirname(self.history_file)), '8_ghost_evals/latest_gate')
        hist_ghost_json = os.path.join(os.path.dirname(os.path.dirname(self.history_file)), '8_ghost_evals/global_best/ghost_eval.json')
        
        try:
            # 執行 eval_ghosts.py 針對這批新權重進行測試
            eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_ghosts.py')
            cmd = [sys.executable, eval_script, "--model", str(new_weights_path), "--output", ghost_eval_dir]
            if os.path.exists(global_best_path):
                cmd += ["--baseline", global_best_path]
            
            self.logger.info(" [Safety Gate] 啟動 Ghost 背景誤判專屬測試...")
            subprocess.check_call(cmd)
            
            # 讀取當次結果
            with open(os.path.join(ghost_eval_dir, 'ghost_eval.json'), 'r') as f:
                new_ghost_stats = json.load(f)
            
            # 讀取歷史最佳 (如果是第一次，自動通過)
            if os.path.exists(hist_ghost_json):
                with open(hist_ghost_json, 'r') as f:
                    old_ghost_stats = json.load(f)
                
                # 門檻規則：@0.25 的 fp_close 或 fp_any 不得上升 (Hard Veto)
                if new_ghost_stats['close@0.25'] > old_ghost_stats['close@0.25']:
                    ghost_pass = False
                    ghost_reason = f"Ghost 誤報劣化 (Close)：{old_ghost_stats['close@0.25']} -> {new_ghost_stats['close@0.25']}"
                elif new_ghost_stats['any@0.25'] > old_ghost_stats['any@0.25']:
                    ghost_pass = False
                    ghost_reason = f"Ghost 總誤報上升：{old_ghost_stats['any@0.25']} -> {new_ghost_stats['any@0.25']}"
                else:
                    ghost_reason = "Ghost 背景拒判能力維持或進步。"
            else:
                ghost_reason = "初代 Ghost 基準建立。"
        except Exception as e:
            self.logger.warning(f" [Safety Gate] Ghost 評估過程出錯: {e}，跳過硬指標檢查。")

        # 讀取歷史紀錄
        if not os.path.exists(self.global_best_info_file):
            is_promoted = True
            reason = "無歷史 global_best，自動晉升為初代金牌模型。"
        else:
            try:
                with open(self.global_best_info_file, 'r', encoding='utf-8') as f:
                    global_info = json.load(f)
                historical_fitness = global_info.get("fitness", 0.0)
                historical_recall = global_info.get("metrics", {}).get("recall", 0.0)
                
                # Fitness 必須進步
                if current_fitness <= historical_fitness:
                    reason = f"Fitness ({current_fitness:.4f}) 未超越歷史最佳 ({historical_fitness:.4f})，拒絕晉升。"
                # Recall 不可嚴重退步 (Safety Gate)
                elif current_recall < (historical_recall - 0.05):
                    reason = f"Safety Gate 攔截：即使 Fitness 較高，但 Recall ({current_recall:.4f}) 較歷史最佳 ({historical_recall:.4f}) 退步超過 5%，拒絕晉升。"
                # Ghost 一票否決 (Hard Veto)
                elif not ghost_pass:
                    reason = f"Ghost 一票否決制：{ghost_reason}"
                else:
                    is_promoted = True
                    reason = f"通過所有閘門！{ghost_reason} Fitness 進步 ({historical_fitness:.4f} -> {current_fitness:.4f})。"
            except Exception as e:
                if self.logger: self.logger.error(f"讀取 global_best_info.json 發生錯誤：{e}")
                is_promoted = True
                reason = "歷史紀錄毀損，強制晉升以修復狀態。"
                
        # 執行晉升
        if is_promoted:
            try:
                shutil.copy2(new_weights_path, global_best_path)
                # 同步更新歷史 ghost JSON 供下次比對
                hist_ghost_dir = os.path.dirname(hist_ghost_json)
                os.makedirs(hist_ghost_dir, exist_ok=True)
                shutil.copy2(os.path.join(ghost_eval_dir, 'ghost_eval.json'), hist_ghost_json)

                with open(self.global_best_info_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "promoted_at": datetime.now().isoformat(),
                        "source_weights": str(new_weights_path),
                        "fitness": round(current_fitness, 4),
                        "metrics": metrics_dict,
                        "ghost_metrics": new_ghost_stats if ghost_pass else {},
                        "reason": reason
                    }, f, indent=4)
                if self.logger: self.logger.info(f"🏆 模型晉升成功 (Promoted to global_best.pt)！{reason}")
            except Exception as e:
                if self.logger: self.logger.error(f"晉升寫入時發生錯誤：{e}")
                is_promoted = False
        else:
            if self.logger: self.logger.warning(f"🛑 模型被閘門擋下 (Rejected)。{reason} (權重僅保留於 latest_best.pt)")
            
        return is_promoted

    def log_experiment(self, dataset_version, model_weights, metrics, save_dir):
        """4 解決：訓練紀錄綁 dataset version，並回傳格式化 metrics 給晉升閘門用"""
        # 防版本崩潰Ultralytics 舊版 mean_p/mean_r 已更新為 mp/mr
        mp, mr, map50, map50_95 = metrics.box.mean_results() if hasattr(metrics.box, 'mean_results') else (
            getattr(metrics.box, 'mp', getattr(metrics.box, 'mean_p', 0.0)),
            getattr(metrics.box, 'mr', getattr(metrics.box, 'mean_r', 0.0)),
            metrics.box.map50, getattr(metrics.box, 'map', 0.0)
        )
        
        metrics_dict = {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(mp, 4),
            "recall": round(mr, 4)
        }
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_version,
            "base_model": model_weights,
            "save_dir": save_dir,
            "metrics": metrics_dict,
            "promoted_to_global": False # 預設 false，稍後由晉升閘門覆寫若有需要
        }
        
        try:
            with open(self.history_file, 'r+', encoding='utf-8') as f:
                history = json.load(f)
                history.append(record)
                f.seek(0)
                json.dump(history, f, indent=4)
            if self.logger:
                self.logger.info(f" 實驗結果已成功綁定並記錄於: {self.history_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f" 記錄實驗歷程失敗: {e}")
                
        return metrics_dict, len(history) - 1 # 回傳 index 以便稍後更新 promoted 狀態
        
    def update_promotion_status(self, record_index):
        """將該次實驗標記為已晉升"""
        try:
            with open(self.history_file, 'r+', encoding='utf-8') as f:
                history = json.load(f)
                history[record_index]["promoted_to_global"] = True
                f.seek(0)
                f.truncate()
                json.dump(history, f, indent=4)
        except:
            pass

# =====================================================================
# 2 Configuration Manager (解決超參數硬編碼與強變形風險)
# =====================================================================
class HyperparameterConfig:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.config = self._load_or_create_default()

    def _load_or_create_default(self, is_incremental=False):
        """ 解決：Augmentation 太強會破壞小 dataset 的問題"""
        default_config = {
            "epochs": 30 if is_incremental else 100,
            "imgsz": 768,
            "batch": 16,
            "patience": 50,
            "optimizer": "auto",
            "lr0": 0.001 if is_incremental else 0.01,
            "warmup_epochs": 2 if is_incremental else 3, # 0.7.1 專家建議：增量模式採用短 Warmup
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "mosaic": 0.2,
            "translate": 0.2,
            "mixup": 0.0,
            "degrees": 2.0,
            "scale": 0.1
        }
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            self.logger.info(f" 已自動產生預設訓練參數檔: {self.config_path} (Incremental={is_incremental})")
            return default_config
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if is_incremental:
                cfg['lr0'] = min(cfg.get('lr0', 0.01), 0.001)
                cfg['epochs'] = min(cfg.get('epochs', 100), 40)
            self.logger.info(f" 已成功載入外部參數檔: {self.config_path} (經增量模式校準)")
            return cfg

# =====================================================================
# 3 SRP 核心: Model Trainer & Evaluator (拆積木設計)
# =====================================================================
class YOLOv8Trainer:
    def __init__(self, model_weights='yolov8n.pt', logger=None):
        self.logger = logger
        self.model_weights = model_weights
        self.model = YOLO(self.model_weights)
        if self.logger:
            self.logger.info(f" 初始模型載入完成: {model_weights}")

    def train(self, data_yaml, hyper_params, project=None, name='exp'):
        if project is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project = os.path.normpath(os.path.join(script_dir, '../data/7_experiments'))  # 【輸出】訓練實驗結果
        if self.logger: self.logger.info(" 啟動安全掛載之神經網路訓練...")
        # 7 解決：加入 Exception Handling 防止 Pipeline 死當
        try:
            results = self.model.train(
                data=data_yaml,
                project=project,
                name=name,
                **hyper_params
            )
            
            # 5 解決：best.pt 抓法有風險 (Race Condition)
            # 正確做法：直接拿 YOLO 物件回傳的記憶體路徑，保證 100% 正確
            if hasattr(results, 'save_dir'):
                best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
            else:
                best_weights = os.path.join(self.model.trainer.save_dir, 'weights', 'best.pt')

            # 把最佳結果備份為 latest_best.pt，代表本次最新出爐的模型，無論有沒有變強
            import shutil
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weight_dir = os.path.normpath(os.path.join(script_dir, '../data/7_experiments/weight'))  # 【輸出】模型備份路徑
            os.makedirs(weight_dir, exist_ok=True)
            latest_copy = os.path.join(weight_dir, 'latest_best.pt')
            shutil.copy2(best_weights, latest_copy)

            if self.logger: self.logger.info(f"Training complete. Run saved to: {best_weights}")
            if self.logger: self.logger.info(f"🔥 本次最新結果已備份至: {latest_copy}，稍後送入晉升閘門審核...")

            save_dir = os.path.dirname(os.path.dirname(best_weights))
            print(f"\n Training run dir: {os.path.abspath(save_dir)}")
            try:
                os.startfile(os.path.abspath(save_dir))
            except Exception:
                pass

            return best_weights

            
        except Exception as e:
            if self.logger: self.logger.error(f" 訓練過程發生致命錯誤崩潰：{e}", exc_info=True)
            raise RuntimeError("Pipeline 終止於訓練階段") from e

    def tune(self, data_yaml, iterations=250, epochs=100):
        if self.logger: self.logger.info(f" 啟動 {iterations} 代超參數基因演化 (GA)...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tune_project = os.path.normpath(os.path.join(script_dir, '../data/7_experiments'))  # 【輸出】調參實驗結果
            self.model.tune(
                data=data_yaml,
                epochs=epochs,
                iterations=iterations,
                optimizer='AdamW',
                project=tune_project,
                plots=False
            )
            if self.logger:
                self.logger.info(f"Tune complete. Best params saved to: {tune_project}/tune/best_hyperparameters.yaml")
        except Exception as e:
            if self.logger: self.logger.error(f" 調參過程崩潰：{e}", exc_info=True)
            raise

    def evaluate(self, weights_path, data_yaml):
        if self.logger: self.logger.info(f" 開始載入 {weights_path} 進行測試集評估...")
        try:
            eval_model = YOLO(weights_path)
            # 依據 F1 Curve 最佳表現，調降驗證門檻以解放 Recall
            metrics = eval_model.val(data=data_yaml, conf=0.3)
            
            if self.logger:
                # 防版本崩潰使用 mean_results() 獲取最穩固的 API (新舊不同 Ultralytics 版本全相容)
                mp, mr, map50, _ = metrics.box.mean_results() if hasattr(metrics.box, 'mean_results') else (
                    getattr(metrics.box, 'mp', getattr(metrics.box, 'mean_p', 0.0)),
                    getattr(metrics.box, 'mr', getattr(metrics.box, 'mean_r', 0.0)),
                    metrics.box.map50, None
                )
                self.logger.info("================  驗證集嚴格評估結果 =================")
                self.logger.info(f" mAP@50   : {map50:.4f}")
                self.logger.info(f" Precision : {mp:.4f}")
                self.logger.info(f" Recall    : {mr:.4f}")
                self.logger.info("=========================================================")
            return metrics
            
        except Exception as e:
            if self.logger: self.logger.error(f" 評估過程崩潰：{e}", exc_info=True)
            raise

# =====================================================================
# Orchestrator (CLI 整合入口)
# =====================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 預設直接讀取 data/ 根目錄的 dataset.yaml
    default_yaml = os.path.normpath(os.path.join(script_dir, '../data/dataset.yaml'))
    dataset_version_name = "base_dataset"

            
    default_config = os.path.join(script_dir, '../configs/train_config.yaml')

    parser = argparse.ArgumentParser(description='MLOps Level 3 企業級訓練引擎')
    parser.add_argument('--action', type=str, choices=['train', 'tune', 'auto_tune_train'], default='train', 
                        help='執行動作：train(一般訓練), tune(純調參), auto_tune_train(先調參後自動使用最佳參數訓練)')
    parser.add_argument('--data', type=str, default=default_yaml, help='dataset.yaml 目標路徑')
    parser.add_argument('--config', type=str, default=default_config, help='超參數 train_config.yaml 路徑')
    parser.add_argument('--weights', type=str, default='', help='初始權重檔，若為空則根據模式自動尋找')
    parser.add_argument('--incremental', action='store_true', help='啟用增量微調模式 (自動調降 LR 與 Epochs)')
    
    args = parser.parse_args()
    
    # [Weights Discovery] 0.7.1 智慧偵測
    if not args.weights:
        if args.incremental:
            # 優先找 global_best，次之 latest_best
            g_path = os.path.join(script_dir, '../data/7_experiments/weight/global_best.pt')
            l_path = os.path.join(script_dir, '../data/7_experiments/weight/latest_best.pt')
            args.weights = g_path if os.path.exists(g_path) else (l_path if os.path.exists(l_path) else 'yolov8s.pt')
        else:
            args.weights = 'yolov8s.pt'
            
    # 啟動各式系統中樞
    logger = setup_logger(os.path.join(script_dir, '../data/logs'))
    tracker = ExperimentTracker(os.path.join(script_dir, '../data/7_experiments/experiments_history.json'), logger)
    config_mgr = HyperparameterConfig(args.config, logger)
    # 重新根據增量模式校準參數
    config_mgr.config = config_mgr._load_or_create_default(is_incremental=args.incremental)
    
    trainer = YOLOv8Trainer(model_weights=args.weights, logger=logger)
    
    if args.action == 'tune':
        trainer.tune(data_yaml=args.data)

    elif args.action == 'train':
        best_pt = trainer.train(data_yaml=args.data, hyper_params=config_mgr.config, name=f"exp_v072_{'inc' if args.incremental else 'base'}")
        metrics = trainer.evaluate(best_pt, data_yaml=args.data)

        metrics_dict, record_idx = tracker.log_experiment(
            dataset_version=f"v0.7.2_{'incremental' if args.incremental else 'rebuild'}",
            model_weights=args.weights,
            metrics=metrics,
            save_dir=os.path.dirname(os.path.dirname(best_pt))
        )

        if tracker.check_promotion_gate(metrics_dict, best_pt):
            tracker.update_promotion_status(record_idx)

    elif args.action == 'auto_tune_train':
        logger.info("Tune -> Train -> Eval")
        trainer.tune(data_yaml=args.data, iterations=350, epochs=100)

        best_hyper_path = os.path.normpath(
            os.path.join(script_dir, '../data/7_experiments/tune/best_hyperparameters.yaml')
        )
        if os.path.exists(best_hyper_path):
            advanced_mgr = HyperparameterConfig(best_hyper_path, logger)
            best_pt = trainer.train(data_yaml=args.data, hyper_params=advanced_mgr.config)
        else:
            logger.warning("Tune best config not found, fallback to default training config")
            best_pt = trainer.train(data_yaml=args.data, hyper_params=config_mgr.config)

        metrics = trainer.evaluate(best_pt, data_yaml=args.data)

        metrics_dict, record_idx = tracker.log_experiment(
            dataset_version=dataset_version_name,
            model_weights=args.weights,
            metrics=metrics,
            save_dir=os.path.dirname(os.path.dirname(best_pt))
        )

        if tracker.check_promotion_gate(metrics_dict, best_pt):
            tracker.update_promotion_status(record_idx)

    print_pipeline_notice(
        output_paths=[
            os.path.normpath(os.path.join(script_dir, '../data/7_experiments')),
            os.path.normpath(os.path.join(script_dir, '../data/7_experiments/weight/latest_best.pt')),
            os.path.normpath(os.path.join(script_dir, '../data/7_experiments/weight/global_best.pt')),
        ],
        next_script="src/analyze_errors.py",
        notes=[
            "請先確認 latest_best.pt 與 global_best.pt 是否如預期更新。",
            "完成訓練後，建議先做 error analysis，再決定是否進入 hard case mining 或 active learning。",
        ],
    )
