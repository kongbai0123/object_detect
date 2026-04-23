# 🚀 Project Evolution & Experiment History (工業門偵測版控日誌)

本文件紀錄 MLOps 管線的演進歷史與歷次回合的實驗除錯紀錄，作為專案推進的核心軌跡。

---

## 📅 版本歷史日誌 (Version History)

### [v0.8.0] 全球標準化與四層存儲重構 (MLOps Infrastructure Modernization) - *當前版本*
**核心目標**：從「雜亂的路徑管理」轉向「專業級 MLOps 基礎建設」，實現資料狀態隔離與自動化管線升級。
*   **🛠️ 決策與行動**：
    1.  **四層存儲架構 (Four-Layer Storage)**：正式廢棄 `data/` 下的數字前綴目錄，建立 `sources` (原始層)、`assets` (資產層)、`workspace` (工作層)、`artifacts` (產出層) 四大嚴格隔離層級，從根本上解決資料污染問題。
    2.  **中央配置單一化 (`settings.py`)**：導入 Pydantic-settings，將所有腳本路徑與超參數收網至單一入口，徹底消除 Hardcoded 路徑，確保 Single Source of Truth。
    3.  **環境編碼與接力指引**：全面強制 UTF-8 編碼解決 Windows 平台衝突，並導入 `pipeline_notice` 模組，強化腳本執行間的邏輯指引。
    4.  **CVAT 整合閉環 v2**：重構 `cvat_import.py` 橋接器，支援自動從原始池 (`raw`) 補位影像至黃金集 (`goldenset`)，大幅提升標註回流效率。
*   **📈 營運現況 (2026-04-20)**：
    - **架構遷移完成**：歷史資料已全數遷移至 `storage/` 體系，新舊路徑映射表已存檔於 `reorg_map.json`。
    - **穩定性驗證**：核心管線（Split > Balance > Augment > Train）已在新架構下測試通過。

---

### [v0.7.2] 語義修正與資料重心轉置 (Semantic Correction & Replay Protection)
**核心目標**：修正 Open 類別語義漂移（路牌誤判）並解決新環境下的 Close 漏報問題。
*   **🛠️ 決策與行動**：
    1.  **CLIP 語義採礦**：跳過 YOLO 的視覺盲點，改用 CLIP 從新影片中精準提煉 116 張「人手開門」高品質樣本。
    2.  **化誤報為負樣本**：將 0.7.1 YOLO 誤判的路牌樣本轉化為 Hard Negative（背景圖），強制模型學習「路牌不是門」。
    3.  **定海神針 (Replay Core) 植入**：建立 200 張黃金樣本池，並修改 `split_dataset.py` 確保每次增量訓練都「強制注入」核心記憶，防止災難性遺忘。
    4.  **數據重平衡 (1:2.7)**：對冗餘關門畫格進行 Scene-aware 下採樣，將 Open/Close 比例優化至健康區間。
*   **📈 營運現況 (2026-04-09)**：
    - **數據工程完工**：完成 Phase 1-3 採礦與重組，Open 類別實施 3 倍增強。
    - **背景訓練啟動**：增量特訓 `exp_v072_inc` 執行中，重點觀察路牌拒判能力。

### [v0.7.1] 營運管線化與增量閉環 (Incremental AL Loop)
**核心目標**：從「單次訓練」轉化為「可持續迭代的生產管線」
*   **🛠️ 決策與行動**：
    1.  **類別感知增強 (Conditioned Aug)**：針對 Open (補強)、Close (保守)、BG (拒判) 實施 3 組不同 Profile。
    2.  **增量微調模式 (--incremental)**：鎖定 `lr0=0.001` 與 30 輪短衝刺。
    3.  **安全閘門制度 (Ghost Promotion Gate)**：整合 `eval_ghosts.py` 至晉升邏輯。
    4.  **Replay Core 策略**：建立初步隔離機制，確保增量訓練包含歷史資料。
*   **📈 歷史總結**：
    - **首戰告捷**：成功執行首次 0.7.1 增量閉環，Recall 達到 0.953。
    - **發現漏洞**：在對抗性測試中發現對「路牌」有語義偏移，觸發 v0.7.2 的修正計畫。

### [v0.6.1] 獵鬼行動與類別再平衡 (修正場景洩漏並重啟)

**硬體目標**：NVIDIA Jetson Orin Nano (TensorRT) / Coral TPU (TFLite INT8)
* **🩺 核心病灶**：發現 v0.5 最佳 F1 門檻極低 (0.085)，且 `close` 樣本量為 `open` 的 4.6 倍，導致嚴重「背景特徵吸收」。模型遇到未知背景便會安全地猜測為 `close`。
* **🛠️ 決策與行動**：
  1. **分層困難挖掘**：實作 `mine_hard_negatives.py`，撈出 200 張高自信度「假關門 (fp_close_high)」影像。
  2. **類別補強平衡**：早期使用 `mine_open_samples.py`，目前已由 `mine_open_v2.py` 取代，從原始影片再抽新開門特徵以補強分布。
  3. **資料庫去毒回流**：將「無車門的純鬼影像 (ghost_)」與「補強開門 (boost_open_)」無損匯入 `3_processed` 黃金池。
  4. **模型訓練特調**：提升輸入解析度 `imgsz=768`，小幅調升損失權重 `cls=0.7`，並設置 `close_mosaic=10` 保留真實背景結構。
* **📈 當前狀態**：YOLOv8s 第 15 次實驗 (exp15) 訓練中。

### [v0.5] 邊緣佈署準備與 YOLOv8s 升級
* **🛠️ 決策與行動**：將主架構由 YOLOv8n 升級為解析力更強的 YOLOv8s。確立量化佈署計畫，規劃導出 TensorRT FP16 與 TFLite INT8，專攻邊緣推論。

### [v0.4] 資料庫災難與淨化復原 (Data Contamination Recovery)
* **🩺 核心病灶**：v0.3 發生致命「同名檔覆蓋」污染（自動標註蓋過了高品質手標黃金資料），導致開門召回率雪崩至 29%。
* **🛠️ 決策與行動**：
  1. 全面清空並重建 `3_processed`，從備份 zip 搶救出 126 張原始人工金標。
  2. 導入 `miner_` 前綴隔離機制，化解後續檔案碰撞風險。重組出 271 張純淨樣本。
  3. 重啟訓練 (exp13)，**開門 Recall 成功暴升至 83%**，模型智力回歸標準值。

### [v0.3] CLIP 語義採礦 (Semantic Mining)
* **🛠️ 決策與行動**：建置 `clip_filter.py` 以文生圖模型自動過濾含有門與無門的場景，意圖依靠資料加量解決誤判問題（但隨後引爆 v0.4 檔名衝突災害）。

### [v0.1 ~ v0.2] MLOps 單階段管線重塑 (Base Pipeline Migration)
* **🛠️ 決策與行動**：
  1. 放棄專案初期的「YOLO 偵測 + CNN 分類」雙引擎過時架構，重新整併為端到端單階段 YOLOv8 推論。
  2. 建立初步資料流金字塔（後於 v0.8.0 升級為 storage 四層架構）與 `cvat_import.py` 整合工具庫。
