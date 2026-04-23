# Industrial Door Detection & Decision System (Antigravity)

![Status](https://img.shields.io/badge/Status-Production--Ready-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/Engine-YOLOv8-red)

這是一個專為工業邊緣端 (Edge) 設計的高可靠度門狀態偵測系統。不同於一般的物件偵測，本專案整合了 **時序決策引擎 (Industrial Decision Engine)**，能有效解決光影干擾、閃爍、與邊界模糊判斷問題。

---

## 🚀 核心優勢 (Key Features)

### 1. 工業級決策引擎 (Industrial Decision Engine)
*   **時序平滑 (Temporal Smoothing)**：採用 10 幀滑動窗口投票，消除單幀跳動。
*   **雙軌權重判定**：Open (紅框) 具備高優先級，Close (綠框) 負責日常穩定監控。
*   **物理幾何約束**：自動過濾面積過小或過大的非理性偵測框。

### 2. 全量吸收訓練鏈路 (Full-Merge Absorption Pipeline)
*   **場景感知切分 (Scene-Aware Split)**：防止視訊序列造成的資料洩漏 (Data Leakage)。
*   **雙向 Open 保護**：動態確保驗證集具備監督能力的同時，守住訓練集的正樣本保底。
*   **領域權重平衡 (Domain Balancing)**：防止單一數據版本 (如 `fif`) 造成模型偏移。

### 3. Edge 端事件優化
*   **低延遲觸發**：專為邊緣設備優化的推論邏輯。
*   **高可信度輸出**：支援 Open-Centric 模式，只在極其確定時觸發警報。

---

## 🛠️ 快速開始 (Quick Start)

### 1. 環境安裝
```powershell
pip install -r requirements.txt
```

### 2. 影片推論測試 (Industrial Evaluation)
執行具備決策引擎的影片推理，結果將存於 `storage/artifacts/evaluations/videos`。
```powershell
python src/video_pt_test.py --source storage/assets/videos --model path/to/best.pt
```

### 3. 啟動增量訓練 (Incremental Training)
```powershell
python src/train_loop.py --start all --mode incremental
```

---

## 📂 專案結構 (Directory Structure)

*   `src/`：核心演算法與執行腳本。
    *   `anti_gravity/`：系統底座、設定管理與數據處理器。
*   `configs/`：所有的訓練與鏈路 YAML 配置。
*   `storage/`：(Git Ignored) 資料存儲區，包含 assets, workspace 與 artifacts。

---

## 📜 決策紀錄 (Architectural Decisions)
詳細的開發歷程與技術決策請參閱 `implementation_plan.md`。
