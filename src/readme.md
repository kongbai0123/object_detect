# 🛠️ Antigravity 核心腳本說明 (Technical Reference)

本目錄包含工業門偵測流水線的所有核心執行腳本。

---

## 🚀 核心執行鏈路 (Workflow)

### 1. 訓練流程編排 (`train_loop.py`)
這是系統的 **「自動化大腦」**，負責串接資料準備與模型訓練。
*   **功能**：自動執行 `Clean -> Split -> Balance -> Augment -> Train` 的全過程。
*   **指令**：`python src/train_loop.py --start all --mode incremental`

### 2. 資料鏈路 (Data Pipeline)
*   **`split_dataset.py`**：實施 **Industrial-Grade Split-then-Merge**。支援跨版本的場景感知切分，具備雙向 Open 樣本保護機制。
*   **`balance_dataset.py`**：對訓練集進行類別平衡，防止 `close` (cls:1) 過度佔據 Batch 權重。
*   **`augment_dataset.py`**：執行離線物理增強，針對 `open` 類別進行加權採樣。

### 3. 推理與決策 (`video_pt_test.py`)
本專案的 **「決策中心」**，將 YOLO 偵測轉化為穩定的工業狀態。
*   **工業決策引擎 (Industrial Decision Engine)**：
    *   **時序平滑**：10 幀投票機制，消除狀態閃爍。
    *   **專屬閾值**：Open (0.35) / Close (0.60) 差別化判定。
    *   **幾何過濾**：排除非理性大小的偵測框。

---

## 📁 腳本一覽表 (Script Reference)

| 腳本名稱 | 角色 | 核心功能 |
| :--- | :--- | :--- |
| `train_loop.py` | 編排器 | 自動化迭代訓練流水線 |
| `train.py` | 執行器 | 封裝 YOLO 訓練與成績回報邏輯 |
| `video_pt_test.py`| 決策器 | 具備時序平滑的工業級推理腳本 |
| `split_dataset.py` | 分配器 | 支援雙向 Safeguard 的場景切分 |
| `balance_dataset.py`| 平衡器 | 調整訓練集 O:C 比例 |
| `augment_dataset.py`| 增強器 | 離線 4 倍物理增強 |
| `mine_dataset.py` | 挖掘器 | 自動偵測未標註影片中的目標並抽幀 |

---

## ⚠️ 開發規範
1.  **路徑引用**：請統一透過 `anti_gravity.settings` 獲取路徑，禁止在腳本中寫死 `storage/` 以外的路徑。
2.  **色彩空間**：`cv2` 使用 **BGR**。繪圖時請注意：Red 為 `(0, 0, 255)`，Green 為 `(0, 255, 0)`。
3.  **編碼建議**：在 Windows 環境下，請確保 `print` 語句不包含非 ASCII 字元，以避免編碼報錯。
