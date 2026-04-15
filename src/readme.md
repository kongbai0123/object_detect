# 🚪 工業門預警 MLOps 流水線 (Latest v2.0)

這套系統專為工業場景設計，旨在透過自動化流程訓練出高精準度的開門偵測模型。

---

## 🚀 核心工作流 (Workflow)

### 【第一階段】資料採集與標註 (Data Acquisition & Labeling)
1.  **影片抽幀**：執行 `python src/video2frames.py` 將原始影片轉為圖片，存入 `data/1_raw/`。
2.  **負樣本挖掘**：執行 `python src/select_negatives.py` 從影片中自動挑選「容易誤判的背景圖」放入 `data/3_processed/`。
3.  **自動標註 (AI 初稿)**：
    *   將生肉圖放入 `data/2_filtered/open`。
    *   執行 `python src/auto_label.py`。
    *   **下一步**：將生成的 ZIP 匯入 CVAT 進行人工複核，完成後存入 `data/3_processed/`。

### 【第二階段】訓練預處理 (Training Pipeline)
當 `data/3_processed/` 累積足夠樣本後，依序執行：
1.  **資料切分 (Split)**：`python src/split_dataset.py`
    *   將熟肉切分為 Train (80%) 與 Val (20%)。
2.  **類別平衡 (Balance)**：`python src/balance_dataset.py`
    *   對訓練集中的「純關門背景」進行降採樣，提升 `open` 類別權重。
3.  **物理增強 (Augment)**：`python src/augment_dataset.py`
    *   對平衡後的資料進行光照、天氣、旋轉等 4 倍擴增。

### 【第三階段】啟動訓練與評估 (Training & Eval)
1.  **開始訓練**：`python src/train.py --action train`
    *   **新功能**：訓練結束後會自動更新 `data/7_experiments/training_history.md`。
    *   **晉升機制**：若表現優異，模型會自動晉升為 `global_best.pt`。

### 【第四階段】實地推論 (Live Inference)
1.  **即時偵測**：`python object_detect/detect.py --source 0`
    *   使用簡約版推論引擎，直接查看模型 raw 輸出。

---

## 📁 目錄結構說明 (Directory Structure)

```text
data/
├── 1_raw/           # 原始影片與抽幀圖片
├── 2_filtered/      # 待自動標註的生肉區
├── 3_processed/     # [關鍵] 已標註完成的「熟肉」黃金池
├── 5_auto_ann/      # Auto Label 產出的 AI 標註初稿
├── 6_augmented/     # 流水線中轉區 (含 Train/Val 最終增強資料)
├── 7_experiments/   # 訓練實驗結果與日誌
└── dataset.yaml     # 訓練設定檔 (nc: 2)
```

## 🛠️ 腳本說明 (Script Reference)

| 腳本名稱 | 功能描述 | 輸出位置 |
| :--- | :--- | :--- |
| `video2frames.py` | 影片轉圖片 | `data/1_raw/` |
| `select_negatives.py`| 挖掘背景負樣本 | `data/3_processed/` |
| `auto_label.py` | YOLO+SAM 自動標註 | `data/5_auto_ann/` |
| `split_dataset.py` | 80/20 場景感知切分 | `data/6_augmented/train_src` |
| `balance_dataset.py` | 關門類別降採樣 | `data/6_augmented/train_src_balanced` |
| `augment_dataset.py` | 離線增強 | `data/6_augmented/train` |
| `train.py` | 核心訓練引擎 | `data/7_experiments/` |

---

## ⚠️ 標註規範與原則
*   **類別鎖死**：`0: open` (開啟/微開), `1: close` (完全關閉)。
*   **拒絕毒化**：Auto Label 的結果**絕對不能**直接拿去跑 `balance_dataset`，必須先經過 CVAT 人工修正。
*   **解析度規範**：訓練與推論請鎖定 `imgsz: 832` 以確保遠方門縫的識別能力。
