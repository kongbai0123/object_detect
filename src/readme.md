# Industrial Dooring Early Warning Pipeline

這套 MLOps 系統旨在建立一個專為嵌入式邊緣設備（如 Jetson）設計的「開門防撞預警」模型流程。從資料蒐集、清洗、增強到主動學習，具備高度自動化與閉環能力。

---

## 🛑 Step 0: 專案規格定義閘門 (Specification Gate)

在開始任何 MLOps 流程前，**必須手動設定**以下規格，這將影響後續所有資料夾名稱與網路定義：

### 1. 鎖死類別定義：嚴格落實「兩分類」 (Strict 2-Class)
YOLO 是空間特徵模型，**不要讓模型嘗試學習 `door_opening` (開啟中) 這種具有時間維度與無窮中間態的類別**。
所有資料標註與程式碼皆強制收斂為以下 2 類：
* `0: open` (車門開啟 / 微開 / 展開)
* `1: close` (車門關閉)

> **邊緣端如何預警？** 部署端的預警引擎（如 `branchs.py`）會使用 Temporal Voting (時間序列防抖)。當它觀測到目標從 `close` 轉變為 `open` 的**瞬間切換**時，就會觸發 `EARLY_WARNING_OPENING`。這是兼顧高召回率與低延遲的最佳實踐。

### 2. 負樣本規範 (Hard Negatives Definition)
本專案的難點不在於「認出車門」，而在於「在複雜背景中抵禦假陽性 (FP)」。以下情況必須大量納入 `close` 類別的標註：
* 靜止車輛且旁邊有行人經過，但門緊閉。
* 路面反光、水灘、樹影、光柵等容易引起邊緣擾動的特徵。

### 3. 手動維護 `data/dataset.yaml`
此檔案為全系統的靈魂中樞。它不會由腳本自動合成，請在專案初期**手動建立與確認**，後續 `train.py` 會絕對信任此檔案：
```yaml
path: C:/antigravity/data
train:
  - 6_augmented/images
val:
  - 4_external/val_frozen/images

nc: 2
names:
  0: open
  1: close
```

---

## 🗺️ 流程全貌 (Visual Pipeline)

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              MLOps Pipeline  ·  Industrial Dooring Early Warning                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  ⚙️  OPTIONAL：資料挖掘 (任何一條路皆可)                                     │
  │                                                                             │
  │  [影片檔 .mp4]         [開源資料集]                                          │
  │       |                     |                                               │
  │       v                     v                                               │
  │  video2frames.py       tools/scrape_pseudo.py  ────────────────────┐        │
  │  (抽幀存至1_raw/)       (爬取存至1_raw/)                            |        │
  │       |                                                            |        │
  │       v                                                            |        │
  │  clip_filter.py                                                    |        │
  │  (語意粗篩)                                                         |        │
  │  2_filtered/                                                       |        │
  │       |                                                            |        │
  │       v                                                            |        │
  │  auto_label.py  ····(初稿，僅供參考)····>                           |        │
  │  (YOLO+SAM 初稿)                                                   |        │
  │  5_auto_ann/                                                       |        │
  └───────────────────────────────|────────────────────────────────────|────────┘
                                  |  (必須人工審核，絕對不可自動回流)   |
                                  v                                    v
  ╔═══════════════════════════════════════════════════════════════════════════════╗
  ║  🧑‍💼  必經閘門：CVAT 人工審查  (cvat_import.py → data/3_processed/)             ║
  ╚═══════════════════════════════════════════════════════════════════════════════╝
                                        |
                                        v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  ✅  MANDATORY：核心訓練流程                                                 │
  │                                                                             │
  │  data/3_processed/          (Ground Truth 黃金池)                           │
  │         |                                                                   │
  │         v                                                                   │
  │  split_dataset.py  ─────────────────────────────────────────┐               │
  │  (train:val = 80:20)                                         |              │
  │         |                                          ⚠️ val_frozen            │
  │         v                                          不可增強、不可污染         │
  │  4_external/train_src/                             4_external/val_frozen/   │
  │         |                                                    |              │
  │         v                                                    |              │
  │  augment_dataset.py                                          |              │
  │  (4x 天氣/光照增強)                                           |              │
  │  6_augmented/                                                |              │
  │         |                                                    |              │
  │         v                                                    |              │
  │      train.py  ──────────────────────────────────────────── + (eval)        │
  │  (YOLOv8 訓練 + 晉升閘門)                                                    │
  │         |                                                                   │
  │         v                                                                   │
  │  7_experiments/weight/                                                      │
  │  ├── global_best.pt   ← 通過晉升閘門的王者                                   │
  │  └── latest_best.pt   ← 本次最新出爐                                         │
  └───────────────────────────┬─────────────────────────────────────────────────┘
                              |
                 ┌────────────┴────────────┐
                 v                         v
  ┌────────────────────────┐  ┌───────────────────────────────────────────────┐
  │  🚗 邊緣部署            │  │  🔄 分析閉環 (Feedback Loop)                  │
  │                        │  │                                               │
  │  branchs.py            │  │  analyze_errors.py     → FiftyOne UI          │
  │  - 時序防抖 Voting      │  │  hard_case_miner.py    → 8_hard_cases/        │
  │  - Priority Scan       │  │               high_conf_review/ (疑 FP)       │
  │  - CLAHE 光照補償       │  │               low_conf_uncertain/ (疑 FN)     │
  │                        │  │  active_learning.py    → 8_hard_cases/        │
  │  close ──→ open        │  │               low_conf_uncertain/             │
  │  = EARLY WARNING 🚨    │  │                        |                      │
  └────────────────────────┘  │               (人工補標後) ················>╗  │
                              │                                            ║   │
                              └─────────────────────────────────────────── ║ ──┘
                                                                           ║
                                                                           ▼
                                                               回流 CVAT 人工閘門
```


---

## 📋 MLOps Level-5 核心流程 (Core Pipeline)

流程分為「必備 (Mandatory)」與「可選 (Optional)」。要練出一個模型，你一定要走完所有的必備流程。

### 【必備階段】核心訓練與品質把關

#### 1. 人工審核閘門 (The Human Gate)
所有資料（不論來源）最終**必須**進入 CVAT 等標註工具人工審核。
* 操作：完成審核後，匯出 YOLO 格式，使用 **`python src/cvat_import.py`** 將其閉環回流。
* 輸出：`data/3_processed/` (這是唯一乾淨的 Ground Truth 黃金池)

#### 2. 資料集切割 (Dataset Splitting)
* 操作：**`python src/split_dataset.py`**
* 輸出：`data/4_external/train_src/` 與 `data/4_external/val_frozen/`
> [!WARNING]  
> **☢️ 資料洩漏極大風險 (Data Leakage)**  
> 預設的 `split_dataset.py` 採取隨機切割 (Random Split)。如果你的資料有大量是由影片抽出的相鄰幀，隨機切割會導致「相同場景」同時存在於 train 和 val，造成驗證指標(mAP)虛高，但實戰徹底盲目。強烈建議未來將隨機切割改寫為「依據影片/場景切割 (Scene-Aware Split)」。

#### 3. 破壞性物理增強 (Offline Augmentation)
* 操作：**`python src/augment_dataset.py`** (預設擴增 4 倍)
* 輸出：`data/6_augmented/`
> [!TIP]
> 增強**絕對只針對 train_src 進行**，val_frozen 是永不污染的跨代 Benchmark。

#### 4. 晉升閘門與訓練 (Training)
* 操作：**`python src/train.py --action train`**
* 輸出：`data/7_experiments/` 
  - `weight/latest_best.pt` (最新一輪最佳產出)
  - `weight/global_best.pt` (通過晉升驗證的黃金權重)
* **[核心機制] 大腦發現系統 (Weights Discovery)**：
  本管線所有推論腳本 (`auto_label`, `video_pt_test`, `active_learning`) 已對齊。它們會自動按以下順序載入權重：
  1. `global_best.pt` (最穩定的王者)
  2. `latest_best.pt` (最新的挑戰者)
  3. `exp*/weights/best.pt` (實驗中的各版本，依時間排序)
  4. `yolov8n.pt` (基礎預訓練)

---

### 【可選階段】資料挖掘與主動學習 (Data Mining)

若模型遇到瓶頸，請利用以下腳本進行提煉，並接回「1. 人工審核閘門」。

#### 📡 生肉收集與清洗
* **`video2frames.py`**：將影片抽幀輸出至 `1_raw/door_opening_frames`。
* **`tools/scrape_pseudo.py`**：從開源影片池大批次挖掘 Hard Negatives。

#### 🧠 自動初稿 (Auto-Labeling)
> [!CAUTION]
> 嚴防自動標註毒化 (Data Poisoning)。當模型 `close` 抓不準時，盲目自動標註會將真實存在的關門車當成背景，讓下一次訓練徹底失明。必須人工覆核！
* **`clip_filter.py`**：使用 NLP 提示詞 (`open`/`close` text prompts) 篩選出明確含有相關事件的圖片，存入 `2_filtered/`。
* **`auto_label.py`**：動用最好的 `global_best.pt` 結合 SAM 生成像素級邊緣吸附的 bbox 初稿，存入 `5_auto_ann/`。

#### 🩻 錯誤分析 (Error Analysis)
* **`analyze_errors.py`**：掛載 `4_external/val_frozen`，啟動 FiftyOne 網頁 (`http://localhost:5151`) 視覺化找出漏標(FN)或誤判(FP)的趨勢。

#### 🥊 困難樣本挖掘 (Active Learning)
* **`active_learning.py`**：掃描 `1_raw/door_opening_frames`（**只收靜態圖片**），依據多類別熵值與不確定性，挑選最值得補標的圖片至 `8_hard_cases/low_conf_uncertain/`。
  - **[核心機制] 智能過濾 (Labeled Filter)**：腳本會自動掃描 `data/3_processed`，若樣本已在已標定池中，將自動剔除，確保不重複推薦已標好的圖，極大化人工標記價值。
* **`hard_case_miner.py`**：讀取影片或圖片，按高/低信心度分流疑似 FP / FN，存入 `8_hard_cases` 供人工覆核。

---

## ✅ 訓練前查檢表 (Pre-Training QA Checklist)
請在按下 `train.py` 之前，在心中勾選以下防呆項目：
- [ ] `dataset.yaml` 內的 `names` 是否確實為 `0: open`, `1: close` 且沒有舊類別？
- [ ] `3_processed/` 是否已無「缺乏標籤檔的幽靈圖片」或「漏框的車輛」？
- [ ] 若資料來自連拍畫面，是否已經人工確保同一台連拍車不會同時出現在 train 和 val？
- [ ] 負面樣本 (`close` 或純背景) 的數量是否有壓倒性優勢，足以抵抗假陽性？
- [ ] 上一版模型遺留下來的 `low_conf_uncertain` 困難樣本，是否已經人工修正完畢並倒回 `3_processed/` 內？ 

---

## 📁 系統目錄結構與腳本對照

```text
data/
├── 1_raw/                           ← video2frames.py (抽幀), tools/scrape_pseudo.py (挖掘)
│   ├── videos/                      
│   └── door_opening_frames/         
├── 2_filtered/                      ← clip_filter.py (使用 open/close prompt 篩選)
│   ├── open/                        
│   └── close/                       
├── 3_processed/                     ← cvat_import.py [必經 GT 黃金池]
│   ├── images/
│   └── labels/
├── 4_external/                      ← tools/download_openimages, tools/parse_bdd10k 
│   ├── openimages/                  
│   ├── bdd10k/                      
│   ├── train_src/                   ← split_dataset.py [Train 80%]
│   └── val_frozen/                  ← split_dataset.py [Val 20% 永不污染]
├── 5_auto_ann/                      ← auto_label.py (SAM 初稿標籤)
├── 6_augmented/                     ← augment_dataset.py [增強訓練集]
├── 7_experiments/                   ← train.py [訓練輸出]
│   └── weight/                      
│       ├── global_best.pt           
│       └── latest_best.pt           
├── 8_hard_cases/                    
│   ├── low_conf_uncertain/          ← active_learning.py, hard_case_miner.py [FN]
│   └── high_conf_review/            ← hard_case_miner.py [FP]
└── dataset.yaml                     ← [架構師手動維護]
```
