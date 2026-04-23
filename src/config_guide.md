# 配置修改指南 (Config Guide)

所有執行腳本的路徑與數值，**統一由以下單一檔案控制**：

```
src/anti_gravity/settings.py
```

修改後存檔，所有腳本下次執行時即自動套用，**不需要個別修改腳本**。

---

## 我想改什麼，要找哪一段？

直接對照下表找到 `settings.py` 中對應的區塊：

1_raw、3_processed 等 | 資料夾路徑 | `class PathConfig` (第 9 行) | 全部腳本 |
balance | 平衡的比例、抽樣邏輯 | `class BalanceSettings` (第 51 行) | `balance_dataset.py` |
augment | 增強的倍率、強度 | `class AugmentSettings` (第 60 行) | `augment_dataset.py` |
train | 訓練的輪次、影像大小 | `class TrainSettings` (第 66 行) | `train.py` |
auto_label | 自動標註的模型、信心閾值 | `class AutoLabelSettings` (第 41 行) | `auto_label.py` |

---

## 各設定區塊說明

### `PathConfig`（第 9 行）— 所有腳本都讀此區

修改資料夾路徑時使用。

| 參數 | 對應路徑 | 預設值 |
|---|---|---|
| `raw` | 原始影像 | `data/1_raw` |
| `filtered` | 篩選後影像 | `data/2_filtered` |
| `processed` | 前處理完成 | `data/3_processed` |
| `auto_ann` | 自動標註結果 | `data/5_auto_ann` |
| `augmented` | 增強/切分輸出 | `data/6_augmented` |
| `experiments` | 訓練實驗紀錄 | `data/7_experiments` |
| `pipeline_yaml` | 主配置 YAML | `configs/pipeline.yaml` |

---

### `BalanceSettings`（第 51 行）— `balance_dataset.py`

控制資料平衡的抽樣邏輯。

| 參數 | 說明 | 預設值 |
|---|---|---|
| `target_ratio` | Close : Open 目標比例 | `2.0` |
| `max_bg_ratio` | 背景圖最大占比 | `0.2` |
| `min_valid_area` | 最小有效框面積（相對） | `0.00005` |
| `min_valid_dim` | 最小有效框邊長（相對） | `0.008` |
| `error_bonus` | 錯誤圖加權分數 | `10.0` |
| `edge_penalty` | 邊緣框懲罰分數 | `-0.6` |
| `scene_cap_close_only` | 每場景最多抽幾張 Close-only | `2` |

---

### `AugmentSettings`（第 60 行）— `augment_dataset.py`

控制資料增強的倍率。

| 參數 | 說明 | 預設值 |
|---|---|---|
| `base_multiplier` | 一般樣本增強倍數 | `3` |
| `close_fp_like_multiplier` | Close 易誤判樣本增強倍數 | `2` |
| `hard_negative_multiplier` | 困難負樣本增強倍數 | `4` |
| `open_hard_threshold` | Open 困難樣本判斷閾值 | `0.05` |

---

### `TrainSettings`（第 66 行）— `train.py`

控制模型訓練的超參數。

| 參數 | 說明 | 預設值 |
|---|---|---|
| `weights` | 初始權重檔名 | `yolov8s.pt` |
| `epochs` | 訓練輪次 | `40` |
| `patience` | Early Stop 耐心值 | `10` |
| `imgsz` | 訓練影像尺寸 | `640` |
| `eval_conf` | 評估信心閾值 | `0.25` |

---

### `AutoLabelSettings`（第 41 行）— `auto_label.py`

控制自動標註的模型與篩選條件。

| 參數 | 說明 | 預設值 |
|---|---|---|
| `det_model` | 偵測模型路徑 | `yolov8s.pt` |
| `sam_model` | SAM 分割模型路徑 | `mobile_sam.pt` |
| `conf` | 偵測信心閾值 | `0.5` |
| `iou` | NMS IoU 閾值 | `0.45` |
| `imgsz` | 推論影像尺寸 | `1024` |
| `min_box_area_px` | 最小有效框面積（像素） | `200` |
| `min_box_dim_px` | 最小有效框邊長（像素） | `10` |

---

## 快速修改流程

1. 開啟 `tools/anti_gravity/settings.py`
2. 找到對應的 Class（參見上表的行號）
3. 修改數值後存檔
4. 直接執行腳本即可，無需其他操作
