# Car Door Open-Centric Edge Detection System (Antigravity)

![Status](https://img.shields.io/badge/Status-Engineering--Prototype-orange)
![Focus](https://img.shields.io/badge/Focus-Open--Centric-red)
![Deployment](https://img.shields.io/badge/Deployment-Edge--AI-blue)

本專案是一個專為 **車門開關狀態 (Car Door State)** 設計的高可靠度邊緣端偵測系統。我們不只提供物件偵測模型，更核心的是解決了從「模型輸出」到「工業決策」之間的穩定性問題。

---

## 🎯 專案定位 (System Positioning)

*   **核心目標**：穩定、低延遲地捕捉 **開門 (OPEN) 事件**。
*   **適用場景**：車載監控、自動化洗車場、停車場安全管理等 Edge AI 設備。
*   **技術哲學**：
    *   **Open-Centric**：將 `OPEN` 視為高優先級觸發事件，`CLOSE` 則作為穩定狀態的參照基準。
    *   **Decision > Detection**：利用時序決策引擎 (Decision Engine) 消除單幀偵測的抖動與誤報。

---

## 🚀 三大核心工作流 (Categorized Pipelines)

### A. 訓練鏈路 (Training Pipeline) - "Full-Merge Absorption"
專為解決工業場景中資料域偏移 (Domain Shift) 與資料洩漏設計。
*   **自動化編排**：`python src/train_loop.py --mode incremental`
*   **場景感知切分 (Scene-Aware Split)**：防止視訊序列造成的訓練/驗證污染。
*   **雙向 Safeguard**：平衡 Open 樣本的監督能力與訓練訊號保底。

### B. 評估與驗證 (Evaluation Pipeline) - "Evidence-Based"
不只看 mAP，更看重實際業務指標。
*   **硬指標報表**：詳細性能數據請參閱 [benchmark.md](./benchmark.md)。
*   **錯誤型態分析**：自動挖掘 False Negatives (漏檢) 以進行難例強化。

### C. 部署推理 (Edge Inference Pipeline) - "Industrial Decision Engine"
針對低功耗設備優化的穩定輸出邏輯。
*   **時序平滑 (Temporal Smoothing)**：10 幀滑動窗口投票，大幅降低誤觸發。
*   **幾何物理過濾**：排除非理性尺寸的噪訊框。
*   **執行指令**：`python src/video_pt_test.py --source storage/assets/videos`

---

## 📊 性能指標 (Benchmark Summary)

| Metric | Value (v0.7.2) | Note |
| :--- | :--- | :--- |
| **mAP50** | **0.873** | Cross-Version Validation |
| **Open Precision** | **0.91** | Conf > 0.35 |
| **Open Recall** | **0.84** | High Reliability |
| **Edge FPS** | **~30** | Tested on laptop GPU |

---

## 📜 決策紀錄 (Architectural Decisions)
詳細的開發歷程與技術決策請參閱 [implementation_plan.md](./implementation_plan.md)。
