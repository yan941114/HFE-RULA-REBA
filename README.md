# ErgoVision AI: 基於電腦視覺之 RULA / REBA 人體工學自動風險評估系統

> **課程專題**：人因工程與實驗設計 (2026 Fall)  
> **研究題目 01**：應用電腦視覺於人體工學風險自動評估 (OpenCV + RULA／REBA)

---

## 專案簡介 (Overview)

本專案旨在結合 **YOLO11-Pose 人體骨架關鍵點偵測** 與國際人體工學風險量表（**RULA** 與 **REBA**），建立一套端到端（End-to-End）的自動化評估系統。系統支援使用者上傳單張照片或工作影片，自動提取人體關節角度、判定正負屈伸方向、進行查表運算，並輸出風險等級（Action Level）與改善建議。

### 核心設計理念：查表邏輯與視覺模型解耦 (Decoupling)
為了避免「角度算錯」與「查表邏輯寫錯」混在一起除錯，本系統將人因評估核心與視覺辨識模組嚴格分離：
1. **`rula_reba_engine.py`**：純 Python 數學邏輯與矩陣查表庫，不依賴任何視覺模型，已完成 100% 單元測試驗證。
2. **雙參數輸入介面**：明確劃分 `auto_angles`（CV 自動估計角度）與 `manual_modifiers`（作業負荷、肌肉活動、支撐、抓握品質等人工補正參數）。
3. **短路快取機制 (State Cache Bypass)**：前端使用者微調作業負荷（如負重由 0kg 改為 5kg）時，系統直接調用查表引擎（幾毫秒內完成），無須重新跑 YOLO 影片推論。

---

## 專案結構 (Directory Structure)

```text
├── src/
│   ├── __init__.py
│   ├── rula_reba_engine.py      # [階段一] RULA & REBA 查表與評分核心邏輯
│   ├── pose_analyzer.py         # [階段二] YOLO11 姿態提取與關節角度推估
│   └── app.py                   # [階段四] 互動式 Web APP 介面
├── tests/
│   ├── __init__.py
│   └── test_rula_reba.py        # 核心評分邏輯單元測試 (對照課堂標準範例)
├── HFcode/
│   ├── yolo11n-pose.pt          # YOLO11-pose 輕量模型權重
│   └── Desktop motion capture example.py # 基礎骨架擷取範例
├── 課堂講義/                     # 課堂指引、RULA/REBA 檢核表、期末報告規範
├── batch_evaluator.py           # [階段三] 專題批次數據導出腳本 (驗證 Kappa / 一致性)
├── .gitignore
└── README.md
```

---

## 開發四階段進度 (Roadmap)

- [x] **階段一：建立 RULA / REBA 演算法查表引擎 (`rula_reba_engine.py`)**
  - [x] 完整實作 RULA Table A, B, C 與 REBA Table A, B, C 矩陣
  - [x] 矢狀面正負方向性判定 (前屈 Flexion vs 後仰 Extension)
  - [x] 單元測試集通過 (包含課堂 P.47 高危害不良姿勢、P.64 搬運案例)
- [ ] **階段二：關節角度提取模組 (`pose_analyzer.py`)**
  - [ ] YOLO11 17 關鍵點推論與解剖學向量夾角計算
  - [ ] 左右側評估與遮蔽側過濾 (Confidence Score 門檻)
  - [ ] 影片時間序列平滑化 (EMA 濾波) 與最高風險關鍵影格 (Peak Risk Frame) 偵測
- [ ] **階段三：專題批次驗證腳本 (`batch_evaluator.py`)**
  - [ ] 資料夾批次處理，匯出包含預測角度、子分數、總分之 CSV
  - [ ] 對接期末專題研究：計算自動評分 vs 人工專家一致性 (Cohen's Kappa / ICC)
- [ ] **階段四：可互動 Web APP 介面**
  - [ ] 支援照片/影片上傳與即時看板
  - [ ] 整合快取機制與人機微調控制 (Human-in-the-loop)

---

## 快速開始 (Quick Start)

### 1. 啟動 Conda 虛擬環境
本專案建議使用已配置好的 `pose` 環境：
```powershell
conda activate pose
```

### 2. 執行單元測試
驗證 RULA 與 REBA 查表模組是否運作正常：
```powershell
python tests/test_rula_reba.py
```

---

## 參考文獻 (References)
1. **RULA**: McAtamney, L., & Corlett, E. N. (1993). RULA: a survey method for the investigation of work-related upper limb disorders. *Applied Ergonomics*, 24(2), 91-99.
2. **REBA**: Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA). *Applied Ergonomics*, 31(2), 201-215.
