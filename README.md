# 🚦 India Road Accident Analysis & 2026 Blackspot Prediction
### Dataset: Kaggle Indian Roads Dataset (2022 – Apr 2025) | 20,000 Records

> **VIT Vellore — Probability & Statistics Project, 2025–26**
> Dataset: `indian_roads_dataset.csv` | Target: Predict accident severity & 2026 blackspot states

---

## 📋 Table of Contents
- [About](#about)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Notebook Cells](#notebook-cells)
- [ML Pipeline](#ml-pipeline)
- [Results](#results)
- [2026 Prediction](#2026-prediction)
- [Output Figures](#output-figures)

---

## 📌 About

End-to-end road accident severity classification and 2026 blackspot forecasting using **only** the Kaggle Indian Roads Dataset (2022–2025). No MoRTH CSVs used.

**Problem:** Classify each accident as `fatal` / `major` / `minor` and forecast which states will be highest-risk blackspots in 2026.

**Key challenges solved:**
- Severe class imbalance → fixed with **BorderlineSMOTE**
- `major` class hard to predict → fixed with **threshold tuning**
- No 2026 data → validated trend model on 2024 (known) before forecasting

---

## 📂 Dataset

| File | Records | Period | Link |
|------|---------|--------|------|
| `indian_roads_dataset.csv` | 20,000 accidents | 2022–Apr 2025 | [Kaggle Indian Roads Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/indian-roads-accident-dataset) |

**Features used (27 total):**
- Scene: `casualties`, `risk_score`, `vehicles_involved`
- Time: `hour` (cyclical sin/cos), `month`, `is_night`, `is_rush`
- Road: `road_type`, `lanes`, `traffic_signal`, `weather`, `visibility`
- Context: `cause`, `traffic_density`, `state`, `festival_flag`
- Engineered: `cas_x_risk`, `risk_sq`, `cas_sq`, `lanes_x_veh`, `high_risk`, `multi_veh`

---

## 🗂️ Project Structure

```
your_project/
│
├── india_road_accidents_2026.ipynb   ← Main notebook (8 cells)
├── indian_roads_dataset.csv          ← Dataset (place here)
├── README.md
│
└── outputs/
    ├── fig1_eda_overview.png
    ├── fig2_eda_deepdive.png
    ├── fig3_smote_balance.png
    ├── fig4_roc_curves.png
    ├── fig5_confusion_matrices.png
    ├── fig6_all_metrics.png
    ├── fig7_feature_importance.png
    └── fig8_2026_prediction.png
```

---

## ▶️ How to Run

1. Place `indian_roads_dataset.csv` in the same folder as the notebook
2. Open `india_road_accidents_2026.ipynb` in Cursor / VS Code
3. `Ctrl+Shift+P` → **Run All Cells**
4. All 8 figures auto-saved to `outputs/`

**Cell 1 auto-installs:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, imbalanced-learn

> ⏱️ **Estimated runtime:** ~8–12 minutes (stacking cv=5 is the slowest step)

---

## 📓 Notebook Cells

| Cell | What it does |
|------|-------------|
| **1** | Setup, imports, GPU detection, helper functions |
| **2** | Load CSV, feature engineering (cyclical, interactions, polynomial), train/test split |
| **3** | EDA — 12 charts across 2 figures (severity, trends, causes, hourly, correlations) |
| **4** | Class imbalance analysis + BorderlineSMOTE balancing |
| **5** | Train 5 models: LR, RF, XGBoost (GPU→CPU), LightGBM, Stacking (cv=5) |
| **6** | ROC curves, confusion matrices, all-metrics bar chart |
| **7** | Feature importance: RF + XGBoost + LightGBM side-by-side |
| **8** | 2026 blackspot prediction + trend validation + dashboard |

---

## 🤖 ML Pipeline

### Split Strategy
```
Train  →  2022–2024  (18,266 records)
Test   →  2025       (1,734 records — real unseen data)
SMOTE  →  Training only (no test leakage)
```

### Class Balancing
```
Before BorderlineSMOTE:   fatal=2,725   major=5,499   minor=10,042
After  BorderlineSMOTE:   fatal=10,042  major=10,042  minor=10,042
```

### Models

| # | Model | Device | Imbalance |
|---|-------|--------|-----------|
| 1 | Logistic Regression | CPU | class_weight='balanced' |
| 2 | Random Forest | CPU (n_jobs=-1) | class_weight='balanced_subsample' |
| 3 | XGBoost | **GPU → CPU fallback** | sample_weight (balanced) |
| 4 | LightGBM | CPU (n_jobs=-1) | class_weight='balanced' |
| 5 | **Stacking (LR+RF+XGB+LGB → LR)** | CPU | cv=5 |

### Threshold Tuning
Default threshold for multiclass is 0.5 per class. We lower the `major` threshold to **0.45** to improve recall for this hard-to-predict middle class.

---

## 📊 Results

### Model Comparison (Tested on 2025 Real Data)

| Rank | Model | Accuracy | F1 Macro | F1 Fatal | F1 Major | F1 Minor | AUC-OVR |
|------|-------|----------|----------|----------|----------|----------|---------|
| 🥇 1 | Random Forest | 0.433 | 0.507 | **1.000** | **0.489** | 0.503 | **0.764** |
| 🥈 2 | XGBoost | 0.501 | 0.605 | 1.000 | 0.449 | 0.698 | 0.760 |
| 🥉 3 | Stacking Classifier | 0.475 | 0.578 | 1.000 | 0.465 | 0.577 | 0.758 |
| 4 | LightGBM | 0.518 | **0.622** | 1.000 | 0.432 | **0.761** | 0.753 |
| 5 | Logistic Regression | 0.415 | 0.462 | 0.969 | 0.449 | 0.405 | 0.738 |

> **Best AUC:** Random Forest (0.764)
> **Best F1 Macro:** LightGBM (0.622)
> **Best F1 Major:** Random Forest (0.489) — highest recall for the hard middle class

### Key Observations
- `fatal` class: All models achieve near-perfect F1 — `risk_score` and `casualties` are strong signals
- `major` class: Genuinely difficult — overlaps heavily with both fatal and minor
- Threshold tuning (0.45 for major) boosts major recall significantly over default 0.5

---

## 📅 2026 Prediction

### Method
- Per-state linear regression on 2022–2024 accident/fatality/risk trends
- Blackspot Score: **BS = 0.5×D̂ + 0.3×FRd + 0.2×Â**
- Validated by training on 2022–2023 and testing against actual 2024 values

### 2026 Projected Blackspot Rankings

| Rank | State | Pred. Accidents | Pred. Fatalities | BS Score | Risk Level | Trend Accuracy |
|------|-------|----------------|-----------------|---------|-----------|----------------|
| 🔴 1 | **Maharashtra** | 496 | 77 | **0.861** | High | 98.6% |
| 🟠 2 | Delhi | 218 | 41 | 0.462 | Moderate | 98.5% |
| 🟠 3 | Punjab | 191 | 37 | 0.436 | Moderate | 90.3% |
| 🟡 4 | Telangana | 197 | 32 | 0.276 | Mod-Low | 85.7% |
| 🟢 5 | Tamil Nadu | 206 | 30 | 0.214 | Low | 97.3% |
| 🟢 6 | Karnataka | 196 | 29 | 0.207 | Low | 96.4% |
| 🟢 7 | West Bengal | 199 | 22 | 0.005 | Low | 97.5% |

**Average Trend Accuracy: 94.9%** (validated against known 2024 data)

> ⚠️ 2026 figures are **forecasts** — actual data will only be available in mid-2027.

---

## 🖼️ Output Figures

| File | Description |
|------|-------------|
| `fig1_eda_overview.png` | Severity distribution, year-wise trends, hourly patterns, state heatmap |
| `fig2_eda_deepdive.png` | Road type, weather, casualties boxplot, monthly, correlations |
| `fig3_smote_balance.png` | Before/after BorderlineSMOTE class distribution |
| `fig4_roc_curves.png` | ROC curves + AUC bar chart for all 5 models |
| `fig5_confusion_matrices.png` | Normalised confusion matrices (all 5 models, 2025 test data) |
| `fig6_all_metrics.png` | Grouped bar: Accuracy, F1 per class, AUC across all models |
| `fig7_feature_importance.png` | RF, XGBoost, LightGBM feature importances side-by-side |
| `fig8_2026_prediction.png` | Trend forecast, blackspot rankings, fatality comparison, accuracy |

---
*Probability and Statistics — VIT Vellore, 2025–26*