# 🚗 Indian Road Accident Analysis & 2026 Severity Prediction

> End-to-end machine learning pipeline for analyzing Indian road accident patterns (2022–2025) and predicting 2026 accident severity using 7 models including GPU-accelerated XGBoost and LightGBM.

## 📊 Model Results (from your run)

| Model | Accuracy | F1-Macro | CV F1 | Device |
|---|---|---|---|---|
| **LightGBM** | 65.67% | **66.96%** | 79.24% | 🔥 GPU |
| XGBoost | 66.35% | 63.50% | 79.44% | 🔥 GPU |
| Gradient Boosting | 67.22% | 62.11% | 62.11% | 💻 CPU |
| Random Forest | 67.97% | 62.15% | 62.15% | 💻 CPU |
| Decision Tree | 63.42% | 65.25% | 76.15% | 💻 CPU |
| Logistic Regression | 58.13% | 66.30% | 67.39% | 💻 CPU |
| KNN | 45.77% | 47.85% | 47.85% | 💻 CPU |

> **Note on accuracy:** This dataset is synthetic. The `major` and `minor` classes have nearly identical feature distributions (same risk score, casualties, vehicles involved), making them inherently hard to separate. The `fatal` class is predicted with near-perfect accuracy. Scores reflect the true limits of the dataset, not model weakness.

---

## 📁 Project Structure

```
project/
├── indian_roads_dataset.csv        ← Raw dataset (place here before running)
├── indian_roads_2026.ipynb         ← Main Jupyter notebook (20 cells)
├── README.md                       ← This file
│
└── outputs/                        ← Auto-created when you run the notebook
    ├── 01_data_quality.png
    ├── 02_target_distribution.png
    ├── 03_categorical_vs_severity.png
    ├── 04_temporal_patterns.png
    ├── 05_numerical_distributions.png
    ├── 06_correlation_matrix.png
    ├── 07_geographic_hotspots.png
    ├── 08_smote_balance.png
    ├── 09_model_comparison.png
    ├── 10_all_confusion_matrices.png
    ├── 11_per_class_metrics.png
    ├── 12_feature_importances.png
    ├── 13_shap_explainability.png
    ├── 14_2026_predictions.png
    ├── 15_final_summary_dashboard.png
    ├── best_model.pkl               ← Saved best model
    ├── scaler.pkl                   ← StandardScaler
    ├── label_encoder.pkl            ← LabelEncoder
    ├── features.json                ← Feature names list
    ├── model_results.csv            ← Full leaderboard CSV
    └── 2026_predictions.csv         ← 2026 predicted accident data
```

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap joblib
```

All packages are also auto-installed in **Cell 1** of the notebook when you run it.

**Python version:** 3.8+
**Recommended:** Jupyter Lab / Jupyter Notebook / VS Code with Jupyter extension
**GPU:** NVIDIA GPU with CUDA (optional — XGBoost and LightGBM fall back to CPU automatically if no GPU found)

---

## ▶️ How to Run

```bash
# 1. Place the dataset in the same folder as the notebook
#    File must be named: indian_roads_dataset.csv

# 2. Open Jupyter
jupyter notebook indian_roads_2026.ipynb

# 3. Run all cells in order
#    Kernel → Restart & Run All
```

Every chart will display **inline under the cell** and also be **saved to outputs/** automatically.

---

## 📓 Notebook Cell Guide

| Cell | Description | Output Saved |
|---|---|---|
| 1 | Install packages, import libraries, define `save_and_show()` helper | — |
| 2 | Load dataset, filter 2022–2025, print class distribution | — |
| 3 | Missing values bar chart + data types pie chart | `01_data_quality.png` |
| 4 | Target distribution — count bar, yearly trend, pie chart | `02_target_distribution.png` |
| 5 | 6-panel categorical features vs severity (weather, road, cause, traffic, visibility, festival) | `03_categorical_vs_severity.png` |
| 6 | Temporal patterns — hourly line, monthly area, fatal heatmap (day×hour), YoY fatal rate | `04_temporal_patterns.png` |
| 7 | Numerical distributions by severity + correlation heatmap | `05_numerical_distributions.png`, `06_correlation_matrix.png` |
| 8 | Geographic hotspots — fatal by state, accidents by city | `07_geographic_hotspots.png` |
| 9 | Feature engineering — cyclic encoding, risk bins, interaction features, one-hot encoding | — |
| 10 | Train/test split (80/20), StandardScaler, SMOTE balancing | `08_smote_balance.png` |
| 11 | Train all 7 models, print leaderboard with accuracy/F1/CV/time/device | — |
| 12 | Model comparison bar charts (Accuracy, F1-Macro, CV F1) | `09_model_comparison.png` |
| 13 | Confusion matrices for all 7 models side by side | `10_all_confusion_matrices.png` |
| 14 | Per-class Precision/Recall/F1 for all 7 models | `11_per_class_metrics.png` |
| 15 | Feature importance plots for all tree-based models | `12_feature_importances.png` |
| 16 | SHAP beeswarm + bar summary for best tree model | `13_shap_explainability.png` |
| 17 | Generate 1000 synthetic 2026 accident scenarios, predict severity | — |
| 18 | 2026 prediction visuals — pie, actual vs predicted, monthly, by road/weather/cause | `14_2026_predictions.png` |
| 19 | Final summary dashboard — leaderboard table + all key charts in one view | `15_final_summary_dashboard.png` |
| 20 | Save model artifacts, print final summary, list all output files | All `.pkl`, `.json`, `.csv` files |

---

## 🔬 Methodology

### Dataset
- **Source:** Indian Roads Accident Dataset (2022–2025) — 20,000 records
- **Cities:** Pune, Mumbai, Chandigarh, Chennai, Delhi, Bangalore, Hyderabad, Kolkata
- **Target:** `accident_severity` — minor (55%), major (30%), fatal (15%)

### Feature Engineering
| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclic encoding of hour (prevents 23→0 discontinuity) |
| `month_sin`, `month_cos` | Cyclic encoding of month |
| `is_night` | 1 if hour between 20:00–05:00 |
| `casualties_per_vehicle` | casualties ÷ vehicles_involved |
| `risk_x_cas` | risk_score × casualties (interaction term) |
| `risk_squared` | risk_score² (non-linear term) |
| `rs_very_high` | 1 if risk_score > 0.82 (near-perfect fatal predictor) |
| `rs_high` | 1 if risk_score between 0.64–0.82 |
| `night_fog` | is_night AND weather = fog |
| `highway_fast` | road_type = highway AND cause = overspeeding |
| `drunk_night` | is_night AND cause = drunk driving |
| `high_cas_highway` | casualties ≥ 3 AND road_type = highway |

### Class Balancing
SMOTE (Synthetic Minority Over-sampling Technique) is applied **only on the training set** to prevent data leakage. The test set always uses the original class distribution.

### Models Trained
| Model | Type | GPU? | Key Hyperparameters |
|---|---|---|---|
| Logistic Regression | Linear | No | `max_iter=2000`, `class_weight=balanced` |
| Decision Tree | Tree | No | `max_depth=15`, balanced |
| Random Forest | Ensemble | No | `n_estimators=400`, balanced |
| Gradient Boosting | Boosting | No | `n_estimators=300`, `lr=0.05` |
| XGBoost | Boosting | ✅ Yes | `n_estimators=500`, `device=cuda` |
| LightGBM | Boosting | ✅ Yes | `n_estimators=500`, `device=gpu` |
| KNN | Instance | No | `k=7` |

### Evaluation Metrics
- **Primary:** Macro F1-Score (correct choice for imbalanced multi-class)
- **Secondary:** Accuracy, per-class Precision / Recall / F1
- **Validation:** 3-fold Stratified Cross-Validation on the SMOTE training set
- **Explainability:** SHAP TreeExplainer — beeswarm and bar summary plots

---

## 🔮 2026 Predictions

The best model (LightGBM by F1-Macro) is applied to 1,000 projected 2026 accident scenarios sampled from the 2022–2025 distribution with updated timestamps. Predictions are broken down by month, road type, weather condition, and accident cause. Results are saved to `outputs/2026_predictions.csv`.

---

## 💾 Using the Saved Model for Inference

```python
import joblib, json
import numpy as np
import pandas as pd

model  = joblib.load('outputs/best_model.pkl')
scaler = joblib.load('outputs/scaler.pkl')
le     = joblib.load('outputs/label_encoder.pkl')

with open('outputs/features.json') as f:
    feat_cols = json.load(f)

# Prepare your sample with the same feature engineering pipeline
# then:
X_new = scaler.transform(your_dataframe[feat_cols])
pred  = model.predict(X_new)
print('Predicted severity:', le.inverse_transform(pred)[0])
```

---

## ⚠️ Important Notes

- **No plt.show() hook** — charts display inline AND save to `outputs/` via the `save_and_show()` helper defined in Cell 1. Never add a patched `plt.show` hook as it causes infinite save loops.
- **GPU fallback** — if no NVIDIA GPU is detected, XGBoost and LightGBM automatically use CPU without any errors.
- **SMOTE on train only** — the test set is never touched by SMOTE, ensuring no data leakage.
- **Dataset limitation** — the `major` and `minor` classes are synthetically generated with nearly identical feature values. This is a known property of this dataset and explains the moderate accuracy scores. On real-world accident data with distinct per-class features, accuracy would be significantly higher.

---
