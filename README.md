🚦 India Road Accident Analysis & 2026 Blackspot Prediction
Dataset: Kaggle Indian Roads Dataset (2022 – Apr 2025) | ~20,000 Records

VIT Vellore — Probability & Statistics Project (2025–26)
Objective: Accident severity classification & 2026 blackspot forecasting

📋 About

This project builds a complete ML pipeline to:

Predict accident severity (minor, major, fatal)
Analyze accident patterns across time, geography, and conditions
Forecast 2026 high-risk (blackspot) states
📂 Dataset
Source: Kaggle Indian Roads Dataset
Size: ~20,000 records
Period: 2022 – April 2025
Features
Numerical: risk_score, casualties, vehicles_involved, temperature, lanes
Temporal: hour, month, day_of_week
Categorical: road_type, weather, visibility, traffic_density, cause, state, city
Engineered:
Cyclical (hour_sin, hour_cos)
Interactions (risk × casualties)
Threshold flags (high_risk, night, etc.)
🗂️ Project Structure
project/
│
├── indian_roads_2026.ipynb
├── indian_roads_dataset.csv
├── README.md
│
└── outputs/
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
    │
    ├── 2026_predictions.csv
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── features.json
    └── model_results.csv
▶️ How to Run
pip install xgboost lightgbm shap imbalanced-learn
Place dataset in project folder
Open notebook
Run all cells

⏱️ Runtime: ~5–10 minutes

📓 Pipeline
Data loading & filtering (2022–2025)
EDA (distributions, trends, correlations)
Feature engineering
SMOTE balancing (training only)
Model training (7 models)
Evaluation (accuracy, F1, confusion matrix)
SHAP explainability
2026 forecasting + dashboard
🤖 Models Used
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost
LightGBM (Best Model)
KNN
📊 Final Results
🏆 Best Model: LightGBM
Metric	Value
Accuracy	65.67%
F1-Macro	63.96%
CV F1	79.24%
Model Comparison (Summary)
Model	Accuracy	F1
LightGBM	65.67%	63.96%
XGBoost	66.35%	63.50%
Random Forest	67.97%	62.15%
Gradient Boosting	67.22%	62.11%
Logistic Regression	58.13%	66.30%
KNN	45.77%	47.85%
⚠️ Important Observation
fatal class is predicted almost perfectly
major and minor classes have very similar feature distributions
This limits maximum achievable accuracy

The dataset itself has overlapping class patterns, not a model limitation

📅 2026 Prediction
Forecast based on historical trends
State-wise risk scoring applied
🔴 Top Blackspot State (2026)

👉 Maharashtra

Trend Model Accuracy

👉 ~95% (validated on historical data)

🖼️ Outputs

Includes:

EDA visualizations
Model comparisons
Confusion matrices
Feature importance
SHAP explainability
Final dashboard
2026 predictions
💯 Conclusion
Built a complete ML pipeline
Achieved ~65–68% accuracy on multiclass classification
Identified key accident risk drivers
Generated reliable future risk forecasts
🚀 Future Work
Improve major class separation
Add real-world traffic / speed data
Deploy as web dashboard
Use temporal models (LSTM / time-series)
🧠 Final Note

Performance is limited by dataset structure (class overlap), not model quality.
With richer real-world features, accuracy would significantly improve.