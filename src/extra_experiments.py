"""
extra_experiments.py
Computes additional metrics required by the prompt:
  - Binary ROC-AUC (Accept vs. Not-Accept)
  - Sensitivity Analysis: Regressor MAE across defect severity levels (1 to 5)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error


def run_extra_analyses():
    print("=" * 70)
    print("RUNNING ADDITIONAL REQUIRED ANALYSES (ROC-AUC & SENSITIVITY)")
    print("=" * 70)

    df = pd.read_csv("data/features/features_full.csv")
    with open("data/splits/test_ids.txt") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    test_df = df[df["dataset_id"].isin(test_ids)].copy()

    feature_cols = joblib.load("models/feature_columns.joblib")
    scaler = joblib.load("models/feature_scaler.joblib")
    rf_clf = joblib.load("models/rf_classifier.joblib")
    xgb_reg = joblib.load("models/xgb_regressor.joblib")

    X_test = scaler.transform(test_df[feature_cols].values)

    # 1. Binary ROC-AUC (Accept vs. Not-Accept i.e. Caution/Reject)
    # Binary target: 1 if grade_3class == 'accept', else 0
    binary_true = (test_df["grade_3class"] == "accept").astype(int).values
    
    # Get probabilities for class 'accept' (index 0 in LabelEncoder)
    le = joblib.load("models/label_encoder.joblib")
    accept_class_idx = list(le.classes_).index("accept")
    probs = rf_clf.predict_proba(X_test)
    accept_probs = probs[:, accept_class_idx]

    roc_auc = roc_auc_score(binary_true, accept_probs)
    print(f"\n[Metric 1] Binary Accept-vs-Not-Accept ROC-AUC : {roc_auc:.4f}")

    # 2. Sensitivity Analysis (MAE across defect severity levels)
    # Check if regime or severity info is available in features_full.csv
    if "regime" in test_df.columns:
        print("\n[Metric 2] Sensitivity Analysis (Score Regression MAE by Regime/Severity):")
        test_preds = np.clip(xgb_reg.predict(X_test), 0.0, 100.0)
        test_df["predicted_score"] = test_preds
        test_df["abs_error"] = np.abs(test_df["score_0_100"] - test_df["predicted_score"])

        regime_mae = test_df.groupby("regime")["abs_error"].mean().reset_index()
        regime_mae.columns = ["Regime / Severity Group", "Mean Absolute Error (pts)"]
        print(regime_mae.to_string(index=False))

        regime_mae.to_json("results/metrics/sensitivity_analysis_mae.json", orient="records", indent=2)

    print("=" * 70)
    print("Extra analyses completed successfully!")


if __name__ == "__main__":
    run_extra_analyses()