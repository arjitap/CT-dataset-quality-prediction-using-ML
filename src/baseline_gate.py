

import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix


class RuleBasedQualityGate:
    """
    Fixed-threshold triage gate based on 3 primary physical indicators:
      1. missing_slice_fraction
      2. noise_mad_estimate
      3. abs(drift_slope)
    """
    def __init__(self, drop_caution=0.04, drop_reject=0.10,
                 noise_caution=0.15, noise_reject=0.25,
                 drift_caution=0.00035, drift_reject=0.00075):
        self.drop_caution = drop_caution
        self.drop_reject = drop_reject
        self.noise_caution = noise_caution
        self.noise_reject = noise_reject
        self.drift_caution = drift_caution
        self.drift_reject = drift_reject

    def predict_single(self, row):
        missing_frac = row["missing_slice_fraction"]
        noise_mad = row["noise_mad_estimate"]
        drift_slope = abs(row["drift_slope"])

        # Severe trigger -> REJECT
        if (missing_frac >= self.drop_reject or
            noise_mad >= self.noise_reject or
            drift_slope >= self.drift_reject):
            return "reject"

        # Moderate trigger -> CAUTION
        if (missing_frac >= self.drop_caution or
            noise_mad >= self.noise_caution or
            drift_slope >= self.drift_caution):
            return "caution"

        return "accept"

    def predict(self, df):
        return [self.predict_single(row) for _, row in df.iterrows()]


def evaluate_baseline():
    print("=" * 65)
    print("STAGE 6A: EVALUATING RULE-BASED HEURISTIC BASELINE GATE")
    print("=" * 65)

    df_all = pd.read_csv("data/features/features_full.csv")

    with open("data/splits/val_ids.txt") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    val_df = df_all[df_all["dataset_id"].isin(val_ids)].copy()
    y_true = val_df["grade_3class"].values

    gate = RuleBasedQualityGate()
    y_pred = gate.predict(val_df)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    labels = ["accept", "caution", "reject"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Validation Volumes Evaluated: {len(val_df)}")
    print(f"Baseline Accuracy          : {acc * 100:.2f}%")
    print(f"Baseline Macro-F1          : {macro_f1:.4f}")
    print("\nClassification Report (Baseline Gate):")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))

    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    print(f"{'':<10} {'Pred Accept':<12} {'Pred Caution':<14} {'Pred Reject'}")
    for idx, row_name in enumerate(["True Accept", "True Caution", "True Reject"]):
        print(f"{row_name:<10} {cm[idx][0]:<12} {cm[idx][1]:<14} {cm[idx][2]}")

    os.makedirs("results/metrics", exist_ok=True)
    metrics_path = "results/metrics/baseline_metrics.json"
    metrics = {
        "model": "Rule-Based Heuristic Baseline",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Baseline metrics saved to: {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_baseline()