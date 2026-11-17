"""
gate.py
Operational Dataset Quality Gate CLI.
Ingests a single CT volume (.npz) or a directory of volumes,
extracts volume-level statistics, and predicts usability grade/score.
"""

import os
import sys
import glob
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from feature_extractor import extract_all_features


class CTDatasetQualityGate:
    def __init__(self, models_dir="models"):
        self.scaler = joblib.load(os.path.join(models_dir, "feature_scaler.joblib"))
        self.le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
        self.feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.joblib"))
        self.classifier = joblib.load(os.path.join(models_dir, "rf_classifier.joblib"))
        self.regressor = joblib.load(os.path.join(models_dir, "xgb_regressor.joblib"))

    def evaluate_volume(self, npz_path):
        """
        Evaluates a single CT dataset container (.npz).
        Returns predicted score, grade, class probabilities, and extracted features.
        """
        data = np.load(npz_path)
        volume = data["volume"]
        slice_indices = data["slice_indices"]

        # Extract features
        raw_feats = extract_all_features(volume, slice_indices)

        # Vectorize and scale
        feat_vector = np.array([[raw_feats[col] for col in self.feature_cols]], dtype=np.float32)
        scaled_vector = self.scaler.transform(feat_vector)

        # Predict grade and probabilities
        pred_class_idx = self.classifier.predict(scaled_vector)[0]
        pred_grade = self.le.inverse_transform([pred_class_idx])[0]
        probs = self.classifier.predict_proba(scaled_vector)[0]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.le.classes_, probs)}

        # Predict continuous usability score
        pred_score = float(np.clip(self.regressor.predict(scaled_vector)[0], 0.0, 100.0))

        report = {
            "dataset_file": os.path.basename(npz_path),
            "predicted_grade": pred_grade.upper(),
            "predicted_score": round(pred_score, 2),
            "class_probabilities": prob_dict,
            "acquired_slices": len(slice_indices),
            "volume_dimensions": list(volume.shape),
            "key_indicators": {
                "missing_slices": int(raw_feats["missing_slice_count"]),
                "noise_mad_estimate": round(float(raw_feats["noise_mad_estimate"]), 4),
                "drift_slope": round(float(raw_feats["drift_slope"]), 6),
                "adjacent_ssim_mean": round(float(raw_feats["adjacent_ssim_mean"]), 4)
            }
        }
        return report


def format_report_console(report):
    print("=" * 65)
    print("AUTOMATED CT DATASET QUALITY GATE REPORT")
    print("=" * 65)
    print(f"Dataset File        : {report['dataset_file']}")
    print(f"Volume Shape        : {report['volume_dimensions']} (Slices, Height, Width)")
    print(f"Acquired Slices     : {report['acquired_slices']}")
    print("-" * 65)
    print(f"PREDICTED GRADE     : [ {report['predicted_grade']} ]")
    print(f"USABILITY SCORE     : {report['predicted_score']:.1f} / 100.0")
    print("-" * 65)
    print("Class Probabilities :")
    for cls, prob in report["class_probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  - {cls.upper():<7}: {prob * 100:5.1f}% | {bar}")
    print("-" * 65)
    print("Key Observable Indicators:")
    print(f"  - Missing Slices  : {report['key_indicators']['missing_slices']}")
    print(f"  - Noise Estimate  : {report['key_indicators']['noise_mad_estimate']}")
    print(f"  - Drift Slope     : {report['key_indicators']['drift_slope']}")
    print(f"  - Inter-Slice SSIM: {report['key_indicators']['adjacent_ssim_mean']}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="CT Dataset Quality Gate CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to a single .npz volume or directory of .npz volumes")
    parser.add_argument("--save_json", action="store_true", help="Save triage report as JSON")
    args = parser.parse_args()

    gate = CTDatasetQualityGate()

    if os.path.isfile(args.input):
        report = gate.evaluate_volume(args.input)
        format_report_console(report)

        if args.save_json:
            os.makedirs("results/reports", exist_ok=True)
            out_name = os.path.splitext(os.path.basename(args.input))[0] + "_report.json"
            out_path = os.path.join("results/reports", out_name)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[OK] Report JSON saved to: {out_path}")

    elif os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
        if not files:
            print(f"No .npz files found in {args.input}")
            return

        print(f"Batch evaluating {len(files)} volumes in {args.input}...")
        summary = []
        for f in files:
            rep = gate.evaluate_volume(f)
            summary.append({
                "file": rep["dataset_file"],
                "grade": rep["predicted_grade"],
                "score": rep["predicted_score"],
                "missing": rep["key_indicators"]["missing_slices"],
                "noise": rep["key_indicators"]["noise_mad_estimate"]
            })

        sum_df = pd.DataFrame(summary)
        print("\n" + sum_df.to_string(index=False))

        os.makedirs("results/reports", exist_ok=True)
        sum_df.to_csv("results/reports/batch_triage_summary.csv", index=False)
        print("\n[OK] Batch triage saved to: results/reports/batch_triage_summary.csv")


if __name__ == "__main__":
    main()