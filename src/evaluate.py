
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, root_mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_evaluation_data():
    
    df = pd.read_csv("data/features/features_full.csv")

    with open("data/splits/test_ids.txt") as f:
        test_ids = [line.strip() for line in f if line.strip()]
    with open("data/splits/unseen_ids.txt") as f:
        unseen_ids = [line.strip() for line in f if line.strip()]

    test_df = df[df["dataset_id"].isin(test_ids)].copy()
    unseen_df = df[df["dataset_id"].isin(unseen_ids)].copy()

    feature_cols = joblib.load("models/feature_columns.joblib")
    scaler = joblib.load("models/feature_scaler.joblib")
    le = joblib.load("models/label_encoder.joblib")

    return test_df, unseen_df, feature_cols, scaler, le


def experiment_1_untouched_test_evaluation(test_df, feature_cols, scaler, le):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: UNTOUCHED TEST SET EVALUATION (IN-DISTRIBUTION)")
    print("=" * 70)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test_cls = le.transform(test_df["grade_3class"].values)
    y_test_reg = test_df["score_0_100"].values

    # Load models
    rf_clf = joblib.load("models/rf_classifier.joblib")
    xgb_reg = joblib.load("models/xgb_regressor.joblib")

    # Classification metrics
    preds_cls = rf_clf.predict(X_test)
    acc = accuracy_score(y_test_cls, preds_cls)
    macro_f1 = f1_score(y_test_cls, preds_cls, average="macro", zero_division=0)

    # Regression metrics
    preds_reg = np.clip(xgb_reg.predict(X_test), 0.0, 100.0)
    mae = mean_absolute_error(y_test_reg, preds_reg)
    rmse = root_mean_squared_error(y_test_reg, preds_reg)
    r2 = r2_score(y_test_reg, preds_reg)

    print(f"Test Samples Evaluated     : {len(test_df)}")
    print(f"Grade Accuracy (RF)        : {acc * 100:.2f}%")
    print(f"Grade Macro-F1 (RF)        : {macro_f1:.4f}")
    print(f"Score MAE (XGB Regressor)  : {mae:.2f} points (Target: <= 6.0)")
    print(f"Score RMSE (XGB Regressor) : {rmse:.2f} points")
    print(f"Score R^2 (XGB Regressor)  : {r2:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_cls, preds_cls, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix: 3-Class Triage (Test Set)")
    plt.ylabel("True Grade")
    plt.xlabel("Predicted Grade")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix_test.png", dpi=150)
    plt.close()

    # Plot Predicted vs True Score Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_reg, preds_reg, alpha=0.75, edgecolors="k", color="#2b5c8f")
    plt.plot([0, 100], [0, 100], "--r", linewidth=2, label="Ideal 1:1 Line")
    plt.axvline(80, color="gray", linestyle=":", label="Accept Boundary (80)")
    plt.axvline(50, color="gray", linestyle="--", label="Caution Boundary (50)")
    plt.title(f"Score Regression: True vs Predicted (MAE = {mae:.2f})")
    plt.xlabel("Ground-Truth Usability Score")
    plt.ylabel("Predicted Usability Score")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/figures/score_regression_scatter.png", dpi=150)
    plt.close()

    return {"accuracy": acc, "macro_f1": macro_f1, "mae": mae, "rmse": rmse, "r2": r2}


def experiment_2_feature_importance(feature_cols):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: FEATURE IMPORTANCE RANKING")
    print("=" * 70)

    rf_clf = joblib.load("models/rf_classifier.joblib")
    importances = rf_clf.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("Top 10 Most Influential Features:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<28} : {row['importance']:.4f}")

    fi_df.to_csv("results/metrics/feature_importances.csv", index=False)

    # Plot Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(15), x="importance", y="feature", palette="viridis")
    plt.title("Top 15 Engineered Features by Gini Importance (Random Forest)")
    plt.xlabel("Mean Impurity Decrease (Importance)")
    plt.ylabel("Aggregate Feature")
    plt.tight_layout()
    plt.savefig("results/figures/feature_importance_top15.png", dpi=150)
    plt.close()


def experiment_3_unseen_combination_generalization(unseen_df, feature_cols, scaler, le):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: UNSEEN DEFECT COMBINATION GENERALIZATION")
    print("  (Evaluated on: Noise + Missing Slices combo held out from training)")
    print("=" * 70)

    X_unseen = scaler.transform(unseen_df[feature_cols].values)
    y_unseen_cls = le.transform(unseen_df["grade_3class"].values)
    y_unseen_reg = unseen_df["score_0_100"].values

    rf_clf = joblib.load("models/rf_classifier.joblib")
    xgb_reg = joblib.load("models/xgb_regressor.joblib")

    preds_cls = rf_clf.predict(X_unseen)
    acc = accuracy_score(y_unseen_cls, preds_cls)
    macro_f1 = f1_score(y_unseen_cls, preds_cls, average="macro", zero_division=0)

    preds_reg = np.clip(xgb_reg.predict(X_unseen), 0.0, 100.0)
    mae = mean_absolute_error(y_unseen_reg, preds_reg)

    print(f"Unseen Samples Evaluated   : {len(unseen_df)}")
    print(f"Grade Accuracy (Unseen)    : {acc * 100:.2f}% (Expected spec range: 75% - 85%)")
    print(f"Grade Macro-F1 (Unseen)    : {macro_f1:.4f}")
    print(f"Score MAE (Unseen)         : {mae:.2f} points")

    return {"unseen_accuracy": acc, "unseen_macro_f1": macro_f1, "unseen_mae": mae}


def experiment_4_feature_ablation_study():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: FEATURE ABLATION STUDY")
    print("  (Measuring impact on Accuracy & Macro-F1 when feature subsets are omitted)")
    print("=" * 70)

    df = pd.read_csv("data/features/features_full.csv")
    with open("data/splits/train_ids.txt") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open("data/splits/test_ids.txt") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    train_df = df[df["dataset_id"].isin(train_ids)].copy()
    test_df = df[df["dataset_id"].isin(test_ids)].copy()

    feature_families = {
        "Noise Features": ["noise_mad_estimate", "noise_local_variance_mean", "laplacian_variance", "high_freq_energy_ratio"],
        "Drift Features": ["drift_slope", "drift_r2", "slice_mean_diff_mean", "slice_mean_diff_max", "slice_mean_cov"],
        "SSIM / Consistency": ["adjacent_ssim_mean", "adjacent_ssim_std", "adjacent_ssim_min", "adjacent_ssim_p10"],
        "Completeness / Drops": ["missing_slice_count", "missing_slice_fraction"],
        "Clipping / Dynamic Range": ["clipped_low_fraction", "clipped_high_fraction", "effective_dynamic_range"]
    }

    all_features = [c for c in df.columns if c not in ["dataset_id", "regime", "score_0_100", "grade_3class"]]

    ablation_results = []

    # 1. Baseline Full Model
    def train_and_eval_rf(feats_to_use):
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_df[feats_to_use].values)
        X_te = sc.transform(test_df[feats_to_use].values)
        y_tr = train_df["grade_3class"].values
        y_te = test_df["grade_3class"].values

        clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        return acc, f1

    full_acc, full_f1 = train_and_eval_rf(all_features)
    ablation_results.append({
        "configuration": "Full Feature Set (All 25)",
        "features_used": len(all_features),
        "test_accuracy": round(full_acc * 100, 2),
        "macro_f1": round(full_f1, 4),
        "f1_drop": 0.0000
    })

    # 2. Drop each family one-by-one
    for family_name, dropped_cols in feature_families.items():
        subset = [c for c in all_features if c not in dropped_cols]
        sub_acc, sub_f1 = train_and_eval_rf(subset)
        f1_drop = full_f1 - sub_f1
        ablation_results.append({
            "configuration": f"Omit {family_name}",
            "features_used": len(subset),
            "test_accuracy": round(sub_acc * 100, 2),
            "macro_f1": round(sub_f1, 4),
            "f1_drop": round(f1_drop, 4)
        })

    abl_df = pd.DataFrame(ablation_results)
    print(abl_df.to_string(index=False))
    abl_df.to_csv("results/metrics/ablation_study_results.csv", index=False)


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    test_df, unseen_df, feature_cols, scaler, le = load_evaluation_data()

    # Run the 4 experiments
    exp1_res = experiment_1_untouched_test_evaluation(test_df, feature_cols, scaler, le)
    experiment_2_feature_importance(feature_cols)
    exp3_res = experiment_3_unseen_combination_generalization(unseen_df, feature_cols, scaler, le)
    experiment_4_feature_ablation_study()

    # Save consolidated test metrics
    summary = {
        "in_distribution_test": exp1_res,
        "unseen_combination_test": exp3_res
    }
    with open("results/metrics/final_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Figures saved in results/figures/ | Metrics saved in results/metrics/")
    print("=" * 70)