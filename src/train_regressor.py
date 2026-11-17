"""
train_regressor.py
Trains and compares 3 regression models for continuous 0-100 scoring:
  1. Linear Regression
  2. Random Forest Regressor
  3. XGBoost Regressor
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb


def run_expanded_regression():
    print("=" * 75)
    print("EXPANDED REGRESSION MODEL COMPARISON (WITH SENSOR JITTER)")
    print("=" * 75)

    df = pd.read_csv("data/features/features_full.csv")
    with open("data/splits/train_ids.txt") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open("data/splits/val_ids.txt") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    train_df = df[df["dataset_id"].isin(train_ids)].copy()
    val_df = df[df["dataset_id"].isin(val_ids)].copy()

    feature_cols = [c for c in df.columns if c not in ["dataset_id", "regime", "score_0_100", "grade_3class"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["score_0_100"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_0_100"].values

    # Inject realistic sensor noise/jitter to training features only
    np.random.seed(42)
    feature_stds = np.std(X_train, axis=0) + 1e-8
    X_train = X_train + np.random.normal(0, 0.02, X_train.shape) * feature_stds

    scaler = joblib.load("models/feature_scaler.joblib")
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Reg": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost Reg": xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    }

    results = {}
    print(f"{'Model Name':<22} | {'Val MAE (pts)':<14} | {'Val RMSE':<12} | {'R^2 Score'}")
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = np.clip(model.predict(X_val_scaled), 0.0, 100.0)

        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        results[name] = {
            "val_mae": round(float(mae), 2),
            "val_rmse": round(float(rmse), 2),
            "val_r2": round(float(r2), 4)
        }
        print(f"{name:<22} | {mae:<14.2f} | {rmse:<12.2f} | {r2:.4f}")

    print("=" * 70)
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/regressor_expanded_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(models["XGBoost Reg"], "models/xgb_regressor.joblib")
    print("Regressors saved successfully!")


if __name__ == "__main__":
    run_expanded_regression()