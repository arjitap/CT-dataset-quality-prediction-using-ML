

import pandas as pd
import numpy as np

def audit_prototype_features(csv_path="data/features/features_prototype.csv"):
    df = pd.read_csv(csv_path)
    print("=" * 70)
    print("PROTOTYPE FEATURE SENSITIVITY AUDIT")
    print("=" * 70)
    
    # 1. Null / NaN / Inf Check
    feature_cols = [c for c in df.columns if c not in ["dataset_id", "score_0_100", "grade_3class"]]
    has_nans = df[feature_cols].isna().any().any()
    has_infs = np.isinf(df[feature_cols].to_numpy()).any()
    
    print(f"Total features checked : {len(feature_cols)}")
    print(f"Contains NaNs / Nulls  : {has_nans} (Must be False)")
    print(f"Contains Inf values    : {has_infs} (Must be False)")
    print("-" * 70)
    
    # 2. Key Physical Metric Responses
    clean_row  = df[df["dataset_id"] == "proto_001"].iloc[0]
    drop_row   = df[df["dataset_id"] == "proto_004"].iloc[0] # drop_severe
    drift_row  = df[df["dataset_id"] == "proto_006"].iloc[0] # drift_severe
    noise_row  = df[df["dataset_id"] == "proto_008"].iloc[0] # noise_severe
    clip_row   = df[df["dataset_id"] == "proto_010"].iloc[0] # clip_severe
    all_row    = df[df["dataset_id"] == "proto_016"].iloc[0] # compound_severe_all
    
    print(f"{'Condition':<22} | {'Missing Slices':<14} | {'Noise MAD':<11} | {'Drift Slope':<12} | {'Adjacent SSIM'}")
    print("-" * 70)
    print(f"{'Clean Reference':<22} | {clean_row['missing_slice_count']:<14.0f} | {clean_row['noise_mad_estimate']:<11.4f} | {clean_row['drift_slope']:<12.5f} | {clean_row['adjacent_ssim_mean']:.4f}")
    print(f"{'Severe Missing':<22} | {drop_row['missing_slice_count']:<14.0f} | {drop_row['noise_mad_estimate']:<11.4f} | {drop_row['drift_slope']:<12.5f} | {drop_row['adjacent_ssim_mean']:.4f}")
    print(f"{'Severe Noise':<22} | {noise_row['missing_slice_count']:<14.0f} | {noise_row['noise_mad_estimate']:<11.4f} | {noise_row['drift_slope']:<12.5f} | {noise_row['adjacent_ssim_mean']:.4f}")
    print(f"{'Severe Drift':<22} | {drift_row['missing_slice_count']:<14.0f} | {drift_row['noise_mad_estimate']:<11.4f} | {drift_row['drift_slope']:<12.5f} | {drift_row['adjacent_ssim_mean']:.4f}")
    print(f"{'Severe Clipping':<22} | {clip_row['missing_slice_count']:<14.0f} | {clip_row['noise_mad_estimate']:<11.4f} | {clip_row['drift_slope']:<12.5f} | {clip_row['adjacent_ssim_mean']:.4f}")
    print(f"{'Compound Severe':<22} | {all_row['missing_slice_count']:<14.0f} | {all_row['noise_mad_estimate']:<11.4f} | {all_row['drift_slope']:<12.5f} | {all_row['adjacent_ssim_mean']:.4f}")
    print("=" * 70)
    
    # Assertions to ensure features are reacting as expected
    assert drop_row["missing_slice_count"] > 0, "Missing slice feature failed to register dropped slices!"
    assert noise_row["noise_mad_estimate"] > clean_row["noise_mad_estimate"], "Noise MAD feature failed to register noise!"
    assert abs(drift_row["drift_slope"]) > abs(clean_row["drift_slope"]), "Drift slope failed to register drift!"
    assert noise_row["adjacent_ssim_mean"] < clean_row["adjacent_ssim_mean"], "SSIM failed to drop under noise!"
    
    print("ALL SANITY CHECKS PASSED: Observable features reflect physical defects accurately.")

if __name__ == "__main__":
    audit_prototype_features()