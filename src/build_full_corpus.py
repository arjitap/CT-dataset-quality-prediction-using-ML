"""
build_full_corpus.py
Generates the complete 600-volume synthetic CT corpus:
  1. Builds 5 distinct master clean ASTRA FBP volumes.
  2. Synthesizes 600 volume packages + JSON recipes across balanced defect cells.
  3. Extracts the 25-feature tabular dataset (data/features/features_full.csv).
  4. Partitions volume IDs into 70/15/15 Train/Val/Test splits + Unseen Combination set.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phantom_generator import generate_clean_reconstructed_volume
from defect_injector import inject_defects_into_volume
from quality_scorer import compute_ground_truth_score
from feature_extractor import extract_all_features


def generate_master_library(num_bases=5, depth=64, height=128, width=128):
    """
    Builds or loads a library of distinct clean ASTRA reconstructed master volumes.
    """
    os.makedirs("data/synthetic", exist_ok=True)
    master_vols = []
    
    for b_id in range(num_bases):
        base_path = f"data/synthetic/master_clean_base_{b_id:02d}.npy"
        if os.path.exists(base_path):
            print(f"[LOAD] Found clean base {b_id}: {base_path}")
            vol = np.load(base_path)
        else:
            print(f"[ASTRA] Reconstructing clean base {b_id + 1}/{num_bases}...")
            vol = generate_clean_reconstructed_volume(depth=depth, height=height, width=width, seed=42 + b_id * 17)
            np.save(base_path, vol)
        master_vols.append(vol)
        
    return master_vols


def build_defect_recipes():
    """
    Constructs 600 balanced defect recipes across 8 distinct regimes:
      1. Clean reference (60)
      2. Dropped slices (80)
      3. Intensity drift (80)
      4. Elevated noise (80)
      5. Clipping / Saturation (80)
      6. Compound: Drift + Noise (90)
      7. Compound: Drift + Clipping (90)
      8. Held-Out Unseen Combo: Noise + Dropped Slices (40)
    Total: 600 datasets
    """
    recipes = []
    
    # 1. Clean reference (60 volumes)
    for _ in range(60):
        recipes.append({
            "regime": "clean",
            "drop_fraction": 0.0,
            "drift_severity": 0.0,
            "noise_sigma": 0.0,
            "clip_percentile": 0.0
        })
        
    # 2. Dropped slices (80 volumes, 5 severities)
    drop_levels = [0.02, 0.05, 0.08, 0.11, 0.14]
    for lvl in drop_levels:
        for _ in range(16):
            recipes.append({
                "regime": "drop_only",
                "drop_fraction": float(lvl),
                "drift_severity": 0.0,
                "noise_sigma": 0.0,
                "clip_percentile": 0.0
            })
            
    # 3. Intensity drift (80 volumes, 5 severities)
    drift_levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    for lvl in drift_levels:
        for _ in range(16):
            recipes.append({
                "regime": "drift_only",
                "drop_fraction": 0.0,
                "drift_severity": float(lvl),
                "noise_sigma": 0.0,
                "clip_percentile": 0.0
            })
            
    # 4. Elevated noise (80 volumes, 5 severities)
    noise_levels = [0.02, 0.04, 0.06, 0.08, 0.11]
    for lvl in noise_levels:
        for _ in range(16):
            recipes.append({
                "regime": "noise_only",
                "drop_fraction": 0.0,
                "drift_severity": 0.0,
                "noise_sigma": float(lvl),
                "clip_percentile": 0.0
            })
            
    # 5. Clipping / saturation (80 volumes, 5 severities)
    clip_levels = [2.0, 5.0, 8.0, 11.0, 14.0]
    for lvl in clip_levels:
        for _ in range(16):
            recipes.append({
                "regime": "clip_only",
                "drop_fraction": 0.0,
                "drift_severity": 0.0,
                "noise_sigma": 0.0,
                "clip_percentile": float(lvl)
            })
            
    # 6. Compound: Drift + Noise (90 volumes)
    for _ in range(90):
        recipes.append({
            "regime": "compound_drift_noise",
            "drop_fraction": 0.0,
            "drift_severity": float(np.random.choice([0.15, 0.25, 0.35])),
            "noise_sigma": float(np.random.choice([0.03, 0.05, 0.08])),
            "clip_percentile": 0.0
        })
        
    # 7. Compound: Drift + Clipping (90 volumes)
    for _ in range(90):
        recipes.append({
            "regime": "compound_drift_clip",
            "drop_fraction": 0.0,
            "drift_severity": float(np.random.choice([0.15, 0.25, 0.35])),
            "noise_sigma": 0.0,
            "clip_percentile": float(np.random.choice([4.0, 7.0, 10.0]))
        })
        
    # 8. Unseen Combination (Held-Out for Generalization Test: 40 volumes)
    for _ in range(40):
        recipes.append({
            "regime": "unseen_noise_drop",
            "drop_fraction": float(np.random.choice([0.04, 0.08, 0.12])),
            "drift_severity": 0.0,
            "noise_sigma": float(np.random.choice([0.03, 0.06, 0.09])),
            "clip_percentile": 0.0
        })
        
    return recipes


def generate_full_dataset(output_csv="data/features/features_full.csv"):
    print("=" * 70)
    print("GENERATING FULL 600-VOLUME SYNTHETIC CT DATASET CORPUS")
    print("=" * 70)
    
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("data/recipes", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    
    # 1. Master base phantoms
    master_bases = generate_master_library(num_bases=5)
    num_bases = len(master_bases)
    
    # 2. Defect recipes
    recipes = build_defect_recipes()
    total_datasets = len(recipes)
    print(f"Total planned datasets: {total_datasets}")
    
    records = []
    
    for idx, recipe in enumerate(recipes):
        dataset_id = f"ds_{idx+1:04d}"
        seed = 2000 + idx
        base_id = idx % num_bases
        clean_vol = master_bases[base_id]
        
        # Inject defects
        corrupted_vol, slice_indices = inject_defects_into_volume(clean_vol, recipe, seed=seed)
        
        # Ground-truth scoring
        score, grade, breakdown = compute_ground_truth_score(recipe)
        
        # Save volume .npz
        vol_path = f"data/synthetic/{dataset_id}.npz"
        np.savez_compressed(vol_path, volume=corrupted_vol, slice_indices=slice_indices)
        
        # Save JSON recipe metadata
        meta = {
            "dataset_id": dataset_id,
            "base_phantom_id": base_id,
            "seed": seed,
            "defect_regime": recipe["regime"],
            "defect_parameters": recipe,
            "ground_truth_score": score,
            "ground_truth_grade": grade,
            "penalty_breakdown": breakdown,
            "volume_shape": list(corrupted_vol.shape),
            "acquired_slice_count": len(slice_indices)
        }
        with open(f"data/recipes/{dataset_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        # Extract features (zero recipe leakage)
        feats = extract_all_features(corrupted_vol, slice_indices)
        
        row = {
            "dataset_id": dataset_id,
            "regime": recipe["regime"],
            **feats,
            "score_0_100": score,
            "grade_3class": grade
        }
        records.append(row)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total_datasets:
            print(f"[{idx+1:03d}/{total_datasets}] Processed {dataset_id} | Regime: {recipe['regime']:<21} | Grade: {grade.upper()}")
            
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Full features table saved to: {output_csv}")
    print(f"Shape: {df.shape} (600 rows, 29 columns)")
    
    # 3. Create Partition Splits
    print("\n" + "=" * 70)
    print("CREATING STRATIFIED TRAIN / VAL / TEST SPLITS (70 / 15 / 15)")
    print("=" * 70)
    
    # Separate the held-out unseen combination first
    unseen_df = df[df["regime"] == "unseen_noise_drop"]
    standard_df = df[df["regime"] != "unseen_noise_drop"]
    
    # 70% Train, 15% Val, 15% Test stratified by 3-class grade
    train_df, temp_df = train_test_split(
        standard_df, test_size=0.30, random_state=42, stratify=standard_df["grade_3class"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["grade_3class"]
    )
    
    # Save ID lists to text files
    with open("data/splits/train_ids.txt", "w") as f:
        f.write("\n".join(train_df["dataset_id"].tolist()))
    with open("data/splits/val_ids.txt", "w") as f:
        f.write("\n".join(val_df["dataset_id"].tolist()))
    with open("data/splits/test_ids.txt", "w") as f:
        f.write("\n".join(test_df["dataset_id"].tolist()))
    with open("data/splits/unseen_ids.txt", "w") as f:
        f.write("\n".join(unseen_df["dataset_id"].tolist()))
        
    print(f"Training set count         : {len(train_df)} volumes (Saved to data/splits/train_ids.txt)")
    print(f"Validation set count       : {len(val_df)} volumes (Saved to data/splits/val_ids.txt)")
    print(f"In-distribution Test count : {len(test_df)} volumes (Saved to data/splits/test_ids.txt)")
    print(f"Unseen Combination Test    : {len(unseen_df)} volumes (Saved to data/splits/unseen_ids.txt)")
    print("=" * 70)
    print("FULL CORPUS AND SPLITS CREATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    generate_full_dataset()