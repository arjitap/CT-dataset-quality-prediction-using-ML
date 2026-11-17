
import os
import json
import numpy as np
import pandas as pd

from phantom_generator import generate_clean_reconstructed_volume
from defect_injector import inject_defects_into_volume
from quality_scorer import compute_ground_truth_score
from feature_extractor import extract_all_features


def create_prototype_recipes():
    """
    Defines 16 distinct defect recipes spanning clean, single-defect,
    and compound multi-defect regimes.
    """
    recipes = [
        # 1-2: Clean References
        {"name": "clean_01", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 0.0},
        {"name": "clean_02", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 0.0},
        
        # 3-4: Dropped / Missing Slices
        {"name": "drop_mild", "drop_fraction": 0.04, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 0.0},
        {"name": "drop_severe", "drop_fraction": 0.12, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 0.0},
        
        # 5-6: Intensity Drift
        {"name": "drift_mild", "drop_fraction": 0.0, "drift_severity": 0.15, "noise_sigma": 0.0, "clip_percentile": 0.0},
        {"name": "drift_severe", "drop_fraction": 0.0, "drift_severity": 0.40, "noise_sigma": 0.0, "clip_percentile": 0.0},
        
        # 7-8: Elevated Noise
        {"name": "noise_mild", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.03, "clip_percentile": 0.0},
        {"name": "noise_severe", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.09, "clip_percentile": 0.0},
        
        # 9-10: Dynamic Range Clipping
        {"name": "clip_mild", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 4.0},
        {"name": "clip_severe", "drop_fraction": 0.0, "drift_severity": 0.0, "noise_sigma": 0.0, "clip_percentile": 12.0},
        
        # 11-16: Compound Defect Mixtures
        {"name": "compound_drift_noise", "drop_fraction": 0.0, "drift_severity": 0.20, "noise_sigma": 0.04, "clip_percentile": 0.0},
        {"name": "compound_drop_noise", "drop_fraction": 0.06, "drift_severity": 0.0, "noise_sigma": 0.04, "clip_percentile": 0.0},
        {"name": "compound_drift_clip", "drop_fraction": 0.0, "drift_severity": 0.25, "noise_sigma": 0.0, "clip_percentile": 6.0},
        {"name": "compound_drop_drift", "drop_fraction": 0.05, "drift_severity": 0.25, "noise_sigma": 0.0, "clip_percentile": 0.0},
        {"name": "compound_triad", "drop_fraction": 0.05, "drift_severity": 0.20, "noise_sigma": 0.03, "clip_percentile": 5.0},
        {"name": "compound_severe_all", "drop_fraction": 0.10, "drift_severity": 0.35, "noise_sigma": 0.08, "clip_percentile": 10.0},
    ]
    return recipes


def run_prototype_generation(output_csv="data/features/features_prototype.csv"):
    """
    Generates 16 prototype datasets, saves .npz files, JSON recipes,
    and extracts the 25-feature tabular CSV.
    """
    print("=" * 65)
    print("STARTING PHASE A: GENERATING 16 PROTOTYPE CT DATASETS")
    print("=" * 65)
    
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("data/recipes", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    
    recipes = create_prototype_recipes()
    
    # Load clean ASTRA base volume
    clean_master_path = "data/synthetic/master_clean_base_00.npy"
    if os.path.exists(clean_master_path):
        print(f"[LOAD] Using existing clean ASTRA base: {clean_master_path}")
        clean_base = np.load(clean_master_path)
    else:
        print("[BUILD] Generating clean ASTRA base volume...")
        clean_base = generate_clean_reconstructed_volume(depth=64, height=128, width=128, seed=42)
        np.save(clean_master_path, clean_base)
        
    records = []
    
    for i, recipe in enumerate(recipes):
        dataset_id = f"proto_{i+1:03d}"
        seed = 1000 + i
        
        # 1. Corrupt volume
        corrupted_vol, slice_indices = inject_defects_into_volume(clean_base, recipe, seed=seed)
        
        # 2. Compute ground-truth score & grade
        score, grade, breakdown = compute_ground_truth_score(recipe)
        
        # 3. Save volume array and slice indices to .npz
        vol_file = f"data/synthetic/{dataset_id}.npz"
        np.savez_compressed(vol_file, volume=corrupted_vol, slice_indices=slice_indices)
        
        # 4. Save JSON recipe
        recipe_metadata = {
            "dataset_id": dataset_id,
            "seed": seed,
            "recipe_name": recipe["name"],
            "defect_parameters": recipe,
            "ground_truth_score": score,
            "ground_truth_grade": grade,
            "penalty_breakdown": breakdown,
            "volume_shape": list(corrupted_vol.shape),
            "acquired_slice_count": len(slice_indices)
        }
        recipe_file = f"data/recipes/{dataset_id}.json"
        with open(recipe_file, "w") as f:
            json.dump(recipe_metadata, f, indent=2)
            
        # 5. Extract 25 features (strictly no recipe leakage)
        feats = extract_all_features(corrupted_vol, slice_indices)
        
        row = {"dataset_id": dataset_id, **feats, "score_0_100": score, "grade_3class": grade}
        records.append(row)
        
        print(f"[{dataset_id}] {recipe['name']:<22} | Score: {score:5.1f} | Grade: {grade.upper():<7} | Saved: {vol_file}")
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print("=" * 65)
    print("SUCCESS: 16 Prototype datasets created!")
    print(f"Features saved to: {output_csv}")
    print(f"Dataset shape: {df.shape} (16 rows, 28 columns)")
    print("=" * 65)
    return df


if __name__ == "__main__":
    df_proto = run_prototype_generation()