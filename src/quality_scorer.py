

import numpy as np


# Maximum realistic reference severities for normalization
MAX_DROP_FRACTION = 0.15     # Up to 15% slices missing
MAX_NOISE_SIGMA = 0.12       # Standard deviation of added noise
MAX_DRIFT_SEVERITY = 0.50    # Up to 50% intensity shift across stack
MAX_CLIP_PERCENTILE = 15.0   # Up to 15% dynamic range clipping


def compute_ground_truth_score(recipe):
    """
    Computes a deterministic usability score in [0, 100] and 3-class grade.
    
    Formula:
      Score = 100 - (P_drop + P_noise + P_drift + P_clip + P_compound)
      
    Penalties:
      - Missing Slices: up to 40 points
      - Noise:          up to 25 points
      - Drift:          up to 25 points
      - Clipping:       up to 20 points
      - Compound:       up to 10 points non-linear interaction
    """
    drop_frac = recipe.get("drop_fraction", 0.0)
    noise_sig = recipe.get("noise_sigma", 0.0)
    drift_sev = recipe.get("drift_severity", 0.0)
    clip_pct  = recipe.get("clip_percentile", 0.0)
    
    # 1. Normalized severities in [0, 1]
    s_drop  = min(1.0, drop_frac / MAX_DROP_FRACTION)
    s_noise = min(1.0, noise_sig / MAX_NOISE_SIGMA)
    s_drift = min(1.0, drift_sev / MAX_DRIFT_SEVERITY)
    s_clip  = min(1.0, clip_pct / MAX_CLIP_PERCENTILE)
    
    # 2. Individual penalties
    p_drop  = 40.0 * s_drop
    p_noise = 25.0 * s_noise
    p_drift = 25.0 * s_drift
    p_clip  = 20.0 * s_clip
    
    # 3. Compound interaction penalty (active when 2 or more defects present)
    active_severities = [s for s in [s_drop, s_noise, s_drift, s_clip] if s > 0.0]
    p_compound = 0.0
    if len(active_severities) >= 2:
        mean_active = float(np.mean(active_severities))
        p_compound = 10.0 * (mean_active ** 2)
        
    total_penalty = p_drop + p_noise + p_drift + p_clip + p_compound
    raw_score = 100.0 - total_penalty
    
    # Bound strictly between 0 and 100
    final_score = round(float(np.clip(raw_score, 0.0, 100.0)), 2)
    
    # 3-class grade mapping (Task 9)
    if final_score >= 80.0:
        grade = "accept"
    elif final_score >= 50.0:
        grade = "caution"
    else:
        grade = "reject"
        
    penalty_breakdown = {
        "p_drop": round(p_drop, 2),
        "p_noise": round(p_noise, 2),
        "p_drift": round(p_drift, 2),
        "p_clip": round(p_clip, 2),
        "p_compound": round(p_compound, 2),
    }
    
    return final_score, grade, penalty_breakdown


if __name__ == "__main__":
    print("Testing Quality Scoring Rule across diverse defect profiles...")
    
    test_cases = [
        {"name": "Clean Reference", "recipe": {}},
        {"name": "Mild Noise", "recipe": {"noise_sigma": 0.02}},
        {"name": "Moderate Drift", "recipe": {"drift_severity": 0.25}},
        {"name": "Severe Missing Slices", "recipe": {"drop_fraction": 0.12}},
        {"name": "Severe Multi-Defect", "recipe": {
            "drop_fraction": 0.10, "noise_sigma": 0.06, "drift_severity": 0.30, "clip_percentile": 8.0
        }}
    ]
    
    for case in test_cases:
        score, grade, breakdown = compute_ground_truth_score(case["recipe"])
        print(f"[{case['name']:<22}] -> Score: {score:5.1f} | Grade: {grade.upper():<7} | Penalties: {breakdown}")