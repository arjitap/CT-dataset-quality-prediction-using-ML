

import numpy as np


def apply_dropped_slices(volume, slice_indices, drop_fraction, rng):
    """
    Simulates missing slices by removing them from the volume array
    and recording the remaining valid physical indices.
    """
    if drop_fraction <= 0.0:
        return volume, slice_indices
        
    num_total = len(slice_indices)
    num_drops = int(round(num_total * drop_fraction))
    if num_drops == 0:
        return volume, slice_indices
        
    # Pick random indices to drop
    drop_positions = rng.choice(num_total, size=num_drops, replace=False)
    keep_mask = np.ones(num_total, dtype=bool)
    keep_mask[drop_positions] = False
    
    corrupted_vol = volume[keep_mask].copy()
    remaining_indices = slice_indices[keep_mask].copy()
    
    return corrupted_vol, remaining_indices


def apply_intensity_drift(volume, drift_severity, drift_type="linear"):
    """
    Simulates thermal ramp or detector drift across the slice stack (Z).
    """
    if drift_severity <= 0.0:
        return volume
        
    depth = volume.shape[0]
    if drift_type == "linear":
        # Multiplicative ramp from 1.0 to (1.0 + drift_severity)
        ramp = np.linspace(0.0, drift_severity, depth, dtype=np.float32)[:, None, None]
    else:
        # Sinusoidal fluctuation
        ramp = (drift_severity * np.sin(np.linspace(0, 2 * np.pi, depth, dtype=np.float32)))[:, None, None]
        
    corrupted_vol = volume * (1.0 + ramp)
    return corrupted_vol


def apply_noise_inflation(volume, noise_sigma, rng):
    """
    Simulates elevated Gaussian electronic noise and Poisson noise surrogate.
    """
    if noise_sigma <= 0.0:
        return volume
        
    # Gaussian electronic noise
    gaussian_noise = rng.normal(0.0, noise_sigma, size=volume.shape).astype(np.float32)
    corrupted_vol = volume + gaussian_noise
    
    # Physical constraint: attenuation cannot be negative
    corrupted_vol = np.clip(corrupted_vol, 0.0, None)
    return corrupted_vol


def apply_dynamic_range_clipping(volume, clip_percentile):
    """
    Simulates ADC clipping, detector saturation, and loss of dynamic range.
    """
    if clip_percentile <= 0.0:
        return volume
        
    # Compute low and high cutoffs from non-zero object voxels
    non_zeros = volume[volume > 0.05]
    if len(non_zeros) == 0:
        return volume
        
    low_val = float(np.percentile(non_zeros, clip_percentile))
    high_val = float(np.percentile(volume, 100.0 - clip_percentile))
    
    corrupted_vol = np.clip(volume, low_val, high_val)
    
    # Rescale back to nominal scale [0, max]
    if high_val > low_val:
        corrupted_vol = (corrupted_vol - low_val) / (high_val - low_val) * high_val
        
    return corrupted_vol.astype(np.float32)


def inject_defects_into_volume(clean_volume, recipe, seed=None):
    """
    Master function: takes a clean CT volume and a defect recipe,
    applies the specified corruptions, and returns the corrupted volume
    along with its physical slice index list.
    """
    rng = np.random.default_rng(seed)
    
    depth = clean_volume.shape[0]
    slice_indices = np.arange(depth, dtype=np.int32)
    vol = clean_volume.copy()
    
    # 1. Apply Intensity Drift
    drift_sev = recipe.get("drift_severity", 0.0)
    drift_type = recipe.get("drift_type", "linear")
    vol = apply_intensity_drift(vol, drift_sev, drift_type)
    
    # 2. Apply Global Noise Inflation
    noise_sig = recipe.get("noise_sigma", 0.0)
    vol = apply_noise_inflation(vol, noise_sig, rng)
    
    # 3. Apply Dynamic Range Clipping
    clip_pct = recipe.get("clip_percentile", 0.0)
    vol = apply_dynamic_range_clipping(vol, clip_pct)
    
    # 4. Apply Dropped Slices
    drop_frac = recipe.get("drop_fraction", 0.0)
    vol, slice_indices = apply_dropped_slices(vol, slice_indices, drop_frac, rng)
    
    return vol, slice_indices


if __name__ == "__main__":
    print("Testing Defect Injection module...")
    
    # Create a small dummy volume of 50 slices
    test_vol = np.ones((50, 64, 64), dtype=np.float32) * 0.5
    
    test_recipe = {
        "drop_fraction": 0.10,
        "drift_severity": 0.30,
        "noise_sigma": 0.05,
        "clip_percentile": 5.0
    }
    
    corrupted, indices = inject_defects_into_volume(test_vol, test_recipe, seed=123)
    print(f"Original shape: {test_vol.shape} | New shape: {corrupted.shape}")
    print(f"Expected slices: 50 | Acquired slices: {len(indices)} | Missing: {50 - len(indices)}")
    print(f"Corrupted intensity range: min={corrupted.min():.3f}, max={corrupted.max():.3f}, mean={corrupted.mean():.4f}")
    print("Defect injection module verified successfully!")