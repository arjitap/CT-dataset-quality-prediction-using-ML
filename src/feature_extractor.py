"""
feature_extractor.py
Extracts 25 volume-level and inter-slice quality statistics from a CT dataset.

"""

import numpy as np
from scipy import stats, ndimage
from skimage.metrics import structural_similarity as ssim


def compute_intensity_features(volume):
    """
    Computes global intensity distribution statistics across the 3D volume.
    (4 features)
    """
    flat = volume.ravel()
    vol_mean = float(np.mean(flat))
    vol_std = float(np.std(flat))
    vol_skew = float(stats.skew(flat))
    
    p5 = float(np.percentile(flat, 5))
    p95 = float(np.percentile(flat, 95))
    p95_p5_ratio = float((p95 + 1e-5) / (p5 + 1e-5))
    
    return {
        "vol_mean": vol_mean,
        "vol_std": vol_std,
        "vol_skewness": vol_skew,
        "intensity_p95_p5_ratio": p95_p5_ratio
    }


def compute_noise_features(volume):
    """
    Computes noise metrics via high-pass filtering, local variance, and Laplacian MAD.
    (4 features)
    """
    depth = volume.shape[0]
    sample_indices = np.linspace(int(depth * 0.15), int(depth * 0.85), min(10, depth), dtype=int)
    
    mad_list = []
    lap_var_list = []
    local_var_list = []
    hf_energy_list = []
    
    for idx in sample_indices:
        slice_2d = volume[idx]
        
        # 1. 2D Laplacian filter for high-frequency edge/noise response
        lap = ndimage.laplace(slice_2d)
        lap_var = float(np.var(lap))
        lap_var_list.append(lap_var)
        
        # 2. Median Absolute Deviation (MAD) of the Laplacian: robust sigma estimate
        mad = float(np.median(np.abs(lap - np.median(lap)))) / 0.6745
        mad_list.append(mad)
        
        # 3. Local background variance (7x7 uniform filter)
        mean_sq = ndimage.uniform_filter(slice_2d ** 2, size=7)
        sq_mean = ndimage.uniform_filter(slice_2d, size=7) ** 2
        local_var = np.maximum(0.0, mean_sq - sq_mean)
        bg_mask = slice_2d < np.percentile(slice_2d, 25)
        if np.any(bg_mask):
            local_var_list.append(float(np.mean(local_var[bg_mask])))
            
        # 4. High-frequency Fourier energy ratio
        f_transform = np.fft.fft2(slice_2d)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift) ** 2
        total_power = np.sum(power_spectrum) + 1e-8
        
        h, w = slice_2d.shape
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        r = min(h, w) * 0.15
        high_pass_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) > (r ** 2)
        hf_power = np.sum(power_spectrum[high_pass_mask])
        hf_energy_list.append(float(hf_power / total_power))
        
    return {
        "noise_mad_estimate": float(np.mean(mad_list)),
        "noise_local_variance_mean": float(np.mean(local_var_list)) if local_var_list else 0.0,
        "laplacian_variance": float(np.mean(lap_var_list)),
        "high_freq_energy_ratio": float(np.mean(hf_energy_list))
    }


def compute_drift_features(volume, slice_indices):
    """
    Computes inter-slice mean drift, linear trend slope, and consistency metrics along Z.
    (5 features)
    """
    slice_means = np.mean(volume, axis=(1, 2))
    n_slices = len(slice_means)
    
    if n_slices < 3:
        return {
            "drift_slope": 0.0, "drift_r2": 0.0,
            "slice_mean_diff_mean": 0.0, "slice_mean_diff_max": 0.0,
            "slice_mean_cov": 0.0
        }
        
    res = stats.linregress(slice_indices, slice_means)
    slope = float(res.slope)
    r2 = float(res.rvalue ** 2)
    
    diffs = np.abs(np.diff(slice_means))
    mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    
    mean_of_means = np.mean(slice_means)
    cov = float(np.std(slice_means) / (mean_of_means + 1e-5))
    
    return {
        "drift_slope": slope,
        "drift_r2": r2,
        "slice_mean_diff_mean": mean_diff,
        "slice_mean_diff_max": max_diff,
        "slice_mean_cov": cov
    }


def compute_ssim_features(volume):
    """
    Computes adjacent-slice structural similarity (SSIM) sequence statistics.
    (4 features)
    """
    depth = volume.shape[0]
    if depth < 2:
        return {
            "adjacent_ssim_mean": 0.0, "adjacent_ssim_std": 0.0,
            "adjacent_ssim_min": 0.0, "adjacent_ssim_p10": 0.0
        }
        
    step = max(1, (depth - 1) // 15)
    sample_pairs = range(0, depth - 1, step)
    
    ssim_values = []
    data_range = float(np.percentile(volume, 99.5) - np.min(volume))
    if data_range <= 0.01:
        data_range = 1.0
        
    for i in sample_pairs:
        s1 = volume[i]
        s2 = volume[i + 1]
        score = ssim(s1, s2, data_range=data_range)
        ssim_values.append(float(score))
        
    return {
        "adjacent_ssim_mean": float(np.mean(ssim_values)),
        "adjacent_ssim_std": float(np.std(ssim_values)),
        "adjacent_ssim_min": float(np.min(ssim_values)),
        "adjacent_ssim_p10": float(np.percentile(ssim_values, 10))
    }


def compute_entropy_features(volume):
    """
    Computes Shannon entropy from normalized voxel histogram distributions.
    (3 features)
    """
    flat = volume.ravel()
    hist, _ = np.histogram(flat, bins=256, density=True)
    hist = hist[hist > 0]
    vol_entropy = float(-np.sum(hist * np.log2(hist)))
    
    depth = volume.shape[0]
    sample_slices = np.linspace(int(depth * 0.1), int(depth * 0.9), min(8, depth), dtype=int)
    slice_entropies = []
    
    for idx in sample_slices:
        s_hist, _ = np.histogram(volume[idx], bins=128, density=True)
        s_hist = s_hist[s_hist > 0]
        s_ent = float(-np.sum(s_hist * np.log2(s_hist)))
        slice_entropies.append(s_ent)
        
    return {
        "slice_entropy_mean": float(np.mean(slice_entropies)),
        "slice_entropy_std": float(np.std(slice_entropies)),
        "vol_entropy": vol_entropy
    }


def compute_clipping_features(volume):
    """
    Computes fraction of clipped/saturated voxels and effective dynamic range.
    (3 features)
    """
    total_voxels = volume.size
    min_val = float(np.min(volume))
    max_val = float(np.max(volume))
    
    tol = (max_val - min_val) * 0.005 + 1e-6
    low_clipped = np.sum(volume <= (min_val + tol)) / total_voxels
    high_clipped = np.sum(volume >= (max_val - tol)) / total_voxels
    
    p1 = float(np.percentile(volume, 1))
    p99 = float(np.percentile(volume, 99))
    eff_range = float(p99 - p1)
    
    return {
        "clipped_low_fraction": float(low_clipped),
        "clipped_high_fraction": float(high_clipped),
        "effective_dynamic_range": eff_range
    }


def compute_completeness_features(slice_indices):
    """
    Derives missing-slice metrics directly from physical slice_indices.
    (2 features)
    """
    if len(slice_indices) == 0:
        return {"missing_slice_count": 0, "missing_slice_fraction": 0.0}
        
    expected_span = int(np.max(slice_indices) - np.min(slice_indices) + 1)
    acquired_count = len(slice_indices)
    missing_count = max(0, expected_span - acquired_count)
    missing_frac = float(missing_count / max(1, expected_span))
    
    return {
        "missing_slice_count": int(missing_count),
        "missing_slice_fraction": missing_frac
    }


def extract_all_features(volume, slice_indices):
    """
    Master extractor: assembles the complete 25-dimensional feature dictionary.
    """
    features = {}
    features.update(compute_intensity_features(volume))
    features.update(compute_noise_features(volume))
    features.update(compute_drift_features(volume, slice_indices))
    features.update(compute_ssim_features(volume))
    features.update(compute_entropy_features(volume))
    features.update(compute_clipping_features(volume))
    features.update(compute_completeness_features(slice_indices))
    return features


if __name__ == "__main__":
    print("Testing Feature Extraction Pipeline...")
    dummy_vol = np.random.uniform(0.1, 0.9, size=(40, 64, 64)).astype(np.float32)
    dummy_indices = np.array([i for i in range(40) if i not in [5, 12, 22, 33]], dtype=np.int32)
    dummy_vol = dummy_vol[:len(dummy_indices)]
    
    feats = extract_all_features(dummy_vol, dummy_indices)
    print(f"Total features extracted: {len(feats)} (Expected: 25)")
    assert len(feats) == 25, f"Error: Expected 25 features, got {len(feats)}"
    print("Feature Extraction pipeline verified successfully!")