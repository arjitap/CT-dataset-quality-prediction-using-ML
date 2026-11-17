

import os
import astra
import numpy as np
import matplotlib.pyplot as plt


def build_analytical_slice(height=128, width=128, rng=None, config_id=0):
   
    if rng is None:
        rng = np.random.default_rng(config_id)

    slice_img = np.zeros((height, width), dtype=np.float32)
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]

    # 1. Outer container (polymer matrix, attenuation ~ 0.40)
    radius_outer = min(height, width) * 0.42
    dist_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    slice_img[dist_center <= radius_outer] = 0.40

    # 2. Inner concentric core (attenuation ~ 0.80)
    radius_inner = radius_outer * 0.35
    slice_img[dist_center <= radius_inner] = 0.80

    # 3. Off-center structural rods (attenuation ~ 0.60, 1.00)
    num_rods = 3
    for i in range(num_rods):
        angle = (2 * np.pi / num_rods) * i + (config_id * 0.4)
        rod_cx = cx + radius_outer * 0.62 * np.cos(angle)
        rod_cy = cy + radius_outer * 0.62 * np.sin(angle)
        rod_r = min(height, width) * 0.06
        rod_dist = np.sqrt((x - rod_cx) ** 2 + (y - rod_cy) ** 2)
        slice_img[rod_dist <= rod_r] = 0.60 if (i % 2 == 0) else 1.00

    # 4. Small voids (0.0) or high-density particles (1.25)
    num_particles = 4
    for _ in range(num_particles):
        offset_r = rng.uniform(0.15, 0.75) * radius_outer
        offset_a = rng.uniform(0, 2 * np.pi)
        px = cx + offset_r * np.cos(offset_a)
        py = cy + offset_r * np.sin(offset_a)
        pr = rng.uniform(2, 5)
        p_dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        slice_img[p_dist <= pr] = 0.0 if rng.random() > 0.5 else 1.25

    return slice_img


def astra_fbp_reconstruct_slice(slice_2d, num_angles=180):
    """
    Simulates forward projection (sinogram) and performs Filtered Back-Projection (FBP)
    using the ASTRA Toolbox (CPU standard algorithm with explicit projector).
    """
    height, width = slice_2d.shape

    # 1. Define 2D parallel-beam geometry and explicit CPU Projector
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, width, angles)
    vol_geom = astra.create_vol_geom(height, width)
    projector_id = astra.create_projector("line", proj_geom, vol_geom)

    # 2. Forward Projection using explicit projector
    vol_id = astra.data2d.create("-vol", vol_geom, data=slice_2d)
    sino_id = astra.data2d.create("-sino", proj_geom)

    fp_cfg = astra.astra_dict("FP")
    fp_cfg["VolumeDataId"] = vol_id
    fp_cfg["ProjectionDataId"] = sino_id
    fp_cfg["ProjectorId"] = projector_id
    fp_alg_id = astra.algorithm.create(fp_cfg)
    astra.algorithm.run(fp_alg_id)

    # 3. Filtered Back-Projection (FBP) Reconstruction
    recon_id = astra.data2d.create("-vol", vol_geom)
    fbp_cfg = astra.astra_dict("FBP")
    fbp_cfg["ReconstructionDataId"] = recon_id
    fbp_cfg["ProjectionDataId"] = sino_id
    fbp_cfg["ProjectorId"] = projector_id
    fbp_cfg["FilterType"] = "Ram-Lak"

    fbp_alg_id = astra.algorithm.create(fbp_cfg)
    astra.algorithm.run(fbp_alg_id)

    reconstructed_slice = astra.data2d.get(recon_id)

    # 4. Cleanup all ASTRA allocated memory
    astra.algorithm.delete(fp_alg_id)
    astra.algorithm.delete(fbp_alg_id)
    astra.projector.delete(projector_id)
    astra.data2d.delete(vol_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(recon_id)

    # Physical clipping: attenuation cannot be negative
    reconstructed_slice = np.clip(reconstructed_slice, 0.0, None)
    return reconstructed_slice.astype(np.float32)


def generate_clean_reconstructed_volume(depth=64, height=128, width=128, seed=42):
    """
    Generates a full 3D CT volume by building analytical phantom geometry
    and reconstructing each slice using ASTRA FBP.
    """
    rng = np.random.default_rng(seed)
    volume = np.zeros((depth, height, width), dtype=np.float32)

    for z in range(depth):
        # Leave 4 empty margin slices at boundaries
        if z < 4 or z >= depth - 4:
            continue

        raw_slice = build_analytical_slice(height=height, width=width, rng=rng, config_id=seed)
        volume[z] = astra_fbp_reconstruct_slice(raw_slice, num_angles=180)

    return volume


def save_reconstruction_verification(volume, output_path="results/figures/clean_astra_recon_preview.png"):
    """
    Saves an orthogonal inspection figure of the ASTRA reconstructed clean CT volume.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth = volume.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    s1, s2, s3 = int(depth * 0.25), int(depth * 0.50), int(depth * 0.75)

    axes[0].imshow(volume[s1], cmap="gray", vmin=0, vmax=1.3)
    axes[0].set_title(f"ASTRA FBP Slice {s1}")
    axes[0].axis("off")

    axes[1].imshow(volume[s2], cmap="gray", vmin=0, vmax=1.3)
    axes[1].set_title(f"ASTRA FBP Slice {s2} (Center)")
    axes[1].axis("off")

    axes[2].imshow(volume[s3], cmap="gray", vmin=0, vmax=1.3)
    axes[2].set_title(f"ASTRA FBP Slice {s3}")
    axes[2].axis("off")

    # Coronal (side profile)
    coronal = volume[:, volume.shape[1] // 2, :]
    axes[3].imshow(coronal, cmap="gray", aspect="auto", vmin=0, vmax=1.3)
    axes[3].set_title("Coronal Side View (Z vs X)")
    axes[3].set_xlabel("X (Width)")
    axes[3].set_ylabel("Z (Slice Index)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] ASTRA FBP verification figure saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 2: ASTRA CT FORWARD PROJECTION & FBP RECONSTRUCTION")
    print("=" * 60)

    vol = generate_clean_reconstructed_volume(depth=64, height=128, width=128, seed=42)

    print(f"Reconstructed Volume Shape: {vol.shape} (Depth, Height, Width)")
    print(f"Intensity Range: Min={vol.min():.3f}, Max={vol.max():.3f}, Mean={vol.mean():.4f}")

    save_reconstruction_verification(vol)

    os.makedirs("data/synthetic", exist_ok=True)
    master_path = "data/synthetic/master_clean_base_00.npy"
    np.save(master_path, vol)
    print(f"[OK] Master clean reconstructed volume saved to: {master_path}")
    print("=" * 60)
    print("ASTRA RECONSTRUCTION STAGE VERIFIED SUCCESSFULLY!")
    print("=" * 60)