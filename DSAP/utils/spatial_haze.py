# Modified: variable names aligned with the method diagram.
import os
import random
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from utils.data_utils import add_Gaussian_noise, add_JPEG_noise, single2uint


def generate_spatial_field(rows=256, cols=256, multiple_iter=None, sigma=None, seed=None):
    """
    Generate spatially varying haze density map beta_tilde in [0, 1].
    This follows the Markov Chain Brownian Motion idea in HazeFlow.
    """
    if seed is not None:
        np.random.seed(seed)

    if multiple_iter is None:
        multiple_iter = random.choice([5, 6, 7])

    if sigma is None:
        sigma = random.choice([15, 25, 35])

    z = np.zeros((rows, cols), dtype=np.float32)

    current_row = np.random.randint(0, rows)
    current_col = np.random.randint(0, cols)
    z[current_row, current_col] = 1.0

    num_points = rows * cols * multiple_iter
    increments = np.random.normal(loc=0.0, scale=1.0, size=(num_points, 2))

    for i in range(num_points):
        r = np.random.rand()

        if r < 0.25 and current_row > 0:
            current_row -= 1
        elif r < 0.50 and current_row < rows - 1:
            current_row += 1
        elif r < 0.75 and current_col > 0:
            current_col -= 1
        elif current_col < cols - 1:
            current_col += 1

        z[current_row, current_col] += 1.0

        current_row = (current_row + int(increments[i, 0])) % rows
        current_col = (current_col + int(increments[i, 1])) % cols
        z[current_row, current_col] += 1.0

    z_smoothed = gaussian_filter(z, sigma=sigma)
    z_int = sigma * z_smoothed + z

    z_min, z_max = z_int.min(), z_int.max()
    if z_max - z_min < 1e-8:
        return np.zeros_like(z_int, dtype=np.float32)

    z_norm = (z_int - z_min) / (z_max - z_min)
    return z_norm.astype(np.float32)


def generate_spatial_field_bank(save_dir, num_maps=1000, size=256):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_maps):
        out_path = os.path.join(save_dir, f"{i:06d}.png")
        if os.path.exists(out_path):
            continue

        m = generate_spatial_field(rows=size, cols=size, seed=i)
        Image.fromarray((m * 255.0).astype(np.uint8)).save(out_path)

    print(f"MCBM maps saved to: {save_dir}")


def synthesize_spatial_haze(
    J: np.ndarray,
    d: np.ndarray,
    beta_tilde: np.ndarray = None,
    beta_range=(0.2, 1.3),
    alpha_range=(0.3, 0.9),
    A_range=(0.70, 1.05),
    gamma_prob=0.5,
    noise_prob=0.5,
    jpeg_prob=0.5,
):
    """
    J: BGR float image in [0, 1]
    d: depth map, H x W
    beta_tilde: MCBM density map in [0, 1], optional

    Return:
        hazy RGB uint8 image, same style as original synthesize()
    """
    h, w = J.shape[:2]

    depth = d.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Effective depth follows the inverted normalized depth convention.
    d_eff = 1.0 - depth

    if beta_tilde is None:
        beta_tilde = generate_spatial_field(rows=256, cols=256)

    if beta_tilde.ndim == 3:
        beta_tilde = beta_tilde[:, :, 0]

    beta_tilde = beta_tilde.astype(np.float32)
    if beta_tilde.max() > 1.0:
        beta_tilde = beta_tilde / 255.0

    beta_tilde = cv2.resize(beta_tilde, (w, h), interpolation=cv2.INTER_CUBIC)
    beta_tilde = np.clip(beta_tilde, 0.0, 1.0)

    beta = np.random.uniform(beta_range[0], beta_range[1])
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    # beta_s(x) = beta + alpha * beta_tilde(x).
    # t_s(x) = exp(-2 * d_eff(x) * beta_s(x)).
    beta_s = beta + alpha * beta_tilde
    t_s = np.exp(-d_eff * 2.0 * beta_s)
    t_s = t_s[:, :, np.newaxis].astype(np.float32)

    A_base = np.random.uniform(A_range[0], A_range[1])
    A_color = np.random.uniform(-0.05, 0.05, size=(3,))
    A_rgb = np.clip(A_base + A_color, 0.25, 1.2).astype(np.float32)

    I_s = J.copy().astype(np.float32)

    # Real-world degradation D in HazeFlow: gamma / noise / JPEG.
    if np.random.rand() < gamma_prob:
        gamma = np.random.uniform(1.2, 2.4)
        I_s = np.power(np.clip(I_s, 0.0, 1.0), gamma)

    if np.random.rand() < noise_prob:
        I_s = add_Gaussian_noise(I_s)

    I_s = I_s * t_s + A_rgb.reshape(1, 1, 3) * (1.0 - t_s)
    I_s = np.clip(I_s, 0.0, 1.0)

    if np.random.rand() < jpeg_prob:
        I_s = add_JPEG_noise(I_s)

    if J.shape[-1] > 3:
        I_s = I_s[:, :, :3]

    # Original synthesize() returns RGB uint8.
    I_s = single2uint(I_s)
    I_s = cv2.cvtColor(I_s, cv2.COLOR_BGR2RGB)
    return I_s
