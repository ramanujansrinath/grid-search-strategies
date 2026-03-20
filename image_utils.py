"""
image_utils.py
--------------
Shared utilities for image-based selection strategies.

Used by:
    strategy_image_salience.py
    strategy_image_contrast.py
    strategy_image_color_concentration.py
    strategy_image_texture.py

Flow summary
------------
These functions handle steps 1-3 (before the sequence loop) and
steps 4-9 (inside the sequence loop).  Each strategy file calls:

    Before the loop:
        img_rgb  = load_image(img_index)
        metric   = compute_<metric>(img_rgb)   # defined in strategy file
        metric_sd = metric.std()

    Inside the loop (once per sequence):
        noisy    = add_noise(metric, metric_sd, A)
        weights  = grid_weights(noisy, grid_size)   # steps 6-9
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# ── Image directory convention ────────────────────────────────────────────────
# Images live in  <this file's directory>/images/img_<index>.png
_IMAGES_DIR = Path(__file__).parent / "images"


def load_image(img_index: int) -> np.ndarray:
    """
    Load images/<img_index>.png as an (H, W, 3) uint8 RGB array.

    Parameters
    ----------
    img_index : 1-based image index (img_1.png, img_2.png, …)

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, RGB channel order
    """
    path = _IMAGES_DIR / f"img_{img_index}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            f"Expected images in {_IMAGES_DIR} named img_1.png, img_2.png, …"
        )
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize_to_multiple(img_rgb: np.ndarray, N: int) -> np.ndarray:
    """
    Resize a square RGB image so its side length is the nearest multiple of N.

    Uses bilinear interpolation (BILINEAR) for speed.

    Parameters
    ----------
    img_rgb : (H, W, 3) uint8 array — input image (assumed square, H == W)
    N       : grid side length

    Returns
    -------
    np.ndarray  shape (P, P, 3) where P = round(H / N) * N
    """
    H = img_rgb.shape[0]
    P = round(H / N) * N
    if P == 0:
        P = N   # safety: ensure at least one cell
    if P == H:
        return img_rgb
    pil = Image.fromarray(img_rgb).resize((P, P), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) uint8 RGB to (H, W) float64 luminance in [0, 1].
    Uses the standard BT.601 coefficients.
    """
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def to_float(img_rgb: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) float64 array with values in [0, 1]."""
    return img_rgb.astype(np.float64) / 255.0


def add_noise(metric: np.ndarray, metric_sd: float, A: float) -> np.ndarray:
    """
    Add symmetric uniform noise to the PxP pixel-level metric map.

    Noise is drawn from Uniform(−SD×A, +SD×A) independently per pixel
    and per sequence call.

    Parameters
    ----------
    metric    : (P, P) float64 pixel-level metric array
    metric_sd : standard deviation of *metric* across all pixels (precomputed)
    A         : noise scaling constant (0 = no noise)

    Returns
    -------
    np.ndarray  same shape as metric, with noise added
    """
    if A == 0.0 or metric_sd == 0.0:
        return metric.copy()
    amplitude = metric_sd * A
    noise = np.random.uniform(-amplitude, amplitude, size=metric.shape)
    return metric + noise


def grid_weights(noisy_metric: np.ndarray, N: int) -> np.ndarray:
    """
    Steps 6-9: divide a PxP noisy metric map into an N×N grid,
    average each cell, then normalise to [0.1, 0.9].

    Parameters
    ----------
    noisy_metric : (P, P) float64 array, P divisible by N
    N            : grid side length

    Returns
    -------
    weights : (N*N,) float64 array indexed by 0-based box number.
              weights[k] is the sampling weight for box k+1.
    """
    P = noisy_metric.shape[0]
    cell = P // N

    # Step 6-7: reshape into (N, N, cell, cell) and average over cell dims
    grid = (noisy_metric
            .reshape(N, cell, N, cell)
            .transpose(0, 2, 1, 3)       # (N, N, cell, cell)
            .reshape(N, N, -1)
            .mean(axis=-1))              # (N, N)

    # Step 8: normalise to [0.1, 0.9]
    a_min = grid.min()
    a_max = grid.max()
    if a_max == a_min:
        # Flat metric — assign uniform weight
        normalised = np.full_like(grid, 0.5)
    else:
        normalised = 0.1 + 0.8 * (grid - a_min) / (a_max - a_min)

    # Step 9: flatten row-wise to match 1-indexed box labelling
    return normalised.flatten()          # shape (N*N,)
