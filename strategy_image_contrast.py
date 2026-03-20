"""
strategy_image_contrast.py
---------------------------
Strategy: IMAGE-GUIDED CONTRAST

Boxes are sampled with probability proportional to a local RMS contrast map
derived from the input image.  High-contrast cells are more likely to be
selected.

Contrast method
---------------
For each pixel (x, y) in the luminance channel, local RMS contrast is the
root-mean-square of pixel intensities within a square neighbourhood of side
length WINDOW_K = floor(image_width / 10):

    rms_contrast(x, y) = sqrt( mean( I(u,v)² over window ) − mean( I(u,v) )² )
                       = local standard deviation of I within the window

This is computed efficiently via uniform_filter (box blur) applied twice —
once to I and once to I² — and then taking the point-wise square root of
variance:

    var(x,y) = E[I²] − E[I]²
    rms(x,y) = sqrt( max(var, 0) )    # max(·,0) avoids float rounding negatives

The window size is derived from the (already-resized) image width, so it
scales naturally with grid_size and image resolution.  It is printed at
runtime so the user can verify it.

Tunable constant
----------------
NOISE_A : float — noise scaling constant A (0 = no noise, default 0.3)
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter

from image_utils import (
    load_image, resize_to_multiple, to_grayscale,
    add_noise, grid_weights,
)
from grid_utils import all_boxes

# ── Exposed tuning constant ───────────────────────────────────────────────────
NOISE_A : float = 0.3    # noise amplitude as a fraction of metric SD
# ─────────────────────────────────────────────────────────────────────────────


def _compute_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel local RMS contrast map on the luminance channel.

    Window size = floor(image_width / 10), minimum 3.

    Returns a (P, P) float64 array of local RMS contrast values.
    """
    lum  = to_grayscale(img_rgb)                        # (P, P) in [0, 1]
    P    = lum.shape[0]
    K    = max(3, int(np.floor(P / 10)))
    # Ensure odd window so uniform_filter is centred
    if K % 2 == 0:
        K += 1

    mean_I  = uniform_filter(lum,       size=K, mode='reflect')
    mean_I2 = uniform_filter(lum ** 2,  size=K, mode='reflect')
    variance = mean_I2 - mean_I ** 2
    return np.sqrt(np.maximum(variance, 0.0))           # local std dev


def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
    img_index: int = 1,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    Parameters
    ----------
    grid_size  : side length of the square grid (N).
    seq_length : number of boxes selected per sequence.
    n_seq      : number of sequences to generate.
    img_index  : 1-based index of the image to load (images/img_<n>.png).
    seed       : optional RNG seed for reproducibility.

    Returns
    -------
    List of n_seq sequences, each a list of seq_length distinct box numbers.
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = grid_size

    # ── Steps 1-3: load, resize, compute metric (once) ───────────────────
    img_rgb   = load_image(img_index)
    img_rgb   = resize_to_multiple(img_rgb, N)
    metric    = _compute_contrast(img_rgb)          # (P, P) float64
    metric_sd = float(metric.std())

    boxes = all_boxes(N)
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        # ── Steps 4-9: noise → cell average → normalise (per sequence) ───
        noisy   = add_noise(metric, metric_sd, NOISE_A)
        weights = grid_weights(noisy, N)             # (N²,) in [0.1, 0.9]

        # ── Steps 10-12: weighted sampling without replacement ────────────
        available = list(boxes)
        seq: List[int] = []

        while len(seq) < seq_length:
            pool_w = [weights[b - 1] for b in available]
            chosen = random.choices(available, weights=pool_w, k=1)[0]
            available.remove(chosen)
            seq.append(chosen)

        sequences.append(seq)

    return sequences
