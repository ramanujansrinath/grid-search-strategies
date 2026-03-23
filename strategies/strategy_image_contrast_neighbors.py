"""
strategy_image_contrast_neighbors.py
--------------------------------------
Strategy: IMAGE-GUIDED CONTRAST + SPATIAL NEIGHBOR BOOST

Combines an image-derived contrast probability map with a dynamic spatial
neighbor-boost mask that grows as boxes are selected.

Algorithm
---------
Before the sequence loop:
  1. Load and resize the image; compute a per-pixel local RMS contrast map.
  2. Derive noise-free N×N base weights via grid_weights (values in [0.1, 0.9]).
     These are fixed for the entire run and capture the image structure.

For each sequence:
  3. Initialise an N×N selected_mask of zeros (tracks which cells have been
     chosen so far in this sequence).

  For each pick within a sequence:
  • Step 1  (uniform random, no mask):
      The first box is drawn uniformly at random from all N² cells, in line
      with the universal first-pick rule shared across all strategies.

  • Steps 2 … seq_length  (contrast × neighbor weighted):
      a. Gaussian-smooth the selected_mask (sigma = NEIGHBOR_SIGMA).
         Each already-chosen cell contributes a Gaussian blob; blobs from
         multiple chosen cells accumulate, so the boost grows as the sequence
         progresses.
      b. Rectify: clip to ≥ 0  (numerical safety; Gaussian of non-negative
         input cannot go negative in practice).
      c. Element-wise multiply the smoothed mask by base_weights_2d.
         Cells neighbouring already-chosen cells inherit a larger multiplier;
         distant cells are multiplied by a near-zero value.
      d. Add a small flat noise (NOISE_LEVEL) to every cell so that no
         available cell ever has exactly zero probability.
      e. Zero out already-selected cells, flatten to 1-D, and draw the
         next box via weighted sampling.

  4. After each pick, record the chosen cell's (row, col) in selected_mask.

Tunable constants
-----------------
NEIGHBOR_SIGMA : float — Gaussian sigma (in grid cells) for the neighbor
                         boost mask.  sigma=1 gives e^−0.5 ≈ 0.6 at
                         immediate neighbours, e^−1 ≈ 0.4 at diagonal
                         neighbours, and decays smoothly beyond.
NOISE_LEVEL    : float — Small additive noise added after mask multiplication
                         to ensure all available cells retain a non-zero
                         selection probability.

Contrast method
---------------
Identical to strategy_image_contrast:
  rms_contrast(x,y) = local std dev of luminance within a square window of
                      side floor(image_width / 10), min 3.  Computed via two
                      uniform_filter passes on I and I².
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from image_utils import (
    load_image, resize_to_multiple, to_grayscale,
    grid_weights,
)
from grid_utils import all_boxes, get_row_col

# ── Exposed tuning constants ──────────────────────────────────────────────────
NEIGHBOR_SIGMA : float = 1.0    # Gaussian sigma for neighbor boost (grid cells)
NOISE_LEVEL    : float = 0.005  # flat additive noise after mask multiplication
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
    if K % 2 == 0:
        K += 1

    mean_I  = uniform_filter(lum,      size=K, mode='reflect')
    mean_I2 = uniform_filter(lum ** 2, size=K, mode='reflect')
    variance = mean_I2 - mean_I ** 2
    return np.sqrt(np.maximum(variance, 0.0))           # local std dev


def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
    img_index: int = 1,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
):
    """
    Parameters
    ----------
    grid_size         : side length of the square grid (N).
    seq_length        : number of boxes selected per sequence.
    n_seq             : number of sequences to generate.
    img_index         : 1-based index of the image to load (images/img_<n>.png).
    seed              : optional RNG seed for reproducibility.
    return_diagnostics: if True, return (sequences, diagnostics) instead of
                        sequences alone.  diagnostics is a dict with keys:
                          "img_rgb"          — (P,P,3) uint8 resized image
                          "metric"           — (P,P) float64 raw metric map
                          "no_noise_weights" — (N²,) float64 noise-free weights

    Returns
    -------
    List of n_seq sequences, each a list of seq_length distinct box numbers.
    If return_diagnostics is True, returns (sequences, diagnostics_dict).
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = grid_size

    # ── Load image and compute contrast metric (once) ─────────────────────
    img_rgb = load_image(img_index)
    img_rgb = resize_to_multiple(img_rgb, N)
    metric  = _compute_contrast(img_rgb)                # (P, P) float64

    # ── Noise-free base weights — fixed for all sequences ─────────────────
    # grid_weights returns (N²,) in row-major order; reshape to (N, N) so
    # we can do element-wise 2-D operations with the smoothed mask.
    no_noise_weights   = grid_weights(metric, N)        # (N²,) in [0.1, 0.9]
    base_weights_2d    = no_noise_weights.reshape(N, N) # (N, N)

    boxes = all_boxes(N)
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        selected_mask = np.zeros((N, N), dtype=float)   # reset each sequence
        available     = list(boxes)
        seq: List[int] = []

        for step in range(seq_length):

            if step == 0:
                # ── First pick: uniform random (universal rule) ───────────
                chosen = random.choice(available)

            else:
                # ── Subsequent picks: contrast × neighbor boost ───────────

                # (a) Smooth the mask of already-selected positions
                smoothed = gaussian_filter(selected_mask, sigma=NEIGHBOR_SIGMA)

                # (b) Rectify (numerical safety)
                smoothed = np.maximum(smoothed, 0.0)

                # (c) Multiply smoothed mask by fixed base weights (2-D)
                effective = base_weights_2d * smoothed  # (N, N)

                # (d) Add small flat noise so every cell keeps non-zero prob
                effective = effective + NOISE_LEVEL

                # (e) Flatten, zero out already-selected, then sample
                w = effective.flatten()                 # (N²,)
                for b in seq:
                    w[b - 1] = 0.0

                pool_w = [w[b - 1] for b in available]
                chosen = random.choices(available, weights=pool_w, k=1)[0]

            # Record chosen cell in the 2-D mask and update state
            row, col = get_row_col(chosen, N)
            selected_mask[row, col] = 1.0
            seq.append(chosen)
            available.remove(chosen)

        sequences.append(seq)

    if return_diagnostics:
        diagnostics = {
            "img_rgb"          : img_rgb,
            "metric"           : metric,
            "no_noise_weights" : no_noise_weights,
        }
        return sequences, diagnostics

    return sequences
