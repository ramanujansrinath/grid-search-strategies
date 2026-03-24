"""
strategy_image_texture_neighbors.py
-------------------------------------
Strategy: IMAGE-GUIDED TEXTURE BUSYNESS + SPATIAL NEIGHBOR BOOST

Combines a local variance texture-busyness probability map with two spatial
Gaussian masks:

  1. Peak-biased first pick — the first box is drawn with probabilities
     weighted by a Gaussian blob centred on the highest-texture cell,
     multiplied element-wise by the base weights.

  2. Neighbor boost for subsequent picks — as boxes are selected, a running
     mask accumulates a Gaussian blob at each chosen cell's position.
     That smoothed mask is multiplied by the base weights so that cells
     neighbouring already-chosen cells are preferentially sampled next.

Algorithm
---------
Before the sequence loop:
  1. Load and resize the image; compute a per-pixel local variance map.
  2. Derive noise-free N×N base weights via grid_weights (values in [0.1, 0.9]).
     These are fixed for the entire run and capture the image structure.
  3. Pre-compute the first-pick weight vector (Gaussian centred on peak cell).

For each sequence:
  4. Initialise an N×N selected_mask of zeros.

  For each pick within a sequence:
  • Step 1  (peak-biased first pick):
      Sample using the pre-computed first-pick weights biased toward the
      most textured cell.

  • Steps 2 … seq_length  (texture × neighbor weighted):
      a. Gaussian-smooth the selected_mask (sigma = NEIGHBOR_SIGMA).
      b. Rectify: clip to ≥ 0.
      c. Element-wise multiply the smoothed mask by base_weights_2d.
      d. Add NOISE_LEVEL so no cell has exactly zero probability.
      e. Zero out already-selected cells, flatten, and sample.

  5. After each pick, record the chosen cell in selected_mask.

Texture method
--------------
Identical to strategy_image_texture:
  var(x,y) = E[I²](x,y) − E[I]²(x,y)
  computed via uniform_filter over a window of floor(P/10), min 3 (forced odd).
  Raw variance (not sqrt) is used so high-frequency texture is emphasised.

Tunable constants
-----------------
NEIGHBOR_SIGMA : float — Gaussian sigma for both the peak-bias and neighbor
                         boost masks (grid cells; default 1.0)
NOISE_LEVEL    : float — flat additive noise added after mask multiplication
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
NEIGHBOR_SIGMA : float = 1.0    # Gaussian sigma for peak-bias + neighbor masks
NOISE_LEVEL    : float = 0.005  # flat additive noise after mask multiplication
# ─────────────────────────────────────────────────────────────────────────────


def _compute_texture(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel local variance map on the luminance channel.

    Window size = floor(image_width / 10), minimum 3 (forced odd).

    Returns a (P, P) float64 array of local variance values.
    """
    lum = to_grayscale(img_rgb)                         # (P, P) in [0, 1]
    P   = lum.shape[0]
    K   = max(3, int(np.floor(P / 10)))
    if K % 2 == 0:
        K += 1

    mean_I  = uniform_filter(lum,      size=K, mode='reflect')
    mean_I2 = uniform_filter(lum ** 2, size=K, mode='reflect')
    variance = mean_I2 - mean_I ** 2
    return np.maximum(variance, 0.0)                    # local variance


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

    # ── Load image and compute texture metric (once) ──────────────────────
    img_rgb = load_image(img_index)
    img_rgb = resize_to_multiple(img_rgb, N)
    metric  = _compute_texture(img_rgb)                 # (P, P) float64

    # ── Noise-free base weights — fixed for all sequences ─────────────────
    no_noise_weights = grid_weights(metric, N)          # (N²,) in [0.1, 0.9]
    base_weights_2d  = no_noise_weights.reshape(N, N)   # (N, N)

    # ── Pre-compute first-pick mask (constant across all sequences) ────────
    peak_idx  = int(np.argmax(no_noise_weights))
    peak_mask = np.zeros((N, N), dtype=float)
    peak_mask[peak_idx // N, peak_idx % N] = 1.0
    smoothed_peak = gaussian_filter(peak_mask, sigma=NEIGHBOR_SIGMA)
    smoothed_peak = np.maximum(smoothed_peak, 0.0)
    first_pick_w  = (base_weights_2d * smoothed_peak + NOISE_LEVEL).flatten()

    boxes = all_boxes(N)
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        selected_mask = np.zeros((N, N), dtype=float)   # reset each sequence
        available     = list(boxes)
        seq: List[int] = []

        for step in range(seq_length):

            if step == 0:
                # ── First pick: Gaussian blob centred on peak-weight cell ─
                pool_w = [first_pick_w[b - 1] for b in available]
                chosen = random.choices(available, weights=pool_w, k=1)[0]

            else:
                # ── Subsequent picks: texture × neighbor boost ────────────

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
