"""
strategy_image_salience_neighbors.py
--------------------------------------
Strategy: IMAGE-GUIDED SALIENCE + SPATIAL NEIGHBOR BOOST

Combines a DoG saliency probability map with two spatial Gaussian masks:

  1. Peak-biased first pick — the first box is drawn with probabilities
     weighted by a Gaussian blob centred on the highest-saliency cell,
     multiplied element-wise by the base weights.  This concentrates the
     first pick on the most salient region.

  2. Neighbor boost for subsequent picks — as boxes are selected, a running
     mask accumulates a Gaussian blob at each chosen cell's position.
     That smoothed mask is multiplied by the base weights so that cells
     neighbouring already-chosen cells are preferentially sampled next.

Algorithm
---------
Before the sequence loop:
  1. Load and resize the image; compute a DoG saliency map (see below).
  2. Derive noise-free N×N base weights via grid_weights (values in [0.1, 0.9]).
     These are fixed for the entire run and capture the image structure.
  3. Identify the peak cell: argmax of no_noise_weights.

For each sequence:
  4. Initialise an N×N selected_mask of zeros.

  For each pick within a sequence:
  • Step 1  (peak-biased first pick):
      A Gaussian blob is centred on the cell with the highest noise-free
      weight (argmax of no_noise_weights) and multiplied element-wise by
      the base weights.  This biases the first pick strongly toward the
      most salient region while keeping all cells reachable.

  • Steps 2 … seq_length  (saliency × neighbor weighted):
      a. Gaussian-smooth the selected_mask (sigma = NEIGHBOR_SIGMA).
         Blobs from all chosen cells accumulate, so the spatial pull grows
         as the sequence progresses.
      b. Rectify: clip to ≥ 0 (numerical safety).
      c. Element-wise multiply the smoothed mask by base_weights_2d.
      d. Add a small flat noise (NOISE_LEVEL) so no available cell ever
         has exactly zero probability.
      e. Zero out already-selected cells, flatten, and sample.

  5. After each pick, record the chosen cell's (row, col) in selected_mask.

Saliency method
---------------
Difference of Gaussians (DoG) on the luminance channel:

    saliency(x,y) = G(x,y; σ_c) − G(x,y; σ_s)

where σ_s = SURROUND_RATIO × σ_c.  The map captures pixels brighter or
darker than their surround; negative values are preserved.  grid_weights
maps the full range to [0.1, 0.9], so both bright-on-dark and dark-on-bright
regions are treated as salient.

Tunable constants
-----------------
SIGMA_CENTER   : float — σ of the centre Gaussian (pixels; default 2.0)
SURROUND_RATIO : float — σ_s / σ_c (default 4.0, giving σ_s = 8.0)
NEIGHBOR_SIGMA : float — Gaussian sigma for both the peak-bias and neighbor
                         boost masks (grid cells; default 1.0)
NOISE_LEVEL    : float — flat additive noise added after mask multiplication
                         to ensure all cells retain non-zero probability
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from image_utils import (
    load_image, resize_to_multiple, to_grayscale,
    grid_weights,
)
from grid_utils import all_boxes, get_row_col

# ── Exposed tuning constants ──────────────────────────────────────────────────
SIGMA_CENTER   : float = 2.0    # centre Gaussian σ (pixels)
SURROUND_RATIO : float = 4.0    # σ_surround = SURROUND_RATIO × SIGMA_CENTER
NEIGHBOR_SIGMA : float = 1.0    # Gaussian sigma for peak-bias + neighbor masks
NOISE_LEVEL    : float = 0.005  # flat additive noise after mask multiplication
# ─────────────────────────────────────────────────────────────────────────────


def _compute_saliency(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel saliency map via DoG on the luminance channel.

    Returns a (P, P) float64 array of raw (unnormalised) saliency values.
    """
    lum      = to_grayscale(img_rgb)                    # (P, P) in [0, 1]
    sigma_c  = SIGMA_CENTER
    sigma_s  = SIGMA_CENTER * SURROUND_RATIO
    centre   = gaussian_filter(lum, sigma=sigma_c)
    surround = gaussian_filter(lum, sigma=sigma_s)
    return centre - surround                            # DoG saliency map


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

    # ── Load image and compute saliency metric (once) ─────────────────────
    img_rgb = load_image(img_index)
    img_rgb = resize_to_multiple(img_rgb, N)
    metric  = _compute_saliency(img_rgb)                # (P, P) float64

    # ── Noise-free base weights — fixed for all sequences ─────────────────
    # grid_weights normalises to [0.1, 0.9], absorbing any negative values
    # in the raw DoG map.  Reshape to (N, N) for 2-D mask operations.
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
                # ── Subsequent picks: saliency × neighbor boost ───────────

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
