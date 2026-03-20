"""
strategy_image_salience.py
---------------------------
Strategy: IMAGE-GUIDED SALIENCE

Boxes are sampled with probability proportional to a saliency map derived
from the input image.  Higher-saliency cells are more likely to be selected.

Saliency method
---------------
A lightweight centre-surround saliency map based on a Difference of Gaussians
(DoG) applied to the luminance channel:

    saliency(x,y) = G(x,y; σ_c) − G(x,y; σ_s)

where σ_c (centre) and σ_s (surround) are chosen to model the classical
centre-surround receptive field structure (σ_s = SURROUND_RATIO × σ_c).

The resulting map captures pixels that are brighter or darker than their
immediate surround.  Negative values (darker-than-surround) and positive
values (brighter-than-surround) are both kept; the normalisation step maps
the full range to [0.1, 0.9], so both bright-on-dark and dark-on-bright
regions are treated as salient.

# TODO: Replace this DoG approximation with a full Itti-Koch saliency
# implementation that includes orientation-selective Gabor channels,
# colour opponency (R-G, B-Y), and the proper iterative normalisation
# operator N(·) described in Itti, Koch & Niebur (1998).

Tunable constants
-----------------
SIGMA_CENTER    : float  — σ of the centre Gaussian  (pixels; default 2.0)
SURROUND_RATIO  : float  — σ_s / σ_c  (default 4.0, giving σ_s = 8.0)
NOISE_A         : float  — noise scaling constant A (0 = no noise, default 0.3)
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from image_utils import (
    load_image, resize_to_multiple, to_grayscale,
    add_noise, grid_weights,
)
from grid_utils import all_boxes

# ── Exposed tuning constants ──────────────────────────────────────────────────
SIGMA_CENTER   : float = 2.0    # centre Gaussian σ (pixels)
SURROUND_RATIO : float = 4.0    # σ_surround = SURROUND_RATIO × SIGMA_CENTER
NOISE_A        : float = 0.3    # noise amplitude as a fraction of metric SD
# ─────────────────────────────────────────────────────────────────────────────


def _compute_saliency(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel saliency map via DoG on the luminance channel.

    Returns a (P, P) float64 array of raw (unnormalised) saliency values.
    """
    lum = to_grayscale(img_rgb)                         # (P, P) in [0, 1]
    sigma_c = SIGMA_CENTER
    sigma_s = SIGMA_CENTER * SURROUND_RATIO
    centre  = gaussian_filter(lum, sigma=sigma_c)
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

    # ── Steps 1-3: load, resize, compute metric (once) ───────────────────
    img_rgb   = load_image(img_index)
    img_rgb   = resize_to_multiple(img_rgb, N)
    metric    = _compute_saliency(img_rgb)          # (P, P) float64
    metric_sd = float(metric.std())

    # ── Noise-free weights (computed once, before the sequence loop) ──────
    no_noise_weights = grid_weights(metric, N)      # (N²,) in [0.1, 0.9]

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

    if return_diagnostics:
        diagnostics = {
            "img_rgb"          : img_rgb,
            "metric"           : metric,
            "no_noise_weights" : no_noise_weights,
        }
        return sequences, diagnostics

    return sequences
