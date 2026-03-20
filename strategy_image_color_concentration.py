"""
strategy_image_color_concentration.py
--------------------------------------
Strategy: IMAGE-GUIDED COLOUR CONCENTRATION

Boxes are sampled with probability proportional to a per-pixel saturation
map derived from the input image.  More colourful (saturated) cells are
more likely to be selected.

Colour metric
-------------
Saturation is computed per pixel from the RGB values using the HSV
definition:

    S(x,y) = ( max(R,G,B) − min(R,G,B) ) / max(R,G,B)
           = 0   when max(R,G,B) = 0  (pure black)

All channels are used in RGB space; no channel is discarded.  The result
is a (P, P) float64 array with values in [0, 1].

Perfectly achromatic pixels (R=G=B) have S=0.  After normalisation these
map to the floor weight of 0.1, so they remain selectable but are the
least likely cells.

Tunable constant
----------------
NOISE_A : float — noise scaling constant A (0 = no noise, default 0.3)
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from image_utils import (
    load_image, resize_to_multiple, to_float,
    add_noise, grid_weights,
)
from grid_utils import all_boxes

# ── Exposed tuning constant ───────────────────────────────────────────────────
NOISE_A : float = 0.3    # noise amplitude as a fraction of metric SD
# ─────────────────────────────────────────────────────────────────────────────


def _compute_saturation(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel HSV saturation from an RGB image.

    Parameters
    ----------
    img_rgb : (P, P, 3) uint8 array

    Returns
    -------
    (P, P) float64 array of saturation values in [0, 1]
    """
    f    = to_float(img_rgb)                            # (P, P, 3) in [0,1]
    Cmax = f.max(axis=-1)                               # (P, P)
    Cmin = f.min(axis=-1)
    delta = Cmax - Cmin

    # S = delta / Cmax; undefined (set to 0) where Cmax == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(Cmax > 0, delta / Cmax, 0.0)

    return S.astype(np.float64)


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
    metric    = _compute_saturation(img_rgb)        # (P, P) float64
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
