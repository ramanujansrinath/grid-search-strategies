"""
strategy_weighted_center_bias.py
---------------------------------
Strategy: WEIGHTED CENTER BIAS

Rule
----
All boxes are assigned a fixed weight w(b) = 1 / dist(b, centre) where
dist is the Euclidean distance from the box centre to the geometric grid
centre ((N-1)/2, (N-1)/2).

Weights are computed *once* at the start of each sequence and never
renormalised as boxes are removed.  On each draw, the probability of
selecting a remaining unvisited box b is:

    P(b) = w(b) / Σ w(j) for j in available

This biases selection towards the centre without being deterministic.

Parameters
----------
WEIGHT_POWER : exponent applied to the inverse distance (default 1.0 for
    1/d weighting).  Increasing this sharpens the centre bias.
"""

from __future__ import annotations
import random
from typing import List, Dict

from grid_utils import all_boxes, dist_from_center


# ── Exposed tuning constant ───────────────────────────────────────────────────
WEIGHT_POWER: float = 3.0   # w(b) = 1 / dist(b, centre) ** WEIGHT_POWER
# ─────────────────────────────────────────────────────────────────────────────


def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
) -> List[List[int]]:
    """
    Parameters
    ----------
    grid_size  : side length of the square grid.
    seq_length : number of boxes per sequence.
    n_seq      : number of sequences to generate.

    Returns
    -------
    List of *n_seq* sequences, each of length *seq_length*.
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")

    N = grid_size

    # Pre-compute weights once (same for every sequence)
    weights: Dict[int, float] = {}
    for b in all_boxes(N):
        d = dist_from_center(b, N)
        weights[b] = 1.0 / (d ** WEIGHT_POWER) if d > 0 else float('inf')

    # Cap the infinite weight (box exactly at centre, rare for even N)
    max_finite = max(v for v in weights.values() if v != float('inf'))
    for b in weights:
        if weights[b] == float('inf'):
            weights[b] = max_finite * 10.0  # large but finite

    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = list(all_boxes(N))
        seq: List[int] = []

        while len(seq) < seq_length:
            # Use pre-computed (fixed) weights, normalised over available pool
            pool_weights = [weights[b] for b in available]
            chosen = random.choices(available, weights=pool_weights, k=1)[0]
            available.remove(chosen)
            seq.append(chosen)

        sequences.append(seq)

    return sequences
