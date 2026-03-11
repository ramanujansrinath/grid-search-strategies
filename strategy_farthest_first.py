"""
strategy_farthest_first.py
---------------------------
Strategy: FARTHEST FIRST

Rule
----
1. First pick: uniformly random from all boxes.
2. At each step, among *all* unvisited boxes, pick the one whose Euclidean
   distance (between box centres) to the *current* box is *largest*.
   Ties are broken uniformly at random.
3. No fallback is needed: there is always at least one unvisited box while
   the sequence is being built (assuming seq_length ≤ N²).

This tends to produce sequences that jump across the grid on each step,
maximising spatial spread.
"""

from __future__ import annotations
from typing import List

from grid_utils import all_boxes, euclidean, random_choice, pick_and_remove


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
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        # ── First pick: uniformly random ──────────────────────────────────
        current = pick_and_remove(available)
        seq.append(current)

        while len(seq) < seq_length:
            max_dist = max(euclidean(current, b, N) for b in available)
            farthest = [b for b in available if euclidean(current, b, N) == max_dist]
            current = random_choice(farthest)
            available.remove(current)
            seq.append(current)

        sequences.append(seq)

    return sequences
