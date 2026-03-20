"""
strategy_rowwise_sequential.py
-------------------------------
Strategy: ROW-WISE SEQUENTIAL (left → right, with wrap)

Rule
----
1. First pick: a random box.  This determines the starting position *within*
   the canonical left-to-right, top-to-bottom reading order.
2. Each subsequent pick is the next unvisited box in that reading order,
   wrapping from the last box (N²) back to box 1 if necessary.
3. Because selections are without replacement in each fresh sequence, and
   seq_length ≤ N², the canonical pointer simply advances; no fallback is
   ever needed.

The "canonical order" is: 1, 2, 3, … N² (left to right, row by row).
Starting at a random offset, the sequence takes *seq_length* consecutive
entries (with modular wrap).
"""

from __future__ import annotations
import random
from typing import List

from grid_utils import all_boxes


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
    canonical = all_boxes(N)   # [1, 2, …, N²]
    total = N * N
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        # Random starting offset in the canonical order
        start_idx = random.randint(0, total - 1)
        seq = [canonical[(start_idx + i) % total] for i in range(seq_length)]
        sequences.append(seq)

    return sequences
