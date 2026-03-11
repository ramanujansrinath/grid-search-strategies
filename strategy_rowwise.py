"""
strategy_rowwise.py
-------------------
Strategy: ROW-WISE (random within row)

Rule
----
1. First pick: uniformly random from all boxes.
2. At each subsequent step, pick uniformly at random from unvisited boxes
   that share the same row as the *current* box.
3. If no unvisited boxes remain in the current row, pick uniformly at random
   from all remaining unvisited boxes (fallback), then reapply the row rule
   from that new box.
"""

from __future__ import annotations
import random
from typing import List

from grid_utils import get_row_col, all_boxes, random_choice


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
        current = random_choice(available)
        available.remove(current)
        seq.append(current)

        while len(seq) < seq_length:
            row_of_current = get_row_col(current, N)[0]

            # Candidates: unvisited boxes in the same row
            same_row = [b for b in available if get_row_col(b, N)[0] == row_of_current]

            if same_row:
                current = random_choice(same_row)
            else:
                # Fallback: pick any remaining box, then the row rule restarts
                current = random_choice(available)

            available.remove(current)
            seq.append(current)

        sequences.append(seq)

    return sequences
