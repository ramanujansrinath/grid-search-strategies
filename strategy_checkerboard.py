"""
strategy_checkerboard.py
-------------------------
Strategy: CHECKERBOARD

Colour convention
-----------------
Box *b* is "white" if (row + col) % 2 == 0, "black" otherwise
(using 0-indexed row and col).  For a 4×4 grid this gives 8 boxes of each
colour.

Rule
----
1. First pick: uniformly random from all boxes.  Its colour determines the
   colour of the entire sequence.
2. All subsequent picks are drawn uniformly at random from the remaining
   unvisited boxes of the *same* colour.
3. If the colour pool is exhausted before *seq_length* boxes are collected
   (possible when seq_length > N²/2), fall back to the remaining boxes of the
   opposite colour (drawn uniformly at random).
"""

from __future__ import annotations
from typing import List

from grid_utils import all_boxes, get_row_col, random_choice, pick_and_remove


def _colour(box: int, N: int) -> int:
    """Return 0 (white) or 1 (black) for a box."""
    r, c = get_row_col(box, N)
    return (r + c) % 2


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
        first = pick_and_remove(available)
        seq.append(first)
        target_colour = _colour(first, N)

        while len(seq) < seq_length:
            same_colour = [b for b in available if _colour(b, N) == target_colour]

            if same_colour:
                chosen = random_choice(same_colour)
            else:
                # Fallback: opposite colour (pool exhausted)
                chosen = random_choice(available)

            available.remove(chosen)
            seq.append(chosen)

        sequences.append(seq)

    return sequences
