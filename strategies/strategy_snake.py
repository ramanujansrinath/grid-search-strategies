"""
strategy_snake.py
-----------------
Strategy: SNAKE (boustrophedon)

Rule
----
1. First pick: uniformly random from all boxes.
2. A snake direction — one of LEFT (L), RIGHT (R), UP (U), DOWN (D) — is
   chosen uniformly at random for each sequence.  This determines the
   *primary sweep axis*:
     • L or R → sweep across rows, reversing direction on each row.
     • U or D → sweep down columns, reversing direction on each column.
3. The full canonical snake order is precomputed for the chosen direction,
   starting the first band in the chosen direction.  The starting box's
   position in that canonical order is found, and the sequence takes
   *seq_length* consecutive entries (wrapping modularly if needed).

Snake direction meanings
------------------------
  R : row 0 goes L→R, row 1 goes R→L, row 2 goes L→R, …
  L : row 0 goes R→L, row 1 goes L→R, row 2 goes R→L, …
  D : col 0 goes T→B, col 1 goes B→T, col 2 goes T→B, …
  U : col 0 goes B→T, col 1 goes T→B, col 2 goes B→T, …

The starting box is located within the canonical snake order; the sequence
then continues forward from that point (with wrap-around).
"""

from __future__ import annotations
import random
from typing import List

from grid_utils import all_boxes, get_row_col, get_box


def _build_snake_order(direction: str, N: int) -> List[int]:
    """Return the full canonical snake order for the given sweep direction."""
    order: List[int] = []

    if direction in ('L', 'R'):
        # Row-based sweep
        # 'R' → first row goes L→R; 'L' → first row goes R→L
        left_to_right = (direction == 'R')
        for row in range(N):
            cols = range(N) if left_to_right else range(N - 1, -1, -1)
            for col in cols:
                order.append(get_box(row, col, N))
            left_to_right = not left_to_right

    else:  # 'U' or 'D'
        # Column-based sweep
        # 'D' → first col goes T→B; 'U' → first col goes B→T
        top_to_bottom = (direction == 'D')
        for col in range(N):
            rows = range(N) if top_to_bottom else range(N - 1, -1, -1)
            for row in rows:
                order.append(get_box(row, col, N))
            top_to_bottom = not top_to_bottom

    return order


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
    total = N * N
    directions = ['L', 'R', 'U', 'D']
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        direction = random.choice(directions)
        snake_order = _build_snake_order(direction, N)

        # Random starting position within the snake order
        start_idx = random.randint(0, total - 1)
        seq = [snake_order[(start_idx + i) % total] for i in range(seq_length)]
        sequences.append(seq)

    return sequences
