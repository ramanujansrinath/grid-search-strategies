"""
strategy_diagonal_sweep.py
---------------------------
Strategy: DIAGONAL SWEEP

Diagonal convention
-------------------
Uses "/" anti-diagonals: all boxes where (row + col) = k share diagonal k.
For a 4×4 grid, k runs from 0 (only box 1) to 6 (only box 16).

  k=0: 1
  k=1: 2, 5
  k=2: 3, 6, 9
  k=3: 4, 7, 10, 13
  k=4: 8, 11, 14
  k=5: 12, 15
  k=6: 16

Within each diagonal, boxes are ordered top-to-bottom (increasing row).

Rule
----
1. First pick: uniformly random from all boxes.  Its diagonal k determines
   the starting diagonal.
2. Sweep diagonals in increasing k order: k, k+1, …, 2*(N-1), 0, 1, …, k-1
   (wrapping around).
3. Within each diagonal, pick uniformly at random from unvisited boxes on
   that diagonal (boxes already visited are skipped).
4. Collect picks until *seq_length* boxes are selected.
5. Fallback: if a diagonal has no unvisited boxes, advance to the next one.
"""

from __future__ import annotations
import random
from typing import List, Dict

from grid_utils import all_boxes, get_row_col, random_choice


def _build_diagonals(N: int) -> Dict[int, List[int]]:
    """Return a dict mapping diagonal index k → sorted list of boxes (by row)."""
    diags: Dict[int, List[int]] = {}
    for box in all_boxes(N):
        r, c = get_row_col(box, N)
        k = r + c
        diags.setdefault(k, []).append(box)
    # Sort each diagonal top-to-bottom (increasing row)
    for k in diags:
        diags[k].sort(key=lambda b: get_row_col(b, N)[0])
    return diags


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
    diagonals = _build_diagonals(N)
    max_k = 2 * (N - 1)
    num_diags = max_k + 1                  # diagonals indexed 0 … 2*(N-1)
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        # ── First pick: uniformly random ──────────────────────────────────
        first = random_choice(available)
        available.remove(first)
        seq.append(first)

        # Starting diagonal
        r0, c0 = get_row_col(first, N)
        start_k = r0 + c0

        # Sweep diagonals: start_k → start_k + num_diags - 1 (mod num_diags)
        for step in range(num_diags):
            k = (start_k + step) % num_diags
            if k not in diagonals:
                continue

            diag_boxes = [b for b in diagonals[k] if b in available]
            random.shuffle(diag_boxes)

            for box in diag_boxes:
                if len(seq) >= seq_length:
                    break
                if box in available:
                    available.remove(box)
                    seq.append(box)

            if len(seq) >= seq_length:
                break

        sequences.append(seq[:seq_length])

    return sequences
