"""
strategy_hilbert_curve.py
--------------------------
Strategy: HILBERT CURVE (space-filling path)

Rule
----
1. Compute the Hilbert curve traversal order for the grid.
   • For grids whose side length N is a power of 2, the standard Hilbert
     curve algorithm (d2xy) produces an exact N×N traversal.
   • For other N, the smallest enclosing power-of-2 grid is used; cells
     outside the N×N region are filtered out.  The resulting path still
     visits every box exactly once and preserves spatial locality.
2. First pick: uniformly random from all boxes.  The box is located in the
   precomputed Hilbert order to determine a starting offset.
3. The sequence takes *seq_length* consecutive entries from the Hilbert
   order starting at that offset, wrapping modularly.

Hilbert order for a 4×4 grid (d=0 → box 1):
  1, 2, 6, 5, 9, 13, 14, 10, 11, 15, 16, 12, 8, 7, 3, 4
"""

from __future__ import annotations
import random
from typing import List, Tuple

from grid_utils import get_box, in_bounds


# ── Hilbert curve helpers ─────────────────────────────────────────────────────

def _d2xy(n: int, d: int) -> Tuple[int, int]:
    """
    Convert Hilbert distance *d* to (x, y) coordinates in an n×n grid
    (n must be a power of 2).  x = col, y = row.
    Reference: Wikipedia "Hilbert curve" pseudocode.
    """
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_hilbert_order(N: int) -> List[int]:
    """
    Return all boxes (1-indexed) in Hilbert curve traversal order for an N×N
    grid.  Works for any N; non-power-of-2 grids use the next larger power-of-2
    curve with out-of-bounds entries filtered.
    """
    h = _next_pow2(N)
    order: List[int] = []
    for d in range(h * h):
        x, y = _d2xy(h, d)   # x=col, y=row
        if in_bounds(y, x, N):
            order.append(get_box(y, x, N))
    return order


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
    hilbert_order = _build_hilbert_order(N)   # length = N²
    total = len(hilbert_order)

    # Build a lookup: box → its index in the Hilbert order
    hilbert_pos = {box: idx for idx, box in enumerate(hilbert_order)}

    sequences: List[List[int]] = []

    for _ in range(n_seq):
        # Random starting box → find its position in the Hilbert path
        start_box = random.randint(1, N * N)
        start_idx = hilbert_pos[start_box]

        seq = [hilbert_order[(start_idx + i) % total] for i in range(seq_length)]
        sequences.append(seq)

    return sequences
