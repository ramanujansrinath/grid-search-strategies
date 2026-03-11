"""
strategy_perimeter_crawl.py
----------------------------
Strategy: PERIMETER CRAWL

Perimeter definition
--------------------
The perimeter consists of the 4*(N-1) boxes on the outermost ring of the
grid, listed in clockwise order starting from the top-left corner:
  top row (L→R), right column (T→B, skipping corner already counted),
  bottom row (R→L, skipping corner), left column (B→T, skipping both corners).

For a 4×4 grid: 1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5

Rule
----
1. First pick: uniformly random from ALL boxes (interior or perimeter).
2a. If the first pick is ON the perimeter:
    • Choose CW or CCW direction uniformly at random.
    • Crawl the perimeter in that direction for the remaining picks.
2b. If the first pick is an INTERIOR box:
    • Find the perimeter box nearest (Euclidean) to the first pick.
      Ties broken uniformly at random.
    • Pick that perimeter box next.
    • Then crawl the perimeter (CW or CCW chosen randomly) for the rest.
3. The crawl wraps around the perimeter ring as needed (only N²/2 picks at
   most, so wrapping is common for small grids).
4. Fallback: if a crawl position has already been visited (due to a previous
   fallback), skip it and advance to the next perimeter position.
"""

from __future__ import annotations
import random
import math
from typing import List

from grid_utils import all_boxes, get_row_col, get_box, euclidean, random_choice, pick_and_remove


def _build_perimeter(N: int) -> List[int]:
    """Return the perimeter boxes in clockwise order."""
    perim: List[int] = []
    # Top row: (0, 0) → (0, N-1)
    for c in range(N):
        perim.append(get_box(0, c, N))
    # Right column: (1, N-1) → (N-1, N-1)
    for r in range(1, N):
        perim.append(get_box(r, N - 1, N))
    # Bottom row: (N-1, N-2) → (N-1, 0)
    for c in range(N - 2, -1, -1):
        perim.append(get_box(N - 1, c, N))
    # Left column: (N-2, 0) → (1, 0)
    for r in range(N - 2, 0, -1):
        perim.append(get_box(r, 0, N))
    return perim


def _is_perimeter(box: int, N: int) -> bool:
    r, c = get_row_col(box, N)
    return r == 0 or r == N - 1 or c == 0 or c == N - 1


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
    perimeter = _build_perimeter(N)   # ordered CW
    perim_index = {box: idx for idx, box in enumerate(perimeter)}
    P = len(perimeter)
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        # ── First pick: uniformly random ──────────────────────────────────
        first = pick_and_remove(available)
        seq.append(first)

        # Determine crawl start on the perimeter
        if _is_perimeter(first, N):
            perim_start = first
        else:
            # Nearest perimeter box (Euclidean distance)
            dists = [(euclidean(first, pb, N), pb) for pb in perimeter if pb in available]
            if not dists:
                # All perimeter boxes already used (only possible if seq_length >> P)
                while len(seq) < seq_length:
                    seq.append(pick_and_remove(available))
                sequences.append(seq)
                continue
            min_d = min(d for d, _ in dists)
            candidates = [pb for d, pb in dists if d == min_d]
            perim_start = random_choice(candidates)
            available.remove(perim_start)
            seq.append(perim_start)

        # CW (+1) or CCW (-1)
        step = random.choice([1, -1])
        idx = perim_index[perim_start]

        # Continue crawling
        while len(seq) < seq_length:
            idx = (idx + step) % P
            box = perimeter[idx]
            if box in available:
                available.remove(box)
                seq.append(box)
            # Skip already-visited perimeter boxes; if we've gone all the way
            # around without finding one, fall back to random
            else:
                # Check if any perimeter box remains
                remaining_perim = [perimeter[(idx + k) % P]
                                   for k in range(1, P)
                                   if perimeter[(idx + k) % P] in available]
                if remaining_perim:
                    continue   # let the while loop advance idx
                else:
                    # No more perimeter boxes — pick randomly from all remaining
                    while len(seq) < seq_length and available:
                        seq.append(pick_and_remove(available))
                    break

        sequences.append(seq)

    return sequences
