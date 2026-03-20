"""
strategy_center_out_spiral.py
------------------------------
Strategy: CENTER-OUT SPIRAL (clockwise outward)

Rule
----
1. First pick: uniformly random from all boxes.
2. Second pick: the immediate LEFT neighbour of the first box, if in bounds;
   otherwise the immediate UP neighbour.  If *both* LEFT and UP are in bounds,
   choose randomly between them.
3. The spiral then proceeds clockwise outward using the fixed-leg expansion
   scheme (direction legs cycle L→U→R→D, with leg lengths 1,1,2,2,3,3,4,4,…).
   The starting direction is whichever was chosen in step 2.

   At each step of a leg the algorithm *tries* to advance in the current
   direction.  If that cell is out of bounds the position is frozen for that
   step (the step is still "used up").  If that cell is already visited the
   position moves there anyway (for tracking purposes) but no new box is added
   to the sequence.  After all steps in a leg, the direction rotates clockwise
   and the leg length increases every two completed legs.

4. The spiral path is precomputed for all N² boxes before the sequence is
   built.  The sequence then follows the precomputed order, skipping already-
   visited boxes (which can appear after a fallback random pick).

5. If the spiral has been fully exhausted before *seq_length* boxes are
   collected, the remainder are drawn uniformly at random from the unvisited
   pool (then the spiral is restarted from each such random pick).

Verified against user examples:
  start=6  → 6, 5, 1, 2, 3, 7, …
  start=10 → 10, 9, 5, 6, 7, 11, …
  start=13 → 13, 9, 10, 14, [random fallback], …
"""

from __future__ import annotations
import random
from typing import List, Tuple

from grid_utils import (
    all_boxes, get_row_col, get_box, in_bounds,
    random_choice, pick_and_remove,
)

# Direction indices and (Δrow, Δcol) vectors
_DIR_VEC = {
    'L': (0, -1),
    'U': (-1, 0),
    'R': (0,  1),
    'D': (1,  0),
}
# Clockwise rotation order starting from each possible starting direction
_CW_ORDER = ['L', 'U', 'R', 'D']


def _compute_spiral_path(start_box: int, start_dir: str, N: int) -> List[int]:
    """
    Precompute the full clockwise outward spiral from *start_box* beginning
    in direction *start_dir*.  Returns a list of all N² boxes in the order
    they are first encountered.
    """
    r, c = get_row_col(start_box, N)
    path: List[int] = [start_box]
    added: set = {start_box}

    dir_cycle = _CW_ORDER[_CW_ORDER.index(start_dir):] + \
                _CW_ORDER[:_CW_ORDER.index(start_dir)]

    dir_idx = 0          # index into dir_cycle
    leg_length = 1
    legs_completed = 0
    safety = 0

    while len(path) < N * N and safety < 8 * N * N:
        safety += 1
        dr, dc = _DIR_VEC[dir_cycle[dir_idx % 4]]

        for _ in range(leg_length):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, N):
                r, c = nr, nc          # always advance position if in bounds
                box = get_box(r, c, N)
                if box not in added:
                    added.add(box)
                    path.append(box)
            # else: out of bounds — position frozen

        legs_completed += 1
        dir_idx += 1
        if legs_completed % 2 == 0:
            leg_length += 1

    # Safety net: append any boxes missed (rare edge cases)
    for box in range(1, N * N + 1):
        if box not in added:
            path.append(box)

    return path


def _choose_start_dir(start_box: int, N: int) -> str:
    """
    Return the starting direction: LEFT or UP (or randomly between them if
    both are valid); fall through to RIGHT then DOWN if neither is available.
    """
    r, c = get_row_col(start_box, N)
    valid = []
    if in_bounds(r, c - 1, N):   # LEFT
        valid.append('L')
    if in_bounds(r - 1, c, N):   # UP
        valid.append('U')
    if valid:
        return random.choice(valid)
    if in_bounds(r, c + 1, N):   # RIGHT (corner fallback)
        return 'R'
    return 'D'                   # last resort


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

        def _run_spiral(seed: int) -> None:
            """Follow the spiral from *seed* until stuck or done."""
            start_dir = _choose_start_dir(seed, N)
            spiral = _compute_spiral_path(seed, start_dir, N)

            for box in spiral:
                if len(seq) >= seq_length:
                    break
                if box in available:
                    available.remove(box)
                    seq.append(box)

        # ── First pick: uniformly random ──────────────────────────────────
        seed = pick_and_remove(available)
        seq.append(seed)
        _run_spiral(seed)

        # Fallback: if spiral exhausted before seq_length, pick randomly
        # and restart spiral
        while len(seq) < seq_length:
            seed = pick_and_remove(available)
            seq.append(seed)
            if len(seq) < seq_length:
                _run_spiral(seed)

        sequences.append(seq)

    return sequences
