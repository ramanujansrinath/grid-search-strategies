"""
strategy_center_out_radial.py
------------------------------
Strategy: CENTER-OUT RADIAL

Rule
----
1. First pick: uniformly random from all boxes.
2. At each step, inspect the unvisited grid neighbours (up/down/left/right)
   of the *current* box.
   • Among those neighbours whose distance from the geometric grid centre
     is *strictly greater* than the current box's distance, pick the one
     with the maximum such distance (ties broken uniformly at random).
   • If no unvisited neighbour has a strictly greater distance (i.e. we
     cannot move "outward"), fall back to a uniformly random pick from all
     remaining unvisited boxes, then reapply the rule from that new box.

Distance is Euclidean from each box centre to the geometric centre
((N-1)/2, (N-1)/2) in 0-indexed row/col coordinates.
"""

from __future__ import annotations
from typing import List

from grid_utils import (
    all_boxes, get_neighbors, dist_from_center,
    random_choice, pick_and_remove,
)


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

    # Pre-compute distances once (they never change)
    center_dist = {b: dist_from_center(b, N) for b in all_boxes(N)}

    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        # ── First pick: uniformly random ──────────────────────────────────
        current = pick_and_remove(available)
        seq.append(current)

        while len(seq) < seq_length:
            cur_dist = center_dist[current]

            # Unvisited neighbours strictly farther from centre
            outward = [
                b for b in get_neighbors(current, N)
                if b in available and center_dist[b] > cur_dist
            ]

            if outward:
                # Pick the one(s) with maximum distance
                max_dist = max(center_dist[b] for b in outward)
                best = [b for b in outward if center_dist[b] == max_dist]
                current = random_choice(best)
                available.remove(current)
            else:
                # Fallback: random pick, then reapply rule
                current = pick_and_remove(available)

            seq.append(current)

        sequences.append(seq)

    return sequences
